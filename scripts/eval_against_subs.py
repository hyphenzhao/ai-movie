#!/usr/bin/env python
"""Score a run against the film's own burned-in subtitles.

The test material carries two lines of hard-coded subtitles (Chinese on top,
Japanese underneath).  That Japanese line is ground truth for what was
actually said, so comparing it with the ASR text separates a recognition
error from a translation error instead of guessing from the Chinese alone.

Nothing here can read pixels on its own — step 1 lays the subtitle band of
every segment out as numbered contact sheets, a human (or Claude) types the
Japanese into a JSON file, and step 2 turns that into a score.  The ground
truth is stored **by time**, not by segment number, so it keeps working
after the pipeline re-segments the film.

    # 1. contact sheets + a skeleton to fill in
    python scripts/eval_against_subs.py output_test --make-tiles

    # 1b. one-off: turn a filled skeleton / legacy index file into time cues
    python scripts/eval_against_subs.py output_test --convert-gt \\
        --gt workspace/output_test/eval/gt_ja.json --gt-out inputs/subs/output_test.gt.json

    # 2. score (reference-translation SRT optional)
    python scripts/eval_against_subs.py output_test \\
        --gt inputs/subs/output_test.gt.json --ref-srt inputs/subs/output_test.zh.srt

    # translation-side metrics only, for a film without ground truth
    python scripts/eval_against_subs.py test_1 --no-gt --summary-json out.json

Both the Japanese ground truth and the reference translation are *editorial*
texts: official subtitlers drop fillers (「ちょっと」「うん」) and reword
freely, so a low score on one line is a lead to read, not a verdict.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import subprocess
import sys
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_movie import units as units_mod  # noqa: E402

# Bottom band of a 1080p frame that holds both subtitle lines.  Both lines
# must land inside it even when the Chinese wraps to two lines, so the band
# is deliberately taller than one line of text.
BAND = {"w": 1920, "h": 210, "y": 870}
TILE_ROWS = 8            # strips per contact sheet
STRIP_W = 1280           # downscaled strip width; text stays legible here
MATCH_TOL = 0.3          # seconds a segment midpoint may fall outside a cue

_PUNCT = re.compile(r"[、。，．！？!?…·・\-—～~「」『』（）()\s　,.]")
_ZH_PUNCT = re.compile(r"[（）()　\s、。，．！？!?…·・\-—～~「」『』]")

BANDS = [
    (0.95, 1.01, "几乎逐字一致"),
    (0.85, 0.95, "很接近（只差语气词）"),
    (0.70, 0.85, "大意对但有出入"),
    (0.50, 0.70, "部分错"),
    (0.00, 0.50, "基本错"),
]


# ── text normalisation ──────────────────────────────────────────────

def fold_ja(text: str | None) -> str:
    """Normalise Japanese so only *content* differences score as errors.

    Katakana folds to hiragana (おチンチン and おちんちん are the same word
    spelled two ways, and Whisper picks either), full-width digits and
    センチ/cm collapse, and punctuation goes away entirely — the burned-in
    subtitles punctuate by house style, not by what was said.
    """
    out = []
    for ch in text or "":
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:          # katakana → hiragana
            ch = chr(code - 0x60)
        out.append(ch)
    s = "".join(out)
    s = s.translate(str.maketrans("０１２３４５６７８９ｃｍ", "0123456789cm"))
    s = s.replace("センチ", "cm").replace("せんち", "cm")
    return _PUNCT.sub("", s)


def fold_zh(text: str | None) -> str:
    return _ZH_PUNCT.sub("", text or "")


def ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ── inputs ──────────────────────────────────────────────────────────

def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    """Return (start, end, text) triples.  Tolerates BOM and CRLF."""
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")

    def ts(token: str) -> float:
        hh, mm, rest = token.split(":")
        ss, ms = rest.replace(".", ",").split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000

    cues = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [x for x in block.split("\n") if x.strip()]
        if len(lines) < 2:
            continue
        i = 1 if re.fullmatch(r"\d+", lines[0].strip()) else 0
        m = re.match(r"(\S+)\s*-->\s*(\S+)", lines[i])
        if not m:
            continue
        cues.append((ts(m.group(1)), ts(m.group(2)), "\n".join(lines[i + 1:])))
    return cues


def load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_segments(state: dict) -> list[dict]:
    """The segments whose Chinese is what the viewer actually hears.

    fit → compact → translate, but a later stage is used only when it is
    aligned with ``translate`` (same count, same start times): a partial
    re-run leaves the previous run's fit/compact lists in state.json until
    those stages run again, and scoring those would silently mix two runs.
    """
    base = (state.get("translate") or {}).get("segments") \
        or (state.get("asr") or {}).get("segments")
    if not base:
        raise SystemExit("state has no asr/translate segments")
    starts = [round(float(s["start"]), 2) for s in base]
    for stage in ("fit", "compact"):
        segs = (state.get(stage) or {}).get("segments")
        if segs and len(segs) == len(base) \
                and [round(float(s["start"]), 2) for s in segs] == starts:
            return segs
    return base


def index_gt_to_cues(gt: dict[str, str], segments: list[dict]) -> list[dict]:
    """Legacy ``{"<segment index>": "<ja>"}`` → time cues.

    Consecutive segments carrying the same subtitle text are one on-screen
    cue; its span runs from the first segment's start to the last one's end.
    """
    cues = []
    for text, run in groupby(range(len(segments)), key=lambda i: gt.get(str(i), "")):
        idxs = list(run)
        if not text:
            continue
        cues.append({"id": len(cues), "ja": text,
                     "start": round(float(segments[idxs[0]]["start"]), 2),
                     "end": round(float(segments[idxs[-1]]["end"]), 2)})
    return cues


def load_gt(path: Path, segments: list[dict] | None = None) -> list[dict]:
    """Accept time cues, a filled skeleton, or the legacy index format."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cues" in data:
        return data["cues"]
    if isinstance(data, dict) and data.get("format") == "segments":
        pseudo = {str(k): r.get("ja", "") for k, r in enumerate(data["segments"])}
        return index_gt_to_cues(pseudo, data["segments"])
    if segments is None:
        raise SystemExit(f"{path} is index-keyed; convert it with --convert-gt first")
    return index_gt_to_cues(data, segments)


def find_video(name: str, state: dict) -> Path:
    src = state.get("input") or state.get("video")
    if src and Path(src).exists():
        return Path(src)
    for ext in (".mp4", ".mkv", ".mov", ".webm"):
        cand = ROOT / "inputs" / f"{name}{ext}"
        if cand.exists():
            return cand
    raise SystemExit(f"cannot locate the source video for {name!r}")


# ── step 1: contact sheets ──────────────────────────────────────────

def make_tiles(video: Path, segments: list[dict], outdir: Path) -> Path:
    """Render one labelled subtitle strip per segment, tiled for reading."""
    strips = outdir / "strips"
    tiles = outdir / "tiles"
    for d in (strips, tiles):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()

    font = subprocess.run(["fc-match", "-f", "%{file}", "DejaVuSans"],
                          capture_output=True, text=True).stdout.strip()
    height = round(BAND["h"] * STRIP_W / BAND["w"])

    for i, seg in enumerate(segments):
        mid = (float(seg["start"]) + float(seg["end"])) / 2.0
        vf = (f"crop={BAND['w']}:{BAND['h']}:0:{BAND['y']},"
              f"scale={STRIP_W}:{height},"
              f"drawbox=x=0:y=0:w=118:h=26:color=black@0.85:t=fill")
        if font:
            vf += (f",drawtext=fontfile={font}:text='#{i} {float(seg['start']):.0f}s'"
                   f":x=5:y=4:fontsize=18:fontcolor=yellow")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{mid:.2f}", "-i", str(video),
             "-frames:v", "1", "-vf", vf, "-y", str(strips / f"{i:03d}.png")],
            check=True)

    files = sorted(strips.glob("*.png"))
    for k in range(0, len(files), TILE_ROWS):
        group = files[k:k + TILE_ROWS]
        args = ["ffmpeg", "-v", "error"]
        for g in group:
            args += ["-i", str(g)]
        chain = "".join(f"[{j}:v]" for j in range(len(group)))
        args += ["-filter_complex", f"{chain}vstack=inputs={len(group)}[o]",
                 "-map", "[o]", "-y", str(tiles / f"t_{k // TILE_ROWS:02d}.png")]
        subprocess.run(args, check=True)

    skeleton = outdir / "gt_skeleton.json"
    if not skeleton.exists():
        skeleton.write_text(json.dumps({
            "format": "segments",
            "segments": [{"i": i, "start": float(s["start"]), "end": float(s["end"]),
                          "ja": ""} for i, s in enumerate(segments)],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    return tiles


# ── step 2: scoring ─────────────────────────────────────────────────

def cue_for(seg: dict, cues: list[dict]) -> int | None:
    """Cue whose span holds the segment midpoint (± MATCH_TOL), else the
    cue with the largest overlap, else None (no subtitle on screen)."""
    a, b = float(seg["start"]), float(seg["end"])
    mid = (a + b) / 2.0
    best, best_ov = None, 0.0
    for c in cues:
        ov = min(b, c["end"]) - max(a, c["start"])
        if c["start"] - MATCH_TOL <= mid <= c["end"] + MATCH_TOL:
            ov += 1000.0                      # containment always wins
        if ov > best_ov:
            best, best_ov = c["id"], ov
    return best


def _slice_by_time(text: str, seg: dict, a: float, b: float) -> str:
    """The share of *text* spoken between *a* and *b*, assuming an even rate."""
    s0, s1 = float(seg["start"]), float(seg["end"])
    if s1 <= s0 or (a <= s0 and b >= s1):
        return text
    n = len(text)
    lo = round(n * max(0.0, (a - s0) / (s1 - s0)))
    hi = round(n * min(1.0, (b - s0) / (s1 - s0)))
    return text[lo:hi]


def build_groups(segments: list[dict], cues: list[dict]) -> list[dict]:
    """One group per on-screen cue, plus runs of segments with no cue.

    A cue gathers every segment that overlaps its span.  A segment spanning
    two cues contributes to each the share of its text that falls inside, so
    the score does not depend on how the pipeline happened to segment — a run
    that merges or splits differently from the one the ground truth was
    typed against is still measured on the same 87 sentences.  On the
    segmentation the ground truth came from, every segment lies inside one
    cue and this reduces to plain concatenation.
    """
    by_seg: dict[int, list[int]] = {}
    groups = []
    for c in sorted(cues, key=lambda c: c["start"]):
        idxs = [i for i, sg in enumerate(segments)
                if cue_for(sg, cues) == c["id"]
                or min(c["end"], float(sg["end"])) - max(c["start"], float(sg["start"])) > 0.1]
        if not idxs:
            groups.append({"cue": c["id"], "idxs": [], "start": c["start"], "end": c["end"],
                           "ref_ja": c["ja"], "asr_ja": "", "ours_zh": "", "sim": 0.0,
                           "fragmented": False})
            continue
        for i in idxs:
            by_seg.setdefault(i, []).append(c["id"])
        ja = "".join(_slice_by_time((segments[i].get("text") or "").strip(), segments[i],
                                    c["start"], c["end"]) for i in idxs)
        pieces = [(segments[i].get("text_translated") or "").strip() for i in idxs]
        groups.append({
            "cue": c["id"], "idxs": idxs,
            "start": float(segments[idxs[0]]["start"]),
            "end": float(segments[idxs[-1]]["end"]),
            "ref_ja": c["ja"], "asr_ja": ja, "ours_zh": " / ".join(pieces),
            "sim": ratio(fold_ja(c["ja"]), fold_ja(ja)),
            "fragmented": len(idxs) > 1 and units_mod.sentence_ends(
                "".join(pieces)) > max(1, units_mod.sentence_ends(c["ja"])),
        })
    # segments under no cue: official subtitles skip fillers and overlaps
    loose = [i for i in range(len(segments)) if i not in by_seg]
    for _, run in groupby(enumerate(loose), key=lambda t: t[1] - t[0]):
        idxs = [i for _, i in run]
        groups.append({
            "cue": None, "idxs": idxs,
            "start": float(segments[idxs[0]]["start"]),
            "end": float(segments[idxs[-1]]["end"]),
            "ref_ja": "", "fragmented": False, "sim": None,
            "asr_ja": "".join((segments[i].get("text") or "").strip() for i in idxs),
            "ours_zh": " / ".join((segments[i].get("text_translated") or "").strip()
                                  for i in idxs),
        })
    groups.sort(key=lambda g: g["start"])
    return groups


def attach_reference_zh(groups: list[dict], cues: list[tuple[float, float, str]]) -> None:
    for g in groups:
        hits = [t for st, en, t in cues
                if min(g["end"], en) - max(g["start"], st) > 0.25]
        g["ref_zh"] = " ".join(t.replace("\n", " ") for t in hits)
        g["zh_sim"] = (ratio(fold_zh(g["ref_zh"]), fold_zh(g["ours_zh"]))
                       if g["ref_zh"] and g["ours_zh"] else None)


def translation_metrics(segments: list[dict]) -> dict:
    """Ground-truth-free translation health, comparable across runs.

    Units come from ``units.group_units`` on the scored segments themselves,
    so a run that did not translate by unit is measured on the same
    utterance boundaries as one that did.
    """
    groups = units_mod.group_units(segments)
    pron = []
    for g in groups:
        ja = "".join((segments[i].get("text") or "") for i in g)
        zh = "".join((segments[i].get("text_translated") or "") for i in g)
        if units_mod.unsupported_pronoun(ja, zh):
            pron.append({"idxs": g, "ja": ja, "zh": zh})
    kana = [i for i, s in enumerate(segments) if units_mod.has_kana(s.get("text_translated"))]
    frag = 0
    for g in groups:
        if len(g) < 2:
            continue
        zh_pieces = [(segments[i].get("text_translated") or "") for i in g]
        ja = "".join((segments[i].get("text") or "") for i in g)
        if sum(1 for p in zh_pieces if units_mod.sentence_ends(p)) > max(
                1, units_mod.sentence_ends(ja)):
            frag += 1
    return {
        "n_segments": len(segments),
        "n_units": len(groups),
        "multi_units": sum(1 for g in groups if len(g) > 1),
        "pronoun_units": len(pron),
        "pronoun_examples": pron[:10],
        "kana_segments": len(kana),
        "fragmented_units": frag,
    }


def summarise(groups: list[dict], segments: list[dict]) -> dict:
    scored = [g for g in groups if g["sim"] is not None]
    out = {"cues": len(scored),
           "covered_segments": len({i for g in scored for i in g["idxs"]}),
           **translation_metrics(segments)}
    if scored:
        sims = [g["sim"] for g in scored]
        out.update({
            "asr_median": round(statistics.median(sims), 4),
            "asr_mean": round(sum(sims) / len(sims), 4),
            "bands": {label: sum(1 for s in sims if lo <= s < hi) for lo, hi, label in BANDS},
            "asr_bad": sum(1 for s in sims if s < 0.70),
            "fragmented_cues": sum(1 for g in scored if g["fragmented"]),
            "split_cues": sum(1 for g in scored if len(g["idxs"]) > 1),
        })
    return out


def render(name: str, groups: list[dict], segments: list[dict],
           summary: dict, worst: int) -> str:
    out = [f"# 字幕对照评测 — {name}", ""]
    scored = [g for g in groups if g["sim"] is not None]
    blank = [g for g in groups if g["sim"] is None]
    if scored:
        covered = summary["covered_segments"]
        out += [
            f"- 画面字幕句数 **{summary['cues']}**，覆盖段 **{covered}/{len(segments)}**",
            f"- 画面无字幕的段 **{len(segments) - covered}**（官方字幕不给语气词/重叠语，不等于幻听）",
            f"- ASR 与画面原文相似度：**中位数 {summary['asr_median']:.2f}**，"
            f"平均 {summary['asr_mean']:.2f}",
            "",
            "| 吻合度 | 句数 | 占比 |",
            "|---|---:|---:|",
        ]
        for _, _, label in BANDS:
            n = summary["bands"][label]
            out.append(f"| {label} | {n} | {n / summary['cues'] * 100:.0f}% |")
        out += [
            "",
            f"- 一句字幕被切成多段：**{summary['split_cues']}** 句；"
            f"其中译文被拆成多个完整中文句（半截话）：**{summary['fragmented_cues']}** 句",
        ]
    out += [
        "",
        "## 翻译健康度（无需字幕，可跨影片比较）",
        "",
        f"- 段 {summary['n_segments']}，句子单元 {summary['n_units']}"
        f"（多段单元 {summary['multi_units']}）",
        f"- 半截话单元（多段单元里出现多个完整中文句）：**{summary['fragmented_units']}**",
        f"- 日文没有、中文凭空出现人称代词的单元：**{summary['pronoun_units']}**",
        f"- 译文残留假名的段：**{summary['kana_segments']}**",
        "",
    ]
    if scored:
        out += [f"## ASR 差异最大的 {worst} 句", ""]
        for g in sorted(scored, key=lambda x: x["sim"])[:worst]:
            out += [
                f"**[{g['start']:.1f}s] #{g['idxs'][0]}–{g['idxs'][-1]}  sim={g['sim']:.2f}**",
                f"- 画面原文：{g['ref_ja']}",
                f"- 我们 ASR：{g['asr_ja']}",
                f"- 我们译文：{g['ours_zh']}",
            ]
            if g.get("ref_zh"):
                out.append(f"- 参考译文：{g['ref_zh']}")
            out.append("")
        good = [g for g in scored if g["sim"] >= 0.85 and g.get("zh_sim") is not None]
        if good:
            out += [
                "## ASR 听对了（sim≥0.85）但译文与参考差最远的句子", "",
                "*两份独立译文措辞本就不同，低分是线索不是结论；"
                "真正的翻译错误多表现为凭空出现的主语或指代。*", "",
            ]
            for g in sorted(good, key=lambda x: x["zh_sim"])[:worst]:
                out += [
                    f"**[{g['start']:.1f}s]  ASR {g['sim']:.2f} / 译文 {g['zh_sim']:.2f}**",
                    f"- 原文：{g['ref_ja']}",
                    f"- 我们译文：{g['ours_zh']}",
                    f"- 参考译文：{g['ref_zh']}",
                    "",
                ]
    if summary["pronoun_examples"]:
        out += ["## 人称代词可疑单元（前 10）", ""]
        for p in summary["pronoun_examples"]:
            out += [f"- #{p['idxs'][0]}–{p['idxs'][-1]}：{p['ja']} → {p['zh']}"]
        out.append("")
    if scored and blank:
        out += [f"## 画面无字幕的 {len(blank)} 组（幻听候选，需人工判断）", ""]
        for g in blank:
            out += [f"**[{g['start']:.1f}s] #{g['idxs'][0]}–{g['idxs'][-1]}**",
                    f"- ASR：{g['asr_ja']}", f"- 译文：{g['ours_zh']}", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="workspace name, e.g. output_test")
    ap.add_argument("--state", type=Path, default=None,
                    help="state.json to score (default: workspace/<name>/state.json)")
    ap.add_argument("--make-tiles", action="store_true",
                    help="render contact sheets + a ground-truth skeleton, then stop")
    ap.add_argument("--convert-gt", action="store_true",
                    help="convert --gt (legacy index or filled skeleton) to time cues")
    ap.add_argument("--gt", type=Path, default=None,
                    help="ground truth (default: inputs/subs/<name>.gt.json)")
    ap.add_argument("--gt-out", type=Path, default=None,
                    help="where --convert-gt writes (default: inputs/subs/<name>.gt.json)")
    ap.add_argument("--no-gt", action="store_true",
                    help="skip the subtitle comparison; translation metrics only")
    ap.add_argument("--ref-srt", type=Path, default=None,
                    help="reference translation SRT, for the translation-side table")
    ap.add_argument("--worst", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None,
                    help="report path (default: deliver/<name>/09_subs_eval.md)")
    ap.add_argument("--summary-json", type=Path, default=None,
                    help="also write the machine-readable summary here")
    args = ap.parse_args()

    workspace = ROOT / "workspace" / args.name
    state_path = args.state or (workspace / "state.json")
    if not state_path.exists():
        raise SystemExit(f"no state at {state_path}")
    state = load_state(state_path)
    segments = final_segments(state)
    evaldir = workspace / "eval"
    evaldir.mkdir(parents=True, exist_ok=True)
    default_gt = ROOT / "inputs" / "subs" / f"{args.name}.gt.json"

    if args.make_tiles:
        tiles = make_tiles(find_video(args.name, state), segments, evaldir)
        print(f"contact sheets: {tiles}  ({len(list(tiles.glob('*.png')))} sheets, "
              f"{len(segments)} segments)")
        print(f"fill in {evaldir / 'gt_skeleton.json'}, then --convert-gt --gt <that file>")
        return

    if args.convert_gt:
        src = args.gt or (evaldir / "gt_ja.json")
        cues = load_gt(src, segments)
        dst = args.gt_out or default_gt
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps({"video": args.name, "source": str(src), "cues": cues},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(cues)} cues → {dst}")
        return

    if args.no_gt:
        cues: list[dict] = []
    else:
        gt_path = args.gt or default_gt
        if not gt_path.exists():
            raise SystemExit(f"no ground truth at {gt_path} — run --make-tiles / --convert-gt, "
                             f"or pass --no-gt")
        cues = load_gt(gt_path)

    groups = build_groups(segments, cues)
    if args.ref_srt:
        attach_reference_zh(groups, parse_srt(args.ref_srt))
    summary = summarise(groups, segments)

    report = render(args.name, groups, segments, summary, args.worst)
    out = args.out or (ROOT / "deliver" / args.name / "09_subs_eval.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    (evaldir / "subs_eval.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                     encoding="utf-8")

    if summary["cues"]:
        print(f"{args.name}: {summary['cues']} 句对照，ASR 中位数 {summary['asr_median']:.2f}，"
              f"平均 {summary['asr_mean']:.2f}，差(<0.70) {summary['asr_bad']}，"
              f"半截话 {summary['fragmented_cues']}")
    print(f"{args.name}: 单元 {summary['n_units']}，半截话单元 {summary['fragmented_units']}，"
          f"代词可疑 {summary['pronoun_units']}，假名残留 {summary['kana_segments']}")
    print(f"报告 {out}")


if __name__ == "__main__":
    main()
