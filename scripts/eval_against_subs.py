#!/usr/bin/env python
"""Score a run against the film's own burned-in subtitles.

The test material carries two lines of hard-coded subtitles (Chinese on top,
Japanese underneath).  That Japanese line is ground truth for what was
actually said, so comparing it with ``asr.segments[*].text`` separates a
recognition error from a translation error instead of guessing from the
Chinese alone.

Nothing here can read pixels on its own — step 1 lays the subtitle band of
every segment out as numbered contact sheets, a human (or Claude) types the
Japanese into a JSON file, and step 2 turns that into a score.  The typing is
the only manual part and it is reusable: the ground truth for a given input
video never changes, so re-score after every pipeline change for free.

    # 1. contact sheets + an empty ground-truth skeleton
    python scripts/eval_against_subs.py output_test --make-tiles

    # 2. after filling in workspace/output_test/eval/gt_ja.json
    python scripts/eval_against_subs.py output_test \
        --ref-srt inputs/output_test.zh.srt

Both the Japanese ground truth and the optional reference-translation SRT are
*editorial* texts: official subtitlers drop fillers ("ちょっと", "うん") and
reword freely, so a low score on one line is a lead to read, not a verdict.
Read ``09_subs_eval.md`` before believing any single row.
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

# Bottom band of a 1080p frame that holds both subtitle lines.  Both lines
# must land inside it even when the Chinese wraps to two lines, so the band
# is deliberately taller than one line of text.
BAND = {"w": 1920, "h": 210, "y": 870}
TILE_ROWS = 8            # strips per contact sheet
STRIP_W = 1280           # downscaled strip width; text stays legible here

_PUNCT = re.compile(r"[、。，．！？!?…·・\-—～~「」『』（）()\s　,.]")
_ZH_PUNCT = re.compile(r"[（）()　\s、。，．！？!?…·・\-—～~「」『』]")


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


def load_segments(workspace: Path) -> list[dict]:
    state = json.loads((workspace / "state.json").read_text(encoding="utf-8"))
    for stage in ("translate", "asr"):
        if stage in state and state[stage].get("segments"):
            return state[stage]["segments"]
    raise SystemExit(f"{workspace}/state.json has no asr/translate segments")


def find_video(name: str, state_dir: Path) -> Path:
    state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
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

    skeleton = outdir / "gt_ja.json"
    if not skeleton.exists():
        skeleton.write_text(
            json.dumps({str(i): "" for i in range(len(segments))},
                       ensure_ascii=False, indent=1),
            encoding="utf-8")
    return tiles


# ── step 2: scoring ─────────────────────────────────────────────────

BANDS = [
    (0.95, 1.01, "几乎逐字一致"),
    (0.85, 0.95, "很接近（只差语气词）"),
    (0.70, 0.85, "大意对但有出入"),
    (0.50, 0.70, "部分错"),
    (0.00, 0.50, "基本错"),
]


def build_groups(segments: list[dict], gt: dict[str, str]) -> list[dict]:
    """Collapse consecutive segments that share one on-screen subtitle.

    A subtitle line usually outlives several ASR segments, so scoring per
    segment would punish us for splitting a sentence the subtitler kept
    whole.  Grouping compares like with like and, as a side effect, counts
    how often our segmentation is finer than the subtitle's.
    """
    groups = []
    for ref, run in groupby(range(len(segments)), key=lambda i: gt.get(str(i), "")):
        idxs = list(run)
        ours = "".join((segments[i].get("text") or "").strip() for i in idxs)
        zh = " / ".join((segments[i].get("text_translated") or "").strip()
                        for i in idxs)
        groups.append({
            "idxs": idxs,
            "start": float(segments[idxs[0]]["start"]),
            "end": float(segments[idxs[-1]]["end"]),
            "ref_ja": ref,
            "asr_ja": ours,
            "ours_zh": zh,
            "sim": ratio(fold_ja(ref), fold_ja(ours)) if ref else None,
        })
    return groups


def attach_reference_zh(groups: list[dict], cues: list[tuple[float, float, str]]) -> None:
    for g in groups:
        hits = [t for st, en, t in cues
                if min(g["end"], en) - max(g["start"], st) > 0.25]
        g["ref_zh"] = " ".join(t.replace("\n", " ") for t in hits)
        g["zh_sim"] = (ratio(fold_zh(g["ref_zh"]), fold_zh(g["ours_zh"]))
                       if g["ref_zh"] and g["ours_zh"] else None)


def render(name: str, groups: list[dict], segments: list[dict],
           worst: int) -> str:
    scored = [g for g in groups if g["sim"] is not None]
    blank = [g for g in groups if g["sim"] is None]
    if not scored:
        raise SystemExit("ground truth is empty — fill in gt_ja.json first")

    covered = sum(len(g["idxs"]) for g in scored)
    sims = [g["sim"] for g in scored]
    out = [f"# 字幕对照评测 — {name}", ""]
    out += [
        f"- 画面字幕句数 **{len(scored)}**，覆盖 ASR 段 **{covered}/{len(segments)}**",
        f"- 画面无字幕的段 **{len(segments) - covered}**（官方字幕不给语气词/重叠语，不等于幻听）",
        f"- ASR 与画面原文相似度：**中位数 {statistics.median(sims):.2f}**，"
        f"平均 {sum(sims) / len(sims):.2f}",
        "",
        "| 吻合度 | 句数 | 占比 |",
        "|---|---:|---:|",
    ]
    for lo, hi, label in BANDS:
        n = sum(1 for s in sims if lo <= s < hi)
        out.append(f"| {label} | {n} | {n / len(sims) * 100:.0f}% |")

    multi = [g for g in scored if len(g["idxs"]) > 1]
    out += [
        "",
        f"- 切分粒度：我们比字幕细 **{covered / len(scored):.2f} 倍**；"
        f"**{len(multi)} 句（{len(multi) / len(scored) * 100:.0f}%）**被切成多段，"
        "这些句子的译文是逐段独立生成的，是「半截话」的来源",
        "",
        f"## ASR 差异最大的 {worst} 句", "",
    ]
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

    if blank:
        out += [f"## 画面无字幕的 {len(blank)} 组（幻听候选，需人工判断）", ""]
        for g in blank:
            out += [f"**[{g['start']:.1f}s] #{g['idxs'][0]}–{g['idxs'][-1]}**",
                    f"- ASR：{g['asr_ja']}", f"- 译文：{g['ours_zh']}", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="workspace name, e.g. output_test")
    ap.add_argument("--make-tiles", action="store_true",
                    help="render contact sheets + a gt_ja.json skeleton, then stop")
    ap.add_argument("--gt", type=Path, default=None,
                    help="ground-truth JSON (default: workspace/<name>/eval/gt_ja.json)")
    ap.add_argument("--ref-srt", type=Path, default=None,
                    help="reference translation SRT, for the translation-side table")
    ap.add_argument("--worst", type=int, default=15,
                    help="how many worst lines to list (default 15)")
    ap.add_argument("--out", type=Path, default=None,
                    help="report path (default: deliver/<name>/09_subs_eval.md)")
    args = ap.parse_args()

    workspace = ROOT / "workspace" / args.name
    if not (workspace / "state.json").exists():
        raise SystemExit(f"no workspace at {workspace}")
    segments = load_segments(workspace)
    evaldir = workspace / "eval"
    evaldir.mkdir(parents=True, exist_ok=True)

    if args.make_tiles:
        video = find_video(args.name, workspace)
        tiles = make_tiles(video, segments, evaldir)
        print(f"contact sheets: {tiles}  ({len(list(tiles.glob('*.png')))} sheets, "
              f"{len(segments)} segments)")
        print(f"now type the on-screen Japanese into {evaldir / 'gt_ja.json'}, "
              f"then re-run without --make-tiles")
        return

    gt_path = args.gt or (evaldir / "gt_ja.json")
    if not gt_path.exists():
        raise SystemExit(f"no ground truth at {gt_path} — run --make-tiles first")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    groups = build_groups(segments, gt)
    if args.ref_srt:
        attach_reference_zh(groups, parse_srt(args.ref_srt))
    else:
        for g in groups:
            g["ref_zh"], g["zh_sim"] = "", None

    report = render(args.name, groups, segments, args.worst)
    out = args.out or (ROOT / "deliver" / args.name / "09_subs_eval.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    (evaldir / "subs_eval.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=1), encoding="utf-8")

    scored = [g for g in groups if g["sim"] is not None]
    sims = [g["sim"] for g in scored]
    print(f"{args.name}: {len(scored)} 句对照，ASR 相似度中位数 "
          f"{statistics.median(sims):.2f}，平均 {sum(sims) / len(sims):.2f}")
    print(f"报告 {out}")
    print(f"明细 {evaldir / 'subs_eval.json'}")


if __name__ == "__main__":
    main()
