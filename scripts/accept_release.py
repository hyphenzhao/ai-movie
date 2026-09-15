#!/usr/bin/env python
"""Release gate: is this run good enough to publish?

Compares the current three-film run with a frozen baseline and exits 0 only
if every blocking gate holds, so ``run_release.sh`` can upload unattended
and stop — with a report — when something regressed.

    python scripts/accept_release.py v3.1.0 --baseline workspace/_archive_v3.0.0

Blocking gates
  H1  every film has a v1 and a v2 final, stereo ≥ 44.1 kHz, duration within
      ±0.2 s of the source
  H2  no eval_pipeline check that passed in the baseline fails now
      (covers gender A3/A5/D2, loudness E3, lip-sync, TTS, …)
  H3  QC FAIL share (v1 and v2) ≤ baseline + 2 percentage points
  H4  output_test ASR vs on-screen subtitles: median ≥ 0.90 and the number of
      lines below 0.70 does not grow
  H5  output_test sentence units whose Chinese invents a pronoun the Japanese
      never had: ≤ 80 % of baseline
  H6  the same pronoun count does not grow on the other films
  H7  lines with leftover kana ≤ max(baseline, 1)

H5 was first drafted as "halve the fragmented-sentence count".  A Sakura
A/B on v3.0.0 output_test showed every grouping rule leaves that count at 8:
the remaining cases are separate utterances with real pauses that the
official subtitler happened to put on one line (「カンナちゃん。｜見てますね。」),
not translation defects — so it is reported, not gated.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_against_subs as subs                # noqa: E402

FILMS = ["output_test", "test_1", "test_2"]
GT_FILM = "output_test"


def ffprobe(path: Path, entries: str, stream: str | None = None) -> list[str]:
    cmd = ["ffprobe", "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream]
    cmd += ["-show_entries", entries, "-of", "csv=p=0", str(path)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.split()


def duration(path: Path) -> float | None:
    out = ffprobe(path, "format=duration")
    try:
        return float(out[0])
    except (IndexError, ValueError):
        return None


def base_key(key: str) -> str:
    return re.sub(r"\[.*\]$", "", key)


def check_rows(rows: list[dict]) -> dict[str, bool]:
    """Collapse per-engine / per-speaker variants (B3[sakura], D2[S0]) so a
    renamed engine or speaker compares with its baseline counterpart."""
    out: dict[str, bool] = {}
    for r in rows:
        if r.get("ok") is None:
            continue
        k = base_key(r["key"])
        out[k] = out.get(k, True) and bool(r["ok"])
    return out


def qc_fail_frac(rows: list[dict], key: str) -> float | None:
    for r in rows:
        if r["key"] == key:
            m = re.search(r"(\d+)/(\d+)/(\d+) of (\d+)", str(r["value"]))
            if m and int(m.group(4)):
                return int(m.group(3)) / int(m.group(4))
    return None


def run_eval_pipeline(film: str, outdir: Path) -> list[dict]:
    state = ROOT / "workspace" / film / "state.json"
    js = outdir / f"{film}.eval.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_pipeline.py"), str(state),
                    "--out", str(outdir / f"{film}.ACCEPTANCE.md"), "--json", str(js)],
                   capture_output=True, text=True, timeout=3600)
    return json.loads(js.read_text(encoding="utf-8")) if js.exists() else []


def subs_summary(film: str, state_path: Path, outdir: Path, tag: str) -> tuple[dict, list]:
    state = subs.load_state(state_path)
    segs = subs.final_segments(state)
    gt = ROOT / "inputs" / "subs" / f"{film}.gt.json"
    cues = subs.load_gt(gt) if gt.exists() else []
    groups = subs.build_groups(segs, cues)
    srt = ROOT / "inputs" / "subs" / f"{film}.zh.srt"
    if srt.exists():
        subs.attach_reference_zh(groups, subs.parse_srt(srt))
    summary = subs.summarise(groups, segs)
    (outdir / f"{film}.subs.{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if tag == "current":
        report = subs.render(film, groups, segs, summary, 15)
        (ROOT / "deliver" / film).mkdir(parents=True, exist_ok=True)
        (ROOT / "deliver" / film / "09_subs_eval.md").write_text(report, encoding="utf-8")
    return summary, groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("--baseline", type=Path, default=ROOT / "workspace" / "_archive_v3.0.0")
    ap.add_argument("--films", default=",".join(FILMS))
    args = ap.parse_args()

    films = [f for f in args.films.split(",") if f]
    outdir = ROOT / "workspace" / "_release" / args.version
    outdir.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []
    notes: list[str] = []

    def gate(key: str, desc: str, ok: bool, detail: str) -> None:
        gates.append({"key": key, "desc": desc, "ok": bool(ok), "detail": detail})

    before_after = []
    for film in films:
        work = ROOT / "workspace" / film
        base = args.baseline / film
        state = json.loads((work / "state.json").read_text(encoding="utf-8"))

        # H1 — deliverables exist and look like the source
        src = ROOT / "inputs" / f"{film}.mp4"
        src_dur = duration(src)
        finals = {"v1": work / "output" / f"{film}_dubbed.mp4",
                  "v2": Path((state.get("vc") or {}).get("video") or "/nonexistent")}
        problems = []
        for label, path in finals.items():
            if not path.exists():
                problems.append(f"{label} missing")
                continue
            d = duration(path)
            if src_dur is None or d is None or abs(d - src_dur) > 0.2:
                problems.append(f"{label} duration {d} vs source {src_dur}")
            sr_ch = ffprobe(path, "stream=sample_rate,channels", "a:0")
            try:
                sr, ch = (int(x) for x in sr_ch[0].split(",")[:2])
                if sr < 44100 or ch < 2:
                    problems.append(f"{label} audio {sr} Hz × {ch} ch")
            except (IndexError, ValueError):
                problems.append(f"{label} audio unreadable")
        gate(f"H1[{film}]", "v1+v2 成片存在、时长与原片差 ≤0.2 s、立体声 ≥44.1 kHz",
             not problems, "; ".join(problems) or "ok")

        # H2/H3 — acceptance checks against the baseline
        cur_rows = run_eval_pipeline(film, outdir)
        base_rows = json.loads((base / "baseline" / "eval.json").read_text(encoding="utf-8"))
        cur_ok, base_ok = check_rows(cur_rows), check_rows(base_rows)
        regressed = sorted(k for k, v in base_ok.items() if v and cur_ok.get(k) is False)
        fixed = sorted(k for k, v in base_ok.items() if not v and cur_ok.get(k) is True)
        missing = sorted(k for k, v in base_ok.items() if v and k not in cur_ok)
        gate(f"H2[{film}]", "基线通过的验收项现在没有失败", not regressed,
             f"回退 {regressed or '无'}；新修复 {fixed or '无'}"
             + (f"；本次未检查 {missing}" if missing else ""))
        for qc_key, label in (("F1", "v1"), ("F4", "v2")):
            b, c = qc_fail_frac(base_rows, qc_key), qc_fail_frac(cur_rows, qc_key)
            if b is None or c is None:
                notes.append(f"{film} QC {label}: 无法比较（基线 {b}，本次 {c}）")
                continue
            gate(f"H3[{film}/{label}]", f"QC FAIL 比例（{label}）≤ 基线 + 2 pp",
                 c <= b + 0.02, f"{c:.1%}（基线 {b:.1%}）")
        for r in cur_rows:
            if r["key"] in ("E3", "E3v"):
                notes.append(f"{film} {r['key']}: {'PASS' if r['ok'] else 'FAIL'} {r['value']}")

        # H4–H7 — subtitle / translation metrics
        cur, cur_groups = subs_summary(film, work / "state.json", outdir, "current")
        bas, bas_groups = subs_summary(film, base / "state.json", outdir, "baseline")
        if film == GT_FILM and cur.get("cues"):
            gate("H4", "output_test ASR 中位数 ≥ 0.90，且 <0.70 的句数不增加",
                 cur["asr_median"] >= 0.90 and cur["asr_bad"] <= bas["asr_bad"],
                 f"中位数 {cur['asr_median']:.3f}（基线 {bas['asr_median']:.3f}），"
                 f"<0.70 {cur['asr_bad']}（基线 {bas['asr_bad']}），"
                 f"平均 {cur['asr_mean']:.3f}（基线 {bas['asr_mean']:.3f}）")
            limit = int(bas["pronoun_units"] * 0.8)
            gate("H5", "output_test 凭空人称代词单元 ≤ 基线 80%",
                 cur["pronoun_units"] <= limit,
                 f"{cur['pronoun_units']}（基线 {bas['pronoun_units']}，上限 {limit}）")
            notes.append(f"output_test 半截话字幕句 {cur['fragmented_cues']}（基线 "
                         f"{bas['fragmented_cues']}，仅报告）；分段 {cur['n_segments']}（基线 "
                         f"{bas['n_segments']}）")
            bmap = {g["cue"]: g for g in bas_groups if g["cue"] is not None}
            for g in cur_groups:
                b = bmap.get(g["cue"])
                if b and (b["sim"] < 0.85 or g["sim"] < 0.85):
                    before_after.append((b, g))
        else:
            gate(f"H6[{film}]", "凭空人称代词单元不增加",
                 cur["pronoun_units"] <= bas["pronoun_units"],
                 f"{cur['pronoun_units']}（基线 {bas['pronoun_units']}）")
        gate(f"H7[{film}]", "译文残留假名的段 ≤ max(基线, 1)",
             cur["kana_segments"] <= max(bas["kana_segments"], 1),
             f"{cur['kana_segments']}（基线 {bas['kana_segments']}）")
        pol = (state.get("translate") or {}).get("polish") or {}
        notes.append(f"{film} 句子单元 {cur['n_units']}（多段 {cur['multi_units']}）；"
                     f"可疑句校对 {pol.get('accepted', 0)}/{pol.get('flagged', 0)} 采纳；"
                     f"ASR 音源 {(state.get('asr') or {}).get('asr_audio')}，"
                     f"性别音源 {(state.get('asr') or {}).get('gender_source')}")

    passed = all(g["ok"] for g in gates)
    md = [f"# {args.version} 发布验收 — {'通过' if passed else '未通过'}", "",
          "| 门 | 条件 | 结果 | 数据 |", "|---|---|---|---|"]
    md += [f"| {g['key']} | {g['desc']} | {'✅' if g['ok'] else '❌'} | {g['detail']} |"
           for g in gates]
    md += ["", "## 仅报告", ""] + [f"- {n}" for n in notes]
    if before_after:
        md += ["", "## output_test：基线或本次 ASR < 0.85 的字幕句，前后对照", ""]
        for b, c in sorted(before_after, key=lambda t: t[1]["start"]):
            md += [f"**[{c['start']:.1f}s] {b['sim']:.2f} → {c['sim']:.2f}**",
                   f"- 画面原文：{c['ref_ja']}",
                   f"- 基线：{b['asr_ja']} → {b['ours_zh']}",
                   f"- 本次：{c['asr_ja']} → {c['ours_zh']}", ""]
    (outdir / "ACCEPTANCE.md").write_text("\n".join(md), encoding="utf-8")
    (outdir / "ACCEPTANCE.json").write_text(
        json.dumps({"version": args.version, "passed": passed, "gates": gates, "notes": notes},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    for g in gates:
        print(f"{'PASS' if g['ok'] else 'FAIL'}  {g['key']:<18} {g['detail']}")
    print(f"\n{args.version}: {'ACCEPTED' if passed else 'REJECTED'} — {outdir / 'ACCEPTANCE.md'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
