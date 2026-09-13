#!/usr/bin/env python
"""Build the static preview site that is served from the VPS.

Why it exists: the ROCm host sits behind a VPN whose *upstream* is ~20 Mbps
with a 169 ms RTT, so streaming a film from it through the SSH tunnel is the
slowest link in the chain (measured 16.5 Mbps).  Copying the finished films
onto the VPS once and serving them straight from Caddy removes that leg
entirely — the VPS pulls/pushes at ~106 Mbps.

The site groups every finished film by release (newest first), and inside a
release by source film and by voice version (原片 / v1 内置音色 / v2 原声音色)
plus its demo clips.  Output goes to ``workspace/_preview_site`` and is then
rsync'd to ``/var/www/preview`` on the VPS.

    python scripts/build_preview_site.py            # build locally
    python scripts/build_preview_site.py --upload   # build + rsync to the VPS
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "workspace" / "_preview_site"
VPS = "root@64.176.52.137"
VPS_DIR = "/var/www/preview"

# ── the curated version map ────────────────────────────────────────
#
# Only *finished films* and their demo clips: the intermediates
# (lipsync.mp4 without final audio, video_silent.mp4, A/B renders, cut
# segments) are development artifacts, not versions of the movie.

FILMS = {
    "output_test": "访谈 390 秒（双人，男主持画外）",
    "test_1": "车内对谈 185 秒",
    "test_2": "剧情多机位 540 秒",
}

RELEASES = [
    {
        "id": "v3.0.0",
        "title": "v3.0.0— 当前版本",
        "date": "2026-09-13",
        "note": "P0/P1 全部改动：立体声全采样率背景床、台词压缩、F0 参考音门、"
                "切镜检测、小脸超分、逐段 QC、重叠语音检测。三部片全自动从零重跑。",
        "items": [
            ("output_test", "原片", "inputs/output_test.mp4"),
            ("output_test", "v1 内置音色", "workspace/output_test/output/output_test_dubbed.mp4"),
            ("output_test", "v2 原声音色", "workspace/output_test/output/v2_cloned_dubbed.mp4"),
            ("test_1", "原片", "inputs/test_1.mp4"),
            ("test_1", "v1 内置音色", "workspace/test_1/output/test_1_dubbed.mp4"),
            ("test_1", "v2 原声音色", "workspace/test_1/output/v2_cloned_dubbed.mp4"),
            ("test_2", "原片", "inputs/test_2.mp4"),
            ("test_2", "v1 内置音色", "workspace/test_2/output/test_2_dubbed.mp4"),
            ("test_2", "v2 原声音色", "workspace/test_2/output/v2_cloned_dubbed.mp4"),
        ],
        "demos": [
            ("output_test", "deliver/output_test_dubbed"),
            ("test_1", "deliver/test_1_dubbed"),
            ("test_2", "deliver/test_2_dubbed"),
        ],
    },
    {
        "id": "v2.0.0",
        "title": "v2.0.0— 上一版本",
        "date": "2026-09-11",
        "note": "分割/说话人日志、术语表、原声克隆、按人脸口型、侧脸门控 + CodeFormer。"
                "output_test 的说话人标签经过人工审核。",
        "items": [
            ("output_test", "v1 内置音色",
             "workspace/_archive_v2/output_test/deliverables/v1_standard/05_final_dubbed.mp4"),
            ("output_test", "v2 原声音色",
             "workspace/_archive_v2/output_test/deliverables/v2_cloned/05_final_dubbed.mp4"),
            ("test_1", "v1 内置音色",
             "workspace/_archive_v2/test_1/deliverables/v1_standard/05_final_dubbed.mp4"),
            ("test_1", "v2 原声音色",
             "workspace/_archive_v2/test_1/deliverables/v2_cloned/05_final_dubbed.mp4"),
            ("test_2", "v1 内置音色",
             "workspace/_archive_v2/test_2/deliverables/v1_standard/05_final_dubbed.mp4"),
            ("test_2", "v2 原声音色",
             "workspace/_archive_v2/test_2/deliverables/v2_cloned/05_final_dubbed.mp4"),
        ],
        "demos": [
            ("output_test", "workspace/_archive_v2/output_test/deliverables/v2_cloned/demo"),
            ("test_1", "workspace/_archive_v2/test_1/deliverables/v2_cloned/demo"),
            ("test_2", "workspace/_archive_v2/test_2/deliverables/v2_cloned/demo"),
        ],
    },
    {
        "id": "v2-prev",
        "title": "v2 早期轮次— 口型清晰度改动之前",
        "date": "2026-09-11",
        "note": "侧脸门控、MuseTalk 质量补丁与 CodeFormer 默认开启*之前*的一轮渲染，"
                "留作对照：嘴部锐度约为源片的一半。",
        "items": [
            ("test_1", "v1 内置音色",
             "workspace/_archive_v2/test_1/deliverables/_prev_round/05_final_dubbed.mp4"),
            ("test_1", "v2 原声音色",
             "workspace/_archive_v2/test_1/deliverables/_prev_round/05_final_dubbed_v2.mp4"),
            ("test_2", "v1 内置音色",
             "workspace/_archive_v2/test_2/deliverables/_prev_round/05_final_dubbed.mp4"),
            ("test_2", "v2 原声音色",
             "workspace/_archive_v2/test_2/deliverables/_prev_round/05_final_dubbed_v2.mp4"),
        ],
        "demos": [],
    },
    {
        "id": "v1",
        "title": "v1— 最早版本",
        "date": "2026-07",
        "note": "单一流程、无说话人日志、无人脸锚定的早期成片，按时间倒序。",
        "items": [
            ("output_test", "手动性别演示 (07-16)", "test_data/results/output_test_MANUAL_GENDER_demo.mp4"),
            ("output_test", "v3 配音 (07-14)", "test_data/results/output_test_v3_dubbed.mp4"),
            ("output_test", "v2 配音 (07-11)", "test_data/results/output_test_v2_dubbed.mp4"),
            ("output_test", "全片配音 (07-11)", "test_data/results/output_test_FULL_dubbed.mp4"),
            ("_pipe_90s", "90 秒验证片 (07-11)", "test_data/results/VALIDATION_90s_dubbed.mp4"),
            ("_pipe_90s", "流水线测试片 (09-08)", "workspace/_pipe_test_90s/output/_pipe_test_90s_dubbed.mp4"),
        ],
        "demos": [],
    },
]

FILMS["_pipe_90s"] = "90 秒测试片"


def probe(p: Path) -> dict:
    try:
        out = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size,bit_rate:stream=codec_type,width,height,sample_rate,channels",
            "-of", "json", str(p)], capture_output=True, text=True, timeout=60).stdout
        d = json.loads(out)
        fmt = d.get("format") or {}
        v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
        a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), {})
        return {
            "duration": round(float(fmt.get("duration") or 0), 1),
            "size": int(fmt.get("size") or p.stat().st_size),
            "bitrate": int(int(fmt.get("bit_rate") or 0) / 1000),
            "width": v.get("width"), "height": v.get("height"),
            "audio": f"{int(a['sample_rate']) // 1000} kHz × {a['channels']}ch" if a.get("sample_rate") else "—",
        }
    except Exception as exc:                            # noqa: BLE001
        print(f"  ffprobe failed for {p}: {exc}", file=sys.stderr)
        return {"duration": 0, "size": p.stat().st_size if p.exists() else 0,
                "bitrate": 0, "width": None, "height": None, "audio": "—"}


def slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")
    return s or "x"


def demo_label(name: str) -> str:
    if "对比" in name:
        return "原片 vs 配音 并排对比"
    m = re.match(r"demo(\d+)_(\d+)-(\d+)s", name)
    return f"片段 {m.group(1)}（{m.group(2)}–{m.group(3)} 秒）" if m else name


def build() -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "media").mkdir(parents=True)
    manifest = {"releases": [], "films": FILMS}
    total = 0

    for rel in RELEASES:
        entry = {k: rel[k] for k in ("id", "title", "date", "note")}
        entry["films"] = {}
        for film, variant, rel_path in rel["items"]:
            src = ROOT / rel_path
            if not src.exists():
                print(f"  缺失，跳过: {rel_path}")
                continue
            meta = probe(src)
            dst_rel = f"media/{slug(rel['id'])}/{slug(film)}/{slug(variant)}.mp4"
            dst = OUT / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)                  # rsync -L copies the real file
            total += meta["size"]
            entry["films"].setdefault(film, {"variants": [], "demos": []})
            entry["films"][film]["variants"].append({"label": variant, "url": dst_rel, **meta})
        for film, ddir in rel["demos"]:
            d = ROOT / ddir
            if not d.is_dir():
                continue
            for i, f in enumerate(sorted(d.glob("*.mp4"))):
                meta = probe(f)
                dst_rel = f"media/{slug(rel['id'])}/{slug(film)}/demo-{i}-{slug(f.stem)[:40]}.mp4"
                dst = OUT / dst_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(f)
                total += meta["size"]
                entry["films"].setdefault(film, {"variants": [], "demos": []})
                entry["films"][film]["demos"].append({"label": demo_label(f.name), "url": dst_rel, **meta})
        if entry["films"]:
            manifest["releases"].append(entry)

    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    manifest["_total_bytes"] = total
    print(f"清单：{len(manifest['releases'])} 个版本，"
          f"{sum(len(f['variants']) + len(f['demos']) for r in manifest['releases'] for f in r['films'].values())} 个文件，"
          f"共 {total / 1e9:.2f} GB")
    return manifest


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Movie 成片预览</title>
<style>
:root { --bg:#11131a; --panel:#1b1f2a; --line:#2c3242; --text:#e6e9f0; --muted:#98a0b3; --blue:#5b9dff; --green:#4ade80; }
*{box-sizing:border-box} html,body{margin:0;background:var(--bg);color:var(--text)}
body{font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif;font-size:15px;line-height:1.6}
header{padding:18px 24px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:10}
h1{margin:0;font-size:19px} .sub{color:var(--muted);font-size:13px;margin-top:4px}
main{padding:18px 24px;max-width:1200px;margin:0 auto}
.rel{margin-bottom:26px;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:var(--panel)}
.rel>summary{padding:14px 18px;cursor:pointer;font-size:17px;font-weight:600;list-style:none;display:flex;gap:10px;align-items:baseline}
.rel>summary::-webkit-details-marker{display:none}
.rel>summary::before{content:"▸";color:var(--muted)} .rel[open]>summary::before{content:"▾"}
.tag{font-size:12px;color:var(--muted);font-weight:400}
.badge{font-size:11px;background:#2b3550;color:var(--blue);padding:2px 8px;border-radius:10px;font-weight:400}
.note{padding:0 18px 12px 40px;color:var(--muted);font-size:13px}
.film{border-top:1px solid var(--line);padding:14px 18px}
.film h3{margin:0 0 8px;font-size:15px} .film h3 small{color:var(--muted);font-weight:400;margin-left:8px}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
button{font:inherit;background:#232939;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
button:hover{background:#2c3446} button.on{background:var(--blue);border-color:var(--blue);color:#0b1020;font-weight:600}
button.demo{background:#1f2a24;border-color:#2d4038}
video{width:100%;max-height:70vh;background:#000;border-radius:10px;display:block}
.meta{color:var(--muted);font-size:12px;margin-top:6px;display:flex;gap:14px;flex-wrap:wrap}
.meta a{color:var(--blue);text-decoration:none} .meta a:hover{text-decoration:underline}
.hint{color:var(--muted);font-size:12px;margin:6px 0 0}
@media(max-width:700px){main{padding:12px}.film{padding:12px}}
</style>
</head>
<body>
<header>
  <h1>AI Movie 成片预览</h1>
  <div class="sub">按版本分组，最新在上。视频直接由本服务器提供，不经过内网隧道。<span id="stat"></span></div>
</header>
<main id="app">加载中…</main>
<script>
const fmtSize = b => b > 1e9 ? (b/1e9).toFixed(2)+' GB' : (b/1e6).toFixed(0)+' MB';
const fmtDur = s => { s = Math.round(s); const m = Math.floor(s/60); return `${m}:${String(s%60).padStart(2,'0')}`; };
const el = (t, a = {}, ...k) => { const e = document.createElement(t);
  for (const [n, v] of Object.entries(a)) { if (n === 'class') e.className = v; else if (n.startsWith('on')) e.addEventListener(n.slice(2), v); else if (v != null) e.setAttribute(n, v); }
  k.flat().forEach(c => c != null && e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c)); return e; };

fetch('manifest.json').then(r => r.json()).then(m => {
  const app = document.getElementById('app'); app.innerHTML = '';
  let n = 0, bytes = 0;
  m.releases.forEach((rel, ri) => {
    const body = el('div');
    for (const [film, data] of Object.entries(rel.films)) {
      const all = [...data.variants.map(v => ({...v, demo:false})), ...data.demos.map(v => ({...v, demo:true}))];
      n += all.length; all.forEach(v => bytes += v.size);
      const video = el('video', { controls:'', preload:'none', playsinline:'' });
      const meta = el('div', { class:'meta' });
      const btns = el('div', { class:'btns' });
      const pick = (v, btn) => {
        const t = video.currentTime, playing = !video.paused;
        btns.querySelectorAll('button').forEach(b => b.classList.remove('on'));
        btn.classList.add('on');
        video.src = v.url;
        if (!v.demo && t > 0) video.addEventListener('loadedmetadata', () => { video.currentTime = t; if (playing) video.play(); }, { once:true });
        meta.innerHTML = '';
        meta.append(
          el('span', {}, `${fmtDur(v.duration)}`),
          el('span', {}, `${v.width || '?'}×${v.height || '?'}`),
          el('span', {}, `${v.bitrate ? v.bitrate + ' kbps' : ''}`),
          el('span', {}, `音轨 ${v.audio}`),
          el('span', {}, fmtSize(v.size)),
          el('a', { href: v.url, download:'' }, '下载'));
      };
      all.forEach((v, i) => { const b = el('button', { class: v.demo ? 'demo' : '', onclick: () => pick(v, b) }, v.label); btns.appendChild(b); if (i === 0) setTimeout(() => pick(v, b), 0); });
      body.appendChild(el('div', { class:'film' },
        el('h3', {}, film, el('small', {}, m.films[film] || '')),
        btns, video, meta,
        el('p', { class:'hint' }, '切换同一部片的不同版本会保持播放位置，方便逐处对比。')));
    }
    app.appendChild(el('details', { class:'rel', ...(ri === 0 ? { open:'' } : {}) },
      el('summary', {}, rel.title, el('span', { class:'tag' }, rel.date),
        ri === 0 ? el('span', { class:'badge' }, '最新') : null),
      el('div', { class:'note' }, rel.note), body));
  });
  document.getElementById('stat').textContent = ` 共 ${m.releases.length} 个版本 / ${n} 个文件 / ${fmtSize(bytes)}。`;
}).catch(e => { document.getElementById('app').textContent = '清单加载失败：' + e.message; });
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build()
    if args.upload or args.dry_run:
        cmd = ["rsync", "-aL", "--info=progress2", "--partial", "--inplace",
               "-e", "ssh -o BatchMode=yes -o Compression=no",
               f"{OUT}/", f"{VPS}:{VPS_DIR}/"]
        print("$ " + " ".join(cmd))
        if args.upload:
            subprocess.run(cmd, check=True)
            subprocess.run(["ssh", "-o", "BatchMode=yes", VPS,
                            f"chown -R caddy:caddy {VPS_DIR} 2>/dev/null || true; du -sh {VPS_DIR}"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
