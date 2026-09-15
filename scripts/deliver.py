#!/usr/bin/env python
"""Assemble (and optionally upload) the customer deliverable for one workspace.

    <out>/<name>_dubbed.mp4          the final film (v2 cloned by default)
    <out>/<name>_dubbed/             demo1..3 (30 s each), one side-by-side
                                     原片 vs 配音, windows.json, the QC review
                                     list and ACCEPTANCE.md

    python scripts/deliver.py workspace/test_1/state.json --out deliver/ --version vc
    python scripts/deliver.py workspace/test_1/state.json --out deliver/ --upload

``--upload`` pushes both to the Google Drive folder with rclone (remote
``gdrive``, root folder id from ``--drive-folder``); ``--dry-run`` prints
the commands instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

DRIVE_FOLDER = "187u7Emwab8T_oc4FEfdFjBoRe1On7igO"


def _rclone() -> str | None:
    for p in (Path.home() / ".local" / "bin" / "rclone", Path("/usr/bin/rclone"),
              Path("/usr/local/bin/rclone")):
        if p.exists():
            return str(p)
    return shutil.which("rclone")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("state")
    ap.add_argument("--out", default="deliver")
    ap.add_argument("--version", default="vc", choices=["vc", "v1"],
                    help="vc = cloned voice (state['vc']), v1 = built-in voice (compose)")
    ap.add_argument("--name", default=None, help="deliverable base name (default: workspace)")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remote", default="gdrive")
    ap.add_argument("--drive-folder", default=DRIVE_FOLDER)
    ap.add_argument("--drive-subdir", default="",
                    help="upload into this subfolder (e.g. v3.1.0) instead of the Drive root")
    ap.add_argument("--extra", action="append", default=[],
                    help="extra report file to copy into the demo folder (repeatable)")
    args = ap.parse_args()

    from make_demo_clips import make_clips, resolve_source

    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    name = args.name or state_path.parent.name
    deliver = state_path.parent / "deliverables"

    if args.version == "vc":
        video = (state.get("vc") or {}).get("video")
        label = "原声音色"
    else:
        video = (state.get("compose") or {}).get("video")
        label = "标准音色"
    if video and not Path(video).is_absolute():
        video = str(ROOT / video)
    if not video or not Path(video).exists():
        print(f"no {args.version} video in state ({video})")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"{name}_dubbed.mp4"
    folder = out / f"{name}_dubbed"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copy2(video, final)
    print(f"final: {final} ({final.stat().st_size / 1e6:.1f} MB)")

    src = resolve_source(state, state_path)
    rc = make_clips(state, folder, video, src, label, args.n)
    if rc != 0:
        return rc

    qc_suffix = "_vc" if args.version == "vc" else ""
    for fn in (f"06_qc{qc_suffix}_review.csv", f"06_qc{qc_suffix}_summary.txt",
               "ACCEPTANCE.md", "03_compact_report.csv"):
        p = deliver / fn
        if p.exists():
            shutil.copy2(p, folder / fn)
    for extra in args.extra:
        if Path(extra).exists():
            shutil.copy2(extra, folder / Path(extra).name)
    print(f"folder: {folder} → {sorted(p.name for p in folder.iterdir())}")

    if args.upload or args.dry_run:
        rclone = _rclone()
        if not rclone:
            print("rclone not installed (expected ~/.local/bin/rclone)")
            return 2
        common = ["--drive-root-folder-id", args.drive_folder, "--checksum", "-P"]
        base = f"{args.drive_subdir.strip('/')}/" if args.drive_subdir else ""
        cmds = [
            [rclone, "copy", str(final), f"{args.remote}:{base}", *common],
            [rclone, "copy", str(folder) + "/", f"{args.remote}:{base}{folder.name}", *common,
             "--transfers", "4"],
            [rclone, "lsl", f"{args.remote}:{base}", "--drive-root-folder-id", args.drive_folder],
        ]
        for c in cmds:
            print("$ " + " ".join(c))
            if not args.dry_run:
                r = subprocess.run(c, text=True, capture_output=True)
                print(r.stdout[-2000:] if r.stdout else "", r.stderr[-800:] if r.returncode else "")
                if r.returncode != 0:
                    print(f"rclone failed (rc={r.returncode})")
                    return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
