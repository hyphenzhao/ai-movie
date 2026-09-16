#!/usr/bin/env python
"""Can the polish model actually answer on this box?  Prints the model to use.

Ollama models above ~60 GB hang on this ROCm build, and a new architecture
(Qwen3.6's hybrid attention) may not load at all, so the release runner asks
this script before committing six hours of GPU time.  It sends three real
flagged lines from v3.0.0 output_test; the model passes if at least two come
back as a single line of Chinese within the timeout.  On failure the fallback
model is checked the same way.

    python scripts/smoke_polish_model.py [--timeout 180] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FALLBACK = "dolphin-mixtral:8x7b"
PROBES = [
    ("好きなのかも。", "我可能喜欢上你了。"),
    ("これは大きいみたいな?", "这个好像是很大？"),
    ("そういう感じでやるわけじゃない。", "不是这种感觉的她。"),
]


def installed(model: str) -> bool:
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    name = model if ":" in model else f"{model}:latest"
    return any(line.split()[0] == name for line in out.splitlines()[1:] if line.split())


def probe(model: str, timeout: int) -> dict:
    from ai_movie import translator as T
    from ai_movie.config import OLLAMA_BASE_URL

    res = {"model": model, "installed": installed(model), "answers": [], "ok": 0}
    if not res["installed"]:
        return res
    t0 = time.time()
    try:
        # stdout carries exactly one thing — the chosen model name — because
        # the release runner captures it with $(...).  Engine chatter would
        # otherwise end up inside AI_MOVIE_POLISH_MODEL and every polish
        # request would come back HTTP 400 (it did, on the first v3.1.0 run).
        with T.exclusive_engine("ollama", ollama_model=model, base_url=OLLAMA_BASE_URL,
                                log_cb=lambda m: print(f"[engine] {m}", file=sys.stderr)):
            for ja, draft in PROBES:
                try:
                    raw = T._call_ollama_chat(
                        model,
                        [{"role": "system", "content": "你是日译中字幕校对。只输出一行中文译文。"},
                         {"role": "user", "content": f"原文：{ja}\n草稿：{draft}\n"
                                                     "只修正错误，只输出一行中文。"}],
                        OLLAMA_BASE_URL, timeout=timeout, think=False,
                        options={"num_predict": 96, "temperature": 0.2})
                    ans = next(iter(T._clean_ollama_output(raw or "").strip().splitlines()), "").strip()
                except Exception as exc:                # noqa: BLE001
                    ans = f"ERROR {type(exc).__name__}: {exc}"
                good = bool(ans) and not ans.startswith("ERROR") and not T._KANA_RE.search(ans) \
                    and any("一" <= c <= "鿿" for c in ans)
                res["answers"].append({"ja": ja, "answer": ans, "good": good})
                res["ok"] += int(good)
    finally:
        subprocess.run(["ollama", "stop", model], capture_output=True)
    res["secs"] = round(time.time() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from ai_movie.config import OLLAMA_POLISH_MODEL
    tried = []
    chosen = None
    for model in [args.model or OLLAMA_POLISH_MODEL, FALLBACK]:
        r = probe(model, args.timeout)
        tried.append(r)
        print(json.dumps(r, ensure_ascii=False), file=sys.stderr)
        if r["ok"] >= 2:
            chosen = model
            break
    if args.json:
        args.json.write_text(json.dumps({"chosen": chosen, "tried": tried},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
    if not chosen:
        return 1
    print(chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
