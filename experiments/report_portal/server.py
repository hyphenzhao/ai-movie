#!/usr/bin/env python3
"""Read-only LAN access to existing text-evaluation reports.

Scope: ../../README.md#用途与处理范围 and AGENTS.md's existing-text test
authorization. This server never runs models, edits evidence, or publishes media.
Only explicit report routes are exposed; there is no general file browser.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.publish_smb import require_mount

STAGES = {
    "asr_polish": "日文转录纠错",
    "translate_raw": "原转录 → 翻译",
    "translate_polished": "纠错转录 → 翻译",
    "polish_translation_raw": "原转录 → 翻译 → 润色",
    "polish_translation_polished": "纠错转录 → 翻译 → 润色",
}
MODEL_NAMES = {
    "qwen3.6-35b-a3b": "Qwen3.6 · 35B-A3B",
    "glm-4.7-flash": "GLM-4.7-Flash",
    "qwen3.8-27b": "Qwen3.8 · 27B",
    "gemma4-31b-it": "Gemma4 · 31B",
}
REPORT_FILES = {
    "/report/": ("00_开始复核.html", "text/html"),
    "/report/结论.txt": ("结论.txt", "text/plain"),
    **{f"/report/{name}.json": (f"{name}.json", "application/json")
       for name in ("analysis", "conclusions", "metrics", "score_details")},
}
LAN = ipaddress.ip_network("192.168.50.0/24")
ACTIVE_STATES = {"running", "waiting_models", "preparing_runtime", "initializing",
                 "finalizing", "waiting_delivery"}


def read_json(path: Path):
    return json.loads(path.read_bytes())


class Portal:
    def __init__(self, run_dir: Path, mount_check=require_mount):
        self.run_dir = run_dir.absolute()
        self.mount_check = mount_check
        self.check_storage()

    def check_storage(self):
        mount = self.mount_check().absolute()
        allowed = mount / "AI-Movie/evaluations/text-models"
        if mount.resolve() != mount or allowed.resolve() != allowed:
            raise RuntimeError("Mounted evaluation ancestors must not redirect to another directory")
        if self.run_dir.is_symlink() or self.run_dir.resolve().parent != allowed:
            raise RuntimeError("Report root is outside the expected mounted evaluation directory")

    def artifact(self, relative: str) -> Path:
        self.check_storage()
        path = self.run_dir / relative
        if not path.resolve().is_relative_to(self.run_dir.resolve()):
            raise RuntimeError("Artifact points outside this evaluation")
        return path

    def snapshot(self):
        state = read_json(self.artifact("STATUS.json"))
        expected = read_json(self.artifact("full-context-b64/EXPECTED_MODELS.json"))
        total = int(state.get("target_units", 0))
        if not total:
            total = len(read_json(self.artifact("inputs/dataset.json"))["units"])
        now = datetime.now(timezone.utc)
        updated = state.get("updated_at")
        try:
            stale = state.get("state") in ACTIVE_STATES and (now - datetime.fromisoformat(updated)).total_seconds() > 120
        except (TypeError, ValueError):
            stale = True
        models = []
        for config in expected:
            mid = config["id"]
            if mid not in MODEL_NAMES:
                raise ValueError("Unexpected model identity in this experiment")
            status = state.get("models", {}).get(mid, {})
            stages = []
            for stage, label in STAGES.items():
                path = self.artifact(f"full-context-b64/{mid}/{stage}.json")
                data = read_json(path) if path.exists() else {}
                units = data.get("units", [])
                counts = Counter(u.get("status", "pending") for u in units)
                counts["pending"] += max(0, total - len(units))
                operations = Counter(u.get("operation") for u in units if u.get("status") == "success")
                stages.append({"id": stage, "label": label, "state": data.get("state", "pending"),
                               "total": total, **{k: counts[k] for k in ("success", "failed", "skipped", "pending")},
                               "operations": {k: operations[k] for k in ("copy", "replace", "host_preserve")}})
            models.append({"id": mid, "name": MODEL_NAMES[mid], "state": status.get("state", "waiting_model"),
                           "phase": status.get("phase"), "active_stage": status.get("stage"), "stages": stages})
        report = self.artifact("full-context-b64/00_开始复核.html")
        available = report.is_file()
        self.check_storage()
        return {"run_id": self.run_dir.name, "updated_at": updated, "now": now.isoformat(),
                "stale": stale, "state": state.get("state", "unknown"), "models": models,
                "total_units": total, "completed_models": state.get("completed_models", 0),
                "target_models": len(expected), "delivery_confirmed": state.get("delivery_confirmed", False),
                "report": {"available": available, "url": "/report/", "updated_at":
                           datetime.fromtimestamp(report.stat().st_mtime, timezone.utc).isoformat() if available else None}}

    def report_bytes(self, name: str):
        data = self.artifact("full-context-b64/" + name).read_bytes()
        self.check_storage()
        if name.endswith(".html"):
            # Adapt a local README link for browsers; never modify stored evidence.
            data = data.decode("utf-8").replace((ROOT / "README.md").as_uri() + "#用途与处理范围", "/scope#用途与处理范围").encode("utf-8")
        return data


def scope_page() -> bytes:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## 用途与处理范围", 1)[1].split("\n## ", 1)[0]
    body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' \
           '<title>项目用途与处理范围</title><style>body{font:17px/1.8 system-ui;max-width:900px;margin:40px auto;padding:0 24px;color:#263331}pre{white-space:pre-wrap;font:inherit}</style>' \
           '<a href="/">← 返回评测入口</a><h1 id="用途与处理范围">项目用途与处理范围</h1>' \
           '<p>本轮依用户 2026-10-07 授权，对已有转录和字幕进行忠实转换与比较。仅发布既有文本评测，不生成媒体，不构成广告、配音、口型或法律验收。</p>' \
           '<p>下文摘自项目 README 的完整范围说明，其中带日期的实现差距为当时记录。</p><pre>' + html.escape(section.strip()) + '</pre></html>'
    return body.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "EvaluationPortal"

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, fmt, *args):
        # Never log raw query strings, request bodies, or source subtitle text.
        print(f"HTTP {self.command} from {self.client_address[0]}", flush=True)

    def reply(self, status, body=b"", content_type="text/plain", head=False, **headers):
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
        for key, value in headers.items():
            self.send_header(key.replace("_", "-"), value)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def do_GET(self):
        self.serve(False)

    def do_HEAD(self):
        self.serve(True)

    def serve(self, head):
        peer = ipaddress.ip_address(self.client_address[0])
        if not (peer.is_loopback or peer in LAN):
            self.reply(403, b"LAN access only", head=head)
            return
        try:
            path = unquote(urlsplit(self.path).path, errors="strict")
            if path == "/":
                self.reply(200, (HERE / "index.html").read_bytes(), "text/html", head=head)
            elif path == "/scope":
                self.reply(200, scope_page(), "text/html", head=head)
            elif path == "/report":
                self.reply(302, head=head, Location="/report/")
            elif path == "/api/status":
                self.reply(200, json.dumps(self.server.portal.snapshot(), ensure_ascii=False).encode(), "application/json", head=head)
            elif path in REPORT_FILES:
                name, mime = REPORT_FILES[path]
                self.reply(200, self.server.portal.report_bytes(name), mime, head=head)
            else:
                self.reply(404, b"Not found", head=head)
        except FileNotFoundError:
            self.reply(404, "报告尚未生成。".encode(), head=head)
        except (RuntimeError, OSError, ValueError, KeyError, TypeError):
            self.reply(503, "共享存储或评测状态暂不可用，请稍后重试。".encode(), head=head)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="192.168.50.241")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    address = ipaddress.ip_address(args.listen)
    if address not in LAN:
        parser.error("Bind to a specific 192.168.50.0/24 LAN address")
    portal = Portal(args.run_dir)
    server = ThreadingHTTPServer((args.listen, args.port), Handler)
    server.daemon_threads = True
    server.portal = portal
    print(f"Serving read-only evaluation at http://{args.listen}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
