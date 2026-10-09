"""Neutral-fixture checks for the read-only LAN evaluation portal.

IMPORTANT — Project scope: ../../README.md#用途与处理范围
Production use is advertising localization; this test exercises only synthetic
text-report access. It never accesses NAS evidence, starts a listening service,
generates media, or validates advertising/legal eligibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import quote

from experiments.report_portal import server


class MemoryConnection:
    """Socket-shaped HTTP transport with no real socket or network listener."""

    def __init__(self, request: bytes):
        self.request = io.BytesIO(request)
        self.response = bytearray()

    def makefile(self, mode, buffering=None):
        if mode != "rb":
            raise AssertionError(f"Unexpected stream mode: {mode}")
        return self.request

    def sendall(self, data):
        self.response.extend(data)

    def settimeout(self, timeout):
        self.timeout = timeout


class PortalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="neutral-report-portal-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mount = self.root / "mounted-fixture"
        self.run = self.mount / "AI-Movie/evaluations/text-models/neutral-run"
        self.full = self.run / "full-context-b64"
        self.full.mkdir(parents=True)
        self.available = True
        self.repo = self.root / "fixture-repo"
        self.ui = self.repo / "experiments/report_portal"
        self.ui.mkdir(parents=True)
        (self.repo / "README.md").write_text(
            "# Neutral fixture\n\n## 用途与处理范围\n"
            "Synthetic project scope <escaped-marker>.\n\n## Next\n"
            "This next section must not be exposed.\n", encoding="utf-8"
        )
        (self.ui / "index.html").write_text(
            '<!doctype html><a href="/report/">Report</a>'
            '<a href="/scope">Scope</a>', encoding="utf-8"
        )
        self.addCleanup(patch.stopall)
        patch.object(server, "ROOT", self.repo).start()
        patch.object(server, "HERE", self.ui).start()
        patch.object(server.Handler, "log_message", lambda *args: None).start()
        self.ids = list(server.MODEL_NAMES)
        self.state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": "running", "target_units": 7, "completed_models": 0,
            "models": {self.ids[0]: {
                "state": "running", "phase": "full", "stage": "asr_polish"
            }},
        }
        self.write_json("STATUS.json", self.state)
        self.write_json("inputs/dataset.json", {"units": [{"text": "INPUT_SENTINEL"}] * 7})
        self.write_json("full-context-b64/EXPECTED_MODELS.json", [
            {"id": mid, "model": "neutral-model"} for mid in self.ids
        ])
        self.write_json(f"full-context-b64/{self.ids[0]}/asr_polish.json", {
            "state": "running", "units": [
                {"id": "unit-1", "text": "SOURCE_SENTINEL", "status": "success", "operation": "copy"},
                {"id": "unit-2", "text": "OUTPUT_SENTINEL", "status": "success", "operation": "replace"},
                {"id": "unit-3", "text": "RETAINED_SENTINEL", "status": "success", "operation": "host_preserve"},
                {"id": "unit-4", "text": "FAILED_SENTINEL", "status": "failed", "operation": "host_preserve"},
                {"id": "unit-5", "text": "SKIPPED_SENTINEL", "status": "skipped"},
            ]
        })
        self.html = (
            '<!doctype html><h1>Neutral report</h1><a href="'
            + (self.repo / "README.md").as_uri()
            + '#用途与处理范围">Scope</a>'
        ).encode("utf-8")
        (self.full / "00_开始复核.html").write_bytes(self.html)
        (self.full / "结论.txt").write_text("Neutral conclusion.\n", encoding="utf-8")
        self.portal = server.Portal(self.run, mount_check=self.mount_check)

    def mount_check(self):
        if not self.available:
            raise RuntimeError("Synthetic mount unavailable")
        return self.mount

    def write_json(self, relative, data):
        path = self.run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def request(self, path, method="GET", body=b"", peer="127.0.0.1"):
        target = quote(path, safe="/%?=&")
        request = (
            f"{method} {target} HTTP/1.1\r\nHost: fixture.invalid\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        ).encode("ascii") + body
        connection = MemoryConnection(request)
        server.Handler(connection, (peer, 12345), SimpleNamespace(portal=self.portal))
        headers, payload = bytes(connection.response).split(b"\r\n\r\n", 1)
        status_line, fields = headers.split(b"\r\n", 1)
        return int(status_line.split()[1]), BytesParser().parsebytes(fields), payload

    def tree_digest(self):
        return {
            str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.root.rglob("*") if path.is_file()
        }

    def test_snapshot_counts_without_source_text(self):
        snapshot = self.portal.snapshot()
        self.assertEqual(snapshot["run_id"], "neutral-run")
        self.assertEqual(snapshot["target_models"], 4)
        self.assertEqual(snapshot["total_units"], 7)
        self.assertFalse(snapshot["stale"])
        self.assertTrue(snapshot["report"]["available"])
        first = snapshot["models"][0]
        self.assertEqual(first["active_stage"], "asr_polish")
        stage = first["stages"][0]
        self.assertEqual({key: stage[key] for key in ("success", "failed", "skipped", "pending")},
                         {"success": 3, "failed": 1, "skipped": 1, "pending": 2})
        self.assertEqual(stage["operations"], {"copy": 1, "replace": 1, "host_preserve": 1})
        self.assertEqual(first["stages"][1]["pending"], 7)
        self.assertEqual(snapshot["models"][1]["state"], "waiting_model")
        self.assertEqual(len(first["stages"]), 5)
        encoded = json.dumps(snapshot)
        for marker in ("SOURCE_SENTINEL", "OUTPUT_SENTINEL", "RETAINED_SENTINEL",
                       "FAILED_SENTINEL", "SKIPPED_SENTINEL", "INPUT_SENTINEL", "unit-1"):
            self.assertNotIn(marker, encoded)

    def test_dataset_count_fallback(self):
        del self.state["target_units"]
        self.write_json("STATUS.json", self.state)
        self.assertEqual(self.portal.snapshot()["total_units"], 7)

    def test_completed_record_is_not_marked_stale(self):
        self.state["updated_at"] = "2001-01-01T00:00:00+00:00"
        for state in ("complete", "complete_with_failures"):
            with self.subTest(state=state):
                self.state["state"] = state
                self.write_json("STATUS.json", self.state)
                self.assertFalse(self.portal.snapshot()["stale"])
        self.state["state"] = "running"
        self.write_json("STATUS.json", self.state)
        self.assertTrue(self.portal.snapshot()["stale"])

    def test_get_head_and_no_store(self):
        for route in ("/", "/api/status", "/report/", "/report/结论.txt", "/scope"):
            with self.subTest(route=route):
                status, headers, body = self.request(route)
                self.assertEqual(status, 200)
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                self.assertEqual(int(headers["Content-Length"]), len(body))
                self.assertTrue(body)
                head_status, head_headers, head_body = self.request(route, "HEAD")
                self.assertEqual(head_status, 200)
                self.assertEqual(head_headers["Cache-Control"], "no-store")
                self.assertGreater(int(head_headers["Content-Length"]), 0)
                self.assertEqual(head_body, b"")

    def test_report_scope_links_and_preserved_evidence(self):
        status, headers, body = self.request("/report")
        self.assertEqual(status, 302)
        self.assertEqual(headers["Location"], "/report/")
        status, _, body = self.request("/report/")
        self.assertEqual(status, 200)
        self.assertIn('href="/scope#用途与处理范围"'.encode(), body)
        self.assertNotIn(b"file://", body)
        self.assertEqual((self.full / "00_开始复核.html").read_bytes(), self.html)
        status, _, body = self.request("/scope")
        self.assertEqual(status, 200)
        self.assertIn(b'&lt;escaped-marker&gt;', body)
        self.assertNotIn(b"This next section must not be exposed", body)
        self.assertIn(b'href="/"', body)

    def test_unknown_and_traversal_routes_are_not_file_browser(self):
        for route in ("/unknown", "/inputs/dataset.json", "/STATUS.json", "/report/STATUS.json",
                      "/report/../STATUS.json", "/report/%2e%2e/STATUS.json",
                      "/report/%252e%252e/STATUS.json", "/api/status/", "/../README.md"):
            with self.subTest(route=route):
                status, headers, body = self.request(route)
                self.assertEqual(status, 404)
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertNotIn(b"SENTINEL", body)

    def test_storage_loss_stops_api_and_reports_but_not_home(self):
        self.available = False
        for route in ("/api/status", "/report/", "/report/结论.txt"):
            with self.subTest(route=route):
                status, headers, body = self.request(route)
                self.assertEqual(status, 503)
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertNotIn(b"SENTINEL", body)
                self.assertEqual(self.request(route, "HEAD")[2], b"")
        self.assertEqual(self.request("/")[0], 200)
        self.available = True
        self.assertEqual(self.request("/api/status")[0], 200)

    def test_report_symlink_outside_run_is_unavailable(self):
        outside = self.root / "outside-report.html"
        outside.write_text("OUTSIDE_SENTINEL", encoding="utf-8")
        report = self.full / "00_开始复核.html"
        report.unlink()
        report.symlink_to(outside)
        for route in ("/report/", "/api/status"):
            with self.subTest(route=route):
                status, _, body = self.request(route)
                self.assertEqual(status, 503)
                self.assertNotIn(b"OUTSIDE_SENTINEL", body)

    def test_post_does_not_write(self):
        before = self.tree_digest()
        for route in ("/", "/api/status", "/report/结论.txt"):
            with self.subTest(route=route):
                status, _, _ = self.request(route, "POST", b"attempted synthetic change")
                self.assertEqual(status, 501)
        self.assertEqual(self.tree_digest(), before)

    def test_evaluation_ancestor_cannot_redirect_outside_mount(self):
        ancestor = self.mount / "AI-Movie"
        moved = self.root / "redirected-evaluations"
        ancestor.rename(moved)
        ancestor.symlink_to(moved, target_is_directory=True)
        for route in ("/api/status", "/report/"):
            self.assertEqual(self.request(route)[0], 503)

    def test_non_lan_peer_is_denied(self):
        for peer in ("192.0.2.10", "10.0.0.10"):
            with self.subTest(peer=peer):
                self.assertEqual(self.request("/api/status", peer=peer)[0], 403)
        self.assertEqual(self.request("/api/status", peer="192.168.50.10")[0], 200)


if __name__ == "__main__":
    unittest.main()
