#!/usr/bin/env python3
"""Resume the four-model private text experiment, one complete model chain at a time.

Scope: [README full scope](../../README.md#用途与处理范围). This evaluates
existing supplied text only; it neither generates media nor certifies advertising,
legality, audio correctness, or the production localization pipeline. It never
downloads models, modifies frozen production sources, or publishes a release.

Run --self-test for neutral in-memory tests. The root operator starts this command;
importing it does not start a runtime or a benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXPECTED_SOURCE_HASH = "aeaebc6c5bc96c6f8f37e17fcdc4cbe3d6e3ec59a81b3eafca34154985464f92"
BASE_URL = "http://127.0.0.1:11435"
VERSION = "0.40.0"
MODELS = {
    "qwen3.6-35b-a3b": "qwen3.6:35b-a3b-q4_K_M",
    "glm-4.7-flash": "glm-4.7-flash:q4_K_M",
    "qwen3.8-27b": "qwen3.8:27b-q4_K_M",
    "gemma4-31b-it": "gemma4:31b-it-q4_K_M",
}
STAGES = ("asr_polish", "translate_raw", "translate_polished",
          "polish_translation_raw", "polish_translation_polished")
TERMINAL = {"success", "failed", "skipped"}
CONFIG_FAILURES = {"configuration", "context_overflow", "model_identity"}
POLL_SECONDS = 30
SCOPE = "Existing supplied text experiment only; README.md#用途与处理范围; not advertising acceptance or a release"
STOP = False


class RuntimeNotReady(Exception):
    """Transient neutral-preparation transport failure; existing evidence is retained."""


def valid_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_runtime_config(config: dict, source: dict) -> None:
    if config.get("id") != source.get("id") or config.get("source_model") != source.get("model") \
            or config.get("source_digest") != source.get("digest") or not valid_digest(config.get("digest")) \
            or not valid_digest(config.get("source_digest")):
        raise ValueError("runtime_source_identity_mismatch")
    direct = config.get("model") == source["model"] and config["digest"] == source["digest"]
    alias = config.get("model") == "llamacpp:" + config["digest"]
    if not (direct or alias) or config.get("think") is not False or config.get("base_url") != BASE_URL:
        raise ValueError("runtime_identity_not_an_explicit_source_or_digest_alias")


def select_runtime_config(source: dict, tags: dict, show: dict) -> tuple[dict, dict]:
    """Bind only the source tag's selected manifest to a unique exact runtime alias."""
    rows = tags.get("models")
    if not isinstance(rows, list):
        raise ValueError("runtime_tags_not_an_array")
    source_rows = [r for r in rows if r.get("name") == source["model"]]
    manifests = show.get("manifests") or []
    if not isinstance(manifests, list):
        raise ValueError("runtime_manifests_not_an_array")
    selected = [m for m in manifests if m.get("selected") is True]
    if len(selected) > 1:
        raise ValueError("multiple_selected_source_manifests")
    binding = {"method": "unchanged_unique_source_manifest", "selected_manifest": selected[0] if selected else None}
    if selected and selected[0].get("runner") == "llamacpp":
        digest = selected[0].get("digest", "").removeprefix("sha256:")
        if not valid_digest(digest) or not any(r.get("digest") == digest for r in source_rows):
            raise ValueError("selected_manifest_not_linked_to_source_tag")
        model = "llamacpp:" + digest
        aliases = [r for r in rows if r.get("name") == model]
        if len(aliases) != 1 or aliases[0].get("digest") != digest:
            raise ValueError("selected_runtime_alias_missing_or_ambiguous")
        binding["method"] = "source_show_selected_llamacpp_manifest_unique_digest_alias"
    else:
        if any(m.get("runner") == "llamacpp" for m in manifests) or (selected and selected[0].get("runner") not in (None, "ggml")):
            raise ValueError("unresolved_or_unknown_runtime_variant")
        if len(source_rows) != 1 or source_rows[0].get("digest") != source["digest"]:
            raise ValueError("source_manifest_changed_without_proven_runtime_alias")
        model, digest = source["model"], source["digest"]
        if selected and selected[0].get("digest", "").removeprefix("sha256:") != digest:
            raise ValueError("selected_native_manifest_disagrees_with_source_identity")
    config = {"id": source["id"], "model": model, "digest": digest,
              "source_model": source["model"], "source_digest": source["digest"],
              "think": False, "base_url": BASE_URL}
    validate_runtime_config(config, source)
    return config, binding


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def options(limit=None) -> dict:
    return {"batch_size": 64, "num_ctx": 131072, "num_predict": 8192,
            "limit": limit, "max_retries": 2, "bisect": True, "timeout": 1800.0,
            "draft_num_predict": 0, "base_url": BASE_URL}


def inventory_configs(inventory: dict) -> tuple[dict, dict]:
    """Only exact preselected ready tags with installed SHA256 identities qualify."""
    if inventory.get("ollama", {}).get("version") != VERSION:
        raise ValueError("inventory_runtime_version_mismatch")
    found, issues = {}, {}
    rows = inventory.get("models")
    if not isinstance(rows, list):
        raise ValueError("inventory_models_not_array")
    for mid, tag in MODELS.items():
        matches = [r for r in rows if isinstance(r, dict) and r.get("model") == tag]
        if len(matches) > 1:
            issues[mid] = "duplicate_inventory_tag"
            continue
        if not matches or matches[0].get("status") != "ready":
            continue
        row = matches[0]
        installed = row.get("installed") or {}
        digest = installed.get("digest")
        if installed.get("name") != tag or not isinstance(digest, str) or len(digest) != 64 \
                or any(c not in "0123456789abcdef" for c in digest):
            issues[mid] = "invalid_ready_model_identity"
            continue
        # Never infer think compatibility or a different quantization/model tag.
        found[mid] = {"id": mid, "model": tag, "digest": digest, "think": False,
                      "base_url": BASE_URL}
    return found, issues


def chain_snapshot(folder: Path, target_ids: list[str], fingerprint: str) -> dict:
    """Reconstruct terminal state from persisted stage artifacts, never their names."""
    result = {"state": "missing", "stages": {}, "configuration_failure": None,
              "successful_units": 0, "failed_units": 0, "skipped_units": 0}
    expected = set(target_ids)
    for stage in STAGES:
        path = folder / f"{stage}.json"
        if not path.exists():
            continue
        data = read_json(path)
        if data.get("fingerprint") != fingerprint:
            raise ValueError("stage_fingerprint_mismatch")
        rows = data.get("units")
        if not isinstance(rows, list):
            raise ValueError("stage_units_not_array")
        ids = [r.get("id") for r in rows]
        if any(not isinstance(uid, str) for uid in ids) or len(ids) != len(set(ids)) or not set(ids) <= expected:
            raise ValueError("stage_unit_identity_mismatch")
        counts = Counter(r.get("status", "unavailable") for r in rows)
        counts["pending"] += len(target_ids) - len(rows)
        terminal = data.get("state") == "complete" and set(ids) == expected \
            and all(r.get("status") in TERMINAL for r in rows)
        result["stages"][stage] = {"state": data.get("state"), "counts": dict(counts),
                                   "terminal": terminal}
        for r in rows:
            if r.get("category") in CONFIG_FAILURES:
                result["configuration_failure"] = r["category"]
        for key in ("success", "failed", "skipped"):
            result[{"success": "successful_units", "failed": "failed_units", "skipped": "skipped_units"}[key]] += counts[key]
    if result["stages"]:
        result["state"] = "partial"
    if len(result["stages"]) == len(STAGES) and all(v["terminal"] for v in result["stages"].values()):
        result["state"] = "complete" if result["successful_units"] == len(target_ids) * len(STAGES) else "complete_with_failures"
    result["zero_success_stages"] = [s for s, v in result["stages"].items()
                                     if v["terminal"] and v["counts"].get("success", 0) == 0]
    return result


def smoke_decision(snapshot: dict, returncode: int) -> str:
    if snapshot.get("configuration_failure"):
        return "pending_human"
    if returncode not in (0, 2) or snapshot["state"] not in ("complete", "complete_with_failures"):
        return "pending_human"
    if snapshot["zero_success_stages"]:
        return "pending_human"
    return "passed" if snapshot["state"] == "complete" else "partial"


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.out = args.root_out.resolve()
        self.dataset_path = self.out / "inputs" / "dataset.json"
        self.references_path = self.out / "inputs" / "references.json"
        self.runner = load_module("benchmark_supervisor_runner", HERE / "runner.py")
        self.eval = load_module("benchmark_supervisor_data_eval", HERE / "data_eval.py")
        self.evidence = load_module("benchmark_supervisor_run_evidence", ROOT / "scripts" / "run_evidence.py")
        self.publisher = load_module("benchmark_supervisor_smb", ROOT / "scripts" / "publish_smb.py")
        self.source = None
        self.units = []
        self.configs = {}
        self.child = None
        self.waited_seconds = 0.0
        self.status = {"schema_version": 1, "scope": SCOPE, "state": "initializing",
                       "delivery_confirmed": False, "models": {mid: {"tag": tag, "state": "waiting_model"}
                                                                  for mid, tag in MODELS.items()}}

    def write_json(self, path: Path, value) -> None:
        self.eval.write_json(path, value)

    def safety_check(self) -> None:
        self.eval.require_nas(self.out)
        self.publisher.require_mount()
        proc = subprocess.run(["systemctl", "--user", "show", "ai-movie-v34-regression.service",
                               "--property=ActiveState", "--property=SubState", "--property=LoadState"],
                              capture_output=True, text=True, check=True, timeout=10)
        service = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
        if service.get("LoadState") != "loaded" or service.get("ActiveState") not in ("inactive", "failed"):
            raise ValueError("v34_regression_service_not_confirmed_stopped")
        if self.evidence.source_hash() != EXPECTED_SOURCE_HASH:
            raise ValueError("frozen_production_source_hash_changed")
        if hasattr(self, "identity"):
            if any(digest_file(HERE / name) != sha for name, sha in self.identity["code_sha256"].items()):
                raise ValueError("benchmark_code_changed_during_run")
            if digest_file(self.dataset_path) != self.identity["dataset_sha256"] \
                    or digest_file(self.references_path) != self.identity["references_sha256"]:
                raise ValueError("benchmark_input_changed_during_run")
            if any(read_json(self.out / "config" / f"{mid}.json") != [config]
                   for mid, config in self.configs.items()):
                raise ValueError("model_configuration_changed_during_run")

    def save_status(self, state: str | None = None, **extra) -> None:
        if state:
            self.status["state"] = state
        self.status.update(extra, updated_at=now(), pid=os.getpid(), waiting_seconds_this_process=round(self.waited_seconds, 1))
        models = self.status["models"]
        self.status["completed_models"] = sum(m.get("state") in ("complete", "complete_with_failures") for m in models.values())
        self.status["target_models"] = len(MODELS)
        self.write_json(self.out / "STATUS.json", self.status)
        lines = ["既有字幕文本四模型实验；不是广告验收或正式发布。",
                 "用途范围：[README](../../README.md#用途与处理范围)",
                 f"更新（北京时间）：{datetime.fromisoformat(self.status['updated_at']).astimezone(ZoneInfo('Asia/Shanghai')).isoformat()}", f"状态：{self.status['state']}",
                 f"完整跑完模型：{self.status['completed_models']}/{len(MODELS)}",
                 f"服务器交付确认：{self.status.get('delivery_confirmed', False)}"]
        for mid, model in models.items():
            lines.append(f"{mid}：{model.get('state')}；阶段={model.get('stage', '—')}；原因={model.get('reason', '—')}")
        if self.status.get("reason"):
            lines.append(f"说明：{self.status['reason']}")
        path = self.out / "状态.txt"
        self.eval.require_nas(path)
        temp = path.with_suffix(".txt.tmp")
        temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.eval.require_nas(path)
        temp.replace(path)

    def initialize(self) -> None:
        self.safety_check()
        self.source, self.units = self.runner.load_dataset(self.dataset_path)
        read_json(self.references_path)
        previous = self.out / "STATUS.json"
        if previous.exists():
            old = read_json(previous)
            for mid in MODELS:
                if old.get("models", {}).get(mid, {}).get("state") == "pending_human":
                    self.status["models"][mid] = old["models"][mid]
        identity = {"schema_version": 1, "scope": SCOPE, "source_hash": EXPECTED_SOURCE_HASH,
                    "dataset_sha256": digest_file(self.dataset_path),
                    "references_sha256": digest_file(self.references_path),
                    "models": MODELS, "options": options(), "smoke_limit": 4,
                    "python": str(Path(self.args.python).absolute()), "ollama_version": VERSION,
                    "code_sha256": {name: digest_file(HERE / name) for name in
                                    ("runner.py", "data_eval.py", "analyze.py", "report.py", "orchestrate.py")}}
        config_path = self.out / "config" / "orchestrator.json"
        if config_path.exists() and read_json(config_path) != identity:
            raise ValueError("experiment_identity_changed_use_new_root_or_review_configuration")
        if not config_path.exists():
            self.write_json(config_path, identity)
        self.identity = identity
        expected_models = [{"id": mid, "model": tag} for mid, tag in MODELS.items()]
        for phase in ("smoke", "full-context-b64"):
            path = self.out / phase / "EXPECTED_MODELS.json"
            if path.exists() and read_json(path) != expected_models:
                raise ValueError("expected_models_changed")
            if not path.exists():
                self.write_json(path, expected_models)
        audit_path = self.out / "AUDIT.json"
        if not audit_path.exists():
            self.write_json(audit_path, self.audit())
        # Completed checkpoints can be recognized without a reachable inference server.
        for mid in MODELS:
            path = self.out / "config" / f"{mid}.json"
            if path.exists():
                entries = self.runner.load_models(path)
                if len(entries) != 1 or entries[0]["id"] != mid or entries[0].get("source_model") != MODELS[mid]:
                    raise ValueError("saved_model_configuration_identity_mismatch")
                validate_runtime_config(entries[0], {"id": mid, "model": MODELS[mid], "digest": entries[0].get("source_digest")})
                self.configs[mid] = entries[0]
                full = self.snapshot(mid, "full-context-b64")
                if full["state"] in ("complete", "complete_with_failures") and not full["configuration_failure"] \
                        and full["successful_units"] > 0:
                    smoke = self.snapshot(mid, "smoke")
                    decision = smoke_decision(smoke, 0 if smoke["state"] == "complete" else 2)
                    if decision == "pending_human":
                        self.status["models"][mid].update(state="pending_human", reason="full_checkpoint_without_usable_smoke")
                    else:
                        self.status["models"][mid].update(state=full["state"], smoke=smoke, smoke_decision=decision,
                                                          **{"full-context-b64": full}, resumed_from_terminal_checkpoints=True)
        self.save_status("waiting_models", source_hash=EXPECTED_SOURCE_HASH, inventory=str(self.args.inventory),
                         target_units=len(self.units), endpoint=BASE_URL, ollama_version=VERSION)

    def audit(self) -> dict:
        rows = self.units + self.source.get("excluded_units", [])
        by_chunk = {}
        for row in rows:
            by_chunk.setdefault(row["chunk"], []).append(row)
        word_sources, word_count, traced, exact = [], 0, 0, 0
        for chunk, selected in sorted(by_chunk.items()):
            path = ROOT / "workspace" / f"{self.source['film']}_p{chunk:02d}" / "asr_words.json"
            words = read_json(path)
            word_count += len(words)
            word_sources.append({"path": str(path.resolve()), "sha256": digest_file(path), "rows": len(words)})
            for row in selected:
                text = self.eval.normalize(row["text"], "ja_nfkc")
                nearby = "".join(word["w"] for word in words
                                 if row["source_start"] - 0.05 <= (word["s"] + word["e"]) / 2 <= row["source_end"] + 0.05)
                nearby = self.eval.normalize(nearby, "ja_nfkc")
                traced += bool(text and text in nearby)
                exact += text == nearby
        reference_stats = {}
        for name, ref in read_json(self.references_path)["sets"].items():
            groups = self.eval.build_groups(self.units, ref["units"])
            covered = sum(bool(g["reference_unit_ids"][r["id"]]) for g in groups for r in g["references"])
            reference_stats[name] = {"references": len(ref["units"]), "with_asr_overlap": covered,
                                    "without_asr_overlap": len(ref["units"]) - covered,
                                    "groups": len(groups), "max_asr_per_group": max((len(g["units"]) for g in groups), default=0),
                                    "max_references_per_group": max((len(g["references"]) for g in groups), default=0)}
        if traced != len(rows):
            raise ValueError("saved_asr_text_not_fully_traceable_to_timed_word_evidence")
        return {"schema_version": 1, "scope": SCOPE, "created_at": now(),
                "dataset_sha256": digest_file(self.dataset_path),
                "production_source_sha256": EXPECTED_SOURCE_HASH,
                "reference_backfill": "No screen/OCR/fansub reverse-translation backfill discovered in read-only code and artifact audit",
                "provenance": {"asr_text_rows": len(rows), "raw_word_rows": word_count,
                               "rows_found_in_timed_word_stream": traced, "exact_timed_window_matches": exact,
                               "word_sources": word_sources}, "reference_coverage": reference_stats,
                "code_evidence": ["scripts/run_pipeline.py:704", "ai_movie/asr.py:619",
                                  "ai_movie/asr.py:1035", "ai_movie/asr.py:1130", "ai_movie/config.py:236"],
                "historical_evidence": "Documentation/claude-session-summary-2026-10-04.md:235,247 describes subtitle guidance as unimplemented advice",
                "limitations": ["Current saved artifacts and code are consistent with audio ASR; historical external manual edits cannot be disproved",
                                "Models receive supplied text, not audio; ASR repair remains context-based inference",
                                "Claude subtitle-tile readings and OCR are not audio-certified ground truth",
                                "OCR reference uses Qwen-family model and may share model-family bias with candidates",
                                "Missing-reference overlap is retained in scores; normalized character similarity is not translation accuracy",
                                "keep_original labels are preserved as metadata and are not treated as proof of nonlexical audio"]}

    def runtime_ready(self) -> bool:
        try:
            with urllib.request.urlopen(BASE_URL + "/api/version", timeout=10) as response:
                version = json.load(response).get("version")
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return False
        if version != VERSION:
            raise ValueError("live_ollama_version_mismatch")
        return True

    def fixed_config(self, mid: str, config: dict) -> Path:
        path = self.out / "config" / f"{mid}.json"
        if path.exists() and read_json(path) != [config]:
            raise ValueError("ready_model_digest_or_configuration_changed")
        if not path.exists():
            self.write_json(path, [config])
        self.runner.load_models(path)
        self.configs[mid] = config
        return path

    def runtime_api_evidence(self, folder: Path, name: str, endpoint: str, payload=None, timeout=60) -> dict:
        """Save full neutral/model-metadata requests and responses while updating status."""
        path = folder / name
        self.write_json(path / "request.json", {"base_url": BASE_URL, "path": endpoint, "payload": payload,
                                               "requested_at": now(), "timeout": timeout})
        result = {}

        def request():
            try:
                result["response"] = self.runner.raw_http(BASE_URL, endpoint, payload, timeout)
            except Exception as exc:
                result["exception"] = exc

        worker = threading.Thread(target=request, daemon=True)
        worker.start()
        next_status = 0.0
        while worker.is_alive():
            if STOP:
                raise InterruptedError("supervisor_stop_requested")
            if time.monotonic() >= next_status:
                self.safety_check()
                self.save_status("preparing_runtime", runtime_preparation_step=name)
                next_status = time.monotonic() + POLL_SECONDS
            worker.join(timeout=1)
        if "exception" in result:
            exc = result["exception"]
            category = getattr(exc, "category", type(exc).__name__)
            self.write_json(path / "failure.json", {"category": category, "detail": str(exc), "at": now()})
            if category in ("transport", "timeout", "URLError", "TimeoutError", "ConnectionError"):
                raise RuntimeNotReady("runtime_neutral_preparation_transport") from exc
            raise ValueError("runtime_neutral_preparation_request_failed") from exc
        raw, status, headers = result["response"]
        response_path = path / "response.bin"
        self.eval.require_nas(response_path)
        response_path.write_bytes(raw)
        self.write_json(path / "http.json", {"status": status, "headers": headers, "received_at": now()})
        try:
            response = self.runner.decode_api(raw, status)
        except self.runner.Failure as exc:
            self.write_json(path / "failure.json", {"category": exc.category, "detail": exc.detail, "at": now()})
            if exc.category == "transport":
                raise RuntimeNotReady("runtime_neutral_preparation_transport") from exc
            raise ValueError("runtime_neutral_preparation_response_failed") from exc
        self.write_json(path / "response.json", response)
        return response

    def prepare_runtime_config(self, mid: str, source: dict) -> dict:
        """Prepare once, then preserve the immutable source/runtime identity binding."""
        existing = self.out / "config" / f"{mid}.json"
        if existing.exists():
            configs = self.runner.load_models(existing)
            if len(configs) != 1:
                raise ValueError("saved_runtime_config_is_not_single_model")
            validate_runtime_config(configs[0], source)
            return configs[0]  # No warmup or identity migration during resume.
        if not isinstance(getattr(self.runner, "SAMPLING", None), dict):
            raise ValueError("runner_shared_sampling_configuration_missing")
        stamp = now().replace(":", "-") + "-" + uuid.uuid4().hex[:8]
        folder = self.out / "config" / "runtime_identity" / mid / stamp
        self.status["models"][mid].update(state="preparing_runtime", phase="neutral_warmup")
        self.save_status("preparing_runtime", active_model=mid, active_phase="neutral_warmup")
        self.write_json(folder / "source_identity.json", {"scope": SCOPE, "source_model": source["model"],
                        "source_registry_digest": source["digest"], "inventory": str(self.args.inventory),
                        "inventory_sha256": digest_file(self.args.inventory), "prepared_at": now()})
        self.runtime_api_evidence(folder, "before-tags", "/api/tags")
        self.runtime_api_evidence(folder, "before-show", "/api/show", {"model": source["model"]})
        payload = {"model": source["model"], "prompt": "Return OK.", "think": False,
                   "stream": False, "keep_alive": "15m", "options": {
                       **self.runner.SAMPLING, "num_ctx": 131072, "num_predict": 8}}
        warm = self.runtime_api_evidence(folder, "neutral-warmup", "/api/generate", payload, timeout=1800)
        if warm.get("done") is not True:
            raise ValueError("neutral_warmup_not_finished")
        after_show = self.runtime_api_evidence(folder, "after-show", "/api/show", {"model": source["model"]})
        after_tags = self.runtime_api_evidence(folder, "after-tags", "/api/tags")
        config, binding = select_runtime_config(source, after_tags, after_show)
        self.write_json(folder / "binding.json", {"scope": SCOPE, "configuration": config,
                        "binding": binding, "created_at": now(), "neutral_warmup_is_not_benchmark_output": True})
        self.write_json(self.out / "config" / f"{mid}.runtime_identity.json", {
            "source_model": source["model"], "source_digest": source["digest"],
            "model": config["model"], "digest": config["digest"],
            "evidence_directory": str(folder.relative_to(self.out)), "binding": binding})
        return config

    def fingerprint(self, mid: str, phase: str) -> str:
        return self.runner.fingerprint(self.source, self.configs[mid], options(4 if phase == "smoke" else None))

    def snapshot(self, mid: str, phase: str) -> dict:
        target = self.units[:4] if phase == "smoke" else self.units
        folder, fingerprint = self.out / phase / mid, self.fingerprint(mid, phase)
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get("fingerprint") != fingerprint or manifest.get("target_units") != len(target):
                raise ValueError("checkpoint_manifest_identity_mismatch")
        elif any((folder / f"{s}.json").exists() for s in STAGES):
            raise ValueError("checkpoint_has_no_manifest")
        return chain_snapshot(folder, [u["id"] for u in target], fingerprint)

    def config_failure(self, folder: Path) -> str | None:
        preflight = folder / "preflight_failure.json"
        if preflight.exists():
            failure = read_json(preflight)
            if failure.get("category") in CONFIG_FAILURES:
                return failure["category"]
        log = folder / "attempts.jsonl"
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("event") == "finished" and entry.get("category") in CONFIG_FAILURES:
                    # A batch may exceed the conservative byte bound while a
                    # smaller target batch fits with the exact same full source.
                    # Let the runner's recorded bisection attempt that recovery.
                    if entry["category"] == "context_overflow" and len(entry.get("ids", [])) > 1:
                        continue
                    return entry["category"]
        return None

    def stop_child(self) -> None:
        if self.child and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait(timeout=15)

    def run_process(self, command: list[str], label: str, mid=None, phase=None,
                    update_status=True) -> int:
        self.safety_check()
        logs = self.out / "logs"
        self.eval.require_nas(logs)
        logs.mkdir(exist_ok=True)
        log_path = logs / (label + ".log")
        with log_path.open("a", encoding="utf-8") as log:
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
            next_update = 0.0
            try:
                while self.child.poll() is None:
                    if STOP:
                        self.stop_child()
                        raise InterruptedError("supervisor_stop_requested")
                    if time.monotonic() >= next_update:
                        self.safety_check()
                        if mid and phase:
                            folder = self.out / phase / mid
                            problem = self.config_failure(folder)
                            if problem:
                                self.stop_child()
                                self.status["models"][mid].update(state="pending_human", reason=problem)
                                self.save_status("running")
                                return 1
                            progress_path = folder / "progress.json"
                            if progress_path.exists():
                                progress = read_json(progress_path)
                                safe = {k: progress[k] for k in ("state", "stage", "updated_at", "counts",
                                        "completed_batches", "total_batches", "batch", "retry") if k in progress}
                                self.status["models"][mid].update(stage=progress.get("stage"), progress=safe,
                                                                  phase=phase)
                        if update_status:
                            self.save_status()
                        next_update = time.monotonic() + POLL_SECONDS
                    try:
                        self.child.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                return self.child.returncode
            finally:
                self.stop_child()
                self.child = None

    def evaluate(self, phase: str) -> bool:
        common = ["--dataset", str(self.dataset_path), "--references", str(self.references_path),
                  "--run-dir", str(self.out / phase)]
        for name, subcommand in (("data_eval.py", ["score"]), ("analyze.py", []), ("report.py", [])):
            code = self.run_process([self.args.python, str(HERE / name), *subcommand, *common],
                                    f"{phase}-{Path(name).stem}")
            if code != 0:
                self.save_status("pending_human", reason=f"{phase}_{Path(name).stem}_exit_{code}")
                return False
        return True

    def run_phase(self, mid: str, phase: str, config_path: Path) -> tuple[int, dict]:
        snapshot = self.snapshot(mid, phase)
        if snapshot["state"] in ("complete", "complete_with_failures"):
            return (0 if snapshot["state"] == "complete" else 2), snapshot
        command = [self.args.python, str(HERE / "runner.py"), "--dataset", str(self.dataset_path),
                   "--models", str(config_path), "--out", str(self.out / phase),
                   "--base-url", BASE_URL, "--batch-size", "64", "--num-ctx", "131072",
                   "--num-predict", "8192", "--timeout", "1800", "--max-retries", "2", "--bisect"]
        if phase == "smoke":
            command += ["--limit", "4"]
        self.status["models"][mid].update(state="running", phase=phase)
        self.save_status("running", active_model=mid, active_phase=phase)
        code = self.run_process(command, f"{mid}-{phase}", mid, phase)
        snapshot = self.snapshot(mid, phase)
        self.status["models"][mid][phase] = {"exit_code": code, **snapshot}
        return code, snapshot

    def run_model(self, mid: str, config: dict) -> None:
        config = self.prepare_runtime_config(mid, config)
        config_path = self.fixed_config(mid, config)
        code, smoke = self.run_phase(mid, "smoke", config_path)
        decision = smoke_decision(smoke, code)
        self.status["models"][mid]["smoke_decision"] = decision
        if smoke["stages"] and not self.evaluate("smoke"):
            raise ValueError("smoke_scoring_or_report_failed")
        if decision == "pending_human":
            if self.preflight_transport_failure(mid, "smoke", code):
                self.status["models"][mid].update(state="waiting_service", reason="smoke_preflight_transport")
            else:
                self.status["models"][mid].update(state="pending_human", reason="smoke_not_usable_for_full_chain")
            self.save_status("waiting_models", active_model=None, active_phase=None)
            return
        code, full = self.run_phase(mid, "full-context-b64", config_path)
        if full["stages"] and not self.evaluate("full-context-b64"):
            raise ValueError("full_scoring_or_report_failed")
        if code in (0, 2) and full["state"] in ("complete", "complete_with_failures") \
                and not full["configuration_failure"] and full["successful_units"] > 0:
            self.status["models"][mid].update(state=full["state"], completed_at=now(), reason=None)
        else:
            if self.preflight_transport_failure(mid, "full-context-b64", code):
                self.status["models"][mid].update(state="waiting_service", reason="full_preflight_transport")
            else:
                self.status["models"][mid].update(state="pending_human", reason="full_chain_error_or_no_successful_output")
        self.save_status("waiting_models", active_model=None, active_phase=None)
        print(json.dumps({"event": "model_finished", "model": mid,
                          "state": self.status["models"][mid]["state"],
                          "completed_models": self.status["completed_models"], "target_models": len(MODELS)}), flush=True)

    def preflight_transport_failure(self, mid: str, phase: str, code: int) -> bool:
        folder = self.out / phase / mid
        preflight, progress = folder / "preflight_failure.json", folder / "progress.json"
        return code == 1 and preflight.exists() and progress.exists() \
            and read_json(progress).get("state") == "preflight" \
            and read_json(preflight).get("category") == "transport"

    def wait_once(self, reason: str) -> bool:
        self.save_status("waiting_models", reason=reason, active_model=None, active_phase=None)
        if self.waited_seconds >= self.args.max_wait_hours * 3600:
            self.save_status("waiting_models", reason="model_wait_deadline_reached_resume_is_available")
            return False
        start = time.monotonic()
        remaining = min(POLL_SECONDS, self.args.max_wait_hours * 3600 - self.waited_seconds)
        while not STOP and time.monotonic() - start < remaining:
            time.sleep(max(0, min(1, remaining - (time.monotonic() - start))))
        self.waited_seconds += time.monotonic() - start
        return True

    def finalize(self) -> int:
        self.save_status("finalizing", reason=None)
        for phase in ("smoke", "full-context-b64"):
            if not self.evaluate(phase):
                return 3
        checksums = []
        for name in ("inputs", "config", "smoke", "full-context-b64"):
            for path in sorted((self.out / name).rglob("*")):
                if path.is_file() and not path.name.endswith((".tmp", ".lock")):
                    self.eval.require_nas(path)
                    checksums.append({"path": str(path.relative_to(self.out)), "bytes": path.stat().st_size,
                                      "sha256": digest_file(path)})
        checksums.append({"path": "AUDIT.json", "bytes": (self.out / "AUDIT.json").stat().st_size,
                          "sha256": digest_file(self.out / "AUDIT.json")})
        self.write_json(self.out / "CHECKSUMS.json", {"schema_version": 1, "scope": SCOPE,
                        "created_at": now(), "files": checksums,
                        "excluded_mutable_files": ["STATUS.json", "状态.txt", "COMPLETE.json", "logs/*"]})
        state = "full" if all(m["state"] == "complete" for m in self.status["models"].values()) else "partial"
        complete = {"schema_version": 1, "scope": SCOPE, "status": state, "completed_at": now(),
                    "models": self.status["models"], "completed_models": len(MODELS),
                    "delivery_confirmed": False, "checksum_manifest": "CHECKSUMS.json",
                    "checksum_manifest_sha256": digest_file(self.out / "CHECKSUMS.json"),
                    "source_hash": EXPECTED_SOURCE_HASH, "not_a_release": True}
        self.write_json(self.out / "COMPLETE.json", complete)
        wait_command = [self.args.python, "-c",
                       "import sys;sys.path.insert(0,sys.argv[1]);from scripts.publish_smb import wait_for_uploads;wait_for_uploads(1800)", str(ROOT)]
        self.save_status("waiting_delivery", reason=None)
        if self.run_process(wait_command, "wait-for-smb-artifacts") != 0:
            self.save_status("waiting_delivery", reason="SMB_upload_confirmation_failed", delivery_confirmed=False)
            return 4
        complete.update(delivery_confirmed=True, delivery_confirmed_at=now())
        self.write_json(self.out / "COMPLETE.json", complete)
        self.save_status("complete" if state == "full" else "complete_with_failures", delivery_confirmed=True,
                         reason=None, active_model=None, active_phase=None)
        # Drain the completion receipt itself. Do not continually enqueue status writes here.
        if self.run_process(wait_command, "wait-for-smb-receipt", update_status=False) != 0:
            complete.update(delivery_confirmed=False, delivery_error="receipt_upload_confirmation_failed")
            self.write_json(self.out / "COMPLETE.json", complete)
            self.save_status("waiting_delivery", reason="SMB_receipt_upload_confirmation_failed", delivery_confirmed=False)
            return 4
        print(json.dumps({"event": "delivery_confirmed", "status": state, "completed_models": len(MODELS)}), flush=True)
        return 0 if state == "full" else 2

    def run(self) -> int:
        self.eval.require_nas(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / ".orchestrator.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.initialize()
            while not STOP:
                self.safety_check()
                if all(m["state"] in ("complete", "complete_with_failures") for m in self.status["models"].values()):
                    return self.finalize()
                candidates = [m for m in MODELS if self.status["models"][m]["state"] not in
                              ("complete", "complete_with_failures", "pending_human")]
                if not candidates:
                    self.save_status("pending_human", reason="one_or_more_models_have_no_completed_full_chain")
                    return 3
                try:
                    inventory = read_json(self.args.inventory)
                    configs, issues = inventory_configs(inventory)
                except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
                    if not self.wait_once("inventory_not_ready_or_atomic_update_in_progress"):
                        return 5
                    continue
                for mid, reason in issues.items():
                    self.status["models"][mid].update(state="pending_human", reason=reason)
                ready = [m for m in candidates if m in configs and m not in issues]
                if not ready:
                    if not self.wait_once("waiting_for_verified_model_downloads"):
                        return 5
                    continue
                if not self.runtime_ready():
                    if not self.wait_once("local_ollama_service_temporarily_unreachable"):
                        return 5
                    continue
                mid = ready[0]
                try:
                    self.run_model(mid, configs[mid])
                except RuntimeNotReady as exc:
                    self.status["models"][mid].update(state="waiting_service", reason=str(exc))
                    self.save_status("waiting_models")
                except (ValueError, json.JSONDecodeError) as exc:
                    self.status["models"][mid].update(state="pending_human", reason=str(exc))
                    self.save_status("waiting_models")
                if self.status["models"][mid]["state"] == "waiting_service":
                    if not self.wait_once("model_preflight_transport_wait"):
                        return 5
            raise InterruptedError("supervisor_stop_requested")


def self_test() -> int:
    """Neutral mocks for identity, failed-smoke handling, and checkpoint resumption."""
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def test_only_exact_ready_digest_qualifies(self):
            tag = next(iter(MODELS.values()))
            inv = {"ollama": {"version": VERSION}, "models": [{"model": tag, "status": "ready",
                   "installed": {"name": tag, "digest": "a" * 64}}]}
            good, bad = inventory_configs(inv)
            self.assertEqual(len(good), 1)
            self.assertEqual(bad, {})
            inv["models"][0]["status"] = "downloading"
            self.assertEqual(inventory_configs(inv)[0], {})
            inv["models"][0].update(status="ready", installed={"name": tag, "digest": "bad"})
            self.assertTrue(inventory_configs(inv)[1])

        def test_source_and_converted_runtime_identities_stay_separate(self):
            source = {"id": "neutral", "model": "neutral:source", "digest": "a" * 64}
            native, _ = select_runtime_config(source, {"models": [{"name": source["model"], "digest": source["digest"]}]}, {})
            self.assertEqual(native["model"], source["model"])
            alias = "llamacpp:" + "b" * 64
            tags = {"models": [{"name": source["model"], "digest": "c" * 64},
                               {"name": source["model"], "digest": "b" * 64},
                               {"name": alias, "digest": "b" * 64}]}
            show = {"manifests": [{"digest": "sha256:" + "c" * 64, "runner": "ggml"},
                                  {"digest": "sha256:" + "b" * 64, "runner": "llamacpp", "selected": True}]}
            config, binding = select_runtime_config(source, tags, show)
            self.assertEqual(config["source_digest"], "a" * 64)
            self.assertEqual(config["digest"], "b" * 64)
            self.assertEqual(config["model"], alias)
            self.assertIn("selected_llamacpp", binding["method"])
            tags["models"].append({"name": alias, "digest": "b" * 64})
            with self.assertRaises(ValueError):
                select_runtime_config(source, tags, show)
            with self.assertRaises(ValueError):
                validate_runtime_config({**config, "source_digest": "d" * 64}, source)

        def test_runtime_resume_reuses_source_binding_without_warmup(self):
            from types import SimpleNamespace
            source = {"id": "neutral", "model": "neutral:source", "digest": "a" * 64}
            config = {**source, "model": "llamacpp:" + "b" * 64, "digest": "b" * 64,
                      "source_model": source["model"], "source_digest": source["digest"],
                      "think": False, "base_url": BASE_URL}
            with tempfile.TemporaryDirectory() as directory:
                supervisor = object.__new__(Supervisor)
                supervisor.out = Path(directory)
                (supervisor.out / "config").mkdir()
                (supervisor.out / "config" / "neutral.json").write_text(json.dumps([config]))
                supervisor.runner = SimpleNamespace(load_models=read_json)
                supervisor.runtime_api_evidence = lambda *a, **k: self.fail("resume must not warm up")
                self.assertEqual(supervisor.prepare_runtime_config("neutral", source), config)
                with self.assertRaises(ValueError):
                    supervisor.prepare_runtime_config("neutral", {**source, "digest": "c" * 64})

        def test_unknown_variant_or_unlinked_alias_fails_closed(self):
            source = {"id": "neutral", "model": "neutral:source", "digest": "a" * 64}
            show = {"manifests": [{"digest": "sha256:" + "b" * 64, "runner": "llamacpp", "selected": True}]}
            with self.assertRaises(ValueError):
                select_runtime_config(source, {"models": [{"name": "llamacpp:" + "b" * 64, "digest": "b" * 64}]}, show)
            with self.assertRaises(ValueError):
                select_runtime_config(source, {"models": [{"name": source["model"], "digest": "c" * 64}]}, {})

        def test_missing_and_resumable_stages_are_not_complete(self):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                self.assertEqual(chain_snapshot(folder, ["neutral1"], "fp")["state"], "missing")
                (folder / "asr_polish.json").write_text(json.dumps({"fingerprint": "fp", "state": "running",
                    "units": [{"id": "neutral1", "status": "success", "text": "天气晴朗"}]}))
                self.assertEqual(chain_snapshot(folder, ["neutral1"], "fp")["state"], "partial")
                with self.assertRaises(ValueError):
                    chain_snapshot(folder, ["neutral1"], "wrong")

        def test_failed_smoke_never_promoted(self):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                for stage in STAGES:
                    (folder / f"{stage}.json").write_text(json.dumps({"fingerprint": "fp", "state": "complete",
                        "units": [{"id": "neutral1", "status": "failed", "text": None}]}))
                snap = chain_snapshot(folder, ["neutral1"], "fp")
                self.assertEqual(smoke_decision(snap, 2), "pending_human")
                self.assertEqual(snap["failed_units"], 5)
                for stage in STAGES:
                    (folder / f"{stage}.json").write_text(json.dumps({"fingerprint": "fp", "state": "complete",
                        "units": [{"id": "neutral1", "status": "success", "text": "晴天"}]}))
                snap = chain_snapshot(folder, ["neutral1"], "fp")
                self.assertEqual(smoke_decision(snap, 0), "passed")
                self.assertEqual(smoke_decision(snap, 1), "pending_human")

        def test_partial_success_is_explicit_and_configuration_blocks(self):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                for stage in STAGES:
                    (folder / f"{stage}.json").write_text(json.dumps({"fingerprint": "fp", "state": "complete",
                        "units": [{"id": "neutral1", "status": "success", "text": "晴天"},
                                  {"id": "neutral2", "status": "failed", "text": None}]}))
                snap = chain_snapshot(folder, ["neutral1", "neutral2"], "fp")
                self.assertEqual(snap["state"], "complete_with_failures")
                self.assertEqual(smoke_decision(snap, 2), "partial")
                snap["configuration_failure"] = "context_overflow"
                self.assertEqual(smoke_decision(snap, 2), "pending_human")

        def test_wait_deadline_leaves_resumable_waiting_state(self):
            from types import SimpleNamespace
            supervisor = object.__new__(Supervisor)
            supervisor.args = SimpleNamespace(max_wait_hours=1)
            supervisor.waited_seconds = 3600
            saved = []
            supervisor.save_status = lambda state, **fields: saved.append((state, fields))
            self.assertFalse(supervisor.wait_once("neutral_download_pending"))
            self.assertEqual(saved[-1][0], "waiting_models")
            self.assertEqual(saved[-1][1]["reason"], "model_wait_deadline_reached_resume_is_available")

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-out", type=Path)
    parser.add_argument("--inventory", type=Path, default=Path("/home/ai-movie/.local/state/ai-movie-text-benchmark/inventory_v040.json"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-wait-hours", type=float, default=12.0,
                        help="maximum cumulative model/service waiting time in this process, excluding inference")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.root_out or not 0 < args.max_wait_hours <= 168:
        parser.error("--root-out and a positive --max-wait-hours up to 168 are required")
    os.umask(0o077)

    def stop(_signum, _frame):
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    supervisor = Supervisor(args)
    try:
        return supervisor.run()
    except BlockingIOError:
        print(json.dumps({"event": "already_running", "state": "not_started"}), flush=True)
        return 6
    except InterruptedError:
        supervisor.stop_child()
        supervisor.save_status("interrupted", reason="checkpoint_resume_available", delivery_confirmed=False)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        supervisor.stop_child()
        # Do not include exception contents which might quote malformed source data.
        reason = str(exc) if type(exc) is ValueError and str(exc).replace("_", "").isalnum() else type(exc).__name__
        try:
            supervisor.save_status("pending_human", reason=reason, delivery_confirmed=False)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            pass  # Mount outage: journal only, never write into the unmounted local directory.
        print(json.dumps({"event": "supervisor_stopped", "state": "pending_human", "reason": reason}), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
