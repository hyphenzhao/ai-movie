#!/usr/bin/env python3
"""Compare existing subtitle benchmark evidence without inference or media.

Scope: [README full scope](../../README.md#用途与处理范围). This is a
user-selected existing-text transformation stress test, not advertising or
multilingual acceptance. Reference agreement is not audio-verified accuracy,
fluency, semantic faithfulness, or legal permission. Failed outputs remain in
the fixed scoring population; incomplete experiments are never final rankings.
"""

from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("benchmark_analysis_runner", HERE / "runner.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
STAGES = runner.STAGES
ZH_STAGES = STAGES[1:]
LABELS = {
    "asr_polish": "日文文本纠错", "translate_raw": "原文直译",
    "translate_polished": "日文校正后翻译", "polish_translation_raw": "原文直译后润色",
    "polish_translation_polished": "日文校正后翻译再润色",
}
REF_LABELS = {"screen_human_zh": "历史 Claude 读取画面中文参考", "fansub_zh": "下载中文字幕参考"}
TERMINAL = {"success", "failed", "skipped"}
OUTPUT_FAILURES = {"json_decode", "schema", "id", "empty", "truncation", "encoding"}
SCOPE = "用户指定既有字幕文本转换压力测试；不生成新情节或媒体；生产范围见 README.md#用途与处理范围。"
COPY_PROTOCOL = "copy/replace 是后端兼容协议：copy 由模型显式选择，主机逐字恢复本阶段原文或保留既有译稿，不是失败后的静默回退，也不表示模型纠错成功；replace 仅表示替换，不自动证明质量改善。评分仍使用主机还原后的完整 text，保留全部源单位及相同参考，不因历史 nonlexical 标签从评分人口剔除。"
HOST_PRESERVE_PROTOCOL = "host_preserve 是主机按保守 rule_v1 执行的独立路由：同时满足既有三项标记及限定字符集合的长纯发声规则，才逐字保留 dataset 原文且不发送模型请求；含词、混合或其他字符不旁路。历史标签不是听音认证。host_preserve 不算 LLM 纠错、翻译或润色成果。status=success 仅说明流程具有完整文本输出，不能当作模型请求成功；LLM 请求成功与失败另由实际请求日志统计。全人口指标仍包含主机保留原文；若参考字幕把发声写成另一语言的字形，其字面差异不等于 LLM 错译。本次不改变评分人口、参考或评分方法。"


def operation_counts(rows: list[dict], stage: str) -> dict:
    """Separate valid protocol outputs from evidence of actual editing or improvement."""
    counts = collections.Counter(copy=0, replace=0, host_preserve=0, copy_source=0, copy_draft=0,
                                 unspecified=0, unknown=0, successful_units=0, model_operation_units=0)
    for row in rows:
        if row.get("status") != "success" or not isinstance(row.get("text"), str) or not row["text"].strip():
            continue
        counts["successful_units"] += 1
        operation = row.get("operation")
        if operation == "host_preserve":
            counts["host_preserve"] += 1
        elif operation == "copy":
            counts["copy"] += 1
            counts["model_operation_units"] += 1
            provenance = row.get("restore_provenance") or {}
            source = provenance.get("source")
            is_draft = source == "target.draft" or (source is None and stage.startswith("polish_translation_"))
            counts["copy_draft" if is_draft else "copy_source"] += 1
        elif operation == "replace":
            counts["replace"] += 1
            counts["model_operation_units"] += 1
        else:
            counts["unspecified" if operation is None else "unknown"] += 1
    return dict(counts)


def backend_repeat_abort(row: dict) -> bool:
    return "prediction aborted, token repeat limit reached" in str(row.get("detail") or "").casefold()


def read(path: Path, default=None):
    return runner.strict_json(path.read_text(encoding="utf-8")) if path.exists() else default


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def verify_score_snapshot(metrics: dict, details: dict, run_dir: Path,
                          dataset_path: Path | None, references_path: Path | None,
                          dataset: dict | None = None, references: dict | None = None) -> dict:
    """Validate the score commit, complete arm set, and both raw input files."""
    problems = []
    snapshot = metrics.get("snapshot")
    required = {"id", "dataset_sha256", "references_sha256", "arms", "method_sha256"}
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        return {"valid": False, "snapshot_id": None, "problems": ["missing_or_invalid_snapshot"]}
    if snapshot != details.get("snapshot"):
        problems.append("metrics_details_snapshot_mismatch")
    if snapshot["id"] != runner.digest({k: v for k, v in snapshot.items() if k != "id"}):
        problems.append("snapshot_identity_hash_mismatch")
    if snapshot["method_sha256"] != runner.digest(metrics.get("method", {})):
        problems.append("scoring_method_hash_mismatch")
    if not metrics.get("details_sha256") or metrics["details_sha256"] != sha(run_dir / "score_details.json"):
        problems.append("score_details_file_hash_mismatch")
    # Also bind the parsed details object, so an in-memory stale read cannot
    # inherit a newer file's valid hash during a concurrent publication.
    try:
        if runner.digest(details) != runner.digest(read(run_dir / "score_details.json", {})):
            problems.append("parsed_details_changed_during_read")
        if runner.digest(metrics) != runner.digest(read(run_dir / "metrics.json", {})):
            problems.append("parsed_metrics_changed_during_read")
    except (OSError, ValueError, runner.Failure):
        problems.append("score_files_unreadable_during_validation")
    for name, path, parsed in (("dataset", dataset_path, dataset), ("references", references_path, references)):
        if path is None or snapshot[name + "_sha256"] != sha(path):
            problems.append(name + "_file_hash_mismatch_or_unverified")
        elif parsed is not None and runner.digest(parsed) != runner.digest(read(path)):
            problems.append(name + "_parsed_content_mismatch")
    spec = importlib.util.spec_from_file_location("benchmark_snapshot_scorer", HERE / "data_eval.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    current = scorer.current_arm_records(run_dir)
    if current != snapshot["arms"]:
        problems.append("current_arm_set_or_hash_mismatch")
    recorded = sorted([{"model": arm.get("model"), "stage": arm.get("stage"),
                        "sha256": arm["artifact"].get("sha256")}
                       for arm in metrics.get("arms", []) if isinstance(arm.get("artifact"), dict)],
                      key=lambda row: (str(row["model"]), str(row["stage"])))
    if recorded != snapshot["arms"]:
        problems.append("metrics_arm_set_not_bound_to_snapshot")
    return {"valid": not problems, "snapshot_id": snapshot["id"], "problems": problems,
            "details_sha256": metrics.get("details_sha256")}


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def fmt(value, precision=4):
    return f"{value:.{precision}f}" if number(value) else "未评分"


def expected_models(run_dir: Path, warnings: list) -> list[dict]:
    expected = read(run_dir / "EXPECTED_MODELS.json")
    if not isinstance(expected, list) or not expected:
        warnings.append("缺少有效 EXPECTED_MODELS.json，无法确认预定四模型范围；不出最终排名。")
        return []
    seen = set()
    for model in expected:
        mid = model.get("id") if isinstance(model, dict) else None
        if not isinstance(mid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", mid) or mid in seen:
            raise ValueError("EXPECTED_MODELS.json contains invalid/duplicate model IDs")
        seen.add(mid)
    if len(expected) != 4:
        warnings.append(f"预定集合有 {len(expected)} 个模型，未满足本实验四模型范围；不出四模型最终排名。")
    return expected


def inspect_stage(path: Path, expected_units: dict, fingerprint: str | None) -> dict:
    artifact = read(path, {})
    rows = artifact.get("units", [])
    if not isinstance(rows, list):
        raise ValueError(f"invalid units array: {path}")
    counts = collections.Counter()
    seen, problems = set(), []
    for unit in rows:
        try:
            key = runner.id_key(unit.get("id"))
        except (ValueError, AttributeError):
            problems.append("invalid_id")
            counts["invalid"] += 1
            continue
        if key in seen:
            problems.append("duplicate_id")
        seen.add(key)
        if key not in expected_units:
            problems.append("unexpected_id")
        else:
            if any(unit.get(field) != expected_units[key].get(field) for field in ("start", "end")):
                problems.append("timestamp_mismatch")
        status = unit.get("status")
        if status == "success" and (not isinstance(unit.get("text"), str) or not unit["text"].strip()):
            status = "invalid"
            problems.append("empty_success")
        counts[status if status in TERMINAL | {"pending"} else "invalid"] += 1
    counts["missing"] = len(set(expected_units) - seen)
    if artifact and (not fingerprint or artifact.get("fingerprint") != fingerprint):
        problems.append("fingerprint_mismatch_or_missing")
    terminal = bool(artifact) and artifact.get("state") == "complete" and not problems \
        and seen == set(expected_units) and all(unit.get("status") in TERMINAL for unit in rows)
    total = len(expected_units)
    return {"state": artifact.get("state", "missing"), "terminal": terminal,
            "target_units": total, "counts": {key: counts[key] for key in ("success", "failed", "skipped", "pending", "missing", "invalid")},
            "success_fraction": counts["success"] / total if total else None,
            "all_success": terminal and counts["success"] == total,
            "operation_counts": operation_counts(rows, artifact.get("stage", path.stem)),
            "identity_problems": sorted(set(problems)), "artifact_sha256": sha(path)}


def request_stats(path: Path, warnings: list) -> dict:
    started, finished, first_line = {}, {}, {}
    if path.exists():
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = runner.strict_json(line)
            except runner.Failure:
                warnings.append(f"{path.parent.name} 请求日志第 {index} 行未完整读取，可能仍在写入。")
                continue
            if not isinstance(row, dict) or not isinstance(row.get("attempt_id"), str):
                warnings.append(f"{path.parent.name} 请求日志第 {index} 行缺少有效 attempt_id，未纳入统计。")
                continue
            first_line.setdefault(row["attempt_id"], index)
            if row.get("event") == "started":
                started[row["attempt_id"]] = row
            elif row.get("event") == "finished":
                finished[row["attempt_id"]] = row
    attempts = [{**started.get(aid, {}), **finished.get(aid, {}), "attempt_id": aid,
                 "closed": aid in finished, "log_first_line": first_line[aid]}
                for aid in set(started) | set(finished)]

    def sort_key(row):
        try:
            stamp = datetime.fromisoformat(row["started_at"])
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            return (stamp.timestamp(), row["log_first_line"])
        except (KeyError, TypeError, ValueError):
            return (float("inf"), row["log_first_line"])

    attempts.sort(key=sort_key)
    seen_batches, audit, corrections = set(), [], []
    for row in attempts:
        key = (row.get("stage"), row.get("batch"))
        depth, retry = row.get("bisection_depth", 0), row.get("retry", 0)
        if depth > 0:
            effective, reason = False, "bisection"
        elif retry > 0:
            effective, reason = False, "ordinary_retry"
        elif key in seen_batches:
            effective, reason = False, "resume_or_repeated_root_batch"
        else:
            effective, reason = True, "first_recorded_root_batch_attempt"
        if depth == 0:
            seen_batches.add(key)
        row["effective_is_initial_request"] = effective
        evidence = {"attempt_id": row["attempt_id"], "stage": key[0], "batch": key[1],
                    "retry": retry, "bisection_depth": depth, "started_at": row.get("started_at"),
                    "closed": row["closed"], "raw_is_initial_request": row.get("is_initial_request"),
                    "effective_is_initial_request": effective, "classification_reason": reason,
                    "raw_category": row.get("category"),
                    "error_subcategory": "backend_repeat_abort" if backend_repeat_abort(row) else None}
        audit.append(evidence)
        if row.get("is_initial_request") is not effective:
            corrections.append(evidence)

    def summarize(selected):
        result = {}
        for label, group in (("first_attempt", [r for r in selected if r["effective_is_initial_request"]]),
                             ("retry_or_bisection", [r for r in selected if not r["effective_is_initial_request"]]),
                             ("all_requests", selected)):
            closed = [r for r in group if r["closed"]]
            failures = [r for r in closed if r.get("status") != "success"]
            categories = collections.Counter(r.get("category", "unknown") for r in failures)
            result[label] = {"requests": len(closed), "started_requests": len(group),
                             "unclosed_requests": len(group) - len(closed), "failures": len(failures),
                             "failure_rate": len(failures) / len(closed) if closed else None,
                             "output_format_failures": sum(r.get("category") in OUTPUT_FAILURES and not backend_repeat_abort(r) for r in failures),
                             "backend_repeat_abort": sum(backend_repeat_abort(r) for r in failures),
                             "categories": dict(categories)}
        return result

    groups = summarize(attempts)
    groups["by_stage"] = {stage: summarize([r for r in attempts if r.get("stage") == stage]) for stage in STAGES}
    groups["classification_audit"] = audit
    groups["classification_corrections"] = corrections
    groups["classification_method"] = "Chronological first root request per stable (stage,batch); replay after interruption is retry; bisection depth>0 is always retry. Raw flags retained. Rates use closed requests only; unclosed counts remain explicit."
    rows = [r for r in attempts if r["closed"]]
    usable = [r for r in rows if number(r.get("eval_count")) and r["eval_count"] >= 0
              and number(r.get("eval_duration")) and r["eval_duration"] > 0]
    tokens = sum(r["eval_count"] for r in usable)
    seconds = sum(r["eval_duration"] / 1e9 for r in usable)
    rates = [r["eval_count"] / (r["eval_duration"] / 1e9) for r in usable]
    timestamps = []
    for row in rows:
        for field in ("started_at", "finished_at"):
            try:
                timestamps.append(datetime.fromisoformat(row[field]))
            except (KeyError, ValueError, TypeError):
                pass
    groups["timing"] = {
        "requests_with_generation_timing": len(usable), "generated_tokens": tokens,
        "generation_seconds_sum": seconds,
        "generation_tokens_per_second_weighted": tokens / seconds if seconds else None,
        "generation_tokens_per_second_request_mean": sum(rates) / len(rates) if rates else None,
        "attempt_wall_seconds_sum": sum(r["elapsed_seconds"] for r in rows if number(r.get("elapsed_seconds"))),
        "recorded_request_span_seconds": (max(timestamps) - min(timestamps)).total_seconds() if timestamps else None,
        "meaning": "Observed request evidence including successful and failed/retried attempts; generation timing excludes prompt prefill/load; not a performance prediction.",
    }
    groups["unclosed_attempts"] = sorted(set(started) - set(finished))
    return groups


def inspect_models(run_dir: Path, expected: list, dataset: dict, warnings: list) -> list[dict]:
    units = {runner.id_key(unit["id"]): unit for unit in dataset["units"]}
    parent_status = read(run_dir.parent / "STATUS.json", {})
    models = []
    for config in expected:
        mid = config["id"]
        folder = run_dir / mid
        manifest = read(folder / "manifest.json", {})
        fingerprint = manifest.get("fingerprint")
        stages = {stage: inspect_stage(folder / (stage + ".json"), units, fingerprint) for stage in STAGES}
        saved_source = manifest.get("source_sha256")
        if manifest and saved_source != runner.digest(dataset):
            warnings.append(f"{mid} 的源数据指纹与本次 dataset 不符，不能确认可比性。")
            for stage in stages.values():
                stage["terminal"] = False
                stage["identity_problems"].append("source_fingerprint_mismatch")
        all_terminal = all(stage["terminal"] for stage in stages.values())
        all_success = all(stage["all_success"] for stage in stages.values()) and all_terminal
        recorded_failure = read(folder / "preflight_failure.json", {})
        supervisor = parent_status.get("models", {}).get(mid, {})
        any_artifact = any(stage["state"] != "missing" for stage in stages.values())
        integration_failure = bool(recorded_failure) and not any_artifact
        if not any_artifact and supervisor.get("state") == "pending_human":
            integration_failure = True
        if all_terminal:
            state = "complete" if all_success else "complete_with_failures"
        elif integration_failure:
            state = "integration_failed"
        else:
            state = "partial" if any_artifact else "pending"
        actual_config = manifest.get("model", {})
        request_evidence = request_stats(folder / "attempts.jsonl", warnings)
        models.append({"id": mid, "state": state, "all_stages_terminal": all_terminal,
                       "complete_success_coverage": all_success, "stages": stages,
                       "host_preserve_target_units": manifest.get("host_preserve_target_units"),
                       "model_target_units": manifest.get("model_target_units"),
                       "source_model": actual_config.get("source_model", config.get("model")),
                       "source_digest": actual_config.get("source_digest", actual_config.get("digest")),
                       "runtime_model": actual_config.get("model"), "runtime_digest": actual_config.get("digest"),
                       "integration_failure": recorded_failure or ({"reason": supervisor.get("reason")} if integration_failure else None),
                       "requests": request_evidence,
                       "final_failed_units": sum(s["counts"]["failed"] for s in stages.values()),
                       "final_skipped_units": sum(s["counts"]["skipped"] for s in stages.values())})
    return models


def metric_index(metrics: dict, models: list[dict], warnings: list, snapshot_valid: bool = False) -> dict:
    stages = {(m["id"], stage): m["stages"][stage] for m in models for stage in STAGES}
    artifacts = {(a.get("model"), a.get("stage")): a.get("artifact", {}).get("sha256") for a in metrics.get("arms", [])}
    index = {}
    for row in metrics.get("results", []):
        key = (row.get("reference_set"), row.get("normalization"), row.get("model"), row.get("stage"))
        if key in index:
            raise ValueError("duplicate metric identity")
        arm = (key[2], key[3])
        fresh = snapshot_valid and (arm[0] == "baseline" or bool(artifacts.get(arm) and artifacts[arm] == stages.get(arm, {}).get("artifact_sha256")))
        if not fresh:
            warnings.append(f"评分快照过期或无有效输出指纹：{key[2]}/{key[3]}；不纳入当前排序。")
        index[key] = {**row, "current_artifact": fresh}
    return index


def detail_index(details: dict) -> dict:
    return {(r.get("reference_set"), r.get("normalization"), r.get("model"), r.get("stage")): r.get("groups", [])
            for r in details.get("results", [])}


def overlap_cer(groups: list[dict]) -> dict:
    selected = [g for g in groups if g.get("unit_ids") and not g.get("missing_source")]
    usable = [g for g in selected if number(g.get("reference_characters")) and number(g.get("edit_distance"))]
    chars = sum(g["reference_characters"] for g in usable)
    edits = sum(g["edit_distance"] for g in usable)
    return {"cer_micro": edits / chars if chars else None, "groups": len(usable),
            "reference_characters": chars, "edit_distance_sum": edits}


def analyze_asr(index: dict, details: dict, models: list[dict]) -> dict:
    ref, norm = "screen_human_ja", "ja_nfkc_kanafold"
    baseline_key = (ref, norm, "baseline", "raw_asr")
    baseline_row = index.get(baseline_key, {})
    base_full = baseline_row.get("metrics", {}).get("cer_micro")
    base_overlap = overlap_cer(details.get(baseline_key, []))
    rows = []
    status = {m["id"]: m["stages"]["asr_polish"] for m in models}
    for mid, stage in [("baseline", "raw_asr")] + [(m["id"], "asr_polish") for m in models]:
        key = (ref, norm, mid, stage)
        metric = index.get(key, {})
        full = metric.get("metrics", {}).get("cer_micro")
        overlap = overlap_cer(details.get(key, []))
        rows.append({"model": mid, "stage": stage, "cer_all_reference_groups": full,
                     "cer_delta_vs_raw": full - base_full if number(full) and number(base_full) else None,
                     "overlap_only": overlap,
                     "overlap_cer_delta_vs_raw": overlap["cer_micro"] - base_overlap["cer_micro"] if number(overlap["cer_micro"]) and number(base_overlap["cer_micro"]) else None,
                     "coverage": metric.get("coverage", {}), "current_artifact": metric.get("current_artifact", False),
                     "comparison_eligible": bool(metric.get("current_artifact") and number(full) and (mid == "baseline" or status[mid]["terminal"]))})
    eligible = [row for row in rows if row["comparison_eligible"]]
    minimum = min((row["cer_all_reference_groups"] for row in eligible), default=None)
    leaders = [{"model": row["model"], "stage": row["stage"], "cer": row["cer_all_reference_groups"]}
               for row in eligible if abs(row["cer_all_reference_groups"] - minimum) <= 1e-9]
    return {"reference_set": ref, "normalization": norm, "rows": rows, "lowest_cer_arms": leaders,
            "eligible_model_corrections": sum(row["model"] != "baseline" for row in eligible),
            "meaning": "All-reference CER retains missing-source deletions; overlap-only CER uses the fixed source-overlap groups, not a model-selected successful subset. Lower means closer text, not verified transcription accuracy."}


def compare_zh(index: dict, models: list[dict], reference: str, final_allowed: bool) -> dict:
    combinations = [(m["id"], stage) for m in models for stage in ZH_STAGES]
    advanced = "zh_nfkc_opencc_t2s"
    norm = advanced if combinations and all((reference, advanced, mid, stage) in index for mid, stage in combinations) else "zh_nfkc"
    model_map = {m["id"]: m for m in models}
    rows = []
    for mid, stage in combinations:
        metric = index.get((reference, norm, mid, stage), {})
        stage_status = model_map[mid]["stages"][stage]
        values, coverage = metric.get("metrics", {}), metric.get("coverage", {})
        valid = metric.get("current_artifact") and number(values.get("chrf_corpus")) and number(values.get("cer_micro"))
        eligible = bool(valid and stage_status["terminal"])
        rows.append({"model": mid, "stage": stage, "chrf": values.get("chrf_corpus"), "cer": values.get("cer_micro"),
                     "coverage": coverage, "stage_counts": stage_status["counts"], "stage_terminal": stage_status["terminal"],
                     "stage_success_fraction": stage_status["success_fraction"], "current_artifact": bool(metric.get("current_artifact")),
                     "comparison_eligible": eligible, "rank": None})
    rows.sort(key=lambda row: (not number(row["chrf"]), -row["chrf"] if number(row["chrf"]) else 0,
                               row["cer"] if number(row["cer"]) else float("inf"), row["model"], row["stage"]))
    eligible = [row for row in rows if row["comparison_eligible"]]
    previous, rank = None, 0
    for position, row in enumerate(eligible, 1):
        if previous is None or abs(row["chrf"] - previous) > 1e-9:
            rank = position
        row["rank"] = rank
        previous = row["chrf"]
    leaders = [{"model": row["model"], "stage": row["stage"], "chrf": row["chrf"], "cer": row["cer"],
                "success_fraction": row["stage_success_fraction"]} for row in eligible if row["rank"] == 1]
    deltas = []
    for model in models:
        by_stage = {row["stage"]: row for row in rows if row["model"] == model["id"]}
        for after, before in (("translate_polished", "translate_raw"), ("polish_translation_raw", "translate_raw"),
                              ("polish_translation_polished", "translate_polished")):
            a, b = by_stage[after], by_stage[before]
            usable = a["comparison_eligible"] and b["comparison_eligible"]
            deltas.append({"model": model["id"], "after": after, "before": before, "comparable_terminal_stages": usable,
                           "chrf_delta": a["chrf"] - b["chrf"] if usable else None,
                           "cer_delta": a["cer"] - b["cer"] if usable else None,
                           "success_fraction_after": a["stage_success_fraction"], "success_fraction_before": b["stage_success_fraction"]})
    return {"reference_set": reference, "normalization": norm, "rows": rows, "leaders": leaders,
            "final_comparison": final_allowed and len(eligible) == len(combinations),
            "comparison_scope": "Completed terminal stages only; missing/failed outputs retain their preset empty-output score penalty. Incomplete or stale stages have no rank.",
            "path_deltas": deltas}


def reference_agreement(chinese: dict) -> dict:
    first, second = [chinese[k] for k in ("screen_human_zh", "fansub_zh")]
    a = {(row["model"], row["stage"]) for row in first["leaders"]}
    b = {(row["model"], row["stage"]) for row in second["leaders"]}
    common = sorted(a & b)
    if not a or not b:
        state = "insufficient_evidence"
    elif len(a) == len(b) == 1 and a == b:
        state = "same_unique_text_agreement_leader"
    elif common:
        state = "shared_tied_text_agreement_leaders"
    else:
        state = "reference_leader_conflict"
    return {"state": state, "shared_leaders": [{"model": mid, "stage": stage} for mid, stage in common],
            "final": first["final_comparison"] and second["final_comparison"],
            "not_a_semantic_quality_verdict": True}


def human_lines(analysis: dict) -> list[str]:
    completed = analysis["completed_experiment"]
    restricted = analysis["restricted_completed_scope"]
    qualifier = "完整实验的流程输出比较" if completed else "受限于已完成模型的比较" if restricted else "临时观察，禁止作为四模型最终排名"
    lines = ["字幕文本模型实验：自动比较观察", SCOPE,
             f"当前状态：{analysis['state']}；{qualifier}。",
             f"完整终态模型 {analysis['terminal_model_count']}/{analysis['expected_model_count']}；全阶段流程完整输出模型 {analysis['fully_successful_model_count']}；接入失败模型 {len(analysis['integration_failed_models'])}。",
             "字符匹配度不是音频核验准确率、自然度或语义忠实度；没有听音复核，也不代表广告、多语言或正式发布验收。",
             "本轮验证 JSON Schema 约束输出与固定 ID，不等于函数调用／工具使用能力评测，也没有进行分段 TTS 合成试听。",
             "失败／缺失输出按预设空文本规则受罚，并另列覆盖率；由流程失败拉低的分数不能解释成语言能力低。"]
    lines.append(COPY_PROTOCOL)
    lines.append(HOST_PRESERVE_PROTOCOL)
    lines.append(f"完整源数据评分人口保持 {analysis['source_units']} 个单位；主机保留与模型操作均使用原有时间、ID 和参考，不剔除任何历史标签类别。")
    if not analysis["score_snapshot"]["valid"]:
        lines.insert(3, "评分指标、明细或输入共同快照未通过验证；暂停当前排名与重叠组 CER，已展示数值仅供追查历史证据。")
    if not completed:
        pending = [m["id"] for m in analysis["models"] if m["state"] in ("pending", "partial")]
        if pending:
            lines.append("尚未完成：" + "、".join(pending) + "。这些模型或阶段不参与最终排名，缺测不能当成劣质输出。")
    for model in analysis["models"]:
        if model["state"] == "integration_failed":
            lines.append(f"{model['id']}：当前接入失败，未证明完整文本链可用；原因 {json.dumps(model['integration_failure'], ensure_ascii=False)}。")
    lines.append("各阶段操作（仅统计有完整文本的流程输出；主机保留与模型返回分开，均不自动等于质量改善）：")
    for model in analysis["models"]:
        if type(model.get("host_preserve_target_units")) is int and type(model.get("model_target_units")) is int:
            lines.append(f"{model['id']} 本次运行清单路由目标：主机保留 {model['host_preserve_target_units']} 个单位、交给模型 {model['model_target_units']} 个单位；smoke 的限量目标不能当作完整人口完成数。")
        for stage in STAGES:
            ops = model["stages"][stage]["operation_counts"]
            lines.append(f"{model['id']} / {LABELS[stage]}：host_preserve {ops['host_preserve']}（主机保留、未发 LLM 请求），模型操作单位 {ops['model_operation_units']}：copy {ops['copy']}（恢复本阶段原文 {ops['copy_source']}、保留译稿 {ops['copy_draft']}），replace {ops['replace']}；流程完整输出 {ops['successful_units']}，旧记录未标操作 {ops['unspecified']}，未知操作 {ops['unknown']}。")
    for ref, comparison in analysis["chinese"].items():
        if comparison["leaders"]:
            winners = "、".join(f"{r['model']} / {LABELS[r['stage']]}（chrF {fmt(r['chrf'])}）" for r in comparison["leaders"])
            lines.append(f"{'最终流程输出比较' if comparison['final_comparison'] else '临时／受限观察'} · {REF_LABELS[ref]}：字面匹配最高为 {winners}。")
    agreement = analysis["two_reference_agreement"]
    if agreement["state"] == "reference_leader_conflict":
        lines.append("主要观察：两个参考的最高匹配方案冲突，不能强选统一质量冠军。")
    elif agreement["state"] == "same_unique_text_agreement_leader":
        winner = agreement["shared_leaders"][0]
        lines.append(f"主要观察：两个参考在{'完整实验' if agreement['final'] else '当前可比范围'}中一致支持 {winner['model']} / {LABELS[winner['stage']]} 的最高字面匹配；不是语义质量认定。")
    lines.extend(["", "日文 ASR 文本纠错：历史 Claude 日文参考，NFKC＋假名折叠；CER 越低越接近。",
                  "全参考 CER 包含无源重叠的缺句；重叠组 CER 只限固定存在 ASR 的组。文本纠错不能可靠补回缺失音频内容。"])
    for row in analysis["asr"]["rows"]:
        lines.append(f"{row['model']}：全参考 CER {fmt(row['cer_all_reference_groups'])}，相对原始日文 ΔCER {fmt(row['cer_delta_vs_raw'])}；固定重叠组 CER {fmt(row['overlap_only']['cer_micro'])}，ΔCER {fmt(row['overlap_cer_delta_vs_raw'])}。")
    if analysis["asr"]["eligible_model_corrections"]:
        lines.append("当前可比的日文方案中，全参考 CER 最低：" + "、".join(
            f"{'原始 ASR 基线' if row['model'] == 'baseline' else row['model']}（{fmt(row['cer'])}）"
            for row in analysis["asr"]["lowest_cer_arms"]) + "。")
    for ref, comparison in analysis["chinese"].items():
        lines.extend(["", f"中文参考：{REF_LABELS[ref]}；归一化 {comparison['normalization']}；{'最终流程输出比较' if comparison['final_comparison'] else '当前已完成阶段的临时／受限比较'}。",
                      "按 chrF 字面匹配值降序；未终态／过期评分无名次。不同参考不混合分数。"])
        for row in comparison["rows"]:
            rank = f"第 {row['rank']} 名" if row["rank"] is not None else "不排名"
            coverage = row["coverage"]
            counts = row["stage_counts"]
            lines.append(f"{rank} · {row['model']} / {LABELS[row['stage']]}：chrF {fmt(row['chrf'])}，CER {fmt(row['cer'])}；参考覆盖 {coverage.get('reference_with_output', '—')}/{coverage.get('reference_total', '—')}，完全覆盖 {coverage.get('reference_fully_covered', '—')}；流程完整输出占比（含主机保留）{fmt(row['stage_success_fraction'])}，失败 {counts['failed']}、跳过 {counts['skipped']}、待处理／缺失 {counts['pending'] + counts['missing']}。")
        if comparison["leaders"]:
            leaders = "、".join(f"{r['model']} / {LABELS[r['stage']]}（chrF {fmt(r['chrf'])}）" for r in comparison["leaders"])
            lines.append("该参考当前可比较流程中字面匹配最高：" + leaders + "。")
        for delta in comparison["path_deltas"]:
            lines.append(f"路径差值 {delta['model']}：{LABELS[delta['after']]} − {LABELS[delta['before']]}：ΔchrF {fmt(delta['chrf_delta'])}，ΔCER {fmt(delta['cer_delta'])}。")
    agreement = analysis["two_reference_agreement"]
    lines.append("")
    if agreement["state"] == "same_unique_text_agreement_leader":
        leader = agreement["shared_leaders"][0]
        lines.append(f"{'两参考最终一致' if agreement['final'] else '临时／受限范围内两参考一致'}：{leader['model']} / {LABELS[leader['stage']]} 的 chrF 字面匹配最高；这不是语义质量冠军结论。")
    elif agreement["state"] == "shared_tied_text_agreement_leaders":
        lines.append("两个参考存在共同并列最高路径，不能强选唯一最佳方案；仅能说明该范围内字面匹配并列。")
    elif agreement["state"] == "reference_leader_conflict":
        lines.append("两个参考的最高匹配方案发生冲突：必须分别报告，不混合评分，不强选统一质量冠军。")
    else:
        lines.append("当前证据不足，尚不能判断两个参考是否支持同一最高匹配方案。")
    lines.extend(["", "请求可靠性与实测开销（包括失败和重试，不作性能预测）："])
    for model in analysis["models"]:
        request = model["requests"]
        first, retry, timing = request["first_attempt"], request["retry_or_bisection"], request["timing"]
        lines.append(f"{model['id']}：首次已结束请求失败 {first['failures']}/{first['requests']}（输出格式类 {first['output_format_failures']}，后端重复保护中止 backend_repeat_abort {first['backend_repeat_abort']}，未闭合首次 {first['unclosed_requests']}）；重试／二分已结束请求失败 {retry['failures']}/{retry['requests']}（输出格式类 {retry['output_format_failures']}，后端重复保护中止 {retry['backend_repeat_abort']}，未闭合重试 {retry['unclosed_requests']}）；最终失败 {model['final_failed_units']}、跳过 {model['final_skipped_units']} 个阶段单位。")
        if first["backend_repeat_abort"] or retry["backend_repeat_abort"]:
            lines.append("  backend_repeat_abort 按错误详情单列；原始 transport 等类别保留，不计为 JSON／输出格式失败。")
        if request["classification_corrections"]:
            lines.append(f"  按稳定阶段／批次和开始时间校正 {len(request['classification_corrections'])} 条首次标记（含续跑重发）；原始标记和校正理由保留在 analysis.json 的 classification_audit。")
        lines.append(f"  实测生成速度 {fmt(timing['generation_tokens_per_second_weighted'], 2)} token/s（按生成时长加权；逐请求均值 {fmt(timing['generation_tokens_per_second_request_mean'], 2)}）；请求墙钟耗时累计 {fmt(timing['attempt_wall_seconds_sum'], 2)} 秒，请求时间跨度 {fmt(timing['recorded_request_span_seconds'], 2)} 秒；未闭合请求 {len(request['unclosed_attempts'])}。")
    if analysis["warnings"]:
        lines.append("证据警告：" + "；".join(analysis["warnings"]))
    return lines


def analyze(dataset: dict, references: dict, metrics: dict, details: dict, run_dir: Path,
            dataset_path: Path | None = None, references_path: Path | None = None) -> dict:
    warnings = []
    expected = expected_models(run_dir, warnings)
    models = inspect_models(run_dir, expected, dataset, warnings)
    snapshot = verify_score_snapshot(metrics, details, run_dir, dataset_path, references_path, dataset, references)
    if not snapshot["valid"]:
        warnings.append("评分共同快照未通过验证：" + ", ".join(snapshot["problems"]))
    index = metric_index(metrics, models, warnings, snapshot["valid"])
    dindex = detail_index(details) if snapshot["valid"] else {}
    terminal = [m for m in models if m["all_stages_terminal"]]
    pending = [m for m in models if m["state"] in ("pending", "partial")]
    failed = [m["id"] for m in models if m["state"] == "integration_failed"]
    completed = len(expected) == 4 and len(terminal) == len(expected)
    restricted = bool(terminal) and bool(failed) and not pending and len(expected) == 4
    success = completed and all(m["complete_success_coverage"] for m in models)
    state = "complete" if success else "complete_with_failures" if completed else "terminal_with_integration_failures" if restricted else "partial" if any(m["state"] != "pending" for m in models) else "pending"
    chinese = {ref: compare_zh(index, models, ref, completed) for ref in REF_LABELS}
    analysis = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "scope": SCOPE,
                "copy_protocol_interpretation": COPY_PROTOCOL,
                "host_preserve_protocol_interpretation": HOST_PRESERVE_PROTOCOL,
                "state": state, "source_units": len(dataset["units"]), "expected_model_count": len(expected),
                "terminal_model_count": len(terminal), "fully_successful_model_count": sum(m["complete_success_coverage"] for m in models),
                "completed_experiment": completed, "complete_success_coverage": success,
                "restricted_completed_scope": restricted, "integration_failed_models": failed,
                "models": models, "score_snapshot": snapshot, "asr": analyze_asr(index, dindex, models), "chinese": chinese,
                "two_reference_agreement": reference_agreement(chinese), "warnings": sorted(set(warnings)),
                "reference_interpretation": "Historical Claude-read screen references and downloaded subtitle references are separate; OCR is supplementary and not used to pick a main leader; none is audio-certified truth.",
                "evidence": {"metrics_sha256": sha(run_dir / "metrics.json"), "details_sha256": sha(run_dir / "score_details.json"),
                             "expected_models_sha256": sha(run_dir / "EXPECTED_MODELS.json"), "analysis_code_sha256": sha(Path(__file__))}}
    analysis["human_readable_lines"] = human_lines(analysis)
    return analysis


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        store = runner.PrivateStore(args.run_dir)
        source, _ = runner.load_dataset(args.dataset)
        references = read(args.references)
        if not isinstance(references, dict) or not isinstance(references.get("sets"), dict):
            raise ValueError("invalid references file")
        result = analyze(source, references, read(args.run_dir / "metrics.json", {}), read(args.run_dir / "score_details.json", {}),
                         args.run_dir, args.dataset, args.references)
        result["evidence"].update(dataset_sha256=sha(args.dataset), references_sha256=sha(args.references))
        store.json(args.run_dir / "analysis.json", result)
        store.write(args.run_dir / "结论.txt", ("\n".join(result["human_readable_lines"]) + "\n").encode("utf-8"))
        print(json.dumps({"analysis": str(args.run_dir / "analysis.json"), "state": result["state"],
                          "completed_experiment": result["completed_experiment"], "complete_success_coverage": result["complete_success_coverage"]}, ensure_ascii=False))
        return 0 if result["score_snapshot"]["valid"] else 1
    except (OSError, ValueError, RuntimeError, runner.Failure) as exc:
        print(f"ANALYSIS_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
