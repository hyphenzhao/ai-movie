#!/usr/bin/env python3
"""Render a private, text-only benchmark review page from existing evidence.

Scope: [README's full scope](../../README.md#用途与处理范围). This isolated
experiment evaluates user-selected existing subtitle text transformations; it
does not generate new story content or media, review legality/content, certify
advertising readiness, or establish multilingual/translation accuracy.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
import importlib.util
import json
import math
import re
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

SPEC = importlib.util.spec_from_file_location("subtitle_benchmark_runner_for_report", Path(__file__).with_name("runner.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
ANALYZE_SPEC = importlib.util.spec_from_file_location("subtitle_benchmark_analysis_for_report", Path(__file__).with_name("analyze.py"))
analysis_tools = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analysis_tools)

STAGES = runner.STAGES
LABELS = {
    "asr_polish": "日文转录文本纠错",
    "translate_raw": "原始日文 → 中文",
    "translate_polished": "校正日文 → 中文",
    "polish_translation_raw": "原始日文 → 中文 → 中文润色",
    "polish_translation_polished": "校正日文 → 中文 → 中文润色",
}
STATUS = {
    "complete": "完成", "complete_with_failures": "完成（含失败或跳过）",
    "partial": "部分完成", "failed": "失败", "not_started": "尚未开始",
    "running": "运行中", "pending": "待处理", "success": "流程输出完整",
    "skipped": "跳过", "unavailable": "无可用记录",
}
WARNING = "字符相似度不是翻译准确率；不同参考文本的分数不能混合。ASR 纠错仅依据字幕文本推断，没有重新听音核验。"
REFERENCE_LABELS = {
    "screen_human_ja": "画面日文：历史 Claude 读取参考",
    "screen_human_zh": "画面中文：历史 Claude 读取参考",
    "screen_ocr_only_ja": "画面日文：OCR 补充层",
    "screen_ocr_only_zh": "画面中文：OCR 补充层",
    "screen_merged_ja": "画面日文：Claude＋OCR 合并参考",
    "screen_merged_zh": "画面中文：Claude＋OCR 合并参考",
    "fansub_zh": "下载中文字幕：独立外部比较参考",
}
METRIC_LABELS = (
    ("metrics", "cer_micro", "字符编辑距离 / 参考字符数 CER（越低越接近）"),
    ("metrics", "sequence_similarity_macro", "分组字符序列相似度均值（0–1）"),
    ("metrics", "chrf_corpus", "字符 n-gram chrF（0–100）"),
    ("metrics", "groups_scored", "纳入评分的对齐组数"),
    ("metrics", "reference_characters", "参考字符数（归一化后）"),
    ("coverage", "reference_total", "参考字幕条数"),
    ("coverage", "reference_with_output", "至少有部分输出的参考条数"),
    ("coverage", "reference_fully_covered", "所关联源单位均有输出的参考条数"),
    ("coverage", "reference_output_fraction", "参考条目输出覆盖比（0–1）"),
    ("coverage", "reference_without_source", "无时间重叠源单位的参考条数"),
    ("coverage", "reference_without_output", "无输出的参考条数（含无源匹配）"),
    ("coverage", "source_units", "该参考对齐范围内的源单位数"),
    ("coverage", "successful_units", "该范围内完整输出单位数（含主机保留）"),
    ("coverage", "failed_units", "该范围内非成功状态单位数"),
    ("coverage", "missing_units", "该范围内缺失单位数"),
    ("coverage", "empty_output_units", "该范围内空文本单位数"),
    ("coverage", "output_units_outside_reference_population", "参考对齐范围之外的输出单位数"),
)


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"), quote=True)


def read_json(path: Path, errors: list[str], required: bool = False) -> dict | list | None:
    if not path.exists():
        if required:
            errors.append(f"缺少文件：{path.name}")
        return None
    try:
        return runner.strict_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, runner.Failure) as exc:
        errors.append(f"读取失败 {path}: {type(exc).__name__}: {exc}")
        return None


def status_counts(units: list[dict], target: int) -> dict[str, int]:
    known = {"success", "failed", "skipped", "pending", "unavailable"}
    counts = collections.Counter(u.get("status") if u.get("status") in known else "unavailable" for u in units)
    counts["pending"] += max(0, target - len(units))
    return {key: counts[key] for key in ("success", "failed", "skipped", "pending", "unavailable")}


def status_text(counts: dict) -> str:
    return " / ".join(f"{STATUS.get(k, k)} {v}" for k, v in counts.items() if v) or "无单位记录"


def model_state(stages: dict, preflight_failure: dict | None, target: int) -> str:
    if preflight_failure and not any(stage.get("units") for stage in stages.values()):
        return "failed"
    if not any(stage.get("units") for stage in stages.values()):
        return "not_started"
    all_terminal = all(stages.get(name, {}).get("state") == "complete"
                       and len(stages[name].get("units", [])) == target
                       and all(u.get("status") in ("success", "failed", "skipped") for u in stages[name]["units"])
                       for name in STAGES)
    if all_terminal:
        all_success = all(u.get("status") == "success" for stage in stages.values() for u in stage.get("units", []))
        return "complete" if all_success else "complete_with_failures"
    return "partial"


def load_models(run_dir: Path, errors: list[str], total_units: int) -> list[dict]:
    models = []
    for folder in sorted(run_dir.iterdir()) if run_dir.exists() else []:
        if not folder.is_dir() or not ((folder / "manifest.json").exists() or any((folder / f"{s}.json").exists() for s in STAGES)):
            continue
        manifest = read_json(folder / "manifest.json", errors) or {}
        stages = {s: read_json(folder / f"{s}.json", errors) or {} for s in STAGES}
        summary = read_json(folder / "summary.json", errors) or {}
        if (folder / "attempts.jsonl").exists():
            summary = summarize_attempt_log(folder / "attempts.jsonl", summary, errors)
        progress = read_json(folder / "progress.json", errors) or {}
        failure = read_json(folder / "preflight_failure.json", errors)
        target = manifest.get("target_units", total_units)
        if type(target) is not int or target < 0:
            target = total_units
        counts = {s: status_counts(stages[s].get("units", []), target) for s in STAGES}
        models.append({"id": folder.name, "folder": folder, "manifest": manifest, "stages": stages,
                       "summary": summary, "progress": progress, "preflight_failure": failure,
                       "target_units": target, "state": model_state(stages, failure, target), "counts": counts})
    expected = read_json(run_dir / "EXPECTED_MODELS.json", errors)
    if expected is None:
        return models
    if not isinstance(expected, list):
        errors.append("EXPECTED_MODELS.json 必须是 [{id, model}] 数组；未能确定完整预定模型集合。")
        return models
    observed = {model["id"]: model for model in models}
    observed_targets = {model["target_units"] for model in models}
    planned_target = next(iter(observed_targets)) if len(observed_targets) == 1 else total_units
    ordered, seen = [], set()
    for config in expected:
        if not isinstance(config, dict) or not isinstance(config.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["id"]):
            errors.append("EXPECTED_MODELS.json 含无效模型 ID；已保留错误提示，未把该条当作已完成。")
            continue
        model_id = config["id"]
        if model_id in seen:
            errors.append(f"EXPECTED_MODELS.json 含重复模型 ID：{model_id}")
            continue
        seen.add(model_id)
        if model_id in observed:
            model = observed[model_id]
            model["expected_config"] = config
            ordered.append(model)
            continue
        # An in-memory placeholder represents the plan, never fabricated run
        # evidence. No model directory, manifest or stage file is created.
        ordered.append({"id": model_id, "folder": run_dir / model_id, "manifest": {},
                        "expected_config": config, "stages": {stage: {} for stage in STAGES},
                        "summary": {}, "progress": {}, "preflight_failure": None,
                        "target_units": planned_target, "state": "not_started",
                        "counts": {stage: status_counts([], planned_target) for stage in STAGES}})
    ordered.extend(model for model in models if model["id"] not in seen)
    return ordered


def summarize_attempt_log(path: Path, prior: dict, errors: list[str]) -> dict:
    """Share the evaluator's chronological resume-aware request classification."""
    audit = analysis_tools.request_stats(path, errors)
    return {**prior, "stages": audit["by_stage"], "interrupted_attempts": audit["unclosed_attempts"],
            "classification_corrections": audit["classification_corrections"],
            "classification_method": audit["classification_method"]}


def format_number(value: Any) -> str:
    if value is None:
        return "—"
    if type(value) is float:
        return f"{value:.4f}" if math.isfinite(value) else "无效数值"
    if isinstance(value, (dict, list)):
        return esc(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return esc(value)


def table(headers: list[str], rows: list[list[str]], css: str = "") -> str:
    return f'<div class="table-wrap {css}"><table><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in headers) + \
           "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows) + \
           "</tbody></table></div>"


def render_status(models: list[dict], total_units: int) -> str:
    if not models:
        return '<p class="notice">尚未发现模型输出；没有可下结论的评测结果。</p>'
    headings = ["阶段 / 覆盖"] + [m["id"] for m in models]
    rows = [["模型链路状态"] + [f'<strong class="state {esc(m["state"])}">{esc(STATUS[m["state"]])}</strong>' for m in models]]
    rows.append(["本次目标 / 全文单位"] + [f'{m["target_units"]} / {total_units}' for m in models])
    rows.append(["本次运行清单路由目标（smoke 按限量目标记录）"] + [
        f'主机保留 {esc(m["manifest"].get("host_preserve_target_units"))} · 交给模型 {esc(m["manifest"].get("model_target_units"))}' for m in models])
    for stage in STAGES:
        row = [esc(LABELS[stage])]
        for model in models:
            counts = model["counts"][stage]
            denominator = model["target_units"]
            pct = f'{100 * counts["success"] / denominator:.1f}%' if denominator else "—"
            row.append(f'<b>{esc(status_text(counts))}</b><br><span class="muted">流程完整输出覆盖（含主机保留）{pct}；阶段记录 {esc(model["stages"][stage].get("state", "missing"))}</span>')
        rows.append(row)
    parts = ["<h2>执行状态与覆盖</h2>", table(headings, rows)]
    for model in models:
        failure = model.get("preflight_failure")
        if failure:
            parts.append(f'<details><summary>{esc(model["id"])} 预检失败记录</summary><pre>{esc(json.dumps(failure, ensure_ascii=False, indent=2))}</pre></details>')
    return "".join(parts)


def render_operations(models: list[dict]) -> str:
    rows = []
    for stage in STAGES:
        row = [esc(LABELS[stage])]
        for model in models:
            counts = analysis_tools.operation_counts(model["stages"][stage].get("units", []), stage)
            row.append(f'<b>主机 host_preserve {counts["host_preserve"]}（未发 LLM 请求）</b><br>'
                       f'<b>模型 copy {counts["copy"]} · replace {counts["replace"]}</b><br>'
                       f'恢复本阶段原文 {counts["copy_source"]} · 保留译稿 {counts["copy_draft"]}<br>'
                       f'模型操作单位 {counts["model_operation_units"]} · 流程完整输出 {counts["successful_units"]}<br>'
                       f'<span class="muted">旧记录未标操作 {counts["unspecified"]} · 未知操作 {counts["unknown"]}</span>')
        rows.append(row)
    return '<h2>逐阶段主机保留与模型操作</h2><p>' + esc(analysis_tools.COPY_PROTOCOL) + '</p><p>' + esc(analysis_tools.HOST_PRESERVE_PROTOCOL) + '</p>' + \
        table(["阶段"] + [m["id"] for m in models], rows)


def render_requests(models: list[dict]) -> str:
    parts = ["<h2>JSON、ID 与请求错误</h2><p>按稳定阶段／批次及开始时间重建首次请求；中断后的同批重发计入重试，二分请求始终另计。原始标记保留。失败率只使用已结束请求，未闭合请求单独列示，不能视为成功。backend_repeat_abort 按后端重复保护错误详情单列，保留原始 transport 类别，不计入 JSON／输出格式失败。</p>"]
    for group, label in (("first_attempt", "首次请求"), ("retry_or_bisection", "重试／二分请求")):
        rows = []
        for stage in STAGES:
            row = [esc(LABELS[stage])]
            for model in models:
                stats = model["summary"].get("stages", {}).get(stage, {}).get(group)
                if not isinstance(stats, dict):
                    row.append("无汇总记录")
                    continue
                rate = stats.get("failure_rate")
                rate_text = f"{100 * rate:.1f}%" if isinstance(rate, (int, float)) else "—"
                categories = "，".join(f"{esc(k)} {esc(v)}" for k, v in sorted(stats.get("categories", {}).items())) or "无已记录错误"
                row.append(f'{esc(stats.get("failures"))} / {esc(stats.get("requests"))} 已结束请求失败（{rate_text}）<br>输出格式类 {esc(stats.get("output_format_failures", 0))} · 后端重复保护中止 {esc(stats.get("backend_repeat_abort", 0))}<br>未闭合 {esc(stats.get("unclosed_requests", 0))}<br><span class="muted">{categories}</span>')
            rows.append(row)
        parts.extend([f"<h3>{label}</h3>", table(["阶段"] + [m["id"] for m in models], rows)])
    interrupted_rows = [[esc(m["id"]), str(len(m["summary"].get("interrupted_attempts", []))),
                         esc(", ".join(m["summary"].get("interrupted_attempts", [])))] for m in models]
    parts.append(table(["模型", "无结束记录的请求（可能运行中或中断）", "请求 ID"], interrupted_rows))
    corrected = []
    for model in models:
        for row in model["summary"].get("classification_corrections", []):
            corrected.append([esc(model["id"]), esc(row["attempt_id"]), esc(f"{row['stage']} / {row['batch']}"),
                              esc(row["raw_is_initial_request"]), esc(row["effective_is_initial_request"]), esc(row["classification_reason"])])
    if corrected:
        parts.append('<details open><summary>续跑等请求分类校正（原始日志保持不变）</summary>' + table(
            ["模型", "请求 ID", "阶段 / 批次", "原始首次标记", "有效首次标记", "校正理由"], corrected) + "</details>")
    return "".join(parts)


def render_manifests(models: list[dict]) -> str:
    rows = []
    for model in models:
        manifest = model["manifest"]
        config = manifest.get("model") or model.get("expected_config", {})
        options = manifest.get("options", {})
        params = {**options, **manifest.get("sampling_parameters", {"temperature": 0.2, "seed": 7}),
                  "think": config.get("think", False)} if manifest else {"state": "预定模型尚无运行清单；参数未取证"}
        source_model = config.get("source_model", model.get("expected_config", {}).get("model", config.get("model")))
        source_digest = config.get("source_digest", config.get("digest"))
        identity = f'<b>官方源标签</b><br>{esc(source_model)}<br><b>实际运行标签／转换别名</b><br>{esc(config.get("model"))}'
        hashes = f'<b>官方源 manifest</b><br><code>{esc(source_digest)}</code><br><b>实际运行 manifest</b><br><code>{esc(config.get("digest"))}</code>'
        rows.append([esc(model["id"]), identity, hashes,
                     f'<pre>{esc(json.dumps(params, ensure_ascii=False, indent=2))}</pre>',
                     f'<code>runner: {esc(manifest.get("code_sha256"))}<br>fingerprint: {esc(manifest.get("fingerprint"))}<br>source: {esc(manifest.get("source_sha256"))}</code><details><summary>完整模型配置</summary><pre>{esc(json.dumps(config, ensure_ascii=False, indent=2))}</pre></details>'])
    return "<h2>模型版本与复现实验参数</h2><p>官方源模型与本机运行身份分别记录；Ollama 自动转换生成的别名不代表改用了另一候选模型。</p>" + table(["模型 ID", "官方来源 / 运行身份", "源 / 运行 digest", "请求参数", "版本指纹"], rows)


def render_analysis(analysis: dict | None, metrics_path: Path, details_path: Path, snapshot_valid: bool = True) -> str:
    if not analysis:
        return ""
    lines = analysis.get("human_readable_lines")
    if not isinstance(lines, list):
        return '<h2>自动比较观察</h2><p>analysis.json 尚无有效可读结论。</p>'
    evidence = analysis.get("evidence", {})
    stale = not snapshot_valid or evidence.get("metrics_sha256") != sha256_file(metrics_path) or evidence.get("details_sha256") != sha256_file(details_path)
    warning = '<p class="notice">自动观察与当前评分文件指纹不同；下文只作为历史快照，不应当作当前最终排名。</p>' if stale else ""
    overview = [line for line in lines[:8] if isinstance(line, str)]
    overview.extend(line for line in lines[8:] if isinstance(line, str) and (line.startswith("主要观察：") or " · " in line and "字面匹配最高为" in line))
    return '<h2>自动比较观察</h2>' + warning + "".join(f"<p>{esc(line)}</p>" for line in overview) + \
        '<p><a href="结论.txt">打开完整中文结论</a> · <a href="analysis.json">机器可读比较</a></p><details><summary>展开全部客观比较、路径差值与请求开销</summary><pre style="max-width:none">' + \
        esc("\n".join(str(line) for line in lines)) + "</pre></details>"


def render_source_details(units: list[dict], models: list[dict]) -> str:
    parts = ['<section class="review-section" data-stage="asr_polish"><h2>逐句日文校正复核</h2>',
             "<p>以下只呈现已有文本与模型改动。无音频复核，无法据此确认原始转录或校正后的听写正确率。</p>"]
    indices = {m["id"]: {runner.id_key(u["id"]): u for u in m["stages"]["asr_polish"].get("units", []) if "id" in u} for m in models}
    parts.append('<details><summary>展开日文原文与各模型校正（可按模型筛选）</summary>')
    for unit in units:
        if "id" not in unit:
            continue
        key = runner.id_key(unit["id"])
        parts.append(f'<details class="alignment"><summary>ID {esc(unit["id"])} · {esc(unit.get("start"))}–{esc(unit.get("end"))} 秒 · {esc(unit.get("speaker"))}</summary>')
        parts.append(f'<div class="source"><b>原始日文</b><p>{esc(unit.get("text"))}</p></div>')
        for model in models:
            result = indices[model["id"]].get(key, {})
            status = result.get("status", "pending")
            text = result.get("text") if status == "success" else f'[{STATUS.get(status, status)}] {result.get("category", result.get("reason", ""))}'
            operation = result.get("operation")
            op_label = " · host_preserve：规则保留 dataset 原文，未调用模型" if status == "success" and operation == "host_preserve" else " · copy：模型选择保留，主机恢复原文" if status == "success" and operation == "copy" else " · replace：模型替换文本" if status == "success" and operation == "replace" else ""
            parts.append(f'<div class="model-output" data-model="{esc(model["id"])}"><b>{esc(model["id"])} · {esc(STATUS.get(status, status))}{esc(op_label)}</b><p>{esc(text)}</p></div>')
        parts.append("</details>")
    parts.append("</details></section>")
    return "".join(parts)


def metric_freshness(metrics: dict, models: list[dict], errors: list[str], snapshot_valid: bool = False) -> dict[tuple, bool]:
    by_id = {m["id"]: m for m in models}
    fresh = {}
    for arm in metrics.get("arms", []):
        key = (arm.get("model"), arm.get("stage"))
        if key[0] == "baseline":
            fresh[key] = snapshot_valid
            continue
        model = by_id.get(key[0])
        path = model["folder"] / f"{key[1]}.json" if model and key[1] in STAGES else None
        expected = arm.get("artifact", {}).get("sha256")
        actual = sha256_file(path) if path else None
        fresh[key] = bool(snapshot_valid and expected and actual == expected)
        if not fresh[key]:
            errors.append(f"评分快照与当前输出指纹不一致或无法核对：{key[0]} / {key[1]}；此项仅显示为历史快照。")
    return fresh


def render_reference_notes(references: dict) -> str:
    rows = []
    for name, refset in references.get("sets", {}).items():
        notes = "未听音验证；不作为已认证真值。"
        if "screen_human" in name:
            notes += " 文件中的历史 human 字段不表示用户逐条人工确认。"
        if "ocr" in name or "merged" in name:
            notes += " OCR 为历史模型读取补充，可能与候选模型家族存在相关偏差；合并层与原分层重复，不能相加计数。"
        rows.append([esc(REFERENCE_LABELS.get(name, name)), esc(refset.get("tier")), str(len(refset.get("units", []))),
                     esc(notes) + f'<details><summary>原始来源说明</summary><p>{esc(refset.get("provenance"))}</p></details>'])
    return "<h2>两类中文参考与画面参考分层</h2><p>画面字幕与下载字幕分开评分；Claude 读取层、OCR 补充层及其合并层也独立展示。分数仅比较输出与该条参考文本的文字接近程度。</p>" + table(["参考集合", "分层", "参考条数", "适用边界"], rows)


def render_metrics(metrics: dict, models: list[dict], freshness: dict) -> str:
    rows = metrics.get("results", [])
    if not rows:
        return '<h2>分层文本指标</h2><p class="notice">尚无 metrics.json 评分结果。不能据此比较模型质量。</p>'
    sections = ["<h2>分层文本指标</h2><p>CER 为字符编辑距离比，可能大于 1；chrF 为字符 n-gram 接近程度。中文同义表达、参考本身错误及时间对齐偏差都会影响指标。失败或缺失输出按既定缺失规则计入，不能只看有输出的句子。</p>"]
    refsets = sorted({row.get("reference_set", "unknown") for row in rows}, key=lambda v: ("screen_human" not in v, v != "fansub_zh", v))
    for refset in refsets:
        refrows = [row for row in rows if row.get("reference_set") == refset]
        initially_open = "screen_human" in refset or refset == "fansub_zh"
        sections.append(f'<details {"open" if initially_open else ""}><summary>{esc(REFERENCE_LABELS.get(refset, refset))} · {esc(refrows[0].get("reference_tier"))}</summary>')
        for normalization in sorted({row.get("normalization", "unspecified") for row in refrows}):
            selected = [row for row in refrows if row.get("normalization") == normalization]
            sections.append(f'<details {"open" if normalization.endswith("_nfkc") else ""}><summary>归一化：{esc(normalization)}</summary>')
            language = selected[0].get("language")
            stages = ("asr_polish",) if language == "ja" else STAGES[1:]
            for stage in stages:
                baseline = next((row for row in selected if row.get("model") == "baseline" and row.get("stage") == "raw_asr"), None) if language == "ja" else None
                columns = [("原始日文 ASR 基线", baseline)] if language == "ja" else []
                columns.extend((model["id"], next((row for row in selected if row.get("model") == model["id"] and row.get("stage") == stage), None)) for model in models)
                if not any(row is not None for _, row in columns):
                    sections.append(f'<h3>{esc(LABELS[stage])}</h3><p>此阶段尚无评分记录。</p>')
                    continue
                metric_rows = [["评分快照状态"] + ["无评分记录" if row is None else
                               ("当前输出" if freshness.get((row.get("model"), row.get("stage")), False) else "历史快照／指纹未确认") for _, row in columns]]
                for section, key, label in METRIC_LABELS:
                    metric_rows.append([esc(label)] + [format_number(row.get(section, {}).get(key)) if row else "—" for _, row in columns])
                sections.extend([f'<h3>{esc(LABELS[stage])}</h3>', table(["同一参考 / 同一归一化指标"] + [label for label, _ in columns], metric_rows)])
            sections.append("</details>")
        sections.append("</details>")
    sections.append(f'<details><summary>完整评分方法</summary><pre>{esc(json.dumps(metrics.get("method", {}), ensure_ascii=False, indent=2))}</pre></details>')
    return "".join(sections)


def render_aligned_details(details: dict, units: list[dict], models: list[dict], freshness: dict) -> str:
    source = {runner.id_key(unit["id"]): unit for unit in units if "id" in unit}
    rows = [row for row in details.get("results", []) if row.get("normalization", "").endswith("_nfkc")]
    if not rows:
        return "<h2>参考对齐组复核</h2><p>暂无 score_details.json 对齐明细；可以先检查上方日文原文与校正。</p>"
    sections = ["<h2>参考对齐组复核</h2><p>以下保留原始文本。每个参考集合独立按时间重叠形成固定多对多对齐组。未匹配参考、缺失输出和失败单位保留显示，不自动替换为其他模型输出。</p>"]
    for refset in sorted({row["reference_set"] for row in rows}):
        refrows = [row for row in rows if row["reference_set"] == refset]
        language = refrows[0].get("language")
        stages = ("asr_polish",) if language == "ja" else STAGES[1:]
        for stage in stages:
            relevant = [row for row in refrows if row.get("stage") == stage or (language == "ja" and row.get("stage") == "raw_asr")]
            if not relevant:
                continue
            by_model = {row["model"]: {group["group_id"]: group for group in row.get("groups", [])} for row in relevant}
            seeds = {}
            for row in relevant:
                for group in row.get("groups", []):
                    seeds.setdefault(group["group_id"], group)
            sections.append(f'<section class="review-section" data-stage="{stage}"><details><summary>{esc(REFERENCE_LABELS.get(refset, refset))} · {esc(LABELS[stage])} · {len(seeds)} 组</summary>')
            for group_id, seed in seeds.items():
                source_text = "\n".join(f'[{uid}] {source.get(runner.id_key(uid), {}).get("text", "[源单位缺失]")}' for uid in seed.get("unit_ids", []))
                sections.append(f'<details class="alignment"><summary>对齐组 {esc(group_id)} · {esc(seed.get("start"))}–{esc(seed.get("end"))} 秒 · 源 {len(seed.get("unit_ids", []))} / 参考 {len(seed.get("reference_ids", []))} 条</summary>')
                sections.append(f'<div class="source"><b>原始日文</b><p>{esc(source_text or "[没有时间重叠的源单位]")}</p></div>')
                sections.append(f'<div class="reference"><b>{esc(REFERENCE_LABELS.get(refset, refset))}</b><p>{esc(seed.get("reference_text"))}</p><span class="muted">参考 ID：{esc(", ".join(map(str, seed.get("reference_ids", []))))}</span></div>')
                for model in models:
                    result = by_model.get(model["id"], {}).get(group_id)
                    if result is None:
                        content = "此模型／阶段暂无该组评分快照。"
                    else:
                        missing = result.get("missing_output_unit_ids", [])
                        score = f'CER {format_number(result.get("cer"))} · 字符序列相似度 {format_number(result.get("sequence_similarity"))} · chrF {format_number(result.get("chrf"))}'
                        stale = "" if freshness.get((model["id"], stage), False) else " · 历史快照／指纹未确认"
                        content = f'<p>{esc(result.get("output_text") or "[无可用输出]")}</p><span class="muted">{score}{esc(stale)}<br>缺失／失败单位：{esc(", ".join(map(str, missing)) or "无")}；源匹配：{"无" if result.get("missing_source") else "有"}</span>'
                    sections.append(f'<div class="model-output" data-model="{esc(model["id"])}"><b>{esc(model["id"])}</b>{content}</div>')
                sections.append("</details>")
            sections.append("</details></section>")
    return "".join(sections)


STYLE = """
:root{color-scheme:light;--ink:#13252d;--muted:#4d6370;--line:#cbd8df;--accent:#006d77}
*{box-sizing:border-box}body{font-family:system-ui,-apple-system,'Noto Sans CJK SC',sans-serif;background:#f4f7f8;color:var(--ink);margin:0;line-height:1.65}
main{max-width:1600px;margin:auto;padding:32px}h1{font-size:30px;margin:0 0 8px}h2{margin-top:38px;font-size:23px;border-bottom:2px solid var(--accent);padding-bottom:8px}h3{font-size:18px}
.notice{background:#fff4cf;border-left:5px solid #a76500;padding:18px 22px;border-radius:5px}.muted{color:var(--muted);font-size:13px}.lead{max-width:1000px}.table-wrap{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:8px;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;vertical-align:top;padding:12px 14px;border-bottom:1px solid var(--line);min-width:150px}th{background:#e3eef1;position:sticky;top:0}td:first-child,th:first-child{min-width:200px}tr:last-child td{border-bottom:0}
code{overflow-wrap:anywhere;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:650px;font-size:12px;margin:5px 0}details{background:white;border:1px solid var(--line);border-radius:6px;padding:12px 16px;margin:12px 0}summary{cursor:pointer;font-weight:600}.alignment{border-left:4px solid #a5c4ca;margin:10px 0}.source,.reference,.model-output{padding:10px 12px;margin:10px 0;border-radius:4px;background:#f1f6f7}.reference{background:#fff8e6}.model-output{background:#f3f3fa}.source p,.reference p,.model-output p{white-space:pre-wrap;margin:5px 0;overflow-wrap:anywhere}
.filters{position:sticky;top:0;z-index:3;background:#e3eef1eF;padding:14px;border:1px solid var(--line);border-radius:6px;display:flex;flex-wrap:wrap;gap:20px;align-items:center}select{font:inherit;padding:4px 9px}.state{padding:3px 8px;border-radius:4px;background:#edf1f3}.complete{background:#d3eee1}.partial,.complete_with_failures{background:#fff1ba}.failed{background:#ffdad6}a{color:#005e73}.hidden{display:none!important}footer{margin-top:40px;padding-top:15px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
@media(max-width:700px){main{padding:15px}h1{font-size:24px}th,td{padding:8px}.filters{position:static}}
"""

SCRIPT = """
(()=>{const model=document.querySelector('#model-filter'),stage=document.querySelector('#stage-filter');
function update(){document.querySelectorAll('[data-model]').forEach(e=>e.classList.toggle('hidden',model.value!==''&&e.dataset.model!==model.value));
document.querySelectorAll('.review-section[data-stage]').forEach(e=>e.classList.toggle('hidden',stage.value!==''&&e.dataset.stage!==stage.value));}
model.addEventListener('change',update);stage.addEventListener('change',update);})();
"""


def make_filters(models: list[dict]) -> str:
    return '<div class="filters"><label>逐句模型 <select id="model-filter"><option value="">所有模型</option>' + \
        "".join(f'<option value="{esc(m["id"])}">{esc(m["id"])}</option>' for m in models) + \
        '</select></label><label>逐句阶段 <select id="stage-filter"><option value="">所有阶段</option>' + \
        "".join(f'<option value="{s}">{esc(LABELS[s])}</option>' for s in STAGES) + \
        '</select></label><span class="muted">筛选仅控制下方逐句复核；统计表保留全部模型以便比较。</span></div>'


def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def generate(dataset_path: Path, references_path: Path, run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    store = runner.PrivateStore(run_dir)
    errors = []
    dataset = read_json(dataset_path, errors, required=True)
    references = read_json(references_path, errors, required=True)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("units"), list):
        raise ValueError("dataset must contain a units array")
    if not isinstance(references, dict) or not isinstance(references.get("sets"), dict):
        raise ValueError("references must contain reference sets")
    units = dataset["units"]
    models = load_models(run_dir, errors, len(units))
    source_fingerprint = runner.digest(dataset)
    for model in models:
        recorded_source = model["manifest"].get("source_sha256")
        if recorded_source and recorded_source != source_fingerprint:
            errors.append(f"当前dataset与模型清单中的源数据指纹不同：{model['id']}；不得把本页当前原文当作该次请求的已确认输入。")
    metrics = read_json(run_dir / "metrics.json", errors) or {}
    details = read_json(run_dir / "score_details.json", errors) or {}
    analysis = read_json(run_dir / "analysis.json", errors)
    if not isinstance(metrics, dict) or not isinstance(details, dict):
        raise ValueError("metrics and score_details must be objects")
    snapshot = analysis_tools.verify_score_snapshot(metrics, details, run_dir, dataset_path, references_path, dataset, references)
    if not snapshot["valid"]:
        errors.append("评分共同快照未通过验证：" + ", ".join(snapshot["problems"]) + "；不标记当前评分，也不把明细与当前原文拼接展示。")
    freshness = metric_freshness(metrics, models, errors, snapshot["valid"])
    generated = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    if not models:
        state = "not_started"
    elif all(m["state"] == "not_started" for m in models):
        state = "not_started"
    elif all(m["state"] == "complete" for m in models):
        state = "complete"
    elif all(m["state"] in ("complete", "complete_with_failures") for m in models):
        state = "complete_with_failures"
    elif all(m["state"] == "failed" for m in models):
        state = "failed"
    else:
        state = "partial"
    if state in ("complete", "complete_with_failures") and any("EXPECTED_MODELS.json" in error for error in errors):
        state = "partial"
    observations = []
    for row in metrics.get("results", []):
        if row.get("reference_set") not in ("screen_human_ja", "screen_human_zh", "fansub_zh"):
            continue
        if not row.get("normalization", "").endswith("_nfkc"):
            continue
        key = (row.get("model"), row.get("stage"))
        observations.append({k: row.get(k) for k in ("model", "stage", "reference_set", "reference_tier", "normalization")}
                            | {"current_output_fingerprint_matches": freshness.get(key, False),
                               "reference_total": row.get("coverage", {}).get("reference_total"),
                               "reference_with_output": row.get("coverage", {}).get("reference_with_output"),
                               "reference_fully_covered": row.get("coverage", {}).get("reference_fully_covered"),
                               "cer_micro": row.get("metrics", {}).get("cer_micro"),
                               "chrf_corpus": row.get("metrics", {}).get("chrf_corpus")})
    conclusions = {
        "schema_version": 1, "generated_at": generated, "state": state,
        "scope": "User-selected existing subtitle text transformation stress test only. README.md#用途与处理范围",
        "copy_protocol_interpretation": analysis_tools.COPY_PROTOCOL,
        "host_preserve_protocol_interpretation": analysis_tools.HOST_PRESERVE_PROTOCOL,
        "source_units": len(units), "models_in_scope": len(models),
        "models_found": sum(bool(model["manifest"] or any(model["stages"].values())) for model in models),
        "expected_models_file_present": (run_dir / "EXPECTED_MODELS.json").is_file(),
        "interpretation": "Reference text agreement only; semantic/manual review pending; no automatic quality ranking or release conclusion.",
        "audio_verified": False, "multilingual_acceptance": False, "advertising_acceptance": False,
        "models": [{"id": m["id"], "state": m["state"], "target_units": m["target_units"],
                    "host_preserve_target_units": m["manifest"].get("host_preserve_target_units"),
                    "model_target_units": m["manifest"].get("model_target_units"),
                    "stage_counts": m["counts"],
                    "stage_operation_counts": {stage: analysis_tools.operation_counts(m["stages"][stage].get("units", []), stage) for stage in STAGES},
                    "unclosed_requests": len(m["summary"].get("interrupted_attempts", [])),
                    "digest": m["manifest"].get("model", {}).get("digest"),
                    "fingerprint": m["manifest"].get("fingerprint")} for m in models],
        "primary_text_agreement_observations": observations,
        "supplementary_metrics": "All OCR, merged-reference and alternate-normalization results remain separate in metrics.json.",
        "errors_or_snapshot_warnings": errors,
        "score_snapshot": snapshot,
        "artifacts": {"dataset_sha256": sha256_file(dataset_path), "references_sha256": sha256_file(references_path),
                      "metrics_sha256": sha256_file(run_dir / "metrics.json"),
                      "details_sha256": sha256_file(run_dir / "score_details.json"),
                      "report_code_sha256": sha256_file(Path(__file__))},
    }
    scope_url = (runner.ROOT / "README.md").as_uri() + "#用途与处理范围"
    report = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
              '<meta name="robots" content="noindex,nofollow"><title>字幕文本模型测试 · 人工复核入口</title>',
              f'<style>{STYLE}</style></head><body><main><h1>字幕文本模型测试 · 人工复核入口</h1>',
              f'<p class="muted">报告生成于 {esc(generated)} · 执行状态：<strong class="state {esc(state)}">{esc(STATUS[state])}</strong> · 源字幕 {len(units)} 个单位 · 比较范围 {len(models)} 个模型（已有运行记录 {conclusions["models_found"]} 个）</p>',
              f'<p class="lead">当前仅进行用户指定既有字幕的文本纠错、忠实翻译与润色压力测试；没有扩写新情节或生成媒体。本轮验证 JSON Schema 约束输出与固定 ID，不是函数调用／工具使用能力评测，也未做分段 TTS 合成试听。本报告不属于广告素材验收、多语言验收、法律审查或发布许可。生产项目完整用途与处理限制见 <a href="{esc(scope_url)}">README：用途与处理范围</a>。</p>',
              f'<p class="lead">{esc(analysis_tools.COPY_PROTOCOL)}</p>',
              f'<p class="lead">{esc(analysis_tools.HOST_PRESERVE_PROTOCOL)}</p>',
              f'<div class="notice"><strong>{esc(WARNING)}</strong><br>历史 Claude 读取、OCR 与下载字幕都不是听音认证真值。OCR 参考可能含有与候选模型家族相关的偏差。请结合具体原文和参考逐组复核，不依据单项字符分数自动判优。</div>',
              '<p class="muted">这是生成时的证据快照，页面不会自动更新；运行中与尚未生成的阶段均保留待处理状态。输出覆盖与文本接近程度分别列示。</p>']
    if errors:
        report.append('<details open class="notice"><summary>读取与快照核对提示</summary><ul>' + "".join(f"<li>{esc(error)}</li>" for error in errors) + "</ul></details>")
    report.extend([render_status(models, len(units)), render_operations(models), render_analysis(analysis, run_dir / "metrics.json", run_dir / "score_details.json", snapshot["valid"]), render_reference_notes(references),
                   render_metrics(metrics, models, freshness), render_requests(models), render_manifests(models),
                   make_filters(models), render_source_details(units, models),
                   render_aligned_details(details, units, models, freshness) if snapshot["valid"] else
                   '<h2>参考对齐组复核</h2><p class="notice">评分与明细尚未通过共同快照验证，暂停跨文件逐组拼接，避免混合新旧证据。原始 JSON 文件仍保留供追查。</p>'])
    report.append('<footer>机器可读观察：<a href="conclusions.json">conclusions.json</a> · 完整指标：<a href="metrics.json">metrics.json</a> · 对齐明细：<a href="score_details.json">score_details.json</a><br>源时间与说话人标签由主机保存；本页只展示现存文本证据，不包含图像、音频或视频。</footer>')
    report.append(f'<script>{SCRIPT}</script></main></body></html>')
    store.json(run_dir / "conclusions.json", conclusions)
    output_path = run_dir / "00_开始复核.html"
    store.write(output_path, "".join(report).encode("utf-8"))
    print(json.dumps({"report": str(output_path), "conclusions": str(run_dir / "conclusions.json"), "state": state, "models": len(models)}, ensure_ascii=False))
    return conclusions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        generate(args.dataset, args.references, args.run_dir)
    except (OSError, ValueError, RuntimeError, runner.Failure) as exc:
        print(f"REPORT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
