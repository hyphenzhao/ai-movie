#!/usr/bin/env python3
"""Prepare existing text and score isolated subtitle experiments, without media reads.

Project scope: [README full scope](../../README.md#用途与处理范围).
This isolated historical-text stress test is not a release or a media pipeline.
Reference similarity is neither audio-certified accuracy nor an advertising permit.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
NAS = Path("/home/ai-movie/mnt/PrivateFiles")
JA_STAGES = {"raw_asr", "asr_polish", "correct", "correction", "corrected", "transcribe"}
ZH_STAGES = {"translate", "translation", "polish", "polished", "translate_raw",
             "translate_polished", "polish_translation_raw", "polish_translation_polished"}
OK_STATUSES = {None, "ok", "success", "completed", "unchanged", "changed", "preserved"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def read_recorded_json(path: Path) -> tuple[dict, dict]:
    """Bind the parsed object to the exact bytes read, even during atomic replacement."""
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def require_nas(path: Path) -> None:
    """Fail closed before writing experiment artifacts if the expected mount is absent."""
    resolved = path.resolve()
    if not resolved.is_relative_to(NAS):
        raise ValueError(f"Experiment output must be under {NAS}")
    proc = subprocess.run(["findmnt", "-T", str(NAS), "-J", "-o", "TARGET,SOURCE,FSTYPE"],
                          check=True, capture_output=True, text=True)
    mounts = json.loads(proc.stdout).get("filesystems", [])
    if not any(m.get("target") == str(NAS) and m.get("fstype") == "fuse.rclone"
               and m.get("source") == "ai-movie-lan:PrivateFiles" for m in mounts):
        raise RuntimeError("Expected NAS mount is unavailable; refusing local fallback writes")


def write_json(path: Path, value) -> None:
    require_nas(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    require_nas(path)
    temp.replace(path)


def place_units(segments: list[dict], chunk: dict, lead: float) -> tuple[list, list]:
    """Keep all textual rows in active units or auditable trimmed-row exclusions."""
    active, excluded = [], []
    index = int(chunk["index"])
    for n, seg in enumerate(segments):
        text = seg.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        local_start, local_end = float(seg["start"]), float(seg["end"])
        original_start = float(chunk["start"]) - lead + local_start
        original_end = float(chunk["start"]) - lead + local_end
        start = max(float(chunk["start"]), original_start)
        end = min(float(chunk["end"]), original_end)
        flags = {k: seg[k] for k in ("keep_original", "no_lipsync", "content", "content_reasons",
                                     "overlap", "asr_conf", "no_speech_prob", "avg_logprob",
                                     "compression_ratio", "source", "pass", "sweep_confirmed") if k in seg}
        unit = {"id": f"p{index:02d}_s{n:04d}", "start": round(start, 6),
                "end": round(end, 6), "text": text,
                "speaker": f"p{index:02d}:{seg.get('speaker') or 'unknown'}", "chunk": index,
                "source_index": n, "source_start": local_start, "source_end": local_end,
                "untrimmed_global_start": round(original_start, 6),
                "untrimmed_global_end": round(original_end, 6), "lead_seconds": lead,
                "flags": flags}
        for k in ("gender", "tts_gender", "speaker_conf"):
            if k in seg:
                unit[k] = seg[k]
        if end <= start:
            unit.update(evaluate=False, exclusion_reason="outside_chunk_after_lead_trim")
            excluded.append(unit)
        else:
            unit["evaluate"] = True
            if start != original_start or end != original_end:
                flags["time_clipped_to_chunk"] = True
            active.append(unit)
    return active, excluded


def prepare(root: Path, film: str, min_conf: float = 0.60) -> tuple[dict, dict]:
    split = root / "workspace" / film / "_split"
    screen = root / "workspace" / film / "_screen"
    plan_path, manifest_path = split / "plan.json", split / "concat_manifest.json"
    plan, manifest = read_json(plan_path), read_json(manifest_path)
    if manifest.get("plan", {}).get("sha256") != sha256(plan_path):
        raise ValueError("Concat manifest does not authenticate current split plan; review timeline first")
    leads = {part["name"]: float(part["lead_seconds"]) for part in manifest["parts"]}
    source_records, units, excluded = [record(plan_path), record(manifest_path)], [], []
    chunks = []
    for chunk in plan["chunks"]:
        name = f"{film}_p{int(chunk['index']):02d}"
        if name not in leads:
            raise ValueError(f"No recorded trim lead for {name}")
        path = root / "workspace" / name / "state.json"
        state = read_json(path)
        keep, trim = place_units(state.get("asr", {}).get("segments", []), chunk, leads[name])
        units.extend(keep)
        excluded.extend(trim)
        source_records.append(record(path))
        chunks.append({"index": chunk["index"], "start": chunk["start"], "end": chunk["end"],
                       "lead_seconds": leads[name], "active_rows": len(keep), "excluded_rows": len(trim)})
    units.sort(key=lambda u: (u["start"], u["end"], u["id"]))
    dataset = {"schema_version": 1, "film": film, "language": "ja", "units": units,
               "excluded_units": excluded, "metadata": {
                   "purpose": "Existing subtitle text transformation stress test; no media processing",
                   "scope": "../../README.md#用途与处理范围", "references_in_prompt": False,
                   "timeline": "global = chunk.start - recorded lead + local time; clip to chunk bounds",
                   "lead_source": "concat_manifest.json authenticated against plan SHA256; no media read",
                   "keep_original_policy": "Retained as metadata; not assumed to prove absence of words",
                   "original_textual_rows": len(units) + len(excluded), "active_rows": len(units),
                   "excluded_rows": len(excluded), "active_characters": sum(len(u["text"]) for u in units),
                   "active_keep_original_rows": sum(bool(u["flags"].get("keep_original")) for u in units),
                   "chunks": chunks, "sources": source_records}}
    scan_path, human_path = screen / "scan.json", screen / "screen.json"
    ocr_path, ocr_meta_path = screen / "screen_ocr.json", screen / "screen_ocr.meta.json"
    fansub_path = split / "srt_cues.json"
    scan = {str(c["id"]): c for c in read_json(scan_path)}
    human, ocr, ocr_meta = read_json(human_path), read_json(ocr_path), read_json(ocr_meta_path)
    reference_sources = [record(p) for p in (scan_path, human_path, ocr_path, ocr_meta_path, fansub_path)]
    sets = {}
    for lang, column in (("zh", 0), ("ja", 1)):
        human_units, ocr_units = [], []
        for source_name, values, target in (("screen_human", human, human_units),
                                             ("screen_ocr_only", ocr, ocr_units)):
            for sid, pair in values.items():
                if source_name == "screen_ocr_only" and sid in human:
                    continue  # Historical primary entry wins, even when one language is empty.
                text = (pair + ["", ""])[column] or ""
                if not text.strip() or sid not in scan:
                    continue
                confidence = None
                if source_name == "screen_ocr_only":
                    confidence = (ocr_meta.get("rows", {}).get(sid, {}).get(lang) or {}).get("score")
                    if confidence is None or float(confidence) < min_conf:
                        continue
                cue = scan[sid]
                target.append({"id": f"cue_{sid}", "cue_id": sid, "start": cue["start"],
                               "end": cue["end"], "text": text, "source": source_name,
                               "confidence": confidence})
        for name, rows, tier in (("screen_human", human_units, "primary"),
                                 ("screen_ocr_only", ocr_units, "supplementary"),
                                 ("screen_merged", human_units + ocr_units, "supplementary")):
            sets[f"{name}_{lang}"] = {"language": lang, "tier": tier,
                 "provenance": ("Historical Claude-assisted reading of subtitle tiles; legacy 'human' "
                                "label does not imply independently verified human or audio ground truth"
                                if name == "screen_human" else
                                "Existing OCR/model subtitle-tile text; primary entries override OCR"
                                if name == "screen_merged" else
                                "Existing OCR/model subtitle-tile text excluding every primary ID"),
                 "audio_verified": False, "units": sorted(rows, key=lambda c: (c["start"], c["id"]))}
    sets["fansub_zh"] = {"language": "zh", "tier": "external_comparison",
        "provenance": "Preserved downloaded subtitle cue JSON; original full SRT file not located",
        "audio_verified": False, "units": [{"id": f"cue_{c['id']}", "cue_id": str(c["id"]),
            "start": c["start"], "end": c["end"], "text": c["zh"], "source": "fansub"}
            for c in read_json(fansub_path) if (c.get("zh") or "").strip()]}
    refs = {"schema_version": 1, "film": film, "sets": sets, "metadata": {
        "scope": "../../README.md#用途与处理范围", "prompt_use_prohibited": True,
        "caveat": "Text reference agreement only; none of these sources is audio-certified ground truth",
        "ocr_min_confidence": min_conf, "ocr_engine": ocr_meta.get("engine"),
        "sources": reference_sources}}
    return dataset, refs


def normalize(text: str, variant: str, converter=None) -> str:
    text = unicodedata.normalize("NFKC", text)
    if variant.endswith("_kanafold"):
        text = "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in text)
    if variant.endswith("_opencc_t2s"):
        if converter is None:
            raise ValueError("OpenCC normalization requested without converter")
        text = converter.convert(text)
    return "".join(c for c in text if not c.isspace() and unicodedata.category(c)[0] not in "PZC")


def distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def chrf_stats(ref: str, hyp: str, order: int = 6) -> list[list[int]]:
    result = []
    for n in range(1, order + 1):
        rn = Counter(ref[i:i + n] for i in range(max(0, len(ref) - n + 1)))
        hn = Counter(hyp[i:i + n] for i in range(max(0, len(hyp) - n + 1)))
        # Match SacreBLEU: an order absent from this reference does not add
        # hypothesis n-grams to corpus precision. Missing hypotheses still
        # retain all reference n-grams and therefore lose recall.
        result.append([sum((rn & hn).values()), sum(hn.values()) if rn else 0, sum(rn.values())])
    return result


def chrf_value(stats: list[list[int]], beta: float = 2.0) -> float:
    # Effective character n-gram order, no word n-grams; score in [0, 100].
    valid = [s for s in stats if s[1] > 0 and s[2] > 0]
    if not valid:
        return 0.0
    precision = sum(s[0] / s[1] for s in valid) / len(valid)
    recall = sum(s[0] / s[2] for s in valid) / len(valid)
    return 100 * (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall) if precision + recall else 0.0


def build_groups(units: list[dict], references: list[dict], min_overlap: float = 0.10) -> list[dict]:
    """Fixed-time bipartite connected components: count each ASR/ref row only once.

    Isolated reference rows form missing-source groups and are included in scores.
    Output times can never move a candidate into or out of this fixed population.
    """
    us = [u for u in units if u.get("evaluate", True)]
    refs = list(references)
    n = len(us)
    parent = list(range(n + len(refs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    order = sorted(range(n), key=lambda i: us[i]["start"])
    starts = [us[i]["start"] for i in order]
    edges = [[] for _ in refs]
    for r, cue in enumerate(refs):
        for pos in range(bisect.bisect_left(starts, float(cue["end"]))):
            i = order[pos]
            overlap = min(float(us[i]["end"]), float(cue["end"])) - max(float(us[i]["start"]), float(cue["start"]))
            if overlap > 0 and overlap >= min_overlap:
                union(i, n + r)
                edges[r].append(us[i]["id"])
    parts = {}
    for r, cue in enumerate(refs):
        group = parts.setdefault(find(n + r), {"references": [], "units": [], "reference_unit_ids": {}})
        group["references"].append(cue)
        group["reference_unit_ids"][cue["id"]] = edges[r]
    for i, unit in enumerate(us):
        if find(i) in parts:
            parts[find(i)]["units"].append(unit)
    groups = sorted(parts.values(), key=lambda g: min(r["start"] for r in g["references"]))
    for i, group in enumerate(groups):
        group["id"] = f"g{i:04d}"
        group["references"].sort(key=lambda c: (c["start"], c["end"], c["id"]))
        group["units"].sort(key=lambda u: (u["start"], u["end"], u["id"]))
    return groups


def score_arm(groups: list[dict], outputs: list[dict], variant: str, converter=None) -> tuple[dict, list]:
    by_id = {}
    for row in outputs:
        uid = row.get("id")
        if uid in by_id:
            raise ValueError(f"Duplicate output ID: {uid}")
        by_id[uid] = row

    def valid(row):
        return bool(row and row.get("status") in OK_STATUSES and isinstance(row.get("text"), str)
                    and normalize(row["text"], variant, converter))

    edit_sum = ref_chars = 0
    sims, details = [], []
    totals = [[0, 0, 0] for _ in range(6)]
    coverage = Counter(reference_total=0, reference_with_source=0, reference_with_output=0,
                       reference_fully_covered=0, groups_total=len(groups), groups_with_output=0,
                       source_units=0, successful_units=0, missing_units=0, failed_units=0,
                       empty_output_units=0, output_only_unknown_units=0)
    population_ids = {u["id"] for g in groups for u in g["units"]}
    coverage["output_units_outside_reference_population"] = len(set(by_id) - population_ids)
    for group in groups:
        matched = [u["id"] for u in group["units"]]
        successful = [uid for uid in matched if valid(by_id.get(uid))]
        success_set = set(successful)
        ref = normalize("".join(c["text"] for c in group["references"]), variant, converter)
        hyp = normalize("".join(by_id[uid]["text"] for uid in successful), variant, converter)
        if not ref:
            continue
        coverage["reference_total"] += len(group["references"])
        for cue in group["references"]:
            linked = group["reference_unit_ids"][cue["id"]]
            coverage["reference_with_source"] += bool(linked)
            coverage["reference_with_output"] += bool(success_set.intersection(linked))
            coverage["reference_fully_covered"] += bool(linked) and all(uid in success_set for uid in linked)
        coverage["source_units"] += len(matched)
        coverage["successful_units"] += len(successful)
        coverage["groups_with_output"] += bool(successful)
        for uid in matched:
            row = by_id.get(uid)
            if row is None:
                coverage["missing_units"] += 1
            elif row.get("status") not in OK_STATUSES:
                coverage["failed_units"] += 1
            elif uid not in success_set:
                coverage["empty_output_units"] += 1
        edits = distance(ref, hyp)
        similarity = difflib.SequenceMatcher(None, ref, hyp, autojunk=False).ratio()
        stats = chrf_stats(ref, hyp)
        for dst, src in zip(totals, stats):
            for j in range(3):
                dst[j] += src[j]
        edit_sum += edits
        ref_chars += len(ref)
        sims.append(similarity)
        details.append({"group_id": group["id"], "reference_ids": [c["id"] for c in group["references"]],
                        "unit_ids": matched, "successful_unit_ids": successful,
                        "start": min(c["start"] for c in group["references"]),
                        "end": max(c["end"] for c in group["references"]),
                        "reference_text": "".join(c["text"] for c in group["references"]),
                        "output_text": "".join(by_id[uid]["text"] for uid in successful),
                        "reference_characters": len(ref), "edit_distance": edits,
                        "cer": edits / len(ref), "sequence_similarity": similarity,
                        "chrf": chrf_value(stats), "missing_source": not matched,
                        "missing_output_unit_ids": [uid for uid in matched if uid not in success_set]})
    total = coverage["reference_total"]
    coverage["reference_without_source"] = total - coverage["reference_with_source"]
    coverage["reference_without_output"] = total - coverage["reference_with_output"]
    coverage["reference_output_fraction"] = coverage["reference_with_output"] / total if total else None
    return {"coverage": dict(coverage), "metrics": {
        "reference_characters": ref_chars, "edit_distance_sum": edit_sum,
        "cer_micro": edit_sum / ref_chars if ref_chars else None,
        "sequence_similarity_macro": sum(sims) / len(sims) if sims else None,
        "chrf_corpus": chrf_value(totals) if ref_chars else None,
        "groups_scored": len(sims)}}, details


def stage_language(stage: str, artifact: dict) -> str | None:
    if stage in JA_STAGES:
        return "ja"
    if stage in ZH_STAGES:
        return "zh"
    return artifact.get("language") or artifact.get("metadata", {}).get("language")


def score_run(dataset: dict, references: dict, run_dir: Path, min_overlap: float = 0.10) -> tuple[dict, dict]:
    try:
        from opencc import OpenCC
        converter = OpenCC("t2s")
    except (ImportError, OSError, RuntimeError):
        converter = None
    arms = [{"model": "baseline", "stage": "raw_asr", "language": "ja", "units": dataset["units"]}]
    skipped = []
    expected_ids = {u["id"] for u in dataset["units"]}
    for path in sorted(run_dir.glob("*/*.json")):
        artifact, artifact_record = read_recorded_json(path)
        if not isinstance(artifact, dict) or "units" not in artifact:
            continue
        language = stage_language(path.stem, artifact)
        if language not in {"ja", "zh"}:
            skipped.append(str(path))
            continue
        extra = {u.get("id") for u in artifact["units"]} - expected_ids
        if extra:
            raise ValueError(f"Unknown output IDs in {path.name}: {len(extra)}")
        arms.append({"model": path.parent.name, "stage": path.stem, "language": language,
                     "units": artifact["units"], "artifact": artifact_record})
    rows, detail_rows = [], []
    for name, refset in references["sets"].items():
        groups = build_groups(dataset["units"], refset["units"], min_overlap)
        variants = [refset["language"] + "_nfkc"]
        if refset["language"] == "ja":
            variants.append("ja_nfkc_kanafold")
        elif converter:
            variants.append("zh_nfkc_opencc_t2s")
        for arm in arms:
            if arm["language"] != refset["language"]:
                continue
            for variant in variants:
                result, details = score_arm(groups, arm["units"], variant, converter)
                identity = {"model": arm["model"], "stage": arm["stage"], "language": arm["language"],
                            "reference_set": name, "reference_tier": refset["tier"], "normalization": variant}
                rows.append({**identity, **result})
                detail_rows.append({**identity, "groups": details})
    metrics = {"schema_version": 1, "film": dataset.get("film"), "results": rows,
        "method": {"alignment": "Fixed source/reference temporal overlap connected components (many-to-many)",
            "minimum_overlap_seconds": min_overlap, "same_population_all_arms": True,
            "missing_reference_policy": "No-overlap reference groups retained; empty output incurs deletions",
            "failed_output_policy": "Explicit non-success status, empty or missing output incurs coverage loss",
            "normalization": "NFKC; remove whitespace and Unicode punctuation/separator/control categories",
            "chrf": "Character orders 1..6, beta=2, effective order, corpus statistics, 0..100",
            "opencc_t2s_available": converter is not None,
            "scorer_code_sha256": sha256(Path(__file__)),
            "caveat": "Reference agreement is not audio-certified correctness; no legal or release conclusion"},
        "arms": [{k: v for k, v in arm.items() if k != "units"} for arm in arms],
        "skipped_unknown_stage_files": skipped, "details_file": "score_details.json"}
    return metrics, {"schema_version": 1, "results": detail_rows}


def current_arm_records(run_dir: Path) -> list[dict]:
    records = []
    for path in sorted(run_dir.glob("*/*.json")):
        artifact, rec = read_recorded_json(path)
        if isinstance(artifact, dict) and "units" in artifact and stage_language(path.stem, artifact) in {"ja", "zh"}:
            records.append({"model": path.parent.name, "stage": path.stem, "sha256": rec["sha256"]})
    return sorted(records, key=lambda r: (r["model"], r["stage"]))


def publish_scores(dataset_path: Path, references_path: Path, run_dir: Path, min_overlap: float = 0.10) -> dict:
    """Commit details first and metrics last; both bind the same complete input snapshot."""
    dataset, dataset_record = read_recorded_json(dataset_path)
    references, references_record = read_recorded_json(references_path)
    metrics, details = score_run(dataset, references, run_dir, min_overlap)
    snapshot = {"dataset_sha256": dataset_record["sha256"],
                "references_sha256": references_record["sha256"],
                "arms": sorted([{"model": arm["model"], "stage": arm["stage"],
                                  "sha256": arm["artifact"]["sha256"]}
                                 for arm in metrics["arms"] if "artifact" in arm],
                                key=lambda arm: (arm["model"], arm["stage"])),
                "method_sha256": canonical_hash(metrics["method"])}
    snapshot["id"] = canonical_hash(snapshot)
    metrics["snapshot"] = details["snapshot"] = snapshot
    metrics["input_sources"] = {"dataset": dataset_record, "references": references_record}

    def unchanged():
        if sha256(dataset_path) != snapshot["dataset_sha256"] \
                or sha256(references_path) != snapshot["references_sha256"] \
                or current_arm_records(run_dir) != snapshot["arms"]:
            raise RuntimeError("Scoring inputs or output arms changed before snapshot commit; score again")

    unchanged()
    details_path = run_dir / "score_details.json"
    write_json(details_path, details)
    metrics["details_sha256"] = sha256(details_path)
    unchanged()
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--root", type=Path, default=ROOT)
    prep.add_argument("--film", default="SONE-846")
    prep.add_argument("--ocr-min-confidence", type=float, default=0.60)
    score = commands.add_parser("score")
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--references", type=Path, required=True)
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--min-overlap", type=float, default=0.10)
    args = parser.parse_args()
    if args.command == "prepare":
        require_nas(args.out)
        dataset, refs = prepare(args.root, args.film, args.ocr_min_confidence)
        write_json(args.out / "dataset.json", dataset)
        write_json(args.out / "references.json", refs)
        print(json.dumps({"dataset": str(args.out / "dataset.json"), "active_rows": len(dataset["units"]),
            "excluded_rows": len(dataset["excluded_units"]),
            "reference_counts": {k: len(v["units"]) for k, v in refs["sets"].items()}}, ensure_ascii=False))
    else:
        require_nas(args.run_dir)
        metrics = publish_scores(args.dataset, args.references, args.run_dir, args.min_overlap)
        print(json.dumps({"metrics": str(args.run_dir / "metrics.json"), "result_rows": len(metrics["results"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
