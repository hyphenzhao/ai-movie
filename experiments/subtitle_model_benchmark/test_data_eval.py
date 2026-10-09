"""Neutral tests; project scope: [README](../../README.md#用途与处理范围)."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_eval import build_groups, canonical_hash, normalize, place_units, publish_scores, score_arm, sha256, stage_language


def unit(uid, start, end, text):
    return {"id": uid, "start": start, "end": end, "text": text}


class DataEvaluationTests(unittest.TestCase):
    def test_split_many_to_many_counts_each_source_once(self):
        source = [unit("a", 0, 2, "春天"), unit("b", 2, 4, "到了")]
        refs = [unit("r1", 0, 1, "春"), unit("r2", 1, 3, "天到"), unit("r3", 3, 4, "了")]
        groups = build_groups(source, refs)
        self.assertEqual(len(groups), 1)
        result, details = score_arm(groups, source, "zh_nfkc")
        self.assertEqual(result["metrics"]["cer_micro"], 0)
        self.assertEqual(result["metrics"]["chrf_corpus"], 100)
        self.assertEqual(result["coverage"]["reference_total"], 3)
        self.assertEqual(details[0]["unit_ids"], ["a", "b"])

    def test_missing_reference_is_not_a_correct_or_dropped_group(self):
        source = [unit("a", 0, 1, "天")]
        refs = [unit("r1", 0, 1, "天"), unit("r2", 5, 6, "气")]
        result, details = score_arm(build_groups(source, refs), source, "zh_nfkc")
        self.assertEqual(result["coverage"]["reference_without_source"], 1)
        self.assertEqual(result["coverage"]["reference_without_output"], 1)
        self.assertEqual(result["metrics"]["cer_micro"], 0.5)
        self.assertEqual(result["metrics"]["sequence_similarity_macro"], 0.5)
        self.assertEqual(len(details), 2)

    def test_failed_output_cannot_get_credit_for_retained_text(self):
        source = [unit("a", 0, 1, "天"), unit("b", 1, 2, "气")]
        refs = [unit("r", 0, 2, "天气")]
        output = [{**source[0], "status": "failed"}, {**source[1], "status": "ok"}]
        result, _ = score_arm(build_groups(source, refs), output, "zh_nfkc")
        self.assertEqual(result["metrics"]["cer_micro"], 0.5)
        self.assertEqual(result["coverage"]["failed_units"], 1)
        self.assertEqual(result["coverage"]["reference_fully_covered"], 0)

    def test_missing_output_and_changed_times_keep_fixed_population(self):
        source = [unit("a", 0, 1, "天"), unit("b", 1, 2, "气")]
        refs = [unit("r1", 0, 1, "天"), unit("r2", 1, 2, "气")]
        output = [unit("a", 100, 101, "天")]
        result, _ = score_arm(build_groups(source, refs), output, "zh_nfkc")
        self.assertEqual(result["coverage"]["reference_total"], 2)
        self.assertEqual(result["coverage"]["reference_with_output"], 1)
        self.assertEqual(result["coverage"]["missing_units"], 1)

    def test_lead_trim_excludes_duplicate_and_retains_boundary_words(self):
        chunk = {"index": 2, "start": 10, "end": 20}
        source = [unit("unused", 0, 2, "重複"), unit("unused", 1, 4, "天気"),
                  {**unit("unused", 4, 5, "晴れ"), "keep_original": True}]
        active, excluded = place_units(source, chunk, 2)
        self.assertEqual(len(active), 2)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(active[0]["start"], 10)
        self.assertEqual(active[0]["end"], 12)
        self.assertTrue(active[0]["flags"]["time_clipped_to_chunk"])
        self.assertTrue(active[1]["flags"]["keep_original"])
        self.assertEqual(active[1]["id"], "p02_s0002")

    def test_normalization_and_stage_language_are_explicit(self):
        self.assertEqual(normalize("Ａ、 カナ！", "ja_nfkc"), "Aカナ")
        self.assertEqual(normalize("Ａ、 カナ！", "ja_nfkc_kanafold"), "Aかな")
        self.assertEqual(stage_language("asr_polish", {}), "ja")
        self.assertEqual(stage_language("polish_translation_raw", {}), "zh")

    def test_duplicate_output_ids_rejected(self):
        source = [unit("a", 0, 1, "天")]
        with self.assertRaises(ValueError):
            score_arm(build_groups(source, source), source * 2, "zh_nfkc")

    def test_snapshot_binds_exact_inputs_arms_and_details_written_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            (run / "neutral").mkdir(parents=True)
            source = unit("a", 0, 1, "晴れ")
            dataset, refs = root / "dataset.json", root / "references.json"
            dataset.write_text(json.dumps({"units": [source]}))
            refs.write_text(json.dumps({"sets": {"neutral_ja": {"language": "ja", "tier": "primary", "units": [source]}}}))
            stage = run / "neutral" / "asr_polish.json"
            stage.write_text(json.dumps({"units": [{**source, "status": "success"}]}))
            writes = []

            def local_write(path, data):
                writes.append(path.name)
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

            with patch("data_eval.write_json", local_write):
                metrics = publish_scores(dataset, refs, run)
            details = json.loads((run / "score_details.json").read_text())
            self.assertEqual(writes, ["score_details.json", "metrics.json"])
            self.assertEqual(metrics["snapshot"], details["snapshot"])
            self.assertEqual(metrics["details_sha256"], sha256(run / "score_details.json"))
            self.assertEqual(metrics["snapshot"]["dataset_sha256"], sha256(dataset))
            self.assertEqual(metrics["snapshot"]["arms"], [{"model": "neutral", "stage": "asr_polish", "sha256": sha256(stage)}])
            self.assertEqual(metrics["snapshot"]["id"], canonical_hash({k: v for k, v in metrics["snapshot"].items() if k != "id"}))

    def test_output_change_between_details_and_metrics_aborts_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            (run / "neutral").mkdir(parents=True)
            source = unit("a", 0, 1, "晴れ")
            dataset, refs = root / "dataset.json", root / "references.json"
            dataset.write_text(json.dumps({"units": [source]}))
            refs.write_text(json.dumps({"sets": {"neutral_ja": {"language": "ja", "tier": "primary", "units": [source]}}}))
            stage = run / "neutral" / "asr_polish.json"
            stage.write_text(json.dumps({"units": [source]}))

            def local_write(path, data):
                path.write_text(json.dumps(data))
                if path.name == "score_details.json":
                    stage.write_text(json.dumps({"units": [{**source, "text": "曇り"}]}))

            with patch("data_eval.write_json", local_write), self.assertRaises(RuntimeError):
                publish_scores(dataset, refs, run)
            self.assertFalse((run / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
