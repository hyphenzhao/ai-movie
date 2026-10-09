"""Neutral offline benchmark checks. Scope: ../../README.md#用途与处理范围.

Mocks contain ordinary supplied dialogue only and never call an inference API.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("subtitle_benchmark_runner", Path(__file__).with_name("runner.py"))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def envelope(items, **extra):
    return {"done": True, "done_reason": "stop", "prompt_eval_count": 200,
            "eval_count": 20, "message": {"role": "assistant", "content": json.dumps({"items": items}, ensure_ascii=False)}, **extra}


def replacement(uid, text):
    return {"id": uid, "op": "replace", "text": text}


def copied(uid):
    return {"id": uid, "op": "copy", "text": ""}


class ParserTests(unittest.TestCase):
    def assert_category(self, category, response, ids=("a", "b")):
        with self.assertRaises(r.Failure) as caught:
            r.parse_response(response, list(ids), 128)
        self.assertEqual(caught.exception.category, category)

    def test_valid_response_preserves_id_type_and_input_order_is_not_required(self):
        response = envelope([replacement(1, "Good morning."), replacement("1", "Hello.")])
        result = r.parse_response(response, ["1", 1], 128)
        self.assertEqual(result[r.id_key("1")], {"op": "replace", "text": "Hello."})
        self.assertEqual(result[r.id_key(1)], {"op": "replace", "text": "Good morning."})

    def test_duplicate_and_missing_and_unknown_ids(self):
        for items in (
            [replacement("a", "Hello."), replacement("a", "Hello.")],
            [replacement("a", "Hello.")],
            [replacement("a", "Hello."), replacement("z", "Hello.")],
            [replacement(True, "Hello.")],
            [copied("a"), copied("a")],
            [copied("a")],
        ):
            self.assert_category("id", envelope(items))

    def test_extra_fields_and_duplicate_json_keys_rejected(self):
        self.assert_category("schema", envelope([{**replacement("a", "Hello."), "note": "extra"}]))
        self.assert_category("schema", envelope([{"id": "a", "text": "Hello."}]))
        response = envelope([])
        response["message"]["content"] = '{"items": [], "items": []}'
        self.assert_category("schema", response)

    def test_invalid_json_empty_encoding_and_truncation(self):
        for content, category in (("```json\n{}\n```", "json_decode"), ("", "empty"), ("\ufffd", "encoding")):
            response = envelope([])
            response["message"]["content"] = content
            self.assert_category(category, response)
        self.assert_category("empty", envelope([replacement("a", " ")]))
        for extra in ({"done": False}, {"done_reason": "length"}, {"eval_count": 128}):
            self.assert_category("truncation", envelope([], **extra))

    def test_copy_requires_exact_empty_text_and_replace_requires_nonempty(self):
        for text in ("copied original", " ", "\n"):
            self.assert_category("schema", envelope([{"id": "a", "op": "copy", "text": text}]), ("a",))
        for text in ("", " ", "\n"):
            self.assert_category("empty", envelope([replacement("a", text)]), ("a",))
        for operation in ("keep", "COPY", None, True, 0):
            self.assert_category("schema", envelope([{"id": "a", "op": operation, "text": "Hello."}]), ("a",))

    def test_long_repeated_copy_is_byte_identical_and_carries_provenance(self):
        source = "  " + "X" * 4096 + "\n"
        parsed = r.parse_response(envelope([copied("a")]), ["a"], 128)
        restored = r.restore_operations(parsed, [{"id": "a", "source": source}], "asr_polish")[r.id_key("a")]
        self.assertEqual(restored["text"], source)
        self.assertEqual(restored["operation"], "copy")
        self.assertEqual(restored["restore_provenance"], {"kind": "host_copy", "source": "target.source",
                         "source_stage": "dataset", "source_id": "a", "text_sha256": hashlib.sha256(source.encode()).hexdigest()})
        self.assertEqual(parsed[r.id_key("a")]["text"], "")

    def test_each_stage_restores_the_correct_host_input(self):
        parsed = r.parse_response(envelope([copied("a")]), ["a"], 128)
        target = {"id": "a", "source": "Original source", "draft": "Existing draft"}
        expected = {"asr_polish": ("source", "dataset"), "translate_raw": ("source", "dataset"),
                    "translate_polished": ("source", "asr_polish"),
                    "polish_translation_raw": ("draft", "translate_raw"),
                    "polish_translation_polished": ("draft", "translate_polished")}
        for stage, (field, source_stage) in expected.items():
            with self.subTest(stage=stage):
                result = r.restore_operations(parsed, [target], stage)[r.id_key("a")]
                self.assertEqual(result["text"], target[field])
                self.assertEqual(result["restore_provenance"]["source"], "target." + field)
                self.assertEqual(result["restore_provenance"]["source_stage"], source_stage)
        with self.assertRaises(r.Failure) as caught:
            r.restore_operations(parsed, [{"id": "a", "source": "Original"}], "polish_translation_raw")
        self.assertEqual(caught.exception.category, "configuration")

    def test_replace_keeps_model_text_and_v3_fingerprint_rejects_old_protocols(self):
        parsed = r.parse_response(envelope([replacement("a", "Updated text")]), ["a"], 128)
        result = r.restore_operations(parsed, [{"id": "a", "source": "Original"}], "asr_polish")[r.id_key("a")]
        self.assertEqual(result["text"], "Updated text")
        self.assertEqual(result["restore_provenance"]["kind"], "model_replace")
        self.assertEqual(result["restore_provenance"]["source"], "model.text")
        args = ({"units": []}, {"digest": "same"}, {}, "same-code")
        current = r.fingerprint(*args)
        for version in (1, 2):
            with patch.object(r, "PROTOCOL_VERSION", version):
                older = r.fingerprint(*args)
            self.assertNotEqual(current, older)
        with patch.dict(r.HOST_PRESERVE_RULE, minimum_longest_identical_character_run=101):
            self.assertNotEqual(current, r.fingerprint(*args))
        self.assertEqual(r.PROTOCOL_VERSION, 3)
        schema = r.schema_for(["a"])["properties"]["items"]["items"]
        self.assertEqual(set(schema["required"]), {"id", "op", "text"})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["op"]["enum"], ["copy", "replace"])

    def test_host_preserve_rule_requires_flags_allowlist_and_long_identical_run(self):
        flags = {"content": "nonlexical", "keep_original": True, "no_lipsync": True}
        for text in ("あ" * 100, "う" * 100, "ー" * 100, "あ" * 100 + "ー", "う" * 100 + "ー"):
            with self.subTest(text_length=len(text)):
                self.assertTrue(r.host_preserve_matches({"text": text, "flags": flags}))
        for text in ("あ" * 99, "う" * 99, "あー" * 100, "あ" * 100 + "う", "う" * 100 + "あ",
                     "あ" * 100 + "ありがとう", "う" * 100 + "言葉", "[声]" + "あ" * 100,
                     "ア" * 100, "あ" * 100 + "☆", "　，。", ""):
            with self.subTest(text_length=len(text)):
                self.assertFalse(r.host_preserve_matches({"text": text, "flags": flags}))
        for wrong in (None, {}, {**flags, "content": "speech"}, {**flags, "keep_original": False},
                      {**flags, "no_lipsync": False}, {**flags, "keep_original": 1},
                      {**flags, "no_lipsync": "true"}):
            self.assertFalse(r.host_preserve_matches({"text": "あ" * 100, "flags": wrong}))

    def test_host_preserve_punctuation_and_whitespace_are_preserved_exactly(self):
        original = " \t「" + "あ、" * 100 + "！」\n"
        unit = {"id": 7, "start": 1.25, "end": 2.5, "speaker": "S2", "text": original,
                "flags": {"content": "nonlexical", "keep_original": True, "no_lipsync": True}}
        result = r.host_preserved_result(unit)
        self.assertEqual(result["text"].encode("utf-8"), original.encode("utf-8"))
        self.assertEqual(result["operation"], "host_preserve")
        self.assertEqual(result["attempt_ids"], [])
        self.assertEqual(result["restore_provenance"], {"kind": "host_preserve", "source": "dataset",
                         "source_stage": "dataset", "source_id": 7,
                         "text_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(), "reason": "rule_v1"})
        for field in ("id", "start", "end", "speaker", "flags"):
            self.assertEqual(result[field], unit[field])
        with self.assertRaises(r.Failure):
            r.host_preserved_result({**unit, "text": "普通の言葉"})

    def test_invalid_http_utf8_classified(self):
        with self.assertRaises(r.Failure) as caught:
            r.decode_api(b"\xff", 200)
        self.assertEqual(caught.exception.category, "encoding")

    def test_context_never_silently_truncated(self):
        payload = {"messages": [{"role": "user", "content": "x" * 1000}], "format": {}}
        with self.assertRaises(r.Failure) as caught:
            r.context_budget(payload, {"num_ctx": 1100, "num_predict": 100})
        self.assertEqual(caught.exception.category, "context_overflow")

    def test_resume_fingerprint_tracks_source_model_options_and_code(self):
        source, model, options = {"units": [{"id": 1, "text": "Hello."}]}, {"digest": "v1"}, {"seed": 7}
        fp = r.fingerprint(source, model, options, "code1")
        r.check_resume({"fingerprint": fp}, fp)
        variants = [
            r.fingerprint({"units": []}, model, options, "code1"),
            r.fingerprint(source, {"digest": "v2"}, options, "code1"),
            r.fingerprint(source, model, {"seed": 8}, "code1"),
            r.fingerprint(source, model, options, "code2"),
        ]
        for changed in variants:
            with self.assertRaises(RuntimeError):
                r.check_resume({"fingerprint": fp}, changed)

    def test_full_context_fixed_prefix_and_no_reference_or_host_metadata(self):
        units = [{"id": "a", "text": "こんにちは。", "speaker": "S1", "start": 0, "end": 1, "reference": "SECRET_REFERENCE"},
                 {"id": "b", "text": "ありがとうございます。", "speaker": "S2", "start": 1, "end": 2}]
        context = r.source_context("translate_raw", units, {})
        options = {"num_ctx": 8192, "num_predict": 128}
        p1 = r.build_payload({"model": "neutral"}, "translate_raw", context, [{"id": "a", "source": units[0]["text"]}], options)
        p2 = r.build_payload({"model": "neutral"}, "translate_raw", context, [{"id": "b", "source": units[1]["text"]}], options)
        self.assertEqual(p1["messages"][0], p2["messages"][0])
        encoded = r.canonical(p1)
        self.assertNotIn("SECRET_REFERENCE", encoded)
        self.assertNotIn('"speaker":', encoded)
        self.assertNotIn('"start":', encoded)
        self.assertIn(units[1]["text"], encoded)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.mount = Path(self.temp.name)
        self.guard_patch = patch.object(r, "require_mount", return_value=self.mount)
        self.guard_patch.start()
        self.addCleanup(self.guard_patch.stop)
        self.units = [{"id": "a", "start": 0, "end": 1, "speaker": "S1", "text": "こんにちは。"},
                      {"id": "b", "start": 1, "end": 2, "speaker": "S2", "text": "ありがとう。"}]
        self.source = {"units": self.units}
        self.model = {"id": "neutral", "model": "neutral:latest", "digest": "digest1"}
        self.options = {"base_url": "http://127.0.0.1:11434", "batch_size": 2, "num_ctx": 8192,
                        "num_predict": 128, "limit": None, "max_retries": 1, "bisect": False, "timeout": 5}

    def runner(self):
        return r.Runner(self.source, self.units, [self.model], self.mount / "AI-Movie" / "bench", self.options)

    def test_upstream_failure_is_skipped_and_independent_raw_branch_runs(self):
        instance = self.runner()
        calls = []

        def mock_http(base, path, payload, timeout):
            self.assertEqual(path, "/api/chat")
            content = payload["messages"][0]["content"]
            targets = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
            is_asr = "Correct Japanese ASR transcription only" in content
            calls.append(is_asr)
            response = envelope([replacement("a", "こんにちは。")]) if is_asr else envelope([replacement(u["id"], "你好。") for u in targets])
            return json.dumps(response, ensure_ascii=False).encode(), 200, {}

        with patch.object(instance, "verify_model", return_value={"digest": "digest1"}), patch.object(r, "raw_http", side_effect=mock_http):
            self.assertEqual(instance.run(), 2)
        polished = json.loads((instance.model_dir / "translate_polished.json").read_text())
        self.assertTrue(all(u["status"] == "skipped" and u["text"] is None for u in polished["units"]))
        raw = json.loads((instance.model_dir / "translate_raw.json").read_text())
        self.assertTrue(all(u["status"] == "success" for u in raw["units"]))
        self.assertEqual(raw["units"][0]["speaker"], "S1")
        summary = json.loads((instance.model_dir / "summary.json").read_text())
        self.assertEqual(summary["stages"]["asr_polish"]["first_attempt"]["failures"], 1)
        self.assertEqual(summary["stages"]["asr_polish"]["retry_or_bisection"]["failures"], 1)
        self.assertEqual(len(calls), 4)

        # Exact resume neither retries terminal failures nor regenerates success.
        resumed = self.runner()
        with patch.object(resumed, "verify_model", return_value={"digest": "digest1"}), patch.object(r, "raw_http", side_effect=AssertionError("unexpected inference")):
            self.assertEqual(resumed.run(), 2)
        changed = self.runner()
        changed.base_options["num_predict"] = 129
        with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
            changed.run()

    def test_bisection_retains_failed_parent_attempts(self):
        instance = self.runner()
        instance.options["max_retries"] = 0
        instance.options["bisect"] = True
        instance.model_dir = instance.out / self.model["id"]
        instance.run_fingerprint = "test"
        context = r.source_context("translate_raw", self.units, {})

        def mock_http(base, path, payload, timeout):
            targets = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
            response = envelope([]) if len(targets) > 1 else envelope([replacement(targets[0]["id"], "你好。")])
            return json.dumps(response).encode(), 200, {}

        with patch.object(r, "raw_http", side_effect=mock_http):
            result = instance.batch(self.model, "translate_raw", "00000", [{"id": u["id"], "source": u["text"]} for u in self.units], context)
        self.assertTrue(all(u["status"] == "success" and u["parent_attempt_ids"] for u in result.values()))
        summary = instance.summarize({"translate_raw": result})
        self.assertEqual(summary["stages"]["translate_raw"]["first_attempt"]["failures"], 1)
        self.assertEqual(summary["stages"]["translate_raw"]["retry_or_bisection"]["requests"], 2)

    def test_complete_chain_copy_restore_counts_and_downstream_full_text(self):
        self.units[1]["text"] = "X" * 400
        instance = self.runner()
        seen = []

        def mock_http(base, path, payload, timeout):
            context = payload["messages"][0]["content"]
            targets = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
            if "Correct Japanese ASR transcription only" in context:
                items = [copied(target["id"]) for target in targets]
                label = "asr"
            elif any("draft" in target for target in targets):
                self.assertEqual(targets[0]["draft"], "你好。")
                self.assertEqual(targets[1]["draft"], "X" * 400)
                items = [copied(target["id"]) for target in targets]
                label = "polish"
            else:
                self.assertEqual(targets[0]["source"], self.units[0]["text"])
                self.assertEqual(targets[1]["source"], "X" * 400)
                items = [replacement("a", "你好。"), copied("b")]
                label = "translate"
            self.assertTrue(all("op" not in target and "operation" not in target for target in targets))
            seen.append(label)
            return json.dumps(envelope(items), ensure_ascii=False).encode(), 200, {}

        with patch.object(instance, "verify_model", return_value={"digest": "digest1"}), patch.object(r, "raw_http", side_effect=mock_http):
            self.assertEqual(instance.run(), 0)
        self.assertEqual(seen, ["asr", "translate", "translate", "polish", "polish"])
        for stage in r.STAGES:
            data = json.loads((instance.model_dir / f"{stage}.json").read_text())
            self.assertEqual(data["protocol_version"], 3)
            self.assertEqual(data["units"][1]["text"], "X" * 400)
            self.assertEqual(data["units"][1]["operation"], "copy")
            expected = {"copy": 1, "replace": 1} if stage in ("translate_raw", "translate_polished") else {"copy": 2, "replace": 0}
            expected["host_preserve"] = 0
            self.assertEqual(data["operation_counts"], expected)
        summary = json.loads((instance.model_dir / "summary.json").read_text())
        self.assertEqual(summary["stages"]["asr_polish"]["operations"], {"copy": 2, "replace": 0, "host_preserve": 0})
        attempts = [json.loads(line) for line in (instance.model_dir / "attempts.jsonl").read_text().splitlines()]
        first = next(row for row in attempts if row["event"] == "finished")
        self.assertEqual(first["operation_counts"], {"copy": 2, "replace": 0})
        raw_response = json.loads((Path(first["request_path"]).parent / "response.json").read_text())
        raw_items = json.loads(raw_response["message"]["content"])["items"]
        self.assertTrue(all(item["text"] == "" for item in raw_items))
        resumed = self.runner()
        with patch.object(resumed, "verify_model", return_value={"digest": "digest1"}), patch.object(r, "raw_http", side_effect=AssertionError("unexpected resumed inference")):
            self.assertEqual(resumed.run(), 0)

    def test_host_preserve_runs_all_stages_without_model_requests_and_resumes(self):
        flags = {"content": "nonlexical", "keep_original": True, "no_lipsync": True}
        for index, unit in enumerate(self.units):
            unit.update(text=" 「" + ("あ" if index == 0 else "う") * 101 + "！」\n", flags=dict(flags))
        instance = self.runner()
        with patch.object(instance, "verify_model", return_value={"digest": "digest1"}), patch.object(r, "raw_http", side_effect=AssertionError("host preservation must not call model")):
            self.assertEqual(instance.run(), 0)
        self.assertFalse((instance.model_dir / "attempts.jsonl").exists())
        self.assertFalse((instance.model_dir / "requests").exists())
        manifest = json.loads((instance.model_dir / "manifest.json").read_text())
        self.assertEqual(manifest["host_preserve_target_units"], 2)
        self.assertEqual(manifest["model_target_units"], 0)
        self.assertFalse(manifest["host_preserve_rule"]["audio_verified"])
        summary = json.loads((instance.model_dir / "summary.json").read_text())
        for stage in r.STAGES:
            artifact = json.loads((instance.model_dir / f"{stage}.json").read_text())
            self.assertEqual(artifact["operation_counts"], {"copy": 0, "replace": 0, "host_preserve": 2})
            self.assertEqual(artifact["full_context_units"], 2)
            for unit, source in zip(artifact["units"], self.units):
                self.assertEqual(unit["text"], source["text"])
                self.assertEqual(unit["status"], "success")
                self.assertEqual(unit["operation"], "host_preserve")
                self.assertEqual(unit["restore_provenance"]["source_stage"], "dataset")
            self.assertEqual(summary["stages"][stage]["operations"], artifact["operation_counts"])
            self.assertEqual(summary["stages"][stage]["all_requests"]["requests"], 0)
        resumed = self.runner()
        with patch.object(resumed, "verify_model", return_value={"digest": "digest1"}), patch.object(r, "raw_http", side_effect=AssertionError("unexpected resumed inference")):
            self.assertEqual(resumed.run(), 0)

    def test_mixed_host_and_model_route_retains_full_context_and_model_only_attempts(self):
        self.units[1].update(text="あ" * 101,
                             flags={"content": "nonlexical", "keep_original": True, "no_lipsync": True})
        instance = self.runner()
        seen = []

        def mock_http(base, path, payload, timeout):
            self.assertIn(self.units[1]["text"], payload["messages"][0]["content"])
            targets = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
            self.assertEqual([target["id"] for target in targets], ["a"])
            seen.append(targets)
            return json.dumps(envelope([copied("a")])).encode(), 200, {}

        with patch.object(instance, "verify_model", return_value={"digest": "digest1"}), patch.object(r, "raw_http", side_effect=mock_http):
            self.assertEqual(instance.run(), 0)
        self.assertEqual(len(seen), 5)
        events = [json.loads(line) for line in (instance.model_dir / "attempts.jsonl").read_text().splitlines()]
        for event in events:
            self.assertEqual(event["ids"], ["a"])
            if event["event"] == "finished":
                self.assertEqual(event["operation_counts"], {"copy": 1, "replace": 0})
        for stage in r.STAGES:
            artifact = json.loads((instance.model_dir / f"{stage}.json").read_text())
            self.assertEqual(artifact["operation_counts"], {"copy": 1, "replace": 0, "host_preserve": 1})

    def test_illegal_copy_is_failure_without_source_fallback(self):
        instance = self.runner()
        instance.options["max_retries"] = 0
        instance.model_dir = instance.out / self.model["id"]
        instance.run_fingerprint = "test"
        context = r.source_context("asr_polish", self.units, {})
        response = envelope([{"id": "a", "op": "copy", "text": "not empty"}])
        with patch.object(r, "raw_http", return_value=(json.dumps(response).encode(), 200, {})):
            result = instance.batch(self.model, "asr_polish", "00000", [{"id": "a", "source": "Original"}], context)
        failed = result[r.id_key("a")]
        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(failed["text"])
        self.assertNotIn("operation", failed)
        self.assertEqual(failed["category"], "schema")

    def test_mount_failure_prevents_output_creation(self):
        out = self.mount / "AI-Movie" / "absent"
        with patch.object(r, "require_mount", side_effect=RuntimeError("not mounted")):
            with self.assertRaises(RuntimeError):
                r.PrivateStore(out)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
