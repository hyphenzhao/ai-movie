"""V7 offline evidence provenance and independent before/after membership; no model loads."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.vc_drift_calib import load_rows, reading
from scripts.vc_drift_probe import _verify_rows, probe_state
from scripts import verify_dub


class EvidenceTests(unittest.TestCase):
    def write_fixture(self, work, *, converted, start=1.0, want="今天天气很好"):
        work.mkdir(parents=True)
        row = {"index": 1, "start": start, "want": want, "heard": want,
               "lang": "zh", "lang_auto": "zh", "forced_zh": None}
        for name in ("verify_v1.json", "verify_v2.json"):
            (work / name).write_text(json.dumps([row]), encoding="utf-8")
        (work / "state.json").write_text(json.dumps({"vc": {"segments": [
            {"start": start, "end": start + 1, "text_translated": want, "vc": converted}]}}), encoding="utf-8")

    def test_before_uses_frozen_membership_after_uses_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "workspace" / "film"
            frozen = work.parent / "_archive_v3.3" / "film"
            self.write_fixture(work, converted=False)
            self.write_fixture(frozen, converted=True)
            self.assertTrue(load_rows(work)[0]["vc"])
            self.assertEqual(load_rows(work)[0]["source"], str(frozen))
            self.assertFalse(load_rows(work, after=True)[0]["vc"])

    def test_historical_rows_without_state_are_not_paired_with_new_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "film"
            self.write_fixture(work, converted=False)
            (work / "verify_v2.pre_v34.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "current state may be overwritten"):
                load_rows(work)

    def test_reused_index_with_changed_text_or_time_is_rejected(self):
        for field, value in (("want", "明天天气不好"), ("start", 2.0)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp) / "film"
                self.write_fixture(work, converted=True)
                p = work / "verify_v2.json"
                rows = json.loads(p.read_text()); rows[0][field] = value
                p.write_text(json.dumps(rows))
                with self.assertRaisesRegex(ValueError, "cannot pair by index"):
                    load_rows(work, after=True)
                self.assertEqual(_verify_rows(work), {})

    def test_new_readings_preserve_auto_when_forced_decode_was_accepted(self):
        row = {"lang": "zh*", "lang_auto": "en", "heard": "今天天气很好",
               "heard_auto": "Hello", "forced_zh": "今天天气很好"}
        r = reading(row)
        self.assertEqual(r["auto"], {"text": "Hello", "language": "en"})
        self.assertEqual(r["zh"]["text"], row["heard"])
        del row["heard_auto"]
        self.assertEqual(reading(row)["auto"]["text"], row["heard"])

    def test_probe_exposes_partial_decode_errors_and_does_not_pair_changed_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "film"
            self.write_fixture(work, converted=True)
            paths = [work / name for name in ("v1.wav", "conv.wav")]
            for path in paths:
                path.write_bytes(b"fixture")
            state = {"fit": {"segments": [{"audio_fit": str(paths[0])}]},
                     "vc": {"converted": 1, "segments": [{"vc": True, "audio": str(paths[1]),
                            "start": 2, "text_translated": "今天天气很好"}]}}
            state_path = work / "state.json"
            state_path.write_text(json.dumps(state))
            judge = SimpleNamespace(decoded=0, model_size="fixture", beam=5, batch=1,
                                    transcribe=lambda paths: {paths[1]: {"auto": {"text": "今天天气很好"}},
                                                              paths[0]: {"auto": {"text": "", "error": "read failed"}}})
            summary = probe_state(state_path, work / "probe", judge=judge, band_sample=0)
            self.assertEqual(summary["error_units"], 1)
            self.assertEqual(summary["judged"], 0)
            self.assertEqual(summary["rejected"], 0)
            self.assertEqual(summary["verify"]["sampled_converted"], 0)
            self.assertEqual(summary["provenance"]["conv_audio_field"], "audio")
            self.assertEqual(list((work / "probe" / "review").glob("*.wav")), [])

    def test_verify_records_raw_text_and_actual_waveform_without_changing_sampling(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fitted, raw = d / "fit.wav", d / "raw.wav"
            fitted.write_bytes(b"fixture fitted audio")
            raw.write_bytes(b"fixture raw audio")
            state = d / "state.json"
            state.write_text(json.dumps({"vc": {"source_key": "audio", "segments": [
                {"start": 0, "text_translated": "今天天气很好", "audio_fit": str(fitted),
                 "audio": str(raw), "gender": "female", "vc": True},
                {"start": 1, "text_translated": "今天天气很好", "audio": str(raw),
                 "gender": "female", "vc": False}]}}))
            calls = []

            def transcribe(path, *, language, **kwargs):
                calls.append((path, language))
                return [SimpleNamespace(text="Hello" if language is None else "今天天气很好")], \
                    SimpleNamespace(language="en" if language is None else "zh", language_probability=0.8)

            with patch.object(sys, "argv", ["verify_dub.py", str(state), "--key", "vc", "-n", "0"]), \
                    patch("ai_movie.asr._load_cpu_model", return_value=SimpleNamespace(transcribe=transcribe)), \
                    patch("ai_movie.diarize._load_mono16k", return_value=np.zeros(1)), \
                    patch("ai_movie.diarize.pitch_track", return_value=(np.full(6, 200), np.ones(6, dtype=bool), None)):
                self.assertEqual(verify_dub.main(), 0)
            rows = json.loads((d / "verify_vc.json").read_text())
            self.assertEqual([r["index"] for r in rows], [1, 2])
            self.assertEqual([r["audio"]["field"] for r in rows], ["audio_fit", "audio"])
            for r, path in zip(rows, (fitted, raw)):
                self.assertEqual(r["heard_auto"], "Hello")
                self.assertEqual(r["heard"], "今天天气很好")
                self.assertEqual(r["lang"], "zh*")
                self.assertEqual(r["audio"], verify_dub.audio_provenance(str(path), r["audio"]["field"]))
            self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
