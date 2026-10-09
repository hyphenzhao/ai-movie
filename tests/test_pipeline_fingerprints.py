"""Selective v3.4 cache invalidation; no media, model loading, or pipeline stages."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_pipeline as P
from ai_movie import config, translator


class StageFingerprintTests(unittest.TestCase):
    def setUp(self):
        # Hashing the dependency names isolates the real cache wiring from imports
        # of optional GPU libraries. A separate test resolves the added pure helpers.
        self.changed_code = set()
        self.hash_patch = patch.object(P, "_source_hash", side_effect=lambda name:
                                      "changed:" + name if name in self.changed_code else "source:" + name)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)
        self.ctx = P.Ctx.__new__(P.Ctx)
        self.ctx.input_fp = "unchanged-input"
        self.ctx.no_fp = False
        self.ctx.args = P.build_parser().parse_args(["unused.mp4"])
        self.ctx.state = {step: {"fixture": True} for step in P.ALL_STEPS}
        for step in P.ALL_STEPS:
            self.ctx.stamp(step)
        self.initial = copy.deepcopy(self.ctx.state["_fp"])

    def stale(self):
        return {step for step in P.ALL_STEPS if self.ctx.explain(step)[0] == "stale"}

    def simulate_recomputed_stamps(self):
        rerun = []
        for step in P.ALL_STEPS:
            if self.ctx.explain(step)[0] == "stale":
                rerun.append(step)
                self.ctx.stamp(step)
        return rerun

    def test_rewrite_option_adoption_invalidates_both_consumers_and_then_downstream(self):
        with patch.dict(translator._REWRITE_OPTIONS, {"repeat_penalty": 1.0}):
            self.assertEqual(self.stale(), {"translate", "compact"})
            self.assertIn("cfg:ai_movie.translator._REWRITE_OPTIONS", self.ctx.explain("translate")[1])
            self.assertEqual(self.simulate_recomputed_stamps(),
                             ["translate", "tts", "compact", "fit", "mix", "faces", "lipsync", "enhance", "compose", "qc"])
            for step in ("demux", "separate", "osd", "asr", "enrol", "glossary"):
                self.assertEqual(self.ctx.state["_fp"][step], self.initial[step])
        # Removing an experimental override also changes the effective generation options.
        self.assertEqual(self.stale(), {"translate", "compact"})

    def test_identical_option_order_and_unrelated_model_cache_do_not_invalidate(self):
        a = {"temperature": 0.0, "repeat_penalty": 1.15}
        b = {"repeat_penalty": 1.15, "temperature": 0.0}
        with patch.object(translator, "_REWRITE_OPTIONS", a):
            fa = self.ctx.fingerprint("translate")
        with patch.object(translator, "_REWRITE_OPTIONS", b):
            self.assertEqual(self.ctx.fingerprint("translate"), fa)
        with patch.dict(translator._model_cache, {"unused-model": object()}):
            self.assertEqual(self.stale(), set())

    def test_inherited_generation_defaults_and_rewrite_acceptance_are_dependencies(self):
        self.changed_code.add("ai_movie.translator._call_ollama_chat")
        self.assertEqual(self.stale(), {"translate", "compact"})
        self.changed_code.clear()
        self.changed_code.add("ai_movie.translator._enforce_edit_ok")
        self.assertEqual(self.stale(), {"translate"})
        self.changed_code.clear()
        self.changed_code.add("ai_movie.translator._char_overlap")
        self.assertEqual(self.stale(), {"compact"})

    def test_l5_pixel_blending_and_switch_report_changes_invalidate_only_picture_branch(self):
        for helper in ("blend_frames", "SwitchLog", "frame_step"):
            with self.subTest(helper=helper):
                self.changed_code.clear()
                self.changed_code.add("ai_movie.switch_fade." + helper)
                self.assertEqual(self.stale(), {"lipsync"})
        self.assertEqual(self.simulate_recomputed_stamps(), ["lipsync", "enhance", "compose", "qc"])

    def test_l5_effective_config_is_already_covered(self):
        with patch.object(config, "LIPSYNC_SWITCH_FADE_FRAMES", config.LIPSYNC_SWITCH_FADE_FRAMES + 1):
            self.assertEqual(self.stale(), {"lipsync"})
        self.ctx.args.switch_fade_frames = 0
        self.assertEqual(self.stale(), {"lipsync"})

    def test_content_held_run_config_and_m3_qc_cutoff_have_distinct_dependencies(self):
        with patch.object(config, "CONTENT_HELD_RUN_MIN", config.CONTENT_HELD_RUN_MIN + 1):
            self.assertEqual(self.stale(), {"asr"})
        with patch.object(config, "VC_CONSIST_OUTLIER", config.VC_CONSIST_OUTLIER + 0.01):
            self.assertEqual(self.stale(), {"qc"})
        self.assertNotEqual(self.initial["asr"]["files"]["ai_movie/content.py"], "missing")
        self.assertNotEqual(self.initial["asr"]["files"]["ai_movie/units.py"], "missing")

    def test_added_source_dependencies_resolve_without_models(self):
        # Use the real inspector only on the pure helpers added for this change.
        self.hash_patch.stop()
        for step, names in {
            "translate": ("ai_movie.translator._call_ollama_chat", "ai_movie.translator._clean_ollama_output",
                          "ai_movie.translator._clean_enforce_candidate", "ai_movie.translator._enforce_edit_ok"),
            "compact": ("ai_movie.translator._glossary_hit", "ai_movie.translator._visible_len",
                        "ai_movie.translator._char_overlap", "ai_movie.composer._visible_chars"),
            "lipsync": ("ai_movie.switch_fade.blend_frames", "ai_movie.switch_fade.SwitchLog",
                        "ai_movie.switch_fade.frame_step"),
        }.items():
            for name in names:
                self.assertIn(name, P.STEP_CODE[step])
                self.assertNotEqual(P._source_hash(name), "missing", name)


if __name__ == "__main__":
    unittest.main()
