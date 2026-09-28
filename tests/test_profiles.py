"""profiles: merging chunk speakers into people and assigning chunks to them (numpy only).

    .venv/bin/python tests/test_profiles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie import profiles as P      # noqa: E402

rng = np.random.default_rng(0)


def unit(v):
    v = np.asarray(v, np.float32); return v / np.linalg.norm(v)


def near(v, eps=0.05):
    return unit(np.asarray(v) + eps * rng.standard_normal(len(v)))


def test_merge_by_voice_or_face_never_across_gender():
    va, vb, vc = unit(rng.standard_normal(192)), unit(rng.standard_normal(192)), unit(rng.standard_normal(192))
    fa, fb = unit(rng.standard_normal(512)), unit(rng.standard_normal(512))
    nodes = [
        {"gender": "female", "voice": va, "face": fa, "seconds": 200},          # 0 actress, chunk 1
        {"gender": "female", "voice": near(va), "face": None, "seconds": 60},   # 1 same voice, no face
        {"gender": "female", "voice": vb, "face": near(fa), "seconds": 50},     # 2 different voice, same face
        {"gender": "male", "voice": near(va), "face": near(fa), "seconds": 30}, # 3 male: never merged
        {"gender": "female", "voice": vc, "face": fb, "seconds": 10},           # 4 someone else
    ]
    groups = P.merge_sources(nodes, voice_link_dist=0.55, face_min_cos=0.5)
    assert sorted(groups[0]) == [0, 1, 2]
    assert [3] in groups and [4] in groups


def test_assign_matched_default_and_only_profile():
    va, vb = unit(rng.standard_normal(192)), unit(rng.standard_normal(192))
    fa, fb = unit(rng.standard_normal(512)), unit(rng.standard_normal(512))
    profiles = {"P0": {"gender": "female", "f0_median": 250, "voice": va, "face": fa, "default_for_gender": True},
                "P1": {"gender": "female", "f0_median": 210, "voice": vb, "face": fb},
                "P2": {"gender": "male", "f0_median": 120, "voice": unit(rng.standard_normal(192)), "face": None,
                       "default_for_gender": True}}
    speakers = {"S0": {"gender": "female", "f0_median": 245, "voice": near(va), "face": near(fa)},   # clear P0
                "S1": {"gender": "female", "f0_median": 230, "voice": unit(rng.standard_normal(192)), "face": None},  # nobody
                "S2": {"gender": "male", "f0_median": 110, "voice": unit(rng.standard_normal(192)), "face": None}}  # only male
    got = P.assign_profiles(speakers, profiles, min_score=0.30, margin=0.08)
    assert got["S0"]["profile"] == "P0" and got["S0"]["how"] == "matched"
    assert got["S1"]["profile"] == "P0" and got["S1"]["how"] == "gender default"
    assert got["S2"]["profile"] == "P2" and got["S2"]["how"] == "only profile of gender"


def test_pitch_match():
    assert P.pitch_match(250, 250) == 1.0
    assert P.pitch_match(250, 125) == 0.0
    assert 0.6 < P.pitch_match(250, 230) < 0.9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
