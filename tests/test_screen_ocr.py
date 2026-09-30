"""screen_ocr helpers and the eval_long truth merge (no GPU, no film needed; the OCR smoke
test skips with a reason when rapidocr_onnxruntime or the Noto CJK font is missing).

    .venv/bin/python tests/test_screen_ocr.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)     # type: ignore[union-attr]
    return mod


so = _load("screen_ocr")
el = _load("eval_long")


def test_strip_rows_roundtrip():
    # exactly screen_subs.py:107 — rows of the mask stacked with a 6-px zero gap
    rng = np.random.default_rng(0)
    r1 = (rng.random((47, 1320)) > 0.7).astype(np.uint8) * 255
    r2 = (rng.random((32, 1320)) > 0.7).astype(np.uint8) * 255
    gap = np.zeros((6, 1320), np.uint8)
    strip = np.vstack([r1, gap, r2, gap])
    lines = [(930, 976), (1005, 1036)]                       # inclusive bounds: h = 47, 32
    rows = so.strip_rows(strip, lines)
    assert len(rows) == 2 and np.array_equal(rows[0], r1) and np.array_equal(rows[1], r2)
    one = so.strip_rows(np.vstack([r2, gap]), [(1006, 1037)])
    assert len(one) == 1 and np.array_equal(one[0], r2)


def test_row_role_matches_human_one_liners():
    # screen.json one-row ids: 241/246/866 (zh only, y0 931/951/951), 239/406 (ja only, y0 1006)
    assert so.row_role(931) == "zh" and so.row_role(951) == "zh"
    assert so.row_role(1006) == "ja"
    assert so.row_role(989) == "zh" and so.row_role(990) == "ja"


def test_cer_and_flags():
    assert so.cer("撮影始まりましたけども、元気ですか?", "撮影始まりましたけども元気ですか？") == 0.0
    assert so.cer("カネちゃん", "かねちゃん") == 0.0                 # katakana folds like L2b
    assert abs(so.cer("早", "早上好") - 2 / 3) < 1e-9
    assert so.cer("", "abc") == 1.0 and so.cer("", "") == 0.0
    # a kana-free ja reading is valid Japanese (id 13 「今回撮影、何本目?」): advisory flag, row kept
    f = so.row_flags("ja", "今回撮影、何本目?", 0.9)
    assert f == ["no_kana"] and so.is_kept(f)
    assert so.is_kept(so.row_flags("ja", "たぶん8本目かな?", 0.9)) and so.row_flags("ja", "たぶん8本目かな?", 0.9) == []
    # below the cutoff → dropped; identical rows → dropped; empty → dropped
    assert not so.is_kept(so.row_flags("zh", "早上好", 0.59, min_score=0.6))
    assert so.is_kept(so.row_flags("zh", "早上好", 0.60, min_score=0.6))
    assert "row_dup" in so.row_flags("ja", "早上好。", 0.9, other="早上好") and not so.is_kept(["row_dup"])
    assert "empty" in so.row_flags("zh", " 。", 0.9)
    # a recogniser whose dictionary has no kana cannot read the ja row at all
    f = so.row_flags("ja", "撮影始、元気？", 0.71, dict_has_kana=False)
    assert "dict_no_kana" in f and not so.is_kept(f)
    assert so.is_kept(so.row_flags("zh", "撮影始", 0.71, dict_has_kana=False))
    # an unscored reading is recorded, not dropped
    assert so.row_flags("zh", "早上好", None) == ["unscored"] and so.is_kept(["unscored"])
    # edge junk from stray mask pixels goes, sentence punctuation stays
    assert so.clean_reading("虽然已经开始了拍摄但还精神吗？：") == "虽然已经开始了拍摄但还精神吗？"
    assert so.clean_reading("　我是S-1专属濑户神流　") == "我是S-1专属濑户神流" and so.clean_reading("·。") == "。"


def test_score_cutoff_calibration():
    # gross errors sit at low scores: the smallest cutoff with ≤ 5 % gross among kept is 0.55
    good = [(0.55 + 0.004 * i, 0.0) for i in range(100)]
    bad = [(0.50, 0.9), (0.52, 0.8), (0.53, 0.7), (0.54, 0.6), (0.62, 0.5), (0.70, 0.4)]
    assert so.calibrate_cutoff(good + bad) == 0.55
    # never below the floor, even when everything is clean; nothing above the floor → no cutoff
    assert so.calibrate_cutoff([(0.55, 0.0)] * 20) == 0.5
    assert so.calibrate_cutoff([(0.3, 0.0)] * 20) is None
    # too few pairs → None; no achievable cutoff → None
    assert so.calibrate_cutoff([(0.9, 0.0)] * 5) is None
    assert so.calibrate_cutoff([(0.9, 0.9)] * 20) is None
    # a few gross errors at high scores must not collapse coverage (the design's 80 % rule would)
    pairs = [(0.6 + 0.003 * i, 0.0) for i in range(100)] + [(0.95, 0.9), (0.96, 0.9), (0.97, 0.9)]
    assert so.calibrate_cutoff(pairs) == 0.5


def test_best_reading_prefers_score_then_agreement():
    best = so.best_reading([{"text": "a", "score": 0.6, "src": "mask"}, {"text": "b", "score": 0.8, "src": "raw"}])
    assert best["text"] == "b"
    agree = so.best_reading([{"text": "早上好", "score": None, "src": "mask"}, {"text": "早上好", "score": None, "src": "mask3x"}])
    assert agree["text"] == "早上好" and abs(agree["score"] - 1.0) < 1e-9
    half = so.best_reading([{"text": "早上好", "score": None, "src": "mask"}, {"text": "早上", "score": None, "src": "mask3x"}])
    assert abs(half["score"] - (1 - 1 / 3)) < 1e-9
    assert so.best_reading([{"text": "x", "score": None, "src": "mask"}])["score"] is None
    assert so.best_reading([]) is None


def test_merge_precedence_human_over_ocr():
    human = {3: ["早上好", "おはようございます。"], 241: ["进去了", ""], 239: ["", "奥まで行くよ。"]}
    ocr = {"3": ["早安", "おはよう"], "241": ["进去", "はい"], "5": ["虽然已经开始了拍摄", ""], "6": ["", ""]}
    read, src = so.merge_truth(human, ocr)
    assert read[3] == ["早上好", "おはようございます。"] and src[3] == "human"
    assert read[241] == ["进去了", ""] and src[241] == "human"       # an empty human row still wins
    assert read[5] == ["虽然已经开始了拍摄", ""] and src[5] == "ocr"
    assert 6 not in read and read[239] == ["", "奥まで行くよ。"]


def test_eval_long_read_truth_gate_and_modes():
    with tempfile.TemporaryDirectory() as d:
        scr = Path(d)
        (scr / "screen.json").write_text(json.dumps({"3": ["早上好", "おはようございます。"], "241": ["进去了", ""]}), encoding="utf-8")
        (scr / "screen_ocr.json").write_text(json.dumps({"3": ["早安", "x"], "5": ["虽然", "撮影始まりました"], "7": ["大家", "こちら"], "9": ["向大家", ""]}), encoding="utf-8")
        (scr / "screen_ocr.meta.json").write_text(json.dumps({"rows": {
            "5": {"zh": {"score": 0.9}, "ja": {"score": 0.95}},
            "7": {"zh": {"score": 0.3}, "ja": {"score": 0.4}},          # both below the gate → dropped
            "9": {"zh": {"score": 0.7}},
        }}), encoding="utf-8")
        read, src = el.read_truth(scr, truth="merged", min_conf=0.6)
        assert read[3] == ["早上好", "おはようございます。"] and src[3] == "human"
        assert read[241] == ["进去了", ""]
        assert read[5] == ["虽然", "撮影始まりました"] and src[5] == "ocr"
        assert 7 not in read
        assert read[9] == ["向大家", ""]
        # a stricter gate at eval time re-cuts without re-running OCR
        read2, _ = el.read_truth(scr, truth="merged", min_conf=0.92)
        assert read2[5] == ["", "撮影始まりました"] and 9 not in read2
        # human-only ignores the OCR file; a screen dir without OCR is unchanged
        read3, src3 = el.read_truth(scr, truth="human")
        assert set(read3) == {3, 241} and set(src3.values()) == {"human"}
        os.remove(scr / "screen_ocr.meta.json"); os.remove(scr / "screen_ocr.json")
        assert el.read_truth(scr)[0] == read3
    assert el.EVAL_TRUTH in ("merged", "human") and 0.5 <= el.OCR_MIN_CONF <= 0.9


def test_ocr_smoke():
    font_p = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
    try:
        import rapidocr_onnxruntime  # noqa: F401
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:                                # noqa: BLE001
        print("  skip test_ocr_smoke:", exc); return
    if not font_p.exists():
        print("  skip test_ocr_smoke: no Noto CJK font"); return
    eng = so.RapidRec()
    assert eng.dict_size > 6000
    font = ImageFont.truetype(str(font_p), 36)

    def render(text: str) -> tuple[np.ndarray, np.ndarray]:
        im = Image.new("RGB", (1320, 52), (110, 105, 100))
        dr = ImageDraw.Draw(im)
        w = dr.textlength(text, font=font); x = int((1320 - w) / 2)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                dr.text((x + dx, 6 + dy), text, font=font, fill=(20, 20, 20))
        dr.text((x, 6), text, font=font, fill=(255, 255, 255))
        raw = np.asarray(im)[:, :, ::-1].copy()
        mask = ((raw.min(axis=2) > 200)).astype(np.uint8) * 255
        return raw, mask
    zh = "大家正在用相机看着我们"
    raw, mask = render(zh)
    (t_raw, s_raw), (t_mask, s_mask) = eng.read([so.prep_raw(raw), so.prep_row(mask)])
    assert so.cer(t_raw, zh) <= 0.15, (t_raw, s_raw)
    assert so.cer(t_mask, zh) <= 0.15, (t_mask, s_mask)
    # the bundled Chinese dictionary cannot spell kana: the ja row must be refused, not mis-read
    ja = "撮影始まりましたけども、元気ですか?"
    (t_ja, s_ja), = eng.read([so.prep_raw(render(ja)[0])])
    if not eng.dict_has_kana:
        assert so.kana_count(t_ja) <= 1 and not so.is_kept(so.row_flags("ja", t_ja, s_ja, dict_has_kana=False))
    else:
        assert so.kana_count(t_ja) >= 8 and so.cer(t_ja, ja) <= 0.15, (t_ja, s_ja)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
