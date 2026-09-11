"""Sentence-level re-segmentation of a Whisper word stream.

Why this exists
---------------
Whisper emits *contiguous* segments inside one VAD chunk — segment *n*'s
``start`` equals segment *n-1*'s ``end``.  The old merge rule in
``asr.py`` ("merge when the gap is <= 0.05 s") therefore chained an entire
chunk into a single blob: a 390 s interview produced 35 segments, the
longest 44 s, each mixing the interviewer and the interviewee.  Everything
downstream inherited that: F0 gender detection saw two voices at once,
translation saw an unpunctuated wall of text, and lip-sync had no usable
speaker boundary.

This module rebuilds segments from word timings instead, cutting on
(in priority order) speaker changes, sentence-final punctuation, pauses,
and finally a hard duration/length cap.
"""

from __future__ import annotations

from ai_movie.config import (
    ASR_MAX_SEGMENT_CHARS,
    ASR_MAX_SEGMENT_DURATION,
    ASR_MIN_SEGMENT_DURATION,
    ASR_PAUSE_SPLIT_SEC,
    ASR_SENTENCE_END,
    ASR_SOFT_BREAK,
)

# Characters that carry no speech and should never form a segment alone.
_PUNCT_ONLY = set(" \t　。、，,．.！!？?…‥・「」『』（）()～~♪♬-—ー")

# Stock phrases Whisper hallucinates on low-energy audio.  They come from the
# YouTube/subtitle text it was trained on and appear verbatim with high
# confidence, so no threshold catches them — only a blocklist does.  Observed
# on this project: 「ご視聴ありがとうございました。」 emitted for a 0.54 s gap
# between two interview turns.
_HALLUCINATION_PHRASES = (
    "ご視聴ありがとうございました",
    "ご覧いただきありがとうございます",
    "最後までご視聴",
    "チャンネル登録",
    "高評価",
    "字幕視聴者",
    "本編は概要欄",
    "Thanks for watching",
    "Subscribe to",
    "请不吝点赞",
    "訂閱",
    "字幕由",
)


def is_hallucination(text: str) -> bool:
    """Whether *text* is one of Whisper's stock filler outputs."""
    t = (text or "").strip().strip("。．.!！?？ 　")
    if not t:
        return False
    return any(p in t for p in _HALLUCINATION_PHRASES)


def _is_punct_only(text: str) -> bool:
    return not text or all(ch in _PUNCT_ONLY for ch in text)


def _visible_len(text: str) -> int:
    """Character count ignoring whitespace and punctuation."""
    return sum(1 for ch in text if ch not in _PUNCT_ONLY)


def speaker_at(
    turns: list[dict] | None,
    start: float,
    end: float,
    fallback: str | None = None,
) -> str | None:
    """Return the speaker label with the largest overlap over ``[start, end]``.

    Falls back to the nearest turn by midpoint distance when nothing
    overlaps (e.g. a word sitting in a diarization gap).
    """
    if not turns:
        return fallback
    best_label, best_ov = None, 0.0
    for t in turns:
        ov = min(end, float(t["end"])) - max(start, float(t["start"]))
        if ov > best_ov:
            best_ov, best_label = ov, t["speaker"]
    if best_label is not None:
        return best_label
    mid = (start + end) / 2.0
    best_label, best_dist = fallback, float("inf")
    for t in turns:
        d = abs(((float(t["start"]) + float(t["end"])) / 2.0) - mid)
        if d < best_dist:
            best_dist, best_label = d, t["speaker"]
    return best_label


def _flush(buf: list[dict], speaker: str | None) -> dict | None:
    """Turn an accumulated word buffer into a segment dict."""
    if not buf:
        return None
    text = "".join(w["w"] for w in buf).strip()
    if _is_punct_only(text) or is_hallucination(text):
        return None
    probs = [float(w.get("p", 0.0)) for w in buf if w.get("p") is not None]
    return {
        "start": round(float(buf[0]["s"]), 2),
        "end": round(float(buf[-1]["e"]), 2),
        "text": text,
        "speaker": speaker or "",
        "asr_conf": round(sum(probs) / len(probs), 3) if probs else 0.0,
        "n_words": len(buf),
    }


def split_into_sentences(
    words: list[dict],
    *,
    speaker_turns: list[dict] | None = None,
    max_duration: float = ASR_MAX_SEGMENT_DURATION,
    max_chars: int = ASR_MAX_SEGMENT_CHARS,
    min_duration: float = ASR_MIN_SEGMENT_DURATION,
    pause_split: float = ASR_PAUSE_SPLIT_SEC,
) -> list[dict]:
    """Split a word stream into sentence-sized segments.

    Parameters
    ----------
    words:
        ``[{"w": str, "s": float, "e": float, "p": float}]`` on the global
        timeline, in ascending time order.
    speaker_turns:
        Optional diarization output ``[{"start","end","speaker"}]``.  When
        given, a speaker change forces a cut — this is what stops one
        segment from containing two people.
    """
    words = [w for w in words if (w.get("w") or "").strip() or w.get("w") == " "]
    if not words:
        return []

    segments: list[dict] = []
    buf: list[dict] = []
    buf_speaker: str | None = None

    def cut() -> None:
        nonlocal buf, buf_speaker
        seg = _flush(buf, buf_speaker)
        if seg is not None:
            segments.append(seg)
        buf = []
        buf_speaker = None

    for i, w in enumerate(words):
        w_start, w_end = float(w["s"]), float(w["e"])
        token = (w.get("w") or "")
        spk = speaker_at(speaker_turns, w_start, w_end, fallback=buf_speaker)

        # (2) Speaker change → cut BEFORE this word.
        if buf and speaker_turns and spk is not None and spk != buf_speaker:
            cut()

        # (3) Long pause → cut before this word.
        if buf:
            gap = w_start - float(buf[-1]["e"])
            if gap >= pause_split:
                cut()

        # (5) Hard cap → cut before this word, at the widest gap available.
        if buf:
            cur_dur = w_end - float(buf[0]["s"])
            cur_chars = _visible_len("".join(x["w"] for x in buf))
            if cur_dur > max_duration or cur_chars >= max_chars:
                _cut_at_widest_gap(buf, segments, buf_speaker)
                if not buf:
                    buf_speaker = None

        if not buf:
            buf_speaker = spk
        buf.append({"w": token, "s": w_start, "e": w_end, "p": w.get("p")})

        stripped = token.strip()
        # (1) Sentence-final punctuation → cut after this word.
        if stripped and stripped[-1] in ASR_SENTENCE_END:
            cut()
            continue

        # (4) Soft punctuation, but only under budget pressure.
        if stripped and stripped[-1] in ASR_SOFT_BREAK:
            dur = float(buf[-1]["e"]) - float(buf[0]["s"])
            chars = _visible_len("".join(x["w"] for x in buf))
            if dur >= 0.7 * max_duration or chars >= 0.8 * max_chars:
                cut()

    cut()

    return _absorb_fragments(segments, min_duration=min_duration,
                             max_duration=max_duration)


def _cut_at_widest_gap(buf: list[dict], segments: list[dict],
                       speaker: str | None) -> None:
    """Emit part of *buf* as a segment, cutting at the widest inter-word gap.

    Mutates *buf* in place, leaving the remainder.  Prefers a gap in the
    last third so the emitted piece stays close to the cap.
    """
    if len(buf) < 2:
        seg = _flush(buf, speaker)
        if seg is not None:
            segments.append(seg)
        del buf[:]
        return

    lo = max(1, (len(buf) * 2) // 3)
    best_i, best_gap = None, -1.0
    for i in range(lo, len(buf)):
        gap = float(buf[i]["s"]) - float(buf[i - 1]["e"])
        if gap > best_gap:
            best_gap, best_i = gap, i
    if best_i is None or best_gap <= 0.0:
        # No usable gap in the tail — search the whole buffer.
        for i in range(1, len(buf)):
            gap = float(buf[i]["s"]) - float(buf[i - 1]["e"])
            if gap > best_gap:
                best_gap, best_i = gap, i
    if best_i is None:
        best_i = len(buf)

    head, tail = buf[:best_i], buf[best_i:]
    seg = _flush(head, speaker)
    if seg is not None:
        segments.append(seg)
    del buf[:]
    buf.extend(tail)


def _absorb_fragments(segments: list[dict], *, min_duration: float,
                      max_duration: float) -> list[dict]:
    """Merge sub-``min_duration`` fragments into an adjacent same-speaker segment."""
    if not segments:
        return []

    out: list[dict] = []
    for seg in segments:
        dur = seg["end"] - seg["start"]
        short = dur < min_duration or _visible_len(seg["text"]) <= 1
        if not short or not out:
            out.append(seg)
            continue

        prev = out[-1]
        can_prev = (prev.get("speaker", "") == seg.get("speaker", "")
                    and seg["end"] - prev["start"] <= max_duration)
        if can_prev:
            prev["end"] = seg["end"]
            prev["text"] = (prev["text"] + seg["text"]).strip()
            prev["n_words"] = prev.get("n_words", 0) + seg.get("n_words", 0)
        else:
            out.append(seg)

    # A leading fragment has no predecessor — fold it into its successor.
    if len(out) >= 2:
        first = out[0]
        if ((first["end"] - first["start"]) < min_duration
                and first.get("speaker", "") == out[1].get("speaker", "")
                and out[1]["end"] - first["start"] <= max_duration):
            out[1]["start"] = first["start"]
            out[1]["text"] = (first["text"] + out[1]["text"]).strip()
            out.pop(0)

    return [s for s in out
            if not _is_punct_only(s["text"]) and not is_hallucination(s["text"])]


def segments_to_words(segments: list[dict]) -> list[dict]:
    """Degrade whole segments into pseudo-words (one per segment).

    Used when ``word_timestamps`` is unavailable (older backends, or a ROCm
    DTW failure) so the same splitter still runs — it can then only cut on
    segment boundaries and speaker changes, but never produces the 44 s
    blobs the old merge rule did.
    """
    words: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        words.append({
            "w": text,
            "s": float(seg["start"]),
            "e": float(seg["end"]),
            "p": seg.get("asr_conf"),
        })
    return words
