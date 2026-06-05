"""Offline text translation using Tencent Hy-MT1.5-1.8B (MIT license).

Lazy-loads the model on first use. GPU preferred (ROCm/CUDA), CPU fallback.
Also supports Ollama-based translation for colloquial / NSFW content.

ChatML prompt template::

    <|im_start|>user
    Translate Japanese to Chinese:
    {segment_text}<|im_end|>
    <|im_start|>assistant

"""

import threading
from pathlib import Path
from typing import Callable

from ai_movie.config import (
    TRANSLATION_MODEL_PATH,
    TRANSLATION_BATCH_SIZE,
    TRANSLATION_MAX_NEW_TOKENS,
    TRANSLATION_CONTEXT_SEGMENTS,
    TRANSLATION_TARGET_LANGS,
)

# Supported target language labels (for UI)
TARGET_LANG_LABELS = list(TRANSLATION_TARGET_LANGS.keys())

# ChatML template markers (kept short to avoid tokenisation issues)
_CHATML_USER = "<|im_start|>user"
_CHATML_ASSISTANT = "<|im_start|>assistant"
_CHATML_END = "<|im_end|>"

# Module-level model cache (lazy-loaded, thread-safe)
_model = None
_tokenizer = None
_lock = threading.Lock()


def _is_model_downloaded() -> bool:
    p = Path(TRANSLATION_MODEL_PATH)
    return p.is_dir() and (p / "model.safetensors").exists()


def _load_model():
    """Lazy-load model & tokenizer (thread-safe, idempotent)."""
    global _model, _tokenizer
    if _model is not None:
        return
    with _lock:
        if _model is not None:          # double-checked locking
            return

        if not _is_model_downloaded():
            raise FileNotFoundError(
                f"Translation model not found at {TRANSLATION_MODEL_PATH}\n"
                f"Download it:\n"
                f"  git clone https://huggingface.co/tencent/Hy-MT1.5-1.8B "
                f"{TRANSLATION_MODEL_PATH}"
            )

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        _tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL_PATH, trust_remote_code=True,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = AutoModelForCausalLM.from_pretrained(
            TRANSLATION_MODEL_PATH,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            _model = _model.to(device)
        _model.eval()


def _build_prompt(text: str, context: str | None,
                  src_lang_name: str, tgt_lang_name: str) -> str:
    """Build a ChatML-formatted translation prompt for one segment."""
    instruction = f"Translate {src_lang_name} to {tgt_lang_name}:"

    if context:
        instruction = f"Context: {context}\n\n{instruction}"

    return (
        f"{_CHATML_USER}\n"
        f"{instruction}\n"
        f"{text}{_CHATML_END}\n"
        f"{_CHATML_ASSISTANT}\n"
    )


def _extract_translation(raw: str, original: str = "") -> str:
    """Strip ChatML tokens, HTML tags, explanations & trailing artifacts.

    Hy-MT sometimes produces grammar explanations or echoes the input
    instead of a real translation — especially for short / ambiguous words.
    This function detects and strips those patterns.
    """
    import re
    text = raw.strip()

    # ── 0. Remove HTML tags ──────────────────────────────────
    text = re.sub(r"<br\s*/?>", "", text)

    # ── 1. Remove ChatML tokens ──────────────────────────────
    text = re.sub(r"</?im_start>", "", text)
    text = re.sub(r"<\|im_end[\|>]*", "", text)
    for suffix in (_CHATML_END, _CHATML_ASSISTANT, _CHATML_USER,
                   "<|im_end>", "<|im_end"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    for marker in ("<|im_start|>", "<|im_end|>", "<|im_end>", "</im_start>"):
        if marker in text:
            text = text.split(marker)[0].strip()

    text = text.strip()
    if not text:
        return ""

    # ── 2. Echo detection: Hy-MT just repeated "原文：..." ───
    if text.startswith("原文：") or text.startswith("原文:"):
        return ""

    # ── 3. Explanation detection ─────────────────────────────
    # If Hy-MT generated an explanation instead of a translation,
    # there is no translation to salvage — return empty.
    _explain_markers = [
        "文法", "这个表达", "这个词语", "这个单词", "这个句子",
        "这个词", "この表現", "この言葉", "この単語",
        "意思是", "语义不明", "意味不明", "语意不明",
        "可以翻译成", "翻译成",
        "通常使用", "通常、この",
        "注意してください", "以下の点",
        "テンプレート",
    ]
    if any(m in text for m in _explain_markers):
        return ""

    # Length heuristic: output >6x longer than input → explanation
    if original:
        ratio = len(text) / max(1, len(original))
        if ratio > 6 and len(text) > 60:
            return ""

    return text


# ── Ollama output cleaner ────────────────────────────────────────

def _clean_ollama_output(raw: str) -> str:
    """Aggressively strip LLM commentary from Ollama output.

    Adult-film translations should be pure dialogue — no explanations,
    no parenthetical notes, no metadata.  This function strips anything
    that looks like model-generated commentary.
    """
    import re as _re
    text = raw.strip()

    # 1. Strip ChatML tokens (including truncated forms)
    text = _re.sub(r"<\|im_start[\|>]*|<\|im_end[\|>]*|</?im_start>|</?im_end>",
                   "", text)

    # 2. Remove ALL parenthetical content — translation dialogue never
    #    needs parentheses.  This catches （粗俗语）, (Translation: ...),
    #    （解释：...）, and any other model commentary.
    text = _re.sub(r"[（(][^)）]*[)）]", "", text)

    # 3. Remove everything after/before common explanation markers.
    #    Split on the first occurrence and keep only what comes before.
    for marker in (
        "解释：", "说明：", "备注：", "注意：", "注：",
        "翻译：", "翻译结果", "译文：",
        "Explanation:", "Note:", "Translation:",
        "（", "(", "【",
    ):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
        elif idx == 0 and marker in (
            "解释：", "说明：", "备注：", "注意：", "注：",
            "翻译：", "翻译结果", "译文：",
        ):
            # Meta prefix at start — strip it and keep whatever follows
            text = text[len(marker):].lstrip("：: ")

    # 4. Strip whole lines that are purely explanatory headers
    text = _re.sub(
        r"(?i)^\s*(Translation|翻译|解释|说明|备注|注意|Note|Explanation)[：:]\s*.*$",
        "", text, flags=_re.MULTILINE,
    )

    # 5. Trim trailing repeated punctuation (LLM rambling artifact)
    text = _re.sub(r"([。！？…\.!\?])\1{4,}$", r"\1", text)

    # 6. Collapse multiple newlines, trim whitespace
    text = _re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # 7. If the result is empty after cleaning, return empty string
    if not text or not text.strip():
        return ""

    # 8. Take only the first substantive line (ignore leading blank lines)
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line

    return ""


def translate(
    segments: list[dict],
    target_lang: str = "Chinese",
    src_lang: str = "Japanese",
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Translate transcript segments to the target language.

    Parameters
    ----------
    segments:
        List of dicts with keys ``text``, ``start``, ``end``, ``source``.
    target_lang:
        Target language name for the prompt (e.g. ``"Chinese"``).
    src_lang:
        Source language name (e.g. ``"Japanese"``).
    progress_cb:
        ``progress_cb(current, total)`` called after each batch completes.
    cancel_check:
        Return ``True`` to abort between batches.

    Returns
    -------
    Same list with ``text_translated`` key added to each segment.
    """
    _load_model()

    import torch
    device = _model.device
    batch_size = TRANSLATION_BATCH_SIZE
    total = len(segments)
    results: list[dict] = list(segments)

    for batch_start in range(0, total, batch_size):
        if cancel_check and cancel_check():
            break

        batch_end = min(batch_start + batch_size, total)
        batch_segs = segments[batch_start:batch_end]

        # Build prompts (with optional preceding-segment context)
        prompts: list[str] = []
        for j, seg in enumerate(batch_segs):
            global_idx = batch_start + j
            context: str | None = None
            if TRANSLATION_CONTEXT_SEGMENTS > 0 and global_idx > 0:
                prev = results[global_idx - 1]
                context = prev.get("text_translated") or prev["text"]
            prompts.append(
                _build_prompt(seg["text"], context, src_lang, target_lang)
            )

        # Tokenize (remove token_type_ids — not used by this model)
        inputs = _tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=TRANSLATION_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )

        # Decode generated portion (strip input prompt)
        for j, seg in enumerate(batch_segs):
            prompt_len = inputs["input_ids"][j].size(0)
            gen_ids = outputs[j][prompt_len:]
            raw = _tokenizer.decode(gen_ids, skip_special_tokens=True)
            global_idx = batch_start + j
            results[global_idx]["text_translated"] = _extract_translation(
                raw, original=seg["text"])

        if progress_cb:
            progress_cb(batch_end, total)

    return results


# ── Ollama translation backend ───────────────────────────────────

def translate_ollama(
    segments: list[dict],
    target_lang: str = "Chinese",
    src_lang: str = "Japanese",
    model: str | None = None,
    base_url: str | None = None,
    system_prompt: str | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    segment_cb: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Translate transcript segments using Ollama (dolphin-mixtral).

    Uses the Ollama HTTP API to translate each segment individually with
    a NSFW-oriented system prompt for colloquial/slang-heavy translation.

    Parameters
    ----------
    segments:
        List of dicts with keys ``text``, ``start``, ``end``, ``source``.
    target_lang:
        Target language name (e.g. ``"Chinese"``).
    src_lang:
        Source language name (e.g. ``"Japanese"``).
    model:
        Ollama model name.  Defaults to ``config.OLLAMA_MODEL``.
    base_url:
        Ollama server URL.  Defaults to ``config.OLLAMA_BASE_URL``.
    system_prompt:
        Override the default system prompt.  Defaults to
        ``config.OLLAMA_NSFW_SYSTEM_PROMPT``.
    progress_cb:
        ``progress_cb(current, total)`` called after each segment.
    cancel_check:
        Return ``True`` to abort between segments.

    Returns
    -------
    Same list with ``text_translated`` key added to each segment.
    """
    import json as _json
    import urllib.request as _urllib

    from ai_movie.config import (
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        OLLAMA_NSFW_SYSTEM_PROMPT,
    )

    if model is None:
        model = OLLAMA_MODEL
    if base_url is None:
        base_url = OLLAMA_BASE_URL
    if system_prompt is None:
        system_prompt = OLLAMA_NSFW_SYSTEM_PROMPT

    total = len(segments)
    results: list[dict] = list(segments)
    chat_url = f"{base_url.rstrip('/')}/api/chat"

    for i, seg in enumerate(results):
        if cancel_check and cancel_check():
            break

        text = seg.get("text", "").strip()
        if not text:
            results[i]["text_translated"] = ""
            if progress_cb:
                progress_cb(i + 1, total)
            continue

        # Build the user instruction with source/target language context
        user_msg = f"Translate {src_lang} to {target_lang}:\n{text}"

        payload = _json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
        }, ensure_ascii=False).encode("utf-8")

        try:
            req = _urllib.Request(
                chat_url,
                data=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            with _urllib.urlopen(req, timeout=300) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
            translated = body.get("message", {}).get("content", "").strip()
        except Exception as exc:
            translated = f"[Ollama error: {exc}]"

        translated = _clean_ollama_output(translated)

        results[i]["text_translated"] = translated

        if segment_cb:
            try:
                segment_cb(i, translated)
            except Exception:
                pass

        if progress_cb:
            progress_cb(i + 1, total)

    return results


def polish_ollama(
    segments: list[dict],
    batch_size: int = 10,
    model: str | None = None,
    base_url: str | None = None,
    system_prompt: str | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    segment_cb: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Polish Hy-MT translated Chinese text with Ollama for NSFW style.

    Sends segments in batches to Ollama, using ``<<<SEG_N>>>`` delimiters
    to separate segments in both the input prompt and the model output.
    This is much faster than per-segment calls (1 HTTP request per batch
    instead of 1 per segment).

    Each segment in the batch includes both the original Japanese text
    and the Hy-MT Chinese translation, so the model can reference the
    original to translate any slang that Hy-MT missed.

    Parameters
    ----------
    segments:
        List of dicts with ``text`` (original) and ``text_translated``
        (Hy-MT Chinese) already set.
    batch_size:
        Number of segments per Ollama request (default 10).
    model / base_url / system_prompt:
        Defaults from ``config.py``.
    progress_cb:
        ``progress_cb(completed, total)`` called after each batch.
    segment_cb:
        ``segment_cb(index, polished_text)`` called per parsed segment.
    cancel_check:
        Return ``True`` to abort between batches.

    Returns
    -------
    Same list with ``text_translated`` replaced by polished version.
    """
    import json as _json
    import re as _re
    import urllib.request as _urllib

    from ai_movie.config import (
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        OLLAMA_POLISH_SYSTEM_PROMPT,
        OLLAMA_NSFW_KEYWORDS,
    )

    if model is None:
        model = OLLAMA_MODEL
    if base_url is None:
        base_url = OLLAMA_BASE_URL
    if system_prompt is None:
        system_prompt = OLLAMA_POLISH_SYSTEM_PROMPT

    total = len(segments)
    results: list[dict] = list(segments)
    chat_url = f"{base_url.rstrip('/')}/api/chat"

    # ── 1. Pre-clean ChatML tokens from Hy-MT output ────────────
    for seg in results:
        t = seg.get("text_translated", "")
        if t:
            t = _re.sub(r"<\|im_start[\|>]*|<\|im_end[\|>]*|</?im_start>|</?im_end>",
                        "", t).strip()
            seg["text_translated"] = t

    # ── 2. Classify: which segments need NSFW polish ────────────
    nsfw_indices: list[int] = []
    keywords_lower = [kw.lower() for kw in OLLAMA_NSFW_KEYWORDS]
    for i, seg in enumerate(results):
        text = seg.get("text", "").lower()
        if any(kw in text for kw in keywords_lower):
            nsfw_indices.append(i)

    nsfw_count = len(nsfw_indices)
    total = len(results)

    # ── 3. Fire segment_cb for non-NSFW segments immediately ────
    if segment_cb:
        for i, seg in enumerate(results):
            if i not in nsfw_indices:
                try:
                    segment_cb(i, seg.get("text_translated", ""))
                except Exception:
                    pass

    # ── 4. Bail early if nothing to polish ──────────────────────
    if not nsfw_indices:
        if progress_cb:
            progress_cb(total, total)
        return results

    batch_size = max(1, min(batch_size, nsfw_count))
    completed = total - nsfw_count  # non-NSFW already counted

    # ── 5. Batch polish NSFW segments ───────────────────────────
    for batch_start in range(0, nsfw_count, batch_size):
        if cancel_check and cancel_check():
            break

        batch_end = min(batch_start + batch_size, nsfw_count)
        batch_indices = nsfw_indices[batch_start:batch_end]
        batch_segs = [results[idx] for idx in batch_indices]

        # Build delimited input: 原文 + 译文
        parts: list[str] = []
        for idx in batch_indices:
            seg = results[idx]
            original = seg.get("text", "").strip()
            translated = seg.get("text_translated", "").strip()
            if not translated:
                continue
            parts.append(
                f"<<<SEG_{idx}>>>\n"
                f"原文：{original}\n"
                f"译文：{translated}"
            )
        if not parts:
            completed += len(batch_segs)
            if progress_cb:
                progress_cb(completed, total)
            continue

        user_msg = "\n\n".join(parts)

        payload = _json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
        }, ensure_ascii=False).encode("utf-8")

        # ── send to Ollama ───────────────────────────────────
        try:
            req = _urllib.Request(
                chat_url,
                data=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            with _urllib.urlopen(req, timeout=300) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
            raw = body.get("message", {}).get("content", "").strip()
        except Exception as exc:
            for idx in batch_indices:
                results[idx]["text_translated"] = f"[Ollama error: {exc}]"
            completed += len(batch_indices)
            if progress_cb:
                progress_cb(completed, total)
            continue

        # ── parse delimited output ───────────────────────────
        parsed = _parse_batch_output(raw, len(batch_indices),
                                     batch_indices[0] if batch_indices else 0)

        for idx in batch_indices:
            polished = parsed.get(idx, "")
            if polished:
                results[idx]["text_translated"] = polished
            if segment_cb:
                try:
                    segment_cb(idx, results[idx]["text_translated"])
                except Exception:
                    pass

        completed += len(batch_indices)
        if progress_cb:
            progress_cb(completed, total)

    return results


def _parse_batch_output(
    raw: str, expected_count: int, start_index: int,
) -> dict[int, str]:
    """Parse Ollama batch output into a {global_index: polished_text} map.

    Expects format like::

        <<<SEG_0>>>
        你的屌真大…
        <<<SEG_1>>>
        要去了…

    Falls back gracefully: if delimiters are missing, returns the whole
    text as segment ``start_index`` (single-segment fallback).
    """
    import re as _re

    result: dict[int, str] = {}

    # Split on <<<SEG_N>>>
    pattern = r"<<<SEG_(\d+)>>>\s*"
    parts = _re.split(pattern, raw)

    # parts[0] = text before first delimiter, parts[1] = first index,
    # parts[2] = first segment text, parts[3] = second index, ...
    for k in range(1, len(parts) - 1, 2):
        try:
            idx = int(parts[k])
            text = parts[k + 1].strip()
            # Remove any trailing delimiter artifacts
            text = _re.sub(r"<<<SEG_\d+>>>.*$", "", text, flags=_re.DOTALL)
            text = _clean_ollama_output(text)
            if text:
                result[idx] = text
        except (ValueError, IndexError):
            continue

    # Fallback: if no delimiters found and only 1 segment expected,
    # treat entire output as that segment
    if not result and expected_count == 1:
        cleaned = _clean_ollama_output(raw)
        if cleaned:
            result[start_index] = cleaned

    return result
