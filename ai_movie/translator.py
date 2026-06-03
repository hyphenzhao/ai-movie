"""Offline text translation using Tencent Hy-MT1.5-1.8B (MIT license).

Lazy-loads the model on first use. GPU preferred (ROCm/CUDA), CPU fallback.

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


def _extract_translation(raw: str) -> str:
    """Strip ChatML tokens, HTML tags & trailing artifacts from model output."""
    import re
    text = raw.strip()
    # Remove HTML tags
    text = re.sub(r"<br\s*/?>", "", text)
    # Remove any ChatML tokens wherever they appear
    text = re.sub(r"</?im_start>", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    # Remove trailing end-of-turn fragments
    for suffix in (_CHATML_END, _CHATML_ASSISTANT, _CHATML_USER):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    # If a ChatML token appears mid-text, take only what's before it
    for marker in ("<|im_start|>", "<|im_end|>", "</im_start>"):
        if marker in text:
            text = text.split(marker)[0].strip()
    return text


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
            results[global_idx]["text_translated"] = _extract_translation(raw)

        if progress_cb:
            progress_cb(batch_end, total)

    return results
