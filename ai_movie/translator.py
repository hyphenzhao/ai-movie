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

# Module-level model cache (lazy-loaded, thread-safe).
# Keyed by model path to support multiple Hy-MT generations.
_model_cache: dict[str, object] = {}
_tokenizer_cache: dict[str, object] = {}
_lock = threading.Lock()


def _is_model_downloaded(path: str) -> bool:
    p = Path(path)
    # Hy-MT2 uses safetensors, Hy-MT1.5 also uses safetensors
    return p.is_dir() and (
        (p / "model.safetensors").exists()
        or any(p.glob("*.safetensors"))
    )


def _load_model(model_path: str | None = None):
    """Lazy-load model & tokenizer (thread-safe, idempotent).

    Parameters
    ----------
    model_path:
        Path to the model directory.  Defaults to
        ``config.TRANSLATION_ACTIVE_MODEL`` (Hy-MT2 if available,
        falling back to Hy-MT1.5).
    """
    from ai_movie.config import (
        TRANSLATION_ACTIVE_MODEL,
        TRANSLATION_MODEL_PATH,
        HYMT2_MODEL_PATH,
    )

    if model_path is None:
        model_path = TRANSLATION_ACTIVE_MODEL

    # Already cached?
    cached = _model_cache.get(model_path)
    if cached is not None:
        return

    with _lock:
        if model_path in _model_cache:          # double-checked locking
            return

        if not _is_model_downloaded(model_path):
            # Build helpful download message with mirror fallback
            model_name = Path(model_path).name
            repo_map = {
                "Hy-MT2-30B-A3B": "tencent/Hy-MT2-30B-A3B",
                "Hy-MT2-30B-A3B-FP8": "tencent/Hy-MT2-30B-A3B-FP8",
                "Hy-MT1.5-1.8B": "tencent/Hy-MT1.5-1.8B",
            }
            repo = repo_map.get(model_name, model_name)
            raise FileNotFoundError(
                f"Translation model not found at {model_path}\n"
                f"Download it (choose one):\n"
                f"  git clone https://huggingface.co/{repo} {model_path}\n"
                f"  git clone https://hf-mirror.com/{repo} {model_path}"
            )

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        _tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
            padding_side="left",  # decoder-only models need left-padding
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Hy-MT2-FP8: force FP16 load to avoid FP8 format mismatch between
        # the checkpoint's quant scheme (input_scale/weight_scale) and what
        # the installed transformers expects (weight_scale_inv/activation_scale).
        _model_cache[model_path] = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            _model_cache[model_path] = _model_cache[model_path].to(device)
        _model_cache[model_path].eval()


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
    #    NOTE: "参考前文", "参考：", "示例" are handled by step 4 instead
    #    (whole-line removal) to avoid accidentally trimming the translation
    #    that follows a context section.
    for marker in (
        "解释：", "说明：", "备注：", "注意：", "注：",
        "翻译：", "翻译结果", "译文：",
        "Explanation:", "Note:", "Translation:",
        "（", "(", "【",
    ):
        idx = text.find(marker)
        if idx == -1:
            continue

        # Meta prefix at very start — strip it, keep whatever follows
        if idx == 0 and marker in (
            "解释：", "说明：", "备注：", "注意：", "注：",
            "翻译：", "翻译结果", "译文：",
        ):
            text = text[len(marker):].lstrip("：: ")
            continue

        # Marker preceded by a newline → likely a section header;
        # extract what comes AFTER it as the real translation
        if idx > 0 and text[idx - 1] == "\n" and marker in (
            "翻译：", "译文：", "翻译结果",
        ):
            after = text[idx + len(marker):].lstrip("：: \t")
            if after:
                text = after
                continue

        # Otherwise marker is in the middle of content → trim from it
        if idx > 0:
            text = text[:idx]

    # 4. Strip whole lines that are purely explanatory headers
    text = _re.sub(
        r"(?i)^\s*(Translation|翻译|解释|说明|备注|注意|Note|Explanation"
        r"|参考前文|参考|示例)[：:]\s*.*$",
        "", text, flags=_re.MULTILINE,
    )

    # 4b. Strip echo of the instruction ("将以下日文翻译为...")
    text = _re.sub(
        r"^\s*将以下(日文|日语|文本).*?(翻译|润色).*?[：:]\s*",
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
            # Strip leading numbering: "1. ", "1)", "①", "1、"
            line = _re.sub(
                r"^\s*(?:\d+[\.\)、．]\s*|[①②③④⑤⑥⑦⑧⑨⑩]"
                r"|[一二三四五六七八九十]+[\.\)、．]\s*)",
                "", line,
            ).strip()
            if line:
                return line

    return ""


def translate(
    segments: list[dict],
    target_lang: str = "Chinese",
    src_lang: str = "Japanese",
    model_path: str | None = None,
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
    model_path:
        Path to the Hy-MT model directory.  Defaults to
        ``config.TRANSLATION_ACTIVE_MODEL`` (Hy-MT2 if available).
    progress_cb:
        ``progress_cb(current, total)`` called after each batch completes.
    cancel_check:
        Return ``True`` to abort between batches.

    Returns
    -------
    Same list with ``text_translated`` key added to each segment.
    """
    _load_model(model_path)

    if model_path is None:
        from ai_movie.config import TRANSLATION_ACTIVE_MODEL
        model_path = TRANSLATION_ACTIVE_MODEL

    model = _model_cache[model_path]
    tokenizer = _tokenizer_cache[model_path]

    import torch
    device = model.device
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
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=TRANSLATION_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode generated portion (strip input prompt)
        for j, seg in enumerate(batch_segs):
            prompt_len = inputs["input_ids"][j].size(0)
            gen_ids = outputs[j][prompt_len:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
            global_idx = batch_start + j
            results[global_idx]["text_translated"] = _extract_translation(
                raw, original=seg["text"])

        if progress_cb:
            progress_cb(batch_end, total)

    return results


# ── Ollama translation backend ───────────────────────────────────

# ═══ common Ollama HTTP helper ═══════════════════════════════════

def _call_ollama_chat(
    model: str,
    messages: list[dict],
    base_url: str,
    timeout: int = 600,
) -> str:
    """Send a single chat request to Ollama; return the assistant reply.

    Parameters
    ----------
    model:
        Ollama model name (e.g. ``"dolphin-mixtral:8x22b"``).
    messages:
        List of ``{"role": ..., "content": ...}`` dicts.
    base_url:
        Ollama server URL (e.g. ``"http://localhost:11434"``).
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    The assistant's ``content`` string, stripped.
    """
    import json as _json
    import urllib.request as _urllib

    chat_url = f"{base_url.rstrip('/')}/api/chat"
    payload = _json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
    }, ensure_ascii=False).encode("utf-8")

    req = _urllib.Request(
        chat_url, data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with _urllib.urlopen(req, timeout=timeout) as resp:
        body = _json.loads(resp.read().decode("utf-8"))
    return body.get("message", {}).get("content", "").strip()


# ═══ JSON output validator ═══════════════════════════════════════

def _parse_json_array(
    raw: str, expected_count: int,
) -> tuple[list[str] | None, str | None]:
    """Try to extract and validate a JSON string array from Ollama output.

    Handles models that wrap JSON in markdown code fences or append
    explanatory text before/after the array.

    Parameters
    ----------
    raw:
        Raw model output text.
    expected_count:
        Expected number of strings in the array.

    Returns
    -------
    ``(translations, error)`` — if ``error`` is ``None``,
    ``translations`` is a list of *expected_count* cleaned strings.
    Otherwise ``translations`` is ``None`` and ``error`` describes the
    problem (suitable for feeding back into a retry prompt).
    """
    import json as _json
    import re as _re

    # Strip markdown code fences (```json ... ```)
    raw = _re.sub(r"```(?:json)?\s*", "", raw)
    raw = _re.sub(r"```", "", raw)
    raw = raw.strip()

    # Locate the outermost JSON array
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None, "输出中未找到 JSON 数组（缺少 [ 或 ]）"

    json_str = raw[start:end + 1]

    try:
        parsed = _json.loads(json_str)
    except _json.JSONDecodeError as exc:
        return None, f"JSON 解析错误: {exc}"

    if not isinstance(parsed, list):
        return None, "输出不是 JSON 数组（可能是对象或其他类型）"

    if len(parsed) != expected_count:
        return None, (
            f"数组长度不匹配：期望 {expected_count} 句，实际输出了 {len(parsed)} 句"
        )

    cleaned: list[str] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, str):
            return None, f"第 {i + 1} 项不是字符串（类型: {type(item).__name__}）"
        text = _clean_ollama_output(item)
        if not text:
            return None, f"第 {i + 1} 句翻译为空或只有注释"
        cleaned.append(text)

    return cleaned, None


# ═══ translate_ollama — batch JSON translation + retry + context ═

def translate_ollama(
    segments: list[dict],
    target_lang: str = "Chinese",
    src_lang: str = "Japanese",
    model: str | None = None,
    base_url: str | None = None,
    system_prompt: str | None = None,
    batch_size: int | None = None,
    max_retries: int | None = None,
    context_sentences: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    segment_cb: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Translate transcript segments using Ollama with JSON batch + retry.

    Sends segments in batches (default 15) as a JSON string array, asks
    the model to return a matching JSON array of translations, and
    validates the output.  On validation failure the model is retried
    with error feedback up to *max_retries* times.

    Cross-batch context (previous translations) is injected to improve
    dialogue coherence.

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
        Override the default system prompt.
    batch_size:
        Segments per Ollama request.  Defaults to ``config.OLLAMA_BATCH_SIZE``.
    max_retries:
        Max JSON-validation retries per batch.  Defaults to
        ``config.OLLAMA_MAX_RETRIES``.
    context_sentences:
        Number of preceding-batch sentence pairs to include as context.
        Defaults to ``config.OLLAMA_CONTEXT_SENTENCES``.  Set to 0 to disable.
    progress_cb:
        ``progress_cb(completed, total)`` called after each batch.
    segment_cb:
        ``segment_cb(index, translated_text)`` called per segment.
    cancel_check:
        Return ``True`` to abort between batches.

    Returns
    -------
    Same list with ``text_translated`` key added to each segment.
    """
    import json as _json

    from ai_movie.config import (
        OLLAMA_BASE_URL,
        OLLAMA_BATCH_SIZE,
        OLLAMA_CONTEXT_SENTENCES,
        OLLAMA_MAX_RETRIES,
        OLLAMA_MODEL,
        OLLAMA_NSFW_SYSTEM_PROMPT,
        OLLAMA_SAKURA_CONCURRENCY,
        OLLAMA_SAKURA_TIMEOUT,
        OLLAMA_SAKURA_TRANSLATE_PROMPT,
    )

    if model is None:
        model = OLLAMA_MODEL
    if base_url is None:
        base_url = OLLAMA_BASE_URL
    if system_prompt is None:
        system_prompt = OLLAMA_NSFW_SYSTEM_PROMPT
    if batch_size is None:
        batch_size = OLLAMA_BATCH_SIZE
    if max_retries is None:
        max_retries = OLLAMA_MAX_RETRIES
    if context_sentences is None:
        context_sentences = OLLAMA_CONTEXT_SENTENCES

    # ── SakuraLLM auto-detection ──────────────────────────────────
    _sakura = _is_sakura_model(model)
    if _sakura and system_prompt == OLLAMA_NSFW_SYSTEM_PROMPT:
        # User didn't override the prompt → use Sakura-specialised one
        system_prompt = OLLAMA_SAKURA_TRANSLATE_PROMPT

    total = len(segments)
    results: list[dict] = list(segments)

    # Rolling context buffer: (original, translated) pairs
    ctx_buf: list[tuple[str, str]] = []

    for batch_start in range(0, total, batch_size):
        if cancel_check and cancel_check():
            break

        batch_end = min(batch_start + batch_size, total)
        batch_indices = list(range(batch_start, batch_end))

        # Collect non-empty texts + their global indices
        non_empty: list[tuple[int, str]] = []
        for idx in batch_indices:
            text = results[idx].get("text", "").strip()
            if text:
                non_empty.append((idx, text))
            else:
                results[idx]["text_translated"] = ""

        if not non_empty:
            if progress_cb:
                progress_cb(batch_end, total)
            continue

        ne_indices = [idx for idx, _ in non_empty]
        ne_texts = [t for _, t in non_empty]
        ne_count = len(ne_texts)

        # ── Build context prefix ──────────────────────────────────
        ctx_prefix = ""
        if ctx_buf and context_sentences > 0:
            lines = []
            for orig, trans in ctx_buf[-context_sentences:]:
                lines.append(f'  "{orig}" → "{trans}"')
            if lines:
                ctx_prefix = (
                    "前面对话的翻译参考（已翻译完毕，不需要再翻译）：\n"
                    + "\n".join(lines) + "\n\n"
                )

        # ── SakuraLLM: per-segment plain-text (no JSON) ───────────
        if _sakura:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _do_one(idx: int, txt: str) -> tuple[int, str, str | None]:
                """Return (index, translation, error_or_None)."""
                try:
                    # Build a single, clean user message (no double-wrapping!)
                    if ctx_buf and context_sentences > 0:
                        ctx_lines = []
                        for orig, trans in ctx_buf[-context_sentences:]:
                            ctx_lines.append(f"「{orig}」→「{trans}」")
                        if ctx_lines:
                            user_msg = (
                                "前文翻译：\n" + "\n".join(ctx_lines)
                                + f"\n\n将以下日文翻译为口语化中文：\n{txt}"
                            )
                        else:
                            user_msg = f"将以下日文翻译为口语化中文：\n{txt}"
                    else:
                        user_msg = f"将以下日文翻译为口语化中文：\n{txt}"

                    raw = _call_ollama_chat(
                        model,
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        base_url,
                        timeout=OLLAMA_SAKURA_TIMEOUT,
                    )
                    return idx, _clean_ollama_output(raw), None
                except Exception as exc:
                    return idx, "", str(exc)

            with ThreadPoolExecutor(max_workers=OLLAMA_SAKURA_CONCURRENCY) as ex:
                futs = {
                    ex.submit(_do_one, idx, txt): idx
                    for idx, txt in non_empty
                }
                for fut in as_completed(futs):
                    if cancel_check and cancel_check():
                        ex.shutdown(wait=False, cancel_futures=True)
                        break
                    idx, trans, error = fut.result()
                    if error:
                        results[idx]["text_translated"] = (
                            f"[翻译失败] {error}"
                        )
                    else:
                        results[idx]["text_translated"] = trans
                        ctx_buf.append((results[idx].get("text", ""), trans))
                    if segment_cb:
                        try:
                            segment_cb(idx, results[idx].get("text_translated", ""))
                        except Exception:
                            pass
            # Keep ctx_buf bounded
            limit = max(context_sentences, 1) * 3
            if len(ctx_buf) > limit:
                ctx_buf = ctx_buf[-limit:]

        else:
            # ── Generic model: JSON batch mode ────────────────────
            input_json = _json.dumps(ne_texts, ensure_ascii=False)

            base_user_msg = (
                f"{ctx_prefix}"
                f"将以下 {ne_count} 句 {src_lang} 翻译为口语化 {target_lang}。\n"
                f"严格输出一个 JSON 字符串数组（共 {ne_count} 个元素），不要任何解释：\n"
                f"{input_json}"
            )

            # ── Retry loop ────────────────────────────────────────
            success = False
            last_error = ""
            user_msg = base_user_msg

            for attempt in range(max_retries):
                try:
                    raw = _call_ollama_chat(
                        model,
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        base_url,
                    )
                except Exception as exc:
                    last_error = f"HTTP 请求失败: {exc}"
                    continue

                translations, error = _parse_json_array(raw, ne_count)
                if error is None:
                    for idx, trans in zip(ne_indices, translations):
                        results[idx]["text_translated"] = trans
                        ctx_buf.append((results[idx].get("text", ""), trans))
                    limit = max(context_sentences, 1) * 3
                    if len(ctx_buf) > limit:
                        ctx_buf = ctx_buf[-limit:]
                    success = True
                    break

                last_error = error
                user_msg = (
                    f"{ctx_prefix}"
                    f"⚠️ 上次输出被拒绝：{error}\n\n"
                    f"请重新将以下 {ne_count} 句 {src_lang} 翻译为口语化 {target_lang}。\n"
                    f"只输出一个纯 JSON 数组（{ne_count} 个字符串），不要 markdown 代码块、"
                    f"不要解释、不要编号：\n"
                    f"{input_json}"
                )

            if not success:
                fail_msg = (
                    f"[翻译失败] Ollama 模型 {model} 在 {max_retries} 次尝试后"
                    f"仍无法返回有效 JSON：{last_error}。\n"
                    f"建议：更换翻译模型，或修改系统提示词。"
                )
                for idx in batch_indices:
                    if not results[idx].get("text_translated"):
                        results[idx]["text_translated"] = fail_msg

            # ── Progress + segment callbacks ──────────────────────
            if segment_cb:
                for idx in batch_indices:
                    try:
                        segment_cb(idx, results[idx].get("text_translated", ""))
                    except Exception:
                        pass

        # ── Batch-level progress ────────────────────────────────────
        if progress_cb:
            progress_cb(batch_end, total)

    return results


# ═══ Ollama model list ════════════════════════════════════════════

def _is_sakura_model(model_name: str) -> bool:
    """Return True if *model_name* matches the Sakura-Qwen family."""
    if not model_name:
        return False
    lower = model_name.lower()
    return "sakura" in lower and "qwen" in lower


def _translate_segment_sakura(
    model: str,
    text: str,
    system_prompt: str,
    base_url: str,
    timeout: int,
    src_lang: str = "日文",
    target_lang: str = "口语化中文",
) -> str:
    """Translate a single segment with SakuraLLM — plain text, no JSON.

    SakuraLLM is fine-tuned for natural-language JP→ZH output.  Forcing
    JSON formatting fights its training and causes near-100% parse
    failures.  This helper sends one segment as a plain-text instruction
    and returns the raw translated text.
    """
    user_msg = f"将以下{src_lang}翻译为{target_lang}：\n{text}"
    raw = _call_ollama_chat(
        model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        base_url,
        timeout=timeout,
    )
    return _clean_ollama_output(raw)


def fetch_ollama_models(base_url: str | None = None) -> list[str]:
    """Fetch the list of available model names from an Ollama server.

    Parameters
    ----------
    base_url:
        Ollama server URL.  Defaults to ``config.OLLAMA_BASE_URL``.

    Returns
    -------
    List of model name strings (e.g. ``["qwen3:14b", "dolphin-mixtral:8x22b"]``).
    Returns an empty list on any error (connection refused, timeout, etc.).
    """
    import json as _json
    import urllib.request as _urllib

    from ai_movie.config import OLLAMA_BASE_URL

    if base_url is None:
        base_url = OLLAMA_BASE_URL

    try:
        tags_url = f"{base_url.rstrip('/')}/api/tags"
        req = _urllib.Request(tags_url)
        with _urllib.urlopen(req, timeout=10) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
        models = body.get("models", [])
        return sorted(
            [m["name"] for m in models if isinstance(m, dict) and "name" in m]
        )
    except Exception:
        return []


# ═══ polish_ollama — JSON batch polish + retry ═══════════════════

def polish_ollama(
    segments: list[dict],
    batch_size: int | None = None,
    model: str | None = None,
    base_url: str | None = None,
    system_prompt: str | None = None,
    max_retries: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    segment_cb: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Polish Hy-MT translated Chinese text with Ollama for NSFW style.

    Sends segments in JSON batches.  Input is ``[{"original": ...,
    "translated": ...}, ...]`` and the model is asked to return a
    matching JSON array of polished strings.  JSON validation with
    retry (up to *max_retries*) guards against malformed output.

    Only segments whose original text matches NSFW keywords are sent to
    Ollama; the rest keep their Hy-MT translation unchanged.

    Parameters
    ----------
    segments:
        List of dicts with ``text`` (original) and ``text_translated``
        (Hy-MT Chinese) already set.
    batch_size:
        Segments per Ollama request.  Defaults to ``config.OLLAMA_BATCH_SIZE``.
    model / base_url / system_prompt:
        Defaults from ``config.py``.
    max_retries:
        Max retries per batch on JSON validation failure.
        Defaults to ``config.OLLAMA_MAX_RETRIES``.
    progress_cb:
        ``progress_cb(completed, total)`` called after each batch.
    segment_cb:
        ``segment_cb(index, polished_text)`` called per segment.
    cancel_check:
        Return ``True`` to abort between batches.

    Returns
    -------
    Same list with ``text_translated`` replaced by polished version.
    """
    import json as _json
    import re as _re

    from ai_movie.config import (
        OLLAMA_BASE_URL,
        OLLAMA_BATCH_SIZE,
        OLLAMA_MAX_RETRIES,
        OLLAMA_MODEL,
        OLLAMA_NSFW_KEYWORDS,
        OLLAMA_POLISH_SYSTEM_PROMPT,
        OLLAMA_SAKURA_CONCURRENCY,
        OLLAMA_SAKURA_POLISH_PROMPT,
        OLLAMA_SAKURA_TIMEOUT,
    )

    if model is None:
        model = OLLAMA_MODEL
    if base_url is None:
        base_url = OLLAMA_BASE_URL
    if system_prompt is None:
        system_prompt = OLLAMA_POLISH_SYSTEM_PROMPT
    if batch_size is None:
        batch_size = OLLAMA_BATCH_SIZE
    if max_retries is None:
        max_retries = OLLAMA_MAX_RETRIES

    # ── SakuraLLM auto-detection ──────────────────────────────────
    _sakura = _is_sakura_model(model)
    if _sakura and system_prompt == OLLAMA_POLISH_SYSTEM_PROMPT:
        system_prompt = OLLAMA_SAKURA_POLISH_PROMPT

    total = len(segments)
    results: list[dict] = list(segments)

    # ── 1. Pre-clean ChatML tokens from Hy-MT output ────────────
    for seg in results:
        t = seg.get("text_translated", "")
        if t:
            t = _re.sub(
                r"<\|im_start[\|>]*|<\|im_end[\|>]*|</?im_start>|</?im_end>",
                "", t,
            ).strip()
            seg["text_translated"] = t

    # ── 2. Classify: which segments need NSFW polish ────────────
    nsfw_indices: list[int] = []
    keywords_lower = [kw.lower() for kw in OLLAMA_NSFW_KEYWORDS]
    for i, seg in enumerate(results):
        text = seg.get("text", "").lower()
        if any(kw in text for kw in keywords_lower):
            nsfw_indices.append(i)

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

    nsfw_count = len(nsfw_indices)
    batch_size = max(1, min(batch_size, nsfw_count))
    completed = total - nsfw_count  # non-NSFW already counted

    # ── 5. Batch polish NSFW segments ───────────────────────────
    for batch_start in range(0, nsfw_count, batch_size):
        if cancel_check and cancel_check():
            break

        batch_end = min(batch_start + batch_size, nsfw_count)
        batch_indices = nsfw_indices[batch_start:batch_end]

        # Build JSON input: [{"original": ..., "translated": ...}, ...]
        input_items = []
        pos_to_idx: dict[int, int] = {}
        pos = 0
        for idx in batch_indices:
            seg = results[idx]
            original = seg.get("text", "").strip()
            translated = seg.get("text_translated", "").strip()
            if not translated:
                continue
            input_items.append({
                "original": original,
                "translated": translated,
            })
            pos_to_idx[pos] = idx
            pos += 1

        if not input_items:
            completed += len(batch_indices)
            if progress_cb:
                progress_cb(completed, total)
            continue

        item_count = len(input_items)

        # ── SakuraLLM: per-segment plain-text polish (no JSON) ──
        if _sakura:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _polish_one(idx: int, original: str, translated: str
                           ) -> tuple[int, str, str | None]:
                """Return (index, polished_text, error_or_None)."""
                try:
                    user_msg = (
                        f"润色以下机器翻译为口语化中文：\n"
                        f"原文：{original}\n"
                        f"机翻：{translated}\n\n"
                        f"只输出润色后的中文。"
                    )
                    raw = _call_ollama_chat(
                        model,
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        base_url,
                        timeout=OLLAMA_SAKURA_TIMEOUT,
                    )
                    return idx, _clean_ollama_output(raw), None
                except Exception as exc:
                    return idx, "", str(exc)

            with ThreadPoolExecutor(max_workers=OLLAMA_SAKURA_CONCURRENCY) as ex:
                futs = {
                    ex.submit(_polish_one, pos_to_idx.get(pos, -1),
                              item.get("original", ""),
                              item.get("translated", "")): pos
                    for pos, item in enumerate(input_items)
                }
                for fut in as_completed(futs):
                    if cancel_check and cancel_check():
                        ex.shutdown(wait=False, cancel_futures=True)
                        break
                    idx, trans, error = fut.result()
                    if idx < 0:
                        continue
                    if error:
                        results[idx]["text_translated"] = (
                            results[idx].get("text_translated", "")
                            + f"\n[润色失败] {error}"
                        )
                    else:
                        results[idx]["text_translated"] = trans
                    if segment_cb:
                        try:
                            segment_cb(idx, results[idx].get("text_translated", ""))
                        except Exception:
                            pass
        else:
            # ── Generic model: JSON batch polish ────────────────
            input_json = _json.dumps(input_items, ensure_ascii=False)

            base_user_msg = (
                f"以下是 {item_count} 句需要润色的字幕（JSON 数组）。\n"
                f"对每句的 translated 字段做口语化润色，严格输出 {item_count} 个字符串的 JSON 数组：\n"
                f"{input_json}"
            )

            success = False
            last_error = ""
            user_msg = base_user_msg

            for attempt in range(max_retries):
                try:
                    raw = _call_ollama_chat(
                        model,
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        base_url,
                    )
                except Exception as exc:
                    last_error = f"HTTP 请求失败: {exc}"
                    continue

                polished_list, error = _parse_json_array(raw, item_count)
                if error is None:
                    for pos, text in enumerate(polished_list):
                        idx = pos_to_idx.get(pos)
                        if idx is not None:
                            results[idx]["text_translated"] = text
                    success = True
                    break

                last_error = error
                user_msg = (
                    f"⚠️ 上次输出被拒绝：{error}\n\n"
                    f"请重新润色以下 {item_count} 句。只输出 {item_count} 个字符串的 "
                    f"纯 JSON 数组，不要任何其他内容：\n"
                    f"{input_json}"
                )

            if not success:
                fail_msg = (
                    f"[润色失败] Ollama 模型 {model} 在 {max_retries} 次尝试后"
                    f"仍无法返回有效 JSON：{last_error}。"
                    f"建议：更换模型或修改润色提示词。"
                )
                for idx in batch_indices:
                    results[idx]["text_translated"] = (
                        results[idx].get("text_translated", "") + f"\n{fail_msg}"
                    )

            if segment_cb:
                for idx in batch_indices:
                    try:
                        segment_cb(idx, results[idx].get("text_translated", ""))
                    except Exception:
                        pass
        completed += len(batch_indices)
        if progress_cb:
            progress_cb(completed, total)

    return results
