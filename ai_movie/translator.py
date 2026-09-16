"""Offline text translation using Tencent Hy-MT1.5-1.8B (MIT license).

Lazy-loads the model on first use. GPU preferred (ROCm/CUDA), CPU fallback.
Also supports Ollama-based translation for colloquial / NSFW content.

ChatML prompt template::

    <|im_start|>user
    Translate Japanese to Chinese:
    {segment_text}<|im_end|>
    <|im_start|>assistant

"""

import re
import sys
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

def _looks_like_context_echo(text: str) -> bool:
    """Whether the model returned a "「原文」→「译文」" pair instead of a translation.

    Completion-style models continue any pattern they are shown; when the
    prompt contained context in that shape they reproduce it verbatim.
    """
    import re as _re
    return bool(_re.search(r"[「\"'']..*?[」\"'']\s*(→|->|=>)\s*[「\"'']", text or ""))


def _clean_ollama_output(raw: str) -> str:
    """Aggressively strip LLM commentary from Ollama output.

    Adult-film translations should be pure dialogue — no explanations,
    no parenthetical notes, no metadata.  This function strips anything
    that looks like model-generated commentary.
    """
    import re as _re
    text = raw.strip()

    # 0. A context echo carries no translation — drop it so the caller can
    #    retry or leave the segment empty rather than emit "「A」→「B」".
    if _looks_like_context_echo(text):
        return ""

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


# ── Memory-exclusive engine routing ──────────────────────────────
#
# This box has 122 GB of unified memory.  Hy-MT2-30B is 57 GB resident and
# gpt-oss-120b is 88 GB — they cannot coexist, and neither can Hy-MT2 plus
# dolphin-mixtral:8x22b (80 GB), which is exactly what the "hy-mt2+polish"
# engine has been asking for.  Every engine switch must therefore evict the
# previous one first.

def unload_local_models(model_path: str | None = None) -> None:
    """Free transformers models held in ``_model_cache``.

    Pass a *model_path* to drop just that one, or ``None`` to drop all.
    Safe to call when nothing is loaded.
    """
    import gc

    with _lock:
        keys = [model_path] if model_path else list(_model_cache.keys())
        for k in keys:
            _model_cache.pop(k, None)
            _tokenizer_cache.pop(k, None)

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:                                   # noqa: BLE001
        pass


def ollama_loaded(base_url: str | None = None) -> list[dict]:
    """Return the models Ollama currently holds resident (``GET /api/ps``)."""
    import json as _json
    import urllib.request as _urllib

    from ai_movie.config import OLLAMA_BASE_URL

    base_url = base_url or OLLAMA_BASE_URL
    try:
        with _urllib.urlopen(f"{base_url.rstrip('/')}/api/ps", timeout=10) as resp:
            return _json.loads(resp.read().decode("utf-8")).get("models", [])
    except Exception:                                   # noqa: BLE001
        return []


def ollama_unload(model: str, base_url: str | None = None) -> None:
    """Evict *model* from Ollama's memory (``keep_alive: 0``)."""
    import json as _json
    import urllib.request as _urllib

    from ai_movie.config import OLLAMA_BASE_URL

    base_url = base_url or OLLAMA_BASE_URL
    payload = _json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    req = _urllib.Request(
        f"{base_url.rstrip('/')}/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with _urllib.urlopen(req, timeout=120):
            pass
    except Exception:                                   # noqa: BLE001
        pass


def free_gpu_for_local_work(base_url: str | None = None,
                            log_cb: Callable[[str], None] | None = None) -> None:
    """Evict every resident Ollama model and transformers model.

    GPU memory here is *unified system memory*: an 83 GB Ollama model leaves
    about 10 MiB for anything else, so CosyVoice, MuseTalk and CodeFormer all
    fail with OOM while an LLM is loaded.  Call this before any local GPU
    stage (TTS, lip-sync, face restore).
    """
    for m in ollama_loaded(base_url):
        name = m.get("name") or m.get("model") or ""
        if name:
            (log_cb or (lambda s: print(f"[engine] {s}", flush=True)))(
                f"evicting ollama model {name} to free GPU")
            ollama_unload(name, base_url)
    unload_local_models()


def _free_gb() -> float:
    """Available system memory in GB (0.0 if unreadable)."""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:                                   # noqa: BLE001
        pass
    return 0.0


def _model_size_gb(model: str) -> float:
    from ai_movie.config import OLLAMA_MODEL_SIZE_GB
    return float(OLLAMA_MODEL_SIZE_GB.get(model, 0.0))


class exclusive_engine:
    """Context manager guaranteeing one heavyweight engine is resident.

    ``kind="ollama"`` evicts every transformers model, plus any *other*
    Ollama model when the target is large.  ``kind="local"`` evicts all
    Ollama models before the transformers model loads.

    Usage::

        with exclusive_engine("ollama", ollama_model=OLLAMA_GPTOSS_MODEL):
            ...
    """

    def __init__(self, kind: str, *, ollama_model: str | None = None,
                 base_url: str | None = None,
                 log_cb: Callable[[str], None] | None = None):
        self.kind = kind
        self.ollama_model = ollama_model
        self.base_url = base_url
        self.log_cb = log_cb

    def _log(self, msg: str) -> None:
        if self.log_cb:
            self.log_cb(msg)
        else:
            print(f"[engine] {msg}", flush=True)

    def __enter__(self):
        from ai_movie.config import OLLAMA_EXCLUSIVE_ABOVE_GB

        before = _free_gb()
        if self.kind == "ollama":
            unload_local_models()
            target = self.ollama_model or ""
            if _model_size_gb(target) >= OLLAMA_EXCLUSIVE_ABOVE_GB:
                for m in ollama_loaded(self.base_url):
                    name = m.get("name") or m.get("model") or ""
                    if name and name != target:
                        self._log(f"evicting ollama model {name}")
                        ollama_unload(name, self.base_url)
        elif self.kind == "local":
            for m in ollama_loaded(self.base_url):
                name = m.get("name") or m.get("model") or ""
                if name:
                    self._log(f"evicting ollama model {name}")
                    ollama_unload(name, self.base_url)
        after = _free_gb()
        self._log(f"enter {self.kind}"
                  f"{'(' + self.ollama_model + ')' if self.ollama_model else ''}: "
                  f"free {before:.0f}→{after:.0f} GB")
        return self

    def __exit__(self, exc_type, exc, tb):
        self._log(f"exit {self.kind}: free {_free_gb():.0f} GB")
        return False


# ── Ollama translation backend ───────────────────────────────────

# ═══ common Ollama HTTP helper ═══════════════════════════════════

def _call_ollama_chat(
    model: str,
    messages: list[dict],
    base_url: str,
    timeout: int = 600,
    options: dict | None = None,
    think: bool | None = None,
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
    # Bound the generation.  Without num_predict Ollama decodes until the
    # context limit: measured on this project, Sakura-14b produced 4 099
    # tokens for a single ~20-token subtitle line, and dolphin-mixtral
    # degenerated into a repeated "MMFMMF..." pattern.  A cap plus a repeat
    # penalty turns those hangs into (at worst) a rejected batch that retries.
    opts = {"num_predict": 512, "temperature": 0.2, "repeat_penalty": 1.15}
    if options:
        opts.update(options)
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": opts,
    }
    if think is not None:
        body["think"] = think
    payload = _json.dumps(body, ensure_ascii=False).encode("utf-8")

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


# ── v2: context-aware, glossary-pinned translation ──────────────────
#
# The v1 path translated each line in isolation with the prompt
# "Translate Japanese to Chinese:\n{text}" and TRANSLATION_CONTEXT_SEGMENTS
# pinned to 0.  That is why the reference run produced 「镰鼬」 for the
# performer's name and rendered 「目ぐらいかな」 as 「大概就是眼睛吧」: no
# surrounding dialogue, no speaker, no terminology.  Everything below feeds
# the model the conversation instead of a fragment.

def build_context_block(
    segments: list[dict],
    translations: list[str],
    idx: int,
    *,
    n_before: int | None = None,
    n_after: int | None = None,
) -> tuple[str, str]:
    """Return ``(already_translated_block, upcoming_source_block)``."""
    from ai_movie.config import TRANSLATION_CTX_AFTER, TRANSLATION_CTX_BEFORE

    n_before = TRANSLATION_CTX_BEFORE if n_before is None else n_before
    n_after = TRANSLATION_CTX_AFTER if n_after is None else n_after

    before = []
    for j in range(max(0, idx - n_before), idx):
        zh = (translations[j] or "").strip() if j < len(translations) else ""
        if not zh:
            continue
        before.append(f"  {_speaker_tag(segments[j])}「{segments[j].get('text','')}」"
                      f" → 「{zh}」")
    after = []
    for j in range(idx, min(len(segments), idx + n_after)):
        after.append(f"  {_speaker_tag(segments[j])}「{segments[j].get('text','')}」")
    return "\n".join(before), "\n".join(after)


def _speaker_tag(seg: dict) -> str:
    """``[S0♀]`` style tag so the model keeps pronouns/register consistent."""
    spk = seg.get("speaker") or ""
    if not spk:
        return ""
    g = seg.get("gender") or seg.get("tts_gender") or ""
    mark = {"female": "女", "male": "男"}.get(g, "")
    return f"[{spk}{mark}]"


def _llm_translate_batches(
    segments: list[dict],
    *,
    model: str,
    base_url: str,
    glossary: dict | None,
    batch_size: int,
    scene_hint: str | None,
    max_retries: int,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[str]:
    """Translate with a chat LLM, feeding it a rolling window of context."""
    from ai_movie.config import TRANSLATION_SCENE_HINT
    from ai_movie.glossary import (
        format_for_prompt, protect_terms, restore_terms,
    )

    scene_hint = scene_hint or TRANSLATION_SCENE_HINT
    out: list[str] = [""] * len(segments)
    total = len(segments)

    system = (
        scene_hint + "\n\n"
        "规则：\n"
        "1. 逐句翻译，输入几句就输出几句，不要合并或拆分。\n"
        "2. 口语化。该用俚语、俗语、语气词就用，不要翻译腔、不要书面语。\n"
        "3. 同一说话人的自称、称呼、语气要前后一致（我会用 [S0女]/[S1男] 标出说话人）。\n"
        "4. 保留语气词、笑声、感叹；不要补充原文没有的内容。\n"
        "5. 遇到术语表里的词，必须按表中的译法翻译。\n"
        "6. 严格只输出一个 JSON 字符串数组，不要编号、不要解释、不要任何其他文字。"
    )

    done = 0
    for start in range(0, total, batch_size):
        if cancel_check and cancel_check():
            break
        batch = segments[start:start + batch_size]
        texts = [(s.get("text") or "").strip() for s in batch]
        pins_per = [{} for _ in batch]
        keep = [i for i, t in enumerate(texts) if t]
        if not keep:
            done += len(batch)
            if progress_cb:
                progress_cb(done, total)
            continue

        before, after = build_context_block(segments, out, start)
        gl = format_for_prompt(glossary or {}, texts)

        parts = []
        if gl:
            parts.append(f"【术语表（必须遵守）】{gl}")
        if before:
            parts.append("【前文（已翻译，仅供参考，不要重复输出）】\n" + before)
        parts.append(
            f"【待翻译】共 {len(keep)} 句，按顺序输出 {len(keep)} 个元素：\n"
            + "\n".join(f"{n}. {_speaker_tag(batch[i])}{texts[i]}"
                        for n, i in enumerate(keep)))
        if after:
            parts.append("【后文（仅供理解语境，禁止翻译）】\n" + after)
        user = "\n\n".join(parts)

        translations = None
        last_err = None
        for attempt in range(max_retries):
            try:
                raw = _call_ollama_chat(
                    model,
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}]
                    + ([{"role": "user",
                         "content": f"⚠️ 上次输出被拒绝：{last_err}。"
                                    f"请重新输出 {len(keep)} 个元素的 JSON 数组。"}]
                       if last_err else []),
                    base_url, timeout=1800,
                    options={"num_predict": 120 * max(4, len(keep)),
                             "temperature": 0.2},
                    think=False)
            except Exception as exc:                    # noqa: BLE001
                last_err = f"请求失败：{exc}"
                continue
            translations, last_err = _parse_json_array(raw, len(keep))
            if translations is not None:
                break

        if translations is None:
            print(f"[translate] batch @{start} failed after {max_retries} "
                  f"attempts: {last_err}", file=sys.stderr)
            translations = ["" for _ in keep]

        for n, i in enumerate(keep):
            out[start + i] = restore_terms(translations[n], pins_per[i])

        done += len(batch)
        if progress_cb:
            progress_cb(done, total)

    return out


def _sakura_translate(
    segments: list[dict],
    *,
    model: str,
    base_url: str,
    glossary: dict | None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[str]:
    """Per-segment translation with SakuraLLM (it cannot do JSON batches).

    Context is snapshotted per batch rather than mutated live: the previous
    implementation appended to a shared ``ctx_buf`` from inside
    ``as_completed`` while four workers ran, so every worker saw a different,
    race-dependent context and reruns were not reproducible.
    """
    from concurrent.futures import ThreadPoolExecutor
    from ai_movie.config import (
        OLLAMA_SAKURA_CONCURRENCY, OLLAMA_SAKURA_TIMEOUT,
        OLLAMA_SAKURA_TRANSLATE_PROMPT,
    )
    from ai_movie.glossary import (
        format_for_prompt, protect_terms, restore_terms,
    )

    out: list[str] = [""] * len(segments)
    total = len(segments)
    ctx: list[tuple[str, str]] = []
    done = 0
    chunk = max(1, OLLAMA_SAKURA_CONCURRENCY)

    for start in range(0, total, chunk):
        if cancel_check and cancel_check():
            break
        batch = list(range(start, min(total, start + chunk)))
        ctx_snapshot = list(ctx[-3:])          # frozen for the whole batch

        def _one(i: int) -> tuple[int, str]:
            txt = (segments[i].get("text") or "").strip()
            if not txt:
                return i, ""
            pins: dict[str, str] = {}
            # Sakura is a completion-style translation model.  Context must be
            # given as real prior chat turns, not as an inline
            # "「原文」→「译文」" block: given that pattern in the user message
            # it *continues the pattern* instead of translating, and 6 of 34
            # segments came back as literal "「A」→「B」" pairs.  Prior turns
            # are unambiguous — the model can only answer the last one.
            msgs = [{"role": "system",
                     "content": OLLAMA_SAKURA_TRANSLATE_PROMPT}]
            for o, t in ctx_snapshot[-2:]:
                msgs.append({"role": "user",
                             "content": f"将以下日文翻译为口语化中文：\n{o}"})
                msgs.append({"role": "assistant", "content": t})

            gl = format_for_prompt(glossary or {}, [txt])
            head = f"固定译名：{gl}\n" if gl else ""
            msgs.append({"role": "user",
                         "content": f"{head}将以下日文翻译为口语化中文：\n{txt}"})
            try:
                raw = _call_ollama_chat(
                    model, msgs, base_url, timeout=OLLAMA_SAKURA_TIMEOUT,
                    options={"num_predict": max(64, len(txt) * 4),
                             "temperature": 0.1})
                return i, restore_terms(_clean_ollama_output(raw), pins)
            except Exception as exc:                    # noqa: BLE001
                print(f"[translate] sakura segment {i} failed: {exc}",
                      file=sys.stderr)
                return i, ""

        with ThreadPoolExecutor(max_workers=chunk) as ex:
            for i, txt in ex.map(_one, batch):
                out[i] = txt

        for i in batch:                                 # extend context in order
            src = (segments[i].get("text") or "").strip()
            if src and out[i]:
                ctx.append((src, out[i]))
        ctx = ctx[-12:]

        done += len(batch)
        if progress_cb:
            progress_cb(done, total)

    return out


def _llm_polish(
    segments: list[dict],
    drafts: list[str],
    *,
    model: str,
    base_url: str,
    glossary: dict | None,
    batch_size: int = 10,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[str]:
    """Rewrite a literal draft into natural, colloquial, coherent Chinese."""
    from ai_movie.config import TRANSLATION_SCENE_HINT
    from ai_movie.glossary import format_for_prompt

    out = list(drafts)
    total = len(segments)
    system = (
        TRANSLATION_SCENE_HINT + "\n\n"
        "我会给你日语原文和一版机器翻译草稿。请逐句润色：\n"
        "1. 改成自然的中文口语，该用俚语、俗语、语气词就用；去掉翻译腔。\n"
        "2. 修正机翻的误译、漏译、残留日文。\n"
        "3. 保持上下文连贯：称呼、自称、语气在整段对话里一致。\n"
        "4. 不要合并或拆分句子，输入几句就输出几句。\n"
        "5. 术语表里的词必须按表中译法。\n"
        "严格只输出一个 JSON 字符串数组，不要任何解释。"
    )

    done = 0
    for start in range(0, total, batch_size):
        if cancel_check and cancel_check():
            break
        idxs = [i for i in range(start, min(total, start + batch_size))
                if (segments[i].get("text") or "").strip()]
        if not idxs:
            done += batch_size
            if progress_cb:
                progress_cb(min(done, total), total)
            continue

        before, _ = build_context_block(segments, out, start)
        gl = format_for_prompt(glossary or {},
                               [segments[i].get("text", "") for i in idxs])
        parts = []
        if gl:
            parts.append(f"【术语表（必须遵守）】{gl}")
        if before:
            parts.append("【前文（已定稿）】\n" + before)
        parts.append("【待润色】共 %d 句：\n%s" % (
            len(idxs),
            "\n".join(f"{n}. {_speaker_tag(segments[i])}原文：{segments[i].get('text','')}"
                      f"\n   草稿：{out[i]}" for n, i in enumerate(idxs))))
        user = "\n\n".join(parts)

        polished = None
        last_err = None
        for _ in range(3):
            try:
                raw = _call_ollama_chat(
                    model,
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    base_url, timeout=1800,
                    options={"num_predict": 120 * max(4, len(idxs)),
                             "temperature": 0.2},
                    think=False)
            except Exception as exc:                    # noqa: BLE001
                last_err = str(exc)
                continue
            polished, last_err = _parse_json_array(raw, len(idxs))
            if polished is not None:
                break

        if polished is None:
            print(f"[translate] polish batch @{start} kept draft: {last_err}",
                  file=sys.stderr)
        else:
            for n, i in enumerate(idxs):
                if polished[n].strip():
                    out[i] = polished[n].strip()

        done += batch_size
        if progress_cb:
            progress_cb(min(done, total), total)

    return out


def _hymt_translate(
    segments: list[dict],
    *,
    target_lang: str,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[str]:
    """Literal draft from the local Hy-MT model (loaded exclusively)."""
    with exclusive_engine("local"):
        res = translate(
            [dict(s) for s in segments], target_lang=target_lang,
            progress_cb=progress_cb, cancel_check=cancel_check,
        )
        out = [(r.get("text_translated") or "").strip() for r in res]
    unload_local_models()
    return out


# Engine table: (draft_fn_key, polish_model_key or None)
_FLAG_HINTS = {
    "F1_pronoun": "译文里出现了人称代词（你/我/他/她…），但日文原文没有主语；请结合上下文确认指代，没有依据就删掉或改正",
    "F2_question": "译文的疑问/陈述语气和原文不一致",
    "F3_kana": "译文残留日文假名，请译成中文",
}
# F4_length stays a metric only: on v3.0.0 those lines were ASR garbage, which
# no amount of Chinese rewriting can repair.


def _polish_flagged(
    segments: list[dict],
    drafts: list[str],
    *,
    model: str,
    base_url: str,
    glossary: dict | None,
    report: list[dict] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[str]:
    """Re-check only the lines a film-independent rule flags as suspicious.

    Polishing every line lets an LLM "improve" the ~80 % that were already
    right (LLM post-editing over-corrects on low-error input), so each
    request carries one flagged line, the reason it was flagged, the four
    finished lines before it and the next two Japanese lines — the look-
    ahead a fragment needs.  Output is one plain line, never JSON: weaker
    models degenerate on structured output.  A candidate replaces the draft
    only if it passes the same fidelity guards as compact_translation, so
    this pass can decline to act but cannot invent content.
    """
    from ai_movie.config import POLISH_CTX_AFTER, POLISH_CTX_BEFORE, POLISH_TIMEOUT
    from ai_movie.glossary import format_for_prompt
    from ai_movie.units import flag_line, polish_edit_ok

    out = [d or "" for d in drafts]
    for i, seg in enumerate(segments):
        if cancel_check and cancel_check():
            break
        ja = (seg.get("text") or "").strip()
        draft = out[i].strip()
        if not ja or not draft:
            continue
        flags = [f for f in flag_line(ja, draft) if f in _FLAG_HINTS]
        if not flags:
            continue
        row = {"idx": i, "flags": ",".join(flags), "ja": ja, "draft": draft,
               "candidate": "", "status": "error"}
        before = "\n".join(
            f"「{(segments[j].get('text') or '').strip()}」→「{out[j]}」"
            for j in range(max(0, i - POLISH_CTX_BEFORE), i) if out[j])
        after = "\n".join(
            f"「{(segments[j].get('text') or '').strip()}」"
            for j in range(i + 1, min(len(segments), i + 1 + POLISH_CTX_AFTER)))
        gl = format_for_prompt(glossary or {}, [ja])
        pins = [v["zh"] for k, v in (glossary or {}).items()
                if v.get("zh") and v["zh"] in draft]
        prompt = (
            (f"【术语表（必须遵守）】\n{gl}\n" if gl else "")
            + (f"【前文（已定稿）】\n{before}\n" if before else "")
            + (f"【后文（仅供理解语境，不要翻译）】\n{after}\n" if after else "")
            + f"【待校对】\n原文：{ja}\n草稿：{draft}\n"
            + "【疑点】\n" + "\n".join(f"- {_FLAG_HINTS[f]}" for f in flags if f in _FLAG_HINTS)
            + "\n\n规则：优先直接删掉没有依据的人称代词，其余每个字原样保留；"
              "实在不能删就换成正确的代词。不要改写句子，不要补充原文没有的信息。\n"
              "示例：原文「好きなのかも。」草稿「我可能喜欢上你了。」→ 输出「我可能喜欢上了。」\n"
              "如果草稿确实没问题，就原样输出草稿。只输出这一句中文，不要输出前文、后文或解释。"
        )
        try:
            raw = _call_ollama_chat(
                model,
                [{"role": "system", "content": "你是日译中字幕校对。只输出一行中文译文。"},
                 {"role": "user", "content": prompt}],
                base_url, timeout=POLISH_TIMEOUT, think=False,
                options={"num_predict": max(96, len(draft) * 4), "temperature": 0.0,
                         "seed": 7})
            cand = next(iter(_clean_ollama_output(raw or "").strip().splitlines()), "").strip()
            # One stricter retry: the guard only accepts deletions and
            # function-word swaps, so a rejected rewrite usually just means
            # the model reworded when it should have deleted.
            if cand and not polish_edit_ok(draft, cand, flags):
                raw2 = _call_ollama_chat(
                    model,
                    [{"role": "system", "content": "你是日译中字幕校对。只输出一行中文译文。"},
                     {"role": "user", "content": prompt
                      + "\n\n注意：上一次的改写改动了太多字。这次只允许删除多余的人称代词，"
                        "其他字符一个都不要改。"}],
                    base_url, timeout=POLISH_TIMEOUT, think=False,
                    options={"num_predict": max(96, len(draft) * 4), "temperature": 0.0,
                             "seed": 11})
                cand2 = next(iter(_clean_ollama_output(raw2 or "").strip().splitlines()), "").strip()
                if cand2 and polish_edit_ok(draft, cand2, flags):
                    cand = cand2
        except Exception as exc:                        # noqa: BLE001
            row["candidate"] = f"{type(exc).__name__}: {exc}"
            if report is not None:
                report.append(row)
            continue
        row["candidate"] = cand
        ok = polish_edit_ok(draft, cand, flags) and all(t in cand for t in pins)
        if not cand or cand == draft:
            row["status"] = "unchanged"
        elif ok:
            out[i] = cand
            row["status"] = "accepted"
        else:
            row["status"] = "rejected"
        if report is not None:
            report.append(row)
    return out


TRANSLATE_ENGINES = {
    "sakura":         ("sakura", None),
    "sakura+gptoss":  ("sakura", "gptoss"),
    "gptoss":         ("gptoss", None),
    "hy-mt2":         ("hymt2", None),
    "hy-mt2+gptoss":  ("hymt2", "gptoss"),
    "hy-mt2+sakura":  ("hymt2", "sakura"),
    "sakura+qwen":    ("sakura", "qwen"),
}

ENGINE_LABELS = {
    "sakura":        "Sakura-14B 直译（快，日→中口语专精）",
    "sakura+gptoss": "Sakura 直译 + gpt-oss-120B 上下文润色（推荐）",
    "gptoss":        "gpt-oss-120B 直译（上下文最强，最慢）",
    "hy-mt2":        "Hy-MT2-30B 直译（原方案）",
    "hy-mt2+gptoss": "Hy-MT2 直译 + gpt-oss-120B 润色",
    "hy-mt2+sakura": "Hy-MT2 直译 + Sakura 润色",
    "sakura+qwen":   "Sakura 直译 + Qwen3.6 可疑句上下文校对（推荐）",
}


def enforce_glossary(
    segments: list[dict],
    translations: list[str],
    glossary: dict[str, dict],
    *,
    model: str | None = None,
    base_url: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> list[str]:
    """Normalise pinned terms in already-translated lines.

    Three cheaper approaches were measured and all failed on Sakura (see
    ``glossary.protect_terms``): instructing the model is ignored, rewriting
    the source makes it re-transliterate, and bracketed placeholders survive
    in isolation but get dropped once the real prompt is assembled — the
    performer's name came back as 卡娜 / 加奈 / 卡恩娜 / 小蓝华.

    Rewriting one finished sentence is a much easier task than translating
    with constraints, and it only runs on the few lines that actually contain
    a pinned term (9 of 128 on the reference video).  The rewrite is accepted
    only if it really contains the pin, so this can never make a line worse.
    """
    from ai_movie.config import GLOSSARY_ENFORCE_MODEL, OLLAMA_BASE_URL

    if not glossary:
        return translations

    model = model or GLOSSARY_ENFORCE_MODEL
    base_url = base_url or OLLAMA_BASE_URL
    out = list(translations)
    fixed = attempted = 0

    for i, (seg, zh) in enumerate(zip(segments, translations)):
        src = (seg.get("text") or "")
        zh = (zh or "").strip()
        if not src or not zh:
            continue
        need = [(ja, v["zh"]) for ja, v in glossary.items()
                if v.get("zh") and _glossary_hit(src, ja) and v["zh"] not in zh]
        if not need:
            continue
        attempted += 1
        pins = "；".join(f"{ja} 必须译作「{t}」" for ja, t in need)
        prompt = (
            f"下面这句中文译文里的专有名词译法不对。参考日文原文：{src}\n"
            f"要求：{pins}。\n"
            f"请只把名字改正，其余措辞保持不变，只输出改正后的整句，不要解释。\n"
            f"待改正：{zh}"
        )
        try:
            raw = _call_ollama_chat(
                model,
                [{"role": "system",
                  "content": "你是中文字幕校对助手。只输出改正后的一句中文，不要解释。"},
                 {"role": "user", "content": prompt}],
                base_url, timeout=300,
                options={"num_predict": max(64, len(zh) * 3), "temperature": 0.0})
        except Exception:                               # noqa: BLE001
            continue
        cand = _clean_ollama_output(raw)
        # Accept only if it actually applied the pin and stayed a sentence.
        if cand and all(t in cand for _, t in need) and \
                0.4 <= len(cand) / max(len(zh), 1) <= 2.5:
            out[i] = cand
            fixed += 1

    if progress_cb and attempted:
        progress_cb(f"术语校正：{fixed}/{attempted} 句已统一")
    return out


def _glossary_hit(src: str, ja: str) -> bool:
    """Whether *ja* occurs in *src* as a real term (see glossary._term_pattern)."""
    from ai_movie.glossary import _term_pattern
    return bool(_term_pattern(ja).search(src))


_KANA_RE = re.compile(r"[぀-ゟ゠-ヿ]")


def _visible_len(text: str) -> int:
    from ai_movie.composer import _visible_chars
    return _visible_chars(text)


def compact_translation(
    ja: str,
    zh: str,
    max_chars: int,
    *,
    glossary: dict[str, dict] | None = None,
    context: list[str] | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_tries: int = 2,
) -> str | None:
    """Rewrite one finished Chinese line to at most *max_chars* visible chars.

    Used by the compact stage on lines whose synthesized duration cannot fit
    their time slot.  Same single-sentence rewrite channel as
    :func:`enforce_glossary` (the drafting model, Sakura, ignores
    constraints).  Returns ``None`` when no acceptable rewrite came back;
    the caller then keeps the full line and lets fit/truncation handle it.

    Accepted only if the candidate is non-empty, has no kana, is really
    shorter than the original, is within budget, keeps every pinned
    glossary term the source contains, and is not absurdly short (≥ 30 % of
    the original) — so a rewrite can drop words but never the sentence.
    """
    from ai_movie.config import COMPACT_MODEL, OLLAMA_BASE_URL

    model = model or COMPACT_MODEL
    base_url = base_url or OLLAMA_BASE_URL
    zh = (zh or "").strip()
    ja = (ja or "").strip()
    if not zh:
        return None
    n_orig = _visible_len(zh)
    if n_orig <= max_chars:
        return None

    pins: list[str] = []
    if glossary and ja:
        pins = [v["zh"] for k, v in glossary.items()
                if v.get("zh") and _glossary_hit(ja, k)]
    pin_txt = ("；人名/术语必须保留：" + "、".join(pins)) if pins else ""
    ctx_txt = ""
    if context:
        ctx_txt = "上下文（仅供理解）：\n" + "\n".join(context[-3:]) + "\n"

    # The instruct model does not count characters reliably (asked for 4 it
    # returns 6), so the budget is a *steer*: the second try asks for less,
    # and the shortest candidate that passes the fidelity checks wins even
    # if it is still over budget — a shorter faithful line always beats the
    # full one, and the fit stage absorbs the remainder.
    best: str | None = None
    best_n = n_orig
    budget = int(max_chars)
    for attempt in range(max_tries):
        ask = budget if attempt == 0 else max(2, int(budget * 0.7))
        prompt = (
            f"{ctx_txt}"
            f"下面这句中文配音台词太长，念不完。请把它改写得更短：保留原意和口语感，"
            f"去掉可有可无的词，不超过 {ask} 个汉字{pin_txt}。\n"
            f"日文原文（参考）：{ja}\n"
            f"待压缩：{zh}\n"
            f"只输出压缩后的一句中文，不要解释，不要引号。"
        )
        try:
            raw = _call_ollama_chat(
                model,
                [{"role": "system",
                  "content": "你是中文配音台词精简助手。只输出压缩后的一句中文，不要解释。"},
                 {"role": "user", "content": prompt}],
                base_url, timeout=300,
                options={"num_predict": max(48, ask * 3), "temperature": 0.0})
        except Exception:                               # noqa: BLE001
            break
        cand = _clean_ollama_output(raw).strip().strip("「」『』\"'“”")
        cand = cand.split("\n")[0].strip()
        n = _visible_len(cand)
        if (cand and not _KANA_RE.search(cand) and n < best_n
                and n >= max(2, int(0.3 * n_orig))
                and all(p in cand for p in pins)
                and _char_overlap(cand, zh) >= 0.3):
            best, best_n = cand, n
            if n <= max_chars:
                break
    return best


def _char_overlap(cand: str, orig: str) -> float:
    """Fraction of the candidate's CJK characters that occur in *orig*.

    A cheap fidelity check for the compact rewrite: a faithful shortening
    reuses the original's words (0.5–1.0), whereas a hallucinated line
    shares almost nothing (measured 0.14 on one that invented content).
    """
    cjk = [c for c in cand if "一" <= c <= "鿿"]
    if not cjk:
        return 0.0
    pool = set(orig)
    return sum(1 for c in cjk if c in pool) / len(cjk)


def translate_segments(
    segments: list[dict],
    *,
    engine: str = "sakura+gptoss",
    glossary: dict | None = None,
    target_lang: str = "Chinese",
    base_url: str | None = None,
    batch_size: int = 8,
    max_retries: int = 4,
    scene_hint: str | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    report: list[dict] | None = None,
) -> list[str]:
    """Translate *segments* with one of :data:`TRANSLATE_ENGINES`.

    Returns one Chinese string per segment (empty string where the source
    was empty or the engine failed).  Engines are run inside
    :class:`exclusive_engine` so a 57 GB local model and an 88 GB Ollama
    model can never be resident at the same time on a 122 GB box.
    """
    from ai_movie.config import (
        OLLAMA_BASE_URL, OLLAMA_GPTOSS_MODEL, OLLAMA_POLISH_MODEL, OLLAMA_SAKURA_MODEL,
    )

    if engine not in TRANSLATE_ENGINES:
        raise ValueError(f"unknown translation engine: {engine!r} "
                         f"(known: {', '.join(TRANSLATE_ENGINES)})")
    base_url = base_url or OLLAMA_BASE_URL
    draft_key, polish_key = TRANSLATE_ENGINES[engine]
    models = {"gptoss": OLLAMA_GPTOSS_MODEL, "sakura": OLLAMA_SAKURA_MODEL,
              "qwen": OLLAMA_POLISH_MODEL}

    # ── draft ───────────────────────────────────────────────────────
    if draft_key == "hymt2":
        drafts = _hymt_translate(segments, target_lang=target_lang,
                                 progress_cb=progress_cb,
                                 cancel_check=cancel_check)
    else:
        m = models[draft_key]
        with exclusive_engine("ollama", ollama_model=m, base_url=base_url):
            if draft_key == "sakura":
                drafts = _sakura_translate(
                    segments, model=m, base_url=base_url, glossary=glossary,
                    progress_cb=progress_cb, cancel_check=cancel_check)
            else:
                drafts = _llm_translate_batches(
                    segments, model=m, base_url=base_url, glossary=glossary,
                    batch_size=batch_size, scene_hint=scene_hint,
                    max_retries=max_retries,
                    progress_cb=progress_cb, cancel_check=cancel_check)

    if not polish_key:
        return enforce_glossary(segments, drafts, glossary or {},
                                base_url=base_url)

    # ── polish ──────────────────────────────────────────────────────
    m = models[polish_key]
    with exclusive_engine("ollama", ollama_model=m, base_url=base_url):
        if polish_key == "qwen":
            polished = _polish_flagged(segments, drafts, model=m, base_url=base_url,
                                       glossary=glossary, report=report,
                                       cancel_check=cancel_check)
        else:
            polished = _llm_polish(segments, drafts, model=m, base_url=base_url,
                                   glossary=glossary, progress_cb=progress_cb,
                                   cancel_check=cancel_check)
    return enforce_glossary(segments, polished, glossary or {},
                            base_url=base_url)
