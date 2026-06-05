# AI Movie — 模型下载说明

本项目使用多个预训练 AI 模型。模型文件较大（总计 ~14 GB），不直接包含在 Git 仓库中。
请按以下说明逐一下载。

> **网络提示**: 如果 GitHub / HuggingFace 直连不可用，可设置镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

---

## 模型清单

| 模型 | 用途 | 大小 | 存放路径 |
|------|------|------|----------|
| `faster-whisper-large-v3` | 语音识别 (ASR CPU) | 2.9 GB | `models/faster-whisper-large-v3/` |
| `Hy-MT1.5-1.8B` | 文本翻译 | 3.9 GB | `models/Hy-MT1.5-1.8B/` |
| `CosyVoice2-0.5B` | 语音合成 (声音克隆) | 4.6 GB | `models/CosyVoice2-0.5B/` |
| `CosyVoice-300M-SFT` | 语音合成 (性别匹配) | 2.5 GB | `models/CosyVoice-300M-SFT/` |
| `CosyVoice` (源码) | TTS 引擎源码 | 已含在仓库 | `models/CosyVoice/` |
| `speechbrain-ecapa` | 说话人识别 | 85 MB | `models/speechbrain-ecapa/` |
| `silero-vad` | 语音活动检测 | 自动下载 | `models/torch_hub/` |

---

## 下载命令

```bash
# 进入项目根目录
cd ai-movie

# ── 1. faster-whisper-large-v3（ASR） ──
# 方式 A：用 faster-whisper 自动下载
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8', download_root='models')"
# 方式 B：手动 clone
git clone https://huggingface.co/Systran/faster-whisper-large-v3 models/faster-whisper-large-v3

# ── 2. Hy-MT1.5-1.8B（翻译） ──
git clone https://huggingface.co/tencent/Hy-MT1.5-1.8B models/Hy-MT1.5-1.8B

# ── 3. CosyVoice2-0.5B（TTS 声音克隆） ──
# 从 ModelScope 或 HuggingFace 下载
git clone https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B models/CosyVoice2-0.5B
# 镜像: git clone https://hf-mirror.com/FunAudioLLM/CosyVoice2-0.5B models/CosyVoice2-0.5B

# ── 4. CosyVoice-300M-SFT（TTS 性别匹配，可选） ──
git clone https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT models/CosyVoice-300M-SFT
# 镜像: git clone https://hf-mirror.com/FunAudioLLM/CosyVoice-300M-SFT models/CosyVoice-300M-SFT

# ── 5. CosyVoice 源码（已在仓库中，无需下载） ──
# models/CosyVoice/ 已包含在 Git 中

# ── 6. SpeechBrain ECAPA（说话人识别） ──
# 下载预训练权重
git clone https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb models/speechbrain-ecapa

# ── 7. silero-vad（VAD，首次运行自动下载） ──
# 无需手动操作，首次使用时会自动缓存到 models/torch_hub/
```

---

## 最小安装（快速开始）

如果只需要基本功能，至少下载前 2 个模型：

| 最小配置 | 需要的模型 |
|----------|-----------|
| ASR + 翻译 | `faster-whisper-large-v3` + `Hy-MT1.5-1.8B` |
| + TTS | 再加 `CosyVoice2-0.5B`（推荐）或 `CosyVoice-300M-SFT` |

---

## 验证安装

```bash
# 检查所有模型目录是否就绪
ls models/faster-whisper-large-v3/model.bin     && echo "✅ faster-whisper"
ls models/Hy-MT1.5-1.8B/model.safetensors        && echo "✅ Hy-MT"
ls models/CosyVoice2-0.5B/llm.pt                 && echo "✅ CosyVoice2"
ls models/speechbrain-ecapa/embedding_model.ckpt  && echo "✅ ECAPA"
```
