# AI视频换语言（Video Dubbing）架构方案

> 📐 **本文档是架构设计参考文档**，编写于项目启动前的技术选型阶段。实际实现可能与方案有差异。
> 当前项目状态见 [project-status.md](project-status.md)。

## 项目目标

加载任意视频，将配音替换为中文，同时：
- 保持原始音色不变
- 尽可能修正口型以匹配新的中文音轨

## 整体架构概览

```
输入视频 → [1.解复用] → 视频流 + 音频流
              ↓
         [2.ASR语音识别] → 带时间戳的字幕文本
              ↓
         [3.机器翻译] → 中文字幕文本
              ↓
         [4.TTS语音合成+音色克隆] → 中文音频
              ↓
         [5.音频对齐] → 对齐后的中文音轨
              ↓
         [6.口型修正] → 最终合成视频
```

---

## Step 1: 视频加载与解复用（Demux）

### 目标

将输入视频分离为独立的视频流（无声）和音频流。

### 技术要点

- 使用 FFmpeg 进行解复用，提取视频流和音频流
- 视频流保留原始编码格式（H.264/H.265/VP9 等）
- 音频流转为标准 PCM WAV（16kHz/24kHz 单声道，方便后续 AI 处理）
- 同时提取 FPS、分辨率、编码参数等元信息，后续合成时需要

### 推荐开源方案

| 方案 | 说明 |
|------|------|
| **[FFmpeg](https://github.com/FFmpeg/FFmpeg)** | 核心工具，几乎所有视频/音频处理的第一步都靠它 |
| **[PyAV](https://github.com/PyAV-Org/PyAV)** | FFmpeg 的 Python 绑定，适合程序化调用 |
| **[moviepy](https://github.com/Zulko/moviepy)** | 更高层的 Python 视频编辑库，底层也是 FFmpeg |

### 示例命令

```bash
# 拆分音视频
ffmpeg -i input.mp4 -an -c:v copy video_only.mp4   # 仅视频（无声）
ffmpeg -i input.mp4 -vn -ar 16000 -ac 1 audio.wav   # 仅音频（16kHz单声道）
```

---

## Step 2: 语音识别（ASR / Speech-to-Text）

### 目标

将音频转为带精确时间戳的文字（词级或句子级时间戳）。

### 技术要点

- 需要一个支持**词级时间戳（word-level timestamps）**的 ASR 模型
- 对多语种视频需要自动语言检测
- 输出格式建议：`[{text, start_ms, end_ms, confidence}, ...]`

### 推荐开源方案

| 方案 | 说明 | 推荐度 |
|------|------|--------|
| **[WhisperX](https://github.com/m-bain/whisperX)** | 基于 OpenAI Whisper + 强制对齐（forced alignment），提供词级时间戳，支持多语言。**强烈推荐** | ⭐⭐⭐⭐⭐ |
| **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** | Whisper 的 CTranslate2 加速版，4x 更快，内存更少 | ⭐⭐⭐⭐ |
| **[FunASR](https://github.com/modelscope/FunASR)** | 阿里达摩院出品，中文场景优化，支持标点、VAD、时间戳 | ⭐⭐⭐⭐⭐ |
| **[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)** | 阿里最新多语言 ASR，情感识别，高精度 | ⭐⭐⭐⭐ |
| **Whisper (OpenAI)** | 原始方案，large-v3 模型效果最好 | ⭐⭐⭐ |

### 建议方案

**WhisperX**（泛语言）+ **FunASR**（中文专优），可以按视频语言动态选择。

---

## Step 3: 文本翻译（NMT）

### 目标

将源语言文本翻译为中文，同时保留时间轴信息。

### 技术要点

- 需要处理口语化表达、俚语、文化梗
- 字幕翻译要控制长度（不能太长，否则口型对不上）
- 最好保留标点符号，方便后续 TTS 处理断句
- 可以结合上下文做篇章级翻译，而不是逐句翻译

### 推荐开源方案

| 方案 | 说明 | 推荐度 |
|------|------|--------|
| **[SeamlessM4T](https://github.com/facebookresearch/seamless_communication)** | Meta 的统一多模态翻译模型，支持 S2ST/S2TT/T2TT，100+ 语言 | ⭐⭐⭐⭐⭐ |
| **[NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb)** | Meta 的 No Language Left Behind 翻译模型，200 种语言 | ⭐⭐⭐⭐ |
| **[argos-translate](https://github.com/argosopentech/argos-translate)** | 轻量离线翻译引擎，支持 30+ 语言 | ⭐⭐⭐ |
| **[LibreTranslate](https://github.com/LibreTranslate/LibreTranslate)** | 自托管翻译 API，基于 Argos/OpenNMT | ⭐⭐⭐ |
| **LLM（GPT-4/Claude/Qwen）** | 通过 API 调用大模型翻译，口语、上下文理解最好，但成本高 | ⭐⭐⭐⭐ |

### 建议方案

离线优先用 **SeamlessM4T v2** 大模型（S2TT 模式）；对质量要求极高时接入 LLM API 做第二轮润色。

---

## Step 4: 语音合成 + 音色克隆（TTS + Voice Cloning）

### 目标

用中文合成新的配音，同时保持原说话人的音色、语速、情感。

**这是整个项目最有技术挑战的环节。**

### 技术要点

- **零样本音色克隆（Zero-shot Voice Cloning）**: 不需要针对每个说话人做微调
- **情感保留**: 源音频中的喜怒哀乐要在合成时尽量传递
- **语速控制**: 中文和英文表达同样的意思通常长度不同，需要做语速调整
- 如果视频是多说话人，还需要先做**说话人分离（Speaker Diarization）**

### 推荐开源方案

| 方案 | 说明 | 推荐度 |
|------|------|--------|
| **[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)** | 目前中文 TTS + 少样本音色克隆最火方案，效果极好 | ⭐⭐⭐⭐⭐ |
| **[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)** | 阿里出品，零样本音色克隆，支持情感/方言/多语言，中文最优之一 | ⭐⭐⭐⭐⭐ |
| **[F5-TTS](https://github.com/SWivid/F5-TTS)** | 基于 Flow Matching 的零样本 TTS，音色克隆自然度高 | ⭐⭐⭐⭐ |
| **[Fish-Speech](https://github.com/fishaudio/fish-speech)** | 开源多语言 TTS，支持零样本克隆 | ⭐⭐⭐⭐ |
| **[XTTS-v2](https://github.com/coqui-ai/TTS)** | Coqui TTS 的 XTTS-v2，支持 17 种语言跨语言克隆 | ⭐⭐⭐⭐ |
| **[ChatTTS](https://github.com/2noise/ChatTTS)** | 对话场景 TTS，口语化好，支持韵律控制 | ⭐⭐⭐ |
| **[OpenVoice](https://github.com/myshell-ai/OpenVoice)** | MIT 开源的即时声音克隆，粒度控制精细 | ⭐⭐⭐ |

### 建议方案

**CosyVoice** 作为中文主力引擎，音色克隆能力极强；**GPT-SoVITS** 作为备选/补充。

> **SOTA 动态**: 这个领域迭代极快，建议关注 CosyVoice 2/3 和 GPT-SoVITS 的最新版本。

---

## Step 4.5: 说话人分离（Speaker Diarization）

多说话人视频必须在此步骤前完成。

| 方案 | 说明 |
|------|------|
| **[pyannote-audio](https://github.com/pyannote/pyannote-audio)** | 基于 PyTorch 的说话人分离，HuggingFace 上有预训练模型 |
| **[wespeaker](https://github.com/wenet-e2e/wespeaker)** | 国产说话人识别/验证工具包 |
| **[whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization)** | WhisperX + Pyannote 的集成方案 |

---

## Step 5: 音频对齐（Audio Alignment）

### 目标

将生成的中文音频在时间轴上拟合到原始音轨中。

### 技术要点

- 中文翻译后的时长和原文时长可能不一致
- 需要在保持时间轴基本对齐的前提下做语速调整
- 静音段处理（原视频的沉默部分如何处理）
- 背景音/环境音的保留（只替换人声，保留背景音）

### 推荐开源方案

| 方案 | 说明 |
|------|------|
| **[UVR (Ultimate Vocal Remover)](https://github.com/Anjok07/ultimatevocalremovergui)** | 人声分离，提取/去除人声，保留背景音，基于 MDX-Net/Demucs |
| **[Demucs](https://github.com/facebookresearch/demucs)** | Meta 的音源分离模型，效果好 |
| **[MFA (Montreal Forced Aligner)](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)** | 音素级文本-音频对齐，做强制对齐 |
| **[sox](https://github.com/sox/sox)** / **[pydub](https://github.com/jiaaro/pydub)** | 音频处理（拉伸/压缩/淡入淡出/混音） |
| **[rubberband](https://github.com/breakfastquay/rubberband)** | 高质量的音频时间拉伸和变调库 |

### 处理流程

```
原音频 → UVR/Demucs 人声分离 → 人声轨 + 背景音轨
合成中文音频 → rubberband 语速调整 → 对齐后的人声轨
对齐后人声轨 + 背景音轨 → 混音 → 最终音轨
```

---

## Step 6: 口型修正（Lip Sync / Talking Face）

### 目标

根据新的中文音频修改视频中人物的口型。

**这是整个项目最难的部分**，目前仍处于学术前沿。

### 技术要点

- 需要检测人脸 + 提取面部关键点/landmarks
- 根据音频驱动口型变化（Audio-Driven Talking Face Generation）
- 只修改嘴部区域，保留其他面部特征不变
- 需要处理多人、侧脸、遮挡等复杂场景

### 推荐开源方案

| 方案 | 说明 | 推荐度 |
|------|------|--------|
| **[Wav2Lip](https://github.com/Rudrabha/Wav2Lip)** | 音频驱动的口型同步，业界标杆方案，效果稳定 | ⭐⭐⭐⭐⭐ |
| **[Video-Retalking](https://github.com/OpenTalker/video-retalking)** | 输入视频+音频，输出口型同步视频，中文支持好 | ⭐⭐⭐⭐⭐ |
| **[MuseTalk](https://github.com/TMElyralab/MuseTalk)** | 腾讯音乐出品，实时高质量口型驱动，支持 Nvidia/CUDA | ⭐⭐⭐⭐ |
| **[EchoMimic](https://github.com/BadToBest/EchoMimic)** | 音频+面部关键点驱动，口型更自然 | ⭐⭐⭐⭐ |
| **[AniPortrait](https://github.com/Zejun-Yang/AniPortrait)** | 腾讯出品，音频驱动肖像动画 | ⭐⭐⭐ |
| **[SadTalker](https://github.com/OpenTalker/SadTalker)** | 生成3D运动系数驱动面部，表情丰富 | ⭐⭐⭐ |

### 建议方案

第一版用 **Wav2Lip**（成熟稳定），后续迭代到 **Video-Retalking**（整体效果更好，特别是中文场景）。

---

## 汇总：整体技术栈

```
┌─────────────────────────────────────────────────────────┐
│  Step 1   解复用      FFmpeg / PyAV                     │
│  Step 2   ASR         WhisperX / FunASR                 │
│  Step 3   翻译         SeamlessM4T / LLM API            │
│  Step 4   TTS 克隆    CosyVoice / GPT-SoVITS            │
│  Step 4.5 说话人分离   pyannote-audio                    │
│  Step 5   音频对齐     UVR + rubberband + pydub         │
│  Step 6   口型修正     Wav2Lip / Video-Retalking         │
└─────────────────────────────────────────────────────────┘
```

---

## 建议的开发路线图

| 阶段 | 目标 | 关键里程碑 |
|------|------|-----------|
| **Phase 1 MVP** | 跑通基础流程 | FFmpeg + WhisperX + 简单翻译 + Edge TTS + 不修口型，直接硬合成 |
| **Phase 2 翻译增强** | 翻译质量升级 | 接入 SeamlessM4T 或 LLM 翻译 |
| **Phase 3 音色克隆** | 中文配音 | 集成 CosyVoice，实现音色克隆 TTS |
| **Phase 4 口型修正** | 完整效果 | 集成 Wav2Lip / Video-Retalking |
| **Phase 5 多说话人** | 复杂视频 | 说话人分离 + 逐人合成 + 场景融合 |

---

## 关键技术风险

1. **音色克隆的跨语言效果**: 大多数 TTS 模型在源语言→中文的跨语言音色克隆上效果还不完美，音色可能会有一定失真
2. **口型修正的稳定性**: Wav2Lip 在侧脸、遮挡、低分辨率场景下效果下降明显
3. **端到端延迟**: 全流程计算量巨大（尤其是口型修正），一个 10 分钟视频可能需要数小时处理
4. **GPU 需求**: Step 4-6 都需要 GPU，建议至少 RTX 4090 24GB
