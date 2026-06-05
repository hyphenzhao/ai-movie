# AI Movie — 项目状态报告

> 更新日期：2026-06-05
> 平台：Ubuntu 24.04 + AMD Ryzen AI MAX+ 395（Radeon 8060S / ROCm 7.2）

---

## 已完成

### 基础架构
- [x] 项目骨架：`ai_movie/` 包结构、`run.py` 入口、集中配置
- [x] 跨平台兼容：从 Windows 迁移到 Linux，修复路径/字体/VLC
- [x] GPU 加速：PyTorch 2.9.1 + ROCm 7.2，Radeon 8060S GPU 推理

### Pipeline 步骤

| 步骤 | 模块 | 状态 | 说明 |
|------|------|:----:|------|
| 切割视频 | `cutter.py` | ✅ | FFmpeg 按 3 分钟切段 + 缩略图 |
| 拆分音轨 | `demuxer.py` | ✅ | 无声视频 + 16kHz 单声道 WAV |
| 转换文字 | `asr.py` | ✅ | 多后端自动选择（ROCm/CUDA/DirectML/CPU） |
| 文本翻译 | `translator.py` | ✅ | Hy-MT1.5-1.8B 本地翻译 + Ollama 可选润色/直翻 |
| 人声分离 | `composer.py` | ✅ | Demucs htdemucs 人声/背景音分离 |
| 人声生成 | `tts.py` | ✅ | CosyVoice 300M-SFT（性别检测）+ CosyVoice2-0.5B（声音克隆） |
| 重新混音 | `composer.py` | ✅ | TTS 人声 + 背景音混音，ducking 压低 |
| 合成音轨 | `app.py` | ✅ | 一键完成人声分离 + 生成 + 混音 |
| 合成视频 | `composer.py` | ✅ | FFmpeg 替换音轨（视频流 copy，音频 AAC 192k） |
| 人物锚定 | `app.py` | 🔲 | 占位 stub（ECAPA 说话人日志已集成到 TTS） |
| 口型匹配 | `app.py` | 🔲 | 占位 stub |

### GUI 功能
- [x] VLC 嵌入式视频播放器（主窗口 + 弹出窗口均带可拖动进度条）
- [x] 步骤工具栏（水平排列，颜色状态：锁定/就绪/运行中/完成/失败）
- [x] 切割视频：缩略图 + 播放按钮
- [x] 拆分音轨：音视频文件列表 + 播放按钮
- [x] 转换文字：语言选择 + 引擎选择（faster-whisper / openai-whisper / 自动）
- [x] 文本翻译：目标语言选择 + **翻译引擎选择**（Hy-MT 本地 / Ollama 直翻 / Hy-MT + Ollama 润色），原文/译文实时对照展示
- [x] 人声分离：人声/背景音文件展示 + 播放按钮
- [x] 人声生成：**三种性别检测算法可选**（pyin 逐段 / 全局 F0 / ECAPA 说话人日志），逐片段展示（原文/译文/时长/性别）+ 播放按钮 + **单片段重新生成**
- [x] 实时进度弹窗（文件进度 + 片段进度）
- [x] 项目保存/加载（`.aimovie.json`），含所有步骤数据恢复
- [x] 可取消的后台任务

### TTS 性别检测（三种算法）
- [x] **pyin 逐段检测**（默认）：对每段音频独立做 pyin F0 分析，自动选择 CosyVoice-300M-SFT 中文男/女声
- [x] **全局 F0 缓存**：整轨预计算 F0 后按片段查表，更稳定
- [x] **ECAPA 说话人日志**：Silero-VAD + SpeechBrain ECAPA-TDNN + 谱聚类，区分不同说话人并分别标注性别

### 翻译引擎（三种模式）
- [x] **Hy-MT 本地**：腾讯 Hy-MT1.5-1.8B 纯本地 GPU 推理，快速准确
- [x] **Ollama 直翻**：通过 Ollama HTTP API（dolphin-mixtral:8x7b），口语/俚语翻译更地道
- [x] **Hy-MT + Ollama 润色**（推荐）：Hy-MT 先翻译，Ollama 对含 NSFW 关键词的片段逐批润色为口语化中文

### 已部署的模型

| 模型 | 用途 | 大小 | 推理引擎 |
|------|------|------|----------|
| `faster-whisper-large-v3` | ASR CPU fallback | 2.9 GB | CTranslate2 int8 |
| `Hy-MT1.5-1.8B` | 文本翻译（日→中/英/韩） | 3.9 GB | PyTorch ROCm fp16 |
| `CosyVoice2-0.5B` | 零样本声音克隆 TTS | 4.6 GB | PyTorch ROCm fp16 |
| `CosyVoice-300M-SFT` | 内置男/女声 TTS（性别检测模式） | 1.2 GB | PyTorch ROCm fp16 |
| Demucs htdemucs | 人声/背景分离 | 80 MB | PyTorch CPU |
| dolphin-mixtral:8x7b | NSFW 俚语翻译/润色（Ollama） | ~40 GB | Ollama 外部服务 |

---

## 待解决问题

### 🔴 高优先级

1. **CosyVoice 2 推理速度慢**
   - 每段 ~28 秒（RTF 2.3x），10 段需 ~5 分钟，100 段需 ~47 分钟
   - 受限于 Flow Matching 解码器（25 步 ODE）+ ROCm MIOpen fallback
   - 可能优化方向：减少 flow steps、INT8 量化、换更快的 TTS 引擎

2. **CosyVoice 2 多线程冲突**
   - 模型内部使用 threading + GPU，从背景线程调用会死循环
   - 已 workaround：强制在主线程运行（`root.after()` 逐段调度）
   - 副作用：每段推理时 GUI 冻结 ~28 秒

3. **网络访问受限**
   - GitHub / HuggingFace 直连不可用
   - 部分 Python 包需要通过清华镜像安装
   - 模型下载需手动配置 HF_ENDPOINT 或用户手动 clone

### 🟡 中优先级

4. **Ollama 翻译延迟高**
   - 每段需单独 HTTP 请求（~2-5 秒/段），大量片段时耗时长
   - `polish_ollama` 的批量模式部分缓解（每批 10 段，仅 NSFW 片段）
   - 可进一步优化：增大 batch size、使用 streaming API

5. **合成音轨声音质量**
   - Demucs 分离后的背景音可能有轻微 artifacts
   - TTS 生成的人声采样率为 24kHz，与原视频可能不一致
   - 混音时简单相加，无音量平衡

6. **依赖版本冲突**
   - Python 3.14 太新，部分包（Coqui TTS、Matcha-TTS）无法正常安装
   - CosyVoice 要求的 `torch==2.3.1` 与已安装的 `torch==2.9.1+rocm7.2` 不一致
   - 通过 symlink / `--no-deps` 等 workaround 绕过

### 🔲 低优先级

7. **人物锚定** — GUI 功能未实现，但 ECAPA 说话人日志已集成到 TTS 步骤
8. **口型匹配** — 未实现，stub 占位
9. **CLI 入口** — 仅有 GUI 模式，无命令行接口
10. **端到端测试** — 缺少自动化测试

---

## 模块清单

| 文件 | 行数 | 功能 | 状态 |
|------|:----:|------|:----:|
| `ai_movie/config.py` | ~180 | 集中配置（路径/字体/模型/Ollama/NSFW） | ✅ |
| `ai_movie/cutter.py` | ~120 | FFmpeg 视频切割 + 缩略图提取 | ✅ |
| `ai_movie/demuxer.py` | ~80 | 音视频分离（无声视频 + 16kHz WAV） | ✅ |
| `ai_movie/asr.py` | ~590 | 语音识别（多后端自动选择） | ✅ |
| `ai_movie/asr_gpu.py` | ~120 | GPU bridge（DirectML，Windows 用） | ✅ |
| `ai_movie/asr_wsl.py` | ~130 | GPU bridge（WSL+ROCm，Windows 用） | ✅ |
| `ai_movie/translator.py` | ~640 | 文本翻译（Hy-MT）+ Ollama 翻译/润色 + NSFW 检测 | ✅ |
| `ai_movie/tts.py` | ~490 | TTS 语音合成 + 三种性别检测算法 + ECAPA 说话人日志 | ✅ |
| `ai_movie/composer.py` | ~200 | 人声分离 + 混音 + 视频合成 | ✅ |
| `ai_movie/project_log.py` | ~130 | 项目日志 + 步骤依赖链 | ✅ |
| `ai_movie/task_manager.py` | ~80 | 线程安全任务注册 | ✅ |
| `ai_movie/utils.py` | ~20 | 文件/JSON 工具 | ✅ |
| `ai_movie/pipeline.py` | ~10 | Pipeline 占位 | 🔲 |
| `ai_movie/gui/app.py` | ~2640 | 主窗口 + 完整 GUI（含翻译引擎选择等） | ✅ |
| `ai_movie/gui/player.py` | ~240 | VLC 嵌入播放器 | ✅ |
| `ai_movie/gui/__init__.py` | ~1 | - | ✅ |

---

## 依赖环境

### 系统依赖
```bash
sudo apt install ffmpeg vlc python3-tk python3-pil.imagetk python3.14-venv
sudo apt install libamdhip64-7  # ROCm HIP runtime
```

### Python 虚拟环境
- 位置：`.venv/`（Python 3.14）
- 安装：`pip install -r requirements.txt`
- GPU：`pip install torch --index-url https://download.pytorch.org/whl/rocm7.2`

### 外部服务（可选）
- **Ollama**：用于 NSFW 俚语翻译/润色，需安装并加载 `dolphin-mixtral:8x7b` 模型
  ```bash
  ollama pull dolphin-mixtral:8x7b
  ```

### 模型文件
```
models/
├── CosyVoice/                  # CosyVoice 源码 (73 MB)
├── CosyVoice2-0.5B/           # 零样本声音克隆 (4.6 GB)
├── CosyVoice-300M-SFT/        # 内置男/女声 TTS (1.2 GB)
├── faster-whisper-large-v3/   # ASR CPU (2.9 GB)
└── Hy-MT1.5-1.8B/             # 文本翻译 (3.9 GB)
```

---

## 下一步建议

1. **优化 TTS 速度**：研究 CosyVoice 2 的 `speed` 参数或减少 flow steps
2. **TTS 降级方案**：如果 CosyVoice 太慢，可尝试 `edge-tts`（免费在线 TTS，速度快但无声音克隆）
3. **完成人物锚定 GUI**：将 ECAPA 说话人日志从 TTS 步骤独立为单独步骤
4. **口型匹配**：研究 Wav2Lip 或类似方案同步口型
5. **添加 CLI**：`python main.py input.mp4 --output output.mp4`
6. **Ollama 性能优化**：增大 polish batch size，探索 streaming API
