# AI Movie — AI 视频换语言（Video Dubbing）

> **一句话**：加载任意视频，用 AI 将配音替换为中文，同时保留原始音色和背景音。

> 📊 v2 质量升级说明见 [Documentation/v2-quality-upgrade.md](Documentation/v2-quality-upgrade.md)；v1 历史文档在 [Documentation/v1/](Documentation/v1/)

---

## 特性

- 🎬 **完整 Pipeline**：视频切割 → 音轨分离 → 语音识别 → 文本翻译 → 人声分离 → 语音合成 → 混音 → 合成视频
- 🧠 **AI 引擎全本地**：语音识别（faster-whisper）、翻译（Hy-MT1.5-1.8B）、TTS（CosyVoice2/CosyVoice-300M-SFT）
- 🌐 **GPU 加速**：AMD ROCm / NVIDIA CUDA / DirectML，自动检测并选择最佳后端
- 🎯 **智能性别检测**：三种算法可选（pyin 逐段 / 全局 F0 / ECAPA 说话人日志），自动匹配男/女声
- 🔞 **NSFW 优化**：Ollama 集成（dolphin-mixtral），支持日语俚语/成人内容的精准口语化翻译
- 🖥️ **完整 GUI**：Tkinter + VLC 嵌入式播放器，步骤工具栏，实时进度，项目保存/加载

---

## 工作流

```
原始视频 → 切割片段 → 拆分音轨 → ASR 语音识别 → 翻译（Hy-MT + Ollama 润色）
                                                      ↓
最终视频 ← 合成视频 ← 重新混音 ← TTS 语音合成 ← 人声分离（Demucs）
```

---

## 环境要求

- **OS**: Ubuntu 24.04+ / Windows 11 (WSL) / macOS
- **GPU**: AMD ROCm 7.2+ / NVIDIA CUDA / DirectML（Windows），CPU 亦可
- **系统依赖**: `ffmpeg`, `vlc`, `python3-tk`, `python3-pil.imagetk`

---

## 快速开始

```bash
# 1. 克隆仓库
git clone git@github.com:hyphenzhao/ai-movie.git
cd ai-movie

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. GPU 加速（AMD ROCm）
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

# 5. 下载模型（详见 models/README.md）
# 需要下载：faster-whisper-large-v3 + Hy-MT1.5-1.8B + CosyVoice2-0.5B

# 6. 运行
python run.py
```

---

## 模型下载

模型需手动下载到 `models/` 目录。详见 **[models/README.md](models/README.md)**。

总下载量约 11 GB：
- `faster-whisper-large-v3` — ASR 语音识别
- `Hy-MT1.5-1.8B` — 文本翻译
- `CosyVoice2-0.5B` — TTS 语音合成

---

## 项目结构

```
ai-movie/
├── run.py                         # 入口
├── ai_movie/
│   ├── main.py                    # 启动 GUI
│   ├── config.py                  # 集中配置
│   ├── cutter.py                  # 视频切割
│   ├── demuxer.py                 # 音轨分离
│   ├── asr.py                     # 语音识别（多后端）
│   ├── translator.py              # 文本翻译（Hy-MT + Ollama）
│   ├── tts.py                     # 语音合成（CosyVoice）
│   ├── composer.py                # 人声分离 + 混音 + 视频合成
│   ├── project_log.py             # 项目日志
│   ├── task_manager.py            # 后台任务管理
│   └── gui/
│       ├── app.py                 # 主窗口
│       └── player.py              # VLC 播放器
├── models/                        # 模型文件（需手动下载）
├── Documentation/                 # 技术文档 & 项目状态
├── scripts/                       # 辅助脚本
├── requirements.txt
```

---

## 技术亮点

1. **TTS 性别检测**: CosyVoice-300M-SFT 逐段分析 F0 基频，自动在「中文男」「中文女」间切换，也支持 ECAPA 说话人日志全局预计算
2. **翻译三引擎**: Hy-MT 本地快速翻译 → Ollama NSFW 润色（批量模式，仅对含关键词片段）→ 口语化中文输出
3. **端到端 Pipeline**: 从原始视频到最终配音视频，一键完成，中间结果可查看/编辑

---

## License

本项目基于 MIT 许可证开源。内含模型的许可证请参见各模型官方仓库。
