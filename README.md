# AI Movie — 广告视频语言本地化（Video Dubbing）

> **项目用途**：面向男女性用品广告的多国家语言本地化；禁止用于色情类电影的制作、翻译或推广。工具不进行内容审核，使用方依据当地广告法律审核素材及投放并承担相关责任。完整处理边界见下文。

## 用途与处理范围

本项目的产品目标是准确识别广告口播、生成可校对文字、忠实翻译、TTS 配音并匹配口型。品牌、型号、数字、单位、产品描述、否定关系及原有语气应准确保留，不擅自增加或删减广告信息。多国家推广是目标，不能据此宣称当前版本已完成所有目标语言的验证。

- **允许用途与禁止用途**：用于男女性用品广告翻译及语言本地化；禁止用于色情类电影的制作、翻译或推广。第三方模型、历史素材或实验名称不扩大本项目用途。
- **无词发声**：喘息、笑声、叹气等无实际词语的声音保留原声，不翻译、不以 TTS/音色转换替换、不修改口型；与真实台词混合时应按时间区间分别处理。这是语音处理路由，不是对声音是否具有性暗示进行审核或分级。
- **图像范围**：仅获取脸部定位、跟踪及口型匹配所必需的信息，并修改目标人物口型。其他画面不进行分析或主动编辑，不做身体/场景语义分析、非目标人物修改、无关人脸美化、字幕或产品画面改写。图像以外的广告文稿和术语可由使用方提供。
- **内容审核归属**：本项目不进行内容审核，不判定广告尺度，不提供各国广告法律审查或投放许可。使用方须根据目标国家或地区的相关广告法律自行审核素材、成品及投放行为；违法内容及其使用、传播属于使用方责任，项目不对使用方违法内容负责。自动技术验收通过不表示内容合法或允许投放。
- **技术质量承诺**：质量工作限于广告语言转换、配音、音画同步和指定保护区域的保全；以约定测试、真实测量及人工复核说明效果，不把“仅保证翻译质量”写成绝无错译、所有语言已经支持或免责条款当然有效的保证。
- **工程说明规范**：自有工程与开发说明均须包含或链接这份用途声明。历史实验事实及第三方原始文档保持可追溯，不将历史技术能力误写成当前用途授权。

**当前实现与目标的差距（2026-10-05）**：v3.4 的三部日语转中文短片已完成生成并交付评测，长片仍在回归，项目尚未成为经过验证的通用多语言广告产品。现有整段发声分类、分离人声回贴、下半脸增强、整帧缩放/编码与全画面切镜检测，尚不能严格保证“无词区间原声原样保留、仅嘴部修改、其余画面不分析处理”。字幕 OCR 为历史独立评测工具，不应成为广告生产流程依赖。这些是待分析和另行实施的差距，不因文档声明而视为已修复；本次未修改正在运行的生成代码。

## 现有工程入口

当前回归与服务器交付状态见 [接手说明](Documentation/codex-handoff.md)；存储路径见 [存储与交付](Documentation/storage-and-delivery.md)。下文保留早期 GUI 安装与模型接入说明，不能用其模型列表或容量估计替代当前 v3.4 的运行配置。

> 📊 v2 质量升级说明见 [Documentation/v2-quality-upgrade.md](Documentation/v2-quality-upgrade.md)；v1 历史文档在 [Documentation/v1/](Documentation/v1/)

---

## 特性

- 🎬 **完整 Pipeline**：视频切割 → 音轨分离 → 语音识别 → 文本翻译 → 人声分离 → 语音合成 → 混音 → 合成视频
- 🧠 **AI 引擎全本地**：语音识别（faster-whisper）、翻译（Hy-MT1.5-1.8B）、TTS（CosyVoice2/CosyVoice-300M-SFT）
- 🌐 **GPU 加速**：AMD ROCm / NVIDIA CUDA / DirectML，自动检测并选择最佳后端
- 🎯 **智能性别检测**：三种算法可选（pyin 逐段 / 全局 F0 / ECAPA 说话人日志），自动匹配男/女声
- 📝 **术语与译文校对**：已有术语表和本地模型校对环节；广告忠实表达与目标语言质量仍需分别验收
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
2. **翻译与校对**: 早期 GUI 支持 Hy-MT 加 Ollama；当前回归使用 Sakura 加 Qwen。已有提示词风格尚需按广告忠实表达收紧，不能把历史风格控制当作广告质量保证
3. **端到端 Pipeline**: 从原始视频到最终配音视频，一键完成，中间结果可查看/编辑

---

## License

本项目基于 MIT 许可证开源。内含模型的许可证请参见各模型官方仓库。
