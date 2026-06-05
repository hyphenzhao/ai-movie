# AI Movie — 项目状态报告

> 更新日期：2026-06-02
> 平台：Ubuntu 24.04 + AMD Ryzen AI MAX+ 395（Radeon 8060S / ROCm 7.2）

---

## 已完成

### 基础架构
- [x] 项目骨架：`ai_movie/` 包结构、`run.py` 入口、集中配置
- [x] 跨平台兼容：从 Windows 迁移到 Linux，修复路径/字体/VLC
- [x] GPU 加速：PyTorch 2.9.1 + ROCm 7.2，Radeon 8060S GPU 推理

### Pipeline 步骤

| 步骤 | 模块 | 状态 |
|------|------|:----:|
| 切割视频 | `cutter.py` | ✅ 完成 |
| 拆分音轨 | `demuxer.py` | ✅ 完成 |
| 转换文字 | `asr.py` | ✅ 完成 |
| 文本翻译 | `translator.py` | ✅ 完成 |
| 人声分离 | `composer.py` | ✅ 完成 |
| 人声生成 | `tts.py` | ⚠️ 功能可用，性能待优化 |
| 重新混音 | `composer.py` | ✅ 完成 |
| 合成音轨 | `app.py` | ✅ 一键完成上面三步 |
| 合成视频 | `composer.py` | ✅ 完成 |
| 人物锚定 | `app.py` | 🔲 占位 stub |
| 口型匹配 | `app.py` | 🔲 占位 stub |

### GUI 功能
- [x] VLC 嵌入式视频播放器（主窗口 + 弹出窗口均带可拖动进度条）
- [x] 步骤工具栏（水平排列，颜色状态：锁定/就绪/运行中/完成/失败）
- [x] 切割视频：缩略图 + 播放按钮
- [x] 拆分音轨：音视频文件列表 + 播放按钮
- [x] 转换文字：语言选择 + 引擎选择（faster-whisper / openai-whisper / 自动）
- [x] 文本翻译：目标语言选择 + 原文/译文对照展示
- [x] 人声分离：人声/背景音文件展示 + 播放按钮
- [x] 人声生成：逐片段展示（原文/译文/时长）+ 播放按钮
- [x] 实时进度弹窗（文件进度 + 片段进度）
- [x] 项目保存/加载（`.aimovie.json`），含所有步骤数据恢复
- [x] 可取消的后台任务

### 已部署的模型

| 模型 | 用途 | 大小 | 推理引擎 |
|------|------|------|----------|
| `faster-whisper-large-v3` | ASR CPU fallback | 2.9 GB | CTranslate2 int8 |
| `Hy-MT1.5-1.8B` | 文本翻译（日→中） | 3.9 GB | PyTorch ROCm fp16 |
| `CosyVoice2-0.5B` | 声音克隆 TTS | 4.6 GB | PyTorch ROCm fp16 |
| Demucs htdemucs | 人声/背景分离 | 80 MB | PyTorch CPU |

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

4. **Hy-MT 翻译 slang 问题**
   - NSFW 俚语（如 `デカちん`）翻译不准确
   - 可通过术语表或 prompt engineering 改善

5. **合成音轨声音质量**
   - Demucs 分离后的背景音可能有轻微 artifacts
   - TTS 生成的人声采样率为 24kHz，与原视频可能不一致
   - 混音时简单相加，无音量平衡

6. **依赖版本冲突**
   - Python 3.14 太新，部分包（Coqui TTS、Matcha-TTS）无法正常安装
   - CosyVoice 要求的 `torch==2.3.1` 与已安装的 `torch==2.9.1+rocm7.2` 不一致
   - 通过 symlink / `--no-deps` 等 workaround 绕过

### 🔲 低优先级

7. **人物锚定** — 未实现，stub 占位
8. **口型匹配** — 未实现，stub 占位
9. **CLI 入口** — 仅有 GUI 模式，无命令行接口
10. **端到端测试** — 缺少自动化测试

---

## 模块清单

| 文件 | 行数 | 功能 | 状态 |
|------|:----:|------|:----:|
| `ai_movie/config.py` | ~110 | 集中配置（路径/字体/模型参数） | ✅ |
| `ai_movie/cutter.py` | ~120 | FFmpeg 视频切割 + 缩略图提取 | ✅ |
| `ai_movie/demuxer.py` | ~80 | 音视频分离（无声视频 + 16kHz WAV） | ✅ |
| `ai_movie/asr.py` | ~450 | 语音识别（多后端自动选择） | ✅ |
| `ai_movie/asr_gpu.py` | ~120 | GPU bridge（DirectML，Windows 用） | ✅ |
| `ai_movie/asr_wsl.py` | ~130 | GPU bridge（WSL+ROCm，Windows 用） | ✅ |
| `ai_movie/translator.py` | ~150 | 文本翻译（Hy-MT1.5-1.8B） | ✅ |
| `ai_movie/tts.py` | ~120 | 声音克隆 TTS（CosyVoice 2） | ⚠️ |
| `ai_movie/composer.py` | ~160 | 人声分离 + 混音 + 视频合成 | ✅ |
| `ai_movie/project_log.py` | ~130 | 项目日志 + 步骤依赖链 | ✅ |
| `ai_movie/task_manager.py` | ~80 | 线程安全任务注册 | ✅ |
| `ai_movie/utils.py` | ~20 | 文件/JSON 工具 | ✅ |
| `ai_movie/pipeline.py` | ~10 | Pipeline 占位 | 🔲 |
| `ai_movie/gui/app.py` | ~1700 | 主窗口 + 完整 GUI | ⚠️ |
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

### 模型文件
```
models/
├── faster-whisper-large-v3/    # 2.9 GB
├── Hy-MT1.5-1.8B/             # 3.9 GB
├── CosyVoice2-0.5B/           # 4.6 GB
└── CosyVoice/                 # 源码 (73 MB)
```

---

## 下一步建议

1. **优化 TTS 速度**：研究 CosyVoice 2 的 `speed` 参数或减少 flow steps
2. **TTS 降级方案**：如果 CosyVoice 太慢，可尝试 `edge-tts`（免费在线 TTS，速度快但无声音克隆）
3. **完成人物锚定**：用说话人识别（speaker diarization）区分不同说话人
4. **口型匹配**：研究 Wav2Lip 或类似方案同步口型
5. **添加 CLI**：`python main.py input.mp4 --output output.mp4`
