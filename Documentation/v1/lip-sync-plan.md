# 口型匹配（Lip Sync）实现计划

> 创建日期：2026-06-08
> 状态：Phase 1 验证中

---

## Context

AI Movie 已完成视频配音 Pipeline（ASR → 翻译 → TTS → 混音），但合成后的视频中人物口型仍匹配原始日语语音，与新的中文配音不同步。需要引入口型同步（Lip Sync）技术，根据 TTS 生成的中文音频驱动视频中人物的口型变化。

## 技术选型

| 方案 | 质量 | ROCm | 集成难度 | 结论 |
|------|:----:|:----:|:--------:|------|
| **Wav2Lip** | ⭐⭐⭐ | ⚠️ 待验证 | 低 | **首选** |
| Wav2Lip + GAN | ⭐⭐⭐⭐ | ⚠️ 待验证 | 中 | 备选 |
| Video-Retalking | ⭐⭐⭐⭐⭐ | ❌ CUDA 强依赖 | 高 | 放弃 |
| MuseTalk | ⭐⭐⭐⭐ | ❌ CUDA | 中 | 放弃 |
| ComfyUI | ⭐⭐⭐ | ⚠️ | 高 | 放弃 |

**结论**: 直调 Wav2Lip Python API，不经过 ComfyUI。

## 执行阶段

### Phase 1: ROCm 兼容性验证（当前）

**目标**: 确认 Wav2Lip 在 AMD ROCm 7.2 上能否正常推理

**步骤**:
1. 从 test_data/output_test.mp4 截取 30 秒测试片段（中间位置）
2. 从 TTS 输出中提取对应 30 秒的中文音轨
3. Clone Wav2Lip 仓库，安装依赖
4. 下载预训练模型（wav2lip_gan.pth / wav2lip.pth）
5. 在 ROCm 上运行推理，验证效果和速度
6. 如果 ROCm 不兼容，测试 CPU 回退方案

**验收标准**:
- [ ] 模型加载成功（无 CUDA 相关错误）
- [ ] 30 秒视频推理完成（ROCm 期望 <15 分钟，CPU 期望 <2 小时）
- [ ] 输出视频口型大致同步

### Phase 2: lip_sync.py 模块封装

**目标**: 将 Wav2Lip 封装为可复用的 Python 模块

**接口设计**:
```python
# ai_movie/lip_sync.py
def wav2lip_sync(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    checkpoint_path: str = "wav2lip_gan.pth",
    face_detector: str = "sfd",  # "sfd" | "retinaface"
    resize_factor: int = 1,
    progress_cb: Callable | None = None,
    cancel_check: Callable | None = None,
) -> Path:
    """Run Wav2Lip inference on a video + audio pair."""
```

**内部流程**:
1. `detect_faces()` — 逐帧人脸检测 + 裁剪（S3FD/RetinaFace）→ 128x128 人脸序列
2. `wav2lip_infer()` — Wav2Lip 生成口型修改后的人脸帧
3. `blend_faces()` — 将修改后的人脸贴回原视频帧
4. `compose_with_audio()` — FFmpeg 合入音轨

### Phase 3: GUI 集成

**目标**: 在工具栏中激活「口型匹配」按钮

**步骤**:
1. app.py 中实现 `_on_lip_sync` 方法
2. 添加进度弹窗（帧进度 + 可取消）
3. 完成回调写入 project log
4. 结果显示 tab

### Phase 4: 质量优化（可选）

- [ ] GFPGAN/CodeFormer 人脸增强
- [ ] 智能跳帧（远景/无脸帧跳过）
- [ ] 批量处理多片段

## 性能预估

| 视频长度 | ROCm (期望) | CPU (回退) |
|----------|:-----------:|:----------:|
| 30 秒 | 8-15 分钟 | 1-2 小时 |
| 3 分钟 | 1-2 小时 | 6-12 小时 |
| 10 分钟 | 3-5 小时 | 20-40 小时 |

## 待解决问题

1. **ROCm MIOpen 兼容性**: Wav2Lip 中使用了一些 CUDA 特有的算子（如双线性插值），ROCm 的 MIOpen fallback 可能导致性能下降或错误
2. **预训练模型**: wav2lip_gan.pth 是 CUDA 训练的，ROCm 上加载通常没问题
3. **人脸检测模型**: S3FD 是 PyTorch 模型，ROCm 兼容性较好
4. **内存占用**: Wav2Lip 推理需要 ~4-6 GB GPU 内存

## 显存爆炸问题分析与修复（2026-06-15）

### 问题现象

Wav2Lip 和 MuseTalk 两个方案在处理视频时显存使用均超标，直接 OOM。

### 根因分析

#### Wav2Lip 三大内存炸弹

**Bug #1: `batches = list(_batches())` — 全部批次同时物化（line 414）**

`_batches()` 是一个生成器，每次 yield 一个 batch。但 `list()` 调用将所有 batch 一次性物化到内存中。
每个 batch 包含 `wav2lip_batch_size` 个原始帧的 `.copy()`。

- 3 分钟 1080p 视频 (4320 帧, batch_size=128): 34 个 batch
- 每个 batch: 128 × 6MB 帧副本 = 768 MB
- **总计: 34 × 768 MB ≈ 26 GB 纯帧副本**

**Bug #2: `fb.append(frames[i].copy())` — 逐帧显式拷贝（line 402）**

每个帧都被 `.copy()` 复制一份放入 batch。与 Bug #1 叠加，全部帧数据在内存中存在两份：
- 原始 `frames` 列表: 26 GB
- batch 中的副本: 26 GB
- **合计: 52 GB**

**Bug #3: 无内存释放**

`face_results` 在 face detection 后、`mel_chunks` 在 batch 构建后、模型加载后的中间张量，均无显式清理。
`_detect_faces` 中的 S3FD detector 返回后未从 GPU 释放。

#### MuseTalk 四大内存问题

**Bug #4: 7+ 模型同时加载到 GPU**

MuseTalk 子进程一次性加载所有模型：
| 模型 | 磁盘 | GPU (float32) |
|------|------|---------------|
| UNet V15 | 3.2 GB | ~6.4 GB |
| SyncNet | 1.4 GB | ~2.8 GB |
| VAE (SD) | 320 MB | ~640 MB |
| Whisper | 145 MB | ~300 MB |
| DWpose | 389 MB | ~780 MB |
| Face Parse | 96 MB | ~190 MB |
| **合计** | **5.6 GB** | **~11.1 GB** |

**Bug #5: ffmpeg 全帧提取 + 重新加载**

MuseTalk 用 `ffmpeg` 将所有帧提取为 PNG 文件，然后 `read_imgs()` 全部加载回 numpy 数组。
相当于帧数据同时在磁盘和内存中各存在一份。

**Bug #6: 帧列表翻倍**

`frame_list_cycle = frame_list + frame_list[::-1]` 将帧列表复制一份逆序拼接，帧数翻倍。

**Bug #7: 未启用 float16**

MuseTalk 支持 `--use_float16` 标志将模型转为半精度（GPU 内存减半至 ~5.5 GB），但我们的调用未传此参数。

### 修复方案

#### Wav2Lip 修复（lip_sync.py）

1. **移除 `list(_batches())`** → 直接迭代生成器，每次仅一个 batch 存活
2. **`fb.append(frames[i].copy())` → `idxs.append(i)`** → 仅存帧索引，合成时按需引用
3. **`del face_results` + `torch.cuda.empty_cache()`** → face detection 后立即释放
4. **`wav2lip_batch_size` 默认值 128 → 32** → 减少单 batch GPU 内存峰值
5. **`_detect_faces` 末尾 `del detector` + `empty_cache()`** → 释放 face detector
6. **大视频警告** → 帧内存 > 4GB 时建议 resize_factor=2

**修复后峰值内存 (3 分钟 1080p)**:
| | 修复前 | 修复后 |
|--|--------|--------|
| frames 数组 | 26 GB | 26 GB |
| batch 帧副本 | 26 GB | **0** (索引) |
| face crops | 117 MB | 117 MB |
| GPU 模型+张量 | ~10 GB | ~6 GB |
| **合计** | **~62 GB** | **~32 GB** |

> 对于 segment-based 模式（每段 5-30s），单段峰值仅 ~5-8 GB。

#### MuseTalk 修复（lip_sync.py）

1. **`musetalk_sync` 新增 `use_float16=True` 参数** → 命令行追加 `--use_float16`，GPU 内存 ~11 GB → ~5.5 GB
2. **`musetalk_sync` 新增 `batch_size=4` 参数** → 显式传递 `--batch_size 4`（原默认 8，减半）
3. **`segment_based_lip_sync` 传递 `use_float16=True, batch_size=4`** → 确保 segment 模式也受益

**修复后 GPU 内存 (MuseTalk)**:
| | 修复前 | 修复后 |
|--|--------|--------|
| 模型 (float32) | ~11 GB | — |
| 模型 (float16) | — | **~5.5 GB** |
| 推理 batch | ~1 GB (bs=8) | **~0.5 GB** (bs=4) |
| **合计** | **~12 GB** | **~6 GB** |

### 待改善（Phase 2）

- [ ] Wav2Lip 流式帧读取：不全量加载 frames 到 RAM（需重构 face detection 为流式）
- [ ] MuseTalk 直接调 Python API 而非 subprocess（避免模型重复加载）
- [ ] 视频分辨率自适应 resize_factor

## 文件清单

| 文件 | 操作 | 说明 |
|------|:----:|------|
| `Documentation/lip-sync-plan.md` | 新增 | 本文档 |
| `test_data/test_clip_30s.mp4` | 新增 | Phase 1 测试片段 |
| `test_data/test_clip_30s_audio.wav` | 新增 | Phase 1 中文音轨 |
| `ai_movie/lip_sync.py` | 新增 | Phase 2 核心模块 |
| `ai_movie/gui/app.py` | 修改 | Phase 3 GUI 集成 |
| `models/wav2lip/` | 新增 | Wav2Lip 模型文件 |
