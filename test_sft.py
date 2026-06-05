"""
CosyVoice-300M-SFT 测试脚本
运行：.venv/bin/python test_sft.py

测试内容：
  1. 列出所有内置说话人
  2. 用每个中文说话人合成同一句话，对比音质
  3. 测试不同语气指令（instruct 模式）
  4. 测试实际视频翻译片段
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_DIR = "models/CosyVoice-300M-SFT"
OUT_DIR   = Path("/tmp/tts_sft_test")
OUT_DIR.mkdir(exist_ok=True)

# ── 检查模型是否下载完整 ────────────────────────────────────────
required = ["cosyvoice.yaml", "llm.pt", "flow.pt", "hift.pt"]
missing  = [f for f in required if not (Path(MODEL_DIR) / f).exists()]
if missing:
    print("❌ 模型文件不完整，缺少：", missing)
    print("   请先运行：")
    print("   HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -c \"")
    print("     from huggingface_hub import snapshot_download")
    print(f"     snapshot_download('FunAudioLLM/CosyVoice-300M-SFT', local_dir='{MODEL_DIR}')\"")
    sys.exit(1)

# ── 加载模型 ────────────────────────────────────────────────────
sys.path.insert(0, "models/CosyVoice")
sys.path.insert(0, "models/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel

print("加载 CosyVoice-300M-SFT ...")
m = AutoModel(model_dir=MODEL_DIR)
print(f"sample_rate = {m.sample_rate}")

# ── 内置说话人列表 ──────────────────────────────────────────────
spks = m.list_available_spks()
print(f"\n内置说话人 ({len(spks)} 个)：{spks}")


def synth(text, spk, out_name, speed=1.0):
    t0 = time.time()
    chunks = []
    for gen in m.inference_sft(text, spk, stream=False, speed=speed):
        chunks.append(gen["tts_speech"].squeeze(0).cpu().numpy())
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    out = OUT_DIR / out_name
    sf.write(str(out), audio, m.sample_rate)
    dur     = len(audio) / m.sample_rate
    elapsed = time.time() - t0
    expected = len(text) / 4  # 约 4 字/秒
    flag = "✓" if 0.5 * expected < dur < 3 * expected else "⚠"
    print(f"  {flag} {out_name:40s}  {dur:.1f}s (期望~{expected:.0f}s)  耗时{elapsed:.0f}s")
    return out


# ═══════════════════════════════════════════════════════════════
# 测试 1：所有中文说话人，同一句话
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("测试 1：所有内置说话人")
print("="*60)
text1 = "大家好，欢迎观看今天的节目，接下来让我们进入正题。"
for spk in spks:
    safe = spk.replace(" ", "_")
    synth(text1, spk, f"1_{safe}.wav")


# ═══════════════════════════════════════════════════════════════
# 测试 2：不同文本类型（标点、数字、感叹）
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("测试 2：不同文本类型（用中文女）")
print("="*60)
spk_f = "中文女" if "中文女" in spks else spks[0]

cases = [
    ("short",    "你好！"),
    ("question", "你今天感觉怎么样？"),
    ("emotion",  "太棒了！我真的非常高兴见到你！"),
    ("number",   "今天是2024年6月4日，温度是28度。"),
    ("long",     "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"),
]
for tag, text in cases:
    synth(text, spk_f, f"2_{tag}.wav")


# ═══════════════════════════════════════════════════════════════
# 测试 3：语速控制
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("测试 3：语速控制（中文女，同一句话）")
print("="*60)
text3 = "大家好，欢迎观看今天的节目，接下来让我们进入正题。"
for speed, tag in [(0.8, "slow"), (1.0, "normal"), (1.2, "fast")]:
    synth(text3, spk_f, f"3_speed_{tag}.wav", speed=speed)


# ═══════════════════════════════════════════════════════════════
# 测试 4：实际翻译片段（从项目数据取前 5 条）
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("测试 4：实际翻译片段（前 5 条）")
print("="*60)
import json
project_file = Path("output_test.aimovie.json")
if project_file.exists():
    data = json.loads(project_file.read_text(encoding="utf-8"))
    segs = data.get("step_data", {}).get("文本翻译", {}).get("segments", [])
    for i, seg in enumerate(segs[:5]):
        t = seg.get("text_translated", "").strip()
        if t:
            synth(t, spk_f, f"4_seg{i+1:02d}.wav")
else:
    print("  (未找到 output_test.aimovie.json，跳过)")


# ═══════════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("汇总 — 输出文件")
print("="*60)
files = sorted(OUT_DIR.glob("*.wav"))
for f in files:
    print(f"  {f}")

print(f"""
播放命令：
  for f in {OUT_DIR}/*.wav; do
    echo "--- $f ---"
    ffplay "$f" -autoexit -nodisp -loglevel quiet
  done
""")
