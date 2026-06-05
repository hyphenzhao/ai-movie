"""
TTS 方案对比测试脚本
运行：.venv/bin/python test.py

输出文件统一放在 /tmp/tts_test/ 目录，每种方案生成一个 WAV。
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

OUT_DIR = Path("/tmp/tts_test")
OUT_DIR.mkdir(exist_ok=True)

# ── 测试文本（取自实际翻译片段，有标点、有情感） ──────────────────
TEXT = "大家好！欢迎观看今天的视频节目。接下来，让我们一起了解这个精彩的故事。"

REF_ZH_WAV  = "models/CosyVoice/asset/zero_shot_prompt.wav"   # 中文女声 274Hz
REF_ZH_TEXT = "希望你以后能够做的比我还好呦。"
REF_EN_WAV  = "models/CosyVoice/asset/cross_lingual_prompt.wav"  # 英文男声 148Hz


def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ─────────────────────────────────────────────────────────────────
# 共用：加载 CosyVoice2 模型（只加载一次）
# ─────────────────────────────────────────────────────────────────
sep("加载 CosyVoice2-0.5B 模型")
sys.path.insert(0, "models/CosyVoice")
sys.path.insert(0, "models/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel
m = AutoModel(model_dir="models/CosyVoice2-0.5B")
print("模型加载完成")


def run_cosyvoice(gen_iter, out_name):
    """收集生成器输出并保存为 WAV，返回时长(s)。"""
    chunks = []
    for gen in gen_iter:
        chunks.append(gen["tts_speech"].squeeze(0).cpu().numpy())
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    out = OUT_DIR / out_name
    sf.write(str(out), audio, m.sample_rate)
    return len(audio) / m.sample_rate, out


# ─────────────────────────────────────────────────────────────────
# 方案 1：zero_shot + 中文女声参考
#   最推荐：参考音频是中文，模型行为稳定，正常速度
# ─────────────────────────────────────────────────────────────────
sep("方案 1 / zero_shot + 中文女声参考")
print(f"文本：{TEXT}")
print(f"参考：{REF_ZH_WAV}（3.5s，中文）")
t0 = time.time()
dur, out = run_cosyvoice(
    m.inference_zero_shot(TEXT, REF_ZH_TEXT, REF_ZH_WAV, stream=False),
    "1_zero_shot_zh.wav",
)
elapsed = time.time() - t0
print(f"✓ 输出：{out}  时长={dur:.1f}s  耗时={elapsed:.0f}s  RTF={elapsed/dur:.2f}")


# ─────────────────────────────────────────────────────────────────
# 方案 2：instruct2 + 中文女声参考 + 情感指令
#   可用自然语言描述说话风格
# ─────────────────────────────────────────────────────────────────
sep("方案 2 / instruct2 + 中文女声参考 + 情感指令")
instruct = "用自然流畅、富有感情的语气朗读这段话。<|endofprompt|>"
print(f"文本：{TEXT}")
print(f"指令：{instruct}")
t0 = time.time()
try:
    dur, out = run_cosyvoice(
        m.inference_instruct2(TEXT, instruct, REF_ZH_WAV, stream=False),
        "2_instruct2_natural.wav",
    )
    elapsed = time.time() - t0
    print(f"✓ 输出：{out}  时长={dur:.1f}s  耗时={elapsed:.0f}s  RTF={elapsed/dur:.2f}")
except Exception as e:
    print(f"✗ 失败：{e}")

# 额外：用稍微不同风格再试一次
sep("方案 2b / instruct2 + 兴奋语气")
instruct2b = "用充满活力和热情的语气朗读，像节目主持人一样。<|endofprompt|>"
print(f"指令：{instruct2b}")
t0 = time.time()
try:
    dur, out = run_cosyvoice(
        m.inference_instruct2(TEXT, instruct2b, REF_ZH_WAV, stream=False),
        "2b_instruct2_lively.wav",
    )
    elapsed = time.time() - t0
    print(f"✓ 输出：{out}  时长={dur:.1f}s  耗时={elapsed:.0f}s  RTF={elapsed/dur:.2f}")
except Exception as e:
    print(f"✗ 失败：{e}")


# ─────────────────────────────────────────────────────────────────
# 方案 3：cross_lingual + 中文女声参考（对比基准）
#   不带语言标记，让模型自行判断
# ─────────────────────────────────────────────────────────────────
sep("方案 3 / cross_lingual + 中文女声参考（对比基准）")
print(f"参考：{REF_ZH_WAV}")
t0 = time.time()
dur, out = run_cosyvoice(
    m.inference_cross_lingual(TEXT, REF_ZH_WAV, stream=False),
    "3_cross_lingual_zh.wav",
)
elapsed = time.time() - t0
print(f"✓ 输出：{out}  时长={dur:.1f}s  耗时={elapsed:.0f}s  RTF={elapsed/dur:.2f}")


# ─────────────────────────────────────────────────────────────────
# 方案 4：ChatTTS（如果模型文件完整则运行）
# ─────────────────────────────────────────────────────────────────
sep("方案 4 / ChatTTS（本地，无需参考音频）")

CHATTTS_FILES = {
    "asset/Decoder.safetensors": 98,
    "asset/DVAE.safetensors":    57,
    "asset/Embed.safetensors":  138,
    "asset/Vocos.safetensors":   51,
    "asset/GPT.pt":             238,
}
missing = []
for f, expected_mb in CHATTTS_FILES.items():
    p = Path(f)
    if not p.exists():
        missing.append(f"  缺失：{f}（{expected_mb} MB）")
    else:
        actual_mb = p.stat().st_size / 1e6
        if actual_mb < expected_mb * 0.9:
            missing.append(f"  不完整：{f}（{actual_mb:.0f}/{expected_mb} MB）")

if missing:
    print("✗ ChatTTS 模型文件未就绪，跳过：")
    for m_msg in missing:
        print(m_msg)
    print("\n  下载命令（需要 HuggingFace 访问）：")
    print("  HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -c \"")
    print("    import ChatTTS; c = ChatTTS.Chat(); c.load()\"")
else:
    print("模型文件完整，开始加载 ChatTTS...")
    try:
        import ChatTTS, torch

        chat = ChatTTS.Chat()
        chat.load()

        t0 = time.time()
        spk = torch.load("asset/spk_stat.pt") if Path("asset/spk_stat.pt").exists() else None
        params = ChatTTS.Chat.InferCodeParams(spk_emb=spk) if spk else ChatTTS.Chat.InferCodeParams()
        wavs = chat.infer([TEXT], params_infer_code=params)
        audio = wavs[0]
        out = OUT_DIR / "4_chattts.wav"
        sf.write(str(out), audio, 24000)
        elapsed = time.time() - t0
        dur = len(audio) / 24000
        print(f"✓ 输出：{out}  时长={dur:.1f}s  耗时={elapsed:.0f}s")
    except Exception as e:
        print(f"✗ ChatTTS 运行失败：{e}")


# ─────────────────────────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────────────────────────
sep("汇总 — 生成文件")
for f in sorted(OUT_DIR.glob("*.wav")):
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name:35s}  {size_kb:5d} KB")

print(f"""
播放命令（逐个听）：
  ffplay {OUT_DIR}/1_zero_shot_zh.wav      -autoexit -nodisp
  ffplay {OUT_DIR}/2_instruct2_natural.wav -autoexit -nodisp
  ffplay {OUT_DIR}/2b_instruct2_lively.wav -autoexit -nodisp
  ffplay {OUT_DIR}/3_cross_lingual_zh.wav  -autoexit -nodisp
""")
