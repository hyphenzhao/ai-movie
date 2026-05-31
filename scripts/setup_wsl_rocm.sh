#!/usr/bin/env bash
# WSL + ROCm + openai-whisper 一键配置脚本
# 在 WSL (Ubuntu 24.04) 内执行
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

MARKER_FILE="$HOME/.config/ai-movie-wsl-rocm"
VENV_DIR="$HOME/ai-movie-venv"

# ── Phase 0: Pre-flight checks ─────────────────────────────────

info "检查运行环境…"

if ! grep -qi microsoft /proc/version 2>/dev/null; then
    error "此脚本必须在 WSL 内运行"
    exit 1
fi

# Check AMD GPU visible via WSL GPU-PV (paravirtualization)
GPU_VISIBLE=false
if [ -d /dev/dri ] && ls /dev/dri/render* >/dev/null 2>&1; then
    info "检测到 GPU 渲染设备: $(ls /dev/dri/render* 2>/dev/null)"
    GPU_VISIBLE=true
elif [ -e /dev/kfd ]; then
    info "检测到 /dev/kfd 设备"
    GPU_VISIBLE=true
else
    warn "未检测到 GPU 设备 (/dev/dri/render*, /dev/kfd)"
    warn ""
    warn "Strix Halo (Radeon 8060S) 需要："
    warn "  1. Windows AMD Adrenalin 25.5.1+ 驱动（WSL GPU-PV 支持）"
    warn "  2. 驱动安装后重启 WSL: 在 PowerShell 中运行 wsl --shutdown"
    warn ""
    warn "如果驱动已是最新但仍看不到 GPU，请确认："
    warn "  - wsl --version >= 2.4.x"
    warn "  - Windows 更新 KB5044380+ 已安装（WSL GPU-PV 增强）"
    warn ""
    warn "脚本将继续安装 CPU-only 模式，GPU 加速可在驱动就绪后重新运行本脚本"
fi

# ── Phase 1: System dependencies ───────────────────────────────

info "安装系统依赖…"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv python3-dev build-essential wget gnupg

# ── Phase 2: Install ROCm userspace ────────────────────────────

ROCM_OK=false

if [ "$GPU_VISIBLE" = true ]; then
    info "配置 AMD ROCm 仓库…"

    # AMD 官方 WSL2 安装方法：amdgpu-install
    if ! command -v amdgpu-install &>/dev/null; then
        info "下载 amdgpu-install…"
        AMD_INSTALL_DEB="amdgpu-install_latest_noble.deb"
        wget -q "https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_6.3.60302-1_all.deb" \
            -O "/tmp/$AMD_INSTALL_DEB" 2>/dev/null || {
            warn "无法下载 amdgpu-install，尝试备选 URL…"
            wget -q "https://repo.radeon.com/amdgpu-install/6.3/ubuntu/noble/amdgpu-install_6.3.60302-1_all.deb" \
                -O "/tmp/$AMD_INSTALL_DEB" 2>/dev/null || true
        }

        if [ -f "/tmp/$AMD_INSTALL_DEB" ] && [ -s "/tmp/$AMD_INSTALL_DEB" ]; then
            sudo dpkg -i "/tmp/$AMD_INSTALL_DEB" 2>/dev/null || {
                sudo apt-get install -y -qq -f  # fix deps
                sudo dpkg -i "/tmp/$AMD_INSTALL_DEB" 2>/dev/null || true
            }
            rm -f "/tmp/$AMD_INSTALL_DEB"
        fi
    fi

    if command -v amdgpu-install &>/dev/null; then
        info "安装 ROCm (WSL2 模式，跳过内核驱动)…"
        sudo amdgpu-install --usecase=wsl,rocm --no-dkms -y 2>&1 || {
            warn "amdgpu-install 失败，尝试仅安装 HIP 运行时…"
            sudo apt-get install -y -qq --no-install-recommends \
                rocm-hip-runtime rocm-hip-libraries 2>/dev/null || true
        }
    else
        warn "amdgpu-install 不可用，尝试直接安装 HIP 运行时…"
        # Add ROCm repo key if not done
        if [ ! -f /etc/apt/trusted.gpg.d/rocm.gpg ]; then
            wget -q https://repo.radeon.com/rocm/rocm.gpg.key -O - \
                | gpg --dearmor 2>/dev/null \
                | sudo tee /etc/apt/trusted.gpg.d/rocm.gpg > /dev/null
        fi
        if [ ! -f /etc/apt/sources.list.d/rocm.list ]; then
            echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/latest noble main" \
                | sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null
            sudo apt-get update -qq
        fi
        sudo apt-get install -y -qq --no-install-recommends \
            rocm-hip-runtime 2>/dev/null || true
    fi

    # Add user to render/video groups
    sudo usermod -a -G render,video "$USER" 2>/dev/null || true

    # Verify
    if [ -f /opt/rocm/bin/rocminfo ]; then
        info "ROCm 运行时已安装"
        ROCM_OK=true
    elif [ -f /opt/rocm/lib/libamdhip64.so ]; then
        info "HIP 运行时库已安装"
        ROCM_OK=true
    else
        warn "ROCm 系统包安装失败 — PyTorch ROCm wheel 自带部分库，将继续尝试"
    fi
else
    info "跳过 ROCm 系统包安装（GPU 不可见）"
fi

# Detect GPU architecture
GFX_ARCH="gfx1151"  # Strix Halo / RDNA 3.5
if command -v /opt/rocm/bin/rocm_agent_enumerator &>/dev/null; then
    GFX_ARCH=$(/opt/rocm/bin/rocm_agent_enumerator 2>/dev/null | head -1 || echo "gfx1151")
fi
info "目标 GPU 架构: $GFX_ARCH (Strix Halo / RDNA 3.5)"

# ── Phase 3: Python venv ───────────────────────────────────────

info "创建 Python 虚拟环境: $VENV_DIR"
python3 -m venv "$VENV_DIR"
PYTHON="$VENV_DIR/bin/python3"

"$PYTHON" -m pip install --upgrade -q pip setuptools wheel

# ── Phase 4: Install PyTorch with ROCm ─────────────────────────

info "安装 PyTorch (ROCm 版本)…"
"$PYTHON" -m pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

info "安装 openai-whisper…"
"$PYTHON" -m pip install -q openai-whisper tqdm numpy

# ── Phase 5: Pre-download whisper model ────────────────────────

info "预下载 Whisper large-v3 模型 (~3 GB)…"

cat << 'PYEOF' | "$PYTHON"
import whisper, os
print("正在下载/验证 large-v3 模型…")
model = whisper.load_model("large-v3")
cache = os.path.expanduser("~/.cache/whisper")
if os.path.isdir(cache):
    files = os.listdir(cache)
    print(f"模型缓存目录: {cache}")
    for f in files:
        fpath = os.path.join(cache, f)
        size_mb = os.path.getsize(fpath) / (1024*1024) if os.path.isfile(fpath) else 0
        print(f"  {f}  ({size_mb:.0f} MB)")
else:
    print("模型加载成功（缓存位置未知）")
PYEOF

MODEL_OK=$?

# ── Phase 6: Verify GPU accessibility ──────────────────────────

GPU_TEST_OK=false
if [ "$GPU_VISIBLE" = true ]; then
    info "验证 ROCm GPU 可用性…"

    cat << 'PYEOF' | HSA_OVERRIDE_GFX_VERSION=11.0.0 "$PYTHON"
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA (ROCm) available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count:    {torch.cuda.device_count()}")
    print(f"Device name:     {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: ROCm GPU 未被 PyTorch 检测到")
    print("可能原因:")
    print("  1. 包安装后需要重启 WSL: 在 PowerShell 中 wsl --shutdown")
    print("  2. 需添加 HSA_OVERRIDE_GFX_VERSION=11.0.0 到 ~/.bashrc")
    print("  3. AMD 驱动版本不适配 Strix Halo (需要 Adrenalin 25.5.1+)")
PYEOF

    if [ $? -eq 0 ]; then
        GPU_TEST_OK=true
    fi
else
    info "跳过 GPU 验证（GPU 不可见）"
fi

# ── Phase 7: Create marker file ────────────────────────────────

mkdir -p "$(dirname "$MARKER_FILE")"

# Write marker even if GPU not ready — user can re-run after fixing drivers
ROCM_VER="unknown"
[ -f /opt/rocm/bin/rocminfo ] && ROCM_VER=$(/opt/rocm/bin/rocminfo 2>/dev/null | grep -i "ROCm" | head -1 || echo "unknown")

cat > "$MARKER_FILE" << EOF
rocm_version=$ROCM_VER
gfx_arch=$GFX_ARCH
gpu_visible=$GPU_VISIBLE
gpu_test=$GPU_TEST_OK
pytorch_version=$("$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
model_downloaded=$([ $MODEL_OK -eq 0 ] && echo "true" || echo "false")
created_at=$(date -Iseconds)
EOF

info "标记文件已创建: $MARKER_FILE"

# ── Done ───────────────────────────────────────────────────────

echo ""
echo "============================================"
if [ "$GPU_VISIBLE" = true ] && [ "$GPU_TEST_OK" = true ]; then
    echo -e "${GREEN}  WSL ROCm GPU 环境配置完成！${NC}"
elif [ "$GPU_VISIBLE" = true ]; then
    echo -e "${YELLOW}  ROCm 已安装，但 GPU 测试未通过${NC}"
    echo ""
    echo "请执行以下步骤后重新运行本脚本:"
    echo "  1. 在 PowerShell 中: wsl --shutdown"
    echo "  2. 重新打开 WSL 终端"
    echo "  3. 重新运行本脚本"
else
    echo -e "${YELLOW}  CPU-only 配置完成（GPU 不可见）${NC}"
    echo ""
    echo "要使 GPU 在 WSL 中可见，请:"
    echo "  1. 更新 Windows AMD 驱动到 25.5.1+"
    echo "  2. PowerShell: wsl --shutdown"
    echo "  3. 重新打开 WSL 并重新运行本脚本"
fi
echo "============================================"
echo ""
echo "环境变量 (添加到 ~/.bashrc):"
echo '  export HSA_OVERRIDE_GFX_VERSION=11.0.0'
echo ""
