#!/usr/bin/env bash
# runpod_bootstrap.sh
# One-shot environment bootstrap for a RunPod pod running the
# runpod/pytorch:2.4.0-py3.11-cuda12.4.1 template (or similar).
#
# Installs PaddlePaddle (matching the pod's CUDA), PaddleFormers (ships
# ERNIEKit), PaddleOCR, then uv + the project venv. Idempotent — re-run
# freely, it skips anything already installed.
#
# Usage (inside the pod, after git-cloning this repo into /workspace):
#   cd /workspace/paddleocr-aksara-jawa
#   bash scripts/runpod_bootstrap.sh
#
# Or paste the whole file via heredoc if you haven't cloned yet:
#   cat > /tmp/bootstrap.sh <<'EOF'
#   ... contents ...
#   EOF
#   bash /tmp/bootstrap.sh
#
# Expected wall-clock: ~3 min on A40 (dominated by pip downloads).

set -euo pipefail

REPO_URL="https://github.com/setilanaji/paddleocr-aksara-jawa.git"
REPO_DIR="/workspace/paddleocr-aksara-jawa"
PADDLE_VERSION="3.2.1"

log() { printf "\n\033[1;34m==>\033[0m %s\n" "$*"; }
warn() { printf "\n\033[1;33m[warn]\033[0m %s\n" "$*"; }

# ── 1. Detect CUDA major version and pick the right Paddle wheel index ───────
log "Detecting CUDA..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found — is this a GPU pod?"
    exit 1
fi
CUDA_MAJOR_MINOR=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | awk '{print $3}')
CUDA_SHORT=$(echo "$CUDA_MAJOR_MINOR" | tr -d '.')
# Paddle only ships cu118 and cu126 wheels (cu121 and cu124 indexes exist but
# are empty). CUDA 12.x runtimes are forward-compatible with cu126 binaries.
case "$CUDA_SHORT" in
    128|127|126|125|124|123|122|121|120) PADDLE_CU=cu126 ;;
    119|118|117|116|115|114|113|112|111|110) PADDLE_CU=cu118 ;;
    *)       warn "Unknown CUDA $CUDA_MAJOR_MINOR, defaulting to cu126"; PADDLE_CU=cu126 ;;
esac
log "CUDA $CUDA_MAJOR_MINOR detected → using $PADDLE_CU Paddle wheel index"

# ── 2. PaddlePaddle-GPU ──────────────────────────────────────────────────────
if python -c "import paddle; assert paddle.__version__.startswith('${PADDLE_VERSION}')" 2>/dev/null; then
    log "paddlepaddle-gpu ${PADDLE_VERSION} already installed, skipping"
else
    log "Installing paddlepaddle-gpu==${PADDLE_VERSION} ($PADDLE_CU)..."
    pip install "paddlepaddle-gpu==${PADDLE_VERSION}" \
        -i "https://www.paddlepaddle.org.cn/packages/stable/${PADDLE_CU}/"
fi

# ── 3. PaddleFormers (ships ERNIEKit) + PaddleOCR ────────────────────────────
# `blinker` on Debian-based images is a distutils install that pip refuses
# to uninstall cleanly. Pre-reinstall it so downstream deps can bump it.
log "Patching distutils-installed blinker (pre-emptive)..."
pip install -q --ignore-installed blinker

log "Installing paddleformers + paddleocr..."
pip install -q paddleformers "paddleocr>=3.0.0"

# ── 4. Sanity check ──────────────────────────────────────────────────────────
log "Paddle sanity check:"
python -c "import paddle; paddle.utils.run_check()" || {
    warn "paddle.utils.run_check() failed — see output above."
    exit 1
}
log "erniekit CLI: $(command -v erniekit || echo NOT FOUND)"

# ── 5. uv + project repo ─────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$REPO_DIR/.git" ]; then
    log "Cloning $REPO_URL into $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    log "Repo already at $REPO_DIR, pulling latest..."
    git -C "$REPO_DIR" pull --ff-only || warn "git pull failed (maybe local changes?) — continuing"
fi

cd "$REPO_DIR"
log "Syncing project venv with uv..."
uv sync

# ── 6. HF login if token is present ──────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
    log "Logging into HuggingFace (HF_TOKEN env var detected)..."
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" \
        && log "HF login OK"
fi

# ── 7. Summary ───────────────────────────────────────────────────────────────
log "Bootstrap complete."
echo
echo "Repo       : $REPO_DIR"
echo "Paddle     : $(python -c 'import paddle; print(paddle.__version__)')"
echo "uv         : $(uv --version 2>&1)"
echo "erniekit   : $(command -v erniekit || echo not on PATH)"
echo
echo "Next steps:"
echo "  cd $REPO_DIR"
echo "  # Regenerate or rsync data — see docs/runpod_setup.md §5"
echo "  # Then:"
echo "  CUDA_VISIBLE_DEVICES=0 erniekit train training/aksara_jawa_lora_config.yaml"
