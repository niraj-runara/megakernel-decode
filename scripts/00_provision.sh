#!/usr/bin/env bash
# Provision a fresh H100 box for the megakernel step-1 reproduction.
# Idempotent: safe to re-run. Verifies cheap preconditions BEFORE expensive installs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MK_DIR="$ROOT/reference-megakernels"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preconditions

say "Checking GPU"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found. This must run on the GPU box."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$GPU_NAME" in
  *H100*) export GPU_ARCH=H100 ;;
  *B200*) export GPU_ARCH=B200 ;;
  *) die "Expected H100 or B200, found '$GPU_NAME'. Megakernel kernels are architecture-specific." ;;
esac
echo "Detected arch: $GPU_ARCH"

say "Checking CUDA toolkit"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/cuda_env.sh"

if export_cuda_env; then
  echo "Found CUDA $(cuda_major_minor "$NVCC") at $CUDA_HOME"
else
  echo "No CUDA 12.x toolkit found. System nvcc is: $(cuda_major_minor nvcc || echo none)"
  echo "Installing cuda-toolkit-${CUDA_SERIES/./-} (userspace only; the driver is untouched)."
  # ThunderKittens targets CUDA 12.x -- see the note in scripts/lib/cuda_env.sh.
  # Vast H100 images ship 13.0 only, which is a major bump past anything upstream tests.
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "cuda-toolkit-${CUDA_SERIES/./-}"
  export_cuda_env || die "cuda-toolkit-${CUDA_SERIES} install did not produce /usr/local/cuda-${CUDA_SERIES}"
  echo "Installed CUDA $(cuda_major_minor "$NVCC") at $CUDA_HOME"
fi

NVCC_VER="$(cuda_major_minor "$NVCC")"
case "$NVCC_VER" in
  12.*) : ;;
  *) die "Resolved nvcc is $NVCC_VER; ThunderKittens expects 12.x." ;;
esac

say "Checking python dev headers (build links -lpython3.12)"
command -v python3-config >/dev/null || die "python3-config missing. Install python3.12-dev."
ls /usr/lib/x86_64-linux-gnu/libpython3.12.so >/dev/null 2>&1 \
  || die "libpython3.12.so missing. Install python3.12-dev."

say "Checking HuggingFace access to the gated model"
# This is the #1 way to waste rented GPU hours. Fail here, before anything slow.
#
# Set ALLOW_NO_HF=1 to downgrade this to a warning: the toolchain install and the
# megakernel build need no HF access, so it is reasonable to provision and compile
# while waiting on a token. Everything from 02_verify.sh onward does need it.
if [ -z "${HF_TOKEN:-}" ]; then
  MSG="HF_TOKEN is not set.
  Llama-3.2-1B-Instruct is GATED. Accept the license at
  https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
  then: export HF_TOKEN=hf_..."
  if [ "${ALLOW_NO_HF:-0}" = "1" ]; then
    printf '\n\033[1;33mWARN: %s\033[0m\n' "$MSG"
    printf '\033[1;33mContinuing (ALLOW_NO_HF=1). Build will work; 02_verify.sh will not.\033[0m\n'
    SKIP_HF_CHECK=1
  else
    die "$MSG"
  fi
fi

[ "${SKIP_HF_CHECK:-0}" = "1" ] || python3 - <<'PY' || die "Cannot access the gated model. Has the license been accepted for THIS account?"
import os, sys, urllib.request
req = urllib.request.Request(
    "https://huggingface.co/api/models/meta-llama/Llama-3.2-1B-Instruct",
    headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
)
try:
    urllib.request.urlopen(req, timeout=20)
    print("HF gated-model access OK")
except Exception as e:
    print(f"HF check failed: {e}", file=sys.stderr)
    sys.exit(1)
PY

# ---------------------------------------------------------------- repo

say "Fetching Megakernels repo"
if [ ! -d "$MK_DIR" ]; then
  git clone --recurse-submodules https://github.com/HazyResearch/Megakernels.git "$MK_DIR"
else
  echo "Already present at $MK_DIR"
  git -C "$MK_DIR" submodule update --init --recursive
fi

# ---------------------------------------------------------------- python envs

say "Installing uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# Three separate envs: vLLM and SGLang have conflicting dependency sets, which is
# why bench_engines.py ships conda-activation machinery.
make_env() {
  local name="$1"
  if [ ! -d "$ROOT/.venvs/$name" ]; then
    say "Creating venv: $name (python 3.12)"
    uv venv --python 3.12 "$ROOT/.venvs/$name"
  else
    echo "venv $name already exists"
  fi
}

mkdir -p "$ROOT/.venvs"
make_env mk
make_env vllm
make_env sglang

say "Installing megakernel env"
# shellcheck disable=SC1091
source "$ROOT/.venvs/mk/bin/activate"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e "$MK_DIR"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
deactivate

say "Installing vLLM env"
source "$ROOT/.venvs/vllm/bin/activate"
uv pip install vllm
python -c "import vllm; print('vllm', vllm.__version__)"
deactivate

say "Installing SGLang env"
source "$ROOT/.venvs/sglang/bin/activate"
uv pip install "sglang[all]"
python -c "import sglang; print('sglang', sglang.__version__)"
deactivate

# ---------------------------------------------------------------- fingerprint

say "Recording environment fingerprint"
mkdir -p "$ROOT/results"
{
  echo "# Environment fingerprint"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(uname -a)"
  echo
  echo "## GPU"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo
  echo "## nvcc"
  nvcc --version | tail -2
  echo
  echo "## versions"
  source "$ROOT/.venvs/mk/bin/activate";     python -c "import torch;  print('torch  ', torch.__version__)"; deactivate
  source "$ROOT/.venvs/vllm/bin/activate";   python -c "import vllm;   print('vllm   ', vllm.__version__)";  deactivate
  source "$ROOT/.venvs/sglang/bin/activate"; python -c "import sglang; print('sglang ', sglang.__version__)"; deactivate
} > "$ROOT/results/environment.txt"

cat "$ROOT/results/environment.txt"

say "Provision complete. Next: scripts/01_build.sh"
