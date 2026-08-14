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
command -v nvcc >/dev/null || die "nvcc not found. Need the CUDA toolkit, not just the driver."
nvcc --version | tail -2

say "Checking HuggingFace access to the gated model"
# This is the #1 way to waste rented GPU hours. Fail here, before anything slow.
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set.
  Llama-3.2-1B-Instruct is GATED. Accept the license at
  https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
  then: export HF_TOKEN=hf_..."

python3 - <<'PY' || die "Cannot access the gated model. Has the license been accepted for THIS account?"
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
