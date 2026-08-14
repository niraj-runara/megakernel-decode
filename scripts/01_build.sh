#!/usr/bin/env bash
# Compile the low-latency Llama megakernel.
# Most likely step to fail on a fresh box.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MK_DIR="$ROOT/reference-megakernels"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

[ -d "$MK_DIR" ] || die "Repo missing. Run scripts/00_provision.sh first."

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$GPU_NAME" in
  *H100*) GPU=H100 ;;
  *B200*) GPU=B200 ;;
  *) die "Unsupported GPU '$GPU_NAME'" ;;
esac

# NOTE: the Makefile defaults to B200 if GPU is unset — an easy way to silently
# build the wrong kernels and then wonder why nothing runs.
export GPU
export THUNDERKITTENS_ROOT="$MK_DIR/ThunderKittens"
export MEGAKERNELS_ROOT="$MK_DIR"
export PYTHON_VERSION=3.12

say "Build configuration"
cat <<EOF
GPU                 = $GPU
THUNDERKITTENS_ROOT = $THUNDERKITTENS_ROOT
MEGAKERNELS_ROOT    = $MEGAKERNELS_ROOT
PYTHON_VERSION      = $PYTHON_VERSION
EOF

# shellcheck disable=SC1091
source "$ROOT/.venvs/mk/bin/activate"

say "Compiling demos/low-latency-llama"
mkdir -p "$ROOT/results"
cd "$MK_DIR/demos/low-latency-llama"
make 2>&1 | tee "$ROOT/results/build.log"

say "Build artifacts"
ls -la ./*.so 2>/dev/null || die "No .so produced — check results/build.log"

say "Build complete. Next: scripts/02_verify.sh"
