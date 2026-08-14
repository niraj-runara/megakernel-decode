# shellcheck shell=bash
# Resolve a CUDA 12.x toolkit and export the build environment.
#
# Why pin to 12.x: ThunderKittens' README states CUDA 12.3+ and its authors
# develop on 12.6. The megakernel is dense Hopper WGMMA inline PTX, which is
# exactly the kind of code that breaks across a major toolkit bump. Vast's H100
# images ship CUDA 13.0 only, so we install 12.8 alongside and build against it.
# 12.8 also matches the torch cu128 wheels upstream tests with, keeping nvcc and
# the torch runtime on the same major.minor.
#
# The driver (580.x, advertising CUDA 13.0) runs 12.8 binaries fine -- CUDA is
# backward compatible with older runtimes.

CUDA_SERIES="${CUDA_SERIES:-12.8}"

resolve_cuda_home() {
  # Prefer the exact series, then any 12.x, newest first.
  local c
  if [ -d "/usr/local/cuda-${CUDA_SERIES}" ]; then
    echo "/usr/local/cuda-${CUDA_SERIES}"
    return 0
  fi
  for c in $(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -Vr); do
    echo "$c"
    return 0
  done
  return 1
}

export_cuda_env() {
  local home
  home="$(resolve_cuda_home)" || return 1
  export CUDA_HOME="$home"
  export CUDA_PATH="$home"
  export PATH="$home/bin:$PATH"
  export LD_LIBRARY_PATH="$home/lib64:${LD_LIBRARY_PATH:-}"
  export NVCC="$home/bin/nvcc"
  return 0
}

cuda_major_minor() {
  "${1:-nvcc}" --version 2>/dev/null \
    | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1
}
