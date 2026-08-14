#!/usr/bin/env bash
# Correctness gate. A fast megakernel that emits garbage is not a result.
# This runs BEFORE any benchmarking, deliberately.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MK_DIR="$ROOT/reference-megakernels"
RESULTS="$ROOT/results"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

# Use MODEL rather than the .l1 config method so an ungated mirror can be
# substituted; .l1 hard-codes meta-llama/Llama-3.2-1B-Instruct.
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"

mkdir -p "$RESULTS"
echo "model: $MODEL"
# shellcheck disable=SC1091
source "$ROOT/.venvs/mk/bin/activate"
cd "$MK_DIR"

say "Numerics: megakernel vs reference path (diff_test.py, all layers)"
# pydra config methods take a LEADING DOT (".full"); a bare "full" is parsed as a
# key=value pair and raises. Default layer_limit=1 checks one layer; .full checks all.
python megakernels/scripts/diff_test.py .full model="$MODEL" 2>&1 | tee "$RESULTS/diff_test.log"

say "Generation sanity: does it produce coherent text?"
# If this emits token soup, the timing numbers downstream are meaningless.
python megakernels/scripts/generate.py \
  mode=mk model="$MODEL" \
  prompt="What is the capital of France?" \
  ntok=40 \
  2>&1 | tee "$RESULTS/sanity_mk.log"

say "Same prompt through PyTorch eager, for comparison"
python megakernels/scripts/generate.py \
  mode=torch model="$MODEL" \
  prompt="What is the capital of France?" \
  ntok=40 \
  2>&1 | tee "$RESULTS/sanity_torch.log"

cat <<'EOF'

------------------------------------------------------------------
MANUAL GATE — read the two sanity logs above before continuing.

  - Is the mk output coherent English?
  - Does it broadly agree with the torch output?

Sampling differences mean they need not be token-identical, but they
should be recognisably the same kind of answer. If mk output is
garbage, STOP. Do not benchmark a broken kernel.
------------------------------------------------------------------

Next: scripts/03_bench_mk.sh
EOF
