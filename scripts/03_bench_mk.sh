#!/usr/bin/env bash
# Benchmark the megakernel and the PyTorch eager baseline.
# Setup mirrors the published one: 32-token prompt, 128 generated tokens, batch 1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MK_DIR="$ROOT/reference-megakernels"
RESULTS="$ROOT/results"

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
PROMPT_TOKENS="${PROMPT_TOKENS:-32}"
NTOK="${NTOK:-128}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

mkdir -p "$RESULTS"
# shellcheck disable=SC1091
source "$ROOT/.venvs/mk/bin/activate"

say "Calibrating a prompt of exactly $PROMPT_TOKENS tokens"
PROMPT="$(python "$ROOT/scripts/lib/make_prompt.py" --model "$MODEL" --ntok "$PROMPT_TOKENS")"
echo "prompt: $PROMPT"
echo "$PROMPT" > "$RESULTS/prompt.txt"

cd "$MK_DIR"

# generate.py prints "Input ids shape" on startup -- confirm it reads
# [1, PROMPT_TOKENS] in the logs below. If not, the comparison is biased.

say "megakernel (mode=mk)"
python megakernels/scripts/generate.py \
  mode=mk l1 \
  prompt="$PROMPT" \
  ntok="$NTOK" \
  batch_size=1 \
  num_warmup="$WARMUP" \
  num_iters="$ITERS" \
  tokens=False \
  2>&1 | tee "$RESULTS/bench_mk.log"

say "PyTorch eager (mode=torch)"
python megakernels/scripts/generate.py \
  mode=torch l1 \
  prompt="$PROMPT" \
  ntok="$NTOK" \
  batch_size=1 \
  num_warmup="$WARMUP" \
  num_iters="$ITERS" \
  tokens=False \
  2>&1 | tee "$RESULTS/bench_torch.log"

say "Summary"
grep -H "Tokens per second\|Average time" "$RESULTS"/bench_mk.log "$RESULTS"/bench_torch.log || true

say "Done. Next: scripts/04_bench_engines.sh"
