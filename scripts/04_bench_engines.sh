#!/usr/bin/env bash
# Benchmark vLLM and SGLang baselines at batch size 1.
#
# bench_engines.py runs from the mk venv (it holds pydra/openai/psutil), but each
# server must start in its OWN venv -- vLLM and SGLang have conflicting deps. The
# launch string carries the activation, since bench_engines.py runs it through bash.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MK_DIR="$ROOT/reference-megakernels"
RESULTS="$ROOT/results"

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
PROMPT_TOKENS="${PROMPT_TOKENS:-32}"
NTOK="${NTOK:-128}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"
PORT="${PORT:-10210}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

mkdir -p "$RESULTS"

free_port() {
  # A server left running from a crashed earlier run will silently serve the
  # next benchmark and produce nonsense numbers.
  if lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
    say "Port $PORT busy — killing stale server"
    lsof -ti tcp:"$PORT" | xargs -r kill -9
    sleep 3
  fi
}

# shellcheck disable=SC1091
source "$ROOT/.venvs/mk/bin/activate"
cd "$MK_DIR"

# ------------------------------------------------------------------ vLLM

free_port
say "vLLM baseline"
# NOTE: --disable-log-requests was removed in vLLM 0.27; passing it exits with
# code 2 (argparse error) and the harness just reports "server crashed".
VLLM_LAUNCH="source $ROOT/.venvs/vllm/bin/activate && \
vllm serve $MODEL --port $PORT --max-model-len 4096"

python megakernels/scripts/bench_engines.py \
  model="$MODEL" \
  prompt_len="$PROMPT_TOKENS" \
  output_len="$NTOK" \
  batch_size=1 \
  num_warmup="$WARMUP" \
  num_iters="$ITERS" \
  port="$PORT" \
  launch="$VLLM_LAUNCH" \
  2>&1 | tee "$RESULTS/bench_vllm.log"

# ------------------------------------------------------------------ SGLang

free_port
say "SGLang baseline"
SGLANG_LAUNCH="source $ROOT/.venvs/sglang/bin/activate && \
python -m sglang.launch_server --model-path $MODEL --port $PORT --context-length 4096"

python megakernels/scripts/bench_engines.py \
  model="$MODEL" \
  prompt_len="$PROMPT_TOKENS" \
  output_len="$NTOK" \
  batch_size=1 \
  num_warmup="$WARMUP" \
  num_iters="$ITERS" \
  port="$PORT" \
  launch="$SGLANG_LAUNCH" \
  2>&1 | tee "$RESULTS/bench_sglang.log"

free_port

say "Summary"
grep -H "Throughput" "$RESULTS"/bench_vllm.log "$RESULTS"/bench_sglang.log || true

cat <<'EOF'

------------------------------------------------------------------
Reminder for the writeup: these baselines are measured through an
HTTP server and include per-request overhead that generate.py does
not. If the megakernel margin is large (>1.5x) this is immaterial.
If it is marginal, re-measure vLLM in-process before claiming a win.
------------------------------------------------------------------

Next: fill in RESULTS.md
EOF
