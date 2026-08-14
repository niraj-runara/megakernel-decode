#!/usr/bin/env bash
# Benchmark vLLM and SGLang baselines at batch size 1.
#
# Uses scripts/lib/bench_openai.py rather than upstream's bench_engines.py:
# upstream sends a bare token-ID array, which SGLang 0.5.9 rejects with
# "Prompt cannot be empty". Ours sends the same TEXT prompt to both engines --
# the identical prompt 03_bench_mk.sh calibrated to 32 tokens -- so the two
# baselines and the megakernel all see the same input.
#
# Each server runs in its own venv; vLLM and SGLang have conflicting deps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$ROOT/results"

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
PROMPT_TOKENS="${PROMPT_TOKENS:-32}"
NTOK="${NTOK:-128}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"
PORT="${PORT:-10210}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

mkdir -p "$RESULTS"

# 03_bench_mk.sh writes the calibrated prompt here. Regenerate if absent so this
# script can run standalone.
PROMPT_FILE="$RESULTS/prompt.txt"
if [ ! -s "$PROMPT_FILE" ]; then
  say "Calibrating prompt ($PROMPT_TOKENS tokens)"
  # shellcheck disable=SC1091
  source "$ROOT/.venvs/mk/bin/activate"
  python "$ROOT/scripts/lib/make_prompt.py" --model "$MODEL" --ntok "$PROMPT_TOKENS" > "$PROMPT_FILE"
  deactivate
fi
echo "prompt: $(cat "$PROMPT_FILE")"

free_port() {
  # A server left over from a crashed run would silently serve the next
  # benchmark and produce nonsense numbers.
  if lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
    say "Port $PORT busy — killing stale server"
    lsof -ti tcp:"$PORT" | xargs -r kill -9
    sleep 5
  fi
}

wait_for_server() {
  local name="$1" pid="$2" tries=180
  for _ in $(seq 1 $tries); do
    if ! kill -0 "$pid" 2>/dev/null; then
      die "$name server died during startup — see $RESULTS/server_${name}.log"
    fi
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "$name is up"
      return 0
    fi
    sleep 5
  done
  die "$name did not become ready in time"
}

run_engine() {
  local name="$1" launch="$2"

  free_port
  say "$name — starting server"
  setsid bash -c "$launch" > "$RESULTS/server_${name}.log" 2>&1 &
  local pid=$!
  wait_for_server "$name" "$pid"

  say "$name — benchmarking"
  # shellcheck disable=SC1091
  source "$ROOT/.venvs/mk/bin/activate"
  python "$ROOT/scripts/lib/bench_openai.py" \
    --model "$MODEL" --port "$PORT" --prompt-file "$PROMPT_FILE" \
    --output-len "$NTOK" --warmup "$WARMUP" --iters "$ITERS" \
    --label "$name" 2>&1 | tee "$RESULTS/bench_${name}.log"
  deactivate

  say "$name — stopping server"
  pkill -P "$pid" 2>/dev/null || true
  kill -9 "$pid" 2>/dev/null || true
  free_port
}

# NOTE: --disable-log-requests was removed in vLLM 0.27; passing it exits with
# code 2 and the failure surfaces only as an opaque "server crashed".
run_engine vllm "source $ROOT/.venvs/vllm/bin/activate && \
exec vllm serve $MODEL --port $PORT --max-model-len 4096"

run_engine sglang "source $ROOT/.venvs/sglang/bin/activate && \
exec python -m sglang.launch_server --model-path $MODEL --port $PORT --context-length 4096"

say "Summary"
grep -H "Tokens per second\|ms per token" "$RESULTS"/bench_vllm.log "$RESULTS"/bench_sglang.log || true

say "Done. Fill in RESULTS.md"
