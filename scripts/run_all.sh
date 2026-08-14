#!/usr/bin/env bash
# Convenience wrapper. Stops at the correctness gate on purpose -- 02_verify.sh
# needs a human to read the sanity output before we spend time benchmarking.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$ROOT/scripts/00_provision.sh"
bash "$ROOT/scripts/01_build.sh"
bash "$ROOT/scripts/02_verify.sh"

cat <<'EOF'

==================================================================
STOP. Read results/sanity_mk.log and results/diff_test.log.

Continue only if numerics pass and the generated text is coherent:

    bash scripts/03_bench_mk.sh
    bash scripts/04_bench_engines.sh
==================================================================
EOF
