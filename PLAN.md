# Step 1 — Reproduce the single-GPU megakernel result

**Goal:** independently verify that collapsing Llama-3.2-1B's decode forward pass into one
persistent kernel beats vLLM and SGLang at batch size 1 on a single H100.

**Why this first:** it is the cheapest possible test of the premise the whole project rests on.
If the launch-overhead thesis does not hold — or we cannot measure it credibly — steps 2 and 3
are not worth funding. Budget is roughly $50 and a few days.

**Status: PASSED** on 2026-08-14 (1× H100 80GB HBM3). See [RESULTS.md](RESULTS.md).

---

## What we are claiming

Not that we invented anything. Step 1 produces **our own latency table**, generated on hardware we
rented, rather than a number quoted from a blog post. That table is the evidence base for the
step 2 proposal.

Target (from Hazy Research's published result on Llama-3.2-1B, batch 1, H100):

| Metric | Their reported figure |
|---|---|
| Forward pass | under 1 ms |
| Memory bandwidth utilisation | ~78% of H100 |
| vs. vLLM | ~2.5× |
| vs. SGLang | >1.5× |

We consider step 1 successful if we land within a reasonable margin of these, **not** if we match
them exactly. Different driver, different CUDA minor version, and different vLLM/SGLang releases
all move the baselines.

---

## Hardware

| | |
|---|---|
| GPU | 1 × H100 80GB (SXM or PCIe — no interconnect needed at this step) |
| Cost | ~$2–3/hr on Lambda, RunPod, Vast, Modal |
| Disk | 60GB+ (CUDA toolchain, three Python envs, model weights) |
| Est. wall time | 2–5 days, dominated by build friction rather than benchmarking |

An H100 is specifically required. The megakernel's `make` takes `GPU={H100,B200}` and defaults to
B200 otherwise; the kernels are architecture-specific, so an A100 will not work.

---

## Known blockers — resolve these BEFORE renting

These are the things that waste rented GPU hours. Each is free to resolve in advance.

1. **`meta-llama/Llama-3.2-1B-Instruct` is a gated model.** Accept the license on HuggingFace and
   generate a read token now. Hitting this at hour one of a rented box is the single most likely
   way to burn money on this step.
2. **Python 3.12 exactly.** `pyproject.toml` requires `>=3.12`. The upstream Makefile defaults
   `PYTHON_VERSION` to **3.13** and links `-lpython$(PYTHON_VERSION)`, so it must be set to 3.12
   explicitly or the link fails. Requires `python3.12-dev` for `libpython3.12.so`.
3. **CUDA 12.x is required — 13.0 is not enough.** ThunderKittens' README specifies CUDA 12.3+
   and its authors develop on 12.6; the megakernel is dense Hopper WGMMA inline PTX, precisely
   the code that breaks across a major toolkit bump. Vast's H100 images ship **CUDA 13.0 only**,
   so `00_provision.sh` installs `cuda-toolkit-12-8` alongside it and the build targets that.
   12.8 also matches the torch cu128 wheels, keeping nvcc and the torch runtime on one version.
   The driver is untouched and runs 12.8 fine — CUDA is backward compatible with older runtimes.
4. **Torch must come from the cu128 index**, per the repo README — not default PyPI.
5. **vLLM and SGLang need separate environments.** Their dependency sets conflict, and
   `bench_engines.py` carries conda-activation machinery for exactly this reason. Three venvs
   total: `mk`, `vllm`, `sglang`.

---

## What the upstream repo already gives us

Significant — we are writing far less harness than expected.

| Tool | What it does |
|---|---|
| `megakernels/scripts/generate.py` | Runs and times the megakernel. `mode=mk` (megakernel), `mode=torch` (PyTorch eager), `mode=pyvm` (Python-interpreted schedule). CUDA-event timed, with warmup. |
| `megakernels/scripts/bench_engines.py` | Benchmarks any OpenAI-compatible server — i.e. vLLM and SGLang. Launches the server, waits for health, tears it down. |
| `megakernels/scripts/diff_test.py` | Layer-by-layer numerics validation of the megakernel against the reference path. |
| `megakernels/scripts/llama_repl.py` | Interactive sanity check that the thing actually generates sane text. |

### A note on measurement methodology

The two benchmark paths measure decode differently, and this must be stated in any writeup:

- `generate.py` runs prefill once, then times only the decode loop with CUDA events.
- `bench_engines.py` runs the same prompt twice — once with `max_tokens=1`, once with
  `max_tokens=N` — and subtracts. This isolates decode by cancelling out prefill and HTTP overhead.

Both are legitimate time-per-output-token measures, but they are not identical instruments.
`bench_engines.py` includes per-request server overhead that `generate.py` does not. If the gap we
measure is large (>1.5×), this does not matter. If it is marginal, it does, and we would need to
re-measure vLLM in-process to make a fair claim.

---

## Execution plan

Scripts are in `scripts/`, numbered in run order. Each is idempotent and fails loudly.

### 0. Provision — `scripts/00_provision.sh`
Verifies the GPU is actually an H100, creates the three venvs, installs torch from cu128,
installs the megakernel package editable, and checks HuggingFace access to the gated model
**before** doing anything expensive.

### 1. Build — `scripts/01_build.sh`
Sets `THUNDERKITTENS_ROOT`, `MEGAKERNELS_ROOT`, `PYTHON_VERSION=3.12`, `GPU=H100`, then runs
`make` in `demos/low-latency-llama`. This compiles the CUDA megakernel and is the most likely
step to fail on a fresh box.

### 2. Correctness — `scripts/02_verify.sh`
Runs `diff_test.py` and a short `llama_repl.py` generation. **A fast megakernel that emits garbage
is not a result.** This gate comes before any benchmarking.

### 3. Benchmark — `scripts/03_bench_mk.sh`
Runs `generate.py` in `mode=mk` and `mode=torch`, 128 output tokens, batch 1, 20 iterations after
warmup. Captures raw output to `results/`.

### 4. Baselines — `scripts/04_bench_engines.sh`
Runs `bench_engines.py` against vLLM and SGLang in their own venvs, same prompt length and output
length. Captures to `results/`.

### 5. Write up — `RESULTS.md`
Table of tokens/s for all four configurations, plus the environment fingerprint (driver, CUDA,
torch, vLLM, SGLang versions) so the numbers are reproducible.

---

## Calibration detail

Hazy Research's published setup is a 32-token prompt with 128 generated tokens. Our two harnesses
specify the prompt differently — `generate.py` takes a prompt *string*, `bench_engines.py` takes a
prompt *length*. `generate.py` prints `Input ids shape` on startup; check it reads 32 and adjust
the prompt string if not. Comparing a 32-token prompt against a 64-token one would quietly bias
the comparison.

---

## Success criteria

Step 1 passes if **all** of these hold:

- [ ] Megakernel builds on H100
- [ ] `diff_test.py` passes and generated text is coherent
- [ ] Megakernel beats PyTorch eager by a wide margin (sanity — this should be easy)
- [ ] Megakernel beats **both** vLLM and SGLang at batch 1
- [ ] The measured forward pass is in the neighbourhood of 1 ms
- [ ] Environment is fully captured so the run is repeatable

## Failure modes and what each would mean

| What happens | Interpretation | Next move |
|---|---|---|
| Build fails on H100 | Toolchain/version drift in the repo | Fixable; open an issue upstream, try pinned CUDA image |
| Numerics fail | Their kernel is broken on our stack, or we mis-built | Do not proceed. Investigate before trusting any timing. |
| Megakernel wins, but only marginally | Our baselines are better tuned than theirs were | Still informative, but weakens the case for step 2 |
| Megakernel loses | Premise does not hold on our stack | **Stop the project.** This is the exit the sequencing exists to provide. |

A negative result here is a cheap and genuinely useful outcome. It is worth saying so explicitly
in the writeup rather than treating it as a failed experiment.
