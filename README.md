# megakernel-decode

Reproducing persistent-kernel (megakernel) low-latency LLM decode, as a tractable
alternative to TileRT — which cannot be extended, because it ships as closed
`.so` binaries pinned to 8× B200.

**Step 1** (this repo, currently): verify the single-GPU megakernel result on
1× H100 with Llama-3.2-1B at batch size 1. See [`PLAN.md`](PLAN.md).

Later steps — multi-GPU fused communication via TileScale/ParallelKittens, then
a custom Tier-2 decode layer — are out of scope until step 1 passes.

---

## Quick start on a fresh H100 box

This repo is private. Connect with **agent forwarding** so the instance can clone
it without a key ever being written to a machine we're renting:

```bash
ssh -A ubuntu@<instance>            # -A forwards the local SSH agent
ssh -T git@github-work              # should greet niraj-runara
```

If forwarding is unavailable, use a fine-grained read-only PAT over HTTPS instead
and revoke it at teardown. Do not copy a private key onto the instance.

```bash
git clone git@github-work:niraj-runara/megakernel-decode.git
cd megakernel-decode

# Llama-3.2-1B-Instruct is GATED. Accept the license first at
# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
export HF_TOKEN=hf_...

bash scripts/run_all.sh      # provision -> build -> verify, then stops
```

> The `github-work` host alias must exist in the instance's `~/.ssh/config`, or
> substitute `git@github.com` — with agent forwarding, either resolves to the
> same forwarded key.

`run_all.sh` deliberately halts at the correctness gate. Read
`results/diff_test.log` and `results/sanity_mk.log`, then continue:

```bash
bash scripts/03_bench_mk.sh
bash scripts/04_bench_engines.sh
```

Fill in [`RESULTS.md`](RESULTS.md) from `results/`.

---

## Scripts

| Script | Does |
|---|---|
| `00_provision.sh` | Verifies GPU is H100/B200 and HF access works, **then** installs. Three venvs (`mk`, `vllm`, `sglang`) — vLLM and SGLang have conflicting deps. |
| `01_build.sh` | Compiles the CUDA megakernel with `GPU=H100`. Most likely step to fail on a fresh box. |
| `02_verify.sh` | Numerics via `diff_test.py` plus a generation sanity check. **Correctness gate — runs before benchmarking.** |
| `03_bench_mk.sh` | Megakernel and PyTorch eager, 32-token prompt / 128 output / batch 1. |
| `04_bench_engines.sh` | vLLM and SGLang baselines, same shape. |
| `lib/make_prompt.py` | Emits a prompt of exactly N tokens, so both harnesses see the same prompt length. |

All scripts are idempotent and write to `results/`.

## Layout

```
PLAN.md                 what we're doing, why, and the failure modes
RESULTS.md              findings (fill in after running)
scripts/                numbered, run in order
results/                logs and environment fingerprint (gitignored)
reference-megakernels/  upstream HazyResearch/Megakernels (gitignored, cloned by 00)
```

## Notes

- **H100 or B200 only.** The kernels are architecture-specific; the upstream
  Makefile silently defaults to B200 if `GPU` is unset.
- **Python 3.12 exactly.** Upstream requires `>=3.12` and the build locates
  headers via `PYTHON_VERSION`.
- **Torch from the cu128 index**, not default PyPI.
- The two benchmark paths measure decode differently — see the methodology note
  in `PLAN.md` before quoting any number.

## Upstream

- [HazyResearch/Megakernels](https://github.com/HazyResearch/Megakernels) — MIT
- [Look Ma, No Bubbles!](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) — the result we're reproducing
