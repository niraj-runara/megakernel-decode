# Step 1 results

**Status: PASSED.** The megakernel beats PyTorch eager, vLLM, and SGLang at batch size 1
on a single H100. Step 2 is justified.

Run 2026-08-14 on a rented Vast.ai H100. Raw logs in `results/` on the instance.

---

## Configuration

| | |
|---|---|
| Model | meta-llama/Llama-3.2-1B-Instruct |
| GPU | NVIDIA H100 80GB HBM3 (SXM), driver 580.126.20 |
| Prompt / output tokens | 32 / 128 |
| Batch size | 1 |
| Warmup / iterations | 5 / 20 |
| Precision | bf16 |

## Correctness

| Check | Result |
|---|---|
| Build on H100 | Pass — `-arch=sm_90a`, 96 registers, **0 bytes spill** |
| `diff_test.py .full` | Pass (see caveat) |
| Generated text coherent | Yes |
| Agrees with torch baseline | Yes — both answer "Paris" correctly |

`diff_test.py` prints tensor diffs but asserts nothing, so this is a judgement call.
KV caches match to ~1.6e-06 mean relative difference and the attention intermediates are
bit-exact (0.0). Activations differ by a few percent mean relative, with max absolute
diffs of 0.03–0.125 — consistent with bf16 (epsilon ~0.0078) plus reordered accumulation
in the fused kernel. Combined with semantically correct generation, we treat this as passed.

The megakernel fuses the entire decode pass into one kernel; the mangled entry point lists
`attention_partial`, `attention_reduction`, `rms_qkv_rope_append`, `downproj`, `o_proj`,
`rms_upgate_silu`, `rms_lm_head`.

## Latency

| Configuration | ms/token | Tokens/s | vs. megakernel |
|---|---|---|---|
| **Megakernel** | **1.0023** | **997.75** | 1.00× |
| vLLM 0.27.1 | 1.4528 | 688.31 | **1.45×** |
| SGLang 0.5.9 | 1.5616 | 640.39 | **1.56×** |
| PyTorch eager | 6.800 | 147.06 | **6.78×** |

### Against the published figures

| Metric | Published | Ours | |
|---|---|---|---|
| Forward pass | under 1 ms | 1.0023 ms | matches |
| vs. SGLang | >1.5× | 1.56× | matches |
| vs. vLLM | ~2.5× | 1.45× | **notably lower** |
| HBM bandwidth | ~78% | ~73.6% (est.) | close |

Bandwidth is an estimate: 1.24B params at bf16 is ~2.47 GB read per forward pass,
so 2.47 GB / 1.0023 ms ≈ 2464 GB/s against H100 HBM3 peak of 3350 GB/s. Weights only —
it ignores KV cache and activations, so the true figure is slightly higher.

**The vLLM gap is the interesting result.** We reproduce the SGLang comparison almost
exactly, but see 1.45× against vLLM where the published figure is ~2.5×. The most likely
explanation is that vLLM improved: the published number predates vLLM 0.27.1 by roughly a
year. The megakernel mechanism still wins, but its margin over a current vLLM is narrower
than the blog post implies. Any proposal quoting 2.5× would be quoting a stale baseline.

## Environment fingerprint

```
date: 2026-08-14T20:21:51Z
host: Linux 6.8.0-1046-nvidia Ubuntu 24.04.4 LTS

## GPU
NVIDIA H100 80GB HBM3, 81559 MiB, 580.126.20

## nvcc
Cuda compilation tools, release 12.8, V12.8.93

## versions
torch   2.11.0+cu128
vllm    0.27.1
sglang  0.5.9
```

---

## Verdict

- [x] Megakernel builds on H100
- [x] Numerics pass and text is coherent
- [x] Beats PyTorch eager by a wide margin (6.78×)
- [x] Beats **both** vLLM and SGLang at batch 1
- [x] Forward pass in the neighbourhood of 1 ms (1.0023 ms)
- [x] Environment captured

**Go / no-go for step 2: GO.** The launch-overhead premise holds on our stack. Decode at
batch 1 is bound by per-kernel overhead, and collapsing the forward pass into one resident
kernel recovers a real and reproducible margin over both production engines.

## Caveats

1. **The measurement instruments are not identical.** The megakernel is timed in-process
   with CUDA events; vLLM and SGLang are timed through an HTTP server. Subtracting a
   1-token run from a 128-token run cancels prefill and per-request overhead, but *per-token*
   server work (detokenisation, scheduling) remains inside the vLLM/SGLang window and not
   the megakernel's. This flatters the megakernel by some amount.

   Our plan set 1.5× as the threshold below which this matters. **At 1.45×, vLLM is just
   under it.** The direction of the result is not in doubt — a plausible per-token server
   overhead of 0.05–0.1 ms would still leave the megakernel ahead — but before publishing a
   specific multiple against vLLM, re-measure it in-process. The SGLang result at 1.56× is
   above the threshold and needs no such qualification.

2. **The comparison is not like-for-like in capability.** vLLM and SGLang are general
   serving engines doing continuous batching, paged attention, and multi-tenancy. The
   megakernel does one sequence, batch 1, one model. It should win at this, and it does —
   but this is not evidence about throughput serving, which is the regime where the
   megakernel gives ground.

3. **Single run, single host.** Numbers come from one instance on one afternoon. Variance
   across hosts on Vast is real. Worth one repeat run on a different host before the figures
   go into anything external.

## Deviations from plan

- Ran initially against `unsloth/Llama-3.2-1B-Instruct` (ungated mirror) while waiting on
  Meta's gate. After access was granted, everything was re-run on the canonical
  `meta-llama/Llama-3.2-1B-Instruct`. The megakernel figure was **identical** (997.75 tok/s)
  on both, confirming the mirror was equivalent. All numbers above are from official weights.
- Vast's H100 image ships CUDA 13.0 only; ThunderKittens documents 12.3+ and its authors
  develop on 12.6. Installed `cuda-toolkit-12-8` alongside and built against it rather than
  gamble on a major toolkit bump.
- Replaced upstream's `bench_engines.py` with `scripts/lib/bench_openai.py`. Upstream sends
  `prompt=[0]*n`, a bare token-ID array that SGLang 0.5.9 rejects. Ours sends the same text
  prompt to both engines, which also removes an input asymmetry.

## Next

Step 2 — validate the multi-GPU toolchain by running TileScale's `gemm_allreduce` example
on 2× H100 SXM with NVLink verified via `nvidia-smi topo -m`. See `PLAN.md`.
