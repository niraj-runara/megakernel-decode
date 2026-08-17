---
title: "Ultra-Low-Latency LLM Decode"
subtitle: "Where TileRT, TileLang and Megakernels Actually Sit — and What We Should Build"
author: "Niraj Dalavi — Runara"
date: "17 August 2026"
---

# 1. Executive summary

We set out to reproduce TileRT's ultra-low-latency inference results at small scale. Three findings:

**TileRT cannot be extended.** Its public repository is MIT-licensed Python wrapping two closed binary blobs, pinned to 8× NVIDIA B200. There is no source to modify.

**The underlying mechanism is real, and we verified it ourselves.** On a single rented H100 we measured a persistent-kernel ("megakernel") implementation of Llama-3.2-1B at **997.75 tokens/s at batch size 1** — 1.45× faster than vLLM, 1.56× faster than SGLang, and 6.78× faster than PyTorch eager. It reaches ~74% of the H100's memory bandwidth where vLLM reaches ~51%.

**No existing project has what we need, but the two halves both exist and are both open.** TileLang has the broad hardware coverage and the per-operation kernels, including for DeepSeek architectures. The megakernel work has the fusion mechanism but only for one model family. Nobody has joined them — which is precisely the layer TileRT keeps closed, and precisely where our contribution would be.

**Recommendation:** build a persistent-kernel scheduler on top of TileLang. For our stated target — DeepSeek V4 Flash on RTX PRO 6000 — this is the *only* viable route, because the alternative toolchain supports neither that GPU nor that model architecture.

---

# 2. Background: why not just use TileRT

TileRT is a runtime from the `tile-ai` group reporting ~500–600 tokens/s per user at batch size 1 on frontier models. Investigating it as a starting point produced a blocking finding.

The repository's loader is explicit:

```python
# tilert/__init__.py
_BACKENDS = {
    "deepseek_v3_2": "libtilert_dsv32.so",
    "glm5":          "libtilert_glm5.so",
}
ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL | os.RTLD_LAZY)
torch.ops.load_library(str(lib_path))
```

Those `.so` files ship in the PyPI wheel, not the repository. Everything of interest — the persistent kernel, warp specialization, tile scheduler, fused communication — is compiled inside them. The MIT licence covers the wrapper only.

## Hard constraints of the shipped artifact

| Constraint | Requirement | Consequence |
|---|---|---|
| Hardware | 8 × NVIDIA B200, single node | Kernels are ahead-of-time compiled for an 8-way tensor-parallel topology |
| Models | DeepSeek-V3.2, GLM-5 / 5.1 | One binary per model family; adding one needs their closed backend rebuilt |
| ABI | CUDA 13.2, Python 3.12, torch 2.11.0+cu130 | Linked against this exact stack |
| Batch size | 1 | The design point — but it rules out throughput evaluation |

**Conclusion:** no porting, no shrinking, no new models. The mechanism has to be rebuilt from open components.

---

# 3. The mechanism: why batch-1 decode is slow

At batch size 1, generating each token requires reading essentially the entire model from memory to do a very small amount of arithmetic. Every matrix multiply is really a matrix–*vector* product: memory-bound, and finished in microseconds.

A conventional engine issues roughly 100+ separate GPU kernel launches per token (for Llama-3.2-1B: 16 layers × ~7 operations). Two costs follow.

**CPU launch overhead.** The CPU must issue each launch. Our measurements show this starkly: PyTorch eager spent **863 ms of CPU time for 863 ms of wall time** — the CPU was launching kernels for the entire run while the GPU waited.

**The inter-kernel dependency barrier.** This is the subtler and more important cost. Every kernel must complete across the whole device before the next begins. The GPU drains and refills ~100+ times per token, and crucially you *cannot* begin loading layer N+1's weights while layer N is still computing.

The second cost is why modern engines remain far from peak even after solving the first. vLLM uses CUDA graphs, which largely eliminate CPU launch cost — yet it still reaches only ~51% of memory bandwidth, because the barrier is what "separate kernel" *means*.

**A megakernel removes both.** The entire forward pass becomes one kernel that launches once and stays resident on every SM, walking an instruction stream in place. Different warp groups specialize: some move data, some compute. Loading the next operation's weights overlaps with computing the current one — overlap that is inexpressible across a kernel boundary.

---

# 4. The three levels

This framework is the most useful thing to take from this document. It separates *kernel quality* from *execution structure*, which are independent and frequently conflated.

## Level 0 — Conventional: one launch per operation

An engine or framework orchestrates many individual kernels. Kernels may be world-class; the gaps between them remain.

- **Has:** excellent per-op kernels (cuBLAS, FlashAttention, TileLang), continuous batching, paged attention, mature serving infrastructure
- **Lacks:** any cross-operation overlap; a full-device barrier between every op
- **Ceiling:** structural. Better kernels raise where you land, but cannot remove the barrier.

Critically, **Level 0 is not one number.** Our measurements span it: PyTorch eager ~11%, SGLang ~47%, vLLM ~51% of roofline. Kernel quality matters enormously *within* the level.

## Level 1 — Megakernel on a single GPU

The whole forward pass is fused into one persistent kernel with an on-GPU scheduler walking an instruction stream.

- **Has:** the same kernels, now as "instructions"; no launch gaps; cross-operation prefetch and overlap
- **Lacks:** any multi-GPU story; the communication problem is untouched
- **Measured:** ~74% of memory bandwidth (this is our Step 1 result)

## Level 2 — Megakernel with fused communication

As Level 1, plus inter-GPU transfers issued *from inside* the resident kernel by dedicated warp groups, overlapping with compute.

- **Has:** everything in Level 1, plus communication that overlaps rather than serializes
- **Requires:** NVLink or equivalent; symmetric memory, device-side signalling, manual memory ordering
- **Status:** this is TileRT. No open project provides it as a complete model runtime.

There is also a **half-step** worth naming: a megakernel per GPU with NCCL collectives between launches. This exits the persistent kernel at every tensor-parallel boundary, reintroducing the barrier, and captures perhaps half the available win.

## What the levels buy

| Capability | Level 0 | Level 1 | Level 2 |
|---|---|---|---|
| Fast individual operations | yes | inherits | inherits |
| No inter-kernel barrier | no | **yes** | yes |
| Cross-op prefetch / overlap | no | **yes** | yes |
| Overlapped inter-GPU communication | no | n/a | **yes** |
| Works at batch > 1 | yes, any batch | yes, but advantage fades | same |

---

# 5. Where every project actually sits

| Project | Hardware coverage | Model coverage | Multi-GPU | Level | Open? |
|---|---|---|---|---|---|
| **TileLang** | Broad — sm_80/89/90a/100a/120, ROCm, CPU, Metal | Large kernel library incl. DeepSeek V3.2/V4, MLA, NSA, fused MoE | via NCCL | **0** | Yes |
| **TileScale** | Same base as TileLang | Same base | Fused collectives, single-node only | 0, with Level-2 *primitives* | Yes |
| **HazyResearch Megakernels** | H100 / B200 only | **Llama only** | No | **1** | Yes (MIT) |
| **ThunderKittens** | Hopper + datacenter Blackwell | Kernel library | via ParallelKittens | 0–1 substrate | Yes (MIT) |
| **Mirage MPK** | Unstated | Qwen3-8B demo | `world_size` param | ~1 | Yes (Apache-2.0) |
| **TileRT** | **8 × B200 only** | DeepSeek-V3.2, GLM-5 | Yes | **2** | **No — closed `.so`** |

**Read the diagonal.** Everything with broad hardware and model coverage is Level 0. The only open Level 1 covers one model family on one vendor's two GPUs. Nothing open is Level 2 as a complete runtime.

This is not an accident. The kernels are not the moat — the scheduler over them is. That is exactly why `tile-ai` open-sources TileLang and ships TileRT as a binary.

---

# 6. What we measured (Step 1)

We provisioned a single H100 and reproduced HazyResearch's megakernel end to end. This is our own evidence, not a citation.

## Configuration

| | |
|---|---|
| Model | meta-llama/Llama-3.2-1B-Instruct (bf16) |
| GPU | NVIDIA H100 80GB HBM3 (SXM), driver 580.126.20 |
| Toolchain | CUDA 12.8.93, torch 2.11.0+cu128, vLLM 0.27.1, SGLang 0.5.9 |
| Workload | 32-token prompt, 128 output tokens, batch size 1 |
| Method | 5 warmup + 20 timed iterations |

## Results

| Configuration | ms/token | Tokens/s | vs. megakernel | % of roofline |
|---|---|---|---|---|
| **Megakernel** | **1.0023** | **997.75** | 1.00× | **~74%** |
| vLLM 0.27.1 | 1.4528 | 688.31 | **1.45×** | ~51% |
| SGLang 0.5.9 | 1.5616 | 640.39 | **1.56×** | ~47% |
| PyTorch eager | 6.800 | 147.06 | **6.78×** | ~11% |

Roofline percentages are estimates: 1.24B parameters at bf16 is ~2.47 GB read per forward pass, against H100 HBM3 peak of 3350 GB/s. Weights only — it ignores KV cache and activations.

## Correctness

The build produced a single fused kernel. `ptxas` reported **96 registers, 16 barriers, 10,384 bytes shared memory, and zero register spills** for `-arch=sm_90a`. The mangled entry point confirms whole-pass fusion, listing `attention_partial`, `attention_reduction`, `rms_qkv_rope_append`, `downproj`, `o_proj`, `rms_upgate_silu`, `rms_lm_head`.

Numerics: KV caches match the reference path to ~1.6e-06 mean relative difference and attention intermediates are bit-exact. Activations differ a few percent mean relative, consistent with bf16 plus reordered accumulation. Generated text is coherent and correct.

## Against the published figures

| Metric | Published | Ours | |
|---|---|---|---|
| Forward pass | under 1 ms | 1.0023 ms | matches |
| vs. SGLang | >1.5× | 1.56× | matches |
| vs. vLLM | ~2.5× | **1.45×** | notably lower |
| HBM bandwidth | ~78% | ~73.6% | close |

**The vLLM discrepancy is a finding in itself.** We reproduce the SGLang comparison almost exactly but see 1.45× against vLLM where the blog reports ~2.5×. The published number predates vLLM 0.27.1 by roughly a year. **The mechanism still wins, but its margin over a current vLLM is materially narrower than the literature implies — anyone quoting 2.5× is quoting a stale baseline.**

## Caveats we should state internally

1. **Instruments are not identical.** The megakernel is timed in-process with CUDA events; the engines are timed through an HTTP server. Subtracting a 1-token run from a 128-token run cancels prefill and per-request overhead, but per-token server work stays inside their window. At 1.45×, vLLM sits just under the 1.5× threshold below which this matters — re-measure vLLM in-process before publishing a specific multiple.
2. **Not like-for-like in capability.** vLLM and SGLang do continuous batching, paged attention and multi-tenancy. The megakernel does one sequence of one model. It should win here, and does — but this says nothing about throughput serving.
3. **Single run, single host.** One instance, one afternoon. Worth a repeat on a different host before figures go external.

---

# 7. What this is and is not good for

## The batch-size relationship

The single most important operating principle:

> **The megakernel's value is roughly inversely proportional to batch size.**

At batch 1 each operation is tiny, so a ~5–10 µs launch plus a device-wide barrier dominates, and removing them is worth ~45%. At batch 64 each kernel does 64× the work on the *same* weights, launch cost becomes noise, and you are bandwidth-bound — where Level 0 is already near-optimal.

Megakernels do not *break* above batch 1; they stop being worth the effort, because there is little left to reclaim. **If we need low latency at batch > 1, this is the wrong tool** — that regime wants good kernels plus continuous batching, which vLLM and SGLang already do well.

Batched megakernels are being built (HazyResearch ships a public `ThroughputScheduleBuilder` with a batch-1024 configuration, though not the corresponding CUDA kernel; TileRT's roadmap lists batch 2/4/8) — but nothing usable is available.

## Where batch-1 latency genuinely pays

- **Long sequential decode.** Time-to-answer is tokens × time-per-token. A reasoning model emitting 10k chain-of-thought tokens takes **10.0 s at our 1.00 ms versus 14.5 s on vLLM** — a difference a user feels directly.
- **Agentic loops.** N sequential tool-calling round-trips, each on the critical path at effectively batch 1, so latency compounds.
- **Speculative decoding drafts.** The draft model is small, sequential, and gates the whole pipeline. A 1B draft model is exactly what we benchmarked.
- **Hard real-time.** Voice assistants with sub-100 ms budgets, robotics control loops, and high-frequency trading — which TileRT's own README names explicitly.

## The economics we must state alongside it

At batch 1 we occupy an entire H100 for one user at ~1000 tokens/s, where a batched engine would push perhaps 10–20k tokens/s aggregate on the same card. That is roughly **10–20× the cost per token.**

This only makes sense where latency carries value exceeding that multiple. Both production TileRT deployments are branded as premium tiers — *Z.ai GLM-5.1 HighSpeed*, *Xiaomi MiMo V2.5 Pro UltraSpeed* — sitting **alongside** normally-priced tiers, not replacing them.

**Positioning: this is a premium latency tier, not a general serving win.** Anyone evaluating it on tokens-per-dollar will conclude it is bad, because on that axis it is.

---

# 8. Path A — extend TileLang upward to Level 1

## What TileLang already gives us

This is a substantial head start, and it is the hard part to write from scratch:

- **Hardware breadth.** Targets `sm_80 / sm_89 / sm_90a / sm_100a / sm_120`, plus ROCm, CPU, Metal and WebGPU backends. An autotuner absorbs most tile-size retuning across architectures.
- **A large kernel library**, including specifically for our target family: `examples/deepseek_v32/` (sparse MLA forward, FP8 lightning indexer, top-k selector), `examples/deepseek_v4/`, `examples/deepseek_mla/`, `examples/deepseek_nsa/`, `examples/fusedmoe/`, `examples/flash_decoding/`.
- **NVFP4 support** via `blockscaled_gemm_sm100` and `gemm_tcgen05`.
- **Warp-specialization primitives** — `tilelang/language/ws_schedule.py`, `tile_schedule.py`.

## What is missing — and would be our contribution

TileLang is a language for writing *individual* kernels. Its own DeepSeek V3.2 inference demo is unambiguously Level 0: a PyTorch module tree calling separate `T.Kernel` launches with `dist.all_reduce` between them, started via `torchrun`.

Reaching Level 1 requires five components that do not exist in any open project:

1. **Model → instruction decomposition.** A pass turning a model graph into a linear stream of typed instructions.
2. **Instruction → SM assignment.** A scheduler mapping instructions onto streaming multiprocessors, balancing load and respecting dependencies.
3. **Persistent kernel driver.** The resident loop that reads the instruction stream and dispatches to the right kernel body.
4. **On-chip buffer management.** Activations must live in shared memory / registers across instruction boundaries rather than round-tripping to HBM.
5. **Dependency and barrier tracking.** Correct ordering without a device-wide sync between operations.

For reference, HazyResearch's implementation of exactly this layer is `scheduler.py`, `instructions.py`, `dispatch.py`, `python_vm.py` and `mk.py` — a few thousand lines above their per-instruction CUDA files. That is the shape and scale of the work.

## Effort and risk

**Effort:** months, not weeks. This is genuine systems engineering.

**Risk:** moderate and well-understood. The mechanism is proven (we proved it), the kernels exist, and a working reference implementation of the scheduler layer is readable and MIT-licensed.

**Payoff:** hardware breadth comes largely for free because TileLang already retargets, and the DeepSeek kernels already exist. New models within a family become cheap.

---

# 9. Path B — extend Level 1 outward from the existing megakernel

The alternative is to take HazyResearch's working Level 1 implementation and widen it. Cost depends entirely on *what* we widen.

## Same architecture family — genuinely just a recompile

The model's dimensions are compile-time C++ template parameters. Our build emitted:

```
globals_t<16, 2048, 8192, 64, 32, 8, 16, 16, 132>
```

That is 16 layers, 2048 hidden, 8192 intermediate, 64 head dim, 32 heads, 8 KV heads — Llama-3.2-1B's configuration — and **132, the SM count of an H100.** Both the model *and* the GPU are baked in at compile time.

Upstream already ships configurations for Llama-3.2-1B and Llama-3.1-8B from the same kernel source. Anything shaped like Llama (RMSNorm + RoPE + GQA + SwiGLU) — Qwen, Mistral, most modern dense models — is **days** of work.

## New architecture — real kernel engineering

Here we write rather than recompile. The megakernel is assembled from per-instruction CUDA files:

```
demos/low-latency-llama/
  attention_partial.cu        attention_reduction.cu
  rms_matvec_rope_append.cu   upgate.cu
  matvec_adds.cu              rms_lm_head.cu
```

plus the Python model definition and scheduler entries. A model with MLA instead of GQA, or an MoE router, needs new instruction kernels and new scheduling decisions: **weeks to months per architecture family.** This is precisely the cost TileRT cites for why it supports two models.

## New hardware — mostly retuning, with one hard blocker

The Makefile already carries `sm_80` (A100), `sm_89` (4090), `sm_90a` (H100), `sm_100a` (B200). Moving between them means changing the arch flag and the SM count, then retuning tile shapes and pipeline depth — **days to weeks.**

**But there is no `sm_120` path**, and ThunderKittens targets Hopper WGMMA and datacenter-Blackwell TCGEN05 only. For workstation Blackwell this is a port of ThunderKittens itself, not a flag change.

---

# 10. Our stated target: DeepSeek V4 Flash on 4× RTX PRO 6000

## The model

DeepSeek V4 Flash is **284B total / 13B active parameters**, MoE, with *hybrid* attention (Compressed Sparse Attention + Heavily Compressed Attention), manifold-constrained hyper-connections, FP4 experts + FP8 non-expert weights, and 1M context.

## The hardware

| | RTX PRO 6000 Blackwell | B200 (for contrast) |
|---|---|---|
| Memory | 96 GB GDDR7 | 180 GB HBM3e |
| Bandwidth | 1,792 GB/s | ~8 TB/s |
| Interconnect | **PCIe Gen5, ~128 GB/s — no NVLink** | NVLink, ~900 GB/s |
| TMA | **No** | Yes |
| NVFP4 | Yes | Yes |
| Compute capability | `sm_120` | `sm_100a` |

## Assessment

**Memory capacity is fine.** 284B at FP4/FP8 mixed is roughly 150–170 GB; 4 × 96 GB = 384 GB, comfortable with room for KV cache. This is presumably why the configuration was chosen.

**Three problems compound, and the third is likely decisive.**

*ThunderKittens has no path for this GPU, and the GPU has no TMA.* TK's warp-specialization design depends on TMA async bulk copies — the mechanism that lets memory warps run ahead of compute warps. Removing it means redesigning data movement.

*The model is maximally difficult.* Nothing about MoE + hybrid sparse attention + mHC is Llama-shaped. TileLang's `examples/deepseek_v4/` contains `sparse_attn_fwd_sm90.py` and `fp8_fp4_gemm_1d1d_sm100.py` — **sm90 and sm100 only.** The reference kernels that exist target the two architectures we are not using.

*No NVLink, and MoE needs all-to-all every layer.* Rough arithmetic:

| | Estimate |
|---|---|
| Active weights per token | ~7.5 GB → ~1.9 GB per GPU sharded 4 ways |
| Time at 1792 GB/s | ~1.05 ms/token (~950 tok/s ceiling) |
| All-to-all, ~60 layers × 2, PCIe-latency-bound | **~2.4 ms/token** |

Compute and memory are adequate. **The interconnect is plausibly 2–3× the entire rest of the forward pass**, landing around 250–330 tok/s. Over NVLink those transfers would be a fraction of that.

Treat these as order-of-magnitude, but the direction is clear: on a PCIe-only box, fused device-side communication stops paying for itself, and MoE all-to-all is the most interconnect-hungry pattern there is.

## Implication

**For this target, Path A is not merely preferable — it is the only viable route.** TileLang supports `sm_120` and already has DeepSeek kernels; ThunderKittens supports neither.

Separately, the hardware choice deserves revisiting. The appeal of RTX PRO 6000 is 96 GB at low cost, which is real — but it buys capacity by giving up interconnect and toolchain support, the two things this workload is most sensitive to. **Datacenter Blackwell (B200) provides NVFP4 *and* NVLink *and* TMA *and* toolchain support in both TileLang and ThunderKittens.** If NVFP4 is the motivation, that is the supported path.

---

# 11. Recommendation and next steps

**Strategic position:** TileLang supplies the best materials; the megakernel work supplies the proven mechanism; nobody has assembled them. That assembly is our contribution, and it is the same layer TileRT keeps closed as its moat.

Sequenced so each step is cheap and can kill the next:

1. **Verify AutoMegaKernel (≈1 day, no GPU).** A recent project ([arXiv 2606.09682](https://arxiv.org/abs/2606.09682)) claims to compile a model into one "provably-correct, self-retargeting" CUDA megakernel that self-tunes past cuBLAS at batch-1 decode. **Unverified by us.** If the retargeting claim holds it targets our exact goal and could collapse much of Path A. Cheapest possible next action; do it first.

2. **Add TensorRT-LLM as a baseline (≈1 day, 1× H100).** We beat vLLM and SGLang; we have *not* tested NVIDIA's own engine, which is typically fastest for latency and is where NVFP4 support landed first. Until we do, we cannot claim to beat NVIDIA. Our harness already speaks the OpenAI API and `trtllm-serve` exposes it.

3. **Validate fused communication (≈2 weeks, 2× H100 SXM).** Run TileScale's existing `gemm_allreduce` example — writing no kernels of our own — against a separate cuBLAS-GEMM-then-NCCL baseline, profiling to confirm genuine overlap. Verify NVLink with `nvidia-smi topo -m` first. This is the gating unknown for anything multi-GPU.

4. **NVFP4 microbenchmark (≈days, 1× Blackwell).** Blockscaled GEMM, NVFP4 versus FP8. Answers the precision question in isolation, on one GPU, no interconnect involved.

5. **Then decide the target hardware and begin the scheduler**, with all four answers in hand.

Steps 1, 2 and 4 are each about a day and can run in parallel with step 3.

## Open questions

- Does AutoMegaKernel's self-retargeting claim survive scrutiny?
- Does the megakernel still win against TensorRT-LLM?
- What is the real PCIe all-to-all cost on 4× RTX PRO 6000? One measurement decides whether that target is reachable.
- Is our hardware choice fixed, or can we still trade capacity for interconnect?
- Where exactly does the megakernel advantage decay as batch goes 1 → 2 → 4 → 8? TileRT's own roadmap admits this is unmapped; it is a publishable question and we are equipped to answer it.

---

# 12. Sources

**Primary — our own work**

- `github.com/niraj-runara/megakernel-decode` — plan, scripts, results and raw logs from Step 1

**Projects**

- `github.com/tile-ai/TileRT` — the closed runtime
- `github.com/tile-ai/tilelang` — kernel DSL, broad hardware coverage
- `github.com/tile-ai/tilescale` — distributed extension, fused collectives
- `github.com/HazyResearch/Megakernels` — the Level 1 implementation we reproduced
- `github.com/HazyResearch/ThunderKittens` — tile primitives; includes ParallelKittens
- `github.com/mirage-project/mirage` — Mirage Persistent Kernel
- `github.com/RightNow-AI/AutoMegaKernel` — self-retargeting claim, unverified

**Writing**

- Hazy Research, *Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B*
- Hazy Research, *ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels*
- SemiAnalysis, *Ultra-High Interactivity on NVIDIA GPUs? — TileRT InferenceX*
- *Compiling LLMs into a MegaKernel: A Path to Low-Latency Inference* (Zhihao Jia)
