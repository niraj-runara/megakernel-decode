# Step 1 results

**Status:** not yet run — awaiting H100.

Fill this in from `results/` after `03_bench_mk.sh` and `04_bench_engines.sh`.
Paste `results/environment.txt` into the fingerprint section; without it the
numbers are not reproducible.

---

## Configuration

| | |
|---|---|
| Model | meta-llama/Llama-3.2-1B-Instruct |
| GPU | _fill in_ |
| Prompt / output tokens | 32 / 128 |
| Batch size | 1 |
| Warmup / iterations | 5 / 20 |

## Correctness

| Check | Result |
|---|---|
| `diff_test.py full` | _pass / fail_ |
| Generated text coherent | _yes / no_ |
| Agrees with torch baseline | _yes / no_ |

## Latency

| Configuration | Tokens/s | Avg time (ms) | vs. megakernel |
|---|---|---|---|
| Megakernel (`mode=mk`) | | | 1.00× |
| PyTorch eager (`mode=torch`) | | | |
| vLLM | | | |
| SGLang | | | |

Published reference for comparison: sub-1ms forward pass, ~78% of H100 memory
bandwidth, ~2.5× vLLM and >1.5× SGLang.

## Environment fingerprint

```
paste results/environment.txt here
```

---

## Verdict

Against the success criteria in `PLAN.md`:

- [ ] Megakernel builds on H100
- [ ] Numerics pass and text is coherent
- [ ] Beats PyTorch eager by a wide margin
- [ ] Beats both vLLM and SGLang at batch 1
- [ ] Forward pass in the neighbourhood of 1 ms
- [ ] Environment captured

**Go / no-go for step 2:** _fill in_

## Notes and deviations

_Anything that differed from plan — build fixes, version pins, retries, and
anything that would change how the numbers should be read._
