"""Benchmark decode throughput against any OpenAI-compatible server.

Why not upstream's bench_engines.py: it sends `prompt=[0] * n_in`, a bare
token-ID array. vLLM accepts that; SGLang 0.5.9 rejects it with "Prompt cannot
be empty". Rather than special-casing one engine, this sends the same *text*
prompt to both -- the server tokenizes it, so both see an identical prompt.

Methodology is upstream's, and it is the right one: time a 1-token completion
and an N-token completion with the same prompt, then subtract. The difference
isolates decode by cancelling prefill, request overhead, and HTTP round-trip.

    tokens/s = (output_len - 1) / (t_N - t_1)

Usage:
    python bench_openai.py --model M --port 10210 --prompt-file p.txt \
        --output-len 128 --warmup 5 --iters 20
"""

import argparse
import statistics
import sys
import time

from openai import OpenAI


def timed_run(client, model, prompt, n_out, warmup, iters):
    """Return (mean, stdev) wall time for a completion of n_out tokens."""
    times = []
    for i in range(warmup + iters):
        start = time.perf_counter()
        resp = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=n_out,
            temperature=0.0,
            # Without this the model may stop early and the timing is meaningless.
            extra_body={"ignore_eos": True},
        )
        end = time.perf_counter()

        got = resp.usage.completion_tokens
        if got != n_out:
            sys.exit(f"Expected {n_out} completion tokens, got {got}. ignore_eos honoured?")

        if i >= warmup:
            times.append(end - start)

    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=10210)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--label", default="engine")
    args = ap.parse_args()

    with open(args.prompt_file) as f:
        prompt = f.read().strip()

    client = OpenAI(api_key="not-used", base_url=f"http://127.0.0.1:{args.port}/v1")

    one_mean, one_std = timed_run(
        client, args.model, prompt, 1, args.warmup, args.iters
    )
    many_mean, many_std = timed_run(
        client, args.model, prompt, args.output_len, args.warmup, args.iters
    )

    decode_time = many_mean - one_mean
    if decode_time <= 0:
        sys.exit(f"Non-positive decode time ({decode_time:.6f}s); timings are unusable.")

    decode_tokens = args.output_len - 1
    tps = decode_tokens / decode_time
    ms_per_token = 1000.0 * decode_time / decode_tokens

    print(f"engine:            {args.label}")
    print(f"prompt tokens:     (server-side) from {args.prompt_file}")
    print(f"1-token run:       {one_mean * 1000:.3f} ms  +/- {one_std * 1000:.3f}")
    print(f"{args.output_len}-token run:     {many_mean * 1000:.3f} ms  +/- {many_std * 1000:.3f}")
    print(f"decode window:     {decode_time * 1000:.3f} ms for {decode_tokens} tokens")
    print(f"ms per token:      {ms_per_token:.4f}")
    print(f"Tokens per second: {tps:.2f}")


if __name__ == "__main__":
    main()
