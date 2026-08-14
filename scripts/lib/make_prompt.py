"""Emit a prompt string that tokenizes to exactly N tokens.

Both benchmark harnesses specify the prompt differently -- generate.py takes a
string, bench_engines.py takes a length. Comparing a 32-token prompt against a
64-token one would quietly bias the result, so we pin the string here.

Usage:
    python make_prompt.py --model meta-llama/Llama-3.2-1B-Instruct --ntok 32
"""

import argparse

from transformers import AutoTokenizer

FILLER = (
    "The history of computing hardware spans a long period, beginning with "
    "mechanical calculating devices and progressing through vacuum tubes, "
    "transistors, integrated circuits, and eventually the massively parallel "
    "processors used for machine learning workloads today. Each generation "
    "changed which optimisations mattered most to practitioners."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--ntok", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    # add_special_tokens=True matches how generate.py encodes in non-chat mode,
    # so the count we print is the count that harness will see.
    ids = tok(FILLER, add_special_tokens=True)["input_ids"]
    if len(ids) < args.ntok:
        raise SystemExit(
            f"Filler text is only {len(ids)} tokens; need at least {args.ntok}."
        )

    truncated = ids[: args.ntok]
    text = tok.decode(truncated, skip_special_tokens=True)

    # Re-encoding a decoded string can shift the count by a token, so verify
    # rather than assume, and trim until it settles.
    for _ in range(8):
        n = len(tok(text, add_special_tokens=True)["input_ids"])
        if n == args.ntok:
            break
        if n > args.ntok:
            text = tok.decode(
                tok(text, add_special_tokens=False)["input_ids"][: -(n - args.ntok)],
                skip_special_tokens=True,
            )
        else:
            raise SystemExit(f"Could not converge: got {n}, wanted {args.ntok}")
    else:
        raise SystemExit("Could not converge on exact token count")

    print(text)


if __name__ == "__main__":
    main()
