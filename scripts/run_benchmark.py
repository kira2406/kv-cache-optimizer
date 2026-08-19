"""
Runs the full policy comparison (baseline + attention/recency/random at
several budgets) and writes results/benchmark.csv, which scripts/plot.py
turns into the memory-vs-quality tradeoff figure for the README.

Example:
    python scripts/run_benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --budgets 128 256 512 --max-new-tokens 150
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kv_cache_optimizer.eval_harness import run_comparison, results_to_csv

LONG_PROMPT = (
    "You are a helpful assistant with expert knowledge of distributed systems. "
    "Below is a long design document. Read it carefully, then continue it with "
    "a detailed section on failure recovery.\n\n"
    "## System Overview\n"
    "The system is a distributed key-value store partitioned across N shards, "
    "each replicated three ways using a Raft-based consensus protocol. Clients "
    "write through a coordinator node, which forwards requests to the shard "
    "leader responsible for the relevant key range. Reads may be served by any "
    "in-sync replica depending on the requested consistency level.\n\n"
    "## Data Model\n"
    "Keys are arbitrary byte strings up to 1KB; values up to 1MB. Each shard "
    "maintains a sorted log-structured merge tree on local disk, with periodic "
    "compaction to bound read amplification. Metadata about shard ownership is "
    "stored in a separate strongly-consistent control plane.\n\n"
    "## Failure Recovery\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--budgets", type=int, nargs="+", default=[48, 96, 160])
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results" / "benchmark.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {args.model} on {device} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # eager attention is required here because run_comparison always includes
    # the "attention" scoring policy, which needs real attention weights back
    # from the forward pass (SDPA/flash kernels never return them).
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.eval()

    prompt_len = tokenizer(LONG_PROMPT, return_tensors="pt")["input_ids"].shape[1]
    print(f"Prompt is {prompt_len} tokens; generating {args.max_new_tokens} more "
          f"(total sequence ~{prompt_len + args.max_new_tokens} tokens).")
    for b in args.budgets:
        if b >= prompt_len + args.max_new_tokens:
            print(f"  NOTE: budget {b} >= total sequence length, eviction will never trigger "
                  f"for this budget (it will look identical to baseline). Use a tighter budget "
                  f"to see the policies actually diverge.")

    results = run_comparison(
        model, tokenizer, LONG_PROMPT, budgets=args.budgets, max_new_tokens=args.max_new_tokens
    )

    print()
    for r in results:
        print(
            f"{r.policy_name:28s} budget={str(r.budget):>6s} "
            f"peak={r.peak_cache_mb:7.3f}MB  tok/s={r.tokens_per_sec:6.2f}  "
            f"evicted={r.evicted_count:5d}  ppl={r.perplexity:.3f}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results_to_csv(results, args.out)
    print(f"\nWrote {args.out} (includes each policy's generated_text column for qualitative reading)")

    # qualitative sanity check: print the tightest-budget attention vs FIFO
    # output side by side, since the perplexity number alone is easy to
    # misread — reading the actual text is what tells you WHY one policy wins
    tightest = min(args.budgets)
    by_name = {r.policy_name: r for r in results}
    ctx = by_name.get(f"context_aware_budget{tightest}")
    fifo = by_name.get(f"fifo_baseline_budget{tightest}")
    if ctx and fifo:
        print(f"\n=== Generated text at tightest budget ({tightest} tokens) ===")
        print(f"[context-aware, ppl={ctx.perplexity:.2f}]:\n{ctx.generated_text[:400]}\n")
        print(f"[FIFO baseline, ppl={fifo.perplexity:.2f}]:\n{fifo.generated_text[:400]}\n")


if __name__ == "__main__":
    main()
