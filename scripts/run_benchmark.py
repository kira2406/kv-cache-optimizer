"""
Runs the full policy comparison (baseline + attention/recency/random at
several budgets) across a diverse set of prompts (see
kv_cache_optimizer.prompts.PROMPTS), and writes:

  results/benchmark_raw.csv      one row per (prompt, policy, budget),
                                  includes each policy's generated_text
  results/benchmark_summary.csv  one row per (policy, budget), aggregated
                                  across prompts: mean/std perplexity,
                                  mean throughput/memory, and a paired
                                  win-count vs. the FIFO baseline

scripts/plot.py reads the summary CSV to draw the memory-vs-quality
tradeoff figure for the README, with error bars from the per-prompt std.

Example:
    python scripts/run_benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --budgets 128 256 512 --max-new-tokens 150

Use --single-prompt to run the old one-prompt-only mode (faster, useful
for a quick smoke test while iterating on the cache manager itself; not
what should end up in the README).
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kv_cache_optimizer.eval_harness import (
    run_comparison,
    run_multi_prompt_comparison,
    results_to_csv,
    summarize_results,
    summary_to_csv,
)
from kv_cache_optimizer.prompts import PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--budgets", type=int, nargs="+", default=[48, 96, 160])
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--recency-window", type=int, default=32,
                     help="Guaranteed most-recent tokens exempt from scoring, applied EQUALLY "
                          "to all three policies (attention, recency/FIFO, random). Set to 0 for "
                          "the strictest fair test: no policy gets a free 'always keep recent N' "
                          "floor, every policy earns its whole non-protected budget on its own "
                          "criterion. Default 32 matches the original setup.")
    ap.add_argument("--single-prompt", action="store_true",
                     help="Run only the original distributed-systems prompt, for a quick smoke test.")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
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

    for prompt_id, category, text in ([PROMPTS[0]] if args.single_prompt else PROMPTS):
        prompt_len = tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
        for b in args.budgets:
            if b >= prompt_len + args.max_new_tokens:
                print(f"  NOTE: prompt '{prompt_id}' budget {b} >= total sequence length "
                      f"({prompt_len + args.max_new_tokens}), eviction will never trigger for "
                      f"this (prompt, budget) pair.")

    if args.single_prompt:
        results = run_comparison(
            model, tokenizer, PROMPTS[0][2], budgets=args.budgets,
            max_new_tokens=args.max_new_tokens, prompt_id=PROMPTS[0][0],
            recency_window=args.recency_window,
        )
    else:
        print(f"\nRunning {len(PROMPTS)} prompts x {len(args.budgets)} budgets x 3 policies "
              f"+ {len(PROMPTS)} baselines (recency_window={args.recency_window}) ...")
        results = run_multi_prompt_comparison(
            model, tokenizer, PROMPTS, budgets=args.budgets, max_new_tokens=args.max_new_tokens,
            recency_window=args.recency_window,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / ("benchmark_raw.csv" if not args.single_prompt else "benchmark.csv")
    results_to_csv(results, str(raw_path))
    print(f"\nWrote {raw_path} ({len(results)} rows, includes generated_text for qualitative reading)")

    if not args.single_prompt:
        summary = summarize_results(results)
        summary_path = out_dir / "benchmark_summary.csv"
        summary_to_csv(summary, str(summary_path))
        print(f"Wrote {summary_path}\n")

        print(f"{'policy':16s} {'budget':>6s} {'n':>3s}  {'ppl (mean±std)':>18s}  "
              f"{'tok/s':>6s}  {'peak MB':>8s}  {'wins vs FIFO':>13s}")
        for s in summary:
            print(
                f"{s.scoring:16s} {str(s.budget):>6s} {s.n_prompts:>3d}  "
                f"{s.perplexity_mean:7.3f} \u00b1 {s.perplexity_std:6.3f}  "
                f"{s.tokens_per_sec_mean:6.2f}  {s.peak_cache_mb_mean:8.3f}  "
                f"{s.wins_vs_fifo or '-':>13s}"
            )
        print(
            "\n'wins vs FIFO' = how many prompts this policy scored a LOWER "
            "(better) perplexity than the FIFO baseline at the same budget. "
            "Check this alongside the mean: a policy can win on average while "
            "losing on most individual prompts if one prompt swings the mean."
        )
    else:
        for r in results:
            print(
                f"{r.policy_name:28s} budget={str(r.budget):>6s} "
                f"peak={r.peak_cache_mb:7.3f}MB  tok/s={r.tokens_per_sec:6.2f}  "
                f"evicted={r.evicted_count:5d}  ppl={r.perplexity:.3f}"
            )


if __name__ == "__main__":
    main()
