"""
Benchmark harness: runs generation under several cache policies on the same
prompt(s) and reports peak KV cache memory, decode throughput, and a
perplexity-based quality proxy, so you can plot the memory/quality tradeoff
that is the headline result of this project.

Usage: see scripts/run_benchmark.py
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from .cache_manager import CacheConfig
from .generate import generate_with_cache_management


@dataclass
class BenchmarkResult:
    policy_name: str
    scoring: str
    budget: Optional[int]
    prompt_len: int
    generated_len: int
    peak_cache_mb: float
    tokens_per_sec: float
    evicted_count: int
    perplexity: Optional[float] = None


@torch.no_grad()
def compute_perplexity(model, tokenizer, text: str) -> float:
    """Perplexity of `text` under `model`, full (unevicted) cache — used only
    as a quality proxy, not as part of the eviction pipeline itself."""
    if not text.strip():
        return float("nan")
    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt").to(device)["input_ids"]
    if ids.shape[1] < 2:
        return float("nan")
    out = model(input_ids=ids, labels=ids)
    return torch.exp(out.loss).item()


def run_policy(
    model,
    tokenizer,
    prompt: str,
    policy_name: str,
    scoring: str,
    budget: Optional[int],
    max_new_tokens: int,
    recency_window: int = 32,
    protected_prefix_len: int = 4,
) -> BenchmarkResult:
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    if budget is None:
        # "no eviction" baseline: budget effectively unbounded
        budget = prompt_len + max_new_tokens + 1

    cfg = CacheConfig(
        budget=budget,
        recency_window=min(recency_window, budget - protected_prefix_len),
        protected_prefix_len=min(protected_prefix_len, budget - 1),
        scoring=scoring,
    )

    start = time.perf_counter()
    text, stats = generate_with_cache_management(
        model, tokenizer, prompt, cfg, max_new_tokens=max_new_tokens
    )
    elapsed = time.perf_counter() - start

    ppl = compute_perplexity(model, tokenizer, text)

    return BenchmarkResult(
        policy_name=policy_name,
        scoring=scoring,
        budget=budget,
        prompt_len=prompt_len,
        generated_len=stats.generated_len,
        peak_cache_mb=stats.peak_cache_bytes / (1024 ** 2),
        tokens_per_sec=stats.generated_len / elapsed if elapsed > 0 else float("nan"),
        evicted_count=stats.evicted_count,
        perplexity=ppl,
    )


def run_comparison(
    model,
    tokenizer,
    prompt: str,
    budgets: List[int],
    max_new_tokens: int = 128,
) -> List[BenchmarkResult]:
    """
    Runs: an unbounded baseline, plus attention/recency/random policies at
    each budget in `budgets`. Returns a flat list of BenchmarkResult ready
    to dump to CSV / plot.
    """
    results = [
        run_policy(model, tokenizer, prompt, "baseline_no_eviction", "attention",
                   budget=None, max_new_tokens=max_new_tokens)
    ]
    for budget in budgets:
        for scoring, label in [
            ("attention", "context_aware"),
            ("recency", "fifo_baseline"),
            ("random", "random_ablation"),
        ]:
            results.append(
                run_policy(
                    model, tokenizer, prompt,
                    policy_name=f"{label}_budget{budget}",
                    scoring=scoring,
                    budget=budget,
                    max_new_tokens=max_new_tokens,
                )
            )
    return results


def results_to_csv(results: List[BenchmarkResult], path: str):
    import csv
    fields = list(BenchmarkResult.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)
