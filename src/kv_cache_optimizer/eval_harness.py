"""
Benchmark harness: runs generation under several cache policies on the same
prompt and reports peak KV cache memory, decode throughput, and a quality
metric, so you can plot the memory/quality tradeoff that is the headline
result of this project.

Quality metric: perplexity is computed via TEACHER FORCING against one
shared, fixed reference continuation (the unconstrained baseline's own
greedy output), not by scoring each policy's own free-running generation.
Scoring self-generated text is a rigged comparison: a policy that degrades
into repetitive output can score an artificially LOW perplexity on itself,
since repetitive text is easy to predict, even though it reads as broken to
a human. Teacher forcing against one shared target removes that confound;
every policy is judged on the same question ("how well can you still
predict what SHOULD come next"), not on how self-consistent its own
possibly-degenerate output is.

The free-running generated text is still produced and stored in the CSV,
for qualitative inspection, but it does not feed into the perplexity number.

Usage: see scripts/run_benchmark.py
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from .cache_manager import CacheConfig
from .generate import generate_with_cache_management, teacher_forced_nll_with_cache_management


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
    perplexity: Optional[float] = None       # teacher-forced against the shared reference
    generated_text: str = ""                  # this policy's own free-running output


@torch.no_grad()
def get_reference_continuation(model, tokenizer, prompt: str, max_new_tokens: int) -> torch.Tensor:
    """
    Runs one unconstrained (no-eviction) greedy generation and returns its
    token ids as a 1-D LongTensor. This becomes the shared, fixed target that
    every policy's teacher-forced perplexity is measured against, so all
    policies are compared on the same question rather than each being
    scored on its own (possibly degenerate) output.
    """
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    unbounded = CacheConfig(budget=prompt_len + max_new_tokens + 1, scoring="recency")
    text, _ = generate_with_cache_management(model, tokenizer, prompt, unbounded, max_new_tokens=max_new_tokens)
    ref_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    return ref_ids


def run_policy(
    model,
    tokenizer,
    prompt: str,
    reference_ids: torch.Tensor,
    policy_name: str,
    scoring: str,
    budget: Optional[int],
    max_new_tokens: int,
    recency_window: int = 32,
    protected_prefix_len: int = 4,
) -> BenchmarkResult:
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    if budget is None:
        budget = prompt_len + max_new_tokens + 1  # "no eviction" baseline: budget effectively unbounded

    cfg = CacheConfig(
        budget=budget,
        recency_window=min(recency_window, max(budget - protected_prefix_len, 1)),
        protected_prefix_len=min(protected_prefix_len, max(budget - 1, 0)),
        scoring=scoring,
    )

    # free-running generation: used for throughput/memory stats and for the
    # human-readable text stored in the CSV, NOT for the perplexity number
    start = time.perf_counter()
    text, gen_stats = generate_with_cache_management(
        model, tokenizer, prompt, cfg, max_new_tokens=max_new_tokens
    )
    elapsed = time.perf_counter() - start

    # teacher-forced perplexity against the shared reference continuation:
    # this is the number that belongs on the tradeoff plot
    _, perplexity, _ = teacher_forced_nll_with_cache_management(
        model, tokenizer, prompt, reference_ids, cfg
    )

    return BenchmarkResult(
        policy_name=policy_name,
        scoring=scoring,
        budget=budget,
        prompt_len=prompt_len,
        generated_len=gen_stats.generated_len,
        peak_cache_mb=gen_stats.peak_cache_bytes / (1024 ** 2),
        tokens_per_sec=gen_stats.generated_len / elapsed if elapsed > 0 else float("nan"),
        evicted_count=gen_stats.evicted_count,
        perplexity=perplexity,
        generated_text=text,
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
    each budget in `budgets`, all scored against ONE shared reference
    continuation. Returns a flat list of BenchmarkResult ready to dump to
    CSV / plot.
    """
    reference_ids = get_reference_continuation(model, tokenizer, prompt, max_new_tokens)

    results = [
        run_policy(model, tokenizer, prompt, reference_ids, "baseline_no_eviction", "recency",
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
                    model, tokenizer, prompt, reference_ids,
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
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)
