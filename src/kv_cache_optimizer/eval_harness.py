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

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    prompt_id: str = "default"                # which prompt this row came from (multi-prompt runs)


@dataclass
class SummaryRow:
    """One (scoring, budget) policy aggregated across every prompt it ran on."""
    scoring: str
    budget: Optional[int]
    n_prompts: int
    perplexity_mean: float
    perplexity_std: float
    peak_cache_mb_mean: float
    tokens_per_sec_mean: float
    evicted_count_mean: float
    # paired win-count vs. the FIFO baseline at the same budget: how many
    # prompts this policy scored a LOWER (better) perplexity than FIFO on.
    # More informative than the mean alone at n=6: a policy can win the
    # average by doing much better on one prompt while losing on most others.
    wins_vs_fifo: Optional[str] = None


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
    prompt_id: str = "default",
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
        prompt_id=prompt_id,
    )


def run_comparison(
    model,
    tokenizer,
    prompt: str,
    budgets: List[int],
    max_new_tokens: int = 128,
    prompt_id: str = "default",
) -> List[BenchmarkResult]:
    """
    Runs: an unbounded baseline, plus attention/recency/random policies at
    each budget in `budgets`, all scored against ONE shared reference
    continuation, for a SINGLE prompt. Returns a flat list of BenchmarkResult
    ready to dump to CSV / plot.

    For a benchmark that generalizes beyond one prompt's writing style, use
    run_multi_prompt_comparison instead.
    """
    reference_ids = get_reference_continuation(model, tokenizer, prompt, max_new_tokens)

    results = [
        run_policy(model, tokenizer, prompt, reference_ids, "baseline_no_eviction", "recency",
                   budget=None, max_new_tokens=max_new_tokens, prompt_id=prompt_id)
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
                    prompt_id=prompt_id,
                )
            )
    return results


def run_multi_prompt_comparison(
    model,
    tokenizer,
    prompts: List[Tuple[str, str, str]],
    budgets: List[int],
    max_new_tokens: int = 128,
) -> List[BenchmarkResult]:
    """
    Runs run_comparison independently for every (prompt_id, category, text)
    in `prompts` (see prompts.PROMPTS) and concatenates the results, tagging
    each row with its prompt_id. Each prompt gets its OWN reference
    continuation (a prompt's teacher-forcing target has to be a plausible
    continuation of that specific prompt).

    Returns the full flat list of per-(prompt, policy, budget) rows. Use
    summarize_results() to aggregate this into per-policy statistics.
    """
    all_results: List[BenchmarkResult] = []
    for i, (prompt_id, category, text) in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] prompt={prompt_id} ({category})")
        all_results.extend(
            run_comparison(
                model, tokenizer, text, budgets=budgets,
                max_new_tokens=max_new_tokens, prompt_id=prompt_id,
            )
        )
    return all_results


def summarize_results(results: List[BenchmarkResult]) -> List[SummaryRow]:
    """
    Aggregates per-prompt BenchmarkResults into one SummaryRow per
    (scoring, budget) policy: mean/std across prompts, plus a paired
    win-count against the FIFO baseline at the same budget.

    The win-count matters at small n: two policies can have very similar
    MEAN perplexity while one wins on 5/6 prompts and the other wins on
    1/6 by a large margin on an easy prompt. The mean alone hides that.
    """
    # group raw perplexity by (scoring, budget) -> {prompt_id: perplexity}.
    # baseline_no_eviction rows get their own bucket, keyed with budget=None
    # and scoring="baseline_no_eviction" (distinct from the "recency" scoring
    # used for the FIFO policy) so the summary keeps a no-eviction reference
    # row without it being mistaken for a FIFO budget.
    by_policy: Dict[Tuple[str, Optional[int]], Dict[str, BenchmarkResult]] = defaultdict(dict)
    for r in results:
        key = ("baseline_no_eviction", None) if r.policy_name == "baseline_no_eviction" else (r.scoring, r.budget)
        by_policy[key][r.prompt_id] = r

    fifo_by_budget: Dict[Optional[int], Dict[str, float]] = defaultdict(dict)
    for (scoring, budget), by_prompt in by_policy.items():
        if scoring == "recency":
            for pid, r in by_prompt.items():
                fifo_by_budget[budget][pid] = r.perplexity

    summary: List[SummaryRow] = []
    for (scoring, budget), by_prompt in sorted(by_policy.items(), key=lambda kv: (kv[0][1] or 0, kv[0][0])):
        ppls = [r.perplexity for r in by_prompt.values()]
        wins_str = None
        if scoring != "recency":
            fifo_scores = fifo_by_budget.get(budget, {})
            shared = [pid for pid in by_prompt if pid in fifo_scores]
            wins = sum(1 for pid in shared if by_prompt[pid].perplexity < fifo_scores[pid])
            if shared:
                wins_str = f"{wins}/{len(shared)}"

        summary.append(SummaryRow(
            scoring=scoring,
            budget=budget,
            n_prompts=len(by_prompt),
            perplexity_mean=statistics.mean(ppls),
            perplexity_std=statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
            peak_cache_mb_mean=statistics.mean(r.peak_cache_mb for r in by_prompt.values()),
            tokens_per_sec_mean=statistics.mean(r.tokens_per_sec for r in by_prompt.values()),
            evicted_count_mean=statistics.mean(r.evicted_count for r in by_prompt.values()),
            wins_vs_fifo=wins_str,
        ))
    return summary


def results_to_csv(results: List[BenchmarkResult], path: str):
    import csv
    fields = list(BenchmarkResult.__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)


def summary_to_csv(summary: List[SummaryRow], path: str):
    import csv
    fields = list(SummaryRow.__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary:
            writer.writerow(r.__dict__)
