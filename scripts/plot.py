"""
Turns results/benchmark_summary.csv into the headline figure: peak cache
memory vs. perplexity, one line per policy, with vertical error bars showing
the std of perplexity across prompts at each budget. This is the plot to
put in the README.

Error bars matter here: without them, a small mean difference between
policies at n=6 prompts is impossible to distinguish from noise just by
looking at the figure.

Usage:
    python scripts/plot.py --csv results/benchmark_summary.csv --out results/tradeoff.png

If you ran with --single-prompt (results/benchmark.csv, no std column),
pass that file explicitly; the plot is drawn without error bars in that case.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(__file__).resolve().parent.parent / "results" / "benchmark_summary.csv"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results" / "tradeoff.png"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    has_std = "perplexity_std" in (rows[0].keys() if rows else [])

    # series: label -> list of (peak_mb, ppl_mean, ppl_std). The no-eviction
    # baseline is drawn separately as a horizontal reference line, not as a
    # point on the memory/quality curve.
    series = defaultdict(list)
    baseline_ppl = None
    for r in rows:
        label = r["scoring"]
        ppl_mean = float(r["perplexity_mean"] if has_std else r["perplexity"])
        ppl_std = float(r["perplexity_std"]) if has_std else 0.0
        peak_mb = float(r["peak_cache_mb_mean"] if has_std else r["peak_cache_mb"])
        if label == "baseline_no_eviction" or r.get("policy_name") == "baseline_no_eviction":
            baseline_ppl = ppl_mean
            continue
        series[label].append((peak_mb, ppl_mean, ppl_std))

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"attention": "o", "recency": "s", "random": "^"}
    labels = {"attention": "Context-aware (attention score)", "recency": "FIFO baseline", "random": "Random (ablation)"}
    for label, pts in series.items():
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        yerr = [p[2] for p in pts]
        ax.errorbar(
            xs, ys, yerr=yerr if has_std else None,
            marker=markers.get(label, "o"), label=labels.get(label, label),
            capsize=3, linewidth=1.5,
        )

    if baseline_ppl is not None:
        ax.axhline(baseline_ppl, color="gray", linestyle="--", linewidth=1, label="No eviction (reference)")

    ax.set_xlabel("Peak KV cache memory (MB)")
    ax.set_ylabel("Perplexity (lower is better)" + (", error bars = std across prompts" if has_std else ""))
    ax.set_title("Cache memory vs. generation quality by eviction policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
