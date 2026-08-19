"""
Turns results/benchmark.csv into the headline figure: peak cache memory vs.
perplexity, one line per policy. This is the plot to put in the README.

Usage:
    python scripts/plot.py --csv results/benchmark.csv --out results/tradeoff.png
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(__file__).resolve().parent.parent / "results" / "benchmark.csv"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results" / "tradeoff.png"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    baseline = [r for r in rows if r["policy_name"] == "baseline_no_eviction"]

    series = defaultdict(list)  # label -> list of (peak_cache_mb, perplexity)
    for r in rows:
        if r["policy_name"] == "baseline_no_eviction":
            continue
        label = r["scoring"]
        series[label].append((float(r["peak_cache_mb"]), float(r["perplexity"])))

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"attention": "o", "recency": "s", "random": "^"}
    labels = {"attention": "Context-aware (attention score)", "recency": "FIFO baseline", "random": "Random (ablation)"}
    for label, pts in series.items():
        pts = sorted(pts)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=markers.get(label, "o"), label=labels.get(label, label))

    if baseline:
        b = baseline[0]
        ax.axhline(float(b["perplexity"]), color="gray", linestyle="--", linewidth=1,
                    label=f"No eviction (peak {float(b['peak_cache_mb']):.0f} MB)")

    ax.set_xlabel("Peak KV cache memory (MB)")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Cache memory vs. generation quality by eviction policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
