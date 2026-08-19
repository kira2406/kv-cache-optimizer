# Context-Aware KV Cache Optimizer

A KV cache eviction system for LLM inference. Instead of letting
the KV cache grow linearly with sequence length (which is the usual memory bottleneck
in autoregressive decoding), this project scores cached tokens by how much
attention they receive and evicts the least important ones once a memory
budget is hit, similar in to the H2O ("heavy hitter oracle") line of
work, implemented here directly against Hugging Face `transformers`' modern
`Cache` API.

## Why this exists

Serving long-context LLMs is memory bound, not compute bound, once the KV
cache dominates GPU memory. This project is a hands-on demonstration of:

- How the KV cache actually works at the tensor level (shapes, per-layer
  storage, how `transformers` manages it internally)
- A concrete memory reduction technique (attention-score-based eviction)
  with measurable tradeoffs against simpler baselines (FIFO / random)
- Correct handling of a subtle correctness issue: RoPE position ids after
  non-contiguous token eviction (see `cache_manager.py` docstring)
- A benchmark harness that produces the tradeoff curve serving teams
  actually care about: memory saved vs. quality lost

## Architecture

```
src/kv_cache_optimizer/
  cache_manager.py   # CacheConfig + KVCacheManager: scoring, eviction, budget logic
  generate.py         # manual decode loop that wires the manager into model.forward()
  eval_harness.py     # runs baseline vs. attention/recency/random policies, computes stats
scripts/
  run_demo.py          # single generation with cache stats printed
  run_benchmark.py     # full policy comparison, writes results/benchmark.csv
  plot.py              # memory-vs-perplexity tradeoff figure
tests/
  test_cache_manager.py  # unit tests for eviction logic (no GPU or model needed)
```

**Why a manual decode loop instead of `model.generate()`:** `generate()` gives
no hook between decode steps to inspect or mutate the cache, and no control
over absolute position ids once tokens are evicted. The loop in `generate.py`
trades convenience for that control.

**Eviction policy:** at each step, a per-token importance score accumulates
attention mass received across layers and heads. When cache length exceeds
the configured budget, the lowest-scoring tokens are dropped, except for a
protected prefix (e.g. a system prompt) and a recency window (very recent
tokens are always kept, since evicting them breaks local coherence even when
their attention score happens to be low). `recency` (pure sliding window) and
`random` policies are included as baselines for the comparison plot.

**Important `transformers` version note:** modern `transformers` (5.x)
dropped the legacy tuple-of-tuples cache format. This project reads and
writes `cache.layers[i].keys` / `.values` directly on the `Cache` object.
It also requires `attn_implementation="eager"` when using the `attention`
scoring policy, because the default SDPA/flash-attention kernels never
materialize attention weights, so `output_attentions=True` would silently
return `None`. The scripts handle this for you; if you write your own
loading code, watch out for this.

## Setup

Tuned for a 6GB GPU (developed against a GTX 1660 Ti, no Tensor Cores,
compute capability 7.5, so FlashAttention-2's official kernels are not
available; eager/SDPA attention is used instead).

```bash
pip install -r requirements.txt
```

Recommended models for a 6GB card: `Qwen/Qwen2.5-0.5B-Instruct`,
`Qwen/Qwen2.5-1.5B-Instruct`, or `meta-llama/Llama-3.2-1B-Instruct`. Keep the
model small on purpose, the point is to make the KV cache the memory
bottleneck at moderate context lengths (4 to 8k tokens), not the weights.

## Usage

```bash
# single run with eviction stats
python scripts/run_demo.py --model Qwen/Qwen2.5-0.5B-Instruct --budget 256 --scoring attention

# compare policies across several budgets, writes results/benchmark.csv
python scripts/run_benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct

# turn that CSV into the headline figure
python scripts/plot.py

# unit tests (no GPU or model download needed)
python -m pytest tests/ -v
```

## Results


![Result displaying the result of the benchmark test](results/tradeoff.png)


| policy_name               | scoring   | budget | prompt_len | generated_len | peak_cache_mb | tokens_per_sec     | evicted_count | perplexity         |
|---------------------------|-----------|--------|------------|---------------|---------------|--------------------|---------------|--------------------|
| baseline_no_eviction      | recency   | 367    | 166        | 200           | 4.2890625     | 12.573347724117724 | 0             | 1.4474139213562012 |
| context_aware_budget48    | attention | 48     | 166        | 200           | 0.5625        | 14.60488260936574  | 318           | 1.8406596183776855 |
| fifo_baseline_budget48    | recency   | 48     | 166        | 200           | 0.5625        | 14.608127449430027 | 318           | 1.8209562301635742 |
| random_ablation_budget48  | random    | 48     | 166        | 200           | 0.5625        | 14.48303896739512  | 318           | 1.8154197931289673 |
| context_aware_budget96    | attention | 96     | 166        | 200           | 1.125         | 14.41047202081317  | 270           | 1.6906108856201172 |
| fifo_baseline_budget96    | recency   | 96     | 166        | 200           | 1.125         | 14.427335067879612 | 270           | 1.7529499530792236 |
| random_ablation_budget96  | random    | 96     | 166        | 200           | 1.125         | 14.711301412240433 | 270           | 1.6857616901397705 |
| context_aware_budget160   | attention | 160    | 166        | 200           | 1.875         | 14.261629144610014 | 206           | 1.5383222103118896 |
| fifo_baseline_budget160   | recency   | 160    | 166        | 200           | 1.875         | 14.188338502474522 | 206           | 1.441357970237732  |
| random_ablation_budget160 | random    | 160    | 166        | 200           | 1.875         | 14.749039125922554 | 206           | 1.5567373037338257 |

## Findings
Initial results (single prompt, Qwen2.5-0.5B, budgets of 48/96/160 tokens)
did not show a clean win for attention-based eviction over the FIFO
baseline. FIFO matched the no-eviction baseline most closely at the
loosest budget, and random eviction tracked the attention-based policy
closely at every budget tested.

Investigating why turned up a likely comparison bug rather than a real
result: `recency_window` is a fixed constant (32 tokens) for the
attention and random policies, but for the FIFO policy the entire budget
acts as a sliding recency window. FIFO therefore keeps far more recent
context than the other two policies at the same memory budget, which
likely explains its advantage at looser budgets. The cumulative attention
score is also unnormalized by token age, so it may be partly acting as a
proxy for "has existed longer" rather than "is currently relevant."


## Notes

This project demonstrates: KV cache memory mechanics, a concrete inference
optimization technique with measured tradeoffs, correct RoPE handling under
cache mutation, and a reproducible benchmark harness, the kind of systems
understanding that's directly relevant to LLM inference optimization roles.
Be ready to discuss the memory vs. quality tradeoff curve and why the
recency window and protected prefix exemptions exist.
