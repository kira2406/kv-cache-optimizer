"""
Context-aware KV cache manager.

Implements budget-constrained eviction directly over a Hugging Face
`transformers.cache_utils.Cache` object (e.g. `DynamicCache`), by mutating
each layer's `.keys` / `.values` tensors in place. Note: modern
`transformers` (5.x) dropped the old legacy tuple-of-tuples cache format
and its `to_legacy_cache()` conversion, so this operates on the `Cache`
object's `.layers[i].keys/.values` directly rather than tuples — if you're
reading H2O/StreamingLLM reference code written against an older
transformers version, this is the main thing that changed. Eviction
decisions are driven by a pluggable importance score (default: cumulative
attention mass received, similar in spirit to the H2O "heavy hitter" idea)
and always protect two groups of tokens:

  1. A fixed prefix ("protected_prefix_len") — e.g. a system prompt — that
     is never evicted regardless of score.
  2. The most recent "recency_window" tokens — evicting very recent context
     tends to break local coherence even when its attention score is still
     low, so it's exempted on principle rather than purely on score.

Everything else competes for the remaining budget purely on score.

Design note on RoPE correctness: eviction removes entries from the cache
but does NOT renumber the survivors. Each retained key vector keeps the
rotary embedding baked in at its true original absolute position. Because
RoPE attention scores depend only on the relative angle between a query
and a key (theta_q - theta_k), a non-contiguous set of retained absolute
positions is still handled correctly — we do not recompute cache-local
position ids for retained tokens. We only need to track the true absolute
position for the *next* generated token (see generate.py).

Design note on scoring granularity: this reference implementation uses one
score per token, averaged over layers and heads, rather than independent
per-layer/per-head scores. That keeps all layers' cache at the same length,
which keeps attention-mask handling trivial. True per-layer eviction (each
layer keeps a different set of tokens) is a natural, well-scoped follow-up
ablation — flagged in the README as future work.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache


@dataclass
class CacheConfig:
    budget: int                     # max tokens retained per layer once eviction kicks in
    recency_window: int = 32        # most-recent tokens always kept
    protected_prefix_len: int = 0   # e.g. system prompt / few-shot header, always kept
    scoring: str = "attention"      # "attention" | "recency" (FIFO baseline) | "random" (ablation control)

    def __post_init__(self):
        if self.protected_prefix_len + self.recency_window > self.budget:
            raise ValueError(
                f"protected_prefix_len ({self.protected_prefix_len}) + recency_window "
                f"({self.recency_window}) must be <= budget ({self.budget})"
            )


class KVCacheManager:
    """
    Wraps eviction bookkeeping around a `transformers` `Cache` object.
    Stateful across a single generation call — create one instance per prompt.
    """

    def __init__(self, config: CacheConfig, num_layers: int, device: torch.device):
        self.config = config
        self.num_layers = num_layers
        self.device = device
        # Cumulative attention received, one score per currently-cached token.
        self.scores: Optional[torch.Tensor] = None
        self.evicted_count = 0

    # ---- score bookkeeping -------------------------------------------------

    def update_scores_from_attentions(self, attentions: Tuple[torch.Tensor, ...]):
        """
        attentions: tuple of per-layer attention weight tensors, each
        [batch, num_heads, q_len, k_len], as returned by output_attentions=True.
        Call this once per forward pass (prefill or a single decode step).
        Adds attention mass received by each key position this step, averaged
        over layers and heads, summed over query positions.
        Assumes batch size 1 (reference implementation; batching is future work).
        """
        if self.config.scoring != "attention":
            return  # recency / random policies don't need attention stats

        per_layer = [attn.sum(dim=2).mean(dim=1) for attn in attentions]  # each -> [batch, k_len]
        received = torch.stack(per_layer, dim=0).mean(dim=0)[0]           # -> [k_len]
        k_len = received.shape[0]

        if self.scores is None:
            # prefill: k_len == q_len == prompt length, one score per prompt token
            self.scores = received.clone()
            return

        old_len = self.scores.shape[0]
        new_tokens = k_len - old_len
        if new_tokens < 0:
            raise RuntimeError(
                f"cache shrank without going through evict() (old={old_len}, new={k_len}); "
                "scores and cache are now out of sync"
            )
        # existing tokens accumulate whatever attention they received this step
        self.scores += received[:old_len]
        if new_tokens > 0:
            # newly appended tokens (normally just 1, the just-generated token)
            # start with the attention they received in this same step
            self.scores = torch.cat([self.scores, received[old_len:]], dim=0)

    # ---- eviction ------------------------------------------------------

    def needs_eviction(self, cache_len: int) -> bool:
        return cache_len > self.config.budget

    def compute_keep_indices(self, cache_len: int) -> torch.Tensor:
        """Sorted 1-D LongTensor of indices (into the current cache) to retain."""
        cfg = self.config
        protected = set(range(min(cfg.protected_prefix_len, cache_len)))

        if cfg.scoring == "recency":
            # Proper sliding-window / FIFO baseline: protected prefix plus as
            # many of the most recent tokens as fit in the remaining budget.
            # (recency_window is not used as a separate exemption here — the
            # whole non-protected budget IS the recency window.)
            n_recent = max(cfg.budget - len(protected), 0)
            recent_start = max(cache_len - n_recent, len(protected))
            keep = protected | set(range(recent_start, cache_len))
            return torch.tensor(sorted(keep), dtype=torch.long, device=self.device)

        recent_start = max(cache_len - cfg.recency_window, cfg.protected_prefix_len)
        recent = set(range(recent_start, cache_len))

        always_keep = protected | recent
        remaining_budget = cfg.budget - len(always_keep)
        candidates = [i for i in range(cache_len) if i not in always_keep]

        if remaining_budget <= 0 or not candidates:
            keep = always_keep
        elif cfg.scoring == "random":
            perm = torch.randperm(len(candidates))[:remaining_budget]
            keep = always_keep | {candidates[i] for i in perm.tolist()}
        else:  # "attention"
            cand_scores = self.scores[candidates]
            k = min(remaining_budget, len(candidates))
            topk = torch.topk(cand_scores, k=k).indices
            keep = always_keep | {candidates[i] for i in topk.tolist()}

        return torch.tensor(sorted(keep), dtype=torch.long, device=self.device)

    def evict(self, cache: Cache, keep_indices: torch.Tensor) -> Cache:
        """Mutates `cache` in place, dropping every position not in keep_indices,
        uniformly across all layers. Returns the same object for chaining."""
        old_len = cache.layers[0].get_seq_length()
        for layer in cache.layers:
            layer.keys = layer.keys[:, :, keep_indices, :]
            layer.values = layer.values[:, :, keep_indices, :]
        self.evicted_count += old_len - keep_indices.shape[0]
        if self.scores is not None:
            self.scores = self.scores[keep_indices]
        return cache

    def maybe_evict(self, cache: Cache) -> Cache:
        cache_len = cache.layers[0].get_seq_length()
        if not self.needs_eviction(cache_len):
            return cache
        keep_indices = self.compute_keep_indices(cache_len)
        return self.evict(cache, keep_indices)

    # ---- utilities -------------------------------------------------

    def cache_bytes(self, cache: Cache) -> int:
        total = 0
        for layer in cache.layers:
            total += layer.keys.element_size() * layer.keys.nelement()
            total += layer.values.element_size() * layer.values.nelement()
        return total
