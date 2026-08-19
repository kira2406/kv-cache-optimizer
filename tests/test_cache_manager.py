import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kv_cache_optimizer.cache_manager import CacheConfig, KVCacheManager


class _FakeLayer:
    """Minimal stand-in for transformers.cache_utils.DynamicLayer — just the
    .keys/.values tensors and get_seq_length(), which is all KVCacheManager uses."""

    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def get_seq_length(self):
        return self.keys.shape[2]


class _FakeCache:
    """Minimal stand-in for transformers.cache_utils.Cache — just `.layers`."""

    def __init__(self, layers):
        self.layers = layers


def make_fake_past_kv(seq_len, num_layers=2, heads=2, head_dim=4, batch=1):
    return _FakeCache(
        [
            _FakeLayer(
                torch.randn(batch, heads, seq_len, head_dim),
                torch.randn(batch, heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        ]
    )


def make_fake_attentions(seq_len, num_layers=2, heads=2, q_len=1, batch=1):
    return tuple(
        torch.rand(batch, heads, q_len, seq_len) for _ in range(num_layers)
    )


def test_no_eviction_below_budget():
    cfg = CacheConfig(budget=100, recency_window=8, protected_prefix_len=4, scoring="attention")
    mgr = KVCacheManager(cfg, num_layers=2, device=torch.device("cpu"))
    past = make_fake_past_kv(seq_len=10)
    mgr.update_scores_from_attentions(make_fake_attentions(seq_len=10, q_len=10))
    out = mgr.maybe_evict(past)
    assert out.layers[0].keys.shape[2] == 10  # unchanged, under budget


def test_eviction_respects_budget():
    cfg = CacheConfig(budget=20, recency_window=5, protected_prefix_len=3, scoring="attention")
    mgr = KVCacheManager(cfg, num_layers=2, device=torch.device("cpu"))
    past = make_fake_past_kv(seq_len=50)
    mgr.update_scores_from_attentions(make_fake_attentions(seq_len=50, q_len=50))
    out = mgr.maybe_evict(past)
    assert out.layers[0].keys.shape[2] == 20
    assert mgr.evicted_count == 30


def test_protected_prefix_and_recency_always_kept():
    cfg = CacheConfig(budget=10, recency_window=3, protected_prefix_len=2, scoring="attention")
    mgr = KVCacheManager(cfg, num_layers=1, device=torch.device("cpu"))
    seq_len = 30
    mgr.update_scores_from_attentions(make_fake_attentions(seq_len=seq_len, num_layers=1, q_len=seq_len))
    # force the protected/recent tokens to have the LOWEST scores, to prove
    # they survive purely from protection, not from score
    mgr.scores[:2] = -1000.0
    mgr.scores[-3:] = -1000.0
    keep = mgr.compute_keep_indices(seq_len)
    for i in [0, 1, 27, 28, 29]:
        assert i in keep.tolist()
    assert len(keep) <= 10


def test_all_layers_stay_same_length_after_eviction():
    cfg = CacheConfig(budget=15, recency_window=4, protected_prefix_len=2, scoring="attention")
    mgr = KVCacheManager(cfg, num_layers=4, device=torch.device("cpu"))
    past = make_fake_past_kv(seq_len=40, num_layers=4)
    mgr.update_scores_from_attentions(make_fake_attentions(seq_len=40, num_layers=4, q_len=40))
    out = mgr.maybe_evict(past)
    lengths = {layer.keys.shape[2] for layer in out.layers}
    assert lengths == {15}


def test_random_and_recency_policies_run():
    for scoring in ["random", "recency"]:
        cfg = CacheConfig(budget=12, recency_window=4, protected_prefix_len=2, scoring=scoring)
        mgr = KVCacheManager(cfg, num_layers=1, device=torch.device("cpu"))
        past = make_fake_past_kv(seq_len=30, num_layers=1)
        out = mgr.maybe_evict(past)
        assert out.layers[0].keys.shape[2] == 12


def test_score_bookkeeping_grows_by_one_per_decode_step():
    cfg = CacheConfig(budget=1000, recency_window=4, protected_prefix_len=0, scoring="attention")
    mgr = KVCacheManager(cfg, num_layers=1, device=torch.device("cpu"))
    mgr.update_scores_from_attentions(make_fake_attentions(seq_len=10, num_layers=1, q_len=10))
    assert mgr.scores.shape[0] == 10
    mgr.update_scores_from_attentions(make_fake_attentions(seq_len=11, num_layers=1, q_len=1))
    assert mgr.scores.shape[0] == 11


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
