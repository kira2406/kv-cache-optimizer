"""
Manual autoregressive decode loop that wires a KVCacheManager into a
Hugging Face causal LM's forward pass.

We don't use model.generate() because it doesn't give us a hook between
decode steps to inspect/evict the cache. This loop trades convenience for
full control over: absolute position ids (needed to keep RoPE correct
across eviction), per-step attention capture, and eviction timing.

Kept deliberately simple (greedy or temperature sampling, batch size 1) so
the eviction logic stays easy to reason about and test. Batching and
beam search are noted as future work in the README.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .cache_manager import KVCacheManager, CacheConfig


@dataclass
class GenerationStats:
    prompt_len: int
    generated_len: int
    peak_cache_len: int
    final_cache_len: int
    evicted_count: int
    peak_cache_bytes: int


@torch.no_grad()
def generate_with_cache_management(
    model,
    tokenizer,
    prompt: str,
    cache_config: CacheConfig,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    eos_token_id: Optional[int] = None,
):
    """
    Returns (generated_text, GenerationStats).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    manager = KVCacheManager(
        config=cache_config,
        num_layers=model.config.num_hidden_layers,
        device=device,
    )

    # ---- prefill -------------------------------------------------------
    out = model(input_ids=input_ids, use_cache=True, output_attentions=True)
    if cache_config.scoring == "attention" and not out.attentions:
        raise RuntimeError(
            "output_attentions=True returned no attention weights. This model was "
            "probably loaded with the default SDPA/flash attention implementation, "
            "which cannot return attention weights. Load it with "
            "`attn_implementation='eager'` to use scoring='attention' "
            "(scoring='recency' or 'random' don't need attentions and work with any "
            "attention implementation)."
        )
    past_key_values = out.past_key_values  # transformers.cache_utils.Cache (e.g. DynamicCache)
    manager.update_scores_from_attentions(out.attentions)
    past_key_values = manager.maybe_evict(past_key_values)

    next_token_logits = out.logits[:, -1, :]
    generated_ids: List[int] = []
    abs_pos = prompt_len  # true absolute position of the NEXT token to generate

    peak_cache_len = past_key_values.layers[0].get_seq_length()
    peak_cache_bytes = manager.cache_bytes(past_key_values)

    eos_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

    # ---- decode ----------------------------------------------------
    for _ in range(max_new_tokens):
        if temperature and temperature > 0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        token_id = next_id.item()
        generated_ids.append(token_id)
        if eos_id is not None and token_id == eos_id:
            break

        cache_len = past_key_values.layers[0].get_seq_length()
        position_ids = torch.tensor([[abs_pos]], device=device)
        attention_mask = torch.ones((1, cache_len + 1), device=device, dtype=torch.long)

        out = model(
            input_ids=next_id,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=True,
        )
        past_key_values = out.past_key_values
        manager.update_scores_from_attentions(out.attentions)
        past_key_values = manager.maybe_evict(past_key_values)

        next_token_logits = out.logits[:, -1, :]
        abs_pos += 1

        cur_len = past_key_values.layers[0].get_seq_length()
        peak_cache_len = max(peak_cache_len, cur_len)
        peak_cache_bytes = max(peak_cache_bytes, manager.cache_bytes(past_key_values))

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    stats = GenerationStats(
        prompt_len=prompt_len,
        generated_len=len(generated_ids),
        peak_cache_len=peak_cache_len,
        final_cache_len=past_key_values.layers[0].get_seq_length(),
        evicted_count=manager.evicted_count,
        peak_cache_bytes=peak_cache_bytes,
    )
    return text, stats


@torch.no_grad()
def teacher_forced_nll_with_cache_management(
    model,
    tokenizer,
    prompt: str,
    reference_ids: torch.Tensor,
    cache_config: CacheConfig,
):
    """
    Measures how well a given eviction policy preserves the model's ability
    to predict a FIXED reference continuation (teacher forcing: at every
    step we feed the true next token, not the model's own guess).

    This is the metric to use for the memory/quality tradeoff plot, instead
    of scoring perplexity on each policy's own free-running output. Scoring
    self-generated text is a rigged comparison: if eviction makes a policy
    degenerate into repetitive output, repetitive text is easy to predict
    and can score a deceptively LOW perplexity despite being obviously worse
    to a human reader. Teacher forcing against one shared, fixed reference
    (normally the unconstrained baseline's own greedy continuation) removes
    that confound, since every policy is judged against the same target.

    Returns (avg_nll, perplexity, GenerationStats). reference_ids should be
    a 1-D LongTensor of continuation token ids (not including the prompt),
    typically produced once by generate_with_cache_management() under an
    unconstrained (no-eviction) policy and reused for every other policy.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    reference_ids = reference_ids.to(device)

    manager = KVCacheManager(
        config=cache_config,
        num_layers=model.config.num_hidden_layers,
        device=device,
    )

    out = model(input_ids=input_ids, use_cache=True, output_attentions=True)
    if cache_config.scoring == "attention" and not out.attentions:
        raise RuntimeError(
            "output_attentions=True returned no attention weights. Load the model "
            "with attn_implementation='eager' to use scoring='attention'."
        )
    past_key_values = out.past_key_values
    manager.update_scores_from_attentions(out.attentions)
    past_key_values = manager.maybe_evict(past_key_values)

    next_token_logits = out.logits[:, -1, :]
    abs_pos = prompt_len
    peak_cache_len = past_key_values.layers[0].get_seq_length()
    peak_cache_bytes = manager.cache_bytes(past_key_values)

    total_nll = 0.0
    n_scored = 0

    for i in range(reference_ids.shape[0]):
        target_id = reference_ids[i : i + 1].unsqueeze(0)  # [1, 1], the KNOWN true next token
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        total_nll += -log_probs[0, target_id.item()].item()
        n_scored += 1

        cache_len = past_key_values.layers[0].get_seq_length()
        position_ids = torch.tensor([[abs_pos]], device=device)
        attention_mask = torch.ones((1, cache_len + 1), device=device, dtype=torch.long)

        out = model(
            input_ids=target_id,  # teacher forcing: feed the TRUE token, not our guess
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=True,
        )
        past_key_values = out.past_key_values
        manager.update_scores_from_attentions(out.attentions)
        past_key_values = manager.maybe_evict(past_key_values)

        next_token_logits = out.logits[:, -1, :]
        abs_pos += 1

        cur_len = past_key_values.layers[0].get_seq_length()
        peak_cache_len = max(peak_cache_len, cur_len)
        peak_cache_bytes = max(peak_cache_bytes, manager.cache_bytes(past_key_values))

    avg_nll = total_nll / max(n_scored, 1)
    perplexity = float(torch.exp(torch.tensor(avg_nll)))

    stats = GenerationStats(
        prompt_len=prompt_len,
        generated_len=n_scored,
        peak_cache_len=peak_cache_len,
        final_cache_len=past_key_values.layers[0].get_seq_length(),
        evicted_count=manager.evicted_count,
        peak_cache_bytes=peak_cache_bytes,
    )
    return avg_nll, perplexity, stats
