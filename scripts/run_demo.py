"""
Quick sanity-check / demo: generate text under a tight cache budget and
print eviction stats.

Example:
    python scripts/run_demo.py --model Qwen/Qwen2.5-1.5B-Instruct --budget 256
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kv_cache_optimizer import CacheConfig, generate_with_cache_management


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--recency-window", type=int, default=32)
    ap.add_argument("--protected-prefix-len", type=int, default=4)
    ap.add_argument("--scoring", default="attention", choices=["attention", "recency", "random"])
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument(
        "--prompt",
        default=(
            "Write a detailed, multi-paragraph explanation of how attention "
            "mechanisms work in transformer models, including the role of "
            "queries, keys, and values, and why KV caching speeds up inference."
        ),
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {args.model} on {device} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # attn_implementation="eager" is required whenever scoring="attention":
    # the default SDPA/flash-attention kernels never materialize attention
    # weights, so output_attentions=True would silently return None.
    attn_impl = "eager" if args.scoring == "attention" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation=attn_impl
    ).to(device)
    model.eval()

    cfg = CacheConfig(
        budget=args.budget,
        recency_window=args.recency_window,
        protected_prefix_len=args.protected_prefix_len,
        scoring=args.scoring,
    )

    text, stats = generate_with_cache_management(
        model, tokenizer, args.prompt, cfg, max_new_tokens=args.max_new_tokens
    )

    print("\n=== Generated text ===")
    print(text)
    print("\n=== Cache stats ===")
    print(f"prompt tokens:      {stats.prompt_len}")
    print(f"generated tokens:   {stats.generated_len}")
    print(f"budget:             {args.budget}")
    print(f"peak cache length:  {stats.peak_cache_len}")
    print(f"final cache length: {stats.final_cache_len}")
    print(f"tokens evicted:     {stats.evicted_count}")
    print(f"peak cache memory:  {stats.peak_cache_bytes / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
