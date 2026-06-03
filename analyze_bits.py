"""Analyze the actual bit-width requirements of trained integer weights."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "self_compression"))

import torch

from modules import SCNNConv2d, bits_needed_for_range
from trainer import MODEL_REGISTRY


def analyze_checkpoint(ckpt_path, model_name, device="cpu"):
    """Load a checkpoint and print per-layer bit-width analysis."""
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    print(f"{'Layer':<25s} | {'Min':>4s} | {'Max':>4s} | {'Unique':>6s} | {'MinBits':>7s} | {'Learned b':>9s}")
    print("-" * 75)

    total_weights = 0
    packed_theoretical_bits = 0

    for name, layer in model.named_modules():
        if isinstance(layer, SCNNConv2d):
            with torch.no_grad():
                W_int = torch.round(layer.qweight()).cpu()
            w_min = int(W_int.min().item())
            w_max = int(W_int.max().item())
            n_unique = len(torch.unique(W_int))
            min_bits = bits_needed_for_range(w_min, w_max)

            n_weights = W_int.numel()
            total_weights += n_weights
            packed_theoretical_bits += n_weights * min_bits

            print(f"{name:<25s} | {w_min:4d} | {w_max:4d} | {n_unique:6d} | {min_bits:7d} | {layer.b.mean().item():9.3f}")

    print("-" * 75)
    print(f"Total weights: {total_weights:,}")
    print(f"Theoretical packed size (min-bits): {packed_theoretical_bits/8/1024:.2f} KB")
    print(f"int8 storage:                       {total_weights/1024:.2f} KB")
    if packed_theoretical_bits > 0:
        print(f"Compression vs int8:                {total_weights / (packed_theoretical_bits/8):.1f}x")

    # Check overall distribution
    all_unique = set()
    for layer in model.modules():
        if isinstance(layer, SCNNConv2d):
            with torch.no_grad():
                W_int = torch.round(layer.qweight()).cpu()
            all_unique.update(W_int.flatten().tolist())
    print(f"\nGlobal unique integer values across all layers: {sorted(all_unique)}")
    print(f"Count: {len(all_unique)} distinct values")


def main():
    parser = argparse.ArgumentParser(description="Analyze bit-widths of a trained SCNN checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True,
                        help="Model architecture")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyze_checkpoint(args.ckpt, args.model, device=device)


if __name__ == "__main__":
    main()
