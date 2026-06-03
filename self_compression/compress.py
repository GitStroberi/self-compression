"""
True weight compression for SCNN models using custom bit-packing.

After training, the model stores full FP32 weights + learned b/e parameters.
This script extracts the actual integer weights and packs them into the minimum
number of bits per layer (2, 3, or 4 bits), achieving near-theoretical sizes.
"""

import argparse
import math
import os
import tempfile

import numpy as np
import torch

from modules import SCNNConv2d, bits_needed_for_range
from trainer import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Bit-packing helpers: pack unsigned integers into the fewest bits possible
# ---------------------------------------------------------------------------

def pack_bits(values, bits_per_value):
    """
    Pack a 1-D array of unsigned integers into a compact byte buffer.

    Args:
        values: numpy array of non-negative integers (dtype=np.uint64 for safety).
        bits_per_value: how many bits each value needs.

    Returns:
        bytes buffer.
    """
    values = np.asarray(values, dtype=np.uint64)
    n = len(values)
    total_bits = n * bits_per_value
    total_bytes = (total_bits + 7) // 8

    buf = bytearray(total_bytes)
    bit_idx = 0
    mask = (1 << bits_per_value) - 1
    for v in values:
        v = int(v) & mask
        for b in range(bits_per_value):
            byte_pos = bit_idx // 8
            bit_pos = bit_idx % 8
            if (v >> b) & 1:
                buf[byte_pos] |= (1 << bit_pos)
            bit_idx += 1
    return bytes(buf)


def unpack_bits(buf, bits_per_value, n_values):
    """Reverse of pack_bits."""
    mask = (1 << bits_per_value) - 1
    values = np.zeros(n_values, dtype=np.int64)
    bit_idx = 0
    for i in range(n_values):
        v = 0
        for b in range(bits_per_value):
            byte_pos = bit_idx // 8
            bit_pos = bit_idx % 8
            if buf[byte_pos] & (1 << bit_pos):
                v |= (1 << b)
            bit_idx += 1
        values[i] = v
    return values


# ---------------------------------------------------------------------------
# Compression / decompression
# ---------------------------------------------------------------------------

def compress_model(model):
    """Extract integer weights and pack into minimum bits per layer."""
    compressed = {}
    for name, module in model.named_modules():
        if isinstance(module, SCNNConv2d):
            with torch.no_grad():
                W_int = torch.round(module.qweight()).cpu()
            w_min = int(W_int.min().item())
            w_max = int(W_int.max().item())
            bits = bits_needed_for_range(w_min, w_max)

            # Map signed integers to unsigned codes: code = value - w_min
            W_uint = (W_int - w_min).to(torch.int64).numpy().flatten()
            packed_bytes = pack_bits(W_uint, bits)

            compressed[name] = {
                "type": "SCNNConv2d",
                "shape": list(W_int.shape),
                "bits": bits,
                "w_min": w_min,
                "packed": packed_bytes,
                "e": module.e.detach().cpu().numpy().astype(np.float32),
                "b": module.b.detach().cpu().numpy().astype(np.float32),
                "bias": module.bias.detach().cpu().numpy() if module.bias is not None else None,
            }
        elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.Linear)):
            state = {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}
            compressed[name] = {"type": module.__class__.__name__, "state": state}

    return compressed


def load_compressed(compressed_dict, model_cls, device="cpu"):
    """Reconstruct an SCNN model from a custom bit-packed checkpoint.

    Args:
        compressed_dict: dict produced by compress_model().
        model_cls: model class (e.g., Net or ResNet20SCNN).
        device: target device.

    Returns:
        Reconstructed model on the requested device.
    """
    model = model_cls().to(device)
    model.eval()

    for name, module in model.named_modules():
        if name not in compressed_dict:
            continue
        ckpt = compressed_dict[name]

        if ckpt["type"] == "SCNNConv2d" and isinstance(module, SCNNConv2d):
            n_vals = int(np.prod(ckpt["shape"]))
            unpacked = unpack_bits(ckpt["packed"], ckpt["bits"], n_vals)
            W_int = torch.from_numpy(unpacked + ckpt["w_min"]).to(device)
            W_int = W_int.view(ckpt["shape"]).to(module.weight.dtype)

            e = torch.from_numpy(ckpt["e"]).to(device)
            b = torch.from_numpy(ckpt["b"]).to(device)

            module.set_from_integer_weights(W_int, e, b)
            if ckpt["bias"] is not None:
                module.bias.data = torch.from_numpy(ckpt["bias"]).to(device)
        else:
            state = {k: torch.from_numpy(v).to(device) for k, v in ckpt["state"].items()}
            module.load_state_dict(state)

    return model


# ---------------------------------------------------------------------------
# Analysis & comparison
# ---------------------------------------------------------------------------

def analyze_model(model):
    """Print per-layer integer weight statistics."""
    print(f"\n{'='*65}")
    print("PER-LAYER INTEGER WEIGHT ANALYSIS")
    print(f"{'='*65}")
    print(f"{'Layer':<25s} | {'Min':>4s} | {'Max':>4s} | {'Unique':>6s} | {'Bits':>4s} | {'Packed':>10s}")
    print("-" * 65)
    total_int8 = 0
    total_packed = 0
    for name, layer in model.named_modules():
        if isinstance(layer, SCNNConv2d):
            with torch.no_grad():
                W_int = torch.round(layer.qweight()).cpu()
            w_min = int(W_int.min().item())
            w_max = int(W_int.max().item())
            n_unique = len(torch.unique(W_int))
            bits = bits_needed_for_range(w_min, w_max)
            n = W_int.numel()
            total_int8 += n
            total_packed += math.ceil(n * bits / 8)
            print(f"{name:<25s} | {w_min:4d} | {w_max:4d} | {n_unique:6d} | {bits:4d} | {math.ceil(n*bits/8):>10,} B")
    print("-" * 65)
    print(f"{'TOTAL':<25s} |      |      |        |      | {total_packed:>10,} B  (int8 would be {total_int8:,} B)")
    print(f"{'='*65}")


def compare_sizes(model, compressed_dict, filepath="compressed.pt"):
    """Compare FP32 checkpoint vs bit-packed vs theoretical limit."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        fp32_path = f.name
    try:
        torch.save(model.state_dict(), fp32_path)
        torch.save(compressed_dict, filepath)

        fp32_size = os.path.getsize(fp32_path)
        comp_size = os.path.getsize(filepath)

        # Theoretical from learned bit-widths
        scnn_layers = [l for l in model.modules() if isinstance(l, SCNNConv2d)]
        qbits = sum(l.qbits() for l in scnn_layers)

        scnn_param_ids = set()
        for l in scnn_layers:
            scnn_param_ids.add(id(l.weight))
            scnn_param_ids.add(id(l.e))
            scnn_param_ids.add(id(l.b))
        other_params = sum(p.numel() for p in model.parameters() if id(p) not in scnn_param_ids)
        theoretical = qbits.item() / 8 + other_params * 4  # SCNN weights in packed bits, rest in FP32 bytes

        print(f"\n{'='*65}")
        print("SIZE COMPARISON")
        print(f"{'='*65}")
        print(f"  FP32 checkpoint:          {fp32_size:>12,} B  ({fp32_size/1024:>6.1f} KB)")
        print(f"  Custom bit-packed:        {comp_size:>12,} B  ({comp_size/1024:>6.1f} KB)")
        print(f"  Theoretical limit:        {theoretical:>12,.0f} B  ({theoretical/1024:>6.1f} KB)")
        print(f"  vs FP32:                {fp32_size/comp_size:>6.1f}x smaller")
        print(f"  vs theoretical:         {comp_size/theoretical:>6.1f}x overhead (pickle+metadata)")
        print(f"{'='*65}")
    finally:
        os.remove(fp32_path)


def verify_equivalence(model, compressed_dict, input_shape, device="cpu"):
    """Verify that a compressed model produces identical outputs."""
    x = torch.randn(2, *input_shape, device=device)
    model.eval()
    with torch.no_grad():
        out_original = model(x)

    model2 = load_compressed(compressed_dict, model_cls=type(model), device=device)
    with torch.no_grad():
        out_reconstructed = model2(x)

    diff = (out_original - out_reconstructed).abs().max().item()
    print(f"\nMax output difference (original vs bit-packed): {diff:.2e}")
    print("PASS: Outputs are equivalent." if diff < 1e-4 else "FAIL: Outputs differ!")


def main():
    parser = argparse.ArgumentParser(description="Compress a trained SCNN model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True,
                        help="Model architecture")
    parser.add_argument("--out", type=str, default="compressed.pt", help="Output compressed path")
    parser.add_argument("--input-shape", type=int, nargs="+", default=None,
                        help="Input shape for verification (e.g. 1 28 28 or 3 32 32)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = MODEL_REGISTRY[args.model]

    # Default shapes per dataset
    default_shapes = {
        "Net": (1, 28, 28),
        "ResNet20SCNN": (3, 32, 32),
    }
    input_shape = tuple(args.input_shape) if args.input_shape else default_shapes.get(args.model)
    if input_shape is None:
        raise ValueError("Please specify --input-shape for this model.")

    print(f"Loading trained model from: {args.ckpt}")
    model = model_cls().to(device)
    state = torch.load(args.ckpt, map_location=device)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    analyze_model(model)

    print("\nExtracting and bit-packing integer weights...")
    compressed = compress_model(model)

    compare_sizes(model, compressed, filepath=args.out)
    verify_equivalence(model, compressed, input_shape=input_shape, device=device)

    print(f"\nCompressed model saved to: {args.out}")


if __name__ == "__main__":
    main()
