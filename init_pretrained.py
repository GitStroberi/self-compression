"""Convert a pretrained torchvision ResNet18 into ResNet18SCNN.

Copies all Conv/BN/FC weights from torchvision and initializes each SCNN
layer's exponent `e` based on the actual weight distribution, so the
quantization grid starts in a numerically safe regime.
"""

import argparse
import math
import os

import torch
import torchvision.models as models

from self_compression.models import ResNet18SCNN


def init_pretrained_resnet18_scnn(init_b=4.0, init_e_fallback=-8.0):
    """Load torchvision ResNet18, map into ResNet18SCNN, init e per-layer."""
    print("Loading torchvision ResNet18 (pretrained on ImageNet-1k)...")
    tv = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    tv.eval()

    print("Creating ResNet18SCNN skeleton...")
    scnn = ResNet18SCNN(num_classes=1000, init_b=init_b, init_e=init_e_fallback)

    tv_state = tv.state_dict()
    scnn_state = scnn.state_dict()

    # Build mapping: torchvision key -> our key
    mapped = {}
    for k in tv_state:
        # torchvision uses 'downsample' for shortcut; we use 'shortcut'
        our_k = k.replace("downsample", "shortcut")
        if our_k in scnn_state:
            mapped[our_k] = tv_state[k]
        else:
            print(f"  [skip] {k} -> no match in SCNN")

    # Copy matched parameters
    for k, v in mapped.items():
        scnn_state[k].copy_(v)

    # Now initialize e per SCNN layer based on copied weight range
    print(f"Initializing per-layer `e` with init_b={init_b} ...")
    for name, module in scnn.named_modules():
        if hasattr(module, "e") and hasattr(module, "b"):
            w = module.weight.data
            max_abs = w.abs().max().item()
            if max_abs > 0:
                # Choose e so that max scaled weight ≈ 2**(b-1)
                # 2**(-e) * max_abs = 2**(b-1)  =>  e = log2(max_abs) - (b-1)
                e_val = math.log2(max_abs) - (init_b - 1)
            else:
                e_val = init_e_fallback
            module.e.data.fill_(e_val)
            module.b.data.fill_(init_b)
            print(f"  {name:45s}  max_abs={max_abs:8.5f}  e={e_val:7.3f}  b={init_b:.2f}")

    return scnn


def main():
    parser = argparse.ArgumentParser(description="Init ResNet18SCNN from pretrained ResNet18")
    parser.add_argument("--init-b", type=float, default=4.0, help="Initial bit-width")
    parser.add_argument("--init-e-fallback", type=float, default=-8.0, help="Fallback e")
    parser.add_argument("--out", type=str, default="pretrained_resnet18_scnn.pt",
                        help="Output checkpoint path")
    args = parser.parse_args()

    scnn = init_pretrained_resnet18_scnn(
        init_b=args.init_b, init_e_fallback=args.init_e_fallback
    )

    out_path = os.path.join("runs", args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    torch.save({
        "model": scnn.state_dict(),
        "init_b": args.init_b,
        "init_e_fallback": args.init_e_fallback,
    }, out_path)

    print(f"\nSaved initialized model to: {out_path}")
    print(f"You can now finetune with: --pretrained {out_path}")


if __name__ == "__main__":
    main()
