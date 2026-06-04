"""Quick smoke test for pruning + lambda warm-up."""
import os
import sys
import tempfile

# Set CUDA device via env before importing torch submodules that might allocate
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch

from self_compression.compress import compress_model, load_compressed
from self_compression.datasets import get_cifar10
from self_compression.models import ResNet20SCNN
from self_compression.trainer import compute_q, evaluate, train_epoch


def test_pruned_model_instantiation():
    print("\n[1] Instantiating ResNet20SCNN with prune=True ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet20SCNN(
        init_b=4.0, init_e=-8.0, prune=True, prune_tau=0.1, init_t=-6.0
    ).to(device)
    scnn_layers = [l for l in model.modules() if hasattr(l, "qbits")]
    pruned_layers = [l for l in scnn_layers if l.prune]
    assert len(pruned_layers) == len(scnn_layers), "Not all layers have pruning enabled"
    print(f"  SCNN layers: {len(scnn_layers)} | Pruned: {len(pruned_layers)}")
    print("  PASS")
    return model, scnn_layers, device


def test_forward_pass(model, device):
    print("\n[2] Forward pass sanity check ...")
    x = torch.randn(2, 3, 32, 32, device=device)
    out = model(x)
    assert out.shape == (2, 10), f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    print("  PASS")


def test_qbits_with_pruning(scnn_layers):
    print("\n[3] qbits() with pruning ...")
    q = compute_q(scnn_layers)
    assert q.item() > 0, "qbits should be positive"
    print(f"  Avg bits/weight: {q.item():.4f}")
    print("  PASS")


def test_integer_weights(scnn_layers):
    print("\n[4] get_integer_weights() with pruning ...")
    layer = scnn_layers[0]
    W_int, e, b = layer.get_integer_weights()
    assert W_int.dtype == torch.int16, f"Wrong dtype: {W_int.dtype}"
    assert W_int.shape == layer.weight.shape, "Shape mismatch"
    zeros = (W_int == 0).sum().item()
    print(f"  W_int: {W_int.shape}, zeros: {zeros}/{W_int.numel()}")
    print("  PASS")


def test_pruned_ratio(scnn_layers):
    print("\n[5] pruned_ratio() ...")
    for i, l in enumerate(scnn_layers[:3]):
        ratio = l.pruned_ratio()
        assert 0.0 <= ratio <= 1.0, f"Invalid pruned_ratio: {ratio}"
        print(f"  Layer {i}: {ratio*100:.2f}%")
    print("  PASS")


def test_training_loop(model, scnn_layers, device):
    print("\n[6] 2-epoch training loop on CIFAR-10 ...")
    dl, dl_test = get_cifar10(batch_size=128, root="./data", num_workers=2)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Epoch 1: warm-up (lambda=0)
    m1 = train_epoch(model, dl, loss_fn, optimizer, scnn_layers, 0.0, grad_clip=5.0, desc="Warm-up")
    print(f"  Warm-up  | loss={m1['loss']:.3f} acc={m1['acc']:.1f}%")

    # Epoch 2: compression active (lambda=0.15)
    m2 = train_epoch(model, dl, loss_fn, optimizer, scnn_layers, 0.15, grad_clip=5.0, desc="Compress")
    print(f"  Compress | loss={m2['loss']:.3f} acc={m2['acc']:.1f}%")

    # Evaluation
    ev = evaluate(model, dl_test, loss_fn, scnn_layers)
    print(f"  Test     | acc={ev['acc']:.1f}% size={ev['model_bytes']/1024:.1f}KB bits/w={ev['bits_per_weight']:.3f}")
    assert ev['acc'] > 10.0, "Test accuracy too low — model is broken"
    print("  PASS")


def test_compression_roundtrip(model, device):
    print("\n[7] Compression round-trip ...")
    model.eval()
    compressed = compress_model(model)
    assert any(k.startswith("conv") or k.startswith("layer") for k in compressed), "No SCNN layers in compressed dict"

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save(compressed, tmp_path)
    loaded = torch.load(tmp_path, weights_only=False)
    os.remove(tmp_path)

    model2 = load_compressed(loaded, ResNet20SCNN, device=device)
    model2.eval()

    x = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    diff = (out1 - out2).abs().max().item()
    print(f"  Max output diff: {diff:.2e}")
    assert diff < 1e-4, f"Compression mismatch: {diff}"
    print("  PASS")


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    model, scnn_layers, device = test_pruned_model_instantiation()
    test_forward_pass(model, device)
    test_qbits_with_pruning(scnn_layers)
    test_integer_weights(scnn_layers)
    test_pruned_ratio(scnn_layers)
    test_training_loop(model, scnn_layers, device)
    test_compression_roundtrip(model, device)

    print("\n=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    main()
