"""Quick integration test: ResNet18SCNN + CLS-LOC ImageNet."""

import sys
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, "/home/afilip/opencode/self-compression")

from self_compression.datasets import get_imagenet1k
from self_compression.models import ResNet18SCNN

ROOT = "/media/volume3/ichitu/data/ImageNet"
BATCH_SIZE = 64

device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=2 maps to cuda:0 internally

print("Loading datasets...")
train_loader, val_loader = get_imagenet1k(
    batch_size=BATCH_SIZE, root=ROOT, num_workers=0
)
print(f"Train batches: {len(train_loader):,}  |  Val batches: {len(val_loader):,}")

print("\nBuilding model...")
model = ResNet18SCNN(num_classes=1000, init_b=4.0, init_e=-8.0).to(device)
model.train()

optimizer = Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()
scnn_layers = [l for l in model.modules() if hasattr(l, "qbits")]

print(f"\nRunning 5 training batches (BS={BATCH_SIZE}) ...")
for i, (imgs, labels) in enumerate(train_loader):
    if i >= 5:
        break
    imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    optimizer.zero_grad()
    out = model(imgs)
    loss = loss_fn(out, labels)
    q = sum(l.qbits() for l in scnn_layers) / sum(l.weight.numel() for l in scnn_layers)
    total_loss = loss + 0.15 * q
    total_loss.backward()
    optimizer.step()
    acc = (out.argmax(dim=1) == labels).float().mean().item() * 100
    print(f"  Batch {i}: loss={loss.item():.4f}  acc={acc:.1f}%  bits/w={q.item():.3f}")

print("\nRunning 1 validation batch...")
model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(imgs)
        preds = out.argmax(dim=1)
        acc = (preds == labels).float().mean().item() * 100
        print(f"  Val batch: acc={acc:.2f}%")
        break

# GPU memory check
torch.cuda.synchronize()
mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
print(f"\nGPU memory allocated: {mem_alloc:.2f} GB")
print(f"GPU memory reserved:  {mem_reserved:.2f} GB")

# Check model stats
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Verify class ordering
print("\nVerifying class ordering (first 10 synsets):")
for i in range(10):
    print(f"  Class {i}: {train_loader.dataset.classes[i]}")

print("\nIntegration test passed!")
