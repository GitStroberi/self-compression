"""Quick sanity check for both MNIST and CIFAR modes."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "self_compression"))

import torch
import torch.nn as nn
from torch.optim import Adam

from modules import SCNNConv2d
from models import Net, ResNet20SCNN
from datasets import get_mnist, get_cifar10


print("=" * 60)
print("SANITY CHECK: MNIST (Net, step-based)")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
dl, dl_test = get_mnist(batch_size=512)
x_test, y_test = next(iter(dl_test))
x_test, y_test = x_test.to(device), y_test.to(device)

optimizer = Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

for step in range(5):
    samples, targets = next(iter(dl))
    samples, targets = samples.to(device), targets.to(device)
    optimizer.zero_grad()
    pred = model(samples)
    loss = loss_fn(pred, targets)
    qbits = sum(l.qbits() for l in model.modules() if hasattr(l, "qbits"))
    comp = qbits / sum(p.numel() for p in model.parameters())
    total = loss + 0.05 * comp
    total.backward()
    optimizer.step()
    print(f"  Step {step}: loss={loss.item():.4f} bits/w={comp.item():.4f}")

test_acc = (model(x_test).argmax(axis=1) == y_test).float().mean().item() * 100
print(f"  Test acc: {test_acc:.2f}%")

print("\n" + "=" * 60)
print("SANITY CHECK: CIFAR-10 (ResNet20, epoch-based)")
print("=" * 60)

del model
model = ResNet20SCNN(num_classes=10).to(device)
dl, dl_test = get_cifar10(batch_size=128)
optimizer = Adam(model.parameters(), lr=3e-4)
model.train()

for i, (samples, targets) in enumerate(dl):
    if i >= 5:
        break
    samples, targets = samples.to(device), targets.to(device)
    optimizer.zero_grad()
    pred = model(samples)
    loss = loss_fn(pred, targets)
    qbits = sum(l.qbits() for l in model.modules() if hasattr(l, "qbits"))
    comp = qbits / sum(p.numel() for p in model.parameters())
    total = loss + 0.05 * comp
    total.backward()
    optimizer.step()
    print(f"  Batch {i}: loss={loss.item():.4f} bits/w={comp.item():.4f}")

# Quick eval
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for samples, targets in dl_test:
        samples, targets = samples.to(device), targets.to(device)
        pred = model(samples)
        _, p = pred.max(1)
        total += targets.size(0)
        correct += p.eq(targets).sum().item()
        break
print(f"  Test acc (1 batch): {100.0 * correct / total:.2f}%")

if torch.cuda.is_available():
    torch.cuda.synchronize()
print("\nAll sanity checks passed!")
