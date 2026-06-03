"""Quick sanity check for SCNN training on CIFAR-10."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import Adam

from self_compression.models.resnet_scnn import ResNet20SCNN
from self_compression.utils.loss import SelfCompressionLoss, compute_model_size_bits

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Tiny dataset
transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Use small subsets for speed
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)

# Model
model = ResNet20SCNN(num_classes=10).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss & optimizer
criterion = SelfCompressionLoss(nn.CrossEntropyLoss(), lambda_compression=0.05)
optimizer = Adam(model.parameters(), lr=1e-3)

# 1 epoch sanity check
print("\nRunning 1 epoch sanity check...")
model.train()
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss, task_loss, comp_loss = criterion(outputs, targets, model)
    loss.backward()
    optimizer.step()
    if batch_idx % 50 == 0:
        print(f"  Batch {batch_idx}: loss={loss.item():.4f} task={task_loss.item():.4f} comp={comp_loss.item():.6f}")
    if batch_idx >= 20:  # only run a few batches
        break

# Eval
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        break  # just one batch

acc = 100.0 * correct / total
bits, bytes_ = compute_model_size_bits(model)
print(f"\nSanity check complete!")
print(f"  Test acc (1 batch): {acc:.2f}%")
print(f"  Model size: {bits:.0f} bits ({bytes_:.0f} bytes)")

# Cleanup
if torch.cuda.is_available():
    torch.cuda.synchronize()
