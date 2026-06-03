"""Debug pretrained loading into SCNN."""
import math
import torch
import torch.nn as nn
from models import ResNet20SCNN
from modules import SCNNConv2d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True, trust_repo=True, verbose=False)
scnn = ResNet20SCNN(num_classes=10, init_b=6.0, init_e=0.0).to(device)

pre_dict = dict(pretrained.named_modules())
scnn_dict = dict(scnn.named_modules())

# Compare layer names
print("PRETRAINED layers:")
for n, m in pre_dict.items():
    if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
        print(f"  {n}: {type(m).__name__}")

print("\nSCNN layers:")
for n, m in scnn_dict.items():
    if isinstance(m, (SCNNConv2d, nn.BatchNorm2d, nn.Linear)):
        print(f"  {n}: {type(m).__name__}")

# Check a specific conv layer
pre_conv1 = pre_dict.get('conv1')
scnn_conv1 = scnn_dict.get('conv1')
if pre_conv1 and scnn_conv1:
    print(f"\nconv1 pretrained weight shape: {pre_conv1.weight.shape}")
    print(f"conv1 scnn weight shape: {scnn_conv1.weight.shape}")
    print(f"conv1 pretrained weight mean: {pre_conv1.weight.mean().item():.4f}, std: {pre_conv1.weight.std().item():.4f}")
    
    # Simulate forward with our layer
    with torch.no_grad():
        scnn_conv1.weight.copy_(pre_conv1.weight)
        W = pre_conv1.weight
        out_ch = W.shape[0]
        e_vals = torch.zeros(out_ch, 1, 1, 1, device=W.device, dtype=torch.float32)
        for c in range(out_ch):
            max_abs = W[c].abs().max().item()
            if max_abs > 0:
                e_vals[c] = math.ceil(math.log2(max_abs)) + 1
        scnn_conv1.e.copy_(e_vals)
        scnn_conv1.b.fill_(8.0)
        
        x_test = torch.randn(1, 3, 32, 32, device=device)
        out_pre = pre_conv1(x_test)
        out_scnn = scnn_conv1(x_test)
        diff = (out_pre - out_scnn).abs().max().item()
        print(f"conv1 output max diff: {diff:.2e}")
        print(f"conv1 e values: min={e_vals.min().item():.2f}, max={e_vals.max().item():.2f}, mean={e_vals.mean().item():.2f}")
