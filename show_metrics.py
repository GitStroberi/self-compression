import torch
h = torch.load("runs/small_test/metrics_history.pt")
print(f"{'Ep':>3s} | {'Loss':>8s} | {'TrAcc':>6s} | {'TeAcc':>6s} | {'Size':>8s} | {'Bits/W':>6s}")
print("-" * 55)
for e in h:
    print(f"{e['epoch']:3d} | {e['train_loss']:8.4f} | {e['train_acc']:5.1f}% | {e['test_acc']:5.1f}% | {e['model_bytes']/1024:7.1f}KB | {e['avg_bitwidth']:6.4f}")
