"""Unified epoch-based training script for Self-Compressing Neural Networks.

Works with any model + dataset combination. All models use the same loop:
  for epoch in range(epochs):
      train_epoch(...)
      evaluate(...)

To add a new model (e.g., YOLOv5, MobileNetV2):
  1. Add it to models.py using SCNNConv2d layers
  2. Add a dataset loader to datasets.py
  3. Register both in the MODEL_REGISTRY and DATASET_REGISTRY below
"""

import argparse
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam, SGD

from .datasets import get_cifar10, get_imagenet1k, get_mnist
from .models import Net, ResNet18SCNN, ResNet20SCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Registry: add new models / datasets here
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "Net": Net,
    "ResNet20SCNN": ResNet20SCNN,
    "ResNet18SCNN": ResNet18SCNN,
}

DATASET_REGISTRY = {
    "mnist": get_mnist,
    "cifar10": get_cifar10,
    "imagenet1k": get_imagenet1k,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _scnn_param_ids(model):
    """Return a set of id()s for all SCNNConv2d parameters."""
    ids = set()
    for module in model.modules():
        if hasattr(module, "qbits"):
            ids.add(id(module.weight))
            ids.add(id(module.e))
            ids.add(id(module.b))
    return ids


def compute_q(scnn_layers):
    """Compression penalty: average bits per SCNN weight."""
    total_qbits = sum(l.qbits() for l in scnn_layers)
    total_scnn_weights = sum(l.weight.numel() for l in scnn_layers)
    return total_qbits / total_scnn_weights


def plot_metrics(run_dir, history, dataset_name=""):
    """Plot accuracy and model size over epochs."""
    epochs = [h["epoch"] for h in history]
    test_accs = [h["test_acc"] for h in history]
    model_bytes = [h["model_bytes"] for h in history]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Model Size (bytes)", color="red")
    ax1.plot(epochs, model_bytes, color="red", label="Model Size")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.plot(epochs, test_accs, color="blue", label="Test Accuracy")
    ax2.set_ylabel("Test Accuracy (%)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Auto-scale y-axis based on data with sensible padding
    if test_accs:
        min_acc = min(test_accs)
        max_acc = max(test_accs)
        padding = max(5, (100 - max_acc) / 2)
        ax2.set_ylim([max(0, min_acc - padding), min(100, max_acc + padding)])

    fig.legend(loc="lower right")
    title = f"SCNN Training {dataset_name.upper()}: Accuracy vs Model Size".strip()
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(run_dir, "training_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, loss_fn, optimizer, scnn_layers, lambda_comp, desc="Epoch"):
    """Train for one epoch. Returns dict with metrics."""
    model.train()
    epoch_loss = 0.0
    epoch_task = 0.0
    correct = 0
    total = 0

    pbar = tqdm.tqdm(loader, desc=desc, unit="batch", leave=False, ascii=True)
    for samples, targets in pbar:
        samples, targets = samples.to(device), targets.to(device)
        optimizer.zero_grad()
        pred = model(samples)
        loss = loss_fn(pred, targets)
        q = compute_q(scnn_layers)
        total_loss = loss + lambda_comp * q
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_task += loss.item()
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(f"{desc} loss:{loss.item():.3f} acc:{100.0*correct/total:.1f}%")

    return {
        "loss": epoch_loss / len(loader),
        "task_loss": epoch_task / len(loader),
        "acc": 100.0 * correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, scnn_layers):
    """Evaluate on test set. Returns dict with metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm.tqdm(loader, desc="Eval", unit="batch", leave=False, ascii=True)
    for samples, targets in pbar:
        samples, targets = samples.to(device), targets.to(device)
        pred = model(samples)
        loss = loss_fn(pred, targets)
        total_loss += loss.item()
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_description(f"Eval acc:{100.0*correct/total:.1f}%")

    q = compute_q(scnn_layers)
    total_scnn_weights = sum(l.weight.numel() for l in scnn_layers)
    scnn_param_ids = _scnn_param_ids(model)
    other_params = sum(p.numel() for p in model.parameters() if id(p) not in scnn_param_ids)
    model_bytes = q.item() * total_scnn_weights / 8 + other_params * 4  # other params in FP32

    return {
        "loss": total_loss / len(loader),
        "acc": 100.0 * correct / total,
        "model_bytes": model_bytes,
        "bits_per_weight": q.item(),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args, run_dir):
    """Main training loop — works for any registered model + dataset."""
    # Build model
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(init_b=args.init_b, init_e=args.init_e).to(device)

    # Build dataset
    dataset_fn = DATASET_REGISTRY[args.dataset]
    dl, dl_test = dataset_fn(args.batch_size, root=args.dataset_root)

    print(f"Device: {device}")
    print(f"Model: {args.model} | Dataset: {args.dataset} | Epochs: {args.epochs}")
    print(f"BS: {args.batch_size} | LR: {args.lr} | optimizer: {args.optimizer} | scheduler: {args.lr_scheduler}")
    print(f"lambda: {args.lambda_comp} | init_b: {args.init_b} | init_e: {args.init_e}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = SGD(
            model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Scheduler
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == "step":
        milestones = [int(s) for s in args.lr_steps.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_scheduler}")

    loss_fn = nn.CrossEntropyLoss()
    scnn_layers = [l for l in model.modules() if hasattr(l, "qbits")]

    history = []
    best_acc = 0.0
    start_time = datetime.now()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, dl, loss_fn, optimizer, scnn_layers, args.lambda_comp,
            desc=f"Epoch {epoch:3d}/{args.epochs}"
        )
        test_metrics = evaluate(model, dl_test, loss_fn, scnn_layers)

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "model_bytes": test_metrics["model_bytes"],
            "bits_per_weight": test_metrics["bits_per_weight"],
        })

        tqdm.tqdm.write(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_metrics['task_loss']:.4f} tr_acc={train_metrics['acc']:.1f}% | "
            f"te_acc={test_metrics['acc']:.1f}% | "
            f"size={test_metrics['model_bytes']/1024:.1f}KB | "
            f"bits/w={test_metrics['bits_per_weight']:.3f} | "
            f"best={best_acc:.1f}%"
        )

        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
            }, os.path.join(run_dir, "best_model.pt"))

        if epoch % args.checkpoint_freq == 0 or epoch == args.epochs:
            path = os.path.join(run_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "args": vars(args),
            }, path)

        scheduler.step()

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Best test accuracy: {best_acc:.2f}%")

    torch.save({"history": history, "args": vars(args)}, os.path.join(run_dir, "metrics_history.pt"))
    plot_metrics(run_dir, history, dataset_name=args.dataset)

    # Print learned bit-widths
    bitwidths = {n: l.b.mean().item() for n, l in model.named_modules() if hasattr(l, "b")}
    print("\nLearned bit-widths per layer:")
    for n, b in bitwidths.items():
        print(f"  {n:40s}: {b:.3f} bits")

    with open(os.path.join(run_dir, "bitwidths.txt"), "w") as f:
        for n, b in bitwidths.items():
            f.write(f"{n}: {b:.4f}\n")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train SCNN (unified epoch-based)")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True,
                        help="Model architecture to train")
    parser.add_argument("--dataset", choices=list(DATASET_REGISTRY.keys()), required=True,
                        help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lambda-comp", type=float, default=0.05, help="Compression loss weight")
    parser.add_argument("--init-b", type=float, default=4.0, help="Initial bit-width for SCNN layers")
    parser.add_argument("--init-e", type=float, default=-8.0, help="Initial exponent for SCNN layers")
    parser.add_argument("--checkpoint-freq", type=int, default=50, help="Checkpoint every N epochs")
    parser.add_argument("--dataset-root", type=str, default="./data", help="Root directory for dataset")
    parser.add_argument("--run-name", type=str, default="", help="Optional suffix for run directory name")
    # Optimizer / scheduler (ImageNet-style SGD support)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="Optimizer to use (default: adam)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step"],
                        help="LR scheduler (default: cosine)")
    parser.add_argument("--lr-steps", type=str, default="30,60,80",
                        help="Step scheduler milestones, comma-separated (default: 30,60,80)")
    parser.add_argument("--gamma", type=float, default=0.1, help="Step LR decay factor (default: 0.1)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.run_name}" if args.run_name else timestamp
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    train(args, run_dir)


if __name__ == "__main__":
    main()
