"""Simple dataset loaders."""

import os

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder, MNIST


# ---------------------------------------------------------------------------
# ImageNet layout helpers
# ---------------------------------------------------------------------------

def _has_imagenet_cls_loc_layout(root):
    """Return True if ``root`` contains the ILSVRC 2012 CLS-LOC layout.

    The canonical structure is::

        root/ILSVRC/Data/CLS-LOC/train/<synset>/
        root/ILSVRC/Data/CLS-LOC/val/<synset>/

    Both splits use standard ``torchvision.datasets.ImageFolder`` structure.
    """
    train_dir = os.path.join(root, "ILSVRC", "Data", "CLS-LOC", "train")
    val_dir = os.path.join(root, "ILSVRC", "Data", "CLS-LOC", "val")
    return os.path.isdir(train_dir) and os.path.isdir(val_dir)


# ---------------------------------------------------------------------------
# Public loader functions
# ---------------------------------------------------------------------------

def get_mnist(batch_size=512, root="./data", num_workers=0):
    """MNIST loaders (reference style).

    Args:
        batch_size: Batch size for both train and test loaders.
        root: Directory to store downloaded data.
        num_workers: DataLoader workers (0 is safest on Windows/ROCm).
    """
    t = T.Compose([T.ToTensor()])
    ds = MNIST(root=root, train=True, download=True, transform=t)
    ds_test = MNIST(root=root, train=False, download=True, transform=t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl, dl_test


def get_cifar10(batch_size=128, root="./data", num_workers=0):
    """CIFAR-10 loaders with standard augmentation.

    Args:
        batch_size: Batch size for both train and test loaders.
        root: Directory to store downloaded data.
        num_workers: DataLoader workers (0 is safest on Windows/ROCm).
    """
    t_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    t_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds = CIFAR10(root=root, train=True, download=True, transform=t_train)
    ds_test = CIFAR10(root=root, train=False, download=True, transform=t_test)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl, dl_test


def get_imagenet1k(batch_size=256, root="./data/imagenet", num_workers=0):
    """ImageNet-1k loaders with standard torchvision transforms.

    Auto-detects the dataset layout in this priority order:

    1. **ILSVRC CLS-LOC** — ``root/ILSVRC/Data/CLS-LOC/train/`` and
       ``root/ILSVRC/Data/CLS-LOC/val/``.  The canonical ImageNet 2012
       challenge format with one subdirectory per synset.
    2. **Standard ImageFolder** — ``root/train/`` and ``root/val/`` with one
       subdirectory per class.

    Args:
        batch_size: Batch size for both train and val loaders.
        root: Path to the ImageNet root directory.
        num_workers: DataLoader workers (0 is safest on Windows/ROCm).
    """
    t_train = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    t_val = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if _has_imagenet_cls_loc_layout(root):
        # ILSVRC 2012 CLS-LOC — the canonical ImageNet challenge format.
        # Already organized into subdirectories per synset, so ImageFolder works
        # directly.  Synset IDs sort alphabetically into the correct 0-999 order.
        train_dir = os.path.join(root, "ILSVRC", "Data", "CLS-LOC", "train")
        val_dir = os.path.join(root, "ILSVRC", "Data", "CLS-LOC", "val")
        ds = ImageFolder(root=train_dir, transform=t_train)
        ds_val = ImageFolder(root=val_dir, transform=t_val)
    else:
        ds = ImageFolder(root=f"{root}/train", transform=t_train)
        ds_val = ImageFolder(root=f"{root}/val", transform=t_val)

    persistent = num_workers > 0
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent,
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent,
    )
    return dl, dl_val
