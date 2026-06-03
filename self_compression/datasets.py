"""Simple dataset loaders."""

import glob
import importlib.util
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, ImageFolder, MNIST


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _load_imagenet_classes(path):
    """Dynamically import ``IMAGENET2012_CLASSES`` from a ``classes.py`` file."""
    spec = importlib.util.spec_from_file_location("_imagenet_classes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    classes = getattr(mod, "IMAGENET2012_CLASSES")
    # Preserve insertion order (Python 3.7+) and map synset -> integer index
    return {synset: idx for idx, synset in enumerate(classes.keys())}


# ---------------------------------------------------------------------------
# Flat-directory ImageNet dataset (HuggingFace-datasets layout)
# ---------------------------------------------------------------------------

class _ImageNetFlatDataset(Dataset):
    """ImageNet from flat directories where the synset is embedded in the filename.

    This matches the HuggingFace ``imagenet-1k`` builder layout:
      - ``data/train_images_*/<filename>_<synset>.JPEG``
      - ``data/val_images/ILSVRC2012_val_<id>_<synset>.JPEG``

    The synset-to-index mapping is read from a ``classes.py`` file placed
    next to the dataset root (the same file the HF builder uses).
    """

    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []

        # ------------------------------------------------------------------
        # 1. Load class mapping from classes.py
        # ------------------------------------------------------------------
        classes_path = os.path.join(root, "classes.py")
        if not os.path.isfile(classes_path):
            raise FileNotFoundError(
                f"ImageNet flat layout detected but no classes.py found in {root}. "
                f"Please place the HuggingFace classes.py next to the dataset."
            )
        synset_to_idx = _load_imagenet_classes(classes_path)

        # ------------------------------------------------------------------
        # 2. Re-use cached file list if available
        # ------------------------------------------------------------------
        cache_path = os.path.join(root, f"._imagenet_flat_{split}_cache.pt")
        if os.path.exists(cache_path):
            self.samples = torch.load(cache_path)
            return

        # ------------------------------------------------------------------
        # 3. Scan directories and parse synsets from filenames
        # ------------------------------------------------------------------
        if split == "train":
            train_dirs = [
                p for p in sorted(glob.glob(os.path.join(root, "data", "train_images_*")))
                if os.path.isdir(p)
            ]
            for d in train_dirs:
                for entry in os.scandir(d):
                    if not entry.is_file():
                        continue
                    fname = entry.name
                    if not fname.lower().endswith((".jpeg", ".jpg")):
                        continue
                    # Parse synset: the last _-separated token before the extension
                    # e.g. n01440764_10159_n01440764.JPEG  ->  n01440764
                    stem, _ = os.path.splitext(fname)
                    parts = stem.rsplit("_", 1)
                    if len(parts) != 2:
                        continue
                    synset = parts[-1]
                    label = synset_to_idx.get(synset)
                    if label is not None:
                        self.samples.append((entry.path, label))
        elif split == "val":
            val_dir = os.path.join(root, "data", "val_images")
            for entry in os.scandir(val_dir):
                if not entry.is_file():
                    continue
                fname = entry.name
                if not fname.lower().endswith((".jpeg", ".jpg")):
                    continue
                stem, _ = os.path.splitext(fname)
                parts = stem.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                synset = parts[-1]
                label = synset_to_idx.get(synset)
                if label is not None:
                    self.samples.append((entry.path, label))
        else:
            raise ValueError(f"Unknown split: {split}")

        # Persist cache for fast subsequent loads
        torch.save(self.samples, cache_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _has_imagenet_flat_layout(root):
    """Return True if ``root`` contains the flat HuggingFace-style ImageNet layout."""
    data_dir = os.path.join(root, "data")
    if not os.path.isdir(data_dir):
        return False
    has_train = any(
        os.path.isdir(p) for p in glob.glob(os.path.join(data_dir, "train_images_*"))
    )
    has_val = os.path.isdir(os.path.join(data_dir, "val_images"))
    return has_train and has_val


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

    Auto-detects the dataset layout:

    1. **Standard ImageFolder** — ``root/train/`` and ``root/val/`` with one
       subdirectory per class.
    2. **Flat HuggingFace layout** — ``root/data/train_images_*/`` and
       ``root/data/val_images/`` where the synset is in the filename.
       Requires ``classes.py`` (from the HuggingFace builder) to be present
       in ``root/``.

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

    if _has_imagenet_flat_layout(root):
        ds = _ImageNetFlatDataset(root, split="train", transform=t_train)
        ds_val = _ImageNetFlatDataset(root, split="val", transform=t_val)
    else:
        ds = ImageFolder(root=f"{root}/train", transform=t_train)
        ds_val = ImageFolder(root=f"{root}/val", transform=t_val)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl, dl_val
