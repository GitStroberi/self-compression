"""Simple dataset loaders."""

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST


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
