"""Self-Compressing Neural Networks package."""
from .modules import SCNNConv2d
from .models import Net, ResNet18SCNN, ResNet20SCNN
from .datasets import get_imagenet1k, get_mnist, get_cifar10

__all__ = ["SCNNConv2d", "Net", "ResNet18SCNN", "ResNet20SCNN", "get_mnist", "get_cifar10", "get_imagenet1k"]
