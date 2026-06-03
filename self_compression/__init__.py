"""Self-Compressing Neural Networks package."""
from .modules import SCNNConv2d
from .models import Net, ResNet20SCNN
from .datasets import get_mnist, get_cifar10

__all__ = ["SCNNConv2d", "Net", "ResNet20SCNN", "get_mnist", "get_cifar10"]
