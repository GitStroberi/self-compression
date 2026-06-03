"""Models that use SCNN quantized layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import SCNNConv2d


class Net(nn.Module):
    """MNIST model using SCNN quantized layers.

    The final convolution size is computed dynamically with a dummy forward
    pass so the architecture is robust to changes in kernel sizes or input
    dimensions.
    """

    def __init__(self, init_b=2.0, init_e=-8.0):
        super().__init__()
        self.conv1 = SCNNConv2d(1, 32, 5, init_b=init_b, init_e=init_e)
        self.conv2 = SCNNConv2d(32, 32, 5, init_b=init_b, init_e=init_e)
        self.conv3 = SCNNConv2d(32, 64, 3, init_b=init_b, init_e=init_e)
        self.conv4 = SCNNConv2d(64, 64, 3, init_b=init_b, init_e=init_e)

        self.bnorm1 = nn.BatchNorm2d(32, affine=False, track_running_stats=False)
        self.bnorm2 = nn.BatchNorm2d(64, affine=False, track_running_stats=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Compute flattened feature size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            out = self._conv_blocks(dummy)
            flat_features = out.view(1, -1).shape[1]
        self.final_conv = SCNNConv2d(flat_features, 10, 1, init_b=init_b, init_e=init_e)

    def _conv_blocks(self, x):
        """Feature extractor backbone (conv + pool blocks)."""
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.bnorm1(out)
        out = self.maxpool1(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.bnorm2(out)
        out = self.maxpool2(out)
        return out

    def forward(self, x):
        """Forward pass. Input: (N, 1, 28, 28). Output: (N, 10)."""
        out = self._conv_blocks(x)
        out = out.view(out.size(0), -1).unsqueeze(-1).unsqueeze(-1)
        out = self.final_conv(out)
        out = torch.flatten(out, 1)
        return out


class BasicBlockSCNN(nn.Module):
    """Basic ResNet block with SCNNConv2d."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, init_b=2.0, init_e=-8.0):
        super().__init__()
        self.conv1 = SCNNConv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False,
            init_b=init_b, init_e=init_e,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SCNNConv2d(
            planes, planes, 3, stride=1, padding=1, bias=False,
            init_b=init_b, init_e=init_e,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                SCNNConv2d(
                    in_planes, self.expansion * planes, 1, stride=stride,
                    bias=False, init_b=init_b, init_e=init_e,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        """Forward pass for a residual block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20SCNN(nn.Module):
    """ResNet20 for CIFAR-10 using SCNN quantized layers."""

    def __init__(self, num_classes=10, init_b=2.0, init_e=-8.0):
        super().__init__()
        self.in_planes = 16
        self.conv1 = SCNNConv2d(
            3, 16, 3, stride=1, padding=1, bias=False,
            init_b=init_b, init_e=init_e,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1, init_b=init_b, init_e=init_e)
        self.layer2 = self._make_layer(32, 3, stride=2, init_b=init_b, init_e=init_e)
        self.layer3 = self._make_layer(64, 3, stride=2, init_b=init_b, init_e=init_e)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride, init_b=2.0, init_e=-8.0):
        """Create a stack of BasicBlockSCNN layers."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlockSCNN(self.in_planes, planes, s, init_b=init_b, init_e=init_e))
            self.in_planes = planes * BasicBlockSCNN.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass. Input: (N, 3, 32, 32). Output: (N, num_classes)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class ResNet18SCNN(nn.Module):
    """ResNet18 for ImageNet-1k using SCNN quantized layers.

    Standard torchvision-style ResNet18 stem (7x7, stride 2) followed by
    maxpool and four residual stages with [2, 2, 2, 2] BasicBlocks.
    """

    def __init__(self, num_classes=1000, init_b=2.0, init_e=-8.0):
        super().__init__()
        self.in_planes = 64
        self.conv1 = SCNNConv2d(
            3, 64, 7, stride=2, padding=3, bias=False,
            init_b=init_b, init_e=init_e,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1, init_b=init_b, init_e=init_e)
        self.layer2 = self._make_layer(128, 2, stride=2, init_b=init_b, init_e=init_e)
        self.layer3 = self._make_layer(256, 2, stride=2, init_b=init_b, init_e=init_e)
        self.layer4 = self._make_layer(512, 2, stride=2, init_b=init_b, init_e=init_e)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride, init_b=2.0, init_e=-8.0):
        """Create a stack of BasicBlockSCNN layers."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlockSCNN(self.in_planes, planes, s, init_b=init_b, init_e=init_e))
            self.in_planes = planes * BasicBlockSCNN.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass. Input: (N, 3, 224, 224). Output: (N, num_classes)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
