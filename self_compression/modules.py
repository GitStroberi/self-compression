"""Custom SCNN layers with both reference (hard clamp) and differentiable (smooth soft-clamp) quantization."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cast_tuple(t, length=1):
    """Coerce a scalar to a tuple of given length."""
    return t if isinstance(t, tuple) else ((t,) * length)


def hard_clamp(x, b):
    """Hard min/max clamp from the original SCNN reference implementation.

    Maps inputs into the range [-2^{b-1}, 2^{b-1} - 1] using torch.min/max.
    This is non-differentiable at the boundaries but matches the paper exactly.

    Args:
        x: Input tensor.
        b: Bit-width parameter (non-negative after ReLU).

    Returns:
        Tensor of the same shape as x, hard-clamped.
    """
    b = F.relu(b)
    c = 2 ** (b - 1)
    return torch.minimum(torch.maximum(x, -c), c - 1)


def smooth_soft_clamp(x, b):
    """Smooth soft-clamp for differentiable quantization (our novelty).

    Maps inputs into the range (-2^{b-1}, 2^{b-1}) using:
        out = x / (1 + |x| / 2^{b-1})

    Args:
        x: Input tensor.
        b: Bit-width parameter (learnable, non-negative after ReLU).

    Returns:
        Tensor of the same shape as x, smoothly clamped.
    """
    b = F.relu(b)
    c = 2 ** (b - 1)
    c = torch.clamp(c, min=1e-8)
    return x / (1.0 + torch.abs(x) / c)


def bits_needed_for_range(vmin, vmax):
    """Minimum signed-integer bits to store range [vmin, vmax].

    Returns the smallest bit-width (2..8) that can represent the given
    signed integer range without overflow.
    """
    max_abs = max(abs(int(vmin)), abs(int(vmax)))
    if max_abs <= 1:
        return 2
    elif max_abs <= 3:
        return 3
    elif max_abs <= 7:
        return 4
    elif max_abs <= 15:
        return 5
    elif max_abs <= 31:
        return 6
    elif max_abs <= 63:
        return 7
    return 8


class SCNNConv2d(nn.Module):
    """Quantized 2D convolution with learnable bit-width and exponent.

    During training the weight is soft-clamped to a learned dynamic range
    and discretized with a straight-through estimator. After training, integer
    weights can be extracted and the layer can run in a fast "compressed" mode
    that skips the quantization math.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, init_b=2.0, init_e=-8.0):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 2)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        scale = 1 / math.sqrt(in_channels * math.prod(self.kernel_size))
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size, dtype=torch.float32)
            .uniform_(-scale, scale)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.e = nn.Parameter(torch.full((out_channels, 1, 1, 1), init_e))
        self.b = nn.Parameter(torch.full((out_channels, 1, 1, 1), init_b))
        # Persistent buffer so state_dict includes it and compressed checkpoints work
        self.register_buffer("_compressed", torch.tensor(False))
        # Python flag avoids GPU sync in forward() (buffer stays for serialization)
        self._is_compressed = False

    def qbits(self):
        """Return total bits consumed by this layer's quantized weights."""
        return F.relu(self.b).sum() * math.prod(self.weight.shape[1:])

    def qweight(self):
        """Return the soft-clamped, scaled weight (before rounding / pruning)."""
        x = (2 ** (-self.e)) * self.weight
        return smooth_soft_clamp(x, self.b)

    def qweight_hard(self):
        """Return the hard-clamped, scaled weight (before rounding).

        Uses the original min/max clamp from the reference implementation.
        """
        x = (2 ** (-self.e)) * self.weight
        return hard_clamp(x, self.b)

    def get_integer_weights(self):
        """Extract the truly quantized integer weights for compressed storage.

        Returns:
            W_int: Integer weight tensor (same shape as self.weight), dtype int16
            e:     Exponent tensor (out_ch, 1, 1, 1)
            b:     Bit-width tensor (out_ch, 1, 1, 1)

        The original FP32 weight can be recovered as:  weight = (2**e) * W_int
        """
        with torch.no_grad():
            qw = self.qweight()
            W_int = torch.round(qw).to(torch.int16)
            return W_int, self.e.detach().cpu(), self.b.detach().cpu()

    def set_from_integer_weights(self, W_int, e, b):
        """Reconstruct FP32 weights from compressed integer representation.

        Call this after loading a compressed checkpoint. Sets the internal
        _compressed flag so forward() skips re-quantization.
        """
        self.e.data = e.to(self.e.device)
        self.b.data = b.to(self.b.device)
        self.weight.data = (2 ** self.e) * W_int.to(self.weight.dtype)
        self._compressed.fill_(True)
        self._is_compressed = True

    def forward(self, x):
        """Forward pass. Uses pre-quantized weights if _compressed is set."""
        if self._is_compressed:
            return F.conv2d(
                x, weight=self.weight, bias=self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups,
            )
        qw = self.qweight()
        w = (torch.round(qw) - qw).detach() + qw  # straight-through estimator
        weight = (2 ** self.e) * w
        return F.conv2d(
            x, weight=weight, bias=self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups,
        )
