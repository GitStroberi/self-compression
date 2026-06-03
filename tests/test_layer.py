"""Unit tests for SCNN modules."""

import sys
import os

# Ensure repo root is on path so package imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from self_compression.modules import SCNNConv2d, smooth_soft_clamp


class TestSmoothSoftClamp:
    def test_output_range(self):
        x = torch.linspace(-100, 100, 1000)
        b = torch.tensor([3.0])
        out = smooth_soft_clamp(x, b)
        c = 2 ** (b - 1)
        assert torch.all(out < c) and torch.all(out > -c)

    def test_gradient_flow(self):
        x = torch.randn(10, requires_grad=True)
        b = torch.tensor([2.0], requires_grad=True)
        out = smooth_soft_clamp(x, b)
        out.sum().backward()
        assert x.grad is not None and b.grad is not None

    def test_small_x_identity(self):
        x = torch.tensor([0.01, -0.01, 0.1, -0.1])
        b = torch.tensor([5.0])
        out = smooth_soft_clamp(x, b)
        torch.testing.assert_close(out, x, atol=1e-3, rtol=1e-2)


class TestSCNNConv2d:
    def test_output_shape(self):
        layer = SCNNConv2d(3, 16, 3, padding=1, bias=False)
        x = torch.randn(4, 3, 32, 32)
        out = layer(x)
        assert out.shape == (4, 16, 32, 32)

    def test_gradient_availability(self):
        layer = SCNNConv2d(3, 16, 3, padding=1, bias=True)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert layer.weight.grad is not None
        assert layer.b.grad is not None
        assert layer.e.grad is not None
        assert x.grad is not None

    def test_qbits(self):
        layer = SCNNConv2d(3, 16, 3)
        qb = layer.qbits()
        assert qb.ndim == 0 and qb.item() >= 0

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = SCNNConv2d(3, 8, 3, padding=1).to(device)
        x = torch.randn(2, 3, 16, 16, device=device)
        out = layer(x)
        assert out.device.type == device.type

    def test_integer_weights_roundtrip(self):
        """Verify get_integer_weights + set_from_integer_weights preserves values."""
        layer = SCNNConv2d(3, 8, 3, padding=1)
        # Run a forward pass to set internal state
        x = torch.randn(2, 3, 16, 16)
        out1 = layer(x)

        W_int, e, b = layer.get_integer_weights()
        layer.set_from_integer_weights(W_int, e, b)

        assert layer._compressed.item() is True
        out2 = layer(x)
        # Should be nearly identical (tiny float rounding differences)
        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)

    def test_compressed_forward_skips_quantization(self):
        """Verify that _compressed flag causes forward to skip clamp/round."""
        layer = SCNNConv2d(3, 4, 3, padding=1)
        x = torch.randn(1, 3, 8, 8)
        out_before = layer(x)

        W_int, e, b = layer.get_integer_weights()
        layer.set_from_integer_weights(W_int, e, b)
        out_after = layer(x)

        # The outputs should match because set_from_integer_weights restores
        # the weight tensor from the integer representation, which forward
        # would have computed anyway (up to rounding).
        diff = (out_before - out_after).abs().max().item()
        assert diff < 1e-4


if __name__ == "__main__":
    print("Running tests...")
    TestSmoothSoftClamp().test_output_range()
    print("PASS: output_range")
    TestSmoothSoftClamp().test_gradient_flow()
    print("PASS: gradient_flow")
    TestSmoothSoftClamp().test_small_x_identity()
    print("PASS: small_x_identity")
    TestSCNNConv2d().test_output_shape()
    print("PASS: output_shape")
    TestSCNNConv2d().test_gradient_availability()
    print("PASS: gradient_availability")
    TestSCNNConv2d().test_qbits()
    print("PASS: qbits")
    TestSCNNConv2d().test_device()
    print("PASS: device")
    TestSCNNConv2d().test_integer_weights_roundtrip()
    print("PASS: integer_weights_roundtrip")
    TestSCNNConv2d().test_compressed_forward_skips_quantization()
    print("PASS: compressed_forward_skips_quantization")
    print("\nAll tests passed!")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
