#!/usr/bin/env python3
"""
Simple test for GSP head without full FlashDepth dependencies
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple GSP implementation for testing
class SimpleGlobalScalePredictor(nn.Module):
    """Standalone GSP implementation for testing"""

    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # [scale, shift]
        )

        # Initialize weights
        with torch.no_grad():
            self.mlp[-1].weight.data.fill_(0.1)
            self.mlp[-1].bias.data[0] = 1.0  # scale
            self.mlp[-1].bias.data[1] = 0.0  # shift

    def forward(self, cls_token):
        if cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)

        output = self.mlp(cls_token)
        scale_raw, shift = output[:, 0:1], output[:, 1:2]
        scale = torch.nn.functional.softplus(scale_raw)

        return scale, shift

    def predict_metric_depth(self, relative_depth, scale, shift):
        if scale.dim() == 2:
            scale = scale.unsqueeze(-1)
        if shift.dim() == 2:
            shift = shift.unsqueeze(-1)

        metric_depth = scale * relative_depth + shift
        return metric_depth


def test_simple_gsp():
    """Test the GSP head implementation"""
    logger.info("Testing Simple Global Scale Predictor...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create GSP head
    gsp_head = SimpleGlobalScalePredictor(input_dim=1024, hidden_dim=256).to(device)

    # Test with dummy CLS tokens
    batch_size = 2
    cls_tokens = torch.randn(batch_size, 1024).to(device)

    # Forward pass
    scale, shift = gsp_head(cls_tokens)

    # Check output shapes
    assert scale.shape == (batch_size, 1), f"Expected scale shape ({batch_size}, 1), got {scale.shape}"
    assert shift.shape == (batch_size, 1), f"Expected shift shape ({batch_size}, 1), got {shift.shape}"

    # Check that scale is positive
    assert torch.all(scale > 0), "Scale values should be positive"

    logger.info(f"✓ GSP head test passed!")
    logger.info(f"  Scale range: {scale.min().item():.4f} - {scale.max().item():.4f}")
    logger.info(f"  Shift range: {shift.min().item():.4f} - {shift.max().item():.4f}")

    return True


def test_metric_conversion():
    """Test metric depth conversion"""
    logger.info("Testing metric depth conversion...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy data
    batch_size, height, width = 2, 64, 64
    relative_depth = torch.rand(batch_size, height, width).to(device) * 10
    scale = torch.tensor([[2.0], [1.5]]).to(device)
    shift = torch.tensor([[0.5], [-0.2]]).to(device)

    # Create GSP head
    gsp_head = SimpleGlobalScalePredictor().to(device)
    metric_depth = gsp_head.predict_metric_depth(relative_depth, scale, shift)

    # Check output shape
    assert metric_depth.shape == relative_depth.shape

    # Verify conversion formula
    expected_0 = scale[0, 0] * relative_depth[0] + shift[0, 0]
    expected_1 = scale[1, 0] * relative_depth[1] + shift[1, 0]

    assert torch.allclose(metric_depth[0], expected_0, atol=1e-6)
    assert torch.allclose(metric_depth[1], expected_1, atol=1e-6)

    logger.info("✓ Metric depth conversion test passed!")

    return True


def test_gradient_flow():
    """Test that gradients flow correctly through GSP"""
    logger.info("Testing gradient flow...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gsp_head = SimpleGlobalScalePredictor().to(device)
    cls_tokens = torch.randn(2, 1024, requires_grad=True).to(device)

    # Forward pass
    scale, shift = gsp_head(cls_tokens)

    # Simple loss (sum of scale and shift)
    loss = (scale.sum() + shift.sum())

    # Backward pass
    loss.backward()

    # Check that gradients exist
    assert cls_tokens.grad is not None, "No gradient for input"
    for param in gsp_head.parameters():
        assert param.grad is not None, f"No gradient for parameter {param.shape}"

    logger.info("✓ Gradient flow test passed!")

    return True


def test_parameter_count():
    """Test parameter count"""
    logger.info("Testing parameter count...")

    gsp_head = SimpleGlobalScalePredictor(input_dim=1024, hidden_dim=256)

    total_params = sum(p.numel() for p in gsp_head.parameters())
    trainable_params = sum(p.numel() for p in gsp_head.parameters() if p.requires_grad)

    expected_params = (1024 * 256) + 256 + (256 * 2) + 2  # Linear layers with bias
    assert total_params == expected_params, f"Expected {expected_params} params, got {total_params}"
    assert trainable_params == total_params, "All parameters should be trainable by default"

    logger.info(f"✓ Parameter count test passed!")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    return True


def run_simple_tests():
    """Run simple tests"""
    logger.info("="*50)
    logger.info("Running Simple GSP Tests")
    logger.info("="*50)

    tests = [
        ("Simple GSP Head", test_simple_gsp),
        ("Metric Conversion", test_metric_conversion),
        ("Gradient Flow", test_gradient_flow),
        ("Parameter Count", test_parameter_count),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED with exception: {e}")

    logger.info("\n" + "="*50)
    logger.info(f"Test Summary: {passed} PASSED, {failed} FAILED")
    logger.info("="*50)

    return failed == 0


if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)