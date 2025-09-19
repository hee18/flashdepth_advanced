#!/usr/bin/env python3
"""
Test script for the Global Scale Predictor (GSP) head implementation
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flashdepth.model import FlashDepth
from flashdepth.heads import GlobalScalePredictor, MetricDepthLoss
from utils.metric_depth_metrics import MetricDepthMetrics, format_metrics
from utils.metric_visualization import MetricDepthVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gsp_head():
    """Test the Global Scale Predictor head in isolation"""
    logger.info("Testing Global Scale Predictor head...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create GSP head
    gsp_head = GlobalScalePredictor(input_dim=1024, hidden_dim=256).to(device)

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


def test_metric_depth_conversion():
    """Test the metric depth conversion functionality"""
    logger.info("Testing metric depth conversion...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy data
    batch_size, height, width = 2, 64, 64
    relative_depth = torch.rand(batch_size, height, width).to(device) * 10  # 0-10 range
    scale = torch.tensor([[2.0], [1.5]]).to(device)  # Different scales for each batch
    shift = torch.tensor([[0.5], [-0.2]]).to(device)  # Different shifts for each batch

    # Create GSP head and test conversion
    gsp_head = GlobalScalePredictor().to(device)
    metric_depth = gsp_head.predict_metric_depth(relative_depth, scale, shift)

    # Check output shape
    assert metric_depth.shape == relative_depth.shape, \
        f"Expected shape {relative_depth.shape}, got {metric_depth.shape}"

    # Verify the conversion formula: D_metric = scale * D_rel + shift
    expected_metric_0 = scale[0, 0] * relative_depth[0] + shift[0, 0]
    expected_metric_1 = scale[1, 0] * relative_depth[1] + shift[1, 0]

    assert torch.allclose(metric_depth[0], expected_metric_0, atol=1e-6), \
        "Metric depth conversion incorrect for batch 0"
    assert torch.allclose(metric_depth[1], expected_metric_1, atol=1e-6), \
        "Metric depth conversion incorrect for batch 1"

    logger.info("✓ Metric depth conversion test passed!")

    return True


def test_flashdepth_integration():
    """Test FlashDepth integration with GSP head"""
    logger.info("Testing FlashDepth integration...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model configuration (using minimal config for testing)
    model_config = {
        'vit_size': 'vits',  # Use smaller model for testing
        'use_mamba': False,  # Disable mamba for simplicity
        'use_metric_head': True,  # Enable GSP head
        'training': False
    }

    try:
        # Create model
        model = FlashDepth(**model_config).to(device)
        model.eval()

        # Test CLS token extraction
        batch_size, channels, height, width = 1, 3, 224, 224
        dummy_image = torch.randn(batch_size, channels, height, width).to(device)

        cls_token = model.get_cls_token(dummy_image)
        expected_embed_dim = 384  # ViT-S embedding dimension
        assert cls_token.shape == (batch_size, expected_embed_dim), \
            f"Expected CLS token shape ({batch_size}, {expected_embed_dim}), got {cls_token.shape}"

        logger.info("✓ CLS token extraction test passed!")

        # Test forward pass with metric head (using dummy video data)
        video_shape = (1, 2, 3, 224, 224)  # B, T, C, H, W
        dummy_video = torch.randn(video_shape).to(device)
        dummy_gt = torch.randn(1, 2, 224, 224).to(device) * 10 + 1  # Positive depth values

        with torch.no_grad():
            outputs = model.forward_with_metric_head((dummy_video, dummy_gt), use_mamba=False)

        # Check output keys and shapes
        expected_keys = ['relative_depth', 'metric_depth', 'scale', 'shift']
        for key in expected_keys:
            assert key in outputs, f"Missing output key: {key}"

        # Check shapes
        expected_depth_shape = (1, 2, 224, 224)  # B, T, H, W
        expected_param_shape = (1, 2)  # B, T

        assert outputs['relative_depth'].shape == expected_depth_shape, \
            f"Wrong relative depth shape: {outputs['relative_depth'].shape}"
        assert outputs['metric_depth'].shape == expected_depth_shape, \
            f"Wrong metric depth shape: {outputs['metric_depth'].shape}"
        assert outputs['scale'].shape == expected_param_shape, \
            f"Wrong scale shape: {outputs['scale'].shape}"
        assert outputs['shift'].shape == expected_param_shape, \
            f"Wrong shift shape: {outputs['shift'].shape}"

        logger.info("✓ FlashDepth integration test passed!")
        logger.info(f"  Relative depth range: {outputs['relative_depth'].min():.3f} - {outputs['relative_depth'].max():.3f}")
        logger.info(f"  Metric depth range: {outputs['metric_depth'].min():.3f} - {outputs['metric_depth'].max():.3f}")
        logger.info(f"  Scale values: {outputs['scale'].flatten()}")
        logger.info(f"  Shift values: {outputs['shift'].flatten()}")

        return True

    except Exception as e:
        logger.error(f"FlashDepth integration test failed: {e}")
        return False


def test_loss_computation():
    """Test the metric depth loss computation"""
    logger.info("Testing loss computation...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy predictions and ground truth
    batch_size, height, width = 2, 32, 32
    pred_metric = torch.randn(batch_size, height, width).to(device) * 5 + 5  # Positive depths
    gt_metric = torch.randn(batch_size, height, width).to(device) * 5 + 5    # Positive depths

    # Create valid mask (some invalid pixels)
    valid_mask = torch.rand(batch_size, height, width).to(device) > 0.1

    # Test L1 loss
    loss_fn_l1 = MetricDepthLoss(loss_type='l1')
    loss_l1 = loss_fn_l1(pred_metric, gt_metric, valid_mask)

    # Test L2 loss
    loss_fn_l2 = MetricDepthLoss(loss_type='l2')
    loss_l2 = loss_fn_l2(pred_metric, gt_metric, valid_mask)

    # Check that losses are positive scalars
    assert loss_l1.dim() == 0, "L1 loss should be a scalar"
    assert loss_l2.dim() == 0, "L2 loss should be a scalar"
    assert loss_l1.item() >= 0, "L1 loss should be non-negative"
    assert loss_l2.item() >= 0, "L2 loss should be non-negative"

    logger.info("✓ Loss computation test passed!")
    logger.info(f"  L1 loss: {loss_l1.item():.4f}")
    logger.info(f"  L2 loss: {loss_l2.item():.4f}")

    return True


def test_metrics_computation():
    """Test the metrics computation"""
    logger.info("Testing metrics computation...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy data with known properties
    height, width = 64, 64

    # Create ground truth with realistic depth values (1-20 meters)
    gt_depth = torch.rand(height, width) * 19 + 1

    # Create prediction with some error
    noise = torch.randn(height, width) * 0.5  # Add noise
    pred_depth = gt_depth + noise
    pred_depth = torch.clamp(pred_depth, min=0.1)  # Ensure positive

    # Create valid mask
    valid_mask = torch.ones_like(gt_depth).bool()

    # Compute comprehensive metrics
    metrics = MetricDepthMetrics.compute_comprehensive_metrics(
        pred_depth, gt_depth, valid_mask
    )

    # Check that essential metrics are present
    essential_metrics = ['mae', 'rmse', 'abs_rel', 'a1']
    for metric in essential_metrics:
        assert metric in metrics, f"Missing essential metric: {metric}"
        assert 0 <= metrics[metric] < float('inf'), f"Invalid {metric} value: {metrics[metric]}"

    # Check that accuracy metrics are between 0 and 1
    accuracy_metrics = ['a1', 'a2', 'a3']
    for metric in accuracy_metrics:
        if metric in metrics:
            assert 0 <= metrics[metric] <= 1, f"{metric} should be between 0 and 1"

    logger.info("✓ Metrics computation test passed!")
    logger.info("Sample metrics:")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  AbsRel: {metrics['abs_rel']:.4f}")
    logger.info(f"  δ1: {metrics['a1']:.4f}")

    return True


def test_parameter_freezing():
    """Test that only GSP head parameters are trainable"""
    logger.info("Testing parameter freezing...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_config = {
        'vit_size': 'vits',
        'use_mamba': False,
        'use_metric_head': True,
        'training': False
    }

    try:
        model = FlashDepth(**model_config).to(device)

        # Simulate freezing (as done in training script)
        trainable_params = []
        frozen_params = []

        for name, param in model.named_parameters():
            if name.startswith('gsp_head'):
                param.requires_grad = True
                trainable_params.append(name)
            else:
                param.requires_grad = False
                frozen_params.append(name)

        # Check that we have both trainable and frozen parameters
        assert len(trainable_params) > 0, "No trainable parameters found!"
        assert len(frozen_params) > 0, "No frozen parameters found!"

        # Check that all GSP head parameters are trainable
        gsp_param_count = sum(1 for name, _ in model.named_parameters()
                             if name.startswith('gsp_head'))
        assert len(trainable_params) == gsp_param_count, \
            f"Expected {gsp_param_count} GSP parameters, got {len(trainable_params)}"

        logger.info("✓ Parameter freezing test passed!")
        logger.info(f"  Trainable parameters: {len(trainable_params)}")
        logger.info(f"  Frozen parameters: {len(frozen_params)}")
        logger.info(f"  Trainable parameter names: {trainable_params}")

        return True

    except Exception as e:
        logger.error(f"Parameter freezing test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    logger.info("="*50)
    logger.info("Running GSP Head Implementation Tests")
    logger.info("="*50)

    tests = [
        ("GSP Head", test_gsp_head),
        ("Metric Depth Conversion", test_metric_depth_conversion),
        ("FlashDepth Integration", test_flashdepth_integration),
        ("Loss Computation", test_loss_computation),
        ("Metrics Computation", test_metrics_computation),
        ("Parameter Freezing", test_parameter_freezing),
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
    success = run_all_tests()
    sys.exit(0 if success else 1)