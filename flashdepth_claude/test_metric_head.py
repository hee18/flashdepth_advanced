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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
from einops import rearrange

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flashdepth.model import FlashDepth
from flashdepth.heads import GlobalScalePredictor, MetricDepthLoss
from utils.metric_depth_metrics import MetricDepthMetrics, format_metrics
try:
    from utils.metric_visualization import MetricDepthVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    MetricDepthVisualizer = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_depth_colormap(depth_map, valid_mask=None, colormap='plasma'):
    """Convert depth map to colored visualization"""
    if valid_mask is not None:
        depth_map = depth_map.clone()
        depth_map[~valid_mask] = 0

    depth_np = depth_map.cpu().numpy()

    # Normalize to 0-1 range
    if valid_mask is not None:
        valid_depth = depth_np[valid_mask.cpu().numpy()]
        if len(valid_depth) > 0:
            min_depth, max_depth = valid_depth.min(), valid_depth.max()
        else:
            min_depth, max_depth = 0, 1
    else:
        min_depth, max_depth = depth_np.min(), depth_np.max()

    if max_depth > min_depth:
        normalized = (depth_np - min_depth) / (max_depth - min_depth)
    else:
        normalized = np.zeros_like(depth_np)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(normalized)

    # Convert to uint8 RGB
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    return rgb


def create_sequence_visualization(images, pred_depths, gt_depths, valid_masks=None,
                                save_path=None, title="Metric Depth Prediction"):
    """
    Create visualization showing sequence of frames with predictions and ground truth

    Args:
        images: [T, 3, H, W] or [T, H, W, 3] - input images
        pred_depths: [T, H, W] - predicted depths
        gt_depths: [T, H, W] - ground truth depths
        valid_masks: [T, H, W] - valid pixel masks (optional)
        save_path: str - path to save visualization
        title: str - title for the plot
    """
    T = len(images)

    # Convert tensors to numpy
    if isinstance(images, torch.Tensor):
        if images.shape[1] == 3:  # [T, 3, H, W]
            images = images.permute(0, 2, 3, 1)  # [T, H, W, 3]
        images = images.cpu().numpy()

    if isinstance(pred_depths, torch.Tensor):
        pred_depths = pred_depths.cpu()
    if isinstance(gt_depths, torch.Tensor):
        gt_depths = gt_depths.cpu()
    if valid_masks is not None and isinstance(valid_masks, torch.Tensor):
        valid_masks = valid_masks.cpu()

    # Normalize images to 0-1 range if needed
    if images.max() > 1.0:
        images = images / 255.0

    # Create figure with subplots
    fig = plt.figure(figsize=(T * 4, 12))
    gs = gridspec.GridSpec(3, T, hspace=0.3, wspace=0.1)

    for t in range(T):
        # Original image
        ax_img = fig.add_subplot(gs[0, t])
        ax_img.imshow(images[t])
        ax_img.set_title(f'Frame {t+1}')
        ax_img.axis('off')

        # Predicted depth
        ax_pred = fig.add_subplot(gs[1, t])
        mask = valid_masks[t] if valid_masks is not None else None
        pred_colored = create_depth_colormap(pred_depths[t], mask, 'plasma')
        ax_pred.imshow(pred_colored)
        ax_pred.set_title(f'Predicted Depth')
        ax_pred.axis('off')

        # Ground truth depth
        ax_gt = fig.add_subplot(gs[2, t])
        gt_colored = create_depth_colormap(gt_depths[t], mask, 'plasma')
        ax_gt.imshow(gt_colored)
        ax_gt.set_title(f'Ground Truth')
        ax_gt.axis('off')

    plt.suptitle(title, fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Sequence visualization saved to {save_path}")

    return fig


def create_comparison_visualization(pred_depth, gt_depth, valid_mask=None,
                                  save_path=None, title="Depth Comparison"):
    """
    Create side-by-side comparison of predicted vs ground truth depth
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Predicted depth
    pred_colored = create_depth_colormap(pred_depth, valid_mask, 'plasma')
    axes[0].imshow(pred_colored)
    axes[0].set_title('Predicted Depth')
    axes[0].axis('off')

    # Ground truth depth
    gt_colored = create_depth_colormap(gt_depth, valid_mask, 'plasma')
    axes[1].imshow(gt_colored)
    axes[1].set_title('Ground Truth Depth')
    axes[1].axis('off')

    # Error map
    if valid_mask is not None:
        error = torch.abs(pred_depth - gt_depth)
        error[~valid_mask] = 0
    else:
        error = torch.abs(pred_depth - gt_depth)

    error_colored = create_depth_colormap(error, valid_mask, 'hot')
    axes[2].imshow(error_colored)
    axes[2].set_title('Absolute Error')
    axes[2].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Comparison visualization saved to {save_path}")

    return fig


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


def test_visualization():
    """Test the visualization functionality"""
    logger.info("Testing visualization functionality...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy sequence data
    T, H, W = 3, 64, 64

    # Create dummy RGB images (normalized to 0-1)
    images = torch.rand(T, 3, H, W) * 0.8 + 0.1  # Avoid pure black/white

    # Create realistic depth maps
    pred_depths = torch.rand(T, H, W) * 10 + 1  # 1-11 meters
    gt_depths = pred_depths + torch.randn_like(pred_depths) * 0.5  # Add noise
    gt_depths = torch.clamp(gt_depths, min=0.1)  # Ensure positive

    # Create valid masks (simulate some invalid pixels)
    valid_masks = torch.rand(T, H, W) > 0.1

    try:
        # Create output directory
        vis_dir = Path("test_results/visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Test sequence visualization
        logger.info("Creating sequence visualization...")
        seq_fig = create_sequence_visualization(
            images, pred_depths, gt_depths, valid_masks,
            save_path=vis_dir / "test_sequence.png",
            title="Test Sequence Visualization"
        )
        plt.close(seq_fig)

        # Test comparison visualization (single frame)
        logger.info("Creating comparison visualization...")
        comp_fig = create_comparison_visualization(
            pred_depths[0], gt_depths[0], valid_masks[0],
            save_path=vis_dir / "test_comparison.png",
            title="Test Depth Comparison"
        )
        plt.close(comp_fig)

        # Test with original MetricDepthVisualizer if available
        if VISUALIZATION_AVAILABLE and MetricDepthVisualizer is not None:
            logger.info("Testing MetricDepthVisualizer...")
            visualizer = MetricDepthVisualizer(save_dir=vis_dir)

            # Create dummy batch data for visualizer
            batch = (images.unsqueeze(0), gt_depths.unsqueeze(0), ['test_dataset'])
            outputs = {
                'metric_depth': pred_depths.unsqueeze(0),
                'relative_depth': pred_depths.unsqueeze(0) * 0.8,  # Slightly different
                'scale': torch.tensor([[2.0, 1.8, 1.5]]),
                'shift': torch.tensor([[0.1, -0.2, 0.3]])
            }

            try:
                visualizer.create_validation_summary(batch, outputs, global_step=0)
                logger.info("✓ MetricDepthVisualizer test passed!")
            except Exception as e:
                logger.warning(f"MetricDepthVisualizer test failed: {e}")

        logger.info("✓ Visualization test passed!")
        logger.info(f"  Visualizations saved to: {vis_dir}")

        return True

    except Exception as e:
        logger.error(f"Visualization test failed: {e}")
        return False


def test_comprehensive_integration():
    """Test comprehensive integration with real-like data and visualization"""
    logger.info("Testing comprehensive integration...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model configuration
    model_config = {
        'vit_size': 'vits',
        'use_mamba': False,
        'use_metric_head': True,
        'training': False
    }

    try:
        # Create model
        model = FlashDepth(**model_config).to(device)
        model.eval()

        # Create realistic test data
        B, T, C, H, W = 1, 3, 3, 224, 224

        # Create video sequence (normalized RGB)
        video = torch.rand(B, T, C, H, W).to(device) * 0.8 + 0.1

        # Create realistic ground truth depth (1-50 meters)
        gt_depth = torch.rand(B, T, H, W).to(device) * 49 + 1

        # Create valid mask
        valid_mask = torch.rand(B, T, H, W).to(device) > 0.05

        # Forward pass
        with torch.no_grad():
            outputs = model.forward_with_metric_head((video, gt_depth), use_mamba=False)

        # Extract outputs
        pred_metric = outputs['metric_depth']
        pred_relative = outputs['relative_depth']
        scale = outputs['scale']
        shift = outputs['shift']

        # Compute metrics
        metrics = {}
        for t in range(T):
            frame_metrics = MetricDepthMetrics.compute_comprehensive_metrics(
                pred_metric[0, t], gt_depth[0, t], valid_mask[0, t]
            )
            for k, v in frame_metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

        # Average metrics across frames
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        # Create comprehensive visualization
        vis_dir = Path("test_results/comprehensive")
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data for visualization (remove batch dimension)
        vis_images = video[0]  # [T, C, H, W]
        vis_pred = pred_metric[0]  # [T, H, W]
        vis_gt = gt_depth[0]  # [T, H, W]
        vis_mask = valid_mask[0]  # [T, H, W]

        # Create sequence visualization
        seq_fig = create_sequence_visualization(
            vis_images, vis_pred, vis_gt, vis_mask,
            save_path=vis_dir / "comprehensive_sequence.png",
            title=f"Comprehensive Test - MAE: {avg_metrics['mae']:.3f}m, AbsRel: {avg_metrics['abs_rel']:.3f}"
        )
        plt.close(seq_fig)

        # Create comparison for each frame
        for t in range(T):
            comp_fig = create_comparison_visualization(
                vis_pred[t], vis_gt[t], vis_mask[t],
                save_path=vis_dir / f"frame_{t+1}_comparison.png",
                title=f"Frame {t+1} - MAE: {metrics['mae'][t]:.3f}m"
            )
            plt.close(comp_fig)

        logger.info("✓ Comprehensive integration test passed!")
        logger.info(f"  Processed video shape: {video.shape}")
        logger.info(f"  Average metrics across {T} frames:")
        for metric_name, value in avg_metrics.items():
            if metric_name in ['mae', 'rmse']:
                logger.info(f"    {metric_name.upper()}: {value:.4f}m")
            elif metric_name in ['abs_rel', 'sq_rel']:
                logger.info(f"    {metric_name.upper()}: {value:.4f}")
            elif metric_name.startswith('a'):
                logger.info(f"    δ{metric_name[1:]}: {value:.4f}")

        logger.info(f"  Scale range: [{scale.min():.3f}, {scale.max():.3f}]")
        logger.info(f"  Shift range: [{shift.min():.3f}, {shift.max():.3f}]")

        return True

    except Exception as e:
        logger.error(f"Comprehensive integration test failed: {e}")
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
        ("Visualization", test_visualization),
        ("Comprehensive Integration", test_comprehensive_integration),
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