"""
Test script for Gear5 architecture with dummy data.
Validates forward pass, loss computation, and gradient flow.
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/cvlab/hsy/flashdepth_advanced/flashdepth_claude')

from flashdepth.gear5_modules import Gear5MetricHead, TemporalScalePredictor, ImportanceMapGenerator

def test_importance_map_generator():
    """Test ImportanceMapGenerator with dummy attention weights."""
    print("\n" + "="*80)
    print("Testing ImportanceMapGenerator...")
    print("="*80)

    generator = ImportanceMapGenerator(num_layers=2)

    # Create dummy attention weights [B, num_heads, N+1, N+1]
    B, T = 2, 5
    num_heads = 16
    patch_h, patch_w = 37, 37
    num_patches = patch_h * patch_w
    N_plus_1 = num_patches + 1  # Including CLS token

    # Attention from 2 layers
    attn_layer_1 = torch.rand(B*T, num_heads, N_plus_1, N_plus_1)
    attn_layer_2 = torch.rand(B*T, num_heads, N_plus_1, N_plus_1)

    # Normalize to sum to 1 (softmax-like)
    attn_layer_1 = F.softmax(attn_layer_1, dim=-1)
    attn_layer_2 = F.softmax(attn_layer_2, dim=-1)

    attention_weights_list = [attn_layer_1, attn_layer_2]

    # Forward pass
    importance_map = generator(attention_weights_list, patch_h, patch_w)

    print(f"  Input attention shapes: {attn_layer_1.shape}, {attn_layer_2.shape}")
    print(f"  Output importance map shape: {importance_map.shape}")
    print(f"  Expected shape: [{B*T}, 1, {patch_h}, {patch_w}]")
    print(f"  Importance map range: [{importance_map.min():.4f}, {importance_map.max():.4f}]")
    print(f"  Importance map mean: {importance_map.mean():.4f}")

    # Check properties
    assert importance_map.shape == (B*T, 1, patch_h, patch_w), "Shape mismatch!"
    assert importance_map.min() >= 0.0 and importance_map.max() <= 1.0, "Values out of [0, 1] range!"

    print("  ✓ ImportanceMapGenerator test passed!")
    return importance_map


def test_temporal_scale_predictor():
    """Test TemporalScalePredictor with dummy CLS tokens."""
    print("\n" + "="*80)
    print("Testing TemporalScalePredictor...")
    print("="*80)

    predictor = TemporalScalePredictor(
        embed_dim=1024,
        feature_dim=256,
        hidden_dim=128,
        num_layers=1
    )

    # Create dummy CLS tokens [B, T, 1024]
    B, T = 2, 5
    cls_tokens = torch.randn(B, T, 1024)

    # Forward pass
    scale, shift = predictor(cls_tokens)

    print(f"  Input CLS tokens shape: {cls_tokens.shape}")
    print(f"  Output scale shape: {scale.shape}")
    print(f"  Output shift shape: {shift.shape}")
    print(f"  Expected shapes: [{B}, {T}] each")
    print(f"  Scale range: [{scale.min():.4f}, {scale.max():.4f}] (should be positive)")
    print(f"  Shift range: [{shift.min():.4f}, {shift.max():.4f}]")

    # Check properties
    assert scale.shape == (B, T), "Scale shape mismatch!"
    assert shift.shape == (B, T), "Shift shape mismatch!"
    assert (scale > 0).all(), "Scale should be positive!"

    # Test gradient flow
    loss = scale.sum() + shift.sum()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.parameters())
    assert has_grad, "No gradients flowing through predictor!"

    print("  ✓ TemporalScalePredictor test passed!")
    print("  ✓ Gradient flow verified!")
    return scale, shift


def test_gear5_metric_head():
    """Test Gear5MetricHead with dummy inputs."""
    print("\n" + "="*80)
    print("Testing Gear5MetricHead...")
    print("="*80)

    head = Gear5MetricHead(
        embed_dim=1024,
        feature_dim=256,
        hidden_dim=128
    )

    # Create dummy inputs
    B, T = 2, 5
    num_heads = 16
    patch_h, patch_w = 37, 37
    num_patches = patch_h * patch_w
    N_plus_1 = num_patches + 1

    # CLS tokens [B, T, 1024]
    cls_tokens = torch.randn(B, T, 1024)

    # Attention weights from 2 layers [B*T, num_heads, N+1, N+1]
    attn_layer_1 = F.softmax(torch.rand(B*T, num_heads, N_plus_1, N_plus_1), dim=-1)
    attn_layer_2 = F.softmax(torch.rand(B*T, num_heads, N_plus_1, N_plus_1), dim=-1)
    attention_weights_list = [attn_layer_1, attn_layer_2]

    # Forward pass
    outputs = head(cls_tokens, attention_weights_list, patch_h, patch_w)

    print(f"  Input CLS tokens: {cls_tokens.shape}")
    print(f"  Input attention: 2 layers × {attn_layer_1.shape}")
    print(f"  Output keys: {list(outputs.keys())}")
    print(f"  Output scale shape: {outputs['scale'].shape}")
    print(f"  Output shift shape: {outputs['shift'].shape}")
    print(f"  Output importance_map shape: {outputs['importance_map'].shape}")
    print(f"  Expected importance_map shape: [{B}, {T}, {patch_h}, {patch_w}]")

    # Check shapes
    assert outputs['scale'].shape == (B, T), "Scale shape mismatch!"
    assert outputs['shift'].shape == (B, T), "Shift shape mismatch!"
    assert outputs['importance_map'].shape == (B, T, patch_h, patch_w), "Importance map shape mismatch!"

    # Test gradient flow
    loss = outputs['scale'].sum() + outputs['shift'].sum() + outputs['importance_map'].sum()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert has_grad, "No gradients flowing through head!"

    print("  ✓ Gear5MetricHead test passed!")
    print("  ✓ Gradient flow verified!")
    return outputs


def test_importance_weighted_loss():
    """Test importance-weighted Log L1 loss computation."""
    print("\n" + "="*80)
    print("Testing Importance-weighted Loss...")
    print("="*80)

    # Create dummy data
    B, T, H, W = 2, 5, 518, 518
    patch_h, patch_w = 37, 37

    # Predicted and GT depth [B*T, 1, H, W]
    pred_depth = torch.rand(B*T, 1, H, W) * 50.0 + 0.1  # 0.1-50m
    gt_depth = torch.rand(B*T, 1, H, W) * 50.0 + 0.1

    # Valid mask (GT >= 0)
    valid_mask = (gt_depth >= 0).flatten()

    # Importance map [B, T, patch_h, patch_w] (requires grad for full test)
    importance_map = torch.rand(B, T, patch_h, patch_w, requires_grad=True)

    # Resize importance map to depth map size
    importance_map_resized = F.interpolate(
        importance_map.view(B*T, 1, patch_h, patch_w),
        size=(H, W),
        mode='bilinear',
        align_corners=True
    )
    importance_flat = importance_map_resized.flatten()

    # Compute fg_ratio (alpha)
    importance_threshold = importance_flat.mean()
    fg_mask = (importance_flat > importance_threshold)
    fg_ratio = fg_mask.float().mean()

    print(f"  Pred depth range: [{pred_depth.min():.2f}, {pred_depth.max():.2f}]")
    print(f"  GT depth range: [{gt_depth.min():.2f}, {gt_depth.max():.2f}]")
    print(f"  Valid ratio: {valid_mask.float().mean():.4f}")
    print(f"  FG ratio (α): {fg_ratio:.4f}")
    print(f"  Importance map range: [{importance_flat.min():.4f}, {importance_flat.max():.4f}]")

    # Flatten for loss
    pred_depth_flat = pred_depth.flatten()
    gt_depth_flat = gt_depth.flatten()

    # Importance-weighted Log L1 Loss
    epsilon = 1e-3
    pred_inv = 1.0 / (pred_depth_flat + epsilon)
    gt_inv = 1.0 / (gt_depth_flat + epsilon)

    loss = torch.abs(torch.log(pred_inv + epsilon) - torch.log(gt_inv + epsilon))
    weighted_loss = loss * (1.0 + fg_ratio * importance_flat)
    final_loss = weighted_loss[valid_mask].mean()

    print(f"  Unweighted loss mean: {loss[valid_mask].mean():.6f}")
    print(f"  Weighted loss mean: {final_loss:.6f}")
    print(f"  Weight range: [{(1.0 + fg_ratio * importance_flat).min():.4f}, {(1.0 + fg_ratio * importance_flat).max():.4f}]")

    # Test gradient flow (backward should work without errors)
    final_loss.backward()
    assert importance_map.grad is not None, "No gradient for importance_map!"

    print("  ✓ Importance-weighted loss computation passed!")
    print("  ✓ Gradient flow verified!")
    return final_loss


def test_full_pipeline():
    """Test full Gear5 pipeline end-to-end."""
    print("\n" + "="*80)
    print("Testing Full Gear5 Pipeline...")
    print("="*80)

    # Setup
    B, T = 2, 5
    H, W = 518, 518
    patch_h, patch_w = 37, 37
    num_heads = 16
    num_patches = patch_h * patch_w
    N_plus_1 = num_patches + 1

    # Create model
    gear5_head = Gear5MetricHead(embed_dim=1024, feature_dim=256, hidden_dim=128)

    # Dummy inputs
    cls_tokens = torch.randn(B, T, 1024)
    attn_1 = F.softmax(torch.rand(B*T, num_heads, N_plus_1, N_plus_1), dim=-1)
    attn_2 = F.softmax(torch.rand(B*T, num_heads, N_plus_1, N_plus_1), dim=-1)
    attention_weights_list = [attn_1, attn_2]

    # Simulate relative depth from frozen FlashDepth
    relative_depth = torch.rand(B*T, 1, H, W) * 0.8 + 0.1  # Normalized [0.1, 0.9]

    # GT depth
    gt_depth = torch.rand(B*T, 1, H, W) * 50.0 + 0.1  # Metric depth [0.1, 50]

    print(f"  Input shapes:")
    print(f"    CLS tokens: {cls_tokens.shape}")
    print(f"    Attention: 2 × {attn_1.shape}")
    print(f"    Relative depth: {relative_depth.shape}")
    print(f"    GT depth: {gt_depth.shape}")

    # Forward pass
    gear5_outputs = gear5_head(cls_tokens, attention_weights_list, patch_h, patch_w)
    scale = gear5_outputs['scale']  # [B, T]
    shift = gear5_outputs['shift']  # [B, T]
    importance_map = gear5_outputs['importance_map']  # [B, T, patch_h, patch_w]

    # Apply scale/shift to relative depth
    scale_flat = scale.view(B*T, 1, 1, 1)
    shift_flat = shift.view(B*T, 1, 1, 1)
    metric_depth = scale_flat * relative_depth + shift_flat

    print(f"  Gear5 outputs:")
    print(f"    Scale: {scale.shape}, range [{scale.min():.3f}, {scale.max():.3f}]")
    print(f"    Shift: {shift.shape}, range [{shift.min():.3f}, {shift.max():.3f}]")
    print(f"    Importance map: {importance_map.shape}")
    print(f"  Metric depth: {metric_depth.shape}, range [{metric_depth.min():.2f}, {metric_depth.max():.2f}]")

    # Compute loss
    pred_depth_flat = metric_depth.flatten()
    gt_depth_flat = gt_depth.flatten()
    valid_mask = (gt_depth_flat >= 0)

    # Resize importance map
    importance_resized = F.interpolate(
        importance_map.view(B*T, 1, patch_h, patch_w),
        size=(H, W), mode='bilinear', align_corners=True
    )
    importance_flat = importance_resized.flatten()

    # FG ratio
    fg_mask = (importance_flat > importance_flat.mean())
    fg_ratio = fg_mask.float().mean()

    # Importance-weighted Log L1
    epsilon = 1e-3
    pred_inv = 1.0 / (pred_depth_flat + epsilon)
    gt_inv = 1.0 / (gt_depth_flat + epsilon)
    loss = torch.abs(torch.log(pred_inv + epsilon) - torch.log(gt_inv + epsilon))
    weighted_loss = loss * (1.0 + fg_ratio * importance_flat)
    final_loss = weighted_loss[valid_mask].mean()

    print(f"  Loss computation:")
    print(f"    FG ratio: {fg_ratio:.4f}")
    print(f"    Unweighted loss: {loss[valid_mask].mean():.6f}")
    print(f"    Final weighted loss: {final_loss:.6f}")

    # Test gradient flow
    final_loss.backward()

    # Check gradients
    param_grads = [(name, p.grad.abs().max().item() if p.grad is not None else 0.0)
                   for name, p in gear5_head.named_parameters()]
    print(f"  Gradient flow:")
    for name, grad_max in param_grads[:5]:  # Show first 5
        print(f"    {name}: max_grad = {grad_max:.6f}")

    has_grad = any(grad > 0 for _, grad in param_grads)
    assert has_grad, "No gradients flowing!"

    print("  ✓ Full pipeline test passed!")
    print("  ✓ End-to-end gradient flow verified!")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("GEAR5 DUMMY DATA VALIDATION")
    print("="*80)

    torch.manual_seed(42)

    try:
        # Component tests
        test_importance_map_generator()
        test_temporal_scale_predictor()
        test_gear5_metric_head()
        test_importance_weighted_loss()

        # Full pipeline
        test_full_pipeline()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nGear5 architecture is ready for training!")
        print("  • TemporalScalePredictor: ✓")
        print("  • ImportanceMapGenerator: ✓")
        print("  • Gear5MetricHead: ✓")
        print("  • Importance-weighted Loss: ✓")
        print("  • Full Pipeline: ✓")
        print("  • Gradient Flow: ✓")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
