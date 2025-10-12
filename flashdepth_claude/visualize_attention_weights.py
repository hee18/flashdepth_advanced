#!/usr/bin/env python3
"""
Importance Map Visualization Script

이 스크립트는 DINOv2 attention으로부터 생성된 Importance map을 시각화하여
FG/BG 분리가 적절히 이루어지는지 확인합니다.

사용법:
    python visualize_attention_weights.py \
        --image-path <path_to_image> \
        --checkpoint <path_to_flashdepth_checkpoint> \
        --output-dir visualizations/attention \
        --config-path configs/gear3

결과:
    - 원본 이미지
    - Importance map (register patches 제거, percentile 정규화)
    - FG mask (mean 기준, register patches 제외)
    - BG mask (mean 기준, register patches 제외)
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import hydra
from omegaconf import OmegaConf
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flashdepth.model import FlashDepth
from flashdepth.gear3_modules import process_attention_to_importance


def load_image(image_path, target_size=518):
    """Load and preprocess image for DINOv2"""
    img = Image.open(image_path).convert('RGB')

    # Resize to target size
    img = img.resize((target_size, target_size), Image.BILINEAR)

    # Convert to tensor and normalize (ImageNet stats)
    img_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_array - mean) / std

    # [H, W, 3] -> [1, 3, H, W]
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()

    return img_tensor, img


def extract_attention_weights(model, img_tensor, device, patch_h=37, patch_w=37):
    """
    Extract CLS→patch attention weights from last block and convert to importance map

    Returns:
        importance_map: [1, 1, patch_h, patch_w] - percentile normalized to [0,1], register patches removed
        attn_scores_raw: [1, num_patches] - raw attention (averaged over heads, before processing)
        attention_weights_full: [1, num_heads, num_patches+1, num_patches+1] - full attention weights
    """
    model.eval()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        # Forward through encoder
        features = model.pretrained.get_intermediate_layers(
            img_tensor,
            n=[4, 11, 17, 23],
            return_class_token=True,
            reshape=True
        )

        # Get last block attention weights
        last_block = model.pretrained.blocks[-1]

        # Check if attention weights are stored
        if not hasattr(last_block.attn, 'attn_weights') or last_block.attn.attn_weights is None:
            raise RuntimeError(
                "Attention weights not found! Make sure to enable "
                "store_attn_weights=True for the last block."
            )

        attn_weights = last_block.attn.attn_weights  # [B, num_heads, num_tokens, num_tokens]

        # Extract CLS→patch attention: [:, :, 0, 1:]
        cls_to_patch_attn = attn_weights[:, :, 0, 1:]  # [B, num_heads, num_patches]

        # Average over heads for raw visualization
        attn_scores_raw = cls_to_patch_attn.mean(dim=1)  # [B, num_patches]

        # Process attention to importance map (train_gear3.py 방식)
        # - Register patches 제거 (highest attention patch)
        # - Percentile normalization (1-99 percentile)
        importance_map = process_attention_to_importance(
            attn_weights, patch_h, patch_w, remove_outliers=True
        )

        return importance_map, attn_scores_raw, attn_weights


def visualize_attention(img_original, attn_scores_raw, importance_map, output_path, patch_size=14, img_size=518):
    """
    Visualize importance map with FG/BG separation (train_gear3.py 방식)

    Args:
        img_original: PIL Image
        attn_scores_raw: [1, num_patches] tensor - raw attention (before processing)
        importance_map: [1, 1, patch_h, patch_w] tensor - processed (register patches removed, percentile normalized)
        output_path: Path to save visualization
        patch_size: DINOv2 patch size (14)
        img_size: Input image size (518)
    """
    # Calculate grid size
    num_patches_per_side = img_size // patch_size  # 518 // 14 = 37

    # Raw attention map (before processing)
    attn_map_raw = attn_scores_raw[0].cpu().numpy().reshape(num_patches_per_side, num_patches_per_side)

    # Processed importance map (register patches removed, percentile normalized)
    importance_map_2d = importance_map[0, 0].cpu().numpy()  # [patch_h, patch_w]

    # FG/BG split: train_gear3.py 방식 (mean 기준, register patches 제외된 importance_map 사용)
    mean_val = importance_map_2d.mean()
    fg_mask = (importance_map_2d > mean_val).astype(np.float32)
    bg_mask = (importance_map_2d <= mean_val).astype(np.float32)

    # Upsample to image resolution for visualization
    importance_map_upsampled = F.interpolate(
        torch.from_numpy(importance_map_2d).unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode='bilinear',
        align_corners=True
    ).squeeze().numpy()

    fg_mask_upsampled = F.interpolate(
        torch.from_numpy(fg_mask).unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode='nearest'
    ).squeeze().numpy()

    bg_mask_upsampled = F.interpolate(
        torch.from_numpy(bg_mask).unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode='nearest'
    ).squeeze().numpy()

    # Create figure with 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Row 1, Col 1: Original Image
    axes[0, 0].imshow(img_original)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Row 1, Col 2: Importance Map (train_gear3.py 방식)
    im_importance = axes[0, 1].imshow(importance_map_upsampled, cmap='jet', interpolation='bilinear',
                                      vmin=0, vmax=1)
    axes[0, 1].set_title(
        f'Importance Map (train_gear3.py)\n'
        f'Register patches removed, Percentile normalized [0,1]\n'
        f'Mean: {mean_val:.4f}',
        fontsize=11, fontweight='bold'
    )
    axes[0, 1].axis('off')
    plt.colorbar(im_importance, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Row 2, Col 1: Foreground Mask (mean 기준, register patches 제외)
    axes[1, 0].imshow(img_original)
    fg_overlay = np.zeros((*fg_mask_upsampled.shape, 3))
    fg_overlay[..., 0] = fg_mask_upsampled  # Red channel
    axes[1, 0].imshow(fg_overlay, alpha=0.5)
    fg_ratio = fg_mask.sum() / fg_mask.size * 100
    axes[1, 0].set_title(
        f'Foreground Mask (Mean-based)\n'
        f'Importance > {mean_val:.3f}: {fg_ratio:.1f}%',
        fontsize=11, fontweight='bold'
    )
    axes[1, 0].axis('off')

    # Row 2, Col 2: Background Mask (mean 기준, register patches 제외)
    axes[1, 1].imshow(img_original)
    bg_overlay = np.zeros((*bg_mask_upsampled.shape, 3))
    bg_overlay[..., 2] = bg_mask_upsampled  # Blue channel
    axes[1, 1].imshow(bg_overlay, alpha=0.5)
    bg_ratio = bg_mask.sum() / bg_mask.size * 100
    axes[1, 1].set_title(
        f'Background Mask (Mean-based)\n'
        f'Importance ≤ {mean_val:.3f}: {bg_ratio:.1f}%',
        fontsize=11, fontweight='bold'
    )
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    plt.close()

    # Analyze register tokens (highest attention patch in raw attention)
    attn_flat = attn_map_raw.flatten()
    sorted_attn = np.sort(attn_flat)[::-1]  # Descending order

    # Find register patch location
    max_idx = np.argmax(attn_map_raw)
    register_row = max_idx // num_patches_per_side
    register_col = max_idx % num_patches_per_side

    # Print statistics
    print(f"\n📊 Importance Map Statistics (train_gear3.py 방식):")
    print(f"\n  Register Patch Detection:")
    print(f"     Highest attention value: {sorted_attn[0]:.6f}")
    print(f"     Register patch location: ({register_row}, {register_col}) in 37×37 grid")
    print(f"     Second highest: {sorted_attn[1]:.6f}")
    print(f"     Gap: {sorted_attn[0] - sorted_attn[1]:.6f}")

    print(f"\n  Processed Importance Map (register removed, percentile normalized):")
    print(f"     Min: {importance_map_2d.min():.6f} (should be ~0.0)")
    print(f"     Max: {importance_map_2d.max():.6f} (should be ~1.0)")
    print(f"     Mean: {mean_val:.6f}")
    print(f"     Std: {importance_map_2d.std():.6f}")

    print(f"\n  FG/BG Split (Mean-based, register patches excluded):")
    print(f"     FG patches (>{mean_val:.4f}): {fg_mask.sum():.0f} ({fg_mask.sum()/fg_mask.size*100:.1f}%)")
    print(f"     BG patches (≤{mean_val:.4f}): {bg_mask.sum():.0f} ({bg_mask.sum()/bg_mask.size*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Visualize Gear3 Importance Map (train_gear3.py 방식)")
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to FlashDepth checkpoint')
    parser.add_argument('--config-path', type=str, default='configs/gear3', help='Config directory')
    parser.add_argument('--output-dir', type=str, default='attn_visualizations', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')

    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    # Load config
    config_path = os.path.join(args.config_path, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = OmegaConf.load(config_path)
    print(f"📄 Config loaded from: {config_path}")

    # Load model
    print(f"🔧 Loading FlashDepth model...")
    # FlashDepth expects model config parameters directly
    model_config = dict(config.model)
    model_config['batch_size'] = config.training.get('batch_size', 1)
    model_config['use_metric_head'] = False  # Don't use GSP head for visualization
    model = FlashDepth(**model_config)

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"✅ Checkpoint loaded: {args.checkpoint}")
    else:
        print(f"⚠️  Checkpoint not found: {args.checkpoint}")
        print(f"    Using randomly initialized model (attention weights still work)")

    model = model.to(device)

    # Enable attention weight storage for last block
    print(f"🔍 Enabling attention weight storage for last block...")
    for i, block in enumerate(model.pretrained.blocks):
        if i == len(model.pretrained.blocks) - 1:
            block.attn.store_attn_weights = True
            print(f"   Block {i} (last): store_attn_weights = True")

    # Load image
    print(f"🖼️  Loading image: {args.image_path}")
    img_tensor, img_original = load_image(args.image_path)

    # Extract attention weights and process to importance map (train_gear3.py 방식)
    print(f"🧠 Processing importance map (register removal + percentile normalization)...")
    importance_map, attn_scores_raw, attn_weights_full = extract_attention_weights(
        model, img_tensor, device, patch_h=37, patch_w=37
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename
    image_name = Path(args.image_path).stem
    output_path = os.path.join(args.output_dir, f'{image_name}_attention.png')

    # Visualize
    print(f"🎨 Creating visualization...")
    visualize_attention(img_original, attn_scores_raw, importance_map, output_path)

    print(f"\n✅ Done!")


if __name__ == '__main__':
    main()
