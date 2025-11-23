#!/usr/bin/env python3
"""
Gear5 Importance Map Visualization Script

이 스크립트는 Gear5 방식으로 2-layer attention averaging을 사용하여
Importance map을 시각화합니다.

사용법:
    # Default: configs/flashdepth-l/iter_10001.pth 체크포인트 사용
    python visualize_attention_weights.py \
        --image-path <path_to_image> \
        --output-dir visualizations/gear5_attention

    # Custom checkpoint 사용
    python visualize_attention_weights.py \
        --image-path <path_to_image> \
        --checkpoint configs/flashdepth-l/iter_10001.pth \
        --output-dir visualizations/gear5_attention

결과 (5개 이미지를 원본 해상도로 저장):
    - layer_11_importance.png: Layer 11 단독 importance map
    - layer_23_importance.png: Layer 23 단독 importance map
    - fusion_importance.png: 2-layer fusion importance map
    - fg_mask.png: FG mask (fusion 기준)
    - original.png: 원본 이미지

주의:
    - FlashDepth 체크포인트를 로드해야 train_gear5.py와 동일한 결과를 얻습니다
    - DINOv2 pretrained만 사용하면 다른 결과가 나올 수 있습니다
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flashdepth.model import FlashDepth


def load_image(image_path, target_size=518):
    """Load and preprocess image for DINOv2"""
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)

    # Resize to target size (518×518)
    img_resized = img.resize((target_size, target_size), Image.BILINEAR)

    # Convert to tensor and normalize (ImageNet stats)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_array - mean) / std

    # [H, W, 3] -> [1, 3, H, W]
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()

    return img_tensor, img, original_size


def process_attention_to_importance(attn_weights, patch_h, patch_w):
    """
    Process attention weights to importance map (Gear5 방식)

    Args:
        attn_weights: [B, num_heads, num_tokens, num_tokens]
        patch_h, patch_w: Patch grid dimensions (37, 37)

    Returns:
        importance_map: [B, 1, patch_h, patch_w] in [0, 1]
    """
    # Extract CLS→patch attention
    cls_to_patch_attn = attn_weights[:, :, 0, 1:]  # [B, num_heads, num_patches]

    # Average over heads
    cls_attention = cls_to_patch_attn.mean(dim=1)  # [B, num_patches]

    # Reshape to spatial
    B = cls_attention.shape[0]
    importance_map = cls_attention.view(B, 1, patch_h, patch_w)  # [B, 1, patch_h, patch_w]

    # Register token removal (3×3 local inpainting)
    for b in range(B):
        max_val = importance_map[b, 0].max()
        outlier_mask = (importance_map[b, 0] == max_val)

        # 3×3 averaging kernel
        kernel = torch.ones(1, 1, 3, 3, device=importance_map.device) / 9.0
        smoothed = F.conv2d(importance_map[b:b+1], kernel, padding=1)
        importance_map[b, 0] = torch.where(outlier_mask, smoothed[0, 0], importance_map[b, 0])

    # Percentile normalization (1-99 percentile)
    for b in range(B):
        p1 = torch.quantile(importance_map[b].flatten(), 0.01)
        p99 = torch.quantile(importance_map[b].flatten(), 0.99)
        importance_map[b] = (importance_map[b] - p1) / (p99 - p1 + 1e-8)
        importance_map[b] = torch.clamp(importance_map[b], 0.0, 1.0)

    return importance_map


def extract_2layer_attention(model, img_tensor, device, patch_h=37, patch_w=37):
    """
    Extract attention weights from 2 layers (11, 23) and generate importance maps

    Returns:
        layer_11_importance: [1, 1, patch_h, patch_w]
        layer_23_importance: [1, 1, patch_h, patch_w]
        fusion_importance: [1, 1, patch_h, patch_w]
    """
    model.eval()
    img_tensor = img_tensor.to(device)

    # ViT-L uses blocks [4, 11, 17, 23], target [11, 23]
    target_blocks = [11, 23]
    intermediate_layers = [4, 11, 17, 23]  # ViT-L intermediate_layer_idx

    # Enable attention storage for target blocks
    for i, block in enumerate(model.pretrained.blocks):
        if i in target_blocks:
            block.attn.store_attn_weights = True

    with torch.no_grad():
        # Forward through encoder
        features = model.pretrained.get_intermediate_layers(
            img_tensor,
            n=intermediate_layers,
            return_class_token=True,
            reshape=True
        )

        # Extract attention weights from target blocks
        attention_weights_list = []
        for block_idx in target_blocks:
            block = model.pretrained.blocks[block_idx]

            if not hasattr(block.attn, 'attn_weights') or block.attn.attn_weights is None:
                raise RuntimeError(
                    f"Attention weights not found for block {block_idx}! "
                    f"Make sure store_attn_weights=True is set."
                )

            attention_weights_list.append(block.attn.attn_weights)

        # Process each layer's attention to importance map
        layer_11_attn = attention_weights_list[0]  # Block 11
        layer_23_attn = attention_weights_list[1]  # Block 23

        layer_11_importance = process_attention_to_importance(layer_11_attn, patch_h, patch_w)
        layer_23_importance = process_attention_to_importance(layer_23_attn, patch_h, patch_w)

        # Fusion: simple average (Gear5 방식)
        fusion_importance = (layer_11_importance + layer_23_importance) / 2.0

        return layer_11_importance, layer_23_importance, fusion_importance


def save_importance_map(importance_map, output_path, original_size):
    """
    Save importance map as image (원본 해상도로 upsample)

    Args:
        importance_map: [1, 1, patch_h, patch_w] tensor
        output_path: Path to save
        original_size: (W, H) tuple
    """
    # Upsample to original resolution
    importance_upsampled = F.interpolate(
        importance_map,
        size=(original_size[1], original_size[0]),  # (H, W)
        mode='bilinear',
        align_corners=True
    ).squeeze().cpu().numpy()  # [H, W]

    # Apply colormap
    cmap = plt.get_cmap('jet')
    colored = cmap(importance_upsampled)  # [H, W, 4] RGBA

    # Convert to RGB and save
    img_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_rgb)
    img_pil.save(output_path)

    print(f"✅ Saved: {output_path}")


def save_fg_mask(fusion_importance, img_original, output_path, original_size):
    """
    Save FG mask overlay on original image

    Args:
        fusion_importance: [1, 1, patch_h, patch_w] tensor
        img_original: PIL Image (original resolution)
        output_path: Path to save
        original_size: (W, H) tuple
    """
    # Upsample to original resolution
    importance_upsampled = F.interpolate(
        fusion_importance,
        size=(original_size[1], original_size[0]),  # (H, W)
        mode='bilinear',
        align_corners=True
    ).squeeze().cpu().numpy()  # [H, W]

    # Compute FG mask (mean threshold)
    mean_val = importance_upsampled.mean()
    fg_mask = (importance_upsampled > mean_val).astype(np.float32)

    # Create overlay
    img_array = np.array(img_original).astype(np.float32) / 255.0

    # Red overlay for FG
    fg_overlay = np.zeros_like(img_array)
    fg_overlay[..., 0] = fg_mask  # Red channel

    # Blend
    blended = img_array * 0.5 + fg_overlay * 0.5
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    # Save
    img_pil = Image.fromarray(blended)
    img_pil.save(output_path)

    fg_ratio = fg_mask.sum() / fg_mask.size * 100
    print(f"✅ Saved: {output_path} (FG: {fg_ratio:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Visualize Gear5 2-Layer Importance Maps")
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to FlashDepth checkpoint (default: configs/flashdepth-l/iter_10001.pth)')
    parser.add_argument('--output-dir', type=str, default='attn_visualizations', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')

    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    # Load model (FlashDepth pretrained)
    print(f"🔧 Loading FlashDepth model (ViT-L)...")
    model = FlashDepth(
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        localhub=True,
        use_mamba=False,  # Temporal module not needed for visualization
    )

    # Load FlashDepth checkpoint (default or user-specified)
    if args.checkpoint is None:
        # Default: Use FlashDepth-L checkpoint from configs/flashdepth-l/
        args.checkpoint = 'configs/flashdepth-l/iter_10001.pth'

    if os.path.exists(args.checkpoint):
        print(f"📦 Loading FlashDepth checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        # Extract state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove module. prefix if present (from DDP training)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load state dict (strict=False to allow missing keys like gear5_metric_head)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"✅ FlashDepth checkpoint loaded successfully")
        if missing:
            print(f"    Missing keys: {len(missing)} (expected for visualization)")
        if unexpected:
            print(f"    Unexpected keys: {len(unexpected)}")
    else:
        print(f"⚠️  Checkpoint not found: {args.checkpoint}")
        print(f"    Using DINOv2 pretrained weights only (may produce different results)")

    model = model.to(device)

    # Load image
    print(f"🖼️  Loading image: {args.image_path}")
    img_tensor, img_resized, original_size = load_image(args.image_path)
    print(f"    Original size: {original_size[0]}×{original_size[1]}")
    print(f"    Resized to: 518×518 for processing")

    # Extract 2-layer attention and generate importance maps
    print(f"🧠 Extracting 2-layer attention (blocks 11, 23)...")
    layer_11_importance, layer_23_importance, fusion_importance = extract_2layer_attention(
        model, img_tensor, device, patch_h=37, patch_w=37
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename prefix
    image_name = Path(args.image_path).stem

    # Load original image (full resolution)
    img_original = Image.open(args.image_path).convert('RGB')

    # Save 5 images
    print(f"\n🎨 Saving visualizations (original resolution: {original_size[0]}×{original_size[1]})...")

    # 1. Layer 11 importance
    save_importance_map(
        layer_11_importance,
        os.path.join(args.output_dir, f'{image_name}_layer_11_importance.png'),
        original_size
    )

    # 2. Layer 23 importance
    save_importance_map(
        layer_23_importance,
        os.path.join(args.output_dir, f'{image_name}_layer_23_importance.png'),
        original_size
    )

    # 3. Fusion importance
    save_importance_map(
        fusion_importance,
        os.path.join(args.output_dir, f'{image_name}_fusion_importance.png'),
        original_size
    )

    # 4. FG mask
    save_fg_mask(
        fusion_importance,
        img_original,
        os.path.join(args.output_dir, f'{image_name}_fg_mask.png'),
        original_size
    )

    # 5. Original image
    img_original.save(os.path.join(args.output_dir, f'{image_name}_original.png'))
    print(f"✅ Saved: {os.path.join(args.output_dir, f'{image_name}_original.png')}")

    # Print statistics
    print(f"\n📊 Importance Map Statistics:")

    for name, importance_map in [
        ("Layer 11", layer_11_importance),
        ("Layer 23", layer_23_importance),
        ("Fusion", fusion_importance)
    ]:
        imp_np = importance_map[0, 0].cpu().numpy()
        print(f"\n  {name}:")
        print(f"     Min: {imp_np.min():.4f}")
        print(f"     Max: {imp_np.max():.4f}")
        print(f"     Mean: {imp_np.mean():.4f}")
        print(f"     Std: {imp_np.std():.4f}")

    print(f"\n✅ Done! All images saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
