#!/usr/bin/env python3
"""
Inference script for avante_images using Gear5 model.
Generates depth maps with 70m max depth threshold and 2-98 percentile colormap.
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flashdepth.model import FlashDepth
from flashdepth.gear5_modules import Gear5MetricHead


def load_gear5_model(checkpoint_path, device, config_variant='l', use_mamba=False, cls_layers=[2, 4]):
    """Load Gear5 model with metric head.

    Returns:
        model: FlashDepth model with Gear5MetricHead
        encoder_indices: List of encoder feature indices for CLS extraction
        target_blocks: List of ViT block indices for attention weights
    """

    # Model configuration based on variant
    if config_variant == 'l':
        vit_size = 'vitl'
        model_embed_dim = 1024
    elif config_variant == 's':
        vit_size = 'vits'
        model_embed_dim = 384
    else:
        vit_size = 'vitl'
        model_embed_dim = 1024

    # Model config matching flashdepth-l/config.yaml (same as test_gear5.py)
    model_config = {
        'vit_size': vit_size,
        'patch_size': 14,
        'attn_class': 'MemEffAttention',
        'use_mamba': True,
        'mamba_type': 'add',
        'num_mamba_layers': 4,
        'downsample_mamba': [0.1],
        'mamba_pos_embed': None,
        'mamba_in_dpt_layer': [3],
        'mamba_d_conv': 4,
        'mamba_d_state': 256,
        'use_hydra': False,
        'use_transformer_rnn': False,
        'use_xlstm': False,
        'batch_size': 1,
        'use_metric_head': False,
    }

    model = FlashDepth(**model_config)

    # Add Gear5 metric head
    model.gear5_metric_head = Gear5MetricHead(
        embed_dim=model_embed_dim,
        feature_dim=256,
        hidden_dim=128,
        use_mamba=use_mamba
    )

    # Enable attention weights storage for CLS token extraction
    intermediate_idx = model.intermediate_layer_idx[model.encoder]
    encoder_indices = [layer - 1 for layer in cls_layers]
    target_blocks = [intermediate_idx[idx] for idx in encoder_indices]

    print(f"CLS layer selection: user specified layers {cls_layers}")
    print(f"  → encoder_indices: {encoder_indices}")
    print(f"  → target_blocks: {target_blocks} (actual ViT block indices)")

    for i, block in enumerate(model.pretrained.blocks):
        if i in target_blocks:
            block.attn.store_attn_weights = True
        else:
            block.attn.store_attn_weights = False

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = model.to(device)
    model.eval()

    return model, encoder_indices, target_blocks


def depth_to_colormap(depth, max_depth=70.0, percentile_range=(2, 98)):
    """
    Convert depth map to colormap visualization.
    - Depth > max_depth is treated as invalid (black)
    - Valid region uses 2-98 percentile normalization

    Args:
        depth: [H, W] numpy array in meters
        max_depth: Maximum valid depth (default: 70m)
        percentile_range: tuple of (low, high) percentiles for auto-scaling

    Returns:
        [H, W, 3] BGR image (uint8)
    """
    # Create valid mask (depth <= max_depth and positive)
    valid_mask = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)

    if not valid_mask.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    valid_depth = depth[valid_mask]

    # Use percentile normalization for valid region
    vmin = np.percentile(valid_depth, percentile_range[0])
    vmax = np.percentile(valid_depth, percentile_range[1])

    # Create depth with NaN for invalid pixels
    depth_vis = np.where(valid_mask, depth, np.nan)

    # Normalize to [0, 1]
    depth_normalized = np.clip((depth_vis - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # Apply colormap (plasma_r)
    cmap = matplotlib.colormaps.get_cmap('plasma_r').copy()
    cmap.set_bad(color='black')  # NaN pixels = black
    depth_colored_rgba = cmap(depth_normalized)
    depth_colored = (depth_colored_rgba[:, :, :3] * 255).astype(np.uint8)

    # Convert RGB to BGR for cv2
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

    return depth_colored


def preprocess_image(image_path, target_size=518):
    """Load and preprocess image for model input."""
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size  # (W, H)

    # Resize to target size while maintaining aspect ratio
    w, h = orig_size
    scale = target_size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Make divisible by 14 (ViT patch size)
    new_w = (new_w // 14) * 14
    new_h = (new_h // 14) * 14

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Convert to tensor and normalize
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std

    # HWC -> CHW
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

    return img_tensor, orig_size, (new_w, new_h)


def run_inference_frame_by_frame(model, images_tensor, encoder_indices, target_blocks, device, canonical_fx=500.0, actual_fx=None):
    """
    Run inference frame-by-frame like test_gear5.py.
    This properly handles Mamba temporal state and de-canonicalization.

    Args:
        model: FlashDepth model with Gear5MetricHead
        images_tensor: [B, T, C, H, W] input images
        encoder_indices: List of encoder feature indices for CLS extraction
        target_blocks: List of ViT block indices for attention weights
        device: torch device
        canonical_fx: Canonical focal length (500.0)
        actual_fx: Actual focal length in pixels (for de-canonicalization)

    Returns:
        metric_depth: [B, T, 1, H, W] numpy array in meters
    """
    import torch.nn.functional as F

    B, T, C, H, W = images_tensor.shape

    # Default focal length
    if actual_fx is None:
        actual_fx = 900.0  # Default for avante_images (original size 1600x1100)

    # De-canonicalization ratio (canonical → actual space for inverse depth)
    de_canon_ratio = canonical_fx / actual_fx  # For inverse depth: multiply by this

    # Initialize Mamba sequence for temporal processing
    if hasattr(model, 'mamba'):
        model.mamba.start_new_sequence()

    pred_depths = []

    with torch.no_grad():
        for t in range(T):
            img_t = images_tensor[0, t]  # [C, H, W]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                h, w = img_t.shape[1:]
                patch_h, patch_w = h // model.patch_size, w // model.patch_size

                # Extract features from DINOv2
                encoder_features = model.pretrained.get_intermediate_layers(
                    img_t.unsqueeze(0), model.intermediate_layer_idx[model.encoder]
                )

                # Extract CLS tokens from specified layers
                cls_tokens_list = [
                    encoder_features[i][:, 0]  # CLS token: [1, embed_dim]
                    for i in encoder_indices
                ]
                # Average and reshape for temporal processing: [1, 1, embed_dim]
                cls_tokens_averaged = torch.stack(cls_tokens_list, dim=1).mean(dim=1)  # [1, embed_dim]
                cls_tokens = cls_tokens_averaged.view(1, 1, -1)  # [1, 1, embed_dim]

                # Get attention weights from specified blocks
                attention_weights_list = [
                    model.pretrained.blocks[block_idx].attn.attn_weights
                    for block_idx in target_blocks
                ]

                # Get DPT features (frozen)
                dpt_features = model.depth_head.get_forward_features(
                    encoder_features, patch_h, patch_w
                )
                path_1 = dpt_features[-1]

                # Apply Mamba temporal processing (frozen)
                path_1_temporal = model.dpt_features_to_mamba(
                    input_shape=(1, 1, None, h, w),
                    dpt_features=path_1,
                    in_dpt_layer=0
                )

                # Get relative depth (frozen) - this is in inverse depth space (100/m)
                out = model.depth_head.scratch.output_conv1(path_1_temporal)
                out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                relative_depth = model.depth_head.scratch.output_conv2(out)  # [1, 1, H, W]

                # Get scale/shift from Gear5MetricHead
                gear5_outputs = model.gear5_metric_head(
                    cls_tokens=cls_tokens,
                    attention_weights_list=attention_weights_list,
                    patch_h=patch_h,
                    patch_w=patch_w
                )

                scale = gear5_outputs['scale']  # [1, 1]
                shift = gear5_outputs['shift']  # [1, 1]

                # Apply scale/shift to relative depth (in canonical inverse depth space)
                scale_expanded = scale.view(1, 1, 1, 1)
                shift_expanded = shift.view(1, 1, 1, 1)
                pred_inverse_100_canonical = scale_expanded * relative_depth + shift_expanded  # [1, 1, H, W]

                # De-canonicalization: canonical → actual space (for inverse depth)
                # pred_inverse_actual = pred_inverse_canonical * (canonical_fx / actual_fx)
                pred_inverse_100_actual = pred_inverse_100_canonical * de_canon_ratio

                # Convert inverse depth to metric depth
                # inverse_depth is in 100/m, so metric_depth = 100 / inverse_depth
                pred_depth_metric = 100.0 / (pred_inverse_100_actual + 1e-8)  # [1, 1, H, W] in meters
                pred_depth_metric = pred_depth_metric.clamp(min=0.01, max=200.0)  # Reasonable range

            pred_depths.append(pred_depth_metric.float().cpu())

    # Stack all frames: [T, 1, H, W]
    pred_depths = torch.cat(pred_depths, dim=0)  # [T, 1, H, W]

    return pred_depths.unsqueeze(0).numpy()  # [1, T, 1, H, W]


def main():
    parser = argparse.ArgumentParser(description='Inference on avante_images using Gear5')
    parser.add_argument('--input-dir', type=str, default='/data/datasets/avante_images',
                        help='Input directory containing images')
    parser.add_argument('--output-dir', type=str, default='/app/test_results/avante_depth',
                        help='Output directory for depth maps')
    parser.add_argument('--checkpoint', type=str,
                        default='train_results/results_21/gear_5/large/best.pth',
                        help='Path to Gear5 checkpoint')
    parser.add_argument('--config-variant', type=str, default='l', choices=['l', 's'],
                        help='Model variant: l (large) or s (small)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--max-depth', type=float, default=70.0,
                        help='Maximum valid depth in meters')
    parser.add_argument('--focal-length', type=float, default=900.0,
                        help='Actual focal length in pixels (default: 900 for avante_images)')
    parser.add_argument('--canonical-fx', type=float, default=500.0,
                        help='Canonical focal length')
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for output GIF')
    parser.add_argument('--section', type=str, default=None,
                        help='Frame section to process (e.g., "450,480" for frames 450-480)')
    parser.add_argument('--mamba', action='store_true',
                        help='Use Mamba2 for temporal modeling')
    parser.add_argument('--cls-layers', type=str, default='2,4',
                        help='CLS token extraction layers (comma-separated)')
    args = parser.parse_args()

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parse CLS layers
    cls_layers = [int(x.strip()) for x in args.cls_layers.split(',')]

    # Load model
    print(f"Loading Gear5 model (variant: {args.config_variant})...")
    model, encoder_indices, target_blocks = load_gear5_model(
        args.checkpoint, device, args.config_variant, args.mamba, cls_layers
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of input images
    input_dir = Path(args.input_dir)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    all_image_files = sorted([f for f in input_dir.iterdir()
                              if f.suffix.lower() in image_extensions])

    print(f"Found {len(all_image_files)} images in {input_dir}")

    # Filter by section if specified (e.g., --section 450,480)
    # Images are named like 00450.png (= frame 450)
    if args.section:
        start_frame, end_frame = map(int, args.section.split(','))

        image_files = []
        for f in all_image_files:
            # Extract number from filename (e.g., 00450.png -> 450)
            try:
                file_num = int(f.stem)
                if start_frame <= file_num <= end_frame:
                    image_files.append(f)
            except ValueError:
                continue

        print(f"Section {start_frame}-{end_frame} selected: {len(image_files)} images")
    else:
        image_files = all_image_files

    if len(image_files) == 0:
        print("No images found!")
        return

    # Preprocess all images first
    print("Preprocessing images...")
    preprocessed = []
    orig_sizes = []
    for img_path in tqdm(image_files):
        img_tensor, orig_size, new_size = preprocess_image(str(img_path))
        preprocessed.append(img_tensor)
        orig_sizes.append(orig_size)

    # Stack all images: [N, C, H, W]
    all_images = torch.stack(preprocessed)
    N, C, H, W = all_images.shape

    # Use actual focal length from args (default: 900 for avante_images)
    actual_fx = args.focal_length

    print(f"Using focal length: {actual_fx:.1f} px (canonical: {args.canonical_fx})")
    print(f"De-canonicalization ratio: {args.canonical_fx / actual_fx:.4f}")
    print(f"Max valid depth: {args.max_depth}m")

    # Run inference frame-by-frame (like test_gear5.py)
    # This properly handles Mamba temporal state and de-canonicalization
    print("Running inference (frame-by-frame with temporal processing)...")

    # Add batch dimension: [1, N, C, H, W]
    all_images_batched = all_images.unsqueeze(0).to(device)

    # Run inference
    metric_depth = run_inference_frame_by_frame(
        model, all_images_batched, encoder_indices, target_blocks, device,
        canonical_fx=args.canonical_fx, actual_fx=actual_fx
    )

    # Extract depth results: [N, H, W]
    depth_results = [metric_depth[0, i, 0] for i in range(N)]

    print(f"Inference complete. Depth range: [{min(d.min() for d in depth_results):.2f}, {max(d.max() for d in depth_results):.2f}] meters")

    # Save results
    print("Saving results...")
    output_dir = Path(args.output_dir)

    gif_frames = []  # For GIF (RGB format)

    for i, (img_path, depth) in enumerate(tqdm(zip(image_files, depth_results))):
        # Resize depth to original size
        orig_w, orig_h = orig_sizes[i]
        depth_resized = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Apply colormap with max depth threshold
        depth_colored = depth_to_colormap(depth_resized, max_depth=args.max_depth)

        # Output filename matches input (e.g., 00006.png -> 00006_depth.png)
        output_name = f"{img_path.stem}_depth.png"

        # Save colormap as PNG
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), depth_colored)

        # Convert BGR to RGB for GIF
        depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        gif_frames.append(Image.fromarray(depth_rgb))

    # Save GIF
    if len(gif_frames) > 1:
        gif_path = output_dir / "depth.gif"
        # Calculate duration in ms from fps
        duration = int(1000 / args.fps)
        gif_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"Saved GIF to {gif_path}")

    print(f"\nDone! Results saved to {output_dir}")
    print(f"  - {len(image_files)} depth colormaps (*_depth.png)")
    print(f"  - 1 GIF (depth.gif)")


if __name__ == '__main__':
    main()
