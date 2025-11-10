"""
Multi-Layer Importance Map & FG Mask Visualization Test
다층 레이어 Importance Map & FG Mask 시각화 테스트

[한글 설명]
이 스크립트는 ViT의 서로 다른 레이어 조합에서 생성된 importance map과 FG mask를 시각화합니다.
각 시퀀스마다 가능한 15가지 레이어 조합(단일, 쌍, 3개, 전체)을 생성하고 5×6 그리드로 표시합니다.

주요 기능:
- 균등 가중치 퓨전: 4개 레이어 → 25%씩, 3개 → 33.3%씩, 2개 → 50%씩, 1개 → 100%
- 자동 이미지 리사이징: GPU OOM 방지를 위해 큰 이미지는 자동으로 518 이하로 축소
- 7개 데이터셋에서 각 2개 시퀀스 샘플링 (총 14개)
- 출력: 조합 분석 그리드 + 입력 이미지 (별도 저장)

[English Description]
This script visualizes importance maps and FG masks from different ViT layer combinations.
For each sequence, it generates all 15 possible layer combinations and displays them in a 5×6 grid.

Key Features:
- Uniform weight fusion: 4 layers → 25% each, 3 → 33.3%, 2 → 50%, 1 → 100%
- Auto image resizing: Large images automatically scaled to ≤518 to prevent GPU OOM
- Samples 2 sequences from each of 7 datasets (14 total)
- Output: Combination analysis grids + input images (saved separately)

Usage:
    python test_multilayer_visualization.py --gpu 0 --data-root /home/cvlab/hsy/Datasets
    python test_multilayer_visualization.py --gpu 1 --output-dir test_results/multilayer_viz
    python test_multilayer_visualization.py --gpu 2 --resolution 518 --seed 42
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random
import logging
from tqdm import tqdm

# FlashDepth imports
from flashdepth.model import FlashDepth
from dataloaders.combined_dataset import CombinedDataset


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_attention_to_importance(attention_weights, patch_h, patch_w, remove_outliers=True):
    """
    Convert raw attention weights to importance map.

    Steps:
    1. Extract CLS→patch attention
    2. Average over heads
    3. Remove register token (highest attention patch)
    4. Percentile normalization (1-99 percentile) to [0, 1]

    Args:
        attention_weights: [B, num_heads, num_patches+1, num_patches+1]
        patch_h, patch_w: Spatial dimensions
        remove_outliers: Whether to remove register token (default: True)

    Returns:
        importance_map: [B, 1, patch_h, patch_w] in range [0, 1]
    """
    B = attention_weights.shape[0]

    # Extract CLS→patch attention
    cls_to_patches = attention_weights[:, :, 0, 1:]  # [B, num_heads, num_patches]

    # Average over heads
    attn_scores = cls_to_patches.mean(dim=1)  # [B, num_patches]

    # Reshape to spatial
    attn_map = attn_scores.reshape(B, 1, patch_h, patch_w)  # [B, 1, patch_h, patch_w]

    if remove_outliers:
        # Remove register token (single highest attention patch)
        for b in range(B):
            attn_2d = attn_map[b, 0]  # [patch_h, patch_w]

            # Find the patch with maximum attention (register token)
            max_val = attn_2d.max()
            outlier_mask = (attn_2d == max_val)

            # Inpaint with local average (3×3 box filter)
            kernel = torch.ones(1, 1, 3, 3, device=attn_map.device) / 9
            attn_smoothed = F.conv2d(
                attn_map[b:b+1], kernel, padding=1
            )
            attn_map[b, 0] = torch.where(
                outlier_mask,
                attn_smoothed[0, 0],
                attn_map[b, 0]
            )

    # Percentile-based normalization to [0, 1] (1-99 percentile)
    for b in range(B):
        attn_flat = attn_map[b].flatten()
        attn_p1 = torch.quantile(attn_flat, 0.01)
        attn_p99 = torch.quantile(attn_flat, 0.99)

        # Normalize to [0, 1] and clip
        attn_map[b] = (attn_map[b] - attn_p1) / (attn_p99 - attn_p1 + 1e-8)
        attn_map[b] = torch.clamp(attn_map[b], 0.0, 1.0)

    return attn_map


def compute_importance_uniform(layer_indices, attention_weights_all, patch_h, patch_w):
    """
    Compute importance map with uniform weights across selected layers.

    Args:
        layer_indices: List of layer numbers (1-indexed), e.g. [1, 3]
        attention_weights_all: List of 4 attention weight tensors [attn_layer1, ..., attn_layer4]
        patch_h, patch_w: Spatial dimensions

    Returns:
        importance_map: [B, 1, patch_h, patch_w]
        weights: numpy array of uniform weights
    """
    # Select attention weights for specified layers
    selected_attns = [attention_weights_all[i-1] for i in layer_indices]

    # Process each layer to get importance map
    importance_maps = []
    for attn in selected_attns:
        imp = process_attention_to_importance(attn, patch_h, patch_w)
        importance_maps.append(imp.squeeze(1))  # [B, H, W]

    # Stack: [B, num_layers, H, W]
    importance_stack = torch.stack(importance_maps, dim=1)

    # Uniform weights
    num_layers = len(layer_indices)
    weights = torch.ones(num_layers, device=importance_stack.device) / num_layers

    # Weighted fusion
    importance_fused = (importance_stack * weights.view(1, -1, 1, 1)).sum(dim=1).unsqueeze(1)

    return importance_fused, weights.cpu().numpy()


def compute_fg_mask(importance_map):
    """
    Compute FG mask using mean threshold (same as train_gear3_upgrade).

    Args:
        importance_map: [B, 1, patch_h, patch_w]

    Returns:
        fg_mask: [B, 1, patch_h, patch_w]
    """
    # Flatten spatial dimensions
    importance_flat = importance_map.flatten(2).squeeze(1)  # [B, num_patches]

    # Mean threshold
    threshold = importance_flat.mean(dim=1, keepdim=True)

    # Binary mask
    fg_mask = (importance_flat > threshold).float().reshape(importance_map.shape)

    return fg_mask


def save_combination_visualization(img, results, combo_list, output_path):
    """
    Save visualization with 2 rows × 8 columns layout.
    Each combination uses 2 columns: left=importance map, right=FG mask overlay.

    Row 0: Input (2 cols) + Layer 2 (2 cols) + Layer 3 (2 cols) + Layer 4 (2 cols)
    Row 1: Layer 2+3 (2 cols) + Layer 2+4 (2 cols) + Layer 3+4 (2 cols) + Layer 2+3+4 (2 cols)

    Args:
        img: [3, H, W] tensor
        results: Dict of {combo_tuple: {'importance': ..., 'fg_mask': ..., 'weights': ...}}
        combo_list: List of 7 layer combinations [[2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]]
        output_path: Path to save
    """
    # Create 2 rows × 8 columns (16 cells total)
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))

    # Prepare input image
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_np = np.clip(img_np, 0, 1)
    H, W = img_np.shape[:2]

    # Row 0, Cols 0-1: Input image (both cells show same image)
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Input Image', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_np)
    axes[0, 1].set_title('Input Image', fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')

    # Layout mapping for 7 combinations (each uses 2 columns)
    # Row 0: cols 2-3, 4-5, 6-7 for [2], [3], [4]
    # Row 1: cols 0-1, 2-3, 4-5, 6-7 for [2,3], [2,4], [3,4], [2,3,4]
    layout = [
        (0, 2), (0, 4), (0, 6),  # [2], [3], [4]
        (1, 0), (1, 2), (1, 4), (1, 6)  # [2,3], [2,4], [3,4], [2,3,4]
    ]

    for idx, combo in enumerate(combo_list):
        row, col_base = layout[idx]

        combo_key = tuple(combo)
        imp_map = results[combo_key]['importance'][0, 0].cpu().numpy()
        fg_mask = results[combo_key]['fg_mask']
        weights = results[combo_key]['weights']

        # Left cell: Importance map
        ax_imp = axes[row, col_base]
        ax_imp.imshow(imp_map, cmap='jet', vmin=0, vmax=1)

        # Title with layer combo and weights
        combo_str = ','.join([str(x) for x in combo])
        weights_str = ':'.join([f'{w:.3f}' for w in weights])
        ax_imp.set_title(f'Layer [{combo_str}]\nWeights: {weights_str}',
                        fontsize=9, fontweight='bold')
        ax_imp.axis('off')

        # Right cell: FG mask overlay
        ax_fg = axes[row, col_base + 1]
        ax_fg.imshow(img_np)

        # Resize FG mask to image resolution
        fg_mask_resized = F.interpolate(
            fg_mask, size=(H, W), mode='bilinear', align_corners=True
        )
        fg_np = fg_mask_resized[0, 0].cpu().numpy()

        # Create red overlay
        overlay = np.zeros((H, W, 3))
        overlay[..., 0] = fg_np  # Red channel
        ax_fg.imshow(overlay, alpha=0.5)

        fg_ratio = fg_np.mean() * 100
        ax_fg.set_title(f'FG Mask ({fg_ratio:.1f}%)',
                       fontsize=9, fontweight='bold')
        ax_fg.axis('off')

    plt.suptitle(f'Multi-Layer Importance & FG Mask Analysis (Layer 1 Excluded)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved combination visualization: {output_path}")


def setup_model(checkpoint_path, device='cuda'):
    """
    Setup FlashDepth-L model from checkpoint.

    Args:
        checkpoint_path: Path to FlashDepth checkpoint file
        device: Device to load model on

    Returns:
        model: FlashDepth model in eval mode
    """
    logger.info("Setting up FlashDepth-L model...")

    # Create FlashDepth-L model
    model = FlashDepth(
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_mamba=False,  # Don't need Mamba for attention extraction
        batch_size=1
    )

    # Load checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract model state dict (handle different checkpoint formats)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        logger.info("✓ Loaded checkpoint successfully")
    else:
        logger.warning(f"Checkpoint not found at: {checkpoint_path}")
        logger.warning("Using random initialization (for testing only!)")

    # Enable attention weight storage for target blocks
    # ViT-L: blocks [4, 11, 17, 23] - same as DPT intermediate layers (model.intermediate_layer_idx)
    target_blocks = [4, 11, 17, 23]
    for i in target_blocks:
        model.pretrained.blocks[i].attn.store_attn_weights = True
    logger.info(f"Enabled attention storage for blocks: {target_blocks}")

    model = model.to(device)
    model.eval()

    return model


def sample_sequences(val_dataset, datasets, num_per_dataset=2, seed=23):
    """
    Randomly sample sequences from each dataset.

    Args:
        val_dataset: CombinedDataset instance
        datasets: List of dataset names
        num_per_dataset: Number of sequences to sample per dataset
        seed: Random seed for reproducibility

    Returns:
        List of (dataset_name, seq_idx) tuples
    """
    random.seed(seed)
    np.random.seed(seed)

    sampled_sequences = []

    for dataset_name in datasets:
        if dataset_name not in val_dataset.pairslist:
            logger.warning(f"Dataset {dataset_name} not found in pairslist, skipping")
            continue

        all_pairs = val_dataset.pairslist[dataset_name]
        num_samples = min(num_per_dataset, len(all_pairs))

        if num_samples == 0:
            logger.warning(f"Dataset {dataset_name} has no pairs, skipping")
            continue

        # Randomly sample indices
        sampled_indices = random.sample(range(len(all_pairs)), num_samples)

        for idx in sampled_indices:
            sampled_sequences.append((dataset_name, idx))

        logger.info(f"{dataset_name}: sampled {num_samples} sequences (indices: {sampled_indices})")

    return sampled_sequences


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Layer Importance Visualization (다층 레이어 Importance 시각화)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제 (Examples):
  # GPU 0번 사용, 기본 설정
  python test_multilayer_visualization.py --gpu 0

  # GPU 1번 사용, 커스텀 출력 디렉토리
  python test_multilayer_visualization.py --gpu 1 --output-dir my_results

  # GPU 2번 사용, 다른 데이터 경로
  python test_multilayer_visualization.py --gpu 2 --data-root /path/to/data

  # GPU 3번 사용, 전체 옵션 지정
  python test_multilayer_visualization.py --gpu 3 --resolution 518 --seed 123
        """
    )

    parser.add_argument('--gpu', type=int, default=0,
                       help='사용할 GPU 번호 (0, 1, 2, 3 등) | GPU device ID to use')
    parser.add_argument('--data-root', type=str, default='/home/cvlab/hsy/Datasets',
                       help='데이터셋 루트 디렉토리 | Root directory for datasets')
    parser.add_argument('--output-dir', type=str, default='test_results/multilayer_viz',
                       help='시각화 결과 저장 디렉토리 | Output directory for visualizations')
    parser.add_argument('--resolution', type=int, default=518,
                       help='이미지 해상도 (최대 크기, 기본값: 518) | Max image resolution (default: 518)')
    parser.add_argument('--seed', type=int, default=23,
                       help='재현성을 위한 랜덤 시드 | Random seed for reproducibility')
    parser.add_argument('--checkpoint', type=str, default='configs/flashdepth-l/iter_10001.pth',
                       help='FlashDepth 체크포인트 경로 | Path to FlashDepth checkpoint')

    args = parser.parse_args()

    # Setup
    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'combination_analysis').mkdir(exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using GPU: {args.gpu}")

    # Setup model
    model = setup_model(checkpoint_path=args.checkpoint, device=device)
    patch_size = model.patch_size  # 14 for ViT-L

    # Setup dataset
    logger.info("Setting up dataset...")
    datasets = ['dynamicreplica', 'sintel', 'spring', 'waymo_seg',
                'mvs-synth', 'tartanair', 'pointodyssey']

    val_dataset = CombinedDataset(
        root_dir=args.data_root,
        enable_dataset_flags=datasets,
        resolution='base',  # 518x518
        split='val',
        video_length=1,  # Only need first frame
        color_aug=False
    )

    # Sample sequences
    sampled_sequences = sample_sequences(val_dataset, datasets, num_per_dataset=2, seed=args.seed)
    logger.info(f"Total sequences to process: {len(sampled_sequences)}")

    # Define 7 layer combinations (excluding Layer 1)
    # Row 1: Layer 2, 3, 4 (single layers)
    # Row 2: Layer 2+3, 2+4, 3+4, 2+3+4 (combinations)
    combinations = [
        # Single (3) - excluding Layer 1
        [2], [3], [4],
        # Pairs (3) - excluding combinations with Layer 1
        [2,3], [2,4], [3,4],
        # Triple (1) - excluding Layer 1
        [2,3,4]
    ]

    # Process each sequence
    for dataset_name, seq_idx in tqdm(sampled_sequences, desc="Processing sequences"):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {dataset_name} sequence {seq_idx}")
        logger.info(f"{'='*60}")

        # Load sequence data
        data = val_dataset[val_dataset.pairs.index((dataset_name, seq_idx))]
        if data is None:
            logger.warning(f"Failed to load data for {dataset_name} seq {seq_idx}, skipping")
            continue

        images, gt_depth, focal_lengths, dataset_idx = data

        # Use first frame only
        img = images[0]  # [3, H, W]

        # Resize if too large to avoid OOM
        max_dim = args.resolution
        _, H_orig, W_orig = img.shape
        if max(H_orig, W_orig) > max_dim:
            scale = max_dim / max(H_orig, W_orig)
            new_H = int(H_orig * scale)
            new_W = int(W_orig * scale)
            # Make dimensions divisible by patch_size (14)
            new_H = (new_H // patch_size) * patch_size
            new_W = (new_W // patch_size) * patch_size
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False
            ).squeeze(0)
            logger.info(f"Resized from {H_orig}×{W_orig} to {new_H}×{new_W}")

        img = img.to(device)
        H, W = img.shape[1:]
        patch_h = H // patch_size
        patch_w = W // patch_size

        logger.info(f"Image shape: {img.shape}, Patch grid: {patch_h}×{patch_w}")

        # Forward pass to collect attention weights
        with torch.no_grad():
            # Extract features (includes attention storage)
            encoder_features = model.pretrained.get_intermediate_layers(
                img.unsqueeze(0),
                model.intermediate_layer_idx[model.encoder]  # [4, 11, 17, 23] for ViT-L
            )

            # Collect attention weights from target blocks [4, 11, 17, 23] - DPT intermediate layers
            target_blocks = [4, 11, 17, 23]
            attention_weights_all = [
                model.pretrained.blocks[i].attn.attn_weights.detach()
                for i in target_blocks
            ]

        logger.info(f"Collected {len(attention_weights_all)} attention weight tensors")

        # Compute importance maps for all 15 combinations
        results = {}
        for combo in tqdm(combinations, desc="Computing combinations", leave=False):
            imp_map, weights = compute_importance_uniform(
                combo, attention_weights_all, patch_h, patch_w
            )
            fg_mask = compute_fg_mask(imp_map)

            results[tuple(combo)] = {
                'importance': imp_map,
                'fg_mask': fg_mask,
                'weights': weights
            }

        logger.info(f"Computed importance maps for {len(results)} combinations")

        # Save visualizations
        output_name = f"{dataset_name}_seq_{seq_idx:02d}"

        # Save combination analysis
        save_combination_visualization(
            img,
            results,
            combinations,
            output_dir / 'combination_analysis' / f"{output_name}_all_combos.png"
        )

        logger.info(f"✓ Completed: {dataset_name} seq {seq_idx}")

        # Clear GPU cache to avoid OOM on next sequence
        del img, images, gt_depth, encoder_features, attention_weights_all, results
        torch.cuda.empty_cache()

    logger.info(f"\n{'='*60}")
    logger.info("All sequences processed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Combination analysis: {output_dir / 'combination_analysis'}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
