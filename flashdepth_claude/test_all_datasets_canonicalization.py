#!/usr/bin/env python3
"""
Test canonicalization for all available datasets and generate Canonicalization.md
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataloaders.base_dataset_pairs import BaseDatasetPairs

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Constants
CANONICAL_FOCAL_LENGTH = 500.0

def get_dataset_info(dataset_name, root_dir, split='val'):
    """Get first frame info from dataset"""
    try:
        dataset = BaseDatasetPairs.create(dataset_name, root_dir=root_dir, split=split, load_cache=None)

        if len(dataset.pairs) == 0:
            return None

        # Get first sequence
        if split == 'val':
            # Validation: pairs is list of sequences (each sequence is list of frames)
            if len(dataset.pairs[0]) == 0:
                return None
            first_pair = dataset.pairs[0][0]
        else:
            # Training: pairs is flat list of frames
            first_pair = dataset.pairs[0]

        # Load image to get original resolution
        from PIL import Image
        img = Image.open(first_pair['image'])
        original_w, original_h = img.size  # PIL: (W, H)

        # Get focal length
        if hasattr(dataset, 'get_focal_length'):
            fx_actual = dataset.get_focal_length(first_pair, (original_h, original_w))
        else:
            # Fallback: estimate from resolution
            fx_actual = original_w * 0.9

        # Get target resolution from reshape_list
        target_resolution = dataset.reshape_list.get('resolution', None)
        resize_factor = dataset.reshape_list.get('resize_factor', 1.0)

        return {
            'original_w': original_w,
            'original_h': original_h,
            'fx_actual': fx_actual,
            'target_resolution': target_resolution,
            'resize_factor': resize_factor,
            'first_frame_path': first_pair['image']
        }
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None


def calculate_canonicalization(original_w, original_h, fx_actual,
                                target_resolution, resize_factor=1.0):
    """Calculate canonicalization ratios"""

    # Step 1: Pre-resize
    pre_h = int(original_h * resize_factor)
    pre_w = int(original_w * resize_factor)

    # Step 2: Unpack target_resolution (FIXED - as (W, H))
    if target_resolution is None:
        return None

    target_w, target_h = target_resolution  # FIXED: Unpack as (W, H)

    # Step 3: Small resize ratio (FIXED)
    small_resize_ratio_code = max(target_w / pre_w, target_h / pre_h)  # W→W, H→H

    # Step 4: fx_ratio
    fx_ratio = CANONICAL_FOCAL_LENGTH / fx_actual

    # Step 5: total_resize_ratio
    total_resize_ratio = resize_factor * small_resize_ratio_code

    # Step 6: depth_correction_ratio
    inverse_depth_correction = total_resize_ratio / fx_ratio  # For inverse depth
    normal_depth_correction = fx_ratio / total_resize_ratio  # For normal depth

    return {
        'pre_h': pre_h,
        'pre_w': pre_w,
        'target_h': target_h,
        'target_w': target_w,
        'small_resize_ratio': small_resize_ratio_code,
        'fx_ratio': fx_ratio,
        'total_resize_ratio': total_resize_ratio,
        'inverse_depth_correction': inverse_depth_correction,
        'normal_depth_correction': normal_depth_correction
    }


def main():
    data_root = "/home/cvlab/hsy/Datasets"

    # List of datasets to test (validation split)
    val_datasets = [
        'sintel',
        'waymo_seg',
        'eth3d',
        'urbansyn',
        'unreal4k',
    ]

    # List of training datasets
    train_datasets = [
        'mvs-synth',
        'spring',
        'pointodyssey',
        'dynamicreplica',
        'tartanair',
    ]

    print("="*80)
    print("Testing Canonicalization for All Datasets")
    print("="*80)

    # Collect info
    dataset_info = {}

    # Test validation datasets (base resolution)
    print("\nLoading validation datasets (base resolution)...")
    # Override resolutions for validation datasets (from combined_dataset.py)
    resolution_base_val_map = {
        'eth3d': (784, 518),
        'waymo_seg': (784, 518),
        'sintel': (1022, 434),
        'urbansyn': (1036, 518),
        'unreal4k': (924, 518),
    }

    for dataset_name in val_datasets:
        print(f"  Loading {dataset_name}...")
        info = get_dataset_info(dataset_name, data_root, split='val')
        if info:
            # Override target resolution with correct value from combined_dataset.py
            if dataset_name in resolution_base_val_map:
                info['target_resolution'] = resolution_base_val_map[dataset_name]
            dataset_info[f"{dataset_name} (val-base)"] = info

    # Test validation datasets (2k resolution) - need to manually set resolution
    print("\nLoading validation datasets (2K resolution)...")
    resolution_2k_map = {
        'eth3d': (1918, 1274),
        'waymo_seg': (1918, 1274),
        'sintel': (1022, 434),
        'urbansyn': (2044, 1022),
        'unreal4k': (2044, 1148)
    }

    for dataset_name in ['sintel', 'waymo_seg', 'eth3d', 'urbansyn', 'unreal4k']:
        if dataset_name not in val_datasets:
            continue
        print(f"  Loading {dataset_name} (2K)...")
        info = get_dataset_info(dataset_name, data_root, split='val')
        if info:
            # Override target resolution for 2K
            info['target_resolution'] = resolution_2k_map.get(dataset_name, info['target_resolution'])
            dataset_info[f"{dataset_name} (val-2k)"] = info

    # Test training datasets (base resolution)
    print("\nLoading training datasets (base resolution)...")
    for dataset_name in train_datasets:
        print(f"  Loading {dataset_name}...")
        info = get_dataset_info(dataset_name, data_root, split='train')
        if info:
            # Training uses 518×518
            info['target_resolution'] = (518, 518)
            dataset_info[f"{dataset_name} (train-base)"] = info

    # Test training datasets (2K resolution)
    print("\nLoading training datasets (2K resolution)...")
    for dataset_name in train_datasets:
        print(f"  Loading {dataset_name} (2K)...")
        info = get_dataset_info(dataset_name, data_root, split='train')
        if info:
            # Training 2K uses 1918×1078
            info['target_resolution'] = (1918, 1078)
            dataset_info[f"{dataset_name} (train-2k)"] = info

    # Generate Markdown file
    print("\n" + "="*80)
    print("Generating Canonicalization.md")
    print("="*80)

    md_lines = []
    md_lines.append("# Canonicalization Test Results")
    md_lines.append("")
    md_lines.append("Test date: 2025-11-13")
    md_lines.append("")
    md_lines.append("## Dataset Overview")
    md_lines.append("")
    md_lines.append("| Dataset | Original Resolution (W×H) | Original Focal Length (fx) |")
    md_lines.append("|---------|---------------------------|----------------------------|")

    # Sort by: base-train, base-val, 2k-train, 2k-val
    def sort_key(item):
        dataset_name = item[0]
        # Extract resolution type and split type
        if 'train-base' in dataset_name:
            return (0, dataset_name)  # base-train first
        elif 'val-base' in dataset_name:
            return (1, dataset_name)  # base-val second
        elif 'train-2k' in dataset_name:
            return (2, dataset_name)  # 2k-train third
        elif 'val-2k' in dataset_name:
            return (3, dataset_name)  # 2k-val fourth
        else:
            return (4, dataset_name)  # others last

    sorted_datasets = sorted(dataset_info.items(), key=sort_key)

    for dataset_name, info in sorted_datasets:
        orig_w = info['original_w']
        orig_h = info['original_h']
        fx = info['fx_actual']
        md_lines.append(f"| {dataset_name} | {orig_w}×{orig_h} | {fx:.2f} |")

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Canonicalization Results by Dataset")
    md_lines.append("")
    md_lines.append("### Formulas")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append("pre_h = original_h × resize_factor")
    md_lines.append("pre_w = original_w × resize_factor")
    md_lines.append("target_w, target_h = target_resolution  # FIXED: target_resolution is (W, H)")
    md_lines.append("small_resize_ratio = max(target_w / pre_w, target_h / pre_h)  # W→W, H→H")
    md_lines.append("fx_ratio = 500.0 / fx_actual")
    md_lines.append("total_resize_ratio = resize_factor × small_resize_ratio")
    md_lines.append("inverse_depth_correction_ratio = total_resize_ratio / fx_ratio")
    md_lines.append("normal_depth_correction_ratio = fx_ratio / total_resize_ratio")
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # Generate results for each dataset with section headers
    current_section = None
    for dataset_name, info in sorted_datasets:
        # Determine section
        if 'train-base' in dataset_name:
            section = 'base-train'
            section_title = '## Base Resolution - Training Datasets'
        elif 'val-base' in dataset_name:
            section = 'base-val'
            section_title = '## Base Resolution - Validation Datasets'
        elif 'train-2k' in dataset_name:
            section = '2k-train'
            section_title = '## 2K Resolution - Training Datasets'
        elif 'val-2k' in dataset_name:
            section = '2k-val'
            section_title = '## 2K Resolution - Validation Datasets'
        else:
            section = 'other'
            section_title = '## Other'

        # Add section header if new section
        if section != current_section:
            if current_section is not None:
                md_lines.append("")
            md_lines.append(section_title)
            md_lines.append("")
            current_section = section
        md_lines.append(f"### {dataset_name}")
        md_lines.append("")
        md_lines.append("**Input:**")
        md_lines.append("")
        md_lines.append(f"- Original resolution: {info['original_w']}×{info['original_h']} (W×H)")
        md_lines.append(f"- Original focal length: fx_actual = {info['fx_actual']:.2f}")
        md_lines.append(f"- Target resolution: {info['target_resolution']}")
        md_lines.append(f"- Resize factor: {info['resize_factor']}")
        md_lines.append(f"- First frame: `{info['first_frame_path']}`")
        md_lines.append("")

        # Calculate canonicalization
        calc = calculate_canonicalization(
            info['original_w'], info['original_h'], info['fx_actual'],
            info['target_resolution'], info['resize_factor']
        )

        if calc:
            md_lines.append("**Calculation:**")
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(f"pre_h = {info['original_h']} × {info['resize_factor']} = {calc['pre_h']}")
            md_lines.append(f"pre_w = {info['original_w']} × {info['resize_factor']} = {calc['pre_w']}")
            md_lines.append(f"target_w, target_h = {info['target_resolution']}")
            md_lines.append(f"target_w = {calc['target_w']}")
            md_lines.append(f"target_h = {calc['target_h']}")
            md_lines.append(f"small_resize_ratio = max({calc['target_w']}/{calc['pre_w']}, {calc['target_h']}/{calc['pre_h']})")
            md_lines.append(f"                   = max({calc['target_w']/calc['pre_w']:.6f}, {calc['target_h']/calc['pre_h']:.6f})")
            md_lines.append(f"                   = {calc['small_resize_ratio']:.6f}")
            md_lines.append("```")
            md_lines.append("")
            md_lines.append("**Results:**")
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(f"fx_ratio = 500.0 / {info['fx_actual']:.2f} = {calc['fx_ratio']:.6f}")
            md_lines.append(f"total_resize_ratio = {info['resize_factor']} × {calc['small_resize_ratio']:.6f} = {calc['total_resize_ratio']:.6f}")
            md_lines.append(f"inverse_depth_correction_ratio = {calc['total_resize_ratio']:.6f} / {calc['fx_ratio']:.6f} = {calc['inverse_depth_correction']:.6f}")
            md_lines.append(f"normal_depth_correction_ratio = {calc['fx_ratio']:.6f} / {calc['total_resize_ratio']:.6f} = {calc['normal_depth_correction']:.6f}")
            md_lines.append("```")
            md_lines.append("")

            # Add interpretation
            expected_inverse_correction = 1.0 / calc['fx_ratio']
            error = abs(calc['inverse_depth_correction'] - expected_inverse_correction)

            md_lines.append("**Interpretation:**")
            md_lines.append("")
            md_lines.append(f"- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = {expected_inverse_correction:.6f}")
            md_lines.append(f"- Actual inverse_depth_correction: {calc['inverse_depth_correction']:.6f}")
            md_lines.append(f"- Error: {error:.6f}")

            if error < 0.01:
                md_lines.append(f"- ✅ **Close to expected value** (resize is minimal)")
            elif calc['total_resize_ratio'] < 0.6:
                md_lines.append(f"- ⚠️ **Significant downsampling** (total_resize_ratio = {calc['total_resize_ratio']:.3f})")
            elif calc['total_resize_ratio'] > 1.5:
                md_lines.append(f"- ⚠️ **Significant upsampling** (total_resize_ratio = {calc['total_resize_ratio']:.3f})")
            else:
                md_lines.append(f"- ⚠️ **Error is significant** - possible bug in calculation")

            md_lines.append("")
        else:
            md_lines.append("**Error:** Could not calculate canonicalization")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

    # Write to file
    output_path = Path(__file__).parent / "Canonicalization.md"
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"\n✅ Generated: {output_path}")
    print(f"   Total datasets tested: {len(dataset_info)}")


if __name__ == "__main__":
    main()
