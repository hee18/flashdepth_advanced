#!/usr/bin/env python3
"""Debug script to check VKITTI segmentation loading"""
import sys
sys.path.insert(0, '.')

from dataloaders.comparison_dataset import ComparisonDataset, comparison_collate_fn
import torch

print("="*80)
print("Testing VKITTI Segmentation Loading")
print("="*80)

# Create dataset with objwise enabled
dataset = ComparisonDataset(
    dataset_name='vkitti',
    data_root='/home/cvlab/hsy/Datasets',
    split='test',
    video_length=50,
    objwise_enabled=True,
    only_clone=True
)

print(f"\nDataset created: {len(dataset)} sequences")
print(f"objwise_enabled: {dataset.objwise_enabled}")

# Load first sequence
if len(dataset) > 0:
    print("\nLoading first sequence...")
    batch = dataset[0]

    print(f"\nBatch keys: {batch.keys()}")
    print(f"Images shape: {batch['images'].shape}")
    print(f"Depths shape: {batch['depths'].shape}")

    if 'segmentations' in batch:
        print(f"✓ Segmentations FOUND!")
        print(f"  Shape: {batch['segmentations'].shape}")
        print(f"  Unique classes: {torch.unique(batch['segmentations'])}")
    else:
        print(f"✗ Segmentations NOT found in batch!")
        print(f"  This is the problem - segmentations should be loaded")
else:
    print("No sequences found!")

print("\n" + "="*80)
