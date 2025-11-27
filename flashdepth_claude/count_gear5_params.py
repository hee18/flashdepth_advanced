#!/usr/bin/env python3
"""
Count exact parameters in Gear5 modules with Mamba2.
"""
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from flashdepth.gear5_modules import Gear5MetricHead, TemporalScalePredictor, ImportanceMapGenerator


def count_parameters(module, name="Module"):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"\n{name}:")
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,}")
    return total, trainable


def detailed_count():
    """Count parameters in each component."""
    print("=" * 60)
    print("Gear5 Parameter Count (Mamba2 configuration)")
    print("=" * 60)

    # 1. TemporalScalePredictor with Mamba2
    print("\n1. TemporalScalePredictor (use_mamba=True)")
    tsp = TemporalScalePredictor(
        embed_dim=1024,
        feature_dim=256,
        hidden_dim=128,
        num_layers=1,
        use_mamba=True
    )
    tsp_total, tsp_trainable = count_parameters(tsp, "TemporalScalePredictor")

    # Break down components
    print("\n  Component breakdown:")
    feature_net_params = sum(p.numel() for p in tsp.feature_net.parameters())
    print(f"    - feature_net (1024→256): {feature_net_params:,}")

    if hasattr(tsp, 'temporal_mamba'):
        mamba_params = sum(p.numel() for p in tsp.temporal_mamba.parameters())
        print(f"    - temporal_mamba:         {mamba_params:,}")

        mamba_proj_params = sum(p.numel() for p in tsp.mamba_proj.parameters())
        print(f"    - mamba_proj (256→128):   {mamba_proj_params:,}")

    scale_head_params = sum(p.numel() for p in tsp.scale_head.parameters())
    shift_head_params = sum(p.numel() for p in tsp.shift_head.parameters())
    print(f"    - scale_head (128→1):     {scale_head_params:,}")
    print(f"    - shift_head (128→1):     {shift_head_params:,}")

    # 2. ImportanceMapGenerator
    print("\n2. ImportanceMapGenerator")
    img = ImportanceMapGenerator(num_layers=2)
    img_total, img_trainable = count_parameters(img, "ImportanceMapGenerator")

    # 3. Full Gear5MetricHead
    print("\n3. Gear5MetricHead (Complete)")
    gear5 = Gear5MetricHead(
        embed_dim=1024,
        feature_dim=256,
        hidden_dim=128,
        use_mamba=True
    )
    gear5_total, gear5_trainable = count_parameters(gear5, "Gear5MetricHead")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Gear5MetricHead (Mamba2): {gear5_trainable:,} trainable parameters")
    print(f"\nFor paper: ~{gear5_trainable/1000:.0f}K parameters")

    # Calculate percentage of total FlashDepth model
    # FlashDepth total: DINOv2 (300M) + DPT (15M) + Mamba (4.3M) + Output (0.3M) = 319.6M
    flashdepth_total = 319_600_000
    percentage = (gear5_trainable / flashdepth_total) * 100

    print(f"\nFlashDepth baseline:    {flashdepth_total/1e6:.1f}M parameters")
    print(f"Gear5 addition:         {gear5_trainable/1e6:.2f}M parameters")
    print(f"Percentage:             {percentage:.3f}% of total")
    print(f"\nTotal with Gear5:       {(flashdepth_total + gear5_trainable)/1e6:.1f}M parameters")
    print("=" * 60)


if __name__ == "__main__":
    detailed_count()
