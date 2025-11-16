#!/usr/bin/env python3
"""
CUT3R Adapter 테스트 스크립트
"""
import sys
import torch
import numpy as np
from pathlib import Path

print("="*80)
print("CUT3R Adapter Test")
print("="*80)

# Import adapter
from adapters.cut3r_adapter import CUT3RAdapter

# Create adapter
print("\n1. Creating CUT3R adapter...")
adapter = CUT3RAdapter(size=512)

# Check checkpoint
checkpoint_path = Path('refer_test/CUT3R/checkpoints/cut3r_512_dpt_4_64.pth')
if not checkpoint_path.exists():
    print(f"✗ Checkpoint not found: {checkpoint_path}")
    print("  Download from: https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view")
    sys.exit(1)

print(f"✓ Checkpoint found: {checkpoint_path}")

# Load model
print("\n2. Loading CUT3R model...")
try:
    adapter.load_model()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create dummy input
print("\n3. Creating dummy input...")
B, T, C, H, W = 1, 3, 3, 375, 1242  # VKITTI resolution
images = torch.rand(B, T, C, H, W).cuda()
print(f"  Input shape: {images.shape}")

# Run inference
print("\n4. Running inference...")
try:
    with torch.no_grad():
        depths = adapter.inference(images)
    print(f"✓ Inference successful")
    print(f"  Output shape: {depths.shape}")
    print(f"  Depth range: [{depths.min():.2f}, {depths.max():.2f}]")
except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Validate output
print("\n5. Validating output...")
assert depths.shape == (B, T, H, W), f"Expected shape {(B, T, H, W)}, got {depths.shape}"
assert not torch.isnan(depths).any(), "Output contains NaN values"
assert not torch.isinf(depths).any(), "Output contains Inf values"
print("✓ Output validation passed")

print("\n" + "="*80)
print("CUT3R Adapter Test: SUCCESS")
print("="*80)
