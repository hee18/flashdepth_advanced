#!/usr/bin/env python3
"""
Calculate exact Gear5 parameters from training log.
Based on actual training log from results_21/gear_5_mamba/large/
"""

print("=" * 70)
print("Gear5 TemporalScalePredictor Parameter Breakdown (Mamba2)")
print("=" * 70)

# 1. feature_net
feature_net_weight = 256 * 1024  # [256, 1024]
feature_net_bias = 256
feature_net_total = feature_net_weight + feature_net_bias
print(f"\n1. feature_net (Linear 1024→256):")
print(f"   - weight: {feature_net_weight:,}")
print(f"   - bias:   {feature_net_bias:,}")
print(f"   - Total:  {feature_net_total:,}")

# 2. temporal_mamba.norm1 (LayerNorm)
norm1_weight = 256
norm1_bias = 256
norm1_total = norm1_weight + norm1_bias
print(f"\n2. temporal_mamba.norm1 (LayerNorm):")
print(f"   - weight: {norm1_weight:,}")
print(f"   - bias:   {norm1_bias:,}")
print(f"   - Total:  {norm1_total:,}")

# 3. temporal_mamba.mamba (Mamba2 core)
dt_bias = 8
A_log = 8
D = 8
in_proj_weight = 1160 * 256  # [1160, 256]
conv1d_weight = 640 * 1 * 4  # [640, 1, 4]
conv1d_bias = 640
norm_weight = 512
out_proj_weight = 256 * 512  # [256, 512]

mamba_total = dt_bias + A_log + D + in_proj_weight + conv1d_weight + conv1d_bias + norm_weight + out_proj_weight
print(f"\n3. temporal_mamba.mamba (Mamba2 core):")
print(f"   - dt_bias:         {dt_bias:,}")
print(f"   - A_log:           {A_log:,}")
print(f"   - D:               {D:,}")
print(f"   - in_proj.weight:  {in_proj_weight:,}")
print(f"   - conv1d.weight:   {conv1d_weight:,}")
print(f"   - conv1d.bias:     {conv1d_bias:,}")
print(f"   - norm.weight:     {norm_weight:,}")
print(f"   - out_proj.weight: {out_proj_weight:,}")
print(f"   - Total:           {mamba_total:,}")

# 4. temporal_mamba.norm2 (LayerNorm)
norm2_weight = 256
norm2_bias = 256
norm2_total = norm2_weight + norm2_bias
print(f"\n4. temporal_mamba.norm2 (LayerNorm):")
print(f"   - weight: {norm2_weight:,}")
print(f"   - bias:   {norm2_bias:,}")
print(f"   - Total:  {norm2_total:,}")

# 5. temporal_mamba.mlp
mlp0_weight = 1024 * 256  # [1024, 256]
mlp0_bias = 1024
mlp2_weight = 256 * 1024  # [256, 1024]
mlp2_bias = 256
mlp_total = mlp0_weight + mlp0_bias + mlp2_weight + mlp2_bias
print(f"\n5. temporal_mamba.mlp:")
print(f"   - mlp.0.weight: {mlp0_weight:,}")
print(f"   - mlp.0.bias:   {mlp0_bias:,}")
print(f"   - mlp.2.weight: {mlp2_weight:,}")
print(f"   - mlp.2.bias:   {mlp2_bias:,}")
print(f"   - Total:        {mlp_total:,}")

# 6. mamba_proj
mamba_proj_weight = 128 * 256  # [128, 256]
mamba_proj_bias = 128
mamba_proj_total = mamba_proj_weight + mamba_proj_bias
print(f"\n6. mamba_proj (Linear 256→128):")
print(f"   - weight: {mamba_proj_weight:,}")
print(f"   - bias:   {mamba_proj_bias:,}")
print(f"   - Total:  {mamba_proj_total:,}")

# 7. scale_head
scale_head_weight = 1 * 128  # [1, 128]
scale_head_bias = 1
scale_head_total = scale_head_weight + scale_head_bias
print(f"\n7. scale_head (Linear 128→1):")
print(f"   - weight: {scale_head_weight:,}")
print(f"   - bias:   {scale_head_bias:,}")
print(f"   - Total:  {scale_head_total:,}")

# 8. shift_head
shift_head_weight = 1 * 128  # [1, 128]
shift_head_bias = 1
shift_head_total = shift_head_weight + shift_head_bias
print(f"\n8. shift_head (Linear 128→1):")
print(f"   - weight: {shift_head_weight:,}")
print(f"   - bias:   {shift_head_bias:,}")
print(f"   - Total:  {shift_head_total:,}")

# Total
total = (feature_net_total + norm1_total + mamba_total + norm2_total +
         mlp_total + mamba_proj_total + scale_head_total + shift_head_total)

print("\n" + "=" * 70)
print("TOTAL TemporalScalePredictor (Mamba2):")
print(f"  {total:,} parameters")
print(f"  ~{total/1000:.0f}K parameters")
print(f"  ~{total/1e6:.2f}M parameters")
print("=" * 70)

# Breakdown by component type
print("\n" + "=" * 70)
print("Component Type Breakdown:")
print("=" * 70)
mamba_block_total = norm1_total + mamba_total + norm2_total + mlp_total
print(f"1. Feature extraction:  {feature_net_total:,} ({feature_net_total/total*100:.1f}%)")
print(f"2. MambaBlock (full):   {mamba_block_total:,} ({mamba_block_total/total*100:.1f}%)")
print(f"   - Mamba2 core:       {mamba_total:,} ({mamba_total/total*100:.1f}%)")
print(f"   - LayerNorms (×2):   {norm1_total + norm2_total:,} ({(norm1_total + norm2_total)/total*100:.1f}%)")
print(f"   - MLP:               {mlp_total:,} ({mlp_total/total*100:.1f}%)")
print(f"3. Projection:          {mamba_proj_total:,} ({mamba_proj_total/total*100:.1f}%)")
print(f"4. Prediction heads:    {scale_head_total + shift_head_total:,} ({(scale_head_total + shift_head_total)/total*100:.1f}%)")

# ImportanceMapGenerator has 0 parameters (no learnable params)
print("\n9. ImportanceMapGenerator:")
print("   - Total: 0 (no learnable parameters)")

print("\n" + "=" * 70)
print("Total Gear5MetricHead = TemporalScalePredictor + ImportanceMapGenerator")
print(f"  {total:,} parameters (verified from training log)")
print("=" * 70)

# FlashDepth baseline comparison
print("\n" + "=" * 70)
print("FlashDepth Model Comparison:")
print("=" * 70)

# From CLAUDE.md and code:
# DINOv2-L: ~300M (ViT-Large)
# DPT: ~15M
# Mamba (FlashDepth original): ~4.3M
# output_conv: ~0.3M
dinov2_params = 300_000_000
dpt_params = 15_000_000
mamba_flashdepth_params = 4_300_000
output_conv_params = 300_000

flashdepth_baseline = dinov2_params + dpt_params + mamba_flashdepth_params + output_conv_params

print(f"\nFlashDepth Baseline (Large model):")
print(f"  - DINOv2-L:        {dinov2_params/1e6:6.1f}M")
print(f"  - DPT:             {dpt_params/1e6:6.1f}M")
print(f"  - Mamba (frozen):  {mamba_flashdepth_params/1e6:6.1f}M")
print(f"  - output_conv:     {output_conv_params/1e6:6.1f}M")
print(f"  - Total:           {flashdepth_baseline/1e6:6.1f}M")

print(f"\nGear5 Addition:")
print(f"  - Gear5MetricHead: {total/1e6:6.2f}M")

total_with_gear5 = flashdepth_baseline + total
percentage = (total / total_with_gear5) * 100

print(f"\nTotal with Gear5:")
print(f"  - {total_with_gear5/1e6:6.1f}M parameters")
print(f"  - Gear5 is {percentage:.2f}% of total")

print("\n" + "=" * 70)
print("PAPER CORRECTION:")
print("=" * 70)
print("논문에 적힌 값:")
print("  - Gear5: ~360K params")
print("  - 전체의 0.1%")
print("  - Total: 319.7M")
print()
print("실제 값 (Mamba2 기준):")
print(f"  - Gear5: ~{total/1000:.0f}K params ({total/1e6:.2f}M)")
print(f"  - 전체의 {percentage:.2f}%")
print(f"  - Total: {total_with_gear5/1e6:.1f}M")
print("=" * 70)
