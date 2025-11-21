#!/usr/bin/env python3
"""Inspect checkpoint to diagnose dimension mismatch."""

import torch
import sys

checkpoint_path = 'train_results/results_20/gear_5_mamba/small/best.pth'

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*80)
print("CHECKPOINT KEYS:")
print("="*80)
for key in checkpoint.keys():
    print(f"  - {key}")

print("\n" + "="*80)
print("MODEL STATE_DICT KEYS (first 20):")
print("="*80)
# Try both 'model_state_dict' and 'model'
state_dict = None
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print("  Using 'model_state_dict'")
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
    print("  Using 'model'")

if state_dict is not None:
    for i, key in enumerate(list(state_dict.keys())[:20]):
        print(f"  {key}: {state_dict[key].shape}")
    print(f"  ... (total {len(state_dict)} keys)")
else:
    print("  No model state dict found")

print("\n" + "="*80)
print("GSP (GEAR5_METRIC_HEAD) PARAMETERS:")
print("="*80)
if state_dict is not None:
    gsp_keys = [k for k in state_dict.keys() if 'gear5_metric_head' in k]
    if gsp_keys:
        for key in gsp_keys:
            print(f"  {key}: {state_dict[key].shape}")
    else:
        print("  No GSP parameters found")
else:
    print("  No state dict available")

print("\n" + "="*80)
print("VIT ENCODER INFORMATION:")
print("="*80)
if state_dict is not None:
    # Check for ViT embedding layer (first layer, indicates hidden dim)
    vit_keys = [k for k in state_dict.keys() if 'pretrained' in k and 'patch_embed' in k]
    if vit_keys:
        print("  ViT Patch Embedding layers:")
        for key in vit_keys[:5]:
            print(f"    {key}: {state_dict[key].shape}")

    # Check for position embeddings (indicates sequence length and hidden dim)
    pos_keys = [k for k in state_dict.keys() if 'pos_embed' in k]
    if pos_keys:
        print("  Position Embedding:")
        for key in pos_keys[:3]:
            print(f"    {key}: {state_dict[key].shape}")

    # Check attention blocks
    attn_keys = [k for k in state_dict.keys() if 'pretrained.blocks' in k and 'qkv.weight' in k]
    if attn_keys:
        print(f"  Attention QKV weight (first block): {state_dict[attn_keys[0]].shape}")
        # QKV weight shape is [3*hidden_dim, hidden_dim]
        # So if shape is [2304, 768], then hidden_dim = 768 (ViT-S)
        # If shape is [3072, 1024], then hidden_dim = 1024 (ViT-L)
        qkv_shape = state_dict[attn_keys[0]].shape
        hidden_dim = qkv_shape[1]
        print(f"  → Inferred ViT hidden_dim: {hidden_dim}")
        if hidden_dim == 768:
            print(f"  → This is ViT-S (Small)")
        elif hidden_dim == 1024:
            print(f"  → This is ViT-L (Large)")
        else:
            print(f"  → Unknown ViT variant")
else:
    print("  No state dict available")

print("\n" + "="*80)
print("CONFIG INFORMATION:")
print("="*80)
if 'config' in checkpoint:
    config = checkpoint['config']
    print(f"  Config type: {type(config)}")
    if hasattr(config, 'model'):
        print(f"  model.vit_size: {getattr(config.model, 'vit_size', 'N/A')}")
        print(f"  model.target_blocks: {getattr(config.model, 'target_blocks', 'N/A')}")
    if hasattr(config, 'config_variant'):
        print(f"  config_variant: {config.config_variant}")
else:
    print("  No config found in checkpoint")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

if state_dict is not None:
    # Get GSP input dimension
    gsp_weight_key = 'gear5_metric_head.temporal_scale_predictor.feature_net.0.weight'
    if gsp_weight_key in state_dict:
        gsp_weight_shape = state_dict[gsp_weight_key].shape
        gsp_input_dim = gsp_weight_shape[1]  # [out_dim, in_dim]
        print(f"  GSP input dimension: {gsp_input_dim}")

        # Get ViT hidden dimension
        attn_keys = [k for k in state_dict.keys() if 'pretrained.blocks' in k and 'qkv.weight' in k]
        if attn_keys:
            vit_hidden_dim = state_dict[attn_keys[0]].shape[1]
            print(f"  ViT hidden dimension: {vit_hidden_dim}")

            # Check for hybrid fusion
            hybrid_keys = [k for k in state_dict.keys() if 'hybrid_fusion' in k]
            teacher_keys = [k for k in state_dict.keys() if 'teacher_model' in k]

            if hybrid_keys or teacher_keys:
                print(f"  Hybrid model detected (has hybrid_fusion or teacher_model)")
                print(f"  → In hybrid models, Student is ViT-S (768) but GSP might use averaged tokens")

                # Check target_blocks from config
                config = checkpoint.get('config', {})
                if isinstance(config, dict):
                    target_blocks = config.get('model', {}).get('target_blocks', None)
                else:
                    target_blocks = getattr(getattr(config, 'model', None), 'target_blocks', None)

                if target_blocks:
                    print(f"  → target_blocks: {target_blocks}")
                    print(f"  → Number of layers averaged: {len(target_blocks)}")

                    if len(target_blocks) == 2 and gsp_input_dim == 384:
                        print(f"\n  🔍 DIAGNOSIS: Hybrid model with 2-layer averaging")
                        print(f"     - Student ViT-S has 768-dim CLS tokens")
                        print(f"     - But config used HALF of the layers (maybe wrong target_blocks?)")
                        print(f"     - Expected: 768-dim input for GSP")
                        print(f"     - Found: 384-dim input in checkpoint")
            else:
                print(f"  Single model (not hybrid)")
                if gsp_input_dim != vit_hidden_dim:
                    print(f"\n  ⚠️  WARNING: Dimension mismatch!")
                    print(f"     GSP expects {gsp_input_dim}-dim but ViT produces {vit_hidden_dim}-dim")
else:
    print("  No state dict available for analysis")
