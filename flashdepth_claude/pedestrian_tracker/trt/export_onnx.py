#!/usr/bin/env python3
"""
Onepiece ONNX Export: BackboneDPT + FinalHead

PC에서 실행하여 ONNX 파일 생성 → Orin으로 전송 → Orin에서 TRT 엔진 빌드

Usage:
    python pedestrian_tracker/trt/export_onnx.py
    python pedestrian_tracker/trt/export_onnx.py --input-h 518 --input-w 756
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

# Project paths
FLASHDEPTH_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(FLASHDEPTH_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─── Wrapper modules for clean ONNX export ───

class BackboneDPT(nn.Module):
    """
    DINOv2 encoder + DPT decoder → dpt_features, cls_token.

    Wraps the frozen backbone into a single traceable module.
    """

    def __init__(self, model):
        super().__init__()
        self.pretrained = model.pretrained
        self.depth_head = model.depth_head
        self.encoder = model.encoder
        self.intermediate_layer_idx = model.intermediate_layer_idx
        self.cls_layer_indices = model.cls_layer_indices
        self.patch_size = model.patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        layer_indices = self.intermediate_layer_idx[self.encoder]

        # DINOv2: get intermediate features + CLS
        raw_outputs = self.pretrained._get_intermediate_layers_not_chunked(x, layer_indices)
        normed = [self.pretrained.norm(out) for out in raw_outputs]

        # CLS token (average of selected layers)
        selected_cls = [normed[idx][:, 0] for idx in self.cls_layer_indices]
        cls_token = torch.stack(selected_cls, dim=0).mean(dim=0)  # [B, embed_dim]

        # Patch features (strip CLS)
        features = [out[:, 1:] for out in normed]

        # DPT decoder → dpt_features
        dpt_features = self.depth_head(features, patch_h, patch_w)  # [B, dpt_dim, h, w]

        return dpt_features, cls_token


class FinalHeadModule(nn.Module):
    """
    Final head: post_mamba [B, dpt_dim, h, w] → relative_depth [B, H, W].

    patch_h, patch_w are baked in at export time (fixed input resolution).
    """

    def __init__(self, model, patch_h, patch_w):
        super().__init__()
        self.output_conv1 = model.depth_head.scratch.output_conv1
        self.output_conv2 = model.depth_head.scratch.output_conv2
        self.target_h = patch_h * model.patch_size
        self.target_w = patch_w * model.patch_size

    def forward(self, post_mamba):
        out = self.output_conv1(post_mamba)
        out = F.interpolate(out, (self.target_h, self.target_w),
                            mode='bilinear', align_corners=True)
        out = self.output_conv2(out)
        return F.relu(out).squeeze(1)  # [B, H, W]


# ─── Model loading ───

def load_onepiece_model(config_path, checkpoint_path, device='cuda'):
    """Load FlashDepth with Onepiece config."""
    import yaml
    from omegaconf import OmegaConf
    from flashdepth.model import FlashDepth

    config_abs = FLASHDEPTH_ROOT / config_path
    with open(config_abs) as f:
        config = OmegaConf.create(yaml.safe_load(f))

    model_config = dict(config.model)
    model_config['batch_size'] = 1
    model_config['use_metric_head'] = False
    model_config['use_onepiece'] = True
    model_config['spatial_mamba_layers'] = config.model.get('spatial_mamba_layers', 4)
    model_config['spatial_mamba_d_state'] = config.model.get('spatial_mamba_d_state', 256)
    model_config['spatial_mamba_d_conv'] = config.model.get('spatial_mamba_d_conv', 4)
    model_config['spatial_mamba_downsample'] = config.model.get('spatial_mamba_downsample', 0.1)
    model_config['onepiece_train_mode'] = config.get('train_mode', 'metric')
    model_config['hybrid_configs'] = config.get('hybrid_configs', None)
    scene_cut = config.get('scene_cut', {})
    model_config['scene_cut_tau'] = scene_cut.get('tau', 0.05) if scene_cut else 0.05
    model_config['scene_cut_k'] = scene_cut.get('k', 80) if scene_cut else 80

    model = FlashDepth(**model_config)

    ckpt_abs = FLASHDEPTH_ROOT / checkpoint_path
    logger.info(f"Loading checkpoint: {ckpt_abs}")
    checkpoint = torch.load(str(ckpt_abs), map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    logger.info("Model loaded")

    return model.to(device).eval()


# ─── ONNX export functions ───

def disable_xformers(model):
    """
    xformers의 memory_efficient_attention은 ONNX 미지원.
    MemEffAttention이 표준 Attention.forward()로 fallback하도록 강제.
    """
    import flashdepth.dinov2_layers.attention as attn_module
    attn_module.XFORMERS_AVAILABLE = False
    logger.info("Disabled xformers → standard attention fallback for ONNX export")


def export_backbone_dpt(model, input_h, input_w, output_path, device='cuda'):
    """BackboneDPT (DINOv2 + DPT) → ONNX."""
    disable_xformers(model)

    backbone = BackboneDPT(model).eval()
    dummy = torch.randn(1, 3, input_h, input_w, device=device)

    logger.info(f"Exporting BackboneDPT: input=[1, 3, {input_h}, {input_w}]")

    # Test forward first
    with torch.no_grad():
        dpt_features, cls_token = backbone(dummy)
    logger.info(f"  dpt_features: {dpt_features.shape}")
    logger.info(f"  cls_token: {cls_token.shape}")

    # ONNX export
    torch.onnx.export(
        backbone, dummy,
        str(output_path),
        input_names=['input'],
        output_names=['dpt_features', 'cls_token'],
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info(f"Saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


def export_final_head(model, input_h, input_w, output_path, device='cuda'):
    """FinalHead (output_conv1 + interpolate + output_conv2) → ONNX."""
    patch_size = model.patch_size
    patch_h = input_h // patch_size
    patch_w = input_w // patch_size

    # dpt_features shape: [B, dpt_dim, patch_h*8, patch_w*8]
    # (DPT resize_layer[0] = ×4 ConvTranspose, then refinenet1 = ×2 interpolate)
    dpt_dim = 256 if model.encoder == 'vitl' else 64
    feat_h = patch_h * 8
    feat_w = patch_w * 8

    head = FinalHeadModule(model, patch_h, patch_w).eval().to(device)
    dummy = torch.randn(1, dpt_dim, feat_h, feat_w, device=device)

    logger.info(f"Exporting FinalHead: input=[1, {dpt_dim}, {feat_h}, {feat_w}]")

    with torch.no_grad():
        out = head(dummy)
    logger.info(f"  output: {out.shape}")

    torch.onnx.export(
        head, dummy,
        str(output_path),
        input_names=['post_mamba'],
        output_names=['relative_depth'],
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info(f"Saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


# ─── Verification ───

def verify_onnx(onnx_path):
    """ONNX 파일 유효성 검증."""
    import onnx
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    logger.info(f"ONNX verification passed: {onnx_path}")

    # Print input/output info
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        logger.info(f"  Input '{inp.name}': {shape}")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        logger.info(f"  Output '{out.name}': {shape}")


def main():
    parser = argparse.ArgumentParser(description='Onepiece ONNX Export')
    parser.add_argument('--config', type=str,
                        default='configs/onepiece/config.yaml')
    parser.add_argument('--checkpoint', type=str,
                        default='train_results/results_34/onepiece/large/last.pth')
    parser.add_argument('--input-h', type=int, default=518,
                        help='Input height (must be multiple of 14)')
    parser.add_argument('--input-w', type=int, default=756,
                        help='Input width (must be multiple of 14)')
    parser.add_argument('--output-dir', type=str,
                        default='pedestrian_tracker/models')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    assert args.input_h % 14 == 0, f"input_h must be multiple of 14, got {args.input_h}"
    assert args.input_w % 14 == 0, f"input_w must be multiple of 14, got {args.input_w}"

    device = f'cuda:{args.gpu}'
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_onepiece_model(args.config, args.checkpoint, device)

    # Export BackboneDPT
    backbone_path = Path(args.output_dir) / 'backbone_dpt.onnx'
    export_backbone_dpt(model, args.input_h, args.input_w, backbone_path, device)
    verify_onnx(backbone_path)

    # Export FinalHead
    final_head_path = Path(args.output_dir) / 'final_head.onnx'
    export_final_head(model, args.input_h, args.input_w, final_head_path, device)
    verify_onnx(final_head_path)

    logger.info("=" * 60)
    logger.info("ONNX export complete!")
    logger.info(f"  BackboneDPT: {backbone_path}")
    logger.info(f"  FinalHead:   {final_head_path}")
    logger.info("")
    logger.info("Next steps (on Jetson Orin):")
    logger.info("  trtexec --onnx=backbone_dpt.onnx --saveEngine=backbone_dpt_fp16.engine --fp16")
    logger.info("  trtexec --onnx=final_head.onnx --saveEngine=final_head_fp16.engine --fp16")


if __name__ == '__main__':
    main()
