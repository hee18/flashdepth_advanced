# Changelog

## 2026-03-05: Onepiece V3 — Spatial Mamba + Dual-Stream Architecture

Major architectural rewrite from V1 (global token Mamba) to V3 (spatial Mamba).

### Architecture Changes

**`flashdepth/onepiece_modules.py`**
- Commented out `UnifiedGlobalMamba` and `OnepieceMetricHead` (V1 classes)
- Added `SpatialMamba`: FlashDepth-style spatial Mamba on 1/10 downsampled DPT features
  - 4 MambaBlock layers, zero-init final_layer, upsample+add residual
  - Per-frame streaming with InferenceParams for inference
  - ViT-S auto-detection: expand=4, headdim=32 when dpt_dim=64
- Added `ConvMetricHead`: Conv1x1-based scale/shift from low-res Mamba output + GAP
  - Supports metric mode (softplus scale, sigmoid shift) and inverse mode
- Kept `SceneCutDetector` unchanged

**`flashdepth/model.py`**
- Replaced `UnifiedGlobalMamba` → `SpatialMamba`, `OnepieceMetricHead` → `ConvMetricHead` in constructor
- Rewrote `forward_with_onepiece()`: DINOv2→DPT→SpatialMamba→ConvMetricHead→final_head→metric
  - No CLS extraction in training, no FiLM modulation
  - Returns `post_mamba_features` instead of `cls_tokens`/`scene_cut_weights`/`d_cls`
- Rewrote `forward_with_onepiece_streaming()`: Frame-by-frame with SCD, returns `reset_frames`
- Added `forward_onepiece_single_frame()`: Single-frame helper for test scripts

### Loss Changes

**`utils/onepiece_losses.py`**
- Renamed `WarpFeatureConsistencyLoss` → `OpticalFlowConsistencyLoss`
- Changed input: `dpt_features` → `post_mamba_features`
- Renamed: `feat_cons_weight` → `ofc_weight`, `feat_cons_loss` → `ofc_loss`

### Training Changes

**`train_onepiece.py`**
- Phase 1 transition: 5000 → **1500 steps**
- Phase 1 trainable: **ConvMetricHead only** (NOT SpatialMamba — frozen with zero-init)
- Phase 2 trainable: SpatialMamba + ConvMetricHead + DPT + output_conv
- Removed Scene Cut Detection from training (no CLS, no W_temporal gating)
- Added `train_mode` support: metric (default) or inverse
- Validation max depth: 70m → **80m**
- Phase 1 skips OFC loss (DPT frozen → no gradient)

### Config Changes

**`configs/onepiece/config.yaml`, `config_l.yaml`, `config_s.yaml`**
- Replaced `unified_mamba_*` → `spatial_mamba_layers`, `spatial_mamba_d_state`, `spatial_mamba_d_conv`, `spatial_mamba_downsample`
- Added `train_mode: metric`
- Changed `phase.auto_transition_step: 1500`
- Changed `loss.ofc_weight: 0.01` (from feat_cons_weight)
- Removed `no_shift`, `cls_layers`

### Test Files

**`test_onepiece.py`**
- Updated for V3 API: removed d_cls, scene_cut_weights, cls_layer_indices
- Added `reset_frames` tracking from SCD at inference
- Replaced all hardcoded `MAX_DEPTH = 70.0` with configurable `self.max_depth` (default 80.0)
- Added `--max-depth` CLI argument (Hydra sys.argv pre-processing)
- Updated best/worst frame visualization: D_cls plot → Scene Cut Resets panel

**`test_gear5.py`** — Copied from onepiece2 branch, added `--max-depth` support
**`test_comparison.py`** — Copied from onepiece2 branch, added `--max-depth` support
**`test_video_comparison.py`** — Copied from onepiece2 branch, added `--max-depth` support

### Shell Scripts

**`run_comparison.sh`**, **`run_video_comparison.sh`**
- Added `--max-depth` argument (default: 80)

**`run_docker.sh`**
- Updated `test_onepiece` command: removed `cls_layers`, `no_shift` from docker command
- Added `--max-depth` pass-through for test_onepiece
- Changed default MAX_DEPTH from 70.0 → 80.0

### Documentation

- Rewrote `Onepiece.md` for V3 architecture
- Created `changelog.md`
