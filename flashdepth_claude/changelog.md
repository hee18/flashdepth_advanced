# Changelog

## 2026-03-09: Validation Loss 통일 + test_onepiece 해상도 처리 개선

### Validation Loss 통일 (`train_onepiece.py`)
- Validation의 LogL1/TGM loss를 training과 동일한 `self.loss_fn` (OnepieceCombinedLoss) 사용으로 변경
  - 기존: validation은 metric meters space에서 inline LogL1 + raw inverse space inline TGM (single scale, no trimming)
  - 변경: training과 동일한 inverse depth 100/m space에서 LogL1Loss + TGMTemporalLoss (log-space, multi-scale, trimmed MAE)
- Valid mask는 validation용 80m threshold 유지 (test evaluation과 일치)

### Visualization 개선 (`utils/onepiece_visualization.py`)
- Depth Metrics 2개씩 한 줄에 배치 (AbsRel+Delta1, Delta2+Delta3, RMSE+MAE)
- TGM과 OFC를 같은 줄에 배치, `feat_cons_loss` → `ofc_loss`로 변경

### test_onepiece.py 해상도 처리 개선
- 기존: GT를 pred 해상도로 downsample → pred 해상도에서 metric 계산
- 변경: pred를 GT 해상도로 upsample → GT 해상도에서 metric 계산 (GT 정보 보존)
- `gt_at_pred_res_cpu` 추가: TC, visualization용 (pred 해상도)
- `gt_depth_metric_cpu` 유지: per-frame metrics, depth range analysis, TAE, optimal scale/shift용 (GT 해상도)
- Sparse dataset (eth3d, waymo_seg) 감지 → nearest interpolation 사용
- GPU 메모리 관리: `del gt_depth_metric` (GPU tensor) 즉시 해제

### test_onepiece.py 기능 추가 (onepiece2 포팅)
- **PSR (Prediction Stability Ratio)**: TC 모드/Full 모드 모두 추가. per-frame scale ratio → 인접 프레임 차이 평균
- **TC-only 모드 TAE 계산**: 기존 0.0 하드코딩 → 실제 reprojection TAE 계산
- **`ea` test mode**: Error Analysis only. TAE/rTC/PSR 스킵 → accuracy metric만 빠르게 측정
- **`_save_tc_summary()`**: rTC + TAE + PSR 통합 요약 JSON 저장
- **`metric_order`**: `psr`, `psr_max` 추가
- **FPS 로깅 개선**: warmup 정보 포함한 상세 로깅

### test_onepiece.py Visualization 개선
- Row 2: Scale/Shift plot → **GT Valid Mask** (density % 표시)
- Row 3: Reset Frames 패널 제거 → Depth Distribution colspan=full
- Sparse 감지: density heuristic (0.5) → **dataset 이름 기반** (eth3d, waymo_seg)

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
