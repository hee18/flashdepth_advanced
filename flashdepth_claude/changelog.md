# Changelog

## 2026-03-11: --save-depth-maps 플래그 추가 (pred depth를 .npy로 저장)

### 변경 파일
- `test_comparison.py`, `test_video_comparison.py`, `test_onepiece.py`
- `run_comparison.sh`, `run_video_comparison.sh`, `run_docker.sh`

### 내용
- `--save-depth-maps` 플래그 추가: 모델 예측 depth map을 float32 `.npy` 파일로 저장
- 저장 경로: `{results_dir}/depth_maps/seq{sequence_id:04d}/pred_{frame:04d}.npy`
- 저장 해상도: 각 모델의 실제 예측 해상도 (GT 해상도로 업샘플하지 않음)
- Shape: `[H, W]` float32 (채널 차원 제거)
- 목적: 나중에 추가 지표 필요 시 모델 재추론 없이 저장된 depth map으로 계산 가능
- GT depth는 데이터셋에 이미 존재하므로 저장하지 않음
- 기본값: off (디스크 부담 없이 필요할 때만 활성화)

## 2026-03-10: FlashDepth test_video_comparison 통합 (scale/shift alignment + rTC/PSR/TAE)

### test_video_comparison.py — FlashDepth scale/shift alignment 추가
- FlashDepth를 `relative_depth_methods`에 등록 (depth_mode='metric' 사용 시 에러)
- `depth_mode='relative'`일 때 시퀀스 전체에 공통 scale/shift alignment 수행 (least-squares)
  - 시퀀스별 독립 scale/shift, 동일 시퀀스 내 전 프레임에 동일 적용
  - GT resolution에서 valid pixels로 alignment 계산 후, pred resolution에 적용
- alignment 후 rTC, PSR, TAE가 모두 aligned metric depth 기준으로 계산됨
- `--test-mode tc`: rTC + PSR + TAE 계산 (per-frame error metrics는 스킵)
- `tc_summary.json`에 시퀀스별 scale/shift 값 기록

### test_video_comparison.py — rTC/PSR은 raw, TAE만 aligned 사용
- alignment 결과를 `pred_depths_aligned`에 별도 보관, `pred_depths_cpu`(raw)는 변경하지 않음
- rTC: raw predictions 사용 (scale-invariant, shift가 개입하면 왜곡)
- PSR: raw predictions 사용 (프레임 간 scale 안정성 측정, alignment 전이 의미있음)
- TAE: aligned predictions 사용 (metric scale 필요한 reprojection error)
- TC 모드와 기본 모드 모두 동일 원칙 적용

### test_video_comparison.py — `del images` 후 참조 에러 수정
- `del images` 후 시각화에서 `images` 참조 시 `UnboundLocalError` 발생
- 수정: `del images` 직후 `images = batch['images']` (CPU 복사본)으로 재할당

### run_docker.sh — test_original_flashdepth --test-mode 경로 수정
- `--depth-mode metric` → `--depth-mode relative` (FlashDepth는 relative depth 출력)
- `--max-depth $EVAL_MAX_DEPTH` 전달 추가

## 2026-03-10: Valid Mask max_depth 하드코딩 수정 + waymo_seg inverse depth 변환

### Valid Mask 시각화 max_depth 수정 (`utils/comparison_visualization.py`)
- `MAX_DEPTH = 70.0` 하드코딩 → `max_depth` 파라미터로 외부에서 전달받도록 변경
- `visualize_sequence_simplified()`, `visualize_best_frame_simplified()` 두 함수 모두 `max_depth` 파라미터 추가
- Valid Mask 타이틀: `GT ≤70m` → `GT ≤{max_depth}m` 동적 표시
- Depth Distribution 타이틀도 동일하게 동적 표시

### 호출부 max_depth 전달 (`test_video_comparison.py`, `test_comparison.py`)
- 두 test 스크립트 모두 `visualize_sequence_simplified()`, `visualize_best_frame_simplified()` 호출 시 `max_depth=MAX_DEPTH` 전달

### SegmentationDataset 코드 제거 (`test_video_comparison.py`, `test_comparison.py`)
- 두 스크립트 모두 이미 ComparisonDataset만 사용 (metric depth, [0,1] 이미지)
- SegmentationDataset fallback 제거: `batch['depth']` fallback, `batch['image']` legacy 경로 삭제
- inverse depth 변환 코드 제거: ComparisonDataset은 이미 metric depth(m) 반환
- ImageNet unnormalization 코드 제거: ComparisonDataset은 이미 [0,1] 범위 반환
- `images_unnorm` → `images`, `img_t_unnorm` → `img_t`로 단순화
- `focal_lengths_actual` SegmentationDataset 전용 fallback 제거

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
