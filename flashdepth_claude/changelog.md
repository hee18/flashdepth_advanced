# Changelog

## 2026-05-06: Hybrid Onepiece에 student CLS 옵션 추가 (--student-cls)

### 변경 파일
- **`flashdepth/model.py`**: `use_teacher_cls` flag 추가. `__init__`에서 `cls_embed_dim` 분기 처리. `forward_with_onepiece`, `forward_with_onepiece_streaming`, `forward_onepiece_single_frame` 3개 forward 메서드에서 `use_teacher_cls=False` 시 student ViT-S CLS 사용 경로 추가
- **`configs/onepiece/config_hybrid.yaml`**: `model.use_teacher_cls: true` 기본값 추가
- **`run_docker.sh`**: `--student-cls` 플래그 추가 (`USE_TEACHER_CLS=false` 설정). `train_onepiece`, `train_onepiece_ddp`, `train_onepiece_fsdp` 3개 명령에 `model.use_teacher_cls=$USE_TEACHER_CLS` 전달

### 내용
- 기존: Hybrid 모드에서 항상 teacher ViT-L CLS(1024-dim) 사용
- 변경: `use_teacher_cls=false`이면 student ViT-S CLS(384-dim) 사용 가능
- DPT fusion용 teacher forward pass는 두 경우 모두 실행 (teacher features는 여전히 필요)
- `use_teacher_cls=false` 시 `cls_projection`은 `Linear(384, 64)` (기존 `Linear(1024, 64)`)

### 사용법
```bash
./run_docker.sh train_onepiece_ddp --config-variant hybrid --student-cls
```

---

## 2026-05-06: run_docker.sh test_onepiece --resolution 누락 버그 수정

### 변경 파일
**`run_docker.sh`**
- `test_onepiece` 배치 모드 docker command에 `+resolution=$RESOLUTION` 추가
- `test_onepiece` 싱글 모드 docker command에 `+resolution=$RESOLUTION`, `training.workers=$WORKERS` 추가

### 버그 원인
- `test_onepiece` 블록(배치/싱글 양쪽)에 `+resolution=$RESOLUTION`이 누락되어 `--resolution 2k` 플래그가 `test_onepiece.py`에 전달되지 않았음
- `test_onepiece.py`는 `self.config.get('resolution', ...)` 으로 해당 값을 읽으므로, 미전달 시 config 기본값(`eval.test_dataset_resolution: 'base'`)으로 폴백되어 항상 base 해상도로 실행되었음
- 싱글 모드에 `training.workers` 도 누락되어 DataLoader workers=0으로 동작하던 문제도 함께 수정

---

## 2026-04-21: FSDP2 forward_with_onepiece → __call__ 우회 버그 수정

### 변경 파일
**`train_onepiece_fsdp.py`**
- `_setup_model()` 끝에 `model.forward = lambda *a, **kw: FlashDepth.forward_with_onepiece(model, *a, **kw)` 추가
- `train_step`, `_save_training_visualization`, `validate` 3군데의 `model.forward_with_onepiece(...)` → `model(...)` 변경

### 버그 원인
- FSDP2는 `register_forward_pre_hook`을 `nn.Module.__call__`에 등록. pre_forward hook이 root-sharded params를 all-gather해서 DTensor → Tensor 복원
- `model.forward_with_onepiece(...)` 직접 호출은 `__call__`을 건너뜀 → hook 미실행 → `patch_embed.proj.weight`가 DTensor 상태 → 입력 x는 Tensor → `RuntimeError: got mixed torch.Tensor and DTensor`
- `model((images,), phase=...)` 형태로 호출하면 `__call__` → FSDP2 hook 정상 실행 → 가중치 복원 후 forward 실행

---

## 2026-04-21: FSDP2+AC DTensor/Tensor 충돌 버그 수정

### 변경 파일
**`train_onepiece_fsdp.py`**
- `_setup_model()` 내 activation checkpointing의 `check_fn=lambda _: True` 제거
- 변경: ViT block 타입과 DPT refinenet 타입만 개별 래핑
  ```python
  ViTBlockType = type(model.pretrained.blocks[0])
  apply_activation_checkpointing(..., check_fn=lambda m: isinstance(m, ViTBlockType))
  DPTRefinenetType = type(model.depth_head.scratch.refinenet1)
  apply_activation_checkpointing(..., check_fn=lambda m: isinstance(m, DPTRefinenetType))
  ```
- Teacher AC 제거 (frozen → 메모리 절약 없음), hybrid_fusion AC 제거 (단일 유닛 샤딩)

### 버그 원인
- `check_fn=lambda _: True`는 `patch_embed.proj`(Conv2d) 등 FSDP2가 개별 샤딩하지 않는 모듈도 AC 래핑
- Root `fully_shard(model)`이 `patch_embed.proj.weight`를 DTensor로 샤딩
- AC re-run forward 시 input=Tensor, weight=DTensor → `RuntimeError: got mixed torch.Tensor and DTensor`
- 해결: FSDP2 샤딩과 동일한 granularity(블록 단위)로만 AC 적용

---

## 2026-04-21: 로딩 제외 목록 수정 (spatial_mamba, onepiece_metric_head 로딩 허용)

### 변경 파일
**`train_onepiece.py`, `train_onepiece_fsdp.py`**
- 기존: `spatial_mamba`, `onepiece_metric_head`, `scene_cut_detector`, `cls_projection`, `unified_global_mamba` 모두 제외
- 변경: `cls_projection`(384→64 vs 1024→64 shape 불일치), `unified_global_mamba`(레거시)만 제외
- 이유: Onepiece-S 체크포인트에서 Hybrid로 로딩 시 SpatialMamba, CLSMetricHead는 아키텍처 동일 → 랜덤 초기화보다 warm start가 유리
- FlashDepth 체크포인트 사용 시는 기존과 동일 (해당 키 없으므로 strict=False로 처리)

---

## 2026-04-21: train_onepiece.py에 load_teacher 분리 로딩 추가

### 변경 파일

**`train_onepiece.py`**
- `load_teacher` config 지원 추가 (train_onepiece_fsdp.py와 동일한 방식)
- Onepiece-L 체크포인트의 `pretrained.*`, `depth_head.*`를 `teacher_model.*`에 별도 로딩
- 원본 FlashDepth `init_setup.py`와 동일: `model.teacher_model.load_state_dict(remapped, strict=False)`

**`configs/onepiece/config_hybrid.yaml`**
- `load: null`, `load_teacher: null` 필드 추가 (FlashDepth hybrid 체크포인트 사용 제거)

### 원본 파일 수정 없음
- `flashdepth/model.py`, `train_onepiece_fsdp.py`, config_hybrid_fsdp.yaml 미변경

---

## 2026-04-21: FSDP2 run_docker.sh 지원 + load_teacher 분리 로딩 추가

### 변경 파일

**`run_docker.sh`**
- `train_onepiece_fsdp` 커맨드 추가 (torchrun --nproc_per_node=2, FSDP2 전용 NCCL env)
- `--teacher-checkpoint PATH` 옵션 추가 (Onepiece-L 또는 FlashDepth-L 체크포인트)
- 2K hybrid 시 batch_size 자동 1로 설정, `--batch-size 2`로 override 가능

**`train_onepiece_fsdp.py`**
- `load_teacher` config 지원: Onepiece-L / FlashDepth-L 체크포인트를 teacher_model에 별도 로딩
- 키 remapping: `pretrained.*`, `depth_head.*` → `teacher_model.pretrained.*`, `teacher_model.depth_head.*`
- `load_teacher=null` 이면 기존대로 단일 `load` 체크포인트에서 teacher weights 함께 로딩

**`configs/onepiece/config_hybrid_fsdp.yaml`**
- `load_teacher: null` 필드 추가 (주석으로 사용법 설명)

---

## 2026-04-21: FSDP2 학습 스크립트 추가 (Onepiece Hybrid 2K 고해상도)

### 배경
- Onepiece Hybrid (ViT-S student + ViT-L teacher) 2K 해상도 학습 시 DDP로는 GPU당 메모리 한계 초과
- PyTorch 2.4 FSDP2 composable API (`torch.distributed._composable.fsdp`)로 파라미터 샤딩

### 신규 파일

**`train_onepiece_fsdp.py`**
- `train_onepiece.py` 기반, DDP → FSDP2 (`fully_shard`) 교체
- 주요 변경사항:
  - `from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy`
  - `from torch.distributed.device_mesh import init_device_mesh`
  - Activation Checkpointing: `CheckpointImpl.NO_REENTRANT` (FSDP2 composable 요구사항)
  - FSDP wrap 순서 (inner→outer): student ViT blocks → teacher ViT blocks → DPT refinenets → SpatialMamba → HybridFusion → root
  - `MixedPrecisionPolicy(param/output=bf16, reduce=fp32)` — autocast 블록 제거
  - `SpatialMamba`: `reshard_after_forward=False` (프레임 루프 내 반복 all-gather 방지)
  - Teacher blocks: `reshard_after_forward=True` (frozen, peak 메모리 절약)
  - `save_checkpoint`: `get_model_state_dict(full_state_dict=True, cpu_offload=True)` — rank 0에서 fp32 full state dict
  - `validate()` / `_save_training_visualization()`: **모든 rank 참여** (FSDP all-gather 동기화 필수)
  - `_get_model()`: DDP `.module` unwrap 불필요 (composable API는 in-place 수정)

**`configs/onepiece/config_hybrid_fsdp.yaml`**
- `config_hybrid.yaml` 기반, FSDP 전용 설정 추가
- `training.batch_size: 1` (2K에서 메모리 여유분 확인 후 2로 증가 가능)
- `fsdp.reshard_after_forward_student: false`, `reshard_after_forward_teacher: true`
- `resume: null` 필드 추가 (FSDP 체크포인트 재개용)

**`configs/onepiece/config_fsdp.yaml`**
- non-hybrid 518×518 smoke test용 (FSDP2 동작 검증)

### 원본 파일 수정 없음
- `train_onepiece.py`, `flashdepth/model.py`, 기존 config 4종 모두 미변경

## 2026-04-14: Onepiece V3 Small / Hybrid 모델 지원 추가

### 배경
- 기존 Onepiece V3는 ViT-L 단독 모델(O-Large)만 지원
- FlashDepth의 Small(ViT-S) / Hybrid(ViT-S student + ViT-L teacher) 변형에 대응하는 Onepiece 버전 필요
- Hybrid에서는 teacher ViT-L의 CLS token을 사용하여 metric head의 표현력 확보

### 변경 파일

**`flashdepth/model.py`**
- `__init__`: `use_onepiece_hybrid` 플래그 추가. Hybrid일 때 `cls_projection`이 teacher embed_dim(1024)을 입력으로 받도록 분기 (`Linear(1024, 64)` vs `Linear(384, 64)`)
- `forward_with_onepiece()`: Hybrid 분기 추가
  - Student ViT-S → encoder features (DPT 입력)
  - Teacher ViT-L → `_get_intermediate_layers_not_chunked` 한 번 호출로 CLS token + teacher features 동시 추출
  - Teacher path_4 fusion → student DPT with fused_path4 (FlashDepth Hybrid와 동일)
  - Onepiece 전용 temporal_layer=[] (DPT 내 Mamba 미사용)
- `forward_with_onepiece_streaming()`: 동일한 Hybrid 분기 추가 (프레임별)
- `forward_onepiece_single_frame()`: 동일한 Hybrid 분기 추가

**`train_onepiece.py`**
- 가중치 로딩: `cls_projection`을 exclude 목록에 추가 (차원 불일치 방지, scratch부터 학습)
- `_configure_parameters_phase2()`: `hybrid_fusion` 파라미터를 trainable로 설정, `teacher_model`은 frozen
- `_setup_optimizer()`: Phase 2에서 `fusion_params` 그룹 추가 (config `lr.fusion`으로 학습률 제어)
- `_set_train_mode()`: Phase 2에서 `hybrid_fusion`을 train mode로 유지

**`configs/onepiece/config_hybrid.yaml`** (신규)
- `model.vit_size: vits`, `hybrid_configs.use_hybrid: true`
- `load`: FlashDepth Hybrid 체크포인트 경로
- `lr.fusion: 1.0e-4` (HybridFusion 학습률)

### 모델별 CLS 소스 및 차원 정리
| 모델 | CLS 소스 | cls_projection | dpt_dim | SpatialMamba d_model |
|------|----------|---------------|---------|---------------------|
| O-Large | ViT-L (self) | Linear(1024→256) | 256 | 256 |
| O-Small | ViT-S (self) | Linear(384→64) | 64 | 64 |
| O-Hybrid | ViT-L (teacher) | Linear(1024→64) | 64 | 64 |

## 2026-04-14: ZoeDepth 로컬 캐시 우선 + rTC SCD 제외 메트릭 추가

### 변경 파일

**`adapters/zoedepth_adapter.py`**
- `load_model()`: `torch.hub.load`에서 `source='local'`로 로컬 캐시 우선 사용, 실패 시 네트워크 fallback
- 원인: Docker/tmux 환경에서 GitHub 504 타임아웃 발생

**`test_onepiece.py`**
- `_compute_excl_scd_metrics()` static method 추가: SCD reset 프레임이 포함된 pair를 제외한 rTC, flickering count 계산
  - Reset frame t → pair index t-1 (frames t-1→t), t (frames t→t+1) 제외
- tc-only 모드와 normal 모드 양쪽에서 excl_scd 메트릭 계산 및 저장
- `metric_order`에 `rtc_excl_scd`, `rtc_gt_excl_scd` 추가
- `_save_temporal_consistency()`: per_sequence, aggregated에 excl_scd 데이터 포함
- `_save_tc_summary()`: per_sequence, aggregated에 excl_scd 데이터 포함

**`utils/temporal_consistency.py`**
- `save_multi_threshold_json()`: per_sequence에 `thr_X.XX_excl_scd` 항목 추가, `aggregate_excl_scd` 섹션 추가

## 2026-03-30: CLS-guided Metric Head — Mamba에 CLS token prepend + CLSMetricHead

### 배경
- 기존 V3: SpatialMamba 출력(downsampled DPT feature)이 DPT temporal alignment과 Metric Head 입력 두 역할을 동시에 수행 → gradient conflict 우려
- 해결: DINOv2 CLS token을 Mamba input에 prepend하여 temporal context를 부여한 후, CLS는 Metric Head로, DPT는 alignment으로 역할 분리

### 변경 파일

**`flashdepth/onepiece_modules.py`**
- `SpatialMamba.forward()`, `forward_single_frame()`: `cls_projected` 파라미터 추가
  - CLS를 DPT 앞에 prepend → Mamba 통과 → split하여 `cls_output` 반환
  - CLS 미전달 시 기존 동작 유지 (legacy 호환)
- `CLSMetricHead` 클래스 추가: MLP(256→64→2) 기반 scale/shift 예측
  - ConvMetricHead(Conv 기반, spatial feature 입력) 대체
  - 동일한 초기화: softplus(0.5413)≈1.0, sigmoid(-5)≈0.0

**`flashdepth/model.py`**
- `__init__`: `cls_projection = Linear(1024, 256)` 추가, `ConvMetricHead` → `CLSMetricHead`
  - `cls_layer_indices = [2, 3]` (ViT layers 17, 23 평균 = fused CLS)
- `forward_with_onepiece()`:
  - Step 1에서 `_get_intermediate_layers_with_cls(cls_layer_indices=[2,3])` 사용
  - Phase 1: CLS는 Mamba bypass (frozen이라 의미 없으므로), projection → MetricHead 직접
  - Phase 2: CLS가 Mamba 통과 후 MetricHead로
- `forward_with_onepiece_streaming()`: fused CLS 사용 (SCD + Mamba 모두)
- `forward_onepiece_single_frame()`: 동일 패턴

**`train_onepiece.py`**
- `_configure_parameters_phase1()`: `cls_projection` trainable 추가
- `_configure_parameters_phase2()`: `cls_projection`을 onepiece param group에 포함
- `_set_train_mode()`: `cls_projection` train mode 유지
- Phase 2 optimizer param group: `cls_projection` → onepiece group (base_lr)

### 설계 결정
- **CLS prepend (앞에 붙이기)**: causal Mamba에서 CLS가 현재 프레임 DPT에 오염되지 않음. 이전 프레임 hidden state를 통한 temporal context만 수신
- **CLS projection 외부 배치**: Phase 1에서 SpatialMamba가 `torch.no_grad()`로 감싸지므로, projection을 FlashDepth 모듈로 분리하여 gradient flow 보장
- **Phase 1 bypass**: Mamba가 frozen + 랜덤 초기화 상태라 CLS 통과 무의미 → 직접 MetricHead로
- **SCD도 fused CLS 사용**: 코드 단순화, 추출 1회로 통일

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
