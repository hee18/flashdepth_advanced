# Onepiece V2 Changelog

## 2026-02-26: Per-Channel Feature Analysis 추가

### analyze_dpt_features.py 확장
- **파일**: `analyze_dpt_features.py`
- **목적**: flickering frame 분석 시 개별 채널의 기여도를 정량적으로 분석, FiLM vs spatial modulation 연구 방향 결정 지원
- **변경 사항**:
  - `compute_affine_alignment()`: per-channel R² `(C,)`, per-channel L2 distance `(C,)` 반환값 추가
  - `analyze_film_validity()`: per_channel_analysis, top_contributing_channels 결과 키 추가. Block 6 중복 compute_affine_alignment 호출 제거 → Block 4 결과 재사용
  - `visualize_part_b()`: 3개 시각화 추가
    - `per_channel_l2_ranking.png`: Top-20 채널 × flicker frames heatmap (Post-Mamba L2 distance)
    - `affine_params_distribution.png`: gamma/beta 분포 scatter (top-5 극단값 annotate, max 6 subplots)
    - `per_channel_r2.png`: Top-20 최저 R² 채널 grouped bar chart
  - `write_summary()`: JSON에 per_channel_analysis (top-30 채널) 추가, summary.txt에 Part C 섹션 추가 (top-5 L2, top-5 lowest R², extreme gamma/beta)

---

## 2026-02-25: DPT Feature Flickering Analysis Script

### analyze_dpt_features.py 추가
- **파일**: `analyze_dpt_features.py` (신규), `run_docker.sh`
- **목적**: Depth flickering이 FlashDepth 파이프라인 어디서 발생하는지 분석, FiLM modulation 타당성 정량적 검증
- **구현**:
  - Monkey-patch `dpt_features_to_mamba` → Pre/Post-Mamba feature 캡처 (원본 코드 수정 없음)
  - Part A: Temporal stability (frame-to-frame cosine sim, Mamba effect per frame)
  - Part B: FiLM validity (channel-wise affine alignment R_affine, channel stats drift, variance decomposition)
  - 자동 flickering frame 감지 (MAD-based outlier) + 수동 override (`--flicker-frames`)
  - Per-flicker-frame heatmaps (pixel-wise cosine sim, affine residual, depth context strip)
- **Docker**: `./run_docker.sh analyze_features --config l --dataset sintel --seq 0 --gpu 0`
- **CLI 옵션**: `--flicker-threshold`, `--flicker-frames`, `--dataset`, `--seq`, `--config`

---

## 2026-02-25: Validation OFC Loss 추가

### Validation에 OFC Loss 포함
- **파일**: `train_onepiece.py` (`validate()`)
- **문제**: Phase 2 training loss = LogL1 + TGM + OFC인데, validation loss = LogL1 + TGM만 계산 → best checkpoint 선택 시 temporal feature consistency 미반영
- **변경**: Phase 2 validation에서 `self.loss_fn.ofc_loss()` 호출하여 OFC loss 계산, `ofc_weight(0.01)` 곱해서 val_loss에 합산
- **추가 항목**: per-dataset OFC 로깅, wandb `val/ofc_loss`, val_loss_dict에 `ofc_loss` 필드, return dict에 `ofc_loss`
- Phase 1에서는 기존과 동일 (LogL1 + TGM만)

---

## 2026-02-24: V2 Loss / Training 개편

### 1. SSIL 제거
- **파일**: `utils/onepiece_losses.py`, `flashdepth/model.py`, `train_onepiece.py`, `configs/onepiece/config.yaml`
- output_conv ~330K params로 SSIL을 제대로 학습하기엔 부족하고, SSIL 발산이 scale 폭발의 직접 원인
- `ScaleAndShiftInvariantLoss` import 및 관련 코드 전부 삭제

### 2. LogL1 Full Graph 전환
- **파일**: `flashdepth/model.py`, `utils/onepiece_losses.py`, `train_onepiece.py`
- **변경 전**: `modulated.detach()` → MetricHead만 gradient 수신
- **변경 후**: full graph `metric_depth`로 LogL1 계산 → 모든 trainable 모듈에 gradient
- `metric_depth_isolated`, `relative_depth_isolated` 출력 삭제

### 3. Scale Cap 1000
- **파일**: `flashdepth/onepiece_modules.py` (`OnepieceMetricHead.forward`)
- `F.softplus(raw_scale)` → `F.softplus(raw_scale).clamp(max=1000.0)`

### 4. WFC → OFC 이름 변경
- **파일**: losses, training, config, visualization, docs 전체
- `WarpFeatureConsistencyLoss` → `OpticalFlowConsistencyLoss`
- 키: `wfc_loss` → `ofc_loss`, `wfc_weight` → `ofc_weight`

### 5. Phase 1 기간 단축: 5000 → 1500 steps
- **파일**: `configs/onepiece/config.yaml`
- Scale 초기값 ~100이 적절하므로 MetricHead range 안정화에 1500 step이면 충분

### 6. Visualization 레이아웃 변경
- **파일**: `utils/onepiece_visualization.py`
- 메트릭 2개씩 한 줄: AbsRel+Delta_1, Delta_2+Delta_3, RMSE+MAE, TGM+OFC

### Loss 체계 변경 요약
```
Before: Phase 1 = LogL1(isolated) + TGM     |  Phase 2 = LogL1(isolated) + TGM + WFC + SSIL
After:  Phase 1 = LogL1(full) + TGM         |  Phase 2 = LogL1(full) + TGM + OFC
```
