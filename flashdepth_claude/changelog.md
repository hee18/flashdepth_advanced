# Onepiece V2 Changelog

## 2026-02-27: Valid-Aware GT Downsampling + GPU 선행 다운샘플 (4개 test script)

### 변경 목적
- GT depth를 pred 해상도로 bilinear downsample하면 invalid pixel(-1.0)이 인접 valid pixel과 혼합되어 fake positive 값 생성
- 예: valid=3.5m + invalid=-1.0m → bilinear=1.25m (gt>0 체크 통과하지만 완전히 틀린 값)
- ETH3D처럼 invalid pixel이 많고 downsample ratio가 큰 데이터셋에서 AbsRel 심각한 성능 저하
- 고해상도 GT를 풀해상도로 CPU 전송 시 불필요한 메모리 부하 (ETH3D 6048×4032 × T frames)

### test_onepiece.py, test_comparison.py, test_video_comparison.py
- **GT downsample**: `mode='bilinear'` → valid-aware area downsample로 교체
  - Invalid pixel을 0으로 마스킹 후 `mode='area'` interpolate
  - Valid ratio로 보정 (`gt_down / valid_ratio`)
  - 50% 미만 valid인 pixel은 invalid(0.0)으로 표시 → 이후 `gt > 0` 체크에서 자동 제외
- **GPU 선행 다운샘플**: `.cpu()` 전에 GPU에서 다운샘플 수행
  - 기존: GPU 풀해상도 → `.cpu()` → CPU 다운샘플
  - 변경: GPU 풀해상도 → GPU 다운샘플 → `.cpu()` (pred 해상도만 전송)
- **Images downsample**: bilinear 유지 (이미지는 invalid pixel 없음)

### test_gear5.py
- **GT downsample**: valid-aware area downsample 동일 적용
- **GPU 선행 불필요**: GT가 batch에서 CPU로 직접 로드됨 (`.to(device)` 없음), GPU 전송 이슈 없음
- `del gt_depth_inverse_100` 추가: metric 변환 후 풀해상도 inverse depth 즉시 해제

## 2026-02-27: TC/EA 모드 temporal 메트릭 처리 통일 (4개 test script)

### 변경 목적
- `--test-mode tc`: temporal 메트릭만 계산 (rTC + TAE + PSR), accuracy 스킵
- `--test-mode ea`: accuracy 메트릭만 계산, temporal 전부 스킵 (rTC + TAE + PSR)
- 기존: tc는 rTC만, ea는 rTC만 스킵 → 불일치

### test_onepiece.py, test_gear5.py, test_comparison.py, test_video_comparison.py
- **TC 모드**: early-return 블록에 TAE + PSR 계산 추가 (rTC는 기존 유지)
  - TAE: `reproj_tae_calculator.compute_tae()` 호출 (dataset 지원 시)
  - PSR: lightweight per-frame scale ratio loop (`per_frame_scale_ratios_tc`)
- **EA 모드**: TAE, PSR 블록에 `if self.test_mode == 'ea':` 가드 추가 → 0.0 설정
- **run_docker.sh**: `--tc-threshold` 옵션 추가 (test_onepiece, test_gear5, test_gear5_bankai, test_original_flashdepth에 전달)

## 2026-02-27: PSR (Prediction Stability Ratio) 메트릭 추가

### test_onepiece.py
- **파일**: `test_onepiece.py`
- **목적**: Onepiece 테스트의 프레임간 스케일 안정성 정량화
- **변경 사항**:
  - Per-frame scale ratio (`r_t = mean(pred) / mean(gt)`) 계산 추가 (valid mask 기준, per-frame loop 내)
  - PSR 블록: `psr = mean(|r_t - r_{t-1}|)`, `psr_max = max(|r_t - r_{t-1}|)` (rTC 블록 뒤)
  - `metric_order` 2곳에 `psr`, `psr_max` 추가 (rtc_gt 뒤)
  - `_per_frame_psr`, `_per_frame_scale_ratio` 상세 데이터 저장 (`_` prefix → JSON에만 포함)

### test_gear5.py
- **파일**: `test_gear5.py`
- **목적**: Gear5 테스트의 프레임간 스케일 안정성 정량화
- **변경 사항**:
  - Per-frame scale ratio (`r_t = mean(pred) / mean(gt)`) 계산 추가 (valid mask 기준, per-frame loop 내)
  - PSR 블록: `psr = mean(|r_t - r_{t-1}|)`, `psr_max = max(|r_t - r_{t-1}|)` (rTC 블록 뒤)
  - `metric_order` 2곳에 `psr`, `psr_max` 추가 (rtc_gt 뒤)
  - TC-only mode 초기 metrics dict에 `psr`, `psr_max` 기본값 추가
  - `_per_frame_psr`, `_per_frame_scale_ratio` 상세 데이터 저장 (`_` prefix → JSON에만 포함)

### test_comparison.py
- **파일**: `test_comparison.py`
- **목적**: 이미지 depth 예측의 프레임간 스케일 안정성 정량화
- **변경 사항**:
  - Per-frame scale ratio (`r_t = mean(pred) / mean(gt)`) 계산 추가 (valid mask 기준, per-frame loop 내)
  - PSR 블록: `psr = mean(|r_t - r_{t-1}|)`, `psr_max = max(|r_t - r_{t-1}|)` (rTC 블록 뒤)
  - `metric_order`에 `psr`, `psr_max` 추가 (rtc_gt 뒤)
  - 평균 메트릭 키 리스트에 `psr`, `psr_max` 추가
  - TC-only mode 초기 metrics dict에 `psr`, `psr_max` 기본값 추가
  - `_per_frame_psr`, `_per_frame_scale_ratio` 상세 데이터 저장 (`_` prefix → JSON에만 포함)

### test_video_comparison.py
- **파일**: `test_video_comparison.py`
- **목적**: 비디오 depth 예측의 프레임간 스케일 안정성 정량화
- **변경 사항**:
  - Per-frame scale ratio (`r_t = mean(pred) / mean(gt)`) 계산 추가 (valid mask 기준)
  - PSR 블록: `psr = mean(|r_t - r_{t-1}|)`, `psr_max = max(|r_t - r_{t-1}|)`
  - `metric_order`에 `psr`, `psr_max` 추가 (rtc_gt 뒤)
  - 평균 메트릭 키 리스트에 `psr`, `psr_max` 추가
  - `_per_frame_psr`, `_per_frame_scale_ratio` 상세 데이터 저장 (`_` prefix → JSON에만 포함)

---

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
