# Onepiece V2 Changelog

## 2026-03-05: DepthPro half precision → TAE dtype mismatch 수정

### 변경 목적
- DepthPro adapter가 `precision=torch.half`로 모델을 생성하여 inference 출력이 float16
- reprojection TAE에서 intrinsics/poses(float32)와 matmul 시 `float != c10::Half` RuntimeError 발생
- DAv2 등 다른 모델은 float32 출력이라 문제 없었음

### 변경 파일
- **test_comparison.py**: `pred_depths_cpu` 생성 시 `.float()` 캐스팅 추가 → half precision 모델에서도 TAE 정상 계산

---

## 2026-03-04: tc_summary.json 추가 + ComparisonDataset image_paths 수정

### 변경 목적
- TC 모드 결과(rTC, TAE, PSR)를 하나의 간결한 JSON 파일로 통합 저장
- ComparisonDataset에서 `image_paths` 누락으로 TAE=0 출력되던 버그 수정

### 변경 파일
- **test_onepiece.py, test_gear5.py, test_comparison.py, test_video_comparison.py**: `_save_tc_summary()` 메서드 추가. TC-only 모드에서 `temporal_consistency.json`과 함께 `tc_summary.json` 저장. 데이터셋 aggregate + per-sequence 결과 포함.
- **dataloaders/comparison_dataset.py**: `frame_data`에 `image_path` 추가, batch dict에 `image_paths` 리스트 포함, collate 함수 업데이트 → TAE 계산 가능하도록 수정.

---

## 2026-03-04: Dual-Resolution 메트릭 계산 전체 확장 (test_gear5, test_comparison, test_video_comparison)

### 변경 목적
- 2026-03-03에 test_onepiece.py에 적용한 GT 해상도 메트릭 계산 패턴을 나머지 3개 test script에 동일 적용
- Sparse GT(eth3d, waymo_seg) downsample 시 valid pixel 소실 문제를 전체 테스트에서 해결
- 업계 표준(KITTI, Depth Anything V2, MiDaS): pred를 GT 해상도로 upsample하여 메트릭 계산

### test_gear5.py
- **GT downsample 제거**: valid-aware area downsample → GT 원본 해상도 유지
- **`gt_at_pred_res_cpu`**: TC/시각화/PSR용 GT를 pred 해상도로 별도 생성 (sparse: nearest, dense: bilinear)
- **Per-frame pred upsample**: 메트릭/depth range/TAE에서 per-frame `F.interpolate(pred, gt_size)` 적용
- **TAE at GT resolution**: pred를 GT 해상도로 upsample 후 TAE 계산, 사용 후 즉시 `del`
- **시각화/object-wise/fgwise**: `gt_at_pred_res_cpu` 사용으로 통일

### test_comparison.py
- **GT CPU only**: GT를 GPU에서 downsample하지 않고 바로 CPU 전송 → GPU 메모리 절약
- **Pred 즉시 CPU**: per-frame 추론 후 `pred_depth_t.cpu()`로 즉시 CPU 이동
- **Per-frame pred upsample**: regular/object-wise/fgwise 메트릭에서 GT 해상도 계산
- **TAE at GT resolution**: TC-mode와 full-mode 모두 적용
- **rTC/PSR/시각화**: `gt_at_pred_res_cpu` 사용

### test_video_comparison.py
- **즉시 메모리 해제**: `del images_unnorm` + `empty_cache()`, pred도 추론 직후 CPU 이동
- **GT CPU only**: test_comparison.py와 동일 패턴
- **Per-frame pred upsample**: regular/object-wise/fgwise 메트릭에서 GT 해상도 계산
- **TAE at GT resolution**: TC-mode와 full-mode 모두 적용
- **rTC/PSR/시각화**: `gt_at_pred_res_cpu` 사용

### 지표별 해상도 전략 (4개 test script 공통)
| 지표 | 해상도 | 이유 |
|------|--------|------|
| 표준 메트릭 (MAE, RMSE, AbsRel, δ1 등) | GT 해상도 | Sparse GT 보존 |
| TAE | GT 해상도 | GT depth를 직접 reprojection에 사용 |
| rTC | pred 해상도 | SEA-RAFT 내부 long edge 960 cap, pred-to-pred 비교 |
| PSR | pred 해상도 | pred 값 기반 scale ratio |
| 시각화 | pred 해상도 | 표시 목적 |

---

## 2026-03-04: Inverse Depth Training Mode 추가

### 변경 목적
- gear5처럼 1/m 공간에서 직접 scale/shift를 학습하는 inverse 모드 추가
- 기존 metric 모드(미터 공간)와 config로 전환 가능: `train_mode: "inverse"` or `"metric"`
- gear5의 *100 스케일링은 사용하지 않고 순수 1/m 공간에서 동작

### 변경 파일

**configs/onepiece/config.yaml, config_l.yaml, config_s.yaml**
- `train_mode: "metric"` 설정 추가 (default)

**flashdepth/onepiece_modules.py**
- `OnepieceMetricHead.__init__`: `train_mode` 파라미터 추가
- `_initialize_weights`: 모드별 초기화 (metric: scale≈100, inverse: scale≈1.0)
- `forward`: 모드별 shift 처리 (metric: sigmoid [0,1], inverse: unconstrained)

**flashdepth/model.py**
- `__init__`: `train_mode`를 모델에 저장, MetricHead에 전달
- `forward_with_onepiece` Step 7: inverse 분기 추가 (`pred_inverse = scale * relative_depth + shift`)
- `forward_with_onepiece_streaming` Step 8: 동일 분기
- `forward_onepiece_single_frame` Step 8: 동일 분기
- Inverse shift clamp: `min=-scale*(1/300), max=1.0`

**train_onepiece.py**
- `__init__`: `self.train_mode` 저장, 로깅 추가
- `_setup_model`: `model_config['train_mode']` 전달
- `_save_training_visualization`: inverse 분기 (1/m → meters 변환, valid mask 간소화)
- `train_step`: inverse 분기 (pred 체크 없는 valid mask, 직접 1/m 공간 loss)
- `validate`: inverse 분기 (gt>0 & pred>0 only, 70m 임계값 제거)
- Validation visualization: inverse 분기

**run_docker.sh**
- `TRAIN_MODE="metric"` 변수 추가
- `--train-mode` CLI 플래그 파싱
- `train_onepiece`, `train_onepiece_ddp` 섹션에 `train_mode=$TRAIN_MODE` 전달

### gear5 vs onepiece inverse 차이
- GT 공간: gear5=100/m, onepiece=1/m (*100 없음)
- Shift lower bound: gear5=-scale*(100/300), onepiece=-scale*(1/300)
- Shift upper bound: gear5=없음, onepiece=1.0
- TGM: gear5=TGMTemporalLoss(multi-scale), onepiece=단순 diff

## 2026-03-03: Dual-Resolution 메트릭 계산 + Sparse 감지 수정 + 시각화 개선 (test_onepiece.py)

### 변경 목적
- eth3d/waymo_seg에서 valid-aware GT downsampling이 sparse GT를 파괴하여 메트릭이 완전히 망가지는 문제 수정
- waymo: lidar ~5% density → valid_ratio < 0.5 threshold에 의해 거의 모든 GT pixel 무효화 → rmse==mae (1 valid pixel)
- eth3d: SfM sparse depth, 6048×4032 → 784×518 downsample에서 동일 문제
- sintel은 dense (~95% coverage)이라 영향 없었음

### test_onepiece.py

**1. Dual-Resolution 메트릭 계산 (lines 515-552)**
- 기존: GT를 pred 해상도로 valid-aware area downsample → sparse GT 파괴
- 변경: Original FlashDepth처럼 pred를 GT 해상도로 per-frame upsample하여 메트릭 계산
  - `pred_depths_cpu`: pred 해상도 (TC/TAE/visualization용)
  - `gt_depth_metric_cpu`: GT 원본 해상도 (메트릭 계산용)
  - `gt_at_pred_res_cpu`: GT를 pred 해상도로 downsample (TC/TAE/visualization용)
    - sparse datasets (eth3d, waymo_seg): nearest interpolation
    - dense datasets: bilinear interpolation

**2. Per-frame 메트릭 루프 (lines 698+)**
- `need_upsample` 체크 후 per-frame F.interpolate(pred, gt_size, bilinear, align_corners=True)
- GT 해상도에서 valid mask 생성 및 메트릭 계산
- OOM 방지: per-frame CPU 처리 (전체 시퀀스 upsample 대신)

**3. Depth range analysis (lines 807+)**
- 동일한 per-frame upsample 패턴 적용

**4. Sparse mode 감지 수정 (lines 1099-1100, 1209-1225)**
- 기존: `gt_density < 0.5` → dense dataset(sintel ambush_2 등)에서 sky가 많으면 잘못 sparse 판정
- 변경: `any(s in dataset_name.lower() for s in ['eth3d', 'waymo_seg'])` → dataset name 기반

**5. FPS 측정 방식 변경 (lines 467-497)**
- 기존: 별도 warmup 호출 + 전체 T frame 재실행 (Mamba state reset)
- 변경: gear5 스타일 단일 forward pass, FPS = (T - warmup_frames) / time

**6. Best/worst frame 시각화 개선 (lines 1267-1272)**
- Row 2: Scale/Shift temporal plot → GT Valid Mask (gear5 스타일)
- dataset_name 기반 sparse 감지 + proper pred_show_mask

**7. TC/TAE/Visualization 참조 변경**
- 모든 TC, TAE, visualization 호출에서 `gt_depth_metric_cpu` → `gt_at_pred_res_cpu`로 교체
- `gt_depth_metric_cpu`는 per-frame 메트릭 계산에서만 사용 (GT 해상도)

### docs/Onepiece.md
- Testing 섹션에 Test Resolutions, Dual-Resolution, Sparse Detection, FPS 측정 항목 추가

## 2026-03-03: GPU 메모리 최적화 - 시퀀스 간 캐시 정리 + 풀해상도 텐서 해제 (test_comparison, test_video_comparison)

### 변경 목적
- ETH3D 등 고해상도 데이터셋(4135×6205)에서 30프레임 처리 시 OOM 발생
- 원인: 시퀀스 간 `torch.cuda.empty_cache()` 미호출 → Reserved 메모리 누적 (Seq0: 14GB → Seq4: 50GB → OOM)
- GT/images downsample 후에도 원본 풀해상도 GPU 텐서가 해제되지 않음

### test_comparison.py
- **시퀀스 간 GPU 캐시 정리**: `test()` 루프에 `finally: torch.cuda.empty_cache()` 추가
- **풀해상도 텐서 해제**: GT downsample + CPU 전송 후 `del gt_depth_gpu, gt_depths, gt_depth_processed` + `empty_cache()`
- **images downsample 방식 변경**: `batch['images']`(CPU) 대신 `images`(GPU) 기준으로 F.interpolate → 원본 GPU 텐서 덮어쓰기 (test_onepiece 패턴)
- **CPU 재할당**: `pred_depths`, `gt_depth_processed`를 CPU 버전으로 재할당하여 이후 visualization/export에서 사용
- fgwise 코드에서 삭제된 `gt_depths` 참조를 `gt_depth_processed_cpu`로 수정

### test_video_comparison.py
- **동일 패턴 적용**: images GPU downsample, 풀해상도 텐서 해제, CPU 재할당
- **`del images_unnorm`**: 추론 완료 후 불필요한 GPU 텐서 즉시 해제
- **`empty_cache()` → `finally` 블록 이동**: 에러 발생 시에도 GPU 메모리 정리 보장

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
