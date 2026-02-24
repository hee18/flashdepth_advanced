# Onepiece V2 Changelog

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
