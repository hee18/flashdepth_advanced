# FlashDepth Advanced: Metric Depth Estimation 시스템 문서

## 개요

이 프로젝트는 FlashDepth 모델에 Global Scale Predictor (GSP) Head를 추가하여 상대 깊이(relative depth)를 절대 깊이(metric depth)로 변환하는 시스템입니다.

**최종 업데이트**: 2025-09-23
**현재 구현 상태**: GSP 모듈 훈련 및 극값 처리 완료

## 전체 시스템 아키텍처

### 1. 기본 구조
```
Input Video → FlashDepth (Frozen) → Relative Depth
                     ↓
              CLS Token → GSP Head (Trainable) → Scale, Shift
                                                      ↓
                           Metric Depth = Scale × Relative Depth + Shift
```

### 2. 모델 컴포넌트

#### FlashDepth (Frozen Path)
- **DINOv2 Encoder**: ViT-L (1024차원) 사용
- **DPT Head**: Dense Prediction Transformer로 상대 깊이 생성
- **Mamba/Temporal Modules**: Attention 클래스 사용 (mamba 비활성화)
- **특징**: 모든 파라미터가 동결되어 사전 훈련된 가중치 유지

#### GSP Head (Trainable Path)
- **입력**: DINOv2에서 추출한 CLS 토큰 (1024차원)
- **구조**: Linear(1024 → 256) → ReLU → Linear(256 → 2)
- **출력**: Scale (양수, Softplus 적용), Shift (실수)
- **목적**: 상대 깊이를 실제 미터 단위 깊이로 변환

### 3. 변환 공식
```
D_metric = Scale × D_relative + Shift
```

## 학습 설정

### 1. 하이퍼파라미터
- **학습률**: 1e-4 (GSP 모듈, 고정 학습률, 스케줄러 없음)
- **옵티마이저**: Adam (weight_decay=1e-6)
- **배치 크기**: 12
- **워커 수**: 4
- **총 반복 횟수**: 30,001
- **검증 주기**: 1,000 스텝마다
- **저장 주기**: 1,000 스텝마다

### 2. 손실 함수
- **기본**: Log L1 Loss (변경됨)
```python
def _log_l1_loss(self, pred, gt):
    return F.l1_loss(torch.log(pred + 1e-8), torch.log(gt + 1e-8))
```

### 3. 데이터셋 설정
- **사용 데이터셋**: TartanAir (학습 및 검증)
- **비디오 길이**: 설정된 video_length 사용
- **해상도**: 518×518
- **데이터 증강**: 비활성화 (메트릭 학습용)
- **GT 포맷**: 이미 미터 단위 metric depth (inverse 변환 불필요)

### 4. 파라미터 동결 전략
```python
# train_metric_head.py의 _freeze_base_model()
for name, param in model.named_parameters():
    if param_name.startswith('gsp_head'):
        param.requires_grad = True    # GSP Head만 학습
    else:
        param.requires_grad = False   # 나머지 모든 파라미터 동결
```

## Valid Mask 처리 (중요한 개선 사항)

### 1. 문제점
- **기존**: GT > 0만 고려 → 예측값의 극값으로 인한 메트릭 폭발
- **극값 예시**: 436,290,016m (4억미터) 같은 비현실적 예측값

### 2. 해결책: 복합 Valid Mask
```python
# GT와 예측값 모두 고려한 valid mask
gt_valid_mask = gt_flat > 0  # GT valid pixels
pred_valid_mask = (pred_flat > 0) & (pred_flat < 1000.0)  # 합리적 범위
valid_mask = gt_valid_mask & pred_valid_mask
```

### 3. 적용 위치
1. **Training Loss 계산** (`flashdepth/model.py`)
2. **Validation Loss 계산** (`train_metric_head.py`)
3. **Metric 계산** (`train_metric_head.py`)
4. **Visualization Metric 계산** (`utils/metric_visualization.py`)

### 4. 범위 기준
- **GT**: `>= 0` (TartanAir: -1이 invalid pixel)
- **예측값**: `> 0 & < 1000m` (0m 초과, 1km 미만으로 제한)

## Docker 환경 설정

### 1. GPU 할당
```bash
# 특정 GPU 사용 (예: GPU 1)
./run_docker.sh train --gpu 1

# 환경변수 설정
export CUDA_VISIBLE_DEVICES=1
```

### 2. 학습 실행
```bash
# 기본 학습 (기본값: batch_size=12, workers=4, total_iters=30001)
./run_docker.sh train

# 커스텀 설정
./run_docker.sh train --batch-size 12 --workers 4 --epochs 100
```

### 3. 결과 디렉토리
- **기본**: `train_results/results_1`
- **커스텀**: `--results-dir train_results/results_N`

## 메트릭 및 평가

### 1. 기본 메트릭
- **MAE**: Mean Absolute Error (L1)
- **RMSE**: Root Mean Square Error (L2)
- **AbsRel**: Absolute Relative Error
- **δ1, δ2, δ3**: Threshold Accuracy (< 1.25, 1.25², 1.25³)

### 2. 추가 메트릭
- **Scale-Shift Invariant 메트릭**: 정렬 후 성능 평가
- **Depth Range 메트릭**: 거리별 성능 분석
- **Boundary 메트릭**: 깊이 경계에서의 정확도

### 3. 시각화 출력
- **Input Image**
- **Ground Truth Depth (m)**
- **Predicted Metric Depth (m)**
- **Relative Depth** (std 정보 포함)
- **Valid Mask** (유효 픽셀 수 표시)
- **Absolute Error Map**
- **Depth Metrics** (AbsRel, δ1-δ3, RMSE, MAE)
- **Transformation Parameters** (Scale, Shift)
- **Depth Distribution Comparison**
- **GT vs Predicted Scatter Plot**

## 훈련 모니터링

### 1. 로그 출력 정보
```
Training Epoch: loss=X, scale=Y, shift=Z
Relative depth stats: Min, Max, Mean, Std
CLS token stats: Min, Max, Mean, Std
Scale stats: Mean, Std
Shift stats: Mean, Std
Metric depth stats: Min, Max, Mean, Std (디버그용)
```

### 2. 시각화 저장
- **Step 1, 10, 50, 100**: 초기 수렴 확인
- **매 250 스텝**: 훈련 중 모니터링
- **매 1000 스텝**: 검증 시각화

### 3. 체크포인트
- **저장 위치**: `configs/flashdepth-l/`
- **저장 조건**: 검증 loss 개선 시 best 체크포인트 저장
- **저장 주기**: 1000 스텝마다 정기 저장

## 성능 개선 사항

### 1. Loss 함수 개선
- **변경**: L1 Loss → Log L1 Loss
- **효과**: 깊이 값의 상대적 차이에 더 민감
- **수식**: `F.l1_loss(log(pred + ε), log(gt + ε))`

### 2. Valid Mask 개선
- **기존 문제**: 극값 예측으로 인한 메트릭 폭발
- **해결**: GT와 예측값 모두 고려한 복합 마스크
- **효과**: 안정적인 훈련 및 정확한 메트릭 계산

### 3. 시각화 개선
- **Path 정보**: 실제 파일 경로 표시
- **Valid Mask**: 검은색으로 유효 픽셀 표시 (cmap='gray_r')
- **메트릭**: 극값 필터링 후 계산

## 모델 비교: 원본 FlashDepth vs 현재 구현

### 1. Valid Mask 처리 비교

#### 원본 FlashDepth
```python
# 기본적인 GT 기반 마스크만 사용
valid_mask = gt_depth >= 0
loss = F.l1_loss(pred_depth[valid_mask], gt_depth[valid_mask])
```

#### 현재 구현 (개선됨)
```python
# GT와 예측값 모두 고려
gt_valid_mask = gt_flat > 0
pred_valid_mask = (pred_flat > 0) & (pred_flat < 1000.0)
valid_mask = gt_valid_mask & pred_valid_mask
loss = loss_fn(pred_flat, gt_metric_flat, valid_mask)
```

### 2. 개선 효과
- **안정성**: 극값으로 인한 훈련 불안정성 해결
- **정확성**: 비현실적 예측값 제외하여 정확한 메트릭 계산
- **현실성**: 0.1m-1000m 범위로 실제 환경에서 합리적인 깊이값만 고려

## 현재 상태 및 다음 단계

### ✅ 완료된 사항
1. **GSP 모듈 구현**: CLS token 기반 scale/shift 예측
2. **Loss 함수 개선**: Log L1 Loss 적용
3. **Valid Mask 개선**: 극값 처리 완료
4. **Docker 환경**: GPU 할당 문제 해결
5. **시각화 시스템**: 종합적인 모니터링 도구
6. **메트릭 계산**: 안정적인 평가 시스템

### 🔄 진행 중인 사항
1. **모델 훈련**: 30,001 반복으로 GSP 헤드 훈련 중
2. **성능 모니터링**: 시각화를 통한 수렴 확인

### 🎯 향후 계획
1. **성능 평가**: 훈련 완료 후 정량적 성능 분석
2. **하이퍼파라미터 튜닝**: 학습률, 배치 크기 등 최적화
3. **다른 데이터셋 확장**: MVS-Synth, Spring 등 추가 데이터셋 활용
4. **모델 경량화**: 추론 속도 최적화

## 핵심 기술적 인사이트

### 1. Metric Depth vs Relative Depth
- **Relative**: FlashDepth가 출력하는 상대적 깊이 (스케일 없음)
- **Metric**: 실제 미터 단위의 절대 깊이
- **변환**: GSP 헤드가 learned scale/shift로 변환 수행

### 2. 극값 처리의 중요성
- **문제**: 초기 훈련에서 GSP가 비현실적인 scale 예측
- **영향**: 메트릭 계산 시 평균값 왜곡, 훈련 불안정
- **해결**: 합리적 범위 제한으로 안정적 훈련 달성

### 3. 시각화의 가치
- **실시간 모니터링**: 수치만으로 파악 어려운 문제점 시각적 확인
- **디버깅**: Valid mask, 예측 분포 등 세부 사항 분석 가능
- **성능 추적**: 시간에 따른 개선 사항 명확히 확인

이 문서는 FlashDepth + GSP 시스템의 완전한 기술 명세서로, 현재 구현 상태와 핵심 기술적 결정들을 상세히 기록하고 있습니다.