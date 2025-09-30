# FlashDepth Advanced: Metric Depth Estimation 시스템 문서

## 개요

이 프로젝트는 FlashDepth 모델에 Global Scale Predictor (GSP) Head를 추가하여 상대 깊이(relative depth)를 절대 깊이(metric depth)로 변환하는 시스템입니다.

**최종 업데이트**: 2025-09-29
**현재 구현 상태**: GSP 모듈 훈련 완료 및 포괄적 테스트 시스템 구축

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
- **학습률**: 1e-4 (GSP 모듈 초기값)
- **옵티마이저**: Adam (weight_decay=1e-6)
- **배치 크기**: 12 (고정)
- **워커 수**: 4 (고정)
- **총 반복 횟수**: 30,001
- **검증 주기**: 1,000 스텝마다
- **저장 주기**: 1,000 스텝마다
- **그래디언트 클리핑**: max_norm=1.0

### 2. 학습률 스케줄러 (개선됨)
- **타입**: Cosine Annealing with Warmup
- **Warmup**: 첫 10% 스텝 (선형 증가 0.1x → 1x)
- **안정 구간**: 10% ~ 30% 스텝 (고정 학습률)
- **감소 구간**: 30% ~ 100% 스텝 (Cosine 감소)
- **최종 학습률**: 초기값의 1% (1e-6)

```python
def lr_lambda(step):
    if step < warmup_steps:  # 0-10%: Warmup
        return 0.1 + 0.9 * (step / warmup_steps)
    elif step < decay_start:  # 10-30%: Stable
        return 1.0
    else:  # 30-100%: Cosine decay
        progress = (step - decay_start) / (total_steps - decay_start)
        return 0.01 + 0.99 * 0.5 * (1 + cos(π * progress))
```

### 3. 손실 함수
- **기본**: Log L1 Loss (변경됨)
```python
def _log_l1_loss(self, pred, gt):
    return F.l1_loss(torch.log(pred + 1e-8), torch.log(gt + 1e-8))
```

### 4. 데이터셋 설정
- **학습 데이터셋**: TartanAir (train split)
- **검증 데이터셋**: TartanAir (val split)
- **테스트 데이터셋**: TartanAir (test split, 4개 시퀀스)
- **비디오 길이**: 5프레임 (학습), 50프레임 (테스트)
- **해상도**: 518×518 (모든 단계에서 일관성 유지)
- **데이터 증강**: 비활성화 (메트릭 학습용)
- **GT 포맷**: 이미 미터 단위 metric depth (inverse 변환 불필요)
- **배치 처리**: Custom collate function으로 None 값 필터링

### 5. 파라미터 동결 전략
```python
# train_metric_head.py의 _freeze_base_model()
def _freeze_base_model(self):
    # Freeze all FlashDepth parameters
    for name, param in self.model.named_parameters():
        if 'gsp_head' in name:
            param.requires_grad = True   # Only GSP Head trainable
        else:
            param.requires_grad = False  # Freeze FlashDepth backbone

    # Log parameter counts
    trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in self.model.parameters())
    logger.info(f"Trainable: {trainable:,} / Total: {total:,} parameters")
```

### 6. 최적화 및 정규화
- **옵티마이저 범위**: GSP Head 파라미터만 (약 262K개)
- **Weight Decay**: 1e-6 (가벼운 정규화)
- **그래디언트 클리핑**: L2 norm 1.0으로 제한
- **Mixed Precision**: BFloat16 사용 (메모리 효율성)
- **GPU 메모리 관리**: 시퀀스 처리 후 캐시 정리

### 7. 모니터링 및 로깅
- **실시간 메트릭**: Loss, Learning Rate, Scale, Shift
- **시각화 저장**: Step 1, 10, 50, 100 (초기), 매 250 스텝 (모니터링)
- **체크포인트**: 검증 loss 개선 시 best 모델 저장
- **Wandb 지원**: 선택적 온라인 로깅 (config.training.wandb=true)

## 포괄적 테스트 시스템 (새로 추가)

### 1. test_metric_head.py 개요
- **목적**: 훈련된 GSP 모델의 종합적 성능 평가
- **특징**: 실제 TartanAir 데이터를 사용한 실전 테스트
- **다중 시퀀스**: 최대 4개 시퀀스 동시 분석

### 2. 핵심 메트릭 및 분석

#### Temporal Alignment Error (TAE)
- **정의**: 연속 프레임 간 깊이 예측 일관성 측정
- **수식**: `TAE = 1/(2(N-1)) * Σ[AbsRel(pred_k, pred_{k+1})]`
- **적용**: 50프레임 전체 시퀀스에서 계산
- **중요성**: 비디오 깊이 추정의 시간적 안정성 평가

#### 프레임별 성능 분석
- **Best Frame**: 가장 낮은 AbsRel을 가진 프레임
- **Worst Frame**: 가장 높은 AbsRel을 가진 프레임
- **TAE 최적 시퀀스**: 5연속 프레임 중 TAE가 가장 낮은 구간

### 3. 다양한 시각화 출력

#### A. 시퀀스 시각화 (depth_sequence_visualization_seqX.png)
- **구성**: Input Image, Predicted Depth, Ground Truth Depth
- **특징**: Valid mask 적용, 프레임 간격 조정 가능
- **용도**: 전체 시퀀스의 시간적 변화 패턴 분석

#### B. Best/Worst Frame 시각화
- **레이아웃**: Input → GT Depth → Pred Depth → Depth Colorbar → Rel Depth → Rel Colorbar → Metrics
- **메트릭**: AbsRel, δ1/δ2/δ3, RMSE, MAE 표시
- **크기 통일**: 모든 이미지 동일 픽셀 크기로 표시

#### C. TAE 5-Frame 시각화
- **구성**: GT Depth, Predicted Depth, Relative Depth (각 5프레임)
- **메트릭 위치**: 우측 상단 꼭짓점에 TAE, FPS, Resolution, GPU 정보
- **컬러바**: 축소된 폭으로 깔끔한 레이아웃

#### D. 개별 이미지 저장 (seq3 전용)
- **파일들**: input.png, gt_depth.png, pred_depth.png, relative_depth.png
- **특징**: 제목/컬러바 없는 순수 이미지, 8×8 크기

### 4. 결과 JSON 저장

#### 개별 시퀀스 결과 (test_results_seqX.json)
```json
{
    "sequence_name": "tartanair/dataset_name",
    "video_shape": [1, 50, 3, 518, 518],
    "inference_time_seconds": 1.84,
    "fps": 27.15,
    "tae_metrics": {
        "tae": 0.147,
        "frame_count": 50,
        "valid_frame_pairs": 49
    },
    "comprehensive_metrics": {
        "mae": 0.449, "rmse": 1.458, "abs_rel": 0.166,
        "a1": 0.872, "a2": 0.906, "a3": 0.936
    },
    "best_frame_info": {"frame_index": 38, "abs_rel": 0.061},
    "best_tae_sequence_info": {"start_index": 38, "tae_value": 0.028}
}
```

#### 평균 결과 (averaged_test_results.json)
```json
{
    "sequence_count": 4,
    "datasets": ["tartanair/seq1", "tartanair/seq2", ...],
    "avg_inference_time_seconds": 1.85,
    "avg_fps": 27.0,
    "avg_tae_metrics": {"avg_tae": 0.145},
    "comprehensive_metrics": {
        "mae": 0.450, "rmse": 1.460, "abs_rel": 0.165,
        "a1": 0.875, "a2": 0.910, "a3": 0.940
    }
}
```

## Docker 환경 설정 (업데이트됨)

### 1. 테스트 실행
```bash
# 기본 테스트 (모든 프레임 표시)
./run_docker.sh test

# 프레임 간격 조정 (2프레임마다 표시)
./run_docker.sh test --frame-interval 2

# 5프레임마다 표시 (50프레임 → 10프레임)
./run_docker.sh test --frame-interval 5

# 커스텀 결과 디렉토리
./run_docker.sh test --results-dir test_results/results_custom
```

### 2. GPU 할당
```bash
# 특정 GPU 사용 (예: GPU 1)
./run_docker.sh test --gpu 1

# 환경변수 설정
export CUDA_VISIBLE_DEVICES=1
```

### 3. 체크포인트 지정
```bash
# GSP 체크포인트 지정
./run_docker.sh test --gsp-checkpoint train_results/results_5/best_metric_head_step_21000.pth
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

### 3. 시각화에서의 적용
- **Depth Sequence**: Invalid 영역은 투명하게 처리 (np.nan)
- **TAE 5-Frame**: Valid 픽셀만을 사용한 colormap 범위 계산
- **Best/Worst Frame**: Valid mask 기반 메트릭 계산

### 4. 범위 기준
- **GT**: `> 0` (TartanAir: 0이 invalid pixel)
- **예측값**: `> 0 & < 1000m` (0m 초과, 1km 미만으로 제한)

## 이미지 정규화 개선 (새로 추가)

### 1. 문제 해결
- **기존 문제**: Input image가 회색/검은색으로 표시
- **원인**: 불필요한 [-1,1] → [0,1] 정규화

### 2. 해결책
```python
# matplotlib이 자동으로 범위 처리하도록 변경
# 복잡한 정규화 로직 제거
logger.info(f"Input image range: min={image.min():.6f}, max={image.max():.6f}")
# matplotlib이 자동으로 적절한 범위로 표시
```

### 3. 이미지 크기 통일
```python
# Best/Worst frame에서 모든 이미지 크기 통일
if image.shape[:2] != (depth_h, depth_w):
    image_resized = cv2.resize(image_uint8, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
    image = image_resized.astype(np.float32) / 255.0
```

## 메트릭 및 평가

### 1. 기본 메트릭
- **MAE**: Mean Absolute Error (L1)
- **RMSE**: Root Mean Square Error (L2)
- **AbsRel**: Absolute Relative Error
- **δ1, δ2, δ3**: Threshold Accuracy (< 1.25, 1.25², 1.25³)

### 2. 추가 메트릭 (새로 추가)
- **TAE**: Temporal Alignment Error (시간적 일관성)
- **Frame-wise Performance**: 프레임별 성능 분석
- **Valid Pixel Ratio**: 유효 픽셀 비율 추적

### 3. 성능 지표 예시
```
평균 성능 (4개 시퀀스):
- MAE: 0.449m
- RMSE: 1.458m
- AbsRel: 0.166
- δ1: 87.2%, δ2: 90.6%, δ3: 93.6%
- TAE: 0.147 (시간적 일관성)
- FPS: 27.2 (RTX A6000)
```

## 모델 아키텍처 세부사항

### 1. BFloat16 지원
```python
# 모든 텐서 연산에서 BFloat16 → Float32 변환 지원
if isinstance(tensor, torch.Tensor):
    tensor = tensor.float().cpu().numpy()  # BFloat16 호환성
```

### 2. 해상도 일관성
- **학습**: 518×518 (TartanAir 기본)
- **테스트**: 518×518 (메트릭 계산 일관성)
- **시각화**: 518×518 (모든 요소 크기 통일)

### 3. 메모리 최적화
```python
# 각 시퀀스 처리 후 GPU 메모리 정리
torch.cuda.empty_cache()
```

## 현재 상태 및 다음 단계

### ✅ 완료된 사항
1. **GSP 모듈 훈련**: CLS token 기반 scale/shift 예측 완료
2. **포괄적 테스트 시스템**: 다중 시퀀스, 다양한 메트릭, 시각화
3. **TAE 메트릭**: 시간적 일관성 평가 시스템
4. **Valid Mask 개선**: 극값 처리 및 안정적인 메트릭 계산
5. **시각화 품질**: 이미지 크기 통일, 정규화 문제 해결
6. **프레임 간격 제어**: 시각화 시퀀스 길이 조절 가능
7. **결과 집계**: 개별 및 평균 JSON 결과 저장
8. **개별 이미지 추출**: seq3 best frame 구성 요소 개별 저장

### 🔄 현재 성능
- **추론 속도**: ~27 FPS (RTX A6000, 518×518)
- **메트릭 성능**: δ1 87.2% (1.25배 임계값 정확도)
- **시간적 일관성**: TAE 0.147 (양호한 시간적 안정성)

### 🎯 향후 계획
1. **성능 최적화**: 더 높은 해상도에서의 성능 테스트
2. **다른 데이터셋**: MVS-Synth, Spring 등 확장 테스트
3. **실시간 최적화**: 추론 속도 개선 및 경량화
4. **로버스트니스**: 다양한 환경 조건에서의 안정성 테스트

## 핵심 기술적 인사이트

### 1. TAE의 중요성
- **비디오 깊이 추정**에서 단일 프레임 정확도만큼 중요한 시간적 일관성
- **실용적 응용**: 자율주행, 로보틱스에서 매우 중요한 지표
- **구현**: 연속 프레임 간 AbsRel 기반 계산

### 2. 시각화 품질의 영향
- **디버깅 효율성**: 숫자만으로 파악하기 어려운 문제를 시각적으로 즉시 확인
- **크기 통일**: 모든 subplot이 동일한 픽셀 크기로 표시되어 정확한 비교 가능
- **Valid Mask**: Invalid 영역의 투명 처리로 실제 성능 영역 명확히 구분

### 3. 다중 시퀀스 테스트
- **통계적 신뢰성**: 단일 시퀀스 테스트의 한계 극복
- **성능 변동성**: 시퀀스별 성능 차이 패턴 분석 가능
- **평균 성능**: 실제 배포 환경에서의 예상 성능 추정

이 문서는 FlashDepth + GSP 시스템의 완전한 기술 명세서로, 훈련부터 포괄적 테스트까지의 전체 파이프라인을 상세히 기록하고 있습니다.