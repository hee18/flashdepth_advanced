# FlashDepth Advanced: Metric Depth Estimation 시스템 문서

## 개요

이 프로젝트는 FlashDepth 모델에 Global Scale Predictor (GSP) Head를 추가하여 상대 깊이(relative depth)를 절대 깊이(metric depth)로 변환하는 시스템입니다.

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
- **DINOv2 Encoder**: ViT-S (384차원) 또는 ViT-L (1024차원)
- **DPT Head**: Dense Prediction Transformer로 상대 깊이 생성
- **Mamba/Temporal Modules**: 비디오 시퀀스 처리 (선택적)
- **특징**: 모든 파라미터가 동결되어 사전 훈련된 가중치 유지

#### GSP Head (Trainable Path)
- **입력**: DINOv2에서 추출한 CLS 토큰 (384 또는 1024차원)
- **구조**: Linear(입력차원 → 256) → ReLU → Linear(256 → 2)
- **출력**: Scale (양수, Softplus 적용), Shift (실수)
- **목적**: 상대 깊이를 실제 미터 단위 깊이로 변환

### 3. 변환 공식
```
D_metric = Scale × D_relative + Shift
```

## 🚨 발견된 문제점들

### 1. **Relative Depth 변환 문제** (중요!)
- **현재 상황**: FlashDepth DPT는 ReLU를 사용하므로 양수 출력이어야 함
- **관찰된 문제**: 시각화에서 -0.1~0.1 범위의 음수 값 출력
- **의심되는 원인**:
  - FlashDepth가 disparity-like 출력을 할 경우 역수 변환 필요 가능성
  - 정규화 과정에서의 스케일링 문제
  - 사전 훈련된 가중치와 현재 설정 간 불일치

### 2. **Dataset 이름 반복 표시**
- **문제**: 시각화에서 'tartanair', 'tartanair', ... 반복
- **원인**: dataset_name이 리스트 형태로 전달되어 직접 출력
- **해결 필요**: 첫 번째 요소만 표시하도록 수정

### 3. **GT Depth 범위 제한**
- **관찰**: 0.1-0.275m (10-27.5cm)로 매우 제한적
- **예상**: TartanAir는 더 넓은 깊이 범위를 가져야 함
- **확인 필요**: 데이터 로딩 및 전처리 과정

## 학습 구조

### 1. 파라미터 동결 전략
```python
# train_metric_head.py의 _freeze_base_model()
for name, param in model.named_parameters():
    if param_name.startswith('gsp_head'):
        param.requires_grad = True    # GSP Head만 학습
    else:
        param.requires_grad = False   # 나머지 모든 파라미터 동결
```

### 2. 데이터셋 설정
- **사용 데이터셋**: TartanAir (학습 및 검증)
- **배치 크기**: 4 (기본값)
- **비디오 길이**: 3 프레임
- **해상도**: 518×518
- **데이터 증강**: 비활성화 (메트릭 학습용)

### 3. 학습 과정
1. **Stage 1**: FlashDepth-L 기반 GSP Head 훈련
2. **손실 함수**: L1 Loss (MAE) 기본 사용
3. **최적화**: Adam optimizer
4. **체크포인트**: 500 스텝마다 저장
5. **시각화**: 500 스텝마다 검증 샘플 시각화

### 4. 학습 흐름
```
TartanAir Video → FlashDepth (Frozen) → Relative Depth
                              ↓
                       CLS Token → GSP Head → Scale, Shift
                                               ↓
                          Predicted Metric Depth
                                  ↓
                     L1 Loss vs GT Metric Depth
```

## 테스트 구조

### 현재 테스트 항목 (8개)

1. **GSP Head 기본 테스트** ✅
   - GSP Head 초기화 및 forward pass
   - Scale 양수 확인, 출력 형태 검증

2. **Metric Depth 변환 테스트** ✅
   - 변환 공식 정확성 검증
   - 배치 차원 처리 확인

3. **FlashDepth 통합 테스트** ✅
   - 전체 모델 통합 확인
   - CLS 토큰 추출 검증
   - forward_with_metric_head 메서드 테스트

4. **손실 함수 테스트** ✅
   - L1, L2 손실 계산 정확성
   - Valid mask 처리 확인

5. **메트릭 계산 테스트** ✅
   - MAE, RMSE, AbsRel, δ1 등 계산
   - 단일 프레임 성능 평가

6. **파라미터 동결 테스트** ✅
   - 추론 모드에서 모든 파라미터 동결 확인
   - GSP Head 파라미터 수 검증

7. **시각화 테스트** ✅
   - 깊이 맵 시각화 생성
   - 시퀀스 비교 시각화 확인

8. **Comprehensive Integration 테스트** ❌
   - **문제**: 복잡한 메트릭 계산에서 딕셔너리 연산 오류
   - **현재 상태**: TartanAir 데이터를 사용한 end-to-end 테스트로 수정 시도
   - **중요성**: 실제 사용 환경 시뮬레이션

### 테스트 결과 분석

#### 성공한 기능들
- GSP Head 기본 동작 완전 검증
- 수학적 변환 공식 정확성 확인
- 전체 아키텍처 통합 성공
- 개별 컴포넌트 안정성 확인

#### 미해결 문제들
- Comprehensive Integration의 복잡한 메트릭 계산 오류
- 실제 비디오 데이터에서의 end-to-end 성능 미검증
- 다중 프레임 시간적 일관성 미확인

## Docker 환경 설정

### 1. 파일 동기화 문제 해결
```yaml
# docker-compose.yml 수정 사항
volumes:
  - ./train_metric_head.py:/app/train_metric_head.py
  - ./test_metric_head.py:/app/test_metric_head.py  # 추가됨
  - ./flashdepth:/app/flashdepth
  - ./utils:/app/utils
  - ./dataloaders:/app/dataloaders
```

### 2. 사용 방법
```bash
# 학습 실행
./run_docker.sh train

# 테스트 실행
./run_docker.sh test

# 대화형 셸
./run_docker.sh shell

# 정리
./run_docker.sh clean
```

## 핵심 발견 사항 및 해결 과제

### 1. **즉시 해결 필요** (Priority 1)

#### Relative Depth 변환 문제
- **문제**: FlashDepth의 상대 깊이가 예상과 다른 형태로 출력
- **현재**: -0.1~0.1 범위 (음수 포함)
- **예상**: 양수 값 (ReLU 때문에)
- **해결 방안**:
  1. FlashDepth의 정확한 출력 형태 분석
  2. Disparity vs Depth 변환 공식 재검토
  3. 정규화/스케일링 과정 확인

#### Dataset 시각화 문제
- **해결 방법**:
```python
# utils/metric_visualization.py 수정
dataset_name = dataset_name[0] if isinstance(dataset_name, list) else dataset_name
```

### 2. **중기 해결 과제** (Priority 2)

#### Comprehensive Integration 테스트
- **목표**: 실제 TartanAir 데이터로 end-to-end 검증
- **방법**: 복잡한 중첩 딕셔너리 계산 단순화
- **중요성**: 실제 사용 환경에서의 안정성 보장

#### 성능 최적화
- **메모리 사용량 최적화**
- **다중 GPU 효율성 개선**
- **배치 처리 안정성 향상**

### 3. **장기 개선 과제** (Priority 3)

#### 모델 성능 향상
- **더 정확한 스케일/시프트 예측**
- **시간적 일관성 개선**
- **다양한 데이터셋에서의 일반화 성능**

#### 평가 메트릭 확장
- **깊이 범위별 성능 분석**
- **경계 영역 정확도 평가**
- **시간적 안정성 메트릭**

## 사용자 검증 포인트

다음 사항들이 의도한 대로 구현되었는지 확인해 주세요:

### ✅ 올바르게 구현된 부분
1. **이중 경로 구조**: Frozen FlashDepth + Trainable GSP
2. **파라미터 동결**: Base 모델 완전 동결, GSP만 학습
3. **TartanAir 데이터 사용**: 실제 메트릭 GT로 학습
4. **Docker 환경**: 개발 및 실험 환경 구성

### ⚠️ 검토 필요한 부분
1. **변환 공식의 정확성**: Disparity-like vs Depth-like 출력 확인
2. **데이터 전처리**: GT 깊이 범위 및 정규화 과정
3. **시각화 결과 해석**: 현재 결과가 예상한 것과 일치하는지

### ❌ 수정 필요한 부분
1. **Relative Depth 음수 문제**: 근본 원인 분석 및 해결
2. **Comprehensive Integration**: 안정적인 end-to-end 테스트 구현
3. **성능 평가**: 실제 메트릭 성능 기준 달성 여부

## 결론

현재 시스템은 기본적인 GSP Head 기능은 완전히 구현되었지만, FlashDepth의 relative depth 특성과 관련된 중요한 문제가 발견되었습니다. 이 문제를 해결하지 않으면 실제 메트릭 깊이 추정 성능에 치명적인 영향을 줄 수 있으므로 최우선으로 해결해야 합니다.

특히 **FlashDepth가 disparity-like 출력을 하는지, depth-like 출력을 하는지**에 따라 변환 공식이 달라질 수 있으므로 이 부분에 대한 정확한 분석이 필요합니다.