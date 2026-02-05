# 🚀 FlashDepth-Metric: 실시간 비디오 절대 깊이 추정 고도화 설계안

## 1. 아키텍처 개요 (Architecture Overview)
본 설계는 기존 FlashDepth의 Mamba 기반 시계열 정렬 능력을 재활용하여 연산 오버헤드를 최소화하고, 절대 깊이 추정을 위한 Metric Head와 고도화된 손실 함수를 결합합니다.

*   **Backbone:** DINOv2 (Frozen)
*   **Temporal Module:** 1/10 Spatial Mamba (FlashDepth 기본 구조 유지)
*   **Decoder:** DPT (Phase 2에서 Unfreeze)
*   **Metric Head:** Mamba Feature와 정제된 CLS 토큰을 결합하여 Scale/Shift 추정

---

## 2. 메트릭 헤드 고도화 (Advanced Metric Head)

### 2.1 입출력 구조
*   **입력 1 (Temporal Base):** Mamba를 통과한 특징 맵의 Global Average Pooling (GAP) [256-dim]
*   **입력 2 (Semantic Base):** **Temporal-Refined CLS Token** [1024-dim]
*   **결합:** `Concat(GAP, Refined-CLS)` → 1280-dim 벡터 생성
*   **출력:** Scale ($S$), Shift ($B$) 스칼라 값

### 2.2 CLS Cross-Attention (CA) Memory Bank
단일 프레임 CLS의 흔들림을 제어하기 위해 과거 문맥을 참조합니다.
*   **구조:** 현재 프레임($t$)의 CLS를 **Query**로, 과거 4프레임($t-1 \dots t-4$)의 CLS를 **Key/Value**로 사용하는 가벼운 Multi-Head Attention 적용.
*   **효과:** CLS 토큰에서 발생하는 미세한 노이즈를 과거 프레임의 문맥으로 필터링하여 안정적인 Scale 추정 가능.

---

## 3. 학습 전략 (Training Strategy)

### Phase 1: Metric Alignment
*   **대상:** Metric Head만 Unfreeze.
*   **목표:** 고정된 Relative Depth 상에서 데이터셋의 절대 깊이 범위를 빠르게 학습.

### Phase 2: Full Video Optimization
*   **대상:** DPT Decoder + Metric Head 모두 Unfreeze.
*   **목표:** 특징 추출 단계부터 시간적 일관성을 학습하여 Flickering의 근본 원인 해결.

---

## 4. 손실 함수 구성 (Loss Functions)

1.  **Log L1 Loss:** 최종 절대 깊이($\hat{D}_{abs} = S \cdot D_{rel} + B$)와 GT 간의 오차 최소화.
2.  **TGM (Temporal Geometric Matching) Loss:** 비디오 전체의 기하학적 일관성 확보.
3.  **Feature-Level Consistency Loss:** 특징 맵 단계에서의 시간적 정렬 강제.

---

## 5. Feature-Level Consistency Loss 상세 설계

상대 깊이 맵의 뼈대가 되는 특징 맵의 시간적 안정성을 위해 다음 세 가지 방안을 제안합니다.

### Method ①: Warp-based Feature Distance (가장 정밀함)
*   **메커니즘:** 외부 Optical Flow 모델(RAFT 등)을 사용하여 프레임 $t-1$의 특징 맵 $F_{t-1}$을 $t$ 시점으로 워핑($\mathcal{W}$)한 후, 현재 특징 $F_t$와의 $L_2$ 거리를 계산합니다.
    $$\mathcal{L}_{feat} = 	ext{mask} \cdot \| F_t - \mathcal{W}(F_{t-1}, 	ext{flow}_{t 	o t-1}) \|^2$$
*   **Teacher-Student 학습:** 학습 시에만 고성능 Flow 모델을 사용하여 Pseudo-GT Flow를 생성하며, 추론 시에는 Flow 모델이 필요 없어 FPS를 유지합니다.
*   **강점:** 카메라 및 물체의 움직임을 물리적으로 완벽히 반영하여 '진정한 의미의 연속성'을 학습시킵니다.

### Method ②: Temporal Cosine Similarity (연산 효율형)
*   **메커니즘:** 워핑 연산 없이, 인접 프레임 간 동일 공간 위치($x, y$)에 있는 특징 벡터들 사이의 Cosine 유사도를 극대화합니다.
*   **예외 처리 (Global Motion Gate):** 특징 맵 전체의 평균 유사도가 특정 임계값 이하로 떨어지면, 급격한 움직임으로 판단하여 해당 프레임의 로스를 차단합니다.
*   **강점:** 움직임이 적은 정적인 시퀀스에서 특징의 방향성(Directionality)을 유지하는 데 효과적입니다.

### Method ③: Feature Correlation (Gram Matrix) (가장 경량화)
*   **메커니즘:** 특징 맵 내 채널 간 상관관계를 나타내는 Gram Matrix $G$를 계산하고, 프레임 간 $G$의 차이를 최소화합니다.
*   **강점:** 픽셀 단위의 정확한 정렬보다는 장면 전체의 '특징 분포'를 일정하게 유지하여 가성비 좋게 Flickering을 억제합니다.

---

## 6. 급격한 변화 및 장면 전환 대응 (Handling Dynamics)

### 6.1 Forward-Backward Validity Masking (Method ① 전용)
*   $t 	o t-1$ 방향과 $t-1 	o t$ 방향의 Flow를 교차 검증하여, 픽셀이 일치하지 않는 영역(Occlusion, 빠른 이동)은 로스 계산에서 제외(Masking)합니다. 이를 통해 잔상(Ghosting) 현상을 방지합니다.

### 6.2 CLS-based Scene Cut Detector (공통 적용)
*   **Scene Cut 감지:** $Distance(CLS_t, CLS_{t-1}) > 	au$ 일 경우 장면 전환으로 정의합니다.
*   **Adaptive Weighting:** CLS 간 거리에 반비례하는 Sigmoid 가중치를 Consistency Loss에 곱하여, 장면이 바뀔 때는 로스를 즉시 차단(Zeroing)하고 새로운 특징을 수용하게 합니다.
    $$W_{temporal} = 1 - \sigma(k \cdot (dist(CLS_t, CLS_{t-1}) - 	au))$$
