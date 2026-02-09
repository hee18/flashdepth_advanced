# 🚀 FlashDepth-Metric: 실시간 비디오 절대 깊이 추정 고도화 설계안

## 1. 아키텍처 개요 (Architecture Overview)
본 설계는 DINOv2의 강력한 의미론적 특징과 DPT의 정밀한 기하학적 특징을 **단일 통합 시계열 Mamba(Unified Global Mamba)**를 통해 결합합니다. 이를 통해 절대 깊이 추정을 위한 Scale/Shift의 안정성을 확보하고, 비디오 전체의 시간적 일관성을 유지합니다.

*   **Backbone:** DINOv2 (Frozen)
*   **Temporal Module:** **Unified Global Mamba (1280-dim)**
*   **Decoder:** DPT (Phase 2에서 Unfreeze)
*   **Guidance:** GAP 기반 FiLM (Feature-wise Linear Modulation)

---

## 2. 통합 시계열 모듈 (Unified Global Mamba)

### 2.1 글로벌 컨텍스트 통합 (Input)
각 프레임($t$)에서 두 종류의 글로벌 정보를 결합하여 1280차원의 시퀀스를 생성합니다.
*   **Semantic Base:** DINOv2 CLS Token [1024-dim]
*   **Geometric Base:** DPT Feature Map의 Global Average Pooling (GAP) [256-dim]
*   **연산:** $F_{global}^{(t)} = \text{Concat}(CLS^{(t)}, GAP^{(t)})$ → $[B, T, 1280]$

### 2.2 시계열 정렬 (Temporal Alignment)
단일 Mamba 블록이 1280차원의 글로벌 컨텍스트 흐름을 학습합니다.
*   **특징:** Mamba의 Parallel Scan을 통해 학습 시 병렬성을 유지하면서도, 숨겨진 상태(Hidden State)를 통해 과거 프레임의 문맥을 현재에 전파합니다.
*   **결과:** 시간적으로 정제된 $\hat{F}_{global}^{(t)}$ 생성.

---

## 3. 이중 경로 주입 전략 (Dual-Path Context Injection)

정제된 글로벌 특징을 각각의 목적에 맞게 분리하여 주입함으로써 품질 하락을 방지합니다.

### 3.1 Path A: 절대 스케일 추정 (Metric Head)
*   **대상:** $\hat{F}_{global}^{(t)}$ 전체 (1280-dim) 사용.
*   **논리:** 장면의 의미(CLS)와 구조(GAP)를 모두 알아야 정확한 절대 스케일($S$)과 오프셋($B$)을 계산할 수 있습니다.
*   **출력:** $Scale (S) = \text{Softplus}$, $Shift (B) = 0.1 \times \text{Sigmoid}$.

### 3.2 Path B: 상대 깊이 가이드 (Spatial Guidance)
*   **대상:** $\hat{F}_{global}^{(t)}$ 중 **GAP 유래 부분(앞 256-dim)**만 사용.
*   **방법 (FiLM):** 정제된 GAP를 통해 $\gamma, \beta$를 추출하여 DPT Feature Map을 변조.
    $$F_{spatial}^{(refined)} = F_{spatial} \cdot \gamma(\hat{F}_{gap}) + \beta(\hat{F}_{gap})$$
*   **논리 (Semantic Hallucination 방지):** 상대 깊이 맵의 기하학적 디테일을 보존하기 위해 CLS(의미) 정보의 직접 개입을 차단하고, 안정화된 GAP(기하) 정보만 가이드로 사용합니다.

---

## 4. 손실 함수 구성 (Loss Functions)

1.  **Log L1 Loss:** 최종 절대 깊이($\hat{D}_{abs} = S \cdot D_{rel} + B$)와 GT 간의 오차 최소화.
2.  **TGM (Temporal Geometric Matching) Loss:** 픽셀 단위 깊이 변화량의 일관성 강제.
    $$\mathcal{L}_{tgm} = \|(D_{pred}^{(t)} - D_{pred}^{(t-1)}) - (D_{gt}^{(t)} - D_{gt}^{(t-1)})\|$$
3.  **Feature-Level Consistency Loss (Warp-based):** 
    *   Sea-RAFT의 **Confidence Map**을 활용한 단방향 워핑 적용 (연산 효율화).
    *   $\mathcal{L}_{feat} = \text{mask}_{conf} \cdot \| F_t - \mathcal{W}(F_{t-1}, \text{flow}) \|^2$

---

## 5. 급격한 변화 대응 (Handling Dynamics)

### 5.1 CLS-based Scene Cut Detector
*   **Metric:** 인접 프레임 CLS 간의 Cosine Distance ($D_{cls}$).
*   **Adaptive Weighting:** $W_{temporal} = 1 - \sigma(k \cdot (D_{cls} - \tau))$
*   **작동:** 장면 전환 시 ($D_{cls} > \tau$) TGM 및 Feature Consistency Loss를 즉시 차단하여 모델 오염 방지.
*   **자동 설정:** 학습 초기 1,000 step 동안 $D_{cls}$ 분포를 측정하여 상위 5% 지점을 $\tau$로 자동 설정.

---

## 6. 학습 단계 (Phase Transition)

*   **Phase 1 (Metric Alignment):** Unified Mamba + Metric Head만 Unfreeze. (빠른 스케일 적응)
*   **Phase 2 (Full Video Optimization):** DPT Decoder까지 Unfreeze. 
    *   DPT의 LR은 Metric Head의 1/10로 설정하여 안정적 미세 조정.
    *   DINOv2 Backbone은 전 과정 Frozen 유지.
