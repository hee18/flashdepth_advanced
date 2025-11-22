# Importance Loss 타당성 분석

## 1. 연구 질문

**주장 1**: "ViT가 중요하게 보는 정도(attention weight)를 살려서 정확도를 높인다"

**주장 2**: "Attention weight가 높다 = 다른 패치와 관계성이 높다"

**주장 3**: "해당 부분의 정확도 가중치를 높이면 전체 정확도 향상으로 이어진다"

---

## 2. 코드 구현의 실제 동작

### 현재 구현

```python
# CLS-to-patch attention 추출 (flashdepth/gear5_modules.py)
cls_to_patch = attn[:, :, 0, 1:]  # [B, num_heads, num_patches]
cls_to_patch = cls_to_patch.mean(dim=1)  # Average over heads
cls_attention = torch.stack(cls_to_patch_list, dim=0).mean(dim=0)  # Average over layers

# Loss weighting (train_gear5.py:1130)
weights = 1.0 + fg_ratio * importance_flat
final_loss = (loss[valid_mask] * weights[valid_mask]).sum() / valid_mask.float().sum()
```

### 실제로 측정하는 것

- **CLS 토큰이 각 패치로부터 정보를 얼마나 가져오는가**
- 이것은 "패치 간 관계성"이 아니라 **"CLS의 정보 집계 패턴"**

---

## 3. 이론적 문제점

### ❌ 문제 1: 주장 2의 오류 - Attention 의미 혼동

당신은 "다른 패치와 관계성이 높다"고 했지만:

- **CLS-to-patch attention**: CLS가 해당 패치에서 정보를 얼마나 가져오는가
- **Patch-to-patch attention**: 패치들 간의 상호작용 (당신이 의도한 의미)

**→ 구현과 설명이 불일치합니다.**

---

### ⚠️ 문제 2: CLS Attention ≠ Depth Importance

#### DINOv2 CLS attention의 특성 (검증된 연구 결과)

1. **Foreground/background 구분에 특화**
   - "Attention density 높음 = foreground object"
   - Background는 낮은 attention
   - 출처: "Upsampling DINOv2 features for unsupervised vision tasks"

2. **Semantic segmentation에 강점**
   - CLS token attention maps에 explicit semantic information 포함
   - Foreground-background differentiability 강함

3. **알려진 문제점**
   - DINOv2는 DINOv1보다 attention map이 덜 semantic함
   - Outlier peaks (attention artifacts)가 background에 나타남
   - 출처: "Vision Transformers Need Registers"

#### Depth estimation의 실제 중요 영역 (다수 논문)

1. **Edge/boundary discontinuities**
   - EGSA-PT: Edge-Guided Spatial Attention
   - BAM: Boundary Attention Module
   - 출처: "Monocular depth estimation with boundary attention mechanism"

2. **Occlusion boundaries**
   - Depth regression models가 가장 취약한 부분
   - "Depth regression models overlook fine details, particularly along object boundaries"

3. **Fine details** (texture, fur, hair)
   - Geometric structure와 무관하게 복잡한 texture

#### 불일치 사례

| 영역 | CLS Attention | Depth Importance |
|------|---------------|------------------|
| 평평한 벽 (background) | **Low** | **High** (accurate plane 필요) |
| 복잡한 texture (foreground) | **High** | **Medium** (geometric은 단순) |
| Object boundary | **Medium** | **Very High** (discontinuity) |
| Sky (background) | **Low** | **Medium** (scale calibration) |

---

### ⚠️ 문제 3: "Attention is not Explanation"

#### NLP에서 검증된 사실 (Jain & Wallace, 2019)

- **Attention weights ≠ Feature importance**
- Attention은 모델이 "어디를 봤는가"를 보여주지만
- "어디가 중요한가"를 보장하지 않음
- 실험: Attention과 gradient-based importance는 상관관계 낮음

#### Vision에서도 유사

> "Self-attention in vision transformers performs perceptual grouping, not attention"
> — Frontiers in Computer Science, 2023

- ViT의 attention은 perceptual grouping (비슷한 패치 묶기)
- Saliency/importance와는 다른 개념

#### 반론 논문 (Wiegreffe & Pinter, 2019)

"Attention is not not Explanation"
- "Explanation"의 정의에 따라 다름
- Task-specific하게 유효할 수 있음
- **하지만 일반적인 feature importance는 아님**

---

### ⚠️ 문제 4: Circular Reasoning

```
CLS token → Scale/Shift 예측
         ↓
CLS attention 높은 영역 → Loss weight 증가
         ↓
CLS의 편향 강화 (?)
```

**질문**: CLS가 이미 본 곳에 더 가중치를 주는 것이, CLS의 잘못된 편향을 오히려 고착화시키는 것은 아닌가?

**예시**:
- CLS가 foreground object에만 집중하도록 학습됨
- Importance weighting으로 foreground loss만 증폭
- Background depth는 무시됨
- → CLS는 계속 foreground만 보게 됨 (악순환)

---

## 4. 관련 연구 증거

### ✅ Edge/Boundary Weighting은 검증됨

실제 depth estimation 논문들:

1. **BAM (Boundary Attention Module)**
   - Edge 정보를 feature fusion에 활용
   - Boundary distortion 방지
   - 출처: "Monocular depth estimation with boundary attention mechanism"

2. **EGSA-PT (Edge-Guided Spatial Attention)**
   - RGB edge → Depth edge로 progressive training
   - Multi-modal fusion에서 boundary 정보 통합
   - 출처: "EGSA-PT: Edge-Guided Spatial Attention with Progressive Training"

3. **Edge Loss Functions**
   - Keras 예제: `edge_loss_weight = 0.9`
   - SSIM + L1 + Edge loss 조합
   - 출처: "Keras documentation: Monocular depth estimation"

**하지만**: 이들은 **edge detector** (Sobel, Canny 등)를 사용하지, **CLS attention**을 사용하지 않음

---

### ❓ CLS Attention Weighting은 증거 부족

웹 검색 및 문헌 조사 결과:

- **DPT (Ranftl et al., 2021)**: Attention 사용하지만 importance weighting 없음
- **DepthAnything (Yang et al., 2024)**: CLS attention weighting 없음
- **Metric3D (Yin et al., 2023)**: Canonical space 사용, attention weighting 없음
- **CLS attention을 depth loss weighting에 사용한 선행 연구를 찾을 수 없음**

---

### 🔍 유사 연구 (Vision-Language Models)

**FasterVLM (2024)**:
- CLS attention으로 visual token importance 평가
- 중요도 낮은 token pruning → inference 가속
- **하지만**: Loss weighting이 아니라 token selection에 사용
- 결과: Performance 유지하면서 속도 향상

**차이점**:
- FasterVLM: Attention 높은 token만 유지 (binary selection)
- 당신 방법: Attention 기반 continuous weighting
- FasterVLM은 검증됐지만, 당신 방법과는 목적이 다름

---

## 5. 잠재적 이점과 위험

### ✅ 잠재적 이점

#### 1. Foreground object 정확도 향상 가능
- CLS attention이 foreground에 집중하므로
- Foreground depth가 더 정확해질 수 있음
- Semantic-aware depth estimation

#### 2. Robustness to background clutter
- 복잡한 background (나뭇잎, 풀 등)의 noise 영향 감소
- Foreground object에 집중

#### 3. Object-centric applications에 유리
- Robotics: Manipulate할 object의 depth가 중요
- AR/VR: Foreground character의 depth가 중요
- Background는 덜 중요한 경우

---

### ⚠️ 위험

#### 1. Background 정확도 저하

```python
fg_mask = (importance_flat > importance_threshold)
fg_ratio = fg_mask.float().mean()
weights = 1.0 + fg_ratio * importance_flat
```

- Background는 weight ≈ 1.0
- Foreground는 weight > 1.0
- **Background depth를 상대적으로 무시**

**문제 시나리오**:
- 자율주행: 도로 surface (background)의 depth 부정확
- Scene reconstruction: 벽/바닥 (background)의 geometry 왜곡

#### 2. Geometric structure 무시

Edge-based weighting vs CLS attention weighting:

| 방법 | 강조하는 영역 | 놓치는 영역 |
|------|--------------|------------|
| **Edge-based** | Depth discontinuity | Flat texture regions |
| **CLS attention** | Semantic foreground | Geometric boundaries |

- Edge가 아니라 semantic foreground에만 집중
- Depth discontinuity가 background에 있으면 놓침

#### 3. Register token artifact (DINOv2 문제)

당신 코드 (gear5_modules.py:80-92):
```python
# Remove register token (highest attention patch) with 3×3 inpainting
max_val = attn_2d.max()
outlier_mask = (attn_2d == max_val)
kernel = torch.ones(1, 1, 3, 3) / 9
attn_smoothed = F.conv2d(importance_map[b:b+1], kernel, padding=1)
importance_map[b, 0] = torch.where(outlier_mask, attn_smoothed[0, 0], importance_map[b, 0])
```

**문제**:
- 3×3 inpainting이 완벽하지 않을 수 있음
- Multiple outliers가 있으면 처리 불완전
- DINOv2의 artifact는 단순 최댓값 제거로 해결 안 될 수 있음

#### 4. Metric 평가의 bias

표준 depth metrics (AbsRel, RMSE 등):
- 전체 픽셀의 평균 오차
- Foreground만 정확하면 좋은 metric 나올 수 있음
- **하지만 실제 사용성은 떨어질 수 있음**

---

## 6. 실험적 검증이 필요한 질문

당신의 방법이 실제로 효과적인지 확인하려면:

### 1. Ablation Study

```python
# A. Baseline (no weighting)
loss_type: log_l1

# B. Importance weighting
loss_type: importance

# C. Edge weighting (대조군)
edge_map = sobel_filter(depth_gt)
weights = 1.0 + alpha * edge_map
```

**측정**:
- Overall metrics (AbsRel, RMSE, δ1)
- Foreground-only metrics
- Background-only metrics
- Boundary-only metrics (±5px around edges)

### 2. Attention-Edge Correlation

```python
cls_attention_map = importance_map_generator(...)
edge_map = sobel_filter(depth_gt)

# Spatial correlation
correlation = torch.corrcoef(
    cls_attention_map.flatten(),
    edge_map.flatten()
)[0, 1]
```

**해석**:
- `correlation > 0.7`: CLS attention이 edge를 잘 찾음 → 타당성 증가
- `correlation < 0.3`: CLS attention과 edge 무관 → 문제

### 3. Alternative Importance Maps

비교 실험:

| Importance Source | 이론적 근거 |
|------------------|-----------|
| CLS attention (yours) | Semantic foreground |
| Sobel edge | Geometric discontinuity |
| Gradient magnitude | True feature importance |
| Depth variance | Local complexity |
| Semantic segmentation mask | Class-based weighting |

### 4. Per-Region Analysis

Dataset: Waymo, Sintel (semantic masks 있음)

```python
# Foreground (person, car, etc.)
fg_mask = (semantic_mask > 0)
fg_absrel = compute_metric(pred[fg_mask], gt[fg_mask])

# Background (sky, road, building)
bg_mask = (semantic_mask == 0)
bg_absrel = compute_metric(pred[bg_mask], gt[bg_mask])

# Boundary (edge ±5px)
boundary_mask = edge_dilation(edge_map, radius=5)
boundary_absrel = compute_metric(pred[boundary_mask], gt[boundary_mask])
```

### 5. Visualization Analysis

```python
# Save comparison
save_comparison(
    image=rgb,
    cls_attention=importance_map,
    edge_map=edge_map,
    depth_error=torch.abs(pred - gt),
    save_path=f"analysis/{seq_name}_{frame_idx}.png"
)
```

**확인 사항**:
- CLS attention 높은 곳 = 실제 depth error 낮은가?
- Background에서 error 증가하는가?
- Edge와 attention의 일치도

---

## 7. 대안 제안

### 대안 1: Edge-based Importance (검증된 방법)

```python
def edge_based_importance(depth_pred, depth_gt):
    """Geometric edge를 기반으로 importance 계산"""
    # GT depth의 gradient
    grad_x = torch.abs(depth_gt[:, :, :, 1:] - depth_gt[:, :, :, :-1])
    grad_y = torch.abs(depth_gt[:, :, 1:, :] - depth_gt[:, :, :-1, :])

    # Gradient magnitude (edge strength)
    edge_map = torch.sqrt(grad_x**2 + grad_y**2)

    # Normalize to [0, 1]
    edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)

    return edge_map

# Loss
weights = 1.0 + alpha * edge_map
loss_weighted = (loss * weights).mean()
```

**장점**:
- 이론적 근거 명확 (실제 depth discontinuity)
- 실제 논문들에서 검증됨 (BAM, EGSA-PT 등)

**단점**:
- Noisy depth GT에서 edge detection 불안정
- Flat texture region 무시 (하지만 depth에선 덜 중요)

---

### 대안 2: Gradient-based Importance (진짜 중요도)

```python
def gradient_based_importance(model, images, depth_gt):
    """Loss gradient 기반 진짜 feature importance"""
    images.requires_grad = True

    # Forward
    depth_pred = model(images)
    loss = torch.abs(depth_pred - depth_gt).mean()

    # Backward
    grad = torch.autograd.grad(loss, images)[0]

    # Importance = gradient magnitude
    importance = grad.abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]

    return importance.detach()
```

**장점**:
- "Attention is not Explanation" 논문이 제안한 방법
- 실제 loss에 영향 미치는 영역 측정
- Task-specific (depth estimation에 진짜 중요한 곳)

**단점**:
- Computational cost 높음 (매 iteration마다 extra backward pass)
- Memory 사용량 증가

---

### 대안 3: Hybrid Approach (추천)

```python
def hybrid_importance(cls_attention, edge_map, alpha=0.5):
    """Semantic + Geometric 조합"""
    # Normalize both to [0, 1]
    cls_norm = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
    edge_norm = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)

    # Weighted combination
    importance = alpha * cls_norm + (1 - alpha) * edge_norm

    return importance

# Ablation: alpha = {0.0, 0.3, 0.5, 0.7, 1.0}
```

**장점**:
- Semantic (CLS attention) + Geometric (edge) 모두 고려
- Alpha로 balance 조절 가능
- 둘의 상호보완

**단점**:
- Hyperparameter 추가 (alpha)
- Edge map 계산 필요

---

### 대안 4: Learned Importance (가장 정교)

```python
class LearnedImportanceGenerator(nn.Module):
    """Importance map을 학습 가능한 network로 생성"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # [0, 1]
        )

    def forward(self, features):
        """features: DPT intermediate features"""
        return self.conv(features)

# Usage
importance_map = learned_generator(dpt_features)
weights = 1.0 + alpha * importance_map
```

**장점**:
- Data-driven (학습을 통해 최적 importance 학습)
- Hand-crafted heuristic 불필요
- Task-specific adaptation

**단점**:
- Trainable parameters 증가
- Overfitting 위험
- Interpretability 감소

---

## 8. 결론: 객관적 판단

### ❌ 이론적 타당성: 약함

1. **주장 2는 틀림**:
   - CLS-to-patch attention ≠ "패치 간 관계성"
   - 실제로는 "CLS의 정보 집계 패턴"

2. **근거 부족**:
   - CLS attention이 depth importance를 나타낸다는 직접 증거 없음
   - "Attention is not Explanation" 연구와 모순

3. **선행 연구 부재**:
   - 이 방법을 사용한 검증된 depth estimation 논문 없음
   - 유사한 FasterVLM은 다른 목적 (token pruning)

---

### ⚠️ 실용적 가능성: 불확실

#### 가능한 긍정적 결과
1. **Foreground 향상**: Semantic object depth 정확도 증가
2. **Object-centric tasks**: Robotics, AR 등에 유리
3. **Robustness**: Background clutter noise 감소

#### 예상되는 부정적 결과
1. **Background 희생**: 전체 metric은 오히려 떨어질 수 있음
2. **Geometric structure 무시**: Edge ≠ Foreground
3. **Circular reasoning**: CLS 편향 강화 위험

---

### ✅ 검증 방법

논문에 쓰려면 **반드시** 다음을 수행해야 함:

1. **Ablation study**:
   ```
   - Baseline (log_l1)
   - CLS attention weighting (importance)
   - Edge weighting (대조군)
   - Hybrid (CLS + Edge)
   ```

2. **Per-region metrics**:
   ```
   - Overall AbsRel, RMSE, δ1
   - Foreground-only metrics
   - Background-only metrics
   - Boundary-only metrics
   ```

3. **Correlation analysis**:
   ```
   - CLS attention vs Edge map correlation
   - CLS attention vs Depth error correlation
   - Statistical significance test (p-value)
   ```

4. **Qualitative analysis**:
   ```
   - Visualization: attention map, depth error map
   - Case study: 성공 사례, 실패 사례
   ```

---

### 📝 논문 작성 시 표현 방법

#### ❌ 피해야 할 표현

- "CLS attention represents the **importance** of each patch"
- "Higher attention means higher **relevance** to depth prediction"
- "This is **obviously** beneficial for depth estimation"

#### ✅ 권장 표현

- "We explore **semantic-aware** loss weighting using CLS attention"
- "CLS attention may serve as a **proxy** for foreground saliency"
- "We **hypothesize** that emphasizing semantically salient regions improves depth estimation"
- "Empirical results suggest that... (실험 결과 기반)"

#### ✅ 논문 구조 제안

```markdown
## Introduction
- Motivation: Foreground objects often more critical for applications
- Hypothesis: Semantic-aware weighting may improve task-specific performance

## Method
- CLS-to-patch attention extraction
- Importance map generation (with register token removal)
- Loss weighting: weights = 1.0 + fg_ratio * importance

## Experiments
- Ablation study: log_l1 vs importance vs edge vs hybrid
- Per-region analysis: foreground, background, boundary
- Correlation analysis: attention vs edge vs error

## Results
- Foreground metrics: +X% improvement
- Background metrics: -Y% degradation (if any)
- Overall: +Z% (or neutral, or negative)

## Discussion
- When it works: Object-centric tasks, clean backgrounds
- When it fails: Geometric-heavy scenes, important backgrounds
- Limitations: Circular reasoning risk, artifact sensitivity
```

---

## 9. 최종 권고사항

### 실험 순서

1. **Week 1: Baseline 확립**
   - `loss_type=log_l1` 학습
   - 모든 dataset에서 metric 측정

2. **Week 2: Importance weighting 테스트**
   - `loss_type=importance` 학습
   - Same datasets, same metric
   - **Per-region analysis 추가**

3. **Week 3: Ablation studies**
   - Edge-based weighting
   - Hybrid approach (CLS + Edge)
   - Alpha sweep (0.0, 0.3, 0.5, 0.7, 1.0)

4. **Week 4: 분석 및 시각화**
   - Correlation plots
   - Error distribution histograms
   - Qualitative examples (best/worst cases)

---

### 논문 작성 전략

#### 시나리오 A: Importance weighting이 효과적인 경우

```markdown
Title: Semantic-Aware Loss Weighting for Monocular Depth Estimation

Contribution:
- Novel use of CLS attention for foreground-focused depth learning
- Empirical validation on object-centric benchmarks
- X% improvement on foreground depth accuracy
```

#### 시나리오 B: Importance weighting이 효과적이지 않은 경우

```markdown
Title: On the Effectiveness of Attention-based Loss Weighting for Depth Estimation

Contribution:
- Comprehensive analysis of CLS attention vs depth importance
- Empirical evidence that semantic attention ≠ geometric importance
- Recommendation: Edge-based weighting for better performance
```

**→ 두 경우 모두 논문 가능! Negative result도 valuable contribution**

---

### 마지막 조언

1. **실험을 먼저 하라**
   - 이론적 타당성이 약해도, 실험적 효과가 있으면 OK
   - "It works in practice" > "It sounds good in theory"

2. **정직하게 작성하라**
   - Limitation을 숨기지 말고 명시
   - Ablation study로 trade-off 보여줌
   - Negative result도 논문 가능

3. **Alternative를 시도하라**
   - Edge-based weighting (이론적으로 더 타당)
   - Hybrid approach (best of both worlds)
   - Learned importance (most flexible)

4. **Reviewer 관점에서 생각하라**
   - "왜 CLS attention이 depth importance를 나타내는가?" → Weak point
   - "실험적으로 효과가 있는가?" → 여기에 집중
   - "다른 방법과 비교했는가?" → Ablation 필수

---

## 10. 참고문헌

### Depth Estimation with Attention/Weighting

1. **Ranftl et al., "Vision Transformers for Dense Prediction"**, ICCV 2021
   - DPT 원본 논문, attention 사용하지만 importance weighting 없음

2. **"Monocular depth estimation with boundary attention mechanism"**, 2024
   - BAM: Boundary attention for edge enhancement
   - Edge-based weighting의 효과 검증

3. **"EGSA-PT: Edge-Guided Spatial Attention"**, 2024
   - Progressive training with RGB→Depth edge
   - Edge information fusion

4. **"Edge loss functions for deep-learning depth-map"**, 2021
   - Edge loss의 다양한 조합 실험
   - Edge weighting의 타당성 검증

### CLS Token and Attention Analysis

5. **"[CLS] Attention is All You Need for Training-Free Visual Token Pruning"**, 2024
   - FasterVLM: CLS attention으로 token importance 평가
   - Token pruning에 사용 (loss weighting 아님)

6. **"Upsampling DINOv2 features for unsupervised vision tasks"**, 2024
   - DINOv2 CLS attention의 foreground/background 특성
   - Attention density로 foreground 판별

7. **"Vision Transformers Need Registers"**, 2022
   - DINOv2 attention artifact 문제
   - Register token의 역할과 제거 필요성

### Attention as Explanation

8. **Jain & Wallace, "Attention is not Explanation"**, NAACL 2019
   - Attention weights ≠ feature importance
   - Gradient-based importance 제안

9. **Wiegreffe & Pinter, "Attention is not not Explanation"**, EMNLP 2019
   - 반론: Task-specific하게 유효할 수 있음
   - Definition of "explanation"에 따라 다름

10. **"Self-attention in vision transformers performs perceptual grouping"**, Frontiers 2023
    - ViT attention은 perceptual grouping
    - Saliency와는 다른 개념

### Vision Transformer Analysis

11. **"How Does Attention Work in Vision Transformers?"**, PMC 2023
    - CLS token과 patch token의 역할 분석
    - Entangled effect: attention + classification

12. **"Class-Discriminative Attention Maps for Vision Transformers"**, 2023
    - Attention visualization 방법
    - Foreground object에 대한 attention 특성

---

## 부록: 코드 스니펫

### A. Edge-based Importance 구현

```python
# flashdepth/importance_generators.py
import torch
import torch.nn.functional as F

class EdgeBasedImportance(nn.Module):
    """Edge-based importance map generator"""
    def __init__(self, edge_threshold=0.1):
        super().__init__()
        self.edge_threshold = edge_threshold

        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float().view(1, 1, 3, 3))

        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]).float().view(1, 1, 3, 3))

    def forward(self, depth_map):
        """
        Args:
            depth_map: [B, 1, H, W] depth prediction or GT
        Returns:
            importance_map: [B, 1, H, W] edge-based importance
        """
        # Compute gradients
        grad_x = F.conv2d(depth_map, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth_map, self.sobel_y, padding=1)

        # Gradient magnitude
        edge_map = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # Normalize to [0, 1]
        B = edge_map.shape[0]
        for b in range(B):
            edge_flat = edge_map[b].flatten()
            edge_min = edge_flat.quantile(0.01)
            edge_max = edge_flat.quantile(0.99)
            edge_map[b] = (edge_map[b] - edge_min) / (edge_max - edge_min + 1e-8)
            edge_map[b] = torch.clamp(edge_map[b], 0.0, 1.0)

        return edge_map
```

### B. Hybrid Importance 구현

```python
class HybridImportance(nn.Module):
    """Hybrid: CLS attention + Edge map"""
    def __init__(self, alpha=0.5, num_layers=2):
        super().__init__()
        self.alpha = alpha
        self.cls_generator = ImportanceMapGenerator(num_layers=num_layers)
        self.edge_generator = EdgeBasedImportance()

    def forward(self, attention_weights_list, depth_map, patch_h, patch_w):
        """
        Args:
            attention_weights_list: For CLS attention
            depth_map: [B, 1, H, W] for edge detection
            patch_h, patch_w: Spatial dimensions
        """
        # CLS attention importance
        cls_importance = self.cls_generator(
            attention_weights_list, patch_h, patch_w
        )  # [B, 1, patch_h, patch_w]

        # Upsample to image resolution
        cls_importance_up = F.interpolate(
            cls_importance, size=depth_map.shape[-2:],
            mode='bilinear', align_corners=True
        )  # [B, 1, H, W]

        # Edge importance
        edge_importance = self.edge_generator(depth_map)  # [B, 1, H, W]

        # Weighted combination
        importance = self.alpha * cls_importance_up + (1 - self.alpha) * edge_importance

        return importance
```

### C. Loss 함수 통합

```python
# train_gear5.py
def compute_loss(self, pred_depth, gt_depth, importance_map=None):
    """
    Args:
        pred_depth: [B*T, 1, H, W]
        gt_depth: [B*T, 1, H, W]
        importance_map: [B*T, 1, H, W] or None
    """
    # Valid mask
    valid_mask = (gt_depth > 0) & (gt_depth < 1000)

    # Log L1 loss
    epsilon = 1e-6
    loss = torch.abs(
        torch.log(pred_depth + epsilon) -
        torch.log(gt_depth + epsilon)
    )

    if self.loss_type == 'importance' and importance_map is not None:
        # Flatten
        loss_flat = loss[valid_mask]
        importance_flat = importance_map[valid_mask]

        # Compute fg_ratio
        importance_threshold = importance_flat.mean()
        fg_mask = (importance_flat > importance_threshold)
        fg_ratio = fg_mask.float().mean()

        # Weighted loss
        weights = 1.0 + fg_ratio * importance_flat
        final_loss = (loss_flat * weights).sum() / valid_mask.float().sum()
    else:
        # Regular loss
        final_loss = loss[valid_mask].mean()
        fg_ratio = torch.tensor(0.0)

    return final_loss, fg_ratio
```

---

**작성일**: 2025-01-21
**분석자**: Claude (Sonnet 4.5)
**검토 대상**: GEAR5 Importance Loss 타당성
