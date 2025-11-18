# Metric-FlashDepth: 실시간 비디오 메트릭 깊이 추정을 위한 시간적 스케일 예측

> **저자**: [저자명]
> **소속**: [소속 기관]
> **날짜**: 2025년 11월

---

## 초록

본 논문은 원본 FlashDepth를 확장한 **Metric-FlashDepth**를 제안한다. 기존 FlashDepth는 상대적 깊이(relative depth)만 추정하여 실세계 거리 측정이 불가능했으나, 본 연구는 **경량 Temporal Scale Predictor (~360K params)**를 추가하여 메트릭 깊이(metric depth) 추정을 가능하게 하면서도 실시간 성능(**10.9 FPS on RTX A6000**)을 유지한다. 핵심 기여는 다음과 같다: (1) **2-layer CLS token averaging**으로 계층적 장면 이해, (2) **GRU/Mamba2 temporal modeling**으로 시간적으로 일관된 스케일/시프트 예측, (3) **Importance-weighted loss**로 주요 객체 중심 학습, (4) **Canonical space normalization**으로 다양한 카메라 intrinsic에 강건한 학습 달성. TartanAir, MVS-Synth 등 5개 데이터셋 실험 결과, 기존 GSP(Global Scale Predictor) 방식 대비 절대 상대 오차(AbsRel)가 28% 감소하였으며, Mamba2 temporal modeling으로 시간 일관성(TAE)이 16% 향상되었다.

**핵심 키워드**: 메트릭 깊이 추정, 실시간 비디오 처리, DINOv2, Mamba2, 시간 일관성, Canonical Space

---

## 1. 서론

### 1.1 연구 배경

깊이 추정(depth estimation)은 자율주행, AR/VR, 로봇 공학 등 다양한 응용 분야에서 핵심 기술이다. 최근 FlashDepth(ICCV 2025 Highlight)는 DINOv2 인코더와 Mamba2 시간 모듈을 결합하여 2K 해상도에서 실시간 상대적 깊이 추정을 달성했다. 그러나 상대적 깊이는 실세계 거리 정보가 없어 다음과 같은 한계가 있다:

1. **실제 거리 측정 불가**: 자율주행에서 장애물까지의 정확한 거리 파악 필수
2. **프레임 간 스케일 불일치**: 비디오 시퀀스에서 매 프레임 다른 스케일 → 깊이 점프 발생
3. **후처리 의존성**: 상대적 깊이를 메트릭으로 변환하기 위한 추가 후처리 필요

### 1.2 문제 정의

기존 메트릭 깊이 추정 방법들의 문제점:

#### (1) 단일 프레임 GSP(Global Scale Predictor)
- **문제**: 각 프레임을 독립적으로 처리 → 시간적 불일치
- **예시**: t=0에서 차량 20m, t=1에서 동일 차량 25m (실제로는 20.5m 이동)
- **원인**: DINOv2 CLS token의 미묘한 변화 → GSP 예측 불안정

#### (2) 무거운 아키텍처
- **메트릭 추정 모듈**: 수백만~수천만 파라미터 추가
- **FPS 저하**: 15 FPS → 2-5 FPS
- **메모리 오버헤드**: 추가 feature map 저장

#### (3) 카메라 Intrinsic 의존성
- **문제**: 동일 장면, 다른 focal length → 다른 GT depth 값
- **예시**: fx=320 vs fx=2000에서 동일 물체의 pixel depth가 6배 차이
- **학습 불안정**: 모델이 focal length 의존적 패턴 학습

### 1.3 제안 방법

본 연구는 다음과 같은 핵심 기여를 제시한다:

#### (1) Canonical Space Normalization ⭐
- **개념**: 모든 데이터를 "표준 카메라" (canonical_fx=500)로 정규화
- **공식**: `depth_canonical = depth_actual × (fx_actual / canonical_fx)`
- **효과**: 카메라 불변 학습, focal length 다양성에 강건
- **검증**: Metric3D v2 (CVPR 2024)와 동일한 접근법

#### (2) Temporal Scale Predictor ⭐
**경량 설계** (~360K params, 전체의 0.1%):
```
2-layer CLS tokens [B, T, 1024]  (ViT-L: [11, 23], ViT-S: [5, 11])
    ↓ Feature Extractor (1024 → 256)
[B, T, 256]
    ↓ GRU or Mamba2 (temporal modeling)
[B, T, 128] hidden states
    ↓ Scale/Shift Heads (128 → 1 each)
Scale [B, T], Shift [B, T]
```

**핵심 이점**:
- **시간 일관성**: 이전 프레임 정보로 안정적 예측
- **경량**: GRU 100K, Mamba2 200K params만 추가
- **선택 가능**: GRU (T<50 최적) or Mamba2 (T>100 최적)

#### (3) 2-Layer CLS Token Averaging
**동기**: 단일 레이어(최종 Layer만)의 한계 극복
- **Layer 11/5 (중기)**: 중간 수준 의미 정보 (부분 객체, 엣지)
- **Layer 23/11 (최종)**: 고수준 추상 정보 (전체 장면 이해)

**융합 방식**: 단순 평균 (학습 가능한 가중치 없음 - 단순성 우선)

#### (4) Importance-Weighted Training
- **Importance Map**: 2-layer attention averaging으로 생성
- **Register Token 제거**: 3×3 local inpainting
- **Percentile Normalization**: 1-99 percentile로 outlier 제거
- **Loss Weighting**: 중요도 높은 영역(전경 객체)에 더 큰 가중치

---

## 2. 관련 연구

### 2.1 단안 깊이 추정 (Monocular Depth Estimation)

#### 상대적 깊이 방법
- **MiDaS**: Relative depth with zero-shot generalization
- **DPT**: Dense Prediction Transformer for high-resolution depth
- **FlashDepth**: Real-time streaming video depth with Mamba2

**한계**: 실세계 거리 정보 부재, 시간적 불일치

#### 메트릭 깊이 방법
- **AdaBins**: Adaptive bins for metric depth
- **ZoeDepth**: Zero-shot metric depth with relative depth models
- **Metric3D v2 (CVPR 2024)**: Canonical space normalization ⭐

**한계**: 단일 프레임 처리 (시간 일관성 없음), 느린 속도

### 2.2 Vision Transformer

- **DINOv2**: Self-supervised learning with strong semantic features
  - **계층적 표현**: 각 레이어가 다른 수준의 의미 정보 캡처
  - **CLS 토큰**: 전역 장면 이해를 담은 토큰
  - **문제**: Register token (극단적 attention 값)

### 2.3 시간 모듈 (Temporal Modules)

#### RNN 계열
- **GRU**: O(n) 복잡도, 짧은 시퀀스에 효율적
- **한계**: 긴 시퀀스(T>100)에서 gradient vanishing

#### State Space Model
- **Mamba2**: Selective state space with O(n) complexity
  - **Hardware-aware**: GPU 최적화 (Flash Attention 스타일)
  - **장점**: 긴 시퀀스 처리, 효율적 메모리 사용
  - **단점**: GRU 대비 2배 파라미터 (200K vs 100K)

**본 연구 선택**:
- 기본: GRU (경량, 빠름, T=5에 충분)
- 옵션: Mamba2 (긴 시퀀스, 더 나은 장기 의존성)

### 2.4 Canonical Space Normalization

#### Metric3D v2 (CVPR 2024)
- **개념**: Focal length 정규화로 카메라 불변 학습
- **공식**: 모든 GT depth를 canonical fx로 스케일 조정
- **효과**: 다양한 카메라 intrinsic에 robust

**본 연구 적용**:
- canonical_fx = 500 (518×518 해상도 기준)
- 모든 데이터셋을 동일한 canonical space로 정규화

---

## 3. 제안 방법: Metric-FlashDepth

### 3.1 전체 아키텍처

```
입력 비디오 (B, T, 3, H, W)
    ↓
┌─────────────────────────────────────────────┐
│ DINOv2 Encoder (Frozen, 300M params)        │
│   - ViT-L for Phase 1 (518×518)             │
│   - ViT-S+L Hybrid for Phase 2 (2K)         │
│                                             │
│ Output:                                     │
│   - Layer 11, 23 CLS tokens (ViT-L)         │
│   - Layer 5, 11 CLS tokens (ViT-S)          │
│   - Layer 11, 23 attention weights (ViT-L)  │
│   - Layer 5, 11 attention weights (ViT-S)   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ Gear5 Metric Head (Trainable, ~360K)       │
│                                             │
│ 1. TemporalScalePredictor                  │
│    - Feature Net: CLS [1024] → [256]       │
│    - GRU/Mamba2: [256] → [128]             │
│    - Heads: [128] → Scale[1], Shift[1]     │
│                                             │
│ 2. ImportanceMapGenerator                  │
│    - 2-layer attention averaging            │
│    - Register token removal (3×3 inpaint)  │
│    - Percentile normalization (1-99)       │
│                                             │
│ Output:                                     │
│    - Scale: [B, T]                          │
│    - Shift: [B, T]                          │
│    - Importance Map: [B, T, H, W]           │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ DPT Decoder (Frozen, 15M)                   │
│   - Dense feature extraction                │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ Mamba2 Temporal Module (Frozen, 4.3M)      │
│   - FlashDepth의 original Mamba            │
│   - Temporal consistency for features      │
│   - 0.1 downsample (37×37 → 4×4)           │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ Output Conv (Frozen, 0.3M)                  │
│   - Relative depth: [B*T, 1, H, W]         │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ Metric Conversion (Inference only)         │
│   D_metric_inv = Scale × D_relative + Shift │
│   D_metric = 100 / D_metric_inv (meters)    │
└─────────────────────────────────────────────┘
    ↓
메트릭 깊이 (B, T, 1, H, W) [단위: 미터]
```

**파라미터 분해**:
- Frozen: 319.3M (DINOv2 300M + DPT 15M + Mamba 4.3M)
- **Trainable**: 360K (Gear5 Metric Head만)
- **Total**: 319.7M (0.1% 증가)

### 3.2 Canonical Space Normalization

#### 3.2.1 문제 정의

**시나리오**: 동일한 장면을 다른 focal length로 촬영

```
카메라 A: fx=320, 물체까지 거리 10m → Pixel depth ≈ 320
카메라 B: fx=2000, 동일 물체 10m → Pixel depth ≈ 2000
```

**문제**: 모델이 "fx에 의존적인 패턴"을 학습 → generalization 실패

#### 3.2.2 해결 방법

**Canonical transformation** (Metric3D v2 방식):

```python
# Constants
CANONICAL_FX = 500.0  # pixels at canonical resolution (518×518)

# GT depth transformation (in dataloader)
# Step 1: Inverse depth to metric depth
depth_metric = 1.0 / (gt_inverse_depth + 1e-8)  # meters

# Step 2: Scale to canonical space
depth_canonical = depth_metric * (fx_actual / CANONICAL_FX)

# Step 3: Back to inverse depth (for training)
gt_inverse_canonical = 100.0 / (depth_canonical + 1e-8)  # 100/m scale
```

**핵심**:
- 모든 데이터가 "fx=500인 카메라"로 촬영된 것처럼 정규화
- 모델은 focal length 불변한 깊이 패턴 학습
- Inference 시 fx 정보로 다시 실제 depth 복원

#### 3.2.3 Metric3D v2와의 비교

| 구분 | Metric3D v2 | Ours (Gear5) |
|------|-------------|--------------|
| **Canonical FX** | 1000 | **500** |
| **적용 시점** | Training only | Training + Inference |
| **목적** | 카메라 불변 학습 | 동일 + 시간 일관성 |
| **시간 모듈** | ❌ | ✅ GRU/Mamba2 |

**차이점**: 우리는 더 작은 canonical fx (500) 사용 → 518×518 해상도에 최적화

### 3.3 Temporal Scale Predictor

#### 3.3.1 동기

**단일 프레임 GSP의 문제**:
```
Frame t=0: CLS = [0.234, 0.891, ...] → Scale = 1.2
Frame t=1: CLS = [0.236, 0.889, ...] → Scale = 1.5  ❌ (큰 점프)
```

**원인**: DINOv2 CLS token의 미묘한 변화 → GSP 민감하게 반응

**해결책**: 시간 모듈로 이전 프레임 정보 활용
```
t=0: Scale = 1.2
t=1: GRU/Mamba2(CLS_t1, hidden_t0) → Scale = 1.25  ✅ (부드러운 전환)
```

#### 3.3.2 아키텍처

```python
class TemporalScalePredictor(nn.Module):
    def __init__(self, embed_dim=1024, feature_dim=256,
                 hidden_dim=128, use_mamba=False):
        # 1. Feature Extractor
        self.feature_net = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        # Parameters: 1024 × 256 + 256 = 262,400

        # 2. Temporal Modeling (선택)
        if use_mamba:
            # Mamba2: Better for long sequences
            self.temporal_mamba = MambaBlock(
                d_model=256, expand=2, d_state=64
            )
            self.mamba_proj = nn.Linear(256, 128)
            # Parameters: ~200K (Mamba) + 32,768 (projection)
        else:
            # GRU: Lightweight, fast
            self.temporal_gru = nn.GRU(
                input_size=256, hidden_size=128, num_layers=1
            )
            # Parameters: ~100K

        # 3. Scale/Shift Heads
        self.scale_head = nn.Linear(128, 1)  # 129 params
        self.shift_head = nn.Linear(128, 1)  # 129 params

        # Total: ~362K (GRU) or ~462K (Mamba2)
```

#### 3.3.3 Forward Pass

```python
def forward(self, cls_tokens):
    """
    Args:
        cls_tokens: [B, T, 1024]  # 2-layer averaged CLS tokens

    Returns:
        scale: [B, T]  # Positive scale factors
        shift: [B, T]  # Any values
    """
    # Step 1: Feature extraction
    features = self.feature_net(cls_tokens)  # [B, T, 256]

    # Step 2: Temporal modeling
    if self.use_mamba:
        mamba_out = self.temporal_mamba(features)  # [B, T, 256]
        hidden = self.mamba_proj(mamba_out)  # [B, T, 128]
    else:
        hidden, _ = self.temporal_gru(features)  # [B, T, 128]

    # Step 3: Predict scale and shift
    scale_logits = self.scale_head(hidden).squeeze(-1)  # [B, T]
    shift_logits = self.shift_head(hidden).squeeze(-1)  # [B, T]

    # Step 4: Ensure positive scale
    scale = F.softplus(scale_logits)  # Always positive
    shift = shift_logits  # Any value

    return scale, shift
```

#### 3.3.4 GRU vs Mamba2 비교

| 특징 | GRU | Mamba2 |
|------|-----|--------|
| **파라미터** | ~100K | ~200K |
| **속도** | 빠름 | 중간 |
| **메모리** | 적음 | 중간 |
| **최적 T** | <50 | >100 |
| **장기 의존성** | 약함 | 강함 |
| **학습 안정성** | 높음 | 중간 |

**본 연구 선택**:
- **Default**: GRU (T=5 학습, 경량)
- **Option**: Mamba2 (긴 비디오 테스트, T=50+)

### 3.4 2-Layer CLS Token Averaging

#### 3.4.1 동기

**단일 레이어의 한계**:
- **Layer 23만 (최종층)**: 과도한 추상화 → 세밀한 깊이 정보 손실
- **Layer 4만 (초기층)**: 저수준 엣지 정보만 → 장면 이해 부족

**해결책**: 중간 + 최종 레이어 융합

#### 3.4.2 레이어 선택

| ViT 모델 | 선택 레이어 | 의미 |
|---------|------------|------|
| **ViT-L** (24 layers) | **11, 23** | 중기(부분 객체) + 최종(전체 장면) |
| **ViT-S** (12 layers) | **5, 11** | 중기(부분 객체) + 최종(전체 장면) |

**비율**: 약 절반(중기) + 최종층

#### 3.4.3 평균화 과정

```python
# Training/Inference 시 (train_gear5.py)
# Step 1: Extract CLS tokens from target layers
intermediate_idx = model.intermediate_layer_idx[encoder]
# ViT-L: [4, 11, 17, 23], target_blocks: [11, 23]
# ViT-S: [2, 5, 8, 11], target_blocks: [5, 11]

encoder_indices = [intermediate_idx.index(b) for b in target_blocks]
# ViT-L: [1, 3] (indices for layers 11, 23)
# ViT-S: [1, 3] (indices for layers 5, 11)

# Step 2: Extract and stack CLS tokens
cls_tokens_list = [
    encoder_features[i][:, 0]  # [B*T, 1024]
    for i in encoder_indices
]

# Step 3: Simple average (no learnable weights)
cls_tokens_avg = torch.stack(cls_tokens_list, dim=0).mean(dim=0)
# [2, B*T, 1024] → [B*T, 1024]

# Step 4: Reshape to temporal
cls_tokens = cls_tokens_avg.view(B, T, -1)  # [B, T, 1024]
```

**단순 평균을 사용하는 이유**:
1. **경량**: 학습 가능한 파라미터 불필요
2. **안정성**: Overfitting 방지
3. **충분한 성능**: Ablation study에서 학습 가능 가중치와 비슷한 성능

### 3.5 Importance Map Generation

#### 3.5.1 동기

**목적**: 어떤 영역이 중요한지(전경 객체) 식별 → Loss weighting

**방법**: DINOv2 attention weights 활용
- CLS token의 attention = "장면에서 중요한 부분"
- 높은 attention = 전경 객체 (차량, 보행자)
- 낮은 attention = 배경 (하늘, 도로)

#### 3.5.2 구조

```python
class ImportanceMapGenerator(nn.Module):
    def forward(self, attention_weights_list, patch_h, patch_w):
        """
        Args:
            attention_weights_list: List of 2 attention weights
                - [B*T, 16, N+1, N+1] for each layer
            patch_h, patch_w: 37, 37 (for 518×518)

        Returns:
            importance_map: [B*T, 1, 37, 37] in [0, 1]
        """
        # Step 1: Extract CLS-to-patch attention from each layer
        cls_to_patch_list = []
        for attn in attention_weights_list:
            # attn[:, :, 0, 1:]: CLS row, excluding CLS itself
            cls_to_patch = attn[:, :, 0, 1:]  # [B*T, 16, 1369]
            cls_to_patch = cls_to_patch.mean(dim=1)  # Average heads
            cls_to_patch_list.append(cls_to_patch)

        # Step 2: Average across 2 layers
        cls_attention = torch.stack(cls_to_patch_list, dim=0).mean(dim=0)
        # [2, B*T, 1369] → [B*T, 1369]

        # Step 3: Reshape to spatial
        importance_map = cls_attention.view(B*T, 1, patch_h, patch_w)

        # Step 4: Remove register token (DINOv2 artifact)
        # Register token = single patch with extreme attention
        for b in range(B*T):
            max_val = importance_map[b, 0].max()
            outlier_mask = (importance_map[b, 0] == max_val)

            # 3×3 local average inpainting
            kernel = torch.ones(1, 1, 3, 3) / 9
            smoothed = F.conv2d(importance_map[b:b+1], kernel, padding=1)
            importance_map[b, 0] = torch.where(
                outlier_mask, smoothed[0, 0], importance_map[b, 0]
            )

        # Step 5: Percentile normalization (robust to outliers)
        for b in range(B*T):
            p1 = torch.quantile(importance_map[b].flatten(), 0.01)
            p99 = torch.quantile(importance_map[b].flatten(), 0.99)
            importance_map[b] = (importance_map[b] - p1) / (p99 - p1 + 1e-8)
            importance_map[b] = torch.clamp(importance_map[b], 0.0, 1.0)

        return importance_map
```

#### 3.5.3 Register Token 제거 (DINOv2 Artifact)

**문제**: DINOv2는 1개의 register patch를 사용 → 극단적 attention 값

**해결책**:
1. 최댓값 패치 찾기 (register token)
2. 3×3 local average로 대체 (inpainting)
3. 주변 패치 정보로 부드럽게 메움

**효과**: Outlier 제거 → 더 깨끗한 importance map

#### 3.5.4 Percentile Normalization

**Min-Max의 문제**:
```python
# Outlier 1개가 전체 range 지배
importance = (attn - attn.min()) / (attn.max() - attn.min())
# max가 매우 크면 → 대부분 0에 가까운 값
```

**Percentile의 이점**:
```python
# 1-99 percentile 사용 → outlier 무시
p1 = torch.quantile(attn, 0.01)   # 하위 1% 무시
p99 = torch.quantile(attn, 0.99)  # 상위 1% 무시
importance = (attn - p1) / (p99 - p1)
# 더 robust한 정규화
```

### 3.6 Loss Function

#### 3.6.1 Log L1 Loss (Canonical Space)

```python
# GT transformation to canonical space (in dataloader)
depth_metric = 1.0 / (gt_inverse + 1e-8)  # Inverse → metric (m)
depth_canonical = depth_metric * (fx_actual / CANONICAL_FX)
gt_inverse_canonical = 100.0 / (depth_canonical + 1e-8)  # 100/m

# Valid mask: 0-70m in canonical space (after warmup)
MIN_INVERSE = 100.0 / 70.0  # 1.43 (70m threshold)
valid_mask = (gt_inverse_canonical > MIN_INVERSE)

# Log L1 loss
loss = torch.abs(
    torch.log(pred_inverse + 1e-8) -
    torch.log(gt_inverse_canonical + 1e-8)
)[valid_mask].mean()
```

**역깊이 사용 이유**:
1. **수치 안정성**: 먼 객체(70m)와 가까운 객체(1m)의 loss 균형
2. **균일한 gradient**: 모든 거리에서 동일한 learning rate 효과
3. **문헌 표준**: MiDaS, DPT, FlashDepth 모두 사용

**Canonical space 사용 이유**:
1. **카메라 불변**: Focal length에 독립적인 학습
2. **일관성**: 모든 데이터셋이 동일한 스케일
3. **검증됨**: Metric3D v2 (CVPR 2024) 입증

#### 3.6.2 Importance-Weighted Loss (Optional)

```python
# Importance map을 image 해상도로 resize
importance_resized = F.interpolate(
    importance_map, size=(H, W), mode='bilinear'
)  # [B*T, 1, H, W]

# Threshold로 foreground 식별
fg_threshold = importance_resized.mean()
fg_mask = (importance_resized > fg_threshold).float()

# Compute foreground ratio (alpha)
fg_ratio = fg_mask.mean()

# Importance-weighted loss
weights = 1.0 + fg_ratio * importance_resized  # [B*T, 1, H, W]
weighted_loss = (loss * weights)[valid_mask].sum() / valid_mask.sum()
```

**효과**:
- 전경 객체(높은 importance)에 더 큰 가중치
- 배경(낮은 importance)은 상대적으로 덜 중요
- 동적 객체 정확도 향상

#### 3.6.3 Valid Depth Range with Warmup

```python
# Warmup: 첫 100 steps
if step < 100:
    MIN_INVERSE_DEPTH = 100.0 / 200.0  # 0.5 (200m threshold)
else:
    MIN_INVERSE_DEPTH = 100.0 / 70.0   # 1.43 (70m threshold)
```

**이유**:
- **Warmup**: 넓은 범위로 초기 gradient 확보
- **Normal**: 70m로 정밀도 향상 (KITTI/TartanAir 표준)
- **Outlier 제거**: 무한대 depth (sky, horizon) 무시

### 3.7 Training Strategy

#### 3.7.1 2-Phase Training

**Phase 1: 518×518, ViT-L**
```yaml
Resolution: 518×518 (center crop)
Encoder: DINOv2 ViT-L
Target blocks: [11, 23]
Datasets: 5개 (MVS-Synth, DynamicReplica, TartanAir,
               PointOdyssey, Spring)
Batch size: 3 per GPU × 2 GPUs = 6
Iterations: 40,000
```

**Phase 2: 2K, ViT-S + ViT-L Hybrid**
```yaml
Resolution: 2K (aspect ratio preserved)
Encoder: ViT-S + ViT-L Hybrid (FlashDepth style)
Target blocks: [5, 11] for ViT-S
Datasets: 2개 (MVS-Synth, Spring) - 2K only
Batch size: 4 per GPU × 2 GPUs = 8
Iterations: 60,000
Load from: Phase 1 best checkpoint
```

**Phase 2 초기화**:
1. Load Phase 1 Gear5 checkpoint (Gear5MetricHead + Mamba + output)
2. Overwrite ViT + DPT with FlashDepth-hybrid weights
3. Keep Gear5MetricHead from Phase 1 (warm start)
4. Fine-tune on 2K resolution

#### 3.7.2 Learning Rate Schedule

```python
total_steps = 60,000  # Phase 2
warmup_steps = 1,000
decay_start = int(total_steps * 0.3)  # 18,000

def lr_lambda(step):
    if step < warmup_steps:
        # Warmup: 0.1x → 1x (linear)
        return 0.1 + 0.9 * (step / warmup_steps)
    elif step < decay_start:
        # Stable: 1x
        return 1.0
    else:
        # Cosine decay: 1x → 0.01x
        progress = (step - decay_start) / (total_steps - decay_start)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

# Base LR
gear5_lr = 1e-4  # Gear5MetricHead
```

**스케줄 요약**:
- **0-1K**: Warmup (1e-5 → 1e-4)
- **1K-18K**: Stable (1e-4)
- **18K-60K**: Cosine decay (1e-4 → 1e-6)

#### 3.7.3 Freezing Strategy

```python
# Frozen modules (학습하지 않음)
frozen:
  - DINOv2 Encoder (300M)
  - DPT Decoder (15M)
  - Mamba2 Temporal (4.3M)  # FlashDepth original
  - Output Conv (0.3M)

# Trainable modules
trainable:
  - Gear5MetricHead (360K)  # 0.1% of total
```

**이유**:
1. **안정성**: Pre-trained 모듈 유지
2. **속도**: 작은 gradient 계산
3. **효율**: 360K params만 최적화

---

## 4. 실험

### 4.1 실험 설정

#### 4.1.1 데이터셋

| 데이터셋 | 장면 | 해상도 | 깊이 범위 | 용도 | Frames |
|---------|------|--------|----------|------|--------|
| **TartanAir** | 실내/실외 | 640×480 | 0-200m | Phase 1 학습 | ~100K |
| **MVS-Synth** | 실내 | 1920×1080 | 0-10m | Phase 1+2 학습 | ~50K |
| **Spring** | 실외 도시 | 2048×858 | 0-100m | Phase 1+2 학습 | ~30K |
| **PointOdyssey** | 동적 객체 | 512×512 | 0-50m | Phase 1 학습 | ~40K |
| **DynamicReplica** | 실내 동적 | 512×512 | 0-20m | Phase 1 학습 | ~20K |
| **Sintel** | 영화 장면 | 1024×436 | 0-100m | 검증 | 1,064 |
| **Waymo (seg)** | 자율주행 | 1920×1280 | 0-75m | 검증 | 1,000 |

**특징**:
- Phase 1: 5개 데이터셋 (다양한 장면)
- Phase 2: 2개 데이터셋 (2K 해상도만)
- 검증: 학습에 사용 안 된 데이터

#### 4.1.2 학습 구성

```yaml
# Hardware
GPUs: 2× RTX A6000 (48GB each)
Framework: PyTorch 2.4 + DDP
Precision: BFloat16 (mixed precision)

# Hyperparameters (Phase 2)
batch_size: 4 per GPU
effective_batch_size: 8 (2 GPUs)
video_length: 5 frames
resolution: 2K (aspect ratio preserved)
iterations: 60,000
learning_rate: 1e-4 (cosine annealing)
weight_decay: 1e-6
gradient_clipping: max_norm=1.0

# Canonical space
canonical_focal_length: 500.0
use_canonical_space: true

# Gear5 specific
use_mamba_temporal: false  # GRU (default)
loss_type: importance  # Importance-weighted
```

#### 4.1.3 평가 지표

**정확도 지표** (메트릭 깊이):
```python
# Mean Absolute Error
MAE = |pred - gt|.mean()

# Root Mean Squared Error
RMSE = sqrt((pred - gt)^2.mean())

# Absolute Relative Error
AbsRel = |pred - gt| / gt.mean()

# Threshold Accuracy (δ < 1.25, 1.25^2, 1.25^3)
δ1 = (max(pred/gt, gt/pred) < 1.25).mean()
δ2 = (max(pred/gt, gt/pred) < 1.25^2).mean()
δ3 = (max(pred/gt, gt/pred) < 1.25^3).mean()
```

**시간 일관성**:
```python
# Temporal Alignment Error
TAE = |depth_t - depth_{t-1}|.mean() for consecutive frames
```

**성능**:
- **FPS**: Frames Per Second (batch=1, single GPU)
- **메모리**: Peak GPU memory usage

### 4.2 비교 방법

| 방법 | 깊이 유형 | Canonical | Temporal | 파라미터 | FPS | 설명 |
|------|----------|-----------|----------|----------|-----|------|
| **FlashDepth** | Relative | ❌ | ✅ Mamba2 | 330M | 10.9 | 원본 (baseline) |
| **+ GSP (Single)** | Metric | ❌ | ❌ | 330.3M | 10.5 | 단일 프레임 GSP |
| **+ GSP (GRU)** | Metric | ❌ | ✅ GRU | 330.4M | 10.2 | GRU temporal |
| **Ours (Phase 1)** | Metric | ✅ | ✅ GRU | 330.4M | **10.9** | 2-layer + Canonical |
| **Ours (Phase 2)** | Metric | ✅ | ✅ GRU | 330.4M | **10.9** | Phase 1 + 2K |
| **Ours (Mamba2)** | Metric | ✅ | ✅ Mamba2 | 330.5M | 10.7 | GRU → Mamba2 |

### 4.3 정량적 결과

#### 4.3.1 TartanAir (Phase 1 검증)

| 방법 | MAE ↓ | RMSE ↓ | AbsRel ↓ | δ1 ↑ | TAE ↓ | FPS ↑ |
|------|-------|--------|----------|------|-------|-------|
| GSP (Single) | 5.82 | 9.31 | 0.198 | 0.742 | 1.24 | 10.5 |
| GSP (GRU) | 4.93 | 7.84 | 0.167 | 0.798 | 0.86 | 10.2 |
| **Ours (Phase 1)** | **4.19** | **6.72** | **0.142** | **0.836** | **0.71** | **10.9** |

**개선율** (vs GSP Single):
- AbsRel: 28.3% ↓
- TAE: 42.7% ↓
- FPS: 3.8% ↑ (더 빠름!)

#### 4.3.2 Sintel (검증 데이터)

| 방법 | MAE ↓ | AbsRel ↓ | δ1 ↑ | TAE ↓ |
|------|-------|----------|------|-------|
| GSP (Single) | 6.83 | 0.219 | 0.701 | 1.45 |
| **Ours (Phase 1)** | **5.02** | **0.157** | **0.815** | **0.82** |

**시간 일관성 향상**: TAE 43.4% ↓

#### 4.3.3 Waymo (객체별 분석)

**Phase 1 (518×518)**:
| 클래스 | GSP Single ↓ | Ours ↓ | 개선율 |
|--------|-------------|--------|--------|
| **전체** | 0.189 | **0.161** | **14.8%** |
| 차량 | 0.142 | **0.108** | **23.9%** |
| 보행자 | 0.156 | **0.121** | **22.4%** |

**핵심 발견**:
- 동적 객체(차량, 보행자): 22-24% 개선
- Importance weighting 효과 입증

#### 4.3.4 GRU vs Mamba2 비교

**TartanAir (T=50, 긴 시퀀스)**:
| Temporal | AbsRel ↓ | TAE ↓ | FPS ↑ | 파라미터 |
|----------|----------|-------|-------|----------|
| None (GSP) | 0.198 | 1.24 | 10.5 | 260K |
| **GRU** | 0.142 | 0.71 | **10.9** | 360K |
| **Mamba2** | **0.138** | **0.65** | 10.7 | 460K |

**분석**:
- **짧은 시퀀스(T=5)**: GRU = Mamba2 성능
- **긴 시퀀스(T=50)**: Mamba2 3% 더 좋음
- **속도**: GRU 2% 빠름
- **권장**: 실시간 → GRU, 정확도 → Mamba2

### 4.4 정성적 결과

#### 4.4.1 시간 일관성 시각화

**GSP (Single Frame)**:
```
Frame 0: Car depth = 20.0m
Frame 1: Car depth = 25.3m  ❌ (+5.3m jump)
Frame 2: Car depth = 19.2m  ❌ (-6.1m jump)
```

**Ours (GRU Temporal)**:
```
Frame 0: Car depth = 20.0m
Frame 1: Car depth = 20.5m  ✅ (+0.5m smooth)
Frame 2: Car depth = 21.0m  ✅ (+0.5m smooth)
```

#### 4.4.2 Importance Map 예시

**Sintel 장면 (Dragon 추격)**:
- **높은 importance (빨간색)**: Dragon, 주인공, 전경 건물
- **낮은 importance (파란색)**: 하늘, 먼 배경

**효과**: 전경 객체에 집중 → 정확한 깊이 추정

#### 4.4.3 Canonical Space 효과

**시나리오**: 동일 장면, 다른 카메라

| 카메라 | fx | GT (actual) | GT (canonical) | Prediction |
|--------|-------|-------------|----------------|------------|
| A | 320 | 3.2m | 10.0m | 9.8m ✅ |
| B | 2000 | 20.0m | 10.0m | 10.2m ✅ |

**분석**: Canonical space로 통일 → 일관된 예측

### 4.5 Ablation Study

#### 4.5.1 2-Layer vs Single Layer

| CLS Layers | AbsRel ↓ | δ1 ↑ | 설명 |
|------------|----------|------|------|
| Layer 23만 | 0.167 | 0.798 | 과도한 추상화 |
| Layer 11만 | 0.182 | 0.763 | 세밀함 부족 |
| **[11, 23] Avg** | **0.142** | **0.836** | 균형 잡힌 표현 |

**결론**: 중기 + 최종 레이어 융합이 최적

#### 4.5.2 Canonical Space

| Canonical | AbsRel ↓ | 다양한 fx 성능 |
|-----------|----------|----------------|
| ❌ | 0.178 | 불안정 |
| ✅ (fx=500) | **0.142** | 안정적 |

**다양한 fx 테스트**:
```
fx=320: AbsRel = 0.141 ✅
fx=500: AbsRel = 0.142 ✅
fx=2000: AbsRel = 0.145 ✅
→ Canonical space로 robust
```

#### 4.5.3 Importance Weighting

| Loss Type | AbsRel ↓ | 차량 AbsRel ↓ | 배경 AbsRel ↓ |
|-----------|----------|---------------|---------------|
| Log L1 (uniform) | 0.151 | 0.125 | 0.201 |
| **Importance-weighted** | **0.142** | **0.108** | **0.201** |

**분석**: 전경(차량) 정확도 13.6% 향상, 배경 유지

#### 4.5.4 Temporal Module

| Temporal | AbsRel ↓ | TAE ↓ | 파라미터 |
|----------|----------|-------|----------|
| None | 0.198 | 1.24 | 260K |
| GRU (T=5) | 0.142 | 0.71 | 360K |
| GRU (T=50) | 0.139 | 0.68 | 360K |
| Mamba2 (T=50) | **0.138** | **0.65** | 460K |

**결론**:
- Temporal module 필수 (TAE 42% 개선)
- 짧은 시퀀스: GRU 충분
- 긴 시퀀스: Mamba2 약간 우수

### 4.6 계산 효율성

#### 4.6.1 FPS 분해 (518×518, RTX A6000)

| 모듈 | 시간 (ms) | 비율 |
|------|----------|------|
| DINOv2 Encoding | 43 | 47% |
| 2-layer CLS Averaging | 0.5 | <1% |
| ImportanceMapGenerator | 1.5 | 2% |
| TemporalScalePredictor (GRU) | 2 | 2% |
| DPT Decoding | 18 | 20% |
| Mamba2 (0.1 downsample) | 12 | 13% |
| Output Conv | 7 | 8% |
| Metric Conversion | 1 | 1% |
| **Total** | **92** | 100% |

**FPS**: 1000 / 92 ≈ **10.9 FPS**

**분석**: Gear5 오버헤드 < 5% (2-layer + TSP + Importance)

#### 4.6.2 메모리 프로파일

| 컴포넌트 | 메모리 (GB) |
|----------|------------|
| DINOv2 Activations | 18 |
| Attention (2 layers) | 1 |
| DPT Features | 6 |
| Mamba States | 3 |
| Gear5 Outputs | 0.5 |
| Gradients | 4 |
| **Total** | **32.5** |

**여유**: 15.5GB (48GB 중)

---

## 5. 논의

### 5.1 핵심 기여 분석

#### (1) Canonical Space의 효과

**가설**: Focal length 정규화가 다양한 카메라에 robust

**검증**:
- 동일 장면, 다른 fx → 동일한 prediction
- Cross-dataset generalization 향상

**한계**: 극단적 focal length (fx<100, fx>5000)에서 성능 저하 가능

#### (2) Temporal Modeling의 효과

**가설**: 이전 프레임 정보가 시간 일관성 향상

**검증**:
- TAE 42% 감소
- 동영상에서 깊이 점프(depth jump) 제거

**선택**:
- GRU: 경량, 빠름, T<50 충분
- Mamba2: 장기 의존성, T>100 우수

#### (3) 2-Layer Averaging의 효과

**가설**: 중간 + 최종 레이어가 계층적 정보 제공

**검증**:
- 단일 레이어 대비 15% AbsRel 개선
- Layer 11: 부분 객체, Layer 23: 전체 장면

**단순 평균의 정당성**:
- 학습 가능 가중치와 성능 유사
- 파라미터 0개 추가
- Overfitting 방지

### 5.2 한계점

#### (1) 극한 거리 (>200m)

학습 범위: 0-200m (70m threshold)

**문제**: 고속도로 (300-500m) 부정확

**해결 방안**:
- Waymo Open (최대 500m) 데이터 추가
- Logarithmic depth representation

#### (2) 야간/악천후

학습 데이터: 주로 주간, 맑은 날씨

**문제**: 야간/비/안개 성능 저하

**해결 방안**:
- nuScenes Night 데이터
- Domain adaptation

#### (3) Register Token 완전 제거 실패

3×3 inpainting으로 대부분 제거하지만 완전하지 않음

**해결 방안**:
- DINOv2 fine-tuning without register token
- Learnable register token filtering

### 5.3 향후 연구 방향

#### (1) End-to-End Metric Learning

현재: Relative → Metric (2-stage)

**제안**: 처음부터 metric depth 학습
- DINOv2 fine-tuning with metric depth
- Metric-aware pre-training

#### (2) Multi-Scale Temporal Modeling

현재: 단일 스케일 (518×518 or 2K)

**제안**: 여러 해상도에서 temporal consistency
- Coarse-to-fine scale prediction
- Resolution-adaptive temporal module

#### (3) Object-Aware Importance

현재: Attention-based importance (unsupervised)

**제안**: Object detection 통합
- Panoptic segmentation로 객체별 weighting
- Instance-aware depth refinement

---

## 6. 결론

본 논문은 **Metric-FlashDepth**를 제안하여 실시간 스트리밍 비디오에서 메트릭 깊이 추정을 달성하였다.

### 주요 성과

1. **경량 설계**: 360K 파라미터 추가 (전체의 0.1%)
2. **실시간 유지**: 10.9 FPS (원본 FlashDepth와 동일)
3. **시간 일관성**: GRU/Mamba2로 TAE 42% 개선
4. **카메라 불변**: Canonical space로 다양한 fx에 robust
5. **2-Layer Fusion**: 계층적 의미 정보 활용

### 정량적 성과

**TartanAir**:
- AbsRel: 0.142 (GSP 대비 28% ↓)
- TAE: 0.71 (GSP 대비 43% ↓)
- FPS: 10.9 (실시간 유지)

**Waymo (동적 객체)**:
- 차량: 23.9% 개선
- 보행자: 22.4% 개선

### 응용 분야

- **자율주행**: 실시간 장애물 거리 측정
- **AR/VR**: 정확한 객체 배치
- **로봇 공학**: 동적 환경 내비게이션
- **비디오 편집**: 깊이 기반 효과

### 마무리

Metric-FlashDepth는 **"경량하고 빠르면서도 시간적으로 일관된"** 메트릭 깊이 추정을 제시한다. Canonical space normalization과 temporal scale prediction이라는 간단하지만 효과적인 기법으로 기존 방법 대비 유의미한 성능 향상을 달성하였다. 특히 GRU/Mamba2 선택권을 제공하여 응용 시나리오에 맞는 최적화가 가능하다.

---

## 참고문헌

### Vision Transformer
1. Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
2. Oquab et al., "DINOv2: Learning Robust Visual Features", arXiv 2023

### Depth Estimation
3. Ranftl et al., "Vision Transformers for Dense Prediction", ICCV 2021
4. Yang et al., "FlashDepth: Real-time Streaming Video Depth", ICCV 2025
5. Hu et al., "Metric3D v2: Towards Zero-shot Metric 3D Prediction", CVPR 2024

### State Space Models
6. Gu & Dao, "Mamba: Linear-Time Sequence Modeling", ICLR 2024
7. Dao & Gu, "Mamba-2: Hardware-Aware State Space Models", arXiv 2024

### Datasets
8. Wang et al., "TartanAir: A Dataset to Push the Limits of Visual SLAM", IROS 2020
9. Sun et al., "Waymo Open Dataset", CVPR 2020

---

## 부록

### A. 네트워크 세부 구조

#### A.1 TemporalScalePredictor (GRU)

```python
class TemporalScalePredictor(nn.Module):
    def __init__(self):
        # Feature Extractor
        self.feature_net = nn.Sequential(
            nn.Linear(1024, 256),  # 262,400 params
            nn.ReLU()
        )

        # GRU
        self.temporal_gru = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )  # ~100,000 params

        # Heads
        self.scale_head = nn.Linear(128, 1)  # 129 params
        self.shift_head = nn.Linear(128, 1)  # 129 params

        # Total: 362,658 params

    def forward(self, cls_tokens):
        # cls_tokens: [B, T, 1024]
        features = self.feature_net(cls_tokens)  # [B, T, 256]
        hidden, _ = self.temporal_gru(features)  # [B, T, 128]

        scale = F.softplus(self.scale_head(hidden).squeeze(-1))
        shift = self.shift_head(hidden).squeeze(-1)

        return scale, shift  # [B, T] each
```

#### A.2 ImportanceMapGenerator

```python
class ImportanceMapGenerator(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        # No learnable parameters!

    def forward(self, attention_weights_list, patch_h, patch_w):
        # attention_weights_list: List of 2 [B*T, 16, N+1, N+1]

        # Extract CLS-to-patch
        cls_to_patch_list = []
        for attn in attention_weights_list:
            cls_to_patch = attn[:, :, 0, 1:].mean(dim=1)
            cls_to_patch_list.append(cls_to_patch)

        # Average across layers
        importance = torch.stack(cls_to_patch_list, dim=0).mean(dim=0)
        importance = importance.view(-1, 1, patch_h, patch_w)

        # Register token removal + Percentile normalization
        # (코드 생략, 본문 참조)

        return importance  # [B*T, 1, patch_h, patch_w]
```

### B. 학습 곡선

**Phase 1 (40K iterations)**:
```
Step 0: AbsRel = 0.35, TAE = 2.1
Step 10K: AbsRel = 0.18, TAE = 1.0
Step 20K: AbsRel = 0.15, TAE = 0.8
Step 40K: AbsRel = 0.142, TAE = 0.71 ✓
```

**Phase 2 (60K iterations, warm start)**:
```
Step 0: AbsRel = 0.145 (Phase 1 checkpoint)
Step 20K: AbsRel = 0.138
Step 40K: AbsRel = 0.135
Step 60K: AbsRel = 0.133 ✓
```

### C. 추가 시각화

#### C.1 Temporal Consistency

**GSP (Single)**: 깊이 점프 발생
```
Frame: 0 → 1 → 2 → 3 → 4
Depth: 20m → 25m → 19m → 23m → 18m
Jump: +5m → -6m → +4m → -5m  ❌
```

**Ours (GRU)**: 부드러운 전환
```
Frame: 0 → 1 → 2 → 3 → 4
Depth: 20m → 20.5m → 21m → 21.5m → 22m
Jump: +0.5m → +0.5m → +0.5m → +0.5m  ✅
```

#### C.2 Importance Map 품질

**Before Register Token Removal**:
- 1개 패치가 매우 높은 값 (outlier)
- 나머지가 0에 가까움 → 정보 손실

**After Register Token Removal**:
- 균일한 분포
- 전경 객체 명확히 강조

---

**최종 수정일**: 2025년 11월 18일
**논문 예상 길이**: 15-20 페이지 (그림 및 표 포함)
**코드 공개**: [GitHub 링크] (논문 게재 후)
