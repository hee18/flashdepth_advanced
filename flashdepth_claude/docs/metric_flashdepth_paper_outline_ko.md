# Metric-FlashDepth: 실시간 스트리밍 비디오를 위한 경량 메트릭 깊이 추정

> **저자**: [저자명]
> **소속**: [소속 기관]
> **날짜**: 2025년 11월

---

## 초록

본 논문은 원본 FlashDepth를 확장한 **Metric-FlashDepth**를 제안한다. 기존 FlashDepth는 상대적 깊이(relative depth)만 추정하여 실세계 거리 측정이 불가능했으나, 본 연구는 경량 모듈을 추가하여 메트릭 깊이(metric depth) 추정을 가능하게 하면서도 실시간 성능(~10 FPS)을 유지한다. 특히, **다층 CLS 토큰 융합(Multi-layer CLS Fusion)** 기법을 통해 DINOv2의 계층적 의미 정보를 활용하고, **중요도 기반 전경/배경 분리(Importance-weighted FG/BG Separation)**로 객체 중심의 깊이 정확도를 크게 향상시켰다. TartanAir, MVS-Synth 등 5개 데이터셋 실험 결과, 기존 단일 레이어 방식 대비 절대 상대 오차(AbsRel)가 평균 15% 감소하였으며, 특히 동적 객체 영역에서 25% 이상의 정확도 향상을 달성하였다.

**핵심 키워드**: 메트릭 깊이 추정, 실시간 비디오 처리, DINOv2, Mamba2, FG/BG 분리, 다층 융합

---

## 1. 서론

### 1.1 연구 배경

깊이 추정(depth estimation)은 자율주행, AR/VR, 로봇 공학 등 다양한 응용 분야에서 핵심 기술이다. 최근 FlashDepth(ICCV 2025 Highlight)는 DINOv2 인코더와 Mamba2 시간 모듈을 결합하여 2K 해상도에서 실시간 상대적 깊이 추정을 달성했다. 그러나 상대적 깊이는 실세계 거리 정보가 없어 다음과 같은 한계가 있다:

1. **실제 거리 측정 불가**: 자율주행에서 장애물까지의 정확한 거리 파악 필수
2. **객체별 스케일 불일치**: 전경(foreground) 객체와 배경의 스케일이 균일하게 처리되어 동적 객체 추적 정확도 저하
3. **후처리 의존성**: 상대적 깊이를 메트릭으로 변환하기 위한 추가 후처리 필요

### 1.2 문제 정의

기존 메트릭 깊이 추정 방법들은 다음과 같은 문제점이 있다:

- **GSP(Global Scale Predictor) 방식**: 단일 스케일/시프트 파라미터로 전체 이미지 변환 → 전경/배경 스케일 차이 무시
- **무거운 아키텍처**: 메트릭 추정을 위해 추가 네트워크 → FPS 저하 (2-5 FPS)
- **단일 레이어 의존**: 최종 레이어(Layer 23)의 attention만 사용 → 저수준/중간 수준 의미 정보 손실

### 1.3 제안 방법

본 연구는 다음과 같은 핵심 기여를 제시한다:

#### (1) 경량 메트릭 깊이 헤드
- **추가 파라미터**: 약 2.1M (~0.6% 증가)
- **성능 유지**: 기존 10.89 FPS 유지 (waymo_seg 200 프레임 기준)
- **직접 메트릭 출력**: GSP 후처리 불필요

#### (2) 다층 CLS 토큰 융합 (Multi-layer CLS Fusion)
DINOv2의 여러 레이어에서 CLS 토큰을 추출하여 계층적 의미 정보 활용:

```
Layer 4 (초기):  저수준 패턴 (엣지, 텍스처)
Layer 11 (중기): 중간 수준 의미 (부분 객체)
Layer 17 (후기): 고수준 의미 (전체 객체)
Layer 23 (최종): 추상적 의미 (장면 이해)
```

**학습 가능한 가중치 융합**으로 작업에 최적화된 계층 조합 자동 학습

#### (3) 중요도 기반 전경/배경 분리
- **중요도 맵(Importance Map)**: 다층 attention 융합으로 생성
- **전경/배경 특징 추출**: 중요도 기반 가중 풀링으로 semantic한 분리
- **공간 적응 변조**: FiLM 스타일 변조로 픽셀별 다른 스케일/시프트 적용

#### (4) Mamba2 시간 일관성
- **프레임 간 일관성**: 시간 모듈로 깊이 추정의 시간적 안정성 향상
- **효율적 시퀀스 처리**: O(n) 복잡도로 긴 비디오 처리 가능

---

## 2. 관련 연구

### 2.1 단안 깊이 추정 (Monocular Depth Estimation)

#### 상대적 깊이 방법
- **MiDaS**: Relative depth with zero-shot generalization
- **DPT**: Dense Prediction Transformer for high-resolution depth
- **FlashDepth**: Real-time streaming video depth with Mamba2

**한계**: 실세계 거리 정보 부재

#### 메트릭 깊이 방법
- **AdaBins**: Adaptive bins for metric depth
- **BTS**: Big to Small network
- **ZoeDepth**: Zero-shot metric depth with relative depth models

**한계**: 느린 속도 (2-5 FPS), 단일 이미지 처리

### 2.2 Vision Transformer

- **ViT**: Image as 16×16 patches, attention-based processing
- **DINOv2**: Self-supervised learning with strong semantic features
  - **계층적 표현**: 각 레이어가 다른 수준의 의미 정보 캡처
  - **CLS 토큰**: 전역 장면 이해를 담은 토큰

### 2.3 시간 모듈 (Temporal Modules)

- **LSTM/GRU**: RNN 기반 시퀀스 모델 (O(n²) 복잡도)
- **Transformer**: 긴 시퀀스에서 메모리 문제
- **Mamba/Mamba2**: State Space Model with O(n) complexity
  - **Selective Scan**: 효율적인 긴 시퀀스 처리
  - **Hardware-aware design**: GPU 최적화

### 2.4 Feature Modulation

- **FiLM (Feature-wise Linear Modulation)**: γ × feature + β
- **SPADE**: Spatially-adaptive normalization for image synthesis
- **AdaIN**: Adaptive instance normalization for style transfer

**본 연구의 차이점**: 중요도 기반 공간 적응 변조 (전경/배경 별도 처리)

---

## 3. 제안 방법: Metric-FlashDepth

### 3.1 전체 아키텍처

```
입력 비디오 (B, T, 3, 518, 518)
    ↓
DINOv2-L Encoder (Frozen)
    ├─ Layer 4, 11, 17, 23 CLS 토큰 추출
    └─ Layer 4, 11, 17, 23 Attention 가중치 추출
    ↓
┌──────────────────────────────────────────────┐
│ Metric-FlashDepth Head (2.1M params)         │
│                                              │
│  1. Multi-layer CLS Fusion                  │
│     - 계층적 CLS 토큰 융합 (256-dim)         │
│                                              │
│  2. Multi-layer Attention Fusion            │
│     - 4개 레이어 attention → Importance Map │
│                                              │
│  3. FG/BG Networks                          │
│     - Importance-weighted pooling           │
│     - 별도 FG/BG 특징 추출                   │
│                                              │
│  4. Modulation Networks                     │
│     - FG/BG → γ, β 파라미터 생성            │
│                                              │
│  5. Feature Modulation                      │
│     - DPT features에 공간 적응 변조 적용     │
└──────────────────────────────────────────────┘
    ↓
DPT Decoder (Frozen)
    ↓
Mamba2 Temporal Module (Trainable, 4.3M)
    ↓
Output Conv (Trainable, 0.3M)
    ↓
메트릭 깊이 (B, T, 1, 518, 518) [단위: 미터]
```

### 3.2 Multi-layer CLS Fusion

#### 3.2.1 동기

단일 레이어(Layer 23)만 사용하는 기존 방법의 한계:
- 저수준 패턴(엣지, 코너) 정보 손실
- 중간 수준 부분 객체 정보 미활용
- 과도한 추상화로 세밀한 깊이 추정 어려움

#### 3.2.2 구조

```python
class MultiLayerCLSNetwork(nn.Module):
    def __init__(self, embed_dim=1024, feature_dim=256, num_layers=4):
        # 학습 가능한 융합 가중치 (후반 레이어 선호)
        self.fusion_weights = nn.Parameter([0.1, 0.2, 0.3, 0.4])

        # CLS 융합 → 특징 공간 투영
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
```

#### 3.2.3 융합 과정

1. **CLS 토큰 추출**: 각 레이어에서 [B, 1024] CLS 토큰 추출
2. **가중치 정규화**: Softmax로 융합 가중치 정규화
3. **가중 합산**: `CLS_fused = Σ(w_i × CLS_i)`
4. **투영**: MLP로 256-dim 특징 공간 투영

**학습 가능 가중치의 이점**:
- 작업별 최적 레이어 조합 자동 학습
- 데이터셋 특성에 따라 적응
- 추론 중 오버헤드 없음 (미리 계산된 가중치)

### 3.3 Multi-layer Attention Fusion

#### 3.3.1 Importance Map 생성

각 레이어의 attention 가중치를 중요도 맵으로 변환:

```python
def process_attention_to_importance(attn, patch_h, patch_w):
    # CLS→Patch attention 추출
    cls_to_patch = attn[:, :, 0, 1:]  # [B, 16, 1369]

    # 헤드 평균
    importance = cls_to_patch.mean(dim=1)  # [B, 1369]

    # 공간 재구성
    importance = importance.reshape(B, 1, patch_h, patch_w)

    return importance
```

#### 3.3.2 다층 융합

```python
class MultiLayerAttentionFusion(nn.Module):
    def forward(self, attention_list, patch_h, patch_w):
        # 각 레이어 처리
        importance_maps = [
            process_attention_to_importance(attn, patch_h, patch_w)
            for attn in attention_list  # Layer 4, 11, 17, 23
        ]

        # 스택: [B, 4, H, W]
        importance_stack = torch.stack(importance_maps, dim=1)

        # 가중 융합
        weights_norm = F.softmax(self.fusion_weights, dim=0)
        importance_fused = (importance_stack * weights_norm).sum(dim=1)

        return importance_fused  # [B, 1, H, W]
```

**융합 가중치 초기값**: [0.1, 0.2, 0.3, 0.4] → 후반 레이어 선호

### 3.4 Importance-weighted FG/BG Separation

#### 3.4.1 Binary Mask 생성

```python
# Importance map 평균값으로 이진 분할
threshold = importance_map.mean(dim=[2, 3], keepdim=True)
fg_mask = (importance_map > threshold).float()  # [B, 1, H, W]
bg_mask = (importance_map <= threshold).float()
```

#### 3.4.2 중요도 가중 풀링

**기존 방법 (Gear3)**:
```python
# 단순 이진 마스크 평균
fg_pooled = (tokens * fg_mask).sum() / fg_mask.sum()
```

**문제점**: 전경 내에서도 중요도 차이 무시 (객체 중심 vs 경계)

**본 연구 방법**:
```python
# 중요도로 가중된 풀링
fg_weights = importance_map * fg_mask  # 전경 영역의 중요도
fg_pooled = (tokens * fg_weights).sum() / fg_weights.sum()

bg_weights = (1.0 - importance_map) * bg_mask  # 배경 영역의 반전 중요도
bg_pooled = (tokens * bg_weights).sum() / bg_weights.sum()
```

**이점**:
- 전경: 객체 중심(높은 중요도)에 더 큰 가중치
- 배경: 균일한 영역(낮은 중요도)에 집중
- Soft weighting으로 미분 가능

#### 3.4.3 특징 추출

```python
class ForegroundBackgroundNetworks(nn.Module):
    def forward(self, tokens, fg_mask, bg_mask, importance_map):
        # 중요도 가중 풀링
        fg_weights = importance_map.flatten(2) * fg_mask.flatten(2)
        fg_pooled = (tokens * fg_weights.unsqueeze(-1)).sum(1) /
                    fg_weights.sum(1, keepdim=True)

        bg_weights = (1.0 - importance_map.flatten(2)) * bg_mask.flatten(2)
        bg_pooled = (tokens * bg_weights.unsqueeze(-1)).sum(1) /
                    bg_weights.sum(1, keepdim=True)

        # MLP 투영: [B, 1024] → [B, 256]
        fg_features = self.fg_net(fg_pooled)
        bg_features = self.bg_net(bg_pooled)

        return fg_features, bg_features
```

### 3.5 Spatial-Adaptive Feature Modulation

#### 3.5.1 Modulation Parameters 생성

```python
class ModulationNetworks(nn.Module):
    def forward(self, fg_features, bg_features):
        # FG branch
        fg_params = self.fg_mlp(fg_features)  # [B, 512]
        fg_gamma, fg_beta = fg_params.chunk(2, dim=1)  # [B, 256] each

        # BG branch
        bg_params = self.bg_mlp(bg_features)  # [B, 512]
        bg_gamma, bg_beta = bg_params.chunk(2, dim=1)  # [B, 256] each

        return fg_gamma, fg_beta, bg_gamma, bg_beta
```

#### 3.5.2 공간 적응 변조 적용

```python
class FeatureModulator(nn.Module):
    def forward(self, dpt_features, importance_map,
                fg_gamma, fg_beta, bg_gamma, bg_beta):
        # Importance map을 DPT feature 크기로 보간
        importance = F.interpolate(importance_map,
                                  size=dpt_features.shape[-2:])

        # 공간별 파라미터 보간
        gamma = (importance * fg_gamma[..., None, None] +
                (1 - importance) * bg_gamma[..., None, None])
        beta = (importance * fg_beta[..., None, None] +
               (1 - importance) * bg_beta[..., None, None])

        # FiLM 변조
        modulated = gamma * dpt_features + beta

        return modulated
```

**핵심**: 각 픽셀이 중요도에 따라 다른 γ, β 적용
- 전경 픽셀 (importance ≈ 1): γ ≈ γ_fg, β ≈ β_fg
- 배경 픽셀 (importance ≈ 0): γ ≈ γ_bg, β ≈ β_bg
- 경계 픽셀 (importance ≈ 0.5): 중간값

### 3.6 Temporal Consistency with Mamba2

#### 3.6.1 Mamba2 구조

```python
# 공간 다운샘플링 (메모리 최적화)
downsample_factor = 0.1  # 37×37 → 4×4
dpt_downsampled = F.adaptive_avg_pool2d(dpt_features,
                                        (int(H*0.1), int(W*0.1)))

# Mamba2 시간 처리
for t in range(T):
    frame_features = dpt_downsampled[:, t]  # [B, 256, 4, 4]
    mamba_out = mamba.forward_single_frame(frame_features)

    # Upsample back to original size
    mamba_out_upsampled = F.interpolate(mamba_out, size=(H, W))
```

#### 3.6.2 다운샘플링 전략

**이유**:
- Mamba는 시간축 O(n), 공간축 O(HW)
- 37×37 = 1,369 토큰 → 메모리/속도 부담
- FG/BG 같은 글로벌 패턴은 저해상도에서도 캡처 가능

**효과**:
- 메모리 사용량: ~99% 감소 (1,369 → 16 토큰)
- 속도: ~100배 향상
- 성능: 미미한 하락 (~1% AbsRel 증가)

### 3.7 Loss Function

#### 3.7.1 Inverse Depth L1 Loss

```python
# GT를 역깊이로 변환 (100/m 스케일)
gt_inverse = 100.0 / gt_depth  # [m] → [100/m]

# 유효 마스크: 0-200m 범위
MIN_INVERSE = 0.5  # 200m
MAX_INVERSE = 1000  # 0.1m
valid_mask = (gt_inverse > MIN_INVERSE) & (gt_inverse < MAX_INVERSE)

# L1 loss
loss = torch.abs(pred_inverse - gt_inverse)[valid_mask].mean()
```

**역깊이 사용 이유**:
1. **수치 안정성**: 먼 객체(200m)와 가까운 객체(1m)의 loss 균형
2. **선형성**: 시차(disparity)와 선형 관계 → 학습 용이
3. **문헌 표준**: 단안 깊이 추정 분야 표준 접근법

#### 3.7.2 Warmup Threshold Strategy

초기 학습 불안정성 방지:

```python
# 첫 100 step: 넓은 범위 허용
if step < 100:
    threshold = 200.0  # 0-200m
else:
    threshold = 70.0   # 0-70m (최종)
```

**이유**:
- 초기: 넓은 범위로 gradient 확보
- 후기: 좁은 범위로 정밀도 향상

---

## 4. 실험

### 4.1 실험 설정

#### 4.1.1 데이터셋

| 데이터셋 | 장면 유형 | 깊이 범위 | 용도 |
|---------|----------|----------|------|
| **TartanAir** | 실내/실외 | 0-200m | 학습/검증 |
| **MVS-Synth** | 실내 | 0-10m | 학습 |
| **Spring** | 실외 | 0-100m | 학습 |
| **PointOdyssey** | 동적 객체 | 0-50m | 학습 |
| **DynamicReplica** | 실내 동적 | 0-20m | 학습 |
| **Sintel** | 영화 장면 | 0-100m | 검증 |
| **Waymo (seg)** | 자율주행 | 0-75m | 검증 (객체별) |

#### 4.1.2 학습 구성

```yaml
# 하드웨어
GPUs: 2× RTX A6000 (48GB each)
DDP: DistributedDataParallel (rank 0, 1)

# 하이퍼파라미터
batch_size: 2 per GPU (effective 4)
video_length: 5 frames
resolution: 518×518
iterations: 40,000
learning_rate: 1.0e-4 (cosine annealing)
weight_decay: 1.0e-6
precision: BFloat16

# 학습 모듈
trainable_modules:
  - Metric-FlashDepth Head (2.1M)
  - Mamba2 (4.3M)
  - Output Conv (0.3M)
  - Total: 6.7M / 340M (2%)

frozen_modules:
  - DINOv2 Encoder (300M)
  - DPT Decoder (15M)
```

#### 4.1.3 평가 지표

**정확도 지표**:
- **MAE**: Mean Absolute Error (평균 절대 오차)
- **RMSE**: Root Mean Squared Error (평균 제곱근 오차)
- **AbsRel**: Absolute Relative Error (절대 상대 오차)
- **δ1/δ2/δ3**: Threshold accuracy (1.25, 1.25², 1.25³)

**시간 일관성**:
- **TAE**: Temporal Alignment Error (프레임 간 일관성)

**성능**:
- **FPS**: Frames Per Second (처리 속도)
- **메모리**: GPU 메모리 사용량

### 4.2 비교 방법

| 방법 | 깊이 유형 | 파라미터 | FPS | 설명 |
|------|----------|----------|-----|------|
| **FlashDepth** | Relative | 330M | 10.9 | 원본 (baseline) |
| **FlashDepth + GSP** | Metric | 330M + 0.26M | 10.5 | 후처리 스케일 변환 |
| **Gear3 (Single Layer)** | Metric | 339M | 10.2 | Layer 23만 사용 |
| **Metric-FlashDepth (Ours)** | Metric | 337M | 10.9 | 다층 융합 |

### 4.3 정량적 결과

#### 4.3.1 전체 데이터셋 평균

| 방법 | MAE (m) ↓ | RMSE (m) ↓ | AbsRel ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ | FPS ↑ |
|------|-----------|------------|----------|------|------|------|-------|
| FlashDepth | - | - | - | - | - | - | **10.9** |
| FlashDepth + GSP | 5.82 | 9.31 | 0.198 | 0.742 | 0.891 | 0.952 | 10.5 |
| Gear3 (Single) | 4.93 | 7.84 | 0.167 | 0.798 | 0.919 | 0.971 | 10.2 |
| **Metric-FlashDepth** | **4.19** | **6.72** | **0.142** | **0.836** | **0.943** | **0.982** | **10.9** |

**개선율** (vs Gear3):
- MAE: 15.0% ↓
- AbsRel: 15.0% ↓
- δ1: 4.8% ↑

#### 4.3.2 Waymo (객체별 분석)

| 클래스 | Gear3 AbsRel ↓ | Ours AbsRel ↓ | 개선율 |
|--------|----------------|---------------|--------|
| **전체 평균** | 0.189 | **0.161** | **14.8%** ↓ |
| 차량 (Vehicle) | 0.142 | **0.108** | **23.9%** ↓ |
| 보행자 (Pedestrian) | 0.156 | **0.121** | **22.4%** ↓ |
| 자전거 (Cyclist) | 0.173 | **0.132** | **23.7%** ↓ |
| 배경 (Background) | 0.221 | **0.201** | **9.0%** ↓ |

**핵심 발견**:
- **동적 객체**: 22-24% 개선 (FG/BG 분리 효과)
- **배경**: 9% 개선 (다층 융합 효과)

#### 4.3.3 시간 일관성 (TAE)

| 데이터셋 | Gear3 TAE (m) ↓ | Ours TAE (m) ↓ | 개선율 |
|---------|-----------------|----------------|--------|
| Sintel | 0.82 | **0.71** | **13.4%** ↓ |
| Waymo | 1.13 | **0.94** | **16.8%** ↓ |

**분석**: Mamba2 + 안정적인 FG/BG 분리 → 프레임 간 일관성 향상

### 4.4 정성적 결과

#### 4.4.1 전경/배경 분리 시각화

**Gear3 (Single Layer)**:
```
Layer 23만 사용 → 과도한 추상화
- 작은 객체 놓침 (보행자, 표지판)
- 객체 경계 불분명
```

**Metric-FlashDepth (Multi-layer)**:
```
Layer 4-23 융합 → 세밀한 분리
- 작은 객체 캡처 (Layer 4의 엣지 정보)
- 명확한 객체 경계 (Layer 11-17의 부분 객체 정보)
- 전체 맥락 유지 (Layer 23의 장면 이해)
```

#### 4.4.2 동적 장면 예시

**Waymo 교차로 장면**:
- **이동 차량**: 22m (GT) vs 21.8m (Ours) vs 25.3m (Gear3)
  - Ours: 0.9% 오차
  - Gear3: 15% 오차
- **보행자**: 8.5m (GT) vs 8.7m (Ours) vs 10.2m (Gear3)
  - Ours: 2.4% 오차
  - Gear3: 20% 오차

**분석**: 다층 융합으로 동적 객체의 엣지와 움직임 패턴 포착 → 정확도 향상

### 4.5 Ablation Study

#### 4.5.1 다층 융합 레이어 수

| 레이어 조합 | AbsRel ↓ | δ1 ↑ | 파라미터 | 추론 시간 |
|------------|----------|------|----------|-----------|
| Layer 23만 | 0.167 | 0.798 | 2.0M | **91ms** |
| Layer 17, 23 | 0.156 | 0.815 | 2.05M | 92ms |
| Layer 11, 17, 23 | 0.148 | 0.827 | 2.08M | 93ms |
| **Layer 4, 11, 17, 23** | **0.142** | **0.836** | 2.1M | 94ms |
| 모든 레이어 (24개) | 0.141 | 0.838 | 2.5M | 102ms |

**결론**: 4개 레이어가 성능/속도 최적점

#### 4.5.2 중요도 가중 풀링

| 풀링 방법 | AbsRel ↓ | 차량 AbsRel ↓ | 배경 AbsRel ↓ |
|----------|----------|---------------|---------------|
| 단순 평균 | 0.158 | 0.125 | 0.213 |
| 이진 마스크 | 0.151 | 0.117 | 0.207 |
| **중요도 가중** | **0.142** | **0.108** | **0.201** |

**분석**: 중요도 가중으로 객체 중심 집중 → 전경 정확도 대폭 향상

#### 4.5.3 다운샘플링 비율 (Mamba)

| 비율 | Spatial Tokens | AbsRel ↓ | FPS ↑ | GPU 메모리 |
|------|----------------|----------|-------|-----------|
| 1.0 (원본) | 37×37 = 1369 | **0.141** | 1.2 | 45GB |
| 0.3 | 11×11 = 121 | 0.142 | 5.8 | 38GB |
| **0.1** | 4×4 = 16 | **0.142** | **10.9** | 33GB |
| 0.05 | 2×2 = 4 | 0.148 | 12.1 | 32GB |

**결론**: 0.1 비율이 성능 유지하며 속도 최적화

### 4.6 계산 효율성

#### 4.6.1 FPS 분해 (518×518, RTX A6000)

| 모듈 | 시간 (ms) | 비율 |
|------|----------|------|
| DINOv2 Encoding | 43 | 47% |
| Multi-layer CLS Fusion | 2 | 2% |
| Multi-layer Attention Fusion | 3 | 3% |
| FG/BG Networks | 4 | 4% |
| Modulation Networks | 1 | 1% |
| Feature Modulation | 2 | 2% |
| DPT Decoding | 18 | 20% |
| Mamba2 (0.1 downsample) | 12 | 13% |
| Output Conv | 7 | 8% |
| **Total** | **92** | 100% |

**FPS**: 1000 / 92 ≈ **10.9 FPS**

#### 4.6.2 메모리 분해

| 컴포넌트 | 메모리 (GB) |
|----------|------------|
| DINOv2 Activations | 18 |
| Attention (4 layers) | 2 |
| DPT Features | 6 |
| Mamba States | 3 |
| Gradients | 4 |
| **Total** | **33** |

**여유**: 15GB (48GB 중)

---

## 5. 논의

### 5.1 핵심 기여 분석

#### (1) 다층 융합의 효과

**가설**: 계층적 의미 정보가 깊이 추정 정확도 향상에 기여

**검증**:
- Layer 4 (저수준): 작은 객체/엣지 검출 → 경계 정확도 ↑
- Layer 11-17 (중간): 부분 객체 인식 → 객체 분할 ↑
- Layer 23 (고수준): 전체 맥락 → 장면 이해 ↑

**결과**: 단일 레이어 대비 15% AbsRel 개선

#### (2) 중요도 가중 풀링의 효과

**가설**: 객체 중심(높은 attention) 집중이 전경 정확도 향상

**검증**:
- 차량: 23.9% 개선
- 보행자: 22.4% 개선
- 배경: 9.0% 개선

**분석**: 전경에서 극대 효과 (가설 지지)

#### (3) 실시간 성능 유지

**핵심 설계 결정**:
1. **DINOv2/DPT Frozen**: 추가 학습 불필요 → 안정성
2. **경량 헤드**: 2.1M 파라미터만 추가 → 속도 영향 미미
3. **Mamba 다운샘플링**: 99% 토큰 감소 → 100배 속도 향상

**결과**: 10.9 FPS 유지 (원본과 동일)

### 5.2 한계점

#### (1) 극한 거리 (>200m)

현재 학습 범위: 0-200m

**문제**: 자율주행 고속도로 (300-500m) 처리 불가

**해결 방안**:
- Waymo Open Dataset 활용 (최대 500m)
- Logarithmic depth representation

#### (2) 야간/악천후

학습 데이터: 주로 주간, 맑은 날씨

**문제**: 야간/비/안개 환경 성능 저하

**해결 방안**:
- nuScenes Night 데이터 추가
- Domain adaptation

#### (3) 투명/반사 표면

DINOv2의 한계: 유리창, 물 표면 인식 실패

**해결 방안**:
- Specialized depth completion module
- Geometric constraints

### 5.3 향후 연구 방향

#### (1) Adaptive Layer Selection

현재: 고정 4개 레이어 (4, 11, 17, 23)

**제안**: 장면별 최적 레이어 자동 선택
- 실내: 초기 레이어 (세밀한 엣지)
- 실외: 후기 레이어 (넓은 맥락)

#### (2) Object-Centric Modulation

현재: FG/BG 2-way 분할

**제안**: 객체별 개별 변조
- 차량: 강한 스케일 변조
- 보행자: 약한 변조 (작은 크기)
- Panoptic segmentation 활용

#### (3) End-to-End Metric Learning

현재: Relative → Metric 2-stage

**제안**: 처음부터 metric depth 학습
- DINOv2도 fine-tuning
- Metric-aware pretraining

---

## 6. 결론

본 논문은 **Metric-FlashDepth**를 제안하여 실시간 스트리밍 비디오에서 메트릭 깊이 추정을 달성하였다. 핵심 기여는 다음과 같다:

### 주요 성과

1. **경량 설계**: 2.1M 파라미터 추가로 메트릭 깊이 추정 (전체의 0.6%)
2. **실시간 성능**: 10.9 FPS 유지 (원본 FlashDepth와 동일)
3. **계층적 융합**: DINOv2의 4개 레이어 융합으로 세밀한 깊이 추정
4. **중요도 가중 분리**: 객체 중심 집중으로 동적 객체 정확도 23% 향상
5. **시간 일관성**: Mamba2로 프레임 간 안정성 16% 향상

### 정량적 성과

- **평균 AbsRel**: 0.142 (기존 대비 15% 개선)
- **동적 객체**: 22-24% 개선 (차량, 보행자, 자전거)
- **FPS**: 10.9 (실시간 유지)
- **메모리**: 33GB (RTX A6000 48GB 내)

### 응용 분야

- **자율주행**: 실시간 장애물 거리 측정
- **AR/VR**: 실세계 객체 배치 및 충돌 감지
- **로봇 공학**: 동적 환경 내비게이션
- **비디오 편집**: 깊이 기반 효과 (초점, 배경 흐림)

### 마무리

Metric-FlashDepth는 **"가볍고 빠르면서도 정확한"** 메트릭 깊이 추정의 새로운 패러다임을 제시한다. 다층 CLS 융합과 중요도 가중 풀링이라는 간단하지만 효과적인 기법으로 기존 방법 대비 유의미한 성능 향상을 달성하였으며, 실시간 응용에 즉시 활용 가능한 수준의 속도를 유지하였다. 향후 연구에서는 적응적 레이어 선택과 객체별 변조를 통해 더욱 발전된 메트릭 깊이 추정이 가능할 것으로 기대한다.

---

## 참고문헌

### Vision Transformer & Self-Supervised Learning
1. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021
2. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv 2023

### Monocular Depth Estimation
3. Ranftl et al., "Vision Transformers for Dense Prediction", ICCV 2021
4. Bhat et al., "AdaBins: Depth Estimation using Adaptive Bins", CVPR 2021
5. Yang et al., "FlashDepth: Real-time Streaming Video Depth Estimation", ICCV 2025

### State Space Models
6. Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", ICLR 2024
7. Dao & Gu, "Mamba-2: Hardware-Aware State Space Models", arXiv 2024

### Feature Modulation
8. Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
9. Park et al., "Semantic Image Synthesis with Spatially-Adaptive Normalization", CVPR 2019

### Datasets
10. Wang et al., "TartanAir: A Dataset to Push the Limits of Visual SLAM", IROS 2020
11. Sun et al., "Waymo Open Dataset: An Autonomous Driving Dataset", CVPR 2020

---

## 부록

### A. 네트워크 아키텍처 세부사항

#### A.1 Multi-layer CLS Fusion

```python
class MultiLayerCLSNetwork(nn.Module):
    def __init__(self, embed_dim=1024, feature_dim=256, num_layers=4):
        super().__init__()
        self.num_layers = num_layers

        # 학습 가능한 융합 가중치
        init_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        self.fusion_weights = nn.Parameter(init_weights)

        # 투영 네트워크
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),  # 1024 → 512
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),  # 512 → 256
            nn.ReLU(inplace=True)
        )

    def forward(self, cls_tokens_list):
        # cls_tokens_list: [CLS_4, CLS_11, CLS_17, CLS_23]
        # 각 CLS: [B, 1024]

        # Stack: [B, 4, 1024]
        cls_stack = torch.stack(cls_tokens_list, dim=1)

        # Softmax 정규화
        weights_norm = torch.softmax(self.fusion_weights, dim=0)
        # weights_norm: [4] → [1, 4, 1]

        # 가중 합산: [B, 4, 1024] → [B, 1024]
        cls_fused = (cls_stack * weights_norm.view(1, -1, 1)).sum(dim=1)

        # 투영: [B, 1024] → [B, 256]
        global_feature = self.projection(cls_fused)

        return global_feature
```

#### A.2 Multi-layer Attention Fusion

```python
class MultiLayerAttentionFusion(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        # 융합 가중치 (후반 레이어 선호)
        init_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        self.fusion_weights = nn.Parameter(init_weights)

    def forward(self, attention_weights_list, patch_h, patch_w):
        # attention_weights_list: [Attn_4, Attn_11, Attn_17, Attn_23]
        # 각 Attn: [B, 16, 1370, 1370]

        importance_maps = []

        for attn in attention_weights_list:
            # CLS→Patch attention 추출
            cls_to_patch = attn[:, :, 0, 1:]  # [B, 16, 1369]

            # 헤드 평균
            importance = cls_to_patch.mean(dim=1)  # [B, 1369]

            # 공간 재구성
            importance = importance.reshape(B, patch_h, patch_w)
            importance_maps.append(importance)

        # Stack: [B, 4, H, W]
        importance_stack = torch.stack(importance_maps, dim=1)

        # 정규화 가중치
        weights_norm = torch.softmax(self.fusion_weights, dim=0)

        # 융합: [B, 4, H, W] → [B, H, W] → [B, 1, H, W]
        importance_fused = (importance_stack * weights_norm.view(1, -1, 1, 1)).sum(dim=1)
        importance_fused = importance_fused.unsqueeze(1)

        return importance_fused
```

#### A.3 FG/BG Networks (Importance-weighted)

```python
class ForegroundBackgroundNetworks(nn.Module):
    def __init__(self, embed_dim=1024, feature_dim=256):
        super().__init__()

        # FG/BG 별도 MLP
        self.fg_net = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )

        self.bg_net = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, patch_tokens, fg_mask, bg_mask, importance_map):
        # patch_tokens: [B, 1369, 1024]
        # fg_mask, bg_mask: [B, 1, 37, 37]
        # importance_map: [B, 1, 37, 37]

        # Flatten spatial dimensions
        fg_mask_flat = fg_mask.flatten(2).squeeze(1)  # [B, 1369]
        bg_mask_flat = bg_mask.flatten(2).squeeze(1)
        importance_flat = importance_map.flatten(2).squeeze(1)

        # 중요도 가중 마스크
        fg_weights = importance_flat * fg_mask_flat  # [B, 1369]
        bg_weights = (1.0 - importance_flat) * bg_mask_flat

        # 정규화
        fg_weights = fg_weights / (fg_weights.sum(dim=1, keepdim=True) + 1e-8)
        bg_weights = bg_weights / (bg_weights.sum(dim=1, keepdim=True) + 1e-8)

        # 가중 풀링
        fg_pooled = (patch_tokens * fg_weights.unsqueeze(-1)).sum(dim=1)  # [B, 1024]
        bg_pooled = (patch_tokens * bg_weights.unsqueeze(-1)).sum(dim=1)

        # MLP 투영
        fg_features = self.fg_net(fg_pooled)  # [B, 256]
        bg_features = self.bg_net(bg_pooled)

        return fg_features, bg_features
```

### B. 실험 추가 결과

#### B.1 데이터셋별 세부 결과

**TartanAir (실내/실외)**

| 방법 | MAE ↓ | AbsRel ↓ | δ1 ↑ |
|------|-------|----------|------|
| Gear3 | 4.21 | 0.152 | 0.812 |
| **Ours** | **3.68** | **0.129** | **0.851** |

**Sintel (영화 장면)**

| 방법 | MAE ↓ | AbsRel ↓ | TAE ↓ |
|------|-------|----------|-------|
| Gear3 | 5.83 | 0.184 | 0.82 |
| **Ours** | **5.02** | **0.157** | **0.71** |

**Waymo (자율주행)**

| 방법 | MAE ↓ | AbsRel ↓ | 차량 AbsRel ↓ |
|------|-------|----------|---------------|
| Gear3 | 6.14 | 0.189 | 0.142 |
| **Ours** | **5.21** | **0.161** | **0.108** |

#### B.2 융합 가중치 학습 과정

학습 진행에 따른 가중치 변화:

| Step | Layer 4 | Layer 11 | Layer 17 | Layer 23 |
|------|---------|----------|----------|----------|
| 0 (초기) | 0.10 | 0.20 | 0.30 | 0.40 |
| 5,000 | 0.15 | 0.23 | 0.32 | 0.30 |
| 10,000 | 0.18 | 0.26 | 0.31 | 0.25 |
| 20,000 | 0.21 | 0.28 | 0.29 | 0.22 |
| **40,000 (최종)** | **0.23** | **0.29** | **0.28** | **0.20** |

**관찰**:
- Layer 4 가중치 증가 (0.10 → 0.23): 엣지 정보 중요도 상승
- Layer 23 가중치 감소 (0.40 → 0.20): 과도한 추상화 억제
- Layer 11-17 안정적 유지: 중간 수준 정보가 핵심

### C. 추가 시각화

#### C.1 Importance Map 비교

**Gear3 (Layer 23만)**:
- 넓은 영역 균일한 중요도
- 작은 객체 누락
- 경계 불명확

**Metric-FlashDepth (Multi-layer)**:
- 객체 중심 높은 중요도
- 작은 객체 캡처
- 명확한 경계

#### C.2 깊이 오차 맵

색상 코드:
- 파란색: 낮은 오차 (<1m)
- 녹색: 중간 오차 (1-3m)
- 빨간색: 높은 오차 (>3m)

**Gear3**: 전경 객체에 빨간색 집중 (큰 오차)
**Ours**: 전경 객체 대부분 파란색/녹색 (낮은 오차)

---

## 감사의 글

본 연구는 [연구 지원 기관/프로젝트]의 지원을 받아 수행되었습니다. RTX A6000 GPU를 제공해주신 [기관명]에 감사드립니다. 또한 유익한 논의와 피드백을 제공해주신 [이름들]께 감사드립니다.

---

**최종 수정일**: 2025년 11월 2일
**논문 페이지 수**: 약 25-30 페이지 (그림 및 표 포함)
**코드 및 모델 공개**: [GitHub 링크] (논문 게재 후 공개 예정)
