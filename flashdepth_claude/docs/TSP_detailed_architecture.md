# TemporalScalePredictor (TSP) 상세 구조

> **버전**: Gear5 with Mamba2
> **총 파라미터**: 1,253,914 (약 1.25M)
> **학습 로그 검증**: `train_results/results_21/gear_5_mamba/large/training.log`

---

## 1. 전체 구조 개요

```
입력: 2-layer CLS tokens [B, T, 1024]
  ↓
[Feature Extractor] 262,400 params (20.9%)
  ↓
Features [B, T, 256]
  ↓
[MambaBlock] 958,360 params (76.4%)
  ├─ norm1 (LayerNorm)
  ├─ mamba (Mamba2 core)
  ├─ norm2 (LayerNorm)
  └─ mlp (FFN)
  ↓
Mamba output [B, T, 256]
  ↓
[Projection] 32,896 params (2.6%)
  ↓
Hidden states [B, T, 128]
  ↓
[Prediction Heads] 258 params (0.02%)
  ├─ scale_head → scale [B, T]
  └─ shift_head → shift [B, T]
  ↓
출력: scale [B, T], shift [B, T]
```

---

## 2. 모듈별 상세 분석

### 2.1 Feature Extractor (입력 처리)

**목적**: CLS token의 차원을 줄여 temporal modeling 효율화

**구조**:
```python
self.feature_net = nn.Sequential(
    nn.Linear(1024, 256),  # in_features=1024, out_features=256
    nn.ReLU(inplace=True)
)
```

**파라미터 계산**:
- `Linear(1024, 256)`:
  - weight: `[256, 1024]` = 262,144 params
  - bias: `[256]` = 256 params
  - **총**: 262,400 params

**차원 흐름**:
```
Input:  cls_tokens [B, T, 1024]
        ↓ Linear(1024→256)
Temp:   linear_out [B, T, 256]
        ↓ ReLU
Output: features [B, T, 256]
```

**예시** (B=2, T=5):
```
[2, 5, 1024] → Linear → [2, 5, 256] → ReLU → [2, 5, 256]
```

---

### 2.2 MambaBlock (Temporal Modeling)

**목적**: 시간적으로 일관된 feature 학습

**전체 구조**:
```python
class MambaBlock(nn.Module):
    def __init__(self, d_model=256, layer_idx=0, expand=2,
                 d_state=64, d_conv=4, headdim=64):
        super().__init__()

        # Pre-normalization
        self.norm1 = nn.LayerNorm(d_model)

        # Mamba2 core (SSM)
        self.mamba = Mamba2(
            d_model=d_model,      # 256
            d_state=d_state,      # 64
            d_conv=d_conv,        # 4
            expand=expand,        # 2
            layer_idx=layer_idx,  # 0
            headdim=headdim       # 64
        )

        # Post-normalization
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward Network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),     # 256 → 1024
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)      # 1024 → 256
        )
```

**총 파라미터**: 958,360 (TSP의 76.4%)

#### 2.2.1 norm1 (Pre-normalization)

**구조**: `LayerNorm(256)`

**파라미터**:
- weight: `[256]` = 256 params
- bias: `[256]` = 256 params
- **총**: 512 params

**차원 흐름**:
```
Input:  features [B, T, 256]
        ↓ LayerNorm
Output: normed1 [B, T, 256]
```

#### 2.2.2 mamba (Mamba2 Core)

**목적**: Selective State Space Model로 temporal dependency 학습

**구조** (from training log):
```python
self.mamba = Mamba2(
    d_model=256,
    d_state=64,
    d_conv=4,
    expand=2,
    headdim=64
)
```

**파라미터 분해** (from training log):
```
1. dt_bias:         [8]                 = 8 params
2. A_log:           [8]                 = 8 params
3. D:               [8]                 = 8 params
4. in_proj.weight:  [1160, 256]         = 296,960 params
5. conv1d.weight:   [640, 1, 4]         = 2,560 params
6. conv1d.bias:     [640]               = 640 params
7. norm.weight:     [512]               = 512 params
8. out_proj.weight: [256, 512]          = 131,072 params
-----------------------------------------------------------
Total:                                  = 431,768 params
```

**차원 흐름**:
```
Input:  normed1 [B, T, 256]
        ↓ in_proj: [256] → [1160]
Proj:   [B, T, 1160]
        ↓ Split into (x, z, B, C, dt)
SSM:    x [B, T, 512], z [B, T, 512], ...
        ↓ Selective SSM processing
        ↓ conv1d: [640, 1, 4]
Conv:   [B, T, 640]
        ↓ SSM state update
        ↓ out_proj: [512] → [256]
Output: mamba_out [B, T, 256]
```

**핵심 연산**:
- **Selective mechanism**: dt, B, C로 input-dependent state 업데이트
- **Conv1d**: Local temporal context (kernel_size=4)
- **State Space**: 긴 시퀀스 효율적 처리

#### 2.2.3 norm2 (Post-normalization)

**구조**: `LayerNorm(256)`

**파라미터**:
- weight: `[256]` = 256 params
- bias: `[256]` = 256 params
- **총**: 512 params

**차원 흐름**:
```
Input:  residual1 [B, T, 256]  # normed1 + mamba_out
        ↓ LayerNorm
Output: normed2 [B, T, 256]
```

#### 2.2.4 mlp (Feed-Forward Network)

**목적**: Channel-wise transformation

**구조**:
```python
self.mlp = nn.Sequential(
    nn.Linear(256, 1024),  # Expansion
    nn.GELU(),
    nn.Linear(1024, 256)   # Projection
)
```

**파라미터 분해** (from training log):
```
1. mlp.0.weight:  [1024, 256]  = 262,144 params
2. mlp.0.bias:    [1024]       = 1,024 params
3. mlp.2.weight:  [256, 1024]  = 262,144 params
4. mlp.2.bias:    [256]        = 256 params
-------------------------------------------------------
Total:                         = 525,568 params
```

**차원 흐름**:
```
Input:  normed2 [B, T, 256]
        ↓ Linear(256→1024)
Expand: [B, T, 1024]
        ↓ GELU
Activ:  [B, T, 1024]
        ↓ Linear(1024→256)
Output: mlp_out [B, T, 256]
```

#### 2.2.5 MambaBlock Forward Pass

**전체 흐름** (residual connections 포함):
```python
def forward(self, x):
    # x: [B, T, 256]

    # First block: Mamba with residual
    residual = x
    x = self.norm1(x)           # [B, T, 256]
    x = self.mamba(x)           # [B, T, 256]
    x = residual + x            # [B, T, 256] (residual connection)

    # Second block: MLP with residual
    residual = x
    x = self.norm2(x)           # [B, T, 256]
    x = self.mlp(x)             # [B, T, 256]
    x = residual + x            # [B, T, 256] (residual connection)

    return x  # [B, T, 256]
```

**MambaBlock 파라미터 총합**:
```
norm1:       512 params
mamba:       431,768 params
norm2:       512 params
mlp:         525,568 params
--------------------------------
Total:       958,360 params (76.4% of TSP)
```

---

### 2.3 Projection Layer

**목적**: Mamba output을 prediction heads 입력 차원으로 축소

**구조**:
```python
self.mamba_proj = nn.Linear(256, 128)
```

**파라미터**:
- weight: `[128, 256]` = 32,768 params
- bias: `[128]` = 128 params
- **총**: 32,896 params

**차원 흐름**:
```
Input:  mamba_out [B, T, 256]
        ↓ Linear(256→128)
Output: hidden [B, T, 128]
```

**예시** (B=2, T=5):
```
[2, 5, 256] → Linear → [2, 5, 128]
```

---

### 2.4 Prediction Heads

**목적**: Hidden states에서 scale과 shift 예측

#### 2.4.1 Scale Head

**구조**:
```python
self.scale_head = nn.Linear(128, 1)
```

**파라미터**:
- weight: `[1, 128]` = 128 params
- bias: `[1]` = 1 param
- **총**: 129 params

**차원 흐름**:
```
Input:  hidden [B, T, 128]
        ↓ Linear(128→1)
Logits: scale_logits [B, T, 1]
        ↓ squeeze(-1)
Temp:   scale_logits [B, T]
        ↓ Softplus (ensure positive)
Output: scale [B, T]
```

**Softplus 함수**:
```python
scale = F.softplus(scale_logits)
# softplus(x) = log(1 + exp(x))
# 항상 양수, 미분 가능
```

#### 2.4.2 Shift Head

**구조**:
```python
self.shift_head = nn.Linear(128, 1)
```

**파라미터**:
- weight: `[1, 128]` = 128 params
- bias: `[1]` = 1 param
- **총**: 129 params

**차원 흐름**:
```
Input:  hidden [B, T, 128]
        ↓ Linear(128→1)
Logits: shift_logits [B, T, 1]
        ↓ squeeze(-1)
Output: shift [B, T]  # 어떤 값도 가능
```

**Prediction Heads 총 파라미터**:
```
scale_head:  129 params
shift_head:  129 params
--------------------------
Total:       258 params (0.02% of TSP)
```

---

## 3. Forward Pass 전체 흐름

### 3.1 입력/출력 스펙

**입력**:
- `cls_tokens`: `[B, T, 1024]` - 2-layer averaged CLS tokens
  - B: Batch size
  - T: Temporal length (e.g., 5 for training, 50 for testing)
  - 1024: DINOv2-ViT-L embedding dimension

**출력**:
- `scale`: `[B, T]` - Positive scale factors
- `shift`: `[B, T]` - Shift values (any real number)

### 3.2 단계별 차원 변화

**전체 흐름표**:
```
Layer/Operation              Input Shape      Output Shape     Parameters
============================================================================
Input: cls_tokens            [B, T, 1024]     -                -

1. Feature Extractor
   ├─ Linear(1024→256)       [B, T, 1024]     [B, T, 256]      262,144
   └─ ReLU                   [B, T, 256]      [B, T, 256]      0
   Subtotal: 262,400 params

2. MambaBlock
   ├─ norm1                  [B, T, 256]      [B, T, 256]      512
   ├─ mamba                  [B, T, 256]      [B, T, 256]      431,768
   │  ├─ in_proj             [B, T, 256]      [B, T, 1160]     296,960
   │  ├─ conv1d              [B, T, 640]      [B, T, 640]      3,200
   │  ├─ ssm_norm            [B, T, 512]      [B, T, 512]      512
   │  └─ out_proj            [B, T, 512]      [B, T, 256]      131,072
   ├─ residual add           [B, T, 256]      [B, T, 256]      0
   ├─ norm2                  [B, T, 256]      [B, T, 256]      512
   ├─ mlp
   │  ├─ Linear(256→1024)    [B, T, 256]      [B, T, 1024]     263,168
   │  ├─ GELU                [B, T, 1024]     [B, T, 1024]     0
   │  └─ Linear(1024→256)    [B, T, 1024]     [B, T, 256]      262,400
   └─ residual add           [B, T, 256]      [B, T, 256]      0
   Subtotal: 958,360 params

3. Projection
   └─ Linear(256→128)        [B, T, 256]      [B, T, 128]      32,896

4. Prediction Heads
   ├─ scale_head
   │  ├─ Linear(128→1)       [B, T, 128]      [B, T, 1]        129
   │  ├─ squeeze(-1)         [B, T, 1]        [B, T]           0
   │  └─ Softplus            [B, T]           [B, T]           0
   └─ shift_head
      ├─ Linear(128→1)       [B, T, 128]      [B, T, 1]        129
      └─ squeeze(-1)         [B, T, 1]        [B, T]           0
   Subtotal: 258 params

============================================================================
Output: scale, shift         [B, T], [B, T]   -                -

TOTAL PARAMETERS: 1,253,914
```

### 3.3 구체적 예시 (B=2, T=5)

```python
# Input
cls_tokens: [2, 5, 1024]

# Step 1: Feature Extractor
features = feature_net(cls_tokens)  # [2, 5, 256]

# Step 2: MambaBlock
# 2.1: First residual block
residual = features                 # [2, 5, 256]
x = norm1(features)                 # [2, 5, 256]
x = mamba(x)                        # [2, 5, 256]
x = residual + x                    # [2, 5, 256]

# 2.2: Second residual block
residual = x                        # [2, 5, 256]
x = norm2(x)                        # [2, 5, 256]
x = mlp(x)                          # [2, 5, 256]
mamba_out = residual + x            # [2, 5, 256]

# Step 3: Projection
hidden = mamba_proj(mamba_out)      # [2, 5, 128]

# Step 4: Prediction Heads
scale_logits = scale_head(hidden)   # [2, 5, 1]
scale_logits = scale_logits.squeeze(-1)  # [2, 5]
scale = F.softplus(scale_logits)    # [2, 5], positive values

shift_logits = shift_head(hidden)   # [2, 5, 1]
shift = shift_logits.squeeze(-1)    # [2, 5], any values

# Output
return scale, shift  # [2, 5], [2, 5]
```

---

## 4. 파라미터 통계

### 4.1 컴포넌트별 분포

| 컴포넌트 | 파라미터 수 | 비율 | 누적 |
|---------|------------|------|------|
| **1. Feature Extractor** | 262,400 | 20.9% | 20.9% |
| **2. MambaBlock** | 958,360 | 76.4% | 97.3% |
| ├─ norm1 | 512 | 0.04% | - |
| ├─ mamba (Mamba2) | 431,768 | 34.4% | - |
| ├─ norm2 | 512 | 0.04% | - |
| └─ mlp | 525,568 | 41.9% | - |
| **3. Projection** | 32,896 | 2.6% | 99.9% |
| **4. Prediction Heads** | 258 | 0.02% | 100.0% |
| **Total** | **1,253,914** | **100.0%** | - |

### 4.2 핵심 인사이트

1. **MambaBlock이 압도적** (76.4%)
   - 그 중 MLP가 가장 큼 (41.9%)
   - Mamba2 core는 34.4%

2. **Feature Extractor도 상당함** (20.9%)
   - 1024→256 차원 축소가 비용 큼

3. **Prediction Heads는 경량** (0.02%)
   - 128→1 변환만 수행

4. **전체 FlashDepth 대비**: 0.39%
   - FlashDepth: 319.6M
   - TSP: 1.25M
   - Total: 320.9M

---

## 5. 메트릭 깊이 변환

TSP의 출력인 scale과 shift는 다음과 같이 사용됩니다:

### 5.1 학습 시 (Inverse Depth Space)

```python
# TSP prediction
scale, shift = temporal_scale_predictor(cls_tokens)  # [B, T]

# FlashDepth relative depth
relative_depth = flashdepth_model(images)  # [B*T, 1, H, W]

# Convert to metric inverse depth
pred_inverse = scale.view(B*T, 1, 1, 1) * relative_depth + shift.view(B*T, 1, 1, 1)
# pred_inverse: [B*T, 1, H, W] in 100/m scale

# Loss computation
gt_inverse = 100.0 / gt_depth  # Ground truth in 100/m scale
loss = log_l1_loss(pred_inverse, gt_inverse, valid_mask)
```

### 5.2 추론 시 (Metric Depth)

```python
# Predict in canonical space
pred_inverse_canonical = scale * relative_depth + shift  # [B*T, 1, H, W]

# De-canonicalize (inverse depth space)
fx_actual = intrinsics['fx']  # [B, T]
de_canonical_ratio = 500.0 / fx_actual  # CANONICAL_FX = 500
pred_inverse_actual = pred_inverse_canonical * de_canonical_ratio.view(B*T, 1, 1, 1)

# Convert to metric depth (meters)
pred_depth_metric = 100.0 / (pred_inverse_actual + 1e-8)  # [B*T, 1, H, W]
```

---

## 6. 학습 로그 검증

**파일**: `train_results/results_21/gear_5_mamba/large/training.log`

**로그 출력**:
```
2025-11-17 02:41:57,856 - INFO - TemporalScalePredictor: 1,253,914 parameters
2025-11-17 02:41:57,857 - INFO - Gear5MetricHead: 1,253,914 / 1,253,914 trainable parameters
2025-11-17 02:41:58,176 - INFO -   - Gear5MetricHead: 1,253,914
```

**레이어별 검증** (from log):
```
feature_net.0.weight:                          [256, 1024]    = 262,144 ✓
feature_net.0.bias:                            [256]          = 256 ✓
temporal_mamba.norm1.weight:                   [256]          = 256 ✓
temporal_mamba.norm1.bias:                     [256]          = 256 ✓
temporal_mamba.mamba.dt_bias:                  [8]            = 8 ✓
temporal_mamba.mamba.A_log:                    [8]            = 8 ✓
temporal_mamba.mamba.D:                        [8]            = 8 ✓
temporal_mamba.mamba.in_proj.weight:           [1160, 256]    = 296,960 ✓
temporal_mamba.mamba.conv1d.weight:            [640, 1, 4]    = 2,560 ✓
temporal_mamba.mamba.conv1d.bias:              [640]          = 640 ✓
temporal_mamba.mamba.norm.weight:              [512]          = 512 ✓
temporal_mamba.mamba.out_proj.weight:          [256, 512]     = 131,072 ✓
temporal_mamba.norm2.weight:                   [256]          = 256 ✓
temporal_mamba.norm2.bias:                     [256]          = 256 ✓
temporal_mamba.mlp.0.weight:                   [1024, 256]    = 262,144 ✓
temporal_mamba.mlp.0.bias:                     [1024]         = 1,024 ✓
temporal_mamba.mlp.2.weight:                   [256, 1024]    = 262,144 ✓
temporal_mamba.mlp.2.bias:                     [256]          = 256 ✓
mamba_proj.weight:                             [128, 256]     = 32,768 ✓
mamba_proj.bias:                               [128]          = 128 ✓
scale_head.weight:                             [1, 128]       = 128 ✓
scale_head.bias:                               [1]            = 1 ✓
shift_head.weight:                             [1, 128]       = 128 ✓
shift_head.bias:                               [1]            = 1 ✓
-----------------------------------------------------------------------
TOTAL:                                                        = 1,253,914 ✓
```

---

## 7. 코드 위치

### 7.1 정의
- **파일**: `flashdepth/gear5_modules.py`
- **클래스**: `TemporalScalePredictor` (line 127-250)
- **MambaBlock**: `flashdepth/mamba.py` (line 31-63)

### 7.2 사용
- **학습**: `train_gear5.py`
- **추론**: `test_gear5.py`

### 7.3 체크포인트
- **Phase 1**: `train_results/results_21/gear_5_mamba/large/best_model.pth`
- **Phase 2**: `train_results/results_21/gear_5_mamba/hybrid/best_model.pth`

---

## 8. 요약

**TemporalScalePredictor (TSP)**:
- **총 파라미터**: 1,253,914 (1.25M)
- **핵심**: Mamba2 기반 temporal modeling
- **입력**: 2-layer CLS tokens [B, T, 1024]
- **출력**: scale [B, T], shift [B, T]
- **학습**: Frozen FlashDepth + Trainable TSP
- **성능**: 10.9 FPS, TAE 47.6% 개선

**파라미터 분포**:
1. MambaBlock: 958K (76.4%)
2. Feature Extractor: 262K (20.9%)
3. Projection: 33K (2.6%)
4. Heads: 258 (0.02%)

**전체 모델**:
- FlashDepth: 319.6M
- TSP: 1.25M (**0.39%**)
- Total: 320.9M
