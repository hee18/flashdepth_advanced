# DPT (Dense Prediction Transformer) 구조

## 개요

DPT (Dense Prediction Transformer)는 Transformer 기반 백본(DINOv2)의 다층 특징을 dense prediction 작업(depth estimation)을 위해 융합하는 디코더입니다.

**핵심 아이디어:**
- ViT의 여러 레이어에서 추출한 특징을 **U-Net 스타일로 상향 융합**
- 저수준 특징 (디테일) + 고수준 특징 (의미) = 정확한 depth 예측

---

## 전체 아키텍처

### FlashDepth-L DPT Head 구조

```
DINOv2 Intermediate Layers
├─ Layer 4  [B, 1369, 1024]  ─┐
├─ Layer 11 [B, 1369, 1024]  ─┼─> Projects + Resize
├─ Layer 17 [B, 1369, 1024]  ─┤
└─ Layer 23 [B, 1369, 1024]  ─┘
          ↓
    ┌─────────────────────────────────────────┐
    │ 1. Projection (1×1 Conv)                │
    │    [B, 1024, 37, 37] → [B, C_out, 37, 37] │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ 2. Resize (ConvTranspose / Conv / Identity) │
    │    다양한 해상도로 조정                 │
    └─────────────────────────────────────────┘
          ↓
    Layer 1 [B, 256, 148, 148]  (4× upsampled)
    Layer 2 [B, 512, 74, 74]    (2× upsampled)
    Layer 3 [B, 1024, 37, 37]   (1× identity)
    Layer 4 [B, 1024, 18, 18]   (0.5× downsampled)
          ↓
    ┌─────────────────────────────────────────┐
    │ 3. layer_rn (Conv to 256 channels)      │
    │    모든 레이어를 256 채널로 통일        │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ 4. RefineNet (Bottom-up Fusion)         │
    │    Path 4 → 3 → 2 → 1                   │
    └─────────────────────────────────────────┘
          ↓
    Path 1 [B, 256, 148, 148]
          ↓
    ┌─────────────────────────────────────────┐
    │ 5. Output Head                          │
    │    Conv → Upsample → Conv → Softplus    │
    └─────────────────────────────────────────┘
          ↓
    Inverse Depth [B, 1, 518, 518]
```

---

## 세부 구성 요소

### 1. Projection Layers

```python
# original_dpt.py:49-57
self.projects = nn.ModuleList([
    nn.Conv2d(
        in_channels=1024,      # DINOv2 embed_dim
        out_channels=out_channel,  # [256, 512, 1024, 1024]
        kernel_size=1,
        stride=1,
        padding=0,
    ) for out_channel in out_channels
])
```

**목적:**
- DINOv2 특징 차원 (1024) → DPT 채널 (256/512/1024/1024)로 변환
- 1×1 Conv = 채널 간 선형 변환

**입출력:**
```
Layer 1: [B, 1024, 37, 37] → [B, 256, 37, 37]
Layer 2: [B, 1024, 37, 37] → [B, 512, 37, 37]
Layer 3: [B, 1024, 37, 37] → [B, 1024, 37, 37]
Layer 4: [B, 1024, 37, 37] → [B, 1024, 37, 37]
```

---

### 2. Resize Layers

```python
# original_dpt.py:59-79
self.resize_layers = nn.ModuleList([
    # Layer 1: 4× upsample (ConvTranspose2d)
    nn.ConvTranspose2d(
        in_channels=256, out_channels=256,
        kernel_size=4, stride=4, padding=0
    ),

    # Layer 2: 2× upsample (ConvTranspose2d)
    nn.ConvTranspose2d(
        in_channels=512, out_channels=512,
        kernel_size=2, stride=2, padding=0
    ),

    # Layer 3: Keep same (Identity)
    nn.Identity(),

    # Layer 4: 2× downsample (Conv2d)
    nn.Conv2d(
        in_channels=1024, out_channels=1024,
        kernel_size=3, stride=2, padding=1
    )
])
```

**목적:**
- 다양한 해상도의 특징 맵 생성
- 멀티스케일 정보 활용

**해상도 변화 (37×37 기준):**
```
Layer 1: 37×37 → 148×148  (4× upsampled)   - 세밀한 디테일용
Layer 2: 37×37 → 74×74    (2× upsampled)   - 중간 스케일
Layer 3: 37×37 → 37×37    (유지)          - 원본 스케일
Layer 4: 37×37 → 18×18    (0.5× downsampled) - 고수준 의미
```

**왜 이런 구조?**
- **Layer 1 (4×):** 저수준 특징 (DINOv2 Layer 4)은 디테일이 많아 고해상도 유지
- **Layer 4 (0.5×):** 고수준 특징 (DINOv2 Layer 23)은 의미적 정보라 저해상도로도 충분

---

### 3. Scratch Layers (layer_rn)

```python
# original_dpt.py:81-86
self.scratch = _make_scratch(
    out_channels=[256, 512, 1024, 1024],
    dpt_dim=256,  # 모든 레이어를 256 채널로 통일
    groups=1,
    expand=False,
)
```

#### layer_rn의 역할

```python
# util/blocks.py (simplified)
layer_1_rn = nn.Sequential(
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.Conv2d(256, 256, kernel_size=3, padding=1)
)
layer_2_rn = nn.Sequential(
    nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 512 → 256
    nn.Conv2d(256, 256, kernel_size=3, padding=1)
)
# ... 동일한 구조 for layer 3, 4
```

**목적:**
- 모든 레이어를 **동일한 채널 수 (256)**로 통일
- 융합(fusion) 준비

**입출력:**
```
Layer 1: [B, 256, 148, 148]  → [B, 256, 148, 148]  (유지)
Layer 2: [B, 512, 74, 74]    → [B, 256, 74, 74]    (512→256)
Layer 3: [B, 1024, 37, 37]   → [B, 256, 37, 37]    (1024→256)
Layer 4: [B, 1024, 18, 18]   → [B, 256, 18, 18]    (1024→256)
```

---

### 4. RefineNet (Feature Fusion Blocks)

DPT의 핵심! U-Net 스타일 bottom-up 융합

```python
# original_dpt.py:90-93
self.scratch.refinenet4 = _make_fusion_block(256, use_bn=False)
self.scratch.refinenet3 = _make_fusion_block(256, use_bn=False)
self.scratch.refinenet2 = _make_fusion_block(256, use_bn=False)
self.scratch.refinenet1 = _make_fusion_block(256, use_bn=False)
```

#### RefineNet 구조

```python
# util/blocks.py (simplified)
class FeatureFusionBlock(nn.Module):
    def forward(self, x, skip=None):
        # 1. Process first input
        out = self.resConfUnit1(x)  # Residual convolution

        if skip is not None:
            # 2. Process skip connection
            out_skip = self.resConfUnit2(skip)  # Residual convolution

            # 3. Add (element-wise)
            out = out + out_skip

        # 4. Upsample to next layer size
        out = F.interpolate(out, size=size, mode='bilinear')

        return out
```

#### 융합 순서 (Bottom-up)

```
Path 4 (18×18)  [고수준, 작은 크기]
  ↓ refinenet4 (no skip, upsample to 37×37)
Path 3 (37×37)  [중간 수준]
  ↓ refinenet3 (+ layer_3_rn, upsample to 74×74)
Path 2 (74×74)  [중간 수준]
  ↓ refinenet2 (+ layer_2_rn, upsample to 148×148)
Path 1 (148×148) [저수준, 큰 크기, 세밀한 디테일]
  ↓ refinenet1 (+ layer_1_rn, 유지)
Final Output (148×148)
```

#### 코드 흐름

```python
# original_dpt.py:140-143
path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
# path_4: [B, 256, 37, 37]

path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
# path_3: [B, 256, 74, 74]

path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
# path_2: [B, 256, 148, 148]

path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
# path_1: [B, 256, 148, 148]
```

**왜 Bottom-up?**
- 고수준 의미 정보 (path_4)를 먼저 확장
- 점진적으로 저수준 디테일 (layer_1~3) 추가
- 최종적으로 고해상도 + 풍부한 의미 정보

---

### 5. Output Head

```python
# original_dpt.py:98-104
self.scratch.output_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

self.scratch.output_conv2 = nn.Sequential(
    nn.Conv2d(128, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 1, kernel_size=1),
    nn.Softplus(),  # Ensure positive output (inverse depth)
)
```

#### Forward Pass

```python
# flashdepth/model.py:389-412 (final_head)
out = self.depth_head.scratch.output_conv1(path_1)
# [B, 256, 148, 148] → [B, 128, 148, 148]

out = F.interpolate(out, (518, 518), mode="bilinear")
# [B, 128, 518, 518] (원본 이미지 크기로)

out = self.depth_head.scratch.output_conv2(out)
# [B, 128, 518, 518] → [B, 1, 518, 518]

depth = F.relu(out).squeeze(1)
# [B, 518, 518] (Inverse depth)
```

**Softplus Activation:**
```python
Softplus(x) = log(1 + exp(x))
```
- 항상 양수 (inverse depth는 0 이상이어야 함)
- ReLU보다 부드러운 gradient (학습 안정성)

---

## 각 레이어의 정보가 다른 이유

### DINOv2 레이어별 특성 복습

```
Layer 4:  저수준 특징  (edges, textures, 세밀한 디테일)
Layer 11: 중간 특징    (object parts, local geometry)
Layer 17: 고수준 특징  (object categories, scene structure)
Layer 23: 최고 특징    (semantic understanding, global context)
```

### DPT에서의 활용

| DPT Layer | DINOv2 Source | 특징 | 해상도 | 역할 |
|-----------|---------------|------|--------|------|
| **Layer 1** | Layer 4 | 저수준 | 148×148 (4×) | **세밀한 경계선, 텍스처** |
| **Layer 2** | Layer 11 | 중간-1 | 74×74 (2×) | **물체 부분, 표면 구조** |
| **Layer 3** | Layer 17 | 중간-2 | 37×37 (1×) | **물체 형태, 의미적 구조** |
| **Layer 4** | Layer 23 | 고수준 | 18×18 (0.5×) | **장면 이해, 전역 컨텍스트** |

### 융합 과정에서의 시너지

```
1. Path 4 (18×18) → 고수준 장면 이해
   "이것은 도로 장면이고, 저기에 차가 있다"
   ↓ Upsample + Fusion

2. Path 3 (37×37) → + Layer 17 (물체 형태)
   "차의 형태와 위치를 정확히 파악"
   ↓ Upsample + Fusion

3. Path 2 (74×74) → + Layer 11 (표면 구조)
   "차 표면의 곡면 구조 이해"
   ↓ Upsample + Fusion

4. Path 1 (148×148) → + Layer 4 (세밀한 디테일)
   "차 경계선, 윈도우 프레임 등 정확한 디테일"
```

**결과:**
- 전역 이해 (고수준) + 지역 디테일 (저수준)
- 의미적으로 일관되고 (semantic consistency)
- 공간적으로 정확한 (spatial accuracy) depth 예측

---

## Mamba 통합 (FlashDepth)

### Mamba 삽입 위치

```python
# original_dpt.py:162-198 (forward_with_mamba)
path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
if 0 in temporal_layer:
    path_4 = mamba_fn(path_4)  # Temporal processing

path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
if 1 in temporal_layer:
    path_3 = mamba_fn(path_3)  # Temporal processing

path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
if 2 in temporal_layer:
    path_2 = mamba_fn(path_2)  # Temporal processing

path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
if 3 in temporal_layer:
    path_1 = mamba_fn(path_1)  # Temporal processing
```

### FlashDepth-L 설정

```python
# configs/flashdepth-l/config.yaml
mamba_in_dpt_layer: [1]  # Only at DPT Layer 1 (path_3)
```

**왜 path_3 (Layer 1)?**
- 중간 수준 특징 (의미 + 구조)
- 적당한 해상도 (37×37)
- 시간적 일관성이 가장 중요한 레벨

**Mamba의 역할:**
- 비디오 프레임 간 시간적 일관성 유지
- 이전 프레임 정보를 현재 프레임에 전달
- Depth flickering 감소

---

## Gear3 Modulation

### Modulation 위치

```python
# train_gear3.py:1085-1092
path_1_modulated, importance_map, fg_features, bg_features = model.gear3_head(
    patch_tokens,      # Layer 23 patch tokens
    attention_weights, # Layer 23 attention
    dpt_features,      # [path_4, path_3, path_2, path_1]
    patch_h, patch_w
)

# Only path_1 is modulated!
out = model.depth_head.scratch.output_conv1(path_1_modulated)
```

### 왜 path_1만 modulate?

```
Path 1 Source: DINOv2 Layer 4 (저수준)
Path 1 Content: 세밀한 경계선, 텍스처, 디테일
    ↓
Modulation Source: DINOv2 Layer 23 (최고수준)
Modulation Content: 장면 이해, 물체 중요도, FG/BG 분리
    ↓
Modulation Formula:
    modulated_feature[x,y] = gamma[x,y] ⊙ path_1[x,y] + beta[x,y]

    where:
    gamma[x,y] = importance[x,y] * fg_gamma + (1-importance[x,y]) * bg_gamma
    beta[x,y]  = importance[x,y] * fg_beta  + (1-importance[x,y]) * bg_beta
```

**Semantic Mismatch 문제:**
- Path 2, 3, 4는 Layer 11, 17, 23에서 나옴 (중간~고수준)
- Layer 23의 CLS attention은 고수준 의미 정보
- 중간 레이어를 고수준 정보로 modulate하면 **semantic mismatch**
- **Path 1만 modulate**: 저수준 디테일에 고수준 의미 주입

---

## 요약

### DPT의 3가지 핵심 아이디어

1. **멀티스케일 특징 추출**
   - DINOv2의 4개 레이어 (4, 11, 17, 23) 사용
   - 저수준→고수준 다양한 추상화 수준

2. **Bottom-up 융합**
   - 고수준부터 시작 (path_4)
   - 점진적으로 저수준 디테일 추가 (path_3→2→1)
   - RefineNet으로 skip connection 융합

3. **해상도 조절**
   - 저수준은 고해상도 (4× upsampled)
   - 고수준은 저해상도 (0.5× downsampled)
   - 효율성 + 정확성

### 각 컴포넌트의 역할

| 컴포넌트 | 입력 | 출력 | 역할 |
|---------|------|------|------|
| **Projects** | [B,1024,37,37] | [B,C,37,37] | 채널 변환 |
| **Resize** | [B,C,37,37] | [B,C,H,W] | 해상도 조절 |
| **layer_rn** | [B,C,H,W] | [B,256,H,W] | 채널 통일 |
| **RefineNet** | 여러 레이어 | [B,256,148,148] | 멀티스케일 융합 |
| **Output** | [B,256,148,148] | [B,1,518,518] | Depth 예측 |

### FlashDepth 확장

- **Mamba**: 시간적 일관성 (비디오)
- **Gear3**: 공간적 modulation (FG/BG 분리)

---

## 참고 자료

- [DPT Paper (Vision Transformers for Dense Prediction)](https://arxiv.org/abs/2103.13413)
- [FlashDepth 구현](../flashdepth/original_dpt.py)
- [Gear3 Modulation](../flashdepth/gear3_modules.py)
