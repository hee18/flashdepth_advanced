# DINOv2 (Vision Transformer) 구조

## 개요

DINOv2는 Meta에서 개발한 자기지도학습(self-supervised learning) Vision Transformer 모델입니다. FlashDepth에서는 **동결된(frozen) 백본**으로 사용되어 이미지에서 다층(multi-scale) 특징을 추출합니다.

---

## 모델 변형

FlashDepth에서 사용하는 2가지 모델:

| 모델 | Embed Dim | Depth (Layers) | Heads | Parameters | Patch Size |
|------|-----------|----------------|-------|------------|------------|
| **ViT-S** (Small) | 384 | 12 | 6 | ~22M | 14×14 |
| **ViT-L** (Large) | 1024 | 24 | 16 | ~300M | 14×14 |

---

## 핵심 아키텍처

### 1. Patch Embedding

```
입력 이미지 [B, 3, 518, 518]
    ↓ (Patch Embed: Conv2d 14×14, stride=14)
Patch Tokens [B, 37×37=1369, embed_dim]
```

- **518 ÷ 14 = 37 patches per dimension**
- 각 patch는 `embed_dim` 차원의 벡터로 변환

---

### 2. Token 구성

```python
# dinov2.py:234-251
x = self.patch_embed(x)  # [B, 1369, embed_dim]
x = torch.cat((self.cls_token, x), dim=1)  # [B, 1370, embed_dim]
x = x + self.pos_embed  # Positional encoding 추가
```

#### Token 종류:
1. **[CLS] Token** (위치 0)
   - 전역 정보를 집약하는 특수 토큰
   - 분류 작업이나 전역 표현에 사용
   - **Gear3에서는 CLS 토큰으로 importance map 생성**

2. **Patch Tokens** (위치 1~1369)
   - 이미지의 각 14×14 패치를 나타냄
   - Spatial 정보 유지

3. **Register Tokens** (선택적, FlashDepth에서는 미사용)
   - DINOv2의 특수 기능 (num_register_tokens=0)
   - 모델의 표현력 향상용

---

### 3. Transformer Blocks

#### ViT-L 기준 (24 layers):

```
Block 0  (Layer 0)   ← 저수준 특징 (edges, textures)
Block 1  (Layer 1)
Block 2  (Layer 2)
Block 3  (Layer 3)
Block 4  (Layer 4)   ← DPT에서 사용 (intermediate_layer_idx[0])
    ⋮
Block 10 (Layer 10)
Block 11 (Layer 11)  ← DPT에서 사용 (intermediate_layer_idx[1])
    ⋮
Block 16 (Layer 16)
Block 17 (Layer 17)  ← DPT에서 사용 (intermediate_layer_idx[2])
    ⋮
Block 22 (Layer 22)
Block 23 (Layer 23)  ← DPT에서 사용 (intermediate_layer_idx[3]) + 고수준 특징
```

#### 각 Block의 구성:

```python
# dinov2_layers/block.py (simplified)
class Block(nn.Module):
    def forward(self, x):
        # Multi-Head Self-Attention (MHSA)
        x = x + self.attn(self.norm1(x))  # Residual connection

        # Feed-Forward Network (MLP)
        x = x + self.mlp(self.norm2(x))   # Residual connection

        return x
```

**구성 요소:**
- **LayerNorm** → **Multi-Head Attention** → **Residual Add**
- **LayerNorm** → **MLP (4× expansion)** → **Residual Add**

---

## 왜 각 레이어의 정보가 다른가?

Vision Transformer는 깊이에 따라 **계층적 표현(hierarchical representations)**을 학습합니다:

### 📊 레이어별 특징 추상화 정도

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 0-4:   저수준 특징 (Low-level Features)             │
│  - Edges, corners, colors                                   │
│  - 텍스처 패턴                                              │
│  - 작은 구조 (small structures)                             │
│  - 높은 공간 해상도 정보                                    │
├─────────────────────────────────────────────────────────────┤
│  Layer 5-11:  중간 수준 특징 (Mid-level Features)          │
│  - 물체의 부분 (object parts)                               │
│  - 표면 기하학 (surface geometry)                          │
│  - 지역적 패턴 (local patterns)                            │
│  - Depth cues (깊이 단서)                                   │
├─────────────────────────────────────────────────────────────┤
│  Layer 12-17: 고수준 특징 (High-level Features)            │
│  - 물체 카테고리 (object categories)                        │
│  - 장면 구조 (scene structure)                             │
│  - 의미적 관계 (semantic relationships)                     │
│  - 전역 컨텍스트 (global context)                          │
├─────────────────────────────────────────────────────────────┤
│  Layer 18-23: 최고수준 특징 (Very High-level Features)     │
│  - 추상적 개념 (abstract concepts)                          │
│  - 장면 이해 (scene understanding)                          │
│  - 물체 간 관계 (inter-object relations)                    │
│  - 의미론적 분할 정보 (semantic segmentation info)          │
│  - **Gear3에서 CLS 토큰 사용 (importance map)**            │
└─────────────────────────────────────────────────────────────┘
```

### 🔬 왜 이런 계층 구조가 생기나?

1. **Self-Attention의 receptive field 확장**
   - 초기 레이어: 가까운 패치들만 참조 (local attention)
   - 깊은 레이어: 먼 패치들까지 참조 (global attention)

2. **Residual Connection의 누적 효과**
   - 각 레이어가 이전 정보를 보존하면서 새로운 정보 추가
   - 깊어질수록 더 추상적인 표현 형성

3. **자기지도학습(self-supervised learning)의 영향**
   - DINOv2는 이미지 재구성, contrastive learning 등으로 학습
   - 자연스럽게 저수준→고수준 특징 계층 형성

---

## FlashDepth에서 사용하는 중간 레이어

### ViT-L Intermediate Layers

```python
# flashdepth/model.py:49-51
self.intermediate_layer_idx = {
    'vitl': [4, 11, 17, 23],  # Layer 4, 11, 17, 23
    'vits': [2, 5, 8, 11],     # Layer 2, 5, 8, 11
}
```

### 왜 이 레이어들을 선택했나?

```
Layer 4  (Early)   → 저수준 특징 → DPT Layer 1 (고해상도 복원용)
Layer 11 (Mid-1)   → 중간 특징   → DPT Layer 2 (구조 정보)
Layer 17 (Mid-2)   → 고수준 특징 → DPT Layer 3 (의미 정보)
Layer 23 (Late)    → 최고수준    → DPT Layer 4 (장면 이해)
```

**멀티스케일 융합의 장점:**
- 저수준: 세밀한 디테일 (경계선, 텍스처)
- 중간수준: 물체 부분, 표면
- 고수준: 전체 장면 구조, 의미

→ **DPT에서 이들을 융합하여 정확한 depth 예측**

---

## CLS 토큰의 역할

### CLS 토큰은 모든 레이어에 존재

```python
# dinov2.py:114
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

# dinov2.py:240
x = torch.cat((self.cls_token, x), dim=1)  # Layer 0에 추가
```

**중요:** CLS 토큰은 **모든 레이어를 거치며 업데이트**됩니다!

```
Input: [CLS][patch1][patch2]...[patch1369]
  ↓ Block 0
[CLS₀][patch1₀][patch2₀]...[patch1369₀]
  ↓ Block 1
[CLS₁][patch1₁][patch2₁]...[patch1369₁]
  ⋮
  ↓ Block 23
[CLS₂₃][patch1₂₃][patch2₂₃]...[patch1369₂₃]
```

### 레이어별 CLS 토큰의 의미

| Layer | CLS 토큰의 정보 내용 |
|-------|---------------------|
| 0-4   | 저수준 전역 통계 (평균 색상, 밝기 등) |
| 5-11  | 중간 수준 장면 정보 (물체 분포, 배치) |
| 12-17 | 고수준 장면 구조 (물체 종류, 관계) |
| 18-23 | **최고 수준 추상화** (장면 카테고리, 중요 영역) |

### Gear3에서 CLS 토큰 사용

```python
# gear3_modules.py:42-50
# Extract CLS→patch attention from LAST LAYER (Layer 23)
cls_to_patches = attention_weights[:, :, 0, 1:]  # [B, num_heads, num_patches]
attn_scores = cls_to_patches.mean(dim=1)  # Average over heads
```

**왜 마지막 레이어의 CLS→patch attention을 사용?**

1. **최고 수준 의미 정보**
   - Layer 23의 CLS는 전체 장면을 가장 잘 이해
   - 어떤 영역이 중요한지(foreground) 판단 가능

2. **Attention Weights의 의미**
   - CLS→patch attention = "각 패치가 전역 표현에 기여하는 정도"
   - 높은 attention = 중요한 물체/영역
   - 낮은 attention = 배경/덜 중요한 영역

3. **Importance Map 생성**
   ```
   Layer 23 CLS Attention → Importance Map (0~1)
                          → FG/BG 분리
                          → Spatial Modulation
   ```

---

## Attention 메커니즘

### Multi-Head Self-Attention

```python
# dinov2_layers/attention.py (simplified)
class Attention(nn.Module):
    def forward(self, x):
        B, N, C = x.shape  # [B, 1370, 1024] for ViT-L

        # Q, K, V projection
        qkv = self.qkv(x).reshape(B, N, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(2)  # Each: [B, N, num_heads, head_dim]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / sqrt(head_dim)  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)

        # Store for Gear3 (only last layer)
        if self.store_attn_weights:
            self.attn_weights = attn

        # Weighted sum
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x
```

### Attention Weights 저장 (Gear3용)

```python
# flashdepth/model.py:594-599
for i, block in enumerate(model.pretrained.blocks):
    if i == len(model.pretrained.blocks) - 1:  # Last block only
        block.attn.store_attn_weights = True
    else:
        block.attn.store_attn_weights = False
```

**메모리 최적화:**
- 24개 레이어 모두 저장하면 ~11GB 추가 메모리
- **마지막 레이어만 저장**으로 메모리 절약

---

## Forward Pass 흐름

### 1. 전체 Forward (forward_features)

```python
# dinov2.py:257-271
def forward_features(self, x, masks=None):
    x = self.prepare_tokens_with_masks(x, masks)  # Patch embed + add CLS

    for blk in self.blocks:
        x = blk(x)  # 24 blocks for ViT-L

    x_norm = self.norm(x)
    return {
        "x_norm_clstoken": x_norm[:, 0],           # CLS token
        "x_norm_patchtokens": x_norm[:, 1:],       # Patch tokens
        "x_prenorm": x,                            # Before final norm
    }
```

### 2. 중간 레이어 추출 (get_intermediate_layers)

```python
# dinov2.py:363-397
def get_intermediate_layers(self, x, n=[4, 11, 17, 23]):
    x = self.prepare_tokens_with_masks(x)
    outputs = []

    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if i in n:  # [4, 11, 17, 23]
            outputs.append(x)

    # Apply final normalization
    outputs = [self.norm(out) for out in outputs]

    # Remove CLS token for DPT
    outputs = [out[:, 1:] for out in outputs]  # Only patch tokens

    return outputs  # 4 tensors: [B, 1369, 1024]
```

**FlashDepth 사용 예:**

```python
# flashdepth/model.py:166
intermediate_features = self.pretrained.get_intermediate_layers(
    x, self.intermediate_layer_idx[self.encoder]
)
# Returns: [layer4, layer11, layer17, layer23]
# Each: [B, 1369, 1024] for ViT-L
```

---

## 주요 특징

### 1. Pre-trained & Frozen

```python
# FlashDepth에서 DINOv2는 항상 frozen
for param in model.pretrained.parameters():
    param.requires_grad = False
```

**장점:**
- 강력한 일반화 능력 (ImageNet-22k 학습)
- 빠른 학습 (백본 업데이트 불필요)
- 메모리 효율적

### 2. Positional Encoding

```python
# dinov2.py:201-232
def interpolate_pos_encoding(self, x, w, h):
    # 다양한 해상도 지원 (518×518, 2K 등)
    # Positional encoding을 입력 크기에 맞춰 보간
```

**유연성:**
- 학습 시: 518×518
- 테스트 시: 임의 해상도 (2K 등)
- Positional encoding을 bicubic interpolation으로 조정

### 3. Memory-Efficient Attention

```python
# dinov2.py:149
attn_class=MemEffAttention  # Memory-efficient attention (xformers)
```

- Flash Attention 사용
- 메모리 사용량 O(N²) → O(N)

---

## 출력 형태

### get_intermediate_layers 출력

```python
outputs = model.pretrained.get_intermediate_layers(
    x,  # [B, 3, 518, 518]
    [4, 11, 17, 23]
)

# Returns: List of 4 tensors
# outputs[0]: [B, 1369, 1024]  # Layer 4  (37×37 patches)
# outputs[1]: [B, 1369, 1024]  # Layer 11
# outputs[2]: [B, 1369, 1024]  # Layer 17
# outputs[3]: [B, 1369, 1024]  # Layer 23
```

**주의:** CLS 토큰은 제거됨 (patch tokens만 반환)

### forward_features 출력

```python
output_dict = model.pretrained.forward_features(x)

# output_dict = {
#     'x_norm_clstoken': [B, 1024],      # CLS token (for Gear3)
#     'x_norm_patchtokens': [B, 1369, 1024],  # Patch tokens
#     'x_prenorm': [B, 1370, 1024],      # Before norm (CLS + patches)
# }
```

**Gear3 사용:**
```python
# flashdepth/model.py:213-218
features = self.pretrained.forward_features(x)
cls_token = features['x_norm_clstoken']  # [B, 1024] for ViT-L
```

---

## 요약

1. **DINOv2는 24-layer Vision Transformer** (ViT-L 기준)
   - 각 레이어는 저수준→고수준으로 추상화
   - Self-attention으로 global receptive field 확보

2. **중간 레이어 추출 (4, 11, 17, 23)**
   - 다양한 추상화 수준의 특징 제공
   - DPT에서 멀티스케일 융합에 사용

3. **CLS 토큰은 모든 레이어에 존재**
   - 각 레이어를 거치며 업데이트
   - 마지막 레이어 (23)의 CLS가 가장 추상적
   - **Gear3는 Layer 23 CLS→patch attention 사용**

4. **Attention Weights 저장**
   - 메모리 절약: 마지막 레이어만 저장
   - Gear3 importance map 생성에 필수

5. **Frozen Backbone**
   - 학습 시 업데이트 안 됨
   - 강력한 일반화 능력 보존

---

## 참고 자료

- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [ViT Paper](https://arxiv.org/abs/2010.11929)
- [FlashDepth 구현](../flashdepth/dinov2.py)
