# Option 5: Multi-scale Features for FG/BG Separation

## 핵심 아이디어

**Early layer (low-level) → BG network**
**Late layer (high-level) → FG network**

서로 다른 semantic level의 features를 사용하여 FG와 BG를 자연스럽게 분리.

---

## 이론적 근거 (2024-2025 최신 연구)

### Vision Transformer Layer별 특성

| Layer | Semantic Level | Features | Attention Pattern | 용도 |
|-------|---------------|----------|-------------------|------|
| **Early (1-8)** | Low-level | Colors, textures, edges, lighting | Uniform, local patches | **Background context** |
| **Middle (9-16)** | Mid-level | Object parts, patterns | Focused on semantic regions | - |
| **Late (17-24)** | High-level | Objects, classes, global semantics | Specific patches, long-range | **Foreground objects** |

### 검증된 사실

1. **Early layers**: "Encode basic features such as **colors and textures**"
2. **Late layers**: "Represent more specific **classes, including objects**"
3. **ViT 특성**: "Unlike CNNs, ViT obtains the **global representation from the shallow layers**"
   - CNN: Local → Global
   - ViT: 처음부터 global하지만, semantic은 나중에 발달

### Depth Estimation 관점

- **BG (Early features)**: 전체 scene의 scale/depth range
  - 실내 vs 야외, 가까운 scene vs 먼 scene
  - Lighting, weather, environment type

- **FG (Late features)**: 개별 객체의 정확한 depth
  - 차는 5m, 사람은 3m, 건물은 20m

---

## 구현 방법

### 1. ViT Layer 정보

**ViT-L** (현재 사용):
```python
intermediate_layer_idx = [4, 11, 17, 23]  # 총 24 layers 중
embed_dim = 1024  # 모든 레이어 동일!
```

**ViT-S**:
```python
intermediate_layer_idx = [2, 5, 8, 11]  # 총 12 layers 중
embed_dim = 384  # 모든 레이어 동일!
```

**중요**: ViT는 모든 레이어가 **같은 dimension**을 가짐!
- Output shape: `[B, num_patches+1, embed_dim]` (모든 레이어 동일)
- CNN처럼 resolution이 줄어들지 않음

### 2. 코드 수정

#### `train_gear3.py` - Forward Pass

```python
# 현재 (line ~765)
with torch.no_grad():
    encoder_features = model.pretrained.get_intermediate_layers(
        img_t, model.intermediate_layer_idx[model.encoder]
    )
    # encoder_features = [layer4, layer11, layer17, layer23] for ViT-L

    # 마지막 layer의 patch tokens만 사용
    patch_tokens = encoder_features[-1]  # [B, num_patches+1, 1024]
    patch_tokens = patch_tokens[:, 1:, :]  # Remove CLS token

# 변경 후
with torch.no_grad():
    encoder_features = model.pretrained.get_intermediate_layers(
        img_t, model.intermediate_layer_idx[model.encoder]
    )

    # Early layer for BG (low-level features)
    early_features = encoder_features[0][:, 1:, :]  # [B, num_patches, 1024]

    # Late layer for FG (high-level features)
    late_features = encoder_features[-1][:, 1:, :]  # [B, num_patches, 1024]
```

#### `flashdepth/gear3_modules.py` - ForegroundBackgroundNetworks

```python
class ForegroundBackgroundNetworks(nn.Module):
    """
    Option 5: Multi-scale Features
    - FG: Late layer features (semantic, object-centric)
    - BG: Early layer features (low-level, scene context)
    """
    def __init__(self, embed_dim=1024, feature_dim=256):
        super().__init__()

        # Foreground network (high-level semantics)
        self.fg_net = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )

        # Background network (low-level context)
        self.bg_net = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )

        logging.info(f"FG/BG Networks (Multi-scale): {embed_dim} -> {feature_dim}")

    def forward(self, early_patch_tokens, late_patch_tokens):
        """
        Args:
            early_patch_tokens: [B, num_patches, embed_dim] - Low-level features
            late_patch_tokens: [B, num_patches, embed_dim] - High-level features

        Returns:
            fg_features: [B, feature_dim] - From late layer (semantic)
            bg_features: [B, feature_dim] - From early layer (context)
        """
        # Global average pooling
        early_global = early_patch_tokens.mean(dim=1)  # [B, embed_dim]
        late_global = late_patch_tokens.mean(dim=1)    # [B, embed_dim]

        # Different semantic levels to different networks
        fg_features = self.fg_net(late_global)   # Semantic features → FG
        bg_features = self.bg_net(early_global)  # Low-level features → BG

        return fg_features, bg_features
```

#### `flashdepth/gear3_modules.py` - Gear3MetricHead

```python
def forward(self, early_patch_tokens, late_patch_tokens, attention_weights,
            dpt_features, patch_h, patch_w):
    """
    Args:
        early_patch_tokens: [B, num_patches, embed_dim] - For BG
        late_patch_tokens: [B, num_patches, embed_dim] - For FG
        attention_weights: [B, num_heads, num_patches+1, num_patches+1]
        dpt_features: List of [B, dpt_dim, H, W] for 4 DPT layers
        patch_h, patch_w: Spatial dimensions

    Returns:
        modulated_dpt_features: List of modulated DPT features
        importance_map: [B, 1, patch_h, patch_w] for visualization
    """
    # 1. Predict importance map (from late layer attention)
    importance_map = self.importance_predictor(attention_weights, patch_h, patch_w)

    # 2. Generate FG/BG features from different layers
    fg_features, bg_features = self.fg_bg_networks(
        early_patch_tokens, late_patch_tokens
    )

    # ... rest same as before
```

---

## 장점

✅ **검증된 hierarchical learning** 활용
✅ 자연스러운 semantic level 분리 (low-level vs high-level)
✅ No chicken-and-egg problem (서로 다른 입력)
✅ No hyperparameter tuning 필요
✅ Depth estimation에 직관적으로 맞음
✅ **레이어 크기 동일** → 구현 간단 (resize 불필요)

## 단점

⚠️ Early layer features가 depth에 얼마나 유용한지 실험 필요
⚠️ Attention weights는 여전히 late layer만 사용 (importance map)

---

## 실험 계획

1. **Baseline**: Option 3 (Attention-based Pooling) 먼저 시도
2. **If Option 3 fails**: Option 5 시도
3. **Comparison**: Importance map의 spatial variance 비교
   - Option 3: Attention-driven separation
   - Option 5: Multi-scale semantic separation

---

## 예상 결과

**Importance map 발전**:
- Early steps: Uniform (0.5) → 두 network가 다른 입력을 받으므로 FG ≠ BG
- Later steps: Spatial variance 증가 → Gradient flow 활성화
- Convergence: Meaningful FG/BG separation

**Depth prediction**:
- BG features: Scene-level scale/shift (전체 depth range)
- FG features: Object-level refinement (개별 객체 depth)

---

## ViT-L Layer 선택 가이드

현재 사용 가능한 layers: `[4, 11, 17, 23]`

| Option | Early (BG) | Late (FG) | 특징 |
|--------|-----------|-----------|------|
| **A** | Layer 4 | Layer 23 | 최대 차이 (가장 극단적) |
| **B** | Layer 4 | Layer 17 | 중간 차이 |
| **C** | Layer 11 | Layer 23 | 약간 차이 |

**추천**: Option A (Layer 4 → BG, Layer 23 → FG)
- 가장 큰 semantic gap
- Low-level vs High-level 명확히 구분

---

## 참고: CNN과의 차이

**CNN** (e.g., ResNet):
```
Layer 1: [B, 64,  H/4,  W/4]  → Low-level, local
Layer 2: [B, 128, H/8,  W/8]  → Mid-level
Layer 3: [B, 256, H/16, W/16] → High-level, global
         ↑ Resolution 감소
```

**ViT** (e.g., DINOv2):
```
Layer 4:  [B, 1024, num_patches] → Low-level BUT global!
Layer 11: [B, 1024, num_patches] → Mid-level, global
Layer 23: [B, 1024, num_patches] → High-level, focused
          ↑ Resolution 동일!
```

**Key difference**: ViT는 처음부터 global attention 가능, 하지만 semantic은 깊은 layer에서 발달.

---

## 마지막 체크리스트

구현 시 확인 사항:
- [ ] `encoder_features[0]`이 Layer 4인지 확인 (ViT-L)
- [ ] `encoder_features[-1]`이 Layer 23인지 확인
- [ ] CLS token 제거 (`[:, 1:, :]`)
- [ ] Forward signature 변경 (2개의 patch_tokens 인자)
- [ ] train_gear3.py에서 early/late features 전달
- [ ] Visualization에서 importance map 확인

**구현 난이도**: ⭐⭐☆☆☆ (쉬움, 레이어 크기 동일하므로)
