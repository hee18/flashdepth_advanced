# FlashDepth Gear3: Feature-level Metric Depth Learning

**작성일**: 2025-10-02
**최종 업데이트**: 2025-10-13
**브랜치**: gear3
**목적**: Feature-level FiLM modulation을 통한 metric depth 학습

**학습 가능 파라미터**: **9.2M / 329M (2.81%)** ⭐
- Gear3 modules: 4.6M
- Mamba: 4.3M
- output_conv: 0.3M

**주요 변경사항 (2025-10-13)**:
- ❌ **Canonicalization 완전 제거** (불필요, raw inverse depth로 충분)
- ✅ **Importance map: Raw attention weights 직접 사용** (학습 불필요)
- ❌ **모든 regularization losses 제거** (importance map 학습 안함)
- ✅ **DPT output_conv만 학습**: Modulated features → depth 변환

**이전 변경사항 (2025-10-09)**:
- ✅ DepthVariancePseudoLabelLoss 추가 (GT depth variance → importance map pseudo-label)
- ❌ BimodalLoss 완전 제거 (continuous modulation 방해: binary 강제 → variance supervision과 충돌)
- ⚠️ EdgeAwareLoss 기본 비활성화 (variance supervision으로 충분)
- ✅ ContrastiveFGBGLoss weight 증가: 0.1 → 0.3 (FG/BG feature separation 강화)

---

## 목차

1. [개요](#개요)
2. [핵심 아이디어](#핵심-아이디어)
3. [아키텍처 설계](#아키텍처-설계)
4. [학습 전략](#학습-전략)
5. [손실 함수 및 Valid Depth Range](#손실-함수-및-valid-depth-range)
6. [사용 방법](#사용-방법)
7. [최적화 설정](#최적화-설정)
8. [기대 효과](#기대-효과)

---

## 개요

Gear3는 FlashDepth에 **Feature-level Metric Injection**을 적용하여 metric depth 추정 성능을 향상시킵니다.
기존 GSP(Global Scale Predictor)와 달리, **DPT feature에 직접 FiLM-style modulation**을 적용합니다.

### 기존 방식 vs Gear3

| 방식            | GSP (기존)              | Gear3 (현재)                  |
|----------------|------------------------|-------------------------------|
| **Metric 주입** | Depth map에 scale/shift | **DPT features에 modulation** |
|  **공간 변화**   |  전역 균일               | **Importance map 기반 spatial modulation** |
| **FG/BG 구분**  |  없음                   | **Separate FG/BG modulation** |
| **학습 파라미터** |  ~0.5M                 | **~9.2M** (Gear3 + Mamba + output_conv) |

---

## 핵심 아이디어

### 1. FiLM-style Feature Modulation

```python
# Spatial-adaptive modulation
gamma[x,y] = importance[x,y] × γ_fg + (1 - importance[x,y]) × γ_bg
beta[x,y] = importance[x,y] × β_fg + (1 - importance[x,y]) × β_bg
modulated_feature[x,y] = gamma[x,y] ⊙ feature[x,y] + beta[x,y]
```

### 2. Attention-based Importance Map (학습 불필요!)

**핵심 변경**: ImportancePredictor 제거, **DINOv2 attention weights를 직접 importance map으로 사용**

```python
# gear3_modules.py - Gear3MetricHead
def forward(self, patch_tokens, attention_weights, dpt_features, patch_h, patch_w):
    # CLS→patch attention (semantic importance)
    cls_to_patch = attention_weights[:, :, 0, 1:]  # [B, num_heads, num_patches]

    # Average over heads and reshape to spatial map
    importance_map = cls_to_patch.mean(dim=1)  # [B, num_patches]
    importance_map = importance_map.view(B, 1, patch_h, patch_w)  # [B, 1, H, W]

    # No learning needed - raw attention already provides semantic importance!
    return modulated_features, importance_map
```

**장점**:
- ❌ **ImportancePredictor 제거**: ~1.2M params 절약
- ✅ **DINOv2의 검증된 semantic attention 활용**: 추가 학습 불필요
- ✅ **Gradient flow 자동 보장**: No zero init tricks needed
- ✅ **메모리 효율**: Last block만 저장 (~11GB 절약)

---

## 아키텍처 설계

### 전체 파이프라인

```
Video Frame → DINOv2-L (frozen) → Patch Tokens + Last Block Attention
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
         Raw Attention → Importance Map    ForegroundBackgroundNetworks
         (No learning!)    [B,1,H,W]      (Attention-based Pooling)
                    ↓                                   ↓
                    ↓                      FG/BG Features [B,256]
                    ↓                   (Top attn → FG, Low attn → BG)
                    └─────────────────┬─────────────────┘
                                      ↓
                            ModulationNetworks (×4 layers)
                                      ↓
                          γ_fg, β_fg, γ_bg, β_bg [B,256]
                                      ↓
         DPT-L Features → FeatureModulator → Modulated Features
                                      ↓
                      DPT Refinement (frozen) + Mamba (trainable)
                                      ↓
                     output_conv1/2 (trainable from scratch)
                                      ↓
                            Inverse Depth (100/m, 직접 출력)
```

### 주요 모듈

#### 1. ~~ImportancePredictor~~ (제거됨!)

**변경**: Raw attention weights를 직접 importance map으로 사용
```python
# gear3_modules.py - Gear3MetricHead.forward()
cls_to_patch = attention_weights[:, :, 0, 1:]  # [B, 16, num_patches]
importance_map = cls_to_patch.mean(dim=1)  # [B, num_patches]
importance_map = importance_map.view(B, 1, patch_h, patch_w)  # [B, 1, H, W]
```

**장점**: 학습 불필요, DINOv2 semantic attention 활용

#### 2. ForegroundBackgroundNetworks

**Option 3: Attention-based Pooling** (현재 사용):
```python
def forward(self, patch_tokens, attention_weights):
    # Extract CLS→patch attention (semantic importance)
    cls_to_patch_attn = attention_weights[:, :, 0, 1:]  # [B, num_heads, num_patches]
    attn_scores = cls_to_patch_attn.mean(dim=1)  # [B, num_patches]

    # Split by median: top 50% = FG, bottom 50% = BG
    median = attn_scores.median(dim=1, keepdim=True).values
    fg_mask = (attn_scores > median).float()
    bg_mask = (attn_scores <= median).float()

    # Attention-weighted pooling
    fg_weights = attn_scores * fg_mask
    bg_weights = (1.0 - attn_scores) * bg_mask
    fg_pooled = (patch_tokens * fg_weights.unsqueeze(-1)).sum(dim=1)  # [B, 1024]
    bg_pooled = (patch_tokens * bg_weights.unsqueeze(-1)).sum(dim=1)  # [B, 1024]

    # Pass through separate networks
    fg_features = self.fg_net(fg_pooled)  # [B, 256]
    bg_features = self.bg_net(bg_pooled)  # [B, 256]
    return fg_features, bg_features
```

**핵심**:
- **DINOv2의 검증된 semantic attention 활용** (frozen but powerful)
- Top attention patches → FG (semantic objects)
- Bottom attention patches → BG (context)
- **FG ≠ BG 자동 보장** → Importance map gradient flow 가능

### 차원 흐름도 (Complete Forward Pass)

**ViT-L 기준, 입력 해상도 518×518**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input Image:                    [B, 3, 518, 518]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. DINOv2 ENCODER (Frozen)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patch Embedding (14×14 patches):
  → num_patches = 518÷14 × 518÷14 = 37 × 37 = 1369
  → Total tokens = 1369 + 1 (CLS) = 1370

Encoder Output (24 layers):
  → Intermediate layers [4, 11, 17, 23]:
      Layer 4:                  [B, 1370, 1024]  ← Early features
      Layer 11:                 [B, 1370, 1024]  ← Mid features
      Layer 17:                 [B, 1370, 1024]  ← Late-mid features
      Layer 23:                 [B, 1370, 1024]  ← Late features (사용)

  → Last Block Attention Weights:
      attention_weights:        [B, 16, 1370, 1370]
                                   ↑ num_heads
      CLS→Patch attention:      [B, 16, 1369]  ← Extract [:, :, 0, 1:]

  → Patch Tokens (without CLS): [B, 1369, 1024]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. DPT FEATURE EXTRACTION (Frozen)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DPT Head processes 4 intermediate layers:

  Path 4 (from Layer 4):        [B, 256, 37, 37]  ← Will be modulated
  Path 3 (from Layer 11):       [B, 256, 37, 37]  ← Will be modulated
  Path 2 (from Layer 17):       [B, 256, 37, 37]  ← Will be modulated
  Path 1 (from Layer 23):       [B, 256, 37, 37]  ← Will be modulated 

  dpt_features = [path_4, path_3, path_2, path_1] 


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. GEAR3 HEAD - Importance Map (No Learning!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raw Attention → Importance Map:
  Input:  attention_weights     [B, 16, 1370, 1370]

  Extract CLS→patch:            [B, 16, 1369]  ← [:, :, 0, 1:]

  Average over heads:           [B, 1369]

  Reshape to spatial:           [B, 1, 37, 37]  ← Importance map

  **No trainable parameters!**  DINOv2 attention provides semantic importance


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. GEAR3 HEAD - FG/BG Feature Extraction (Trainable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ForegroundBackgroundNetworks (Option 3: Attention-based Pooling):

  Input:
    patch_tokens:               [B, 1369, 1024]
    attention_weights:          [B, 16, 1369]  ← CLS→patch attention

  Average over heads:           [B, 1369]  ← attn_scores

  Median split (binary masks):
    fg_mask (top 50%):          [B, 1369]  ← Binary: 1 if attn > median, 0 otherwise (~685 ones)
    bg_mask (bottom 50%):       [B, 1369]  ← Binary: 1 if attn ≤ median, 0 otherwise (~684 ones)

  Weighted pooling:
    fg_pooled:                  [B, 1024]  ← Sum over FG patches only (masked sum)
    bg_pooled:                  [B, 1024]  ← Sum over BG patches only (masked sum)

  FG Network (MLP):
    fg_pooled → Linear(1024→512) → ReLU → Linear(512→256) → ReLU
    fg_features:                [B, 256]   ← FG semantic features

  BG Network (MLP):
    bg_pooled → Linear(1024→512) → ReLU → Linear(512→256) → ReLU
    bg_features:                [B, 256]   ← BG context features


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. GEAR3 HEAD - Modulation Parameters (Trainable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ModulationNetworks (for each DPT layer, layer_idx ∈ {0,1,2,3}):

  Input:
    fg_features:                [B, 256]
    bg_features:                [B, 256]

  FG Modulation Network:
    fg_features → Linear(256→512) → ReLU → Linear(512→512)
    fg_params:                  [B, 512]
    Split:
      fg_gamma:                 [B, 256]  ← First half
      fg_beta:                  [B, 256]  ← Second half

  BG Modulation Network:
    bg_features → Linear(256→512) → ReLU → Linear(512→512)
    bg_params:                  [B, 512]
    Split:
      bg_gamma:                 [B, 256]
      bg_beta:                  [B, 256]

  Output per layer: (fg_gamma, fg_beta, bg_gamma, bg_beta)
                    각각 [B, 256]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. FEATURE MODULATION (Trainable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FeatureModulator (for each DPT layer):

  Input:
    features:                   [B, 256, 37, 37]  ← DPT path features
    importance_map:             [B, 1, 37, 37]
    fg_gamma, fg_beta:          [B, 256]
    bg_gamma, bg_beta:          [B, 256]

  Resize importance_map (해상도 달라질 경우):        [B, 1, 37, 37]  ← Bilinear interpolation

  Expand modulation params:
    fg_gamma:                   [B, 256, 1, 1]  → Broadcast
    fg_beta:                    [B, 256, 1, 1]  → Broadcast
    bg_gamma:                   [B, 256, 1, 1]  → Broadcast
    bg_beta:                    [B, 256, 1, 1]  → Broadcast

  Compute spatial-adaptive params:
    gamma[b,c,h,w] = importance[b,0,h,w] × fg_gamma[b,c,0,0]
                   + (1 - importance[b,0,h,w]) × bg_gamma[b,c,0,0]
    beta[b,c,h,w]  = importance[b,0,h,w] × fg_beta[b,c,0,0]
                   + (1 - importance[b,0,h,w]) × bg_beta[b,c,0,0]

    gamma:                      [B, 256, 37, 37]
    beta:                       [B, 256, 37, 37]

  Apply FiLM modulation:
    modulated_features:         [B, 256, 37, 37]
                                = gamma ⊙ features + beta


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. DPT OUTPUT HEAD (Trainable from scratch)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use modulated path_1:           [B, 256, 37, 37]

output_conv1 (Conv 3×3):        [B, 256, 37, 37]
                                  ↓ ReLU

Interpolate to input size:      [B, 256, 518, 518]  ← Bilinear upsampling

output_conv2 (Conv 3×3):        [B, 1, 518, 518]
                                  ↓ Softplus (positive)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. FINAL OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Predicted Inverse Depth:        [B, 1, 518, 518]
  (in 100/meters scale, NO canonicalization)

Ground Truth (for training):    [B, 1, 518, 518]
  (in 100/meters scale, NO canonicalization)

Valid Mask:                     [B, 1, 518, 518]
  (inverse_depth > 0.5, i.e., depth < 200m)
```

### 메모리 사용량 분석 (Batch=20, BFloat16)

| Component | Shape | Memory | Notes |
|-----------|-------|--------|-------|
| **Encoder outputs** | [20, 1370, 1024] × 4 | ~432MB | 4 intermediate layers |
| **Attention weights** | [20, 16, 1370, 1370] | ~1.0GB | Only last block |
| **DPT features** | [20, 256, 37, 37] × 4 | ~1.0MB | 4 DPT paths |
| **Importance map** | [20, 1, 37, 37] | ~0.06MB | Spatial importance |
| **Modulation params** | [20, 256] × 4 × 4 | ~0.16MB | FG/BG γ/β per layer |
| **Output** | [20, 1, 518, 518] | ~10.5MB | Final depth |
| **Total (forward)** | - | **~1.7GB** | Per GPU |

**Peak Memory (training)**:
- Forward: ~1.7GB
- Gradients: ~1.7GB (trainable params only)
- Optimizer states: ~3.4GB (Adam: 2× gradients)
- Activations (gradient checkpointing): ~5GB
- **Total per GPU**: ~34GB / 48GB ✅

---

## 학습 전략

### 파라미터 설정

| 모듈 | 학습 여부 | LR | 파라미터 수 | 비고 |
|------|----------|-----|------------|------|
| DINOv2 Encoder | ❌ Frozen | - | ~300M | FlashDepth-L 로드 |
| DPT projects/resize | ❌ Frozen | - | ~5M | FlashDepth-L 로드 |
| DPT refinenet | ❌ Frozen | - | ~15M | FlashDepth-L 로드 |
| **DPT output_conv** | ✅ **Train from scratch** | **1e-4** | ~0.3M | **로드 안함** |
| **Mamba** | ✅ **Train from scratch** | **1e-4** | ~4.3M | **로드 안함** |
| **Gear3 modules** | ✅ Train | **1e-4** | ~4.6M | 신규 |

**총 학습 가능**: ~9.2M
**이유**: Modulated features를 받는 모듈은 사전학습 가중치 불가

### 학습 설정 (최적화됨)

**Hardware**: 2× RTX A6000 (48GB each), 96 CPU cores, 503GB RAM

```yaml
# configs/gear3/config.yaml
training:
  batch_size: 20        # Per GPU (effective 40 with DDP)
  workers: 8            # Optimized for 96 cores
  iterations: 60001

  # Learning rates
  gear3_lr: 1.0e-4      # Gear3 modules
  mamba_lr: 1.0e-4      # Mamba (from scratch)
```

**Scheduler**: Cosine Annealing with Warmup
```
Warmup (0-10%):    1e-5 → 1e-4
Stable (10-30%):   1e-4
Decay (30-100%):   1e-4 → 1e-6
```

### Multi-GPU (DDP) 설정

```bash
# train_gear3_ddp.sh
export CUDA_VISIBLE_DEVICES=0,1

torchrun \
    --nproc_per_node=2 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_gear3.py \
    --config-path configs/gear3 \
    dataset.data_root=/data/datasets \
    phase=1 \
    training.batch_size=20 \
    training.workers=8 \
    +results_dir=train_results/results_7 \
    load=configs/flashdepth-l/iter_10001.pth
```

**Memory 최적화**:
- **Shared memory**: 16GB (Docker `shm_size`)
- **Attention weights**: Last block만 저장 (~11GB 절약)
- **BFloat16 autocast**: Training & validation

---

## 손실 함수 및 Valid Depth Range

### Primary Loss: Inverse Depth Loss (100/m scale)

```python
# Training loop (NO canonicalization!)
gt_depth_inverse = gt_depth * 100.0  # Dataloader gives 1/m, scale to 100/m

# Valid mask: 0.5 < inverse_depth (i.e., depth < 200m)
MIN_INVERSE_DEPTH = 0.5  # 100/200m = 0.5
valid_mask = (gt_inverse > MIN_INVERSE_DEPTH)

depth_loss = LogL1Loss(pred_inverse, gt_inverse, valid_mask)
```

### Regularization Losses: **모두 제거됨!** ❌

**핵심 변경 (2025-10-13)**:
- ✅ **Importance map = Raw attention weights** (학습 불필요)
- ❌ **모든 regularization losses 제거**:
  - DepthVariancePseudoLabelLoss ❌
  - EdgeAwareLoss ❌
  - ContrastiveFGBGLoss ❌

**이유**:
- Importance map은 DINOv2 attention weights를 직접 사용 → **학습 파라미터 없음**
- Regularization은 importance map 학습을 위한 것 → **불필요**
- FG/BG networks와 modulation networks만 학습

**이전 시도 (참고용)**:
1. EntropyLoss (2025-10-07): Uniform 장려 → 제거
2. BimodalLoss (2025-10-09): Binary 강제 → 제거
3. DepthVarianceLoss (2025-10-09): Variance supervision → 제거 (importance map 학습 안함)
4. ContrastiveFGBGLoss (2025-10-09): FG/BG separation → 제거 (modulation만으로 충분)

### Total Loss (현재)

```python
# 현재 (2025-10-13): Depth loss만 사용!
total_loss = depth_loss  # Log L1 loss on inverse depth

# Regularization losses 모두 제거됨
```

**핵심 단순화**:
- ✅ Depth loss만으로 학습
- ❌ Importance map regularization 불필요 (raw attention 사용)
- ❌ FG/BG contrastive loss 불필요 (modulation만으로 충분)

---

### 이전 Loss (참고용 - 더 이상 사용 안 함)

<details>
<summary><b>DepthVariancePseudoLabelLoss</b> (제거됨)</summary>

Importance map 학습을 위한 loss였으나, raw attention 사용으로 불필요해짐.
</details>

<details>
<summary><b>EdgeAwareLoss</b> (제거됨)</summary>

Depth edge alignment을 위한 loss였으나, importance map 학습 안 함.
</details>

<details>
<summary><b>ContrastiveFGBGLoss</b> (제거됨)</summary>

FG/BG feature separation을 위한 loss였으나, modulation 학습만으로 충분.
</details>

### Valid Depth Range: **200m 이하**

**배경**:
- 원본 FlashDepth: 70m (KITTI/NYUv2 기준)
- Gear3: **200m** (TartanAir, Spring 고려)

**적용**:
1. **Training loss**: `gt_inverse > 0.5` (depth < 200m)
2. **Validation loss**: 동일
3. **Visualization**: `0 < depth < 200` 필터링

**이유**:
- TartanAir: 대부분 200m 이내
- Spring (outdoor): 200m로 커버 가능
- Infinity depth (10억m) 제거
- 70m보다 넓지만 안정적

```python
# utils/gear3_visualization.py
MAX_DEPTH = 200.0  # meters
gt_valid_mask = (gt_depth > 0) & (gt_depth < MAX_DEPTH)
```

### ~~Canonical Space~~ (완전 제거)

**변경 (2025-10-13)**: Canonicalization 로직 완전 제거
- ❌ `CanonicalSpaceNormalizer` 제거
- ❌ `canonicalize_inverse()` 제거
- ✅ **Raw inverse depth 직접 사용**: `gt_depth * 100.0`

**이유**: Focal length 정규화 불필요, 학습 간단화

---

## 사용 방법

### Phase 1 학습 (5개 dataset)

```bash
# Native (추천)
bash train_gear3_ddp.sh \
  --config-path configs/gear3 \
  dataset.data_root=/data/datasets \
  phase=1 \
  training.batch_size=20 \
  training.workers=8 \
  +results_dir=train_results/results_7 \
  load=configs/flashdepth-l/iter_10001.pth

# Docker
./run_docker.sh train_gear3_ddp \
  --batch-size 20 \
  --workers 8 \
  --results-dir train_results/results_7
```

**Datasets (Phase 1)** (순서: 원본 FlashDepth-L과 동일):
1. mvs-synth
2. dynamicreplica
3. tartanair
4. pointodyssey
5. spring

**Validation**: sintel, waymo (각 dataset에서 1개씩 시각화)

### Phase 2 학습 (nuScenes)

```bash
bash train_gear3_ddp.sh \
  --config-path configs/gear3 \
  phase=2 \
  load=train_results/results_7/best_checkpoint.pth
```

### 커스텀 옵션

```bash
# Batch size 변경
--batch-size 20              # Per GPU

# Workers 변경
--workers 6

# Results directory
--results-dir train_results/custom

# Loss 옵션 (기본값: depth-variance + contrastive 활성화, edge-aware 비활성화)
--depth-variance-loss true   # DepthVariancePseudoLabelLoss 활성화/비활성화
--depth-variance-weight 0.5  # DepthVarianceLoss weight (0.3-1.0 권장)
--variance-kernel-size 15    # Gaussian kernel size on GT resolution (7-21 권장)

--edge-aware-loss false      # EdgeAwareLoss 활성화/비활성화 (기본 비활성화)
--edge-aware-weight 0.3      # EdgeAwareLoss weight (0.1-0.5 권장)

--contrastive-loss true      # ContrastiveFGBGLoss 활성화/비활성화
--contrastive-weight 0.3     # ContrastiveFGBGLoss weight (0.1-0.5 권장)
```

**Loss 실험 예시**:

```bash
# 기본 설정 (variance + contrastive)
./run_docker.sh train_gear3_ddp \
  --results-dir train_results/results_default

# 모든 loss 사용 (EdgeAwareLoss 포함)
./run_docker.sh train_gear3_ddp \
  --depth-variance-loss true \
  --edge-aware-loss true \
  --contrastive-loss true \
  --results-dir train_results/results_all_losses

# Weight 조정
./run_docker.sh train_gear3_ddp \
  --depth-variance-weight 0.7 \
  --contrastive-weight 0.5 \
  --results-dir train_results/results_high_weight

# Kernel size 실험
./run_docker.sh train_gear3_ddp \
  --variance-kernel-size 21 \
  --results-dir train_results/results_kernel21
```

### 테스트

```bash
python test_gear3.py \
  --config-path configs/gear3 \
  +flashdepth_checkpoint=train_results/results_7/final.pth \
  +results_dir=test_results/results_7 \
  +gpu=0
```

---

## 최적화 설정

### GPU 메모리 사용량

**Current (batch_size=20 per GPU)**:
- GPU 0: ~41GB / 48GB (**7GB 여유**)
- GPU 1: ~33GB / 48GB (15GB 여유)
- GPU utilization: **100%** ✅

**가능한 최대**: batch_size=24 (GPU 0 기준 ~46GB)

### RAM 사용량

- Total: 503GB
- Used: ~36GB
- Shared memory: 24GB / 252GB (10%)
- **여유 충분** ✅

### CPU 사용량

- 사용률: ~14.5% (96 cores)
- Workers: 22 processes (2 GPUs × 8 + overhead)
- **병목 없음** ✅

### Validation 최적화

**문제 해결**:
1. ✅ BFloat16 autocast (메모리 절약)
2. ✅ Frame-by-frame tensor deletion
3. ✅ Attention weights 초기화 (last block만 사용)
4. ✅ Cache clearing before/after validation
5. ✅ **각 dataset별 시각화** (sintel, waymo 각각 1개)

**Validation 시간**: ~32초 (전체 dataset)

**시각화 개선**:
- **이전**: `validation_step_005000.png` (항상 첫 번째 batch, sintel만)
- **현재**:
  - `validation_sintel_step_005000.png` (sintel 첫 번째 batch)
  - `validation_waymo_step_005000.png` (waymo 첫 번째 batch)
- **장점**: 각 dataset의 학습 진행 상황을 개별적으로 추적 가능

---

## 시각화 및 메트릭

### Visualization 출력

**위치**: `train_results/results_X/visualizations/`

**내용**:
```
Row 1: Input | GT Depth | Pred Depth | Importance Map
Row 2: Valid Mask | Error Map | Metrics | Training Info
Row 3: Depth Distribution | Importance Distribution
```

**Importance Map 디버깅**:
```
Step X:
  GT raw range: 2.676 - 200.000
  Pred raw range: 126.291 - 141.559
  Invalid GT pixels (>200m or <0): 123456
  GT valid range: 2.676 - 178.234
  Valid pixels: 2000000 / 2109744
```

### 메트릭

```json
{
  "mae": 1.234,           // Mean Absolute Error
  "rmse": 2.345,          // Root Mean Squared Error
  "abs_rel": 0.0567,      // Absolute Relative Error
  "a1": 0.945,            // δ < 1.25
  "a2": 0.987,            // δ < 1.25²
  "a3": 0.995             // δ < 1.25³
}
```

**Max depth = 200m** 적용됨 (visualization & metrics)

---

## 기대 효과

### 1. 학습 속도

**이전 (batch 8, single GPU)**:
- Training time: ~140시간

**현재 (batch 20×2, DDP)**:
- Effective batch: **40** (5배!)
- Training time: **~28시간** (80% 감소)

### 2. 메모리 효율

- Attention weights: **11GB 절약** (last block만 저장)
- BFloat16: ~30% 메모리 감소
- Shared memory: 2GB → 16GB (OOM 해결)

### 3. Metric Depth 정확도

- **FG/BG 분리**: Importance-based modulation
- **Valid range**: 200m (기존 70m보다 현실적)
- **Feature-level**: Scale/shift보다 표현력 풍부

---

## 주요 버그 수정 이력

### 2025-10-13 ⭐⭐⭐⭐⭐

1. **Importance map 학습 완전 제거** ⭐⭐⭐⭐⭐
   - 변경: ImportancePredictor 제거 → **Raw attention weights 직접 사용**
   - 구현:
     ```python
     cls_to_patch = attention_weights[:, :, 0, 1:]
     importance_map = cls_to_patch.mean(dim=1).view(B, 1, H, W)
     ```
   - 효과:
     - ~1.2M params 절약
     - DINOv2 semantic attention 활용
     - Gradient flow 자동 보장 (zero init tricks 불필요)

2. **모든 regularization losses 제거** ⭐⭐⭐⭐
   - 제거됨:
     - DepthVariancePseudoLabelLoss ❌
     - EdgeAwareLoss ❌
     - ContrastiveFGBGLoss ❌
   - 이유: Importance map 학습 안 함 → Regularization 불필요
   - 결과: **Depth loss만 사용** (학습 단순화)

3. **Canonicalization 완전 제거** ⭐⭐⭐
   - 제거됨:
     - `CanonicalSpaceNormalizer` class
     - `canonicalize_inverse()` / `decanonicalize_inverse()`
   - 변경: `gt_depth * 100.0` (직접 스케일링)
   - 이유: Focal length 정규화 불필요, raw inverse depth로 충분

4. **학습 가능 파라미터 정리**
   - 총 9.2M / 329M (2.81%)
   - Gear3 modules: 4.6M
   - Mamba: 4.3M
   - output_conv: 0.3M

### 2025-10-09

1. **BimodalLoss 설계 오류 발견 및 완전 제거** ⭐⭐⭐⭐
   - 문제: Binary 0/1 강제 → Continuous spatial modulation 이점 상실
   - 분석:
     - BimodalLoss는 importance map을 segmentation mask처럼 만듦
     - FiLM modulation의 핵심인 continuous spatial adaptation을 방해
     - Variance supervision과 근본적으로 충돌 (binary vs continuous)
   - 해결: **완전 제거** (no replacement needed, variance supervision is sufficient)

2. **DepthVariancePseudoLabelLoss 도입** (→ 2025-10-13에 제거됨)
   - 아이디어: GT depth의 local variance를 importance map의 pseudo-label로 사용
   - 구현:
     - Gaussian-weighted variance computation (kernel_size=15, sigma=3.0)
     - **CRITICAL**: `torch.no_grad()` for GT depth to prevent gradient flow
     - L1 loss between importance_map and normalized variance
   - 효과: Continuous spatial modulation, natural edge/object emphasis
   - Weight: 0.5 (main supervision for importance map)
   - **상태**: 제거됨 (importance map 학습 안 함)

3. **ContrastiveFGBGLoss weight 증가** (→ 2025-10-13에 제거됨)
   - 변경: 0.1 → 0.3
   - 이유: Loss range [-14.3, 14.3] with temp=0.07 → depth loss 대비 약함
   - 효과: FG/BG feature separation 강화, modulation parameters 차이 증가
   - **상태**: 제거됨 (modulation만으로 충분)

4. **EdgeAwareLoss 기본 비활성화** (→ 2025-10-13에 완전 제거됨)
   - 이유: Variance loss가 edge 정보를 암묵적으로 제공
   - 상태: Optional (재활성화 가능하지만 기본적으로 불필요)
   - Weight: 0.3 (활성화 시)
   - **상태**: 제거됨

### 2025-10-07

1. **EntropyLoss 설계 오류 발견 및 수정** ⭐⭐⭐
   - 문제: Shannon entropy maximization → Uniform distribution 장려
   - 결과: Importance map std=0.001, FG/BG 구분 실패
   - 분석:
     - Uniform (p≈0.5): H=log(HW)=6.93 → Loss=-6.93 (Best!) ❌
     - Bimodal (p≈0 or 1): H=log(HW/2)=6.24 → Loss=-6.24 (Worse) ❌
   - 해결: 3가지 새로운 loss 도입 (BimodalLoss는 2025-10-09에 제거됨)
     - ~~**BimodalLoss**~~: min(p, 1-p) → 0 또는 1로 push (제거됨)
     - **EdgeAwareLoss**: Depth edge와 importance edge align (기본 비활성화)
     - **ContrastiveFGBGLoss**: FG/BG features embedding space 분리 (weight 증가)

2. **Validation 시각화 개선**
   - 문제: 항상 동일한 dataset (sintel)만 시각화
   - 해결: 각 dataset (sintel, waymo)에서 1개씩 시각화
   - 파일명: `validation_{dataset}_step_{step:06d}.png`

3. **Dataset 순서 원본과 통일**
   - 변경: 원본 FlashDepth-L과 동일한 순서로 정렬
   - 순서: `[mvs-synth, dynamicreplica, tartanair, pointodyssey, spring]`

4. **Command-line loss 옵션 추가**
   - Docker script에 모든 loss toggle 추가
   - 2025-10-09 업데이트: bimodal → depth-variance 교체

### 2025-10-06

1. **Attention weights 메모리 누수**
   - 문제: 24 blocks × 11.5GB = OOM
   - 해결: Last block만 저장 (`store_attn_weights` flag)

2. **Importance map uniform 문제 (초기 발견)**
   - 문제: std=0.000 (모든 픽셀 동일)
   - 임시 해결: Zero initialization (gradient flow 확보)
   - 근본 원인: EntropyLoss 설계 오류 (2025-10-07 발견)

3. **Shared memory 부족**
   - 문제: 2GB → Bus error (batch 16+)
   - 해결: 16GB로 증가

4. **Valid depth range 불일치**
   - 문제: 10억m outliers
   - 해결: 200m 상한선 통일 (training, val, viz)

5. **BFloat16 dtype 에러**
   - 문제: `importance_map (float) vs gamma (bfloat16)`
   - 해결: `.to(bg_gamma.dtype)` 추가

---

## 파일 구조

```
flashdepth_claude/
├── flashdepth/
│   ├── gear3_modules.py          # Gear3 핵심 모듈
│   ├── dinov2_layers/
│   │   └── attention.py          # Attention weights 최적화
│   └── original_dpt.py
├── utils/
│   ├── gear3_visualization.py    # Visualization (200m 필터)
│   └── metric_depth_metrics.py   # Metrics 계산
├── dataloaders/
│   ├── combined_dataset.py       # Multi-dataset loader
│   └── spring_dataset.py         # Validation dataset
├── configs/gear3/
│   └── config.yaml               # Gear3 설정 (batch 20, workers 8)
├── train_gear3.py                # Main training script
├── train_gear3_ddp.sh            # DDP launch script
├── run_docker.sh                 # Docker runner
├── docker-compose.yml            # Docker config (shm 16GB)
└── flashdepth_gear3.md           # 이 문서
```

---

## 참고 문헌

1. **FiLM (2018)**: "FiLM: Visual Reasoning with a General Conditioning Layer"
2. **FlashDepth (2024)**: DINOv2 + DPT + Mamba 기반 아키텍처
3. **PyTorch DDP**: Distributed Data Parallel 공식 문서

---

## 요약

### 최종 아키텍처 (2025-10-13)

```
DINOv2 (frozen) → Attention → Raw Importance Map (no learning!)
                           ↓
                   FG/BG Networks (trainable)
                           ↓
               Modulation Networks (trainable)
                           ↓
         DPT Features → Feature Modulator
                           ↓
           Mamba (trainable) + DPT Refinement (frozen)
                           ↓
                output_conv1/2 (trainable)
                           ↓
              Inverse Depth (100/m scale)
```

**핵심**:
- ✅ **Importance map = Raw attention** (학습 불필요)
- ✅ **Depth loss만 사용** (regularization 제거)
- ✅ **Canonicalization 제거** (raw inverse depth)
- ✅ **학습 파라미터: 9.2M / 329M (2.81%)**

---

**Last Update**: 2025-10-13
**Branch**: gear3
**Developer**: hsy
