# FlashDepth Gear3: Feature-level Metric Depth Learning

**작성일**: 2025-10-02
**최종 업데이트**: 2025-10-06
**브랜치**: gear3
**목적**: Feature-level FiLM modulation을 통한 metric depth 학습

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

### 2. Attention-based Importance Prediction

**입력**: DINOv2 **last block의 CLS attention weights만** 사용 (메모리 최적화)
- 이전: 24 blocks × 480MB = **11.5GB** 낭비
- 현재: 1 block × 480MB = **0.5GB** ✅

```python
# flashdepth/dinov2_layers/attention.py
class MemEffAttention(Attention):
    def __init__(self):
        self.store_attn_weights = False  # Default: 저장 안 함

# train_gear3.py - Last block만 활성화
for i, block in enumerate(model.pretrained.blocks):
    if i == len(model.pretrained.blocks) - 1:
        block.attn.store_attn_weights = True
```

### 3. Zero Initialization for Gradient Flow

```python
# gear3_modules.py - ImportancePredictor
nn.init.zeros_(self.conv3.weight)  # 마지막 layer zero init
nn.init.zeros_(self.conv3.bias)    # sigmoid(0) = 0.5 시작
```

**이유**: Random init → sigmoid saturation → no gradient flow → uniform importance map

---

## 아키텍처 설계

### 전체 파이프라인

```
Video Frame → DINOv2-L (frozen) → Patch Tokens + Last Block Attention
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
         ImportancePredictor         ForegroundBackgroundNetworks
         (zero init last layer)      (Attention-based Pooling)
                    ↓                                   ↓
         Importance Map [B,1,H,W]          FG/BG Features [B,256]
                    ↓                   (Top attn → FG, Low attn → BG)
                    └─────────────────┬─────────────────┘
                                      ↓
                            ModulationNetworks (×4 layers)
                                      ↓
                          γ_fg, β_fg, γ_bg, β_bg [B,256]
                                      ↓
         DPT-L Features → FeatureModulator → Modulated Features
                                      ↓
                      DPT Refinement + Mamba (trainable)
                                      ↓
                     output_conv1/2 (trainable from scratch)
                                      ↓
                            Metric Depth (직접 출력)
```

### 주요 모듈

#### 1. ImportancePredictor

```python
class ImportancePredictor(nn.Module):
    def __init__(self, num_heads=16, hidden_dim=128):
        self.conv1 = nn.Conv2d(num_heads, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim//2)

        self.conv3 = nn.Conv2d(hidden_dim//2, 1, 1)
        self.sigmoid = nn.Sigmoid()

        # Zero init for gradient flow
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
```

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

  Path 4 (from Layer 4):        [B, 256, 37, 37]
  Path 3 (from Layer 11):       [B, 256, 37, 37]
  Path 2 (from Layer 17):       [B, 256, 37, 37]
  Path 1 (from Layer 23):       [B, 256, 37, 37]  ← Will be modulated

  dpt_features = [path_4, path_3, path_2, path_1]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. GEAR3 HEAD - Importance Prediction (Trainable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ImportancePredictor:
  Input:  attention_weights     [B, 16, 1369]

  Reshape:                      [B, 16, 37, 37]

  Conv1 + BN + ReLU:            [B, 128, 37, 37]  ← hidden_dim
  Conv2 + BN + ReLU:            [B, 64, 37, 37]   ← hidden_dim//2
  Conv3 (zero init):            [B, 1, 37, 37]
  Sigmoid:                      [B, 1, 37, 37]    ← Importance map (0~1)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. GEAR3 HEAD - FG/BG Feature Extraction (Trainable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ForegroundBackgroundNetworks (Option 3: Attention-based Pooling):

  Input:
    patch_tokens:               [B, 1369, 1024]
    attention_weights:          [B, 16, 1369]  ← CLS→patch attention

  Average over heads:           [B, 1369]  ← attn_scores

  Median split:
    fg_mask (top 50%):          [B, 1369]  ← High attention patches
    bg_mask (bottom 50%):       [B, 1369]  ← Low attention patches

  Weighted pooling:
    fg_pooled:                  [B, 1024]  ← Attention-weighted sum
    bg_pooled:                  [B, 1024]  ← Inverse attention-weighted sum

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

  Resize importance_map:        [B, 1, 37, 37]  ← Bilinear interpolation

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
  (in 100/meters scale, canonicalized)

Ground Truth (for training):    [B, 1, 518, 518]
  (in 100/meters scale, canonicalized)

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

### Inverse Depth Loss (100/m scale)

```python
# Training loop
gt_depth_inverse_canonical = canonicalize_inverse(gt_depth, focal_length)
gt_depth_inverse = gt_depth_inverse_canonical * 100.0  # Scale to 100/m

# Valid mask: 0.5 < inverse_depth (i.e., depth < 200m)
MIN_INVERSE_DEPTH = 0.5  # 100/200m = 0.5
valid_mask = (gt_inverse > MIN_INVERSE_DEPTH)

loss = L1(pred_inverse, gt_inverse, valid_mask)
```

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

### Canonical Space (현재 비활성화)

```yaml
# configs/gear3/config.yaml
use_canonical_space: false  # 사용 안 함
canonical_focal_length: 1000.0
```

**이유**: Raw inverse depth로 충분히 학습 가능

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

**Datasets (Phase 1)**:
- mvs-synth
- pointodyssey
- spring (train split)
- tartanair
- dynamicreplica

**Validation**: spring (val split, 12 sequences)

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

**Validation 시간**: ~32초 (6 sequences, video_length=5)

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

### 2025-10-06

1. **Attention weights 메모리 누수**
   - 문제: 24 blocks × 11.5GB = OOM
   - 해결: Last block만 저장 (`store_attn_weights` flag)

2. **Importance map uniform 문제**
   - 문제: std=0.000 (모든 픽셀 동일)
   - 해결: Zero initialization (gradient flow 확보)

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

**Last Update**: 2025-10-06
**Branch**: gear3
**Developer**: hsy
