# FlashDepth Gear5: Two-Stage Global + Foreground Modulation for Metric Depth

**작성일**: 2025-11-10
**최종 업데이트**: 2025-11-10
**브랜치**: gear5
**목적**: 2단계 학습을 통한 metric depth 정확도 향상 (Global Scale Predictor + Foreground-only Modulation)

**학습 가능 파라미터**:
- Phase 1 Step 1: **GSP (0.26M) + Mamba (4.3M) + output_conv (0.3M) = 4.86M / 340M (1.43%)**
- Phase 1 Step 2: **FFM (2.9M) + Mamba (4.3M) + output_conv (0.3M) = 7.5M / 340M (2.21%)**

## 주요 특징 ⭐⭐⭐⭐⭐

### 1. 2단계 학습 전략

**Step 1 (Global Scale Predictor)**:
- Multi-layer CLS tokens → Global scale/shift 예측
- ViT-L: [4, 11, 17, 23] 4개 레이어 사용
- ViT-S: [2, 5, 8, 11] 4개 레이어 사용
- 전역적 metric scale을 먼저 학습

**Step 2 (Foreground-only Modulation)**:
- Multi-layer attention weights → Foreground feature 추출
- ViT-L: [11, 17] 중간 2개 레이어 사용
- ViT-S: [5, 8] 중간 2개 레이어 사용
- Global modulation 위에 FG-only modulation 추가
- GSP는 frozen, FFM만 학습

### 2. Two-Phase Training (518×518 → 2K Hybrid)

**Phase 1 (518×518, ViT-L)**:
```bash
# Step 1: Global Scale Predictor 학습
python train_gear5.py --config-path configs/gear5 step=1 load=configs/flashdepth-l/iter_10001.pth

# Step 2: Foreground-only Modulation 학습 (GSP frozen)
python train_gear5.py --config-path configs/gear5 step=2 gear_checkpoint=configs/gear5/checkpoint_step40000_step1.pth
```

**Phase 2 (2K, ViT-S Hybrid)**:
```bash
# Step 1: 2K resolution에서 GSP 재학습 (FFM 제외)
python train_gear5.py --config-path configs/gear5/hybrid \
  step=1 \
  gear_checkpoint=configs/gear5/checkpoint_step40000_step2.pth \
  load=configs/flashdepth/iter_43002.pth

# Step 2: FFM 추가하여 전체 모델 학습
python train_gear5.py --config-path configs/gear5/hybrid \
  step=2 \
  gear_checkpoint=configs/gear5/hybrid/checkpoint_step40000_step1.pth \
  phase1_step2_checkpoint=configs/gear5/checkpoint_step40000_step2.pth
```

### 3. Gear5 vs Gear3 비교

| 특징 | Gear3 | Gear5 |
|-----|-------|-------|
| **학습 전략** | Single-stage | Two-stage (GSP → FFM) |
| **Multi-layer** | Step 2만 사용 | Step 1, 2 모두 사용 |
| **CLS tokens** | 사용 안함 | Step 1에서 사용 (4-layer) |
| **Attention layers** | Step 2: 4-layer | Step 2: 2-layer (중간만) |
| **GSP** | Global feature 기반 | Multi-layer CLS 기반 |
| **FFM** | 전체 feature 대상 | Foreground-only |
| **학습 파라미터** | ~9.2M | Step1: 4.86M, Step2: 7.5M |

---

## 목차

1. [개요](#개요)
2. [핵심 아이디어](#핵심-아이디어)
3. [아키텍처 설계](#아키텍처-설계)
4. [2단계 학습 전략](#2단계-학습-전략)
5. [손실 함수](#손실-함수)
6. [사용 방법](#사용-방법)
7. [Dimension Flow](#dimension-flow)
8. [Testing](#testing)
9. [기대 효과](#기대-효과)

---

## 개요

Gear5는 **2단계 학습 전략**을 통해 metric depth 추정 성능을 극대화합니다:

1. **Step 1 (Global Scale Predictor)**: Multi-layer CLS tokens를 사용하여 전역적 scale/shift 예측
2. **Step 2 (Foreground-only Modulation)**: Multi-layer attention weights로 foreground feature를 추출하고, FG-only modulation 적용

### Gear5의 핵심 설계 원칙

1. **Multi-layer Feature Fusion**:
   - Single layer 대신 multiple layers (4개 또는 2개)의 정보를 융합
   - ViT의 계층적 특징을 활용하여 더 풍부한 semantic 정보 획득

2. **Foreground-only Modulation**:
   - Background는 global modulation만으로 충분
   - Foreground만 추가 modulation → 효율적 파라미터 사용

3. **Two-stage Learning**:
   - Step 1: Global scale을 먼저 정확하게 학습
   - Step 2: Global 위에 FG-specific refinement 추가

---

## 핵심 아이디어

### 1. Multi-layer Global Scale Predictor (Step 1)

**기존 GSP의 한계**:
- Single CLS token (last layer) 사용
- 계층적 feature의 이점 활용 못함

**Gear5 GSP**:
```python
# GlobalScalePredictorMultiLayer
# Input: CLS tokens from 4 layers [4, 11, 17, 23] or [2, 5, 8, 11]
# Output: scale (B,), shift (B,)

# 1. Extract CLS tokens from multiple layers
cls_tokens = [encoder_features[i][:, 0] for i in [0, 1, 2, 3]]  # 4 CLS tokens

# 2. Uniform weight fusion (equal ratio)
cls_fused = torch.stack(cls_tokens, dim=1).mean(dim=1)  # [B, embed_dim]

# 3. Predict scale/shift
scale = F.softplus(self.scale_head(cls_fused))  # [B] positive
shift = self.shift_head(cls_fused)  # [B] real number
```

**장점**:
- ✅ 4개 layer의 계층적 정보 활용
- ✅ Uniform weight → 학습 안정성 ↑
- ✅ 추가 파라미터 최소 (~0.26M)

### 2. Multi-layer Foreground Feature Extraction (Step 2)

**기존 attention-based FG mask의 한계**:
- Last block attention만 사용 → 정보 제한적
- FG/BG 구분이 명확하지 않을 수 있음

**Gear5 FG Feature Extraction**:
```python
# ForegroundOnlyModulationHead
# Input: Attention weights from [11, 17] or [5, 8]
# Output: FG features (B, 256), importance map, FG mask

# 1. Extract attention weights from middle 2 layers
attn_weights_multi = [
    model.pretrained.blocks[11].attn.attn_weights,
    model.pretrained.blocks[17].attn.attn_weights
]

# 2. Multi-layer fusion (uniform weights)
attn_fused = torch.stack(attn_weights_multi, dim=0).mean(dim=0)  # [B, H, N, N]

# 3. CLS→patch attention for importance
importance_map = process_attention_to_importance(attn_fused, patch_h, patch_w)

# 4. FG mask generation (threshold-based)
fg_mask = (importance_map > 0.5).float()  # [B, 1, H, W]

# 5. FG patch tokens extraction (ViT의 patch tokens 사용)
fg_patches = extract_foreground_patches(patch_tokens, fg_mask)  # [B, N_fg, C]

# 6. FG feature pooling
fg_features = fg_patches.mean(dim=1)  # [B, C]
```

**장점**:
- ✅ 중간 2개 layer의 attention 융합 → 더 robust한 FG detection
- ✅ ViT patch tokens 직접 사용 → 고품질 semantic feature
- ✅ Binary FG mask → 명확한 FG/BG 구분

### 3. Foreground-only Modulation

**핵심 아이디어**: Background는 global modulation만으로 충분, Foreground만 추가 modulation

```python
# Step 1: Global modulation (전체 영역)
dpt_global = dpt_features * scale.view(B,1,1,1) + shift.view(B,1,1,1)

# Step 2: FG-only modulation (Foreground만)
# 1. FG features → FG-specific γ_fg, β_fg 예측
gamma_fg, beta_fg = self.fg_modulation_predictor(fg_features)  # [B, dpt_dim]

# 2. Spatial broadcast + FG mask 적용
gamma_fg_spatial = gamma_fg.view(B, -1, 1, 1) * fg_mask  # [B, C, H, W]
beta_fg_spatial = beta_fg.view(B, -1, 1, 1) * fg_mask

# 3. FG-only modulation on top of global
dpt_fg_modulated = dpt_global + gamma_fg_spatial * dpt_global + beta_fg_spatial
```

**수식**:
```
Global:  F_global(x,y) = γ_global × F(x,y) + β_global
FG-only: F_final(x,y) = F_global(x,y) + M_fg(x,y) × (γ_fg × F_global(x,y) + β_fg)
```

**장점**:
- ✅ Background는 untouched → global만으로 처리
- ✅ Foreground에만 집중 → 파라미터 효율 ↑
- ✅ Residual connection → 학습 안정성 ↑

---

## 아키텍처 설계

### Overall Architecture

```
Video Frames [B, T, 3, H, W]
    ↓
DINOv2 Encoder (frozen)
    ├→ CLS tokens [Layer 4, 11, 17, 23] → Step 1 (GSP)
    ├→ Attention weights [Layer 11, 17] → Step 2 (FFM)
    └→ Patch tokens [Layer 23] → Step 2 (FG feature extraction)
    ↓
DPT Features (frozen): path_1 [B*T, dpt_dim, H/14, W/14]
    ↓
═══════════════════════════════════════════════════════════
Step 1: Global Scale Predictor (GSP)
═══════════════════════════════════════════════════════════
GlobalScalePredictorMultiLayer:
    Input:  CLS tokens [4, 11, 17, 23] (4 layers)
    Fusion: Uniform weights (equal ratio)
    Output: scale [B], shift [B]

Global Modulation:
    path_1_global = path_1 * scale + shift

    ↓ (if step == 1, return here)

═══════════════════════════════════════════════════════════
Step 2: Foreground-only Modulation (FFM)
═══════════════════════════════════════════════════════════
ForegroundOnlyModulationHead:
    Input:
        - Patch tokens [B*T, N, embed_dim]
        - Attention weights from [11, 17] (2 layers)
        - path_1_global [B*T, dpt_dim, H/14, W/14]

    Process:
        1. Multi-layer attention fusion → importance_map [B, 1, H, W]
        2. Binary FG mask (threshold 0.5)
        3. FG patch extraction → fg_features [B, 256]
        4. FG modulation params: γ_fg, β_fg [B, dpt_dim]
        5. Spatial broadcast + mask → FG-only modulation

    Output: path_1_fg_modulated [B*T, dpt_dim, H/14, W/14]

═══════════════════════════════════════════════════════════
Temporal Modeling (Mamba) - Both Steps
═══════════════════════════════════════════════════════════
Mamba:
    Input:  path_1_modulated (global or global+FG)
    Output: path_1_temporal [B*T, dpt_dim, H/14, W/14]

═══════════════════════════════════════════════════════════
DPT Output Head (Trainable) - Both Steps
═══════════════════════════════════════════════════════════
output_conv1 → upsample to (H, W) → output_conv2
    ↓
Metric Depth [B*T, 1, H, W]
```

### Module Details

#### 1. GlobalScalePredictorMultiLayer

```python
class GlobalScalePredictorMultiLayer(nn.Module):
    """
    Multi-layer CLS token fusion for global scale/shift prediction.

    Architecture:
        4 CLS tokens → Uniform fusion → MLP → scale/shift

    Parameters:
        - embed_dim: DINOv2 embedding dimension (1024 for ViT-L, 384 for ViT-S)

    Trainable params: ~0.26M
    """
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.scale_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.shift_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, cls_tokens_list):
        """
        Args:
            cls_tokens_list: List of [B, embed_dim] CLS tokens from 4 layers

        Returns:
            scale: [B] positive values
            shift: [B] real values
        """
        # Uniform weight fusion
        cls_fused = torch.stack(cls_tokens_list, dim=1).mean(dim=1)  # [B, embed_dim]

        # Predict scale/shift
        scale = F.softplus(self.scale_head(cls_fused).squeeze(-1))  # [B]
        shift = self.shift_head(cls_fused).squeeze(-1)  # [B]

        return scale, shift
```

#### 2. ForegroundOnlyModulationHead

```python
class ForegroundOnlyModulationHead(nn.Module):
    """
    Multi-layer attention fusion for foreground-only modulation.

    Architecture:
        Multi-layer attention → FG mask → FG features → FG modulation

    Parameters:
        - embed_dim: 1024 (ViT-L) or 384 (ViT-S)
        - dpt_dim: 256 (ViT-L) or 64 (ViT-S)

    Trainable params: ~2.9M
    """
    def __init__(self, embed_dim=1024, dpt_dim=256):
        super().__init__()

        # FG feature aggregator
        self.fg_feature_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        # FG modulation predictor
        self.fg_modulation_predictor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dpt_dim * 2)  # γ_fg and β_fg
        )

    def forward(self, patch_tokens, attention_weights_multi_layer,
                dpt_features_global, patch_h, patch_w):
        """
        Args:
            patch_tokens: [B, N, embed_dim]
            attention_weights_multi_layer: List of [B, H, N, N] (2 layers)
            dpt_features_global: [B, dpt_dim, H, W]

        Returns:
            path_1_fg_modulated: [B, dpt_dim, H, W]
            importance_map: [B, 1, H, W]
            fg_features: [B, 256]
            fg_mask: [B, 1, H, W]
        """
        B = dpt_features_global.shape[0]

        # 1. Multi-layer attention fusion
        attn_fused = torch.stack(attention_weights_multi_layer, dim=0).mean(dim=0)

        # 2. Importance map
        importance_map = process_attention_to_importance(attn_fused, patch_h, patch_w)

        # 3. FG mask (binary threshold)
        fg_mask = (importance_map > 0.5).float()

        # 4. Extract FG patch tokens
        fg_patches = extract_foreground_patches(patch_tokens, fg_mask, patch_h, patch_w)

        # 5. FG feature aggregation
        fg_features_raw = self.fg_feature_mlp(fg_patches)  # [B, N_fg, 256]
        fg_features = fg_features_raw.mean(dim=1)  # [B, 256] (average pooling)

        # 6. Predict FG modulation parameters
        fg_params = self.fg_modulation_predictor(fg_features)  # [B, dpt_dim*2]
        gamma_fg = fg_params[:, :self.dpt_dim]  # [B, dpt_dim]
        beta_fg = fg_params[:, self.dpt_dim:]   # [B, dpt_dim]

        # 7. Spatial broadcast + FG mask
        gamma_fg_spatial = gamma_fg.view(B, -1, 1, 1) * fg_mask
        beta_fg_spatial = beta_fg.view(B, -1, 1, 1) * fg_mask

        # 8. FG-only modulation (residual connection)
        path_1_fg_modulated = dpt_features_global + \\
            gamma_fg_spatial * dpt_features_global + beta_fg_spatial

        return path_1_fg_modulated, importance_map, fg_features, fg_mask
```

---

## 2단계 학습 전략

### Phase 1 (518×518, ViT-L)

#### Step 1: Global Scale Predictor 학습

**목표**: Multi-layer CLS tokens로 전역 scale/shift 학습

**Trainable**:
- ✅ GlobalScalePredictorMultiLayer (~0.26M)
- ✅ Mamba (~4.3M)
- ✅ output_conv (~0.3M)

**Frozen**:
- ❌ DINOv2 encoder
- ❌ DPT decoder

**Config**:
```yaml
step: 1
load: configs/flashdepth-l/iter_10001.pth
dataset:
  resolution: 'base'  # 518×518
  train_datasets: [mvs-synth, dynamicreplica, tartanair, pointodyssey, spring]
training:
  batch_size: 20
  iterations: 40001
  gsp_lr: 1.0e-4
  mamba_lr: 1.0e-4
  output_lr: 1.0e-4
```

**Command**:
```bash
python train_gear5.py --config-path configs/gear5 \\
  step=1 \\
  load=configs/flashdepth-l/iter_10001.pth \\
  dataset.data_root=/path/to/data
```

#### Step 2: Foreground-only Modulation 학습

**목표**: FG-only modulation 추가 (GSP frozen)

**Trainable**:
- ✅ ForegroundOnlyModulationHead (~2.9M)
- ✅ Mamba (~4.3M)
- ✅ output_conv (~0.3M)

**Frozen**:
- ❌ DINOv2 encoder
- ❌ DPT decoder
- ❌ GlobalScalePredictorMultiLayer (Step 1에서 학습됨)

**Config**:
```yaml
step: 2
gear_checkpoint: configs/gear5/checkpoint_step40000_step1.pth  # Step 1 checkpoint
training:
  fg_mod_lr: 1.0e-4  # FG modulation LR
  mamba_lr: 5.0e-5   # Lower for fine-tuning
  output_lr: 5.0e-5
```

**Command**:
```bash
python train_gear5.py --config-path configs/gear5 \\
  step=2 \\
  gear_checkpoint=configs/gear5/checkpoint_step40000_step1.pth \\
  dataset.data_root=/path/to/data
```

### Phase 2 (2K, ViT-S Hybrid)

#### Step 1: 2K resolution에서 GSP 재학습 (FFM 제외)

**목표**: 고해상도에서 GSP + Mamba + output 재학습

**Checkpoint Loading**:
1. Phase1 Step2 checkpoint에서 GSP + Mamba + output 로드 (FFM 제외)
2. FlashDepth-hybrid에서 DINOv2 + DPT 덮어쓰기

**Trainable**:
- ✅ GlobalScalePredictorMultiLayer
- ✅ Mamba
- ✅ output_conv

**Frozen**:
- ❌ DINOv2 + DPT (FlashDepth-hybrid)

**Config**:
```yaml
# configs/gear5/hybrid.yaml
step: 1
load: configs/flashdepth/iter_43002.pth  # FlashDepth-hybrid (DINOv2+DPT)
gear_checkpoint: configs/gear5/checkpoint_step40000_step2.pth  # Phase1 Step2 (GSP+Mamba+output, NO FFM)
model:
  vit_size: 'vits'  # ViT-S for hybrid
dataset:
  resolution: '2k'
  train_datasets: [mvs-synth, spring]
training:
  batch_size: 3
  gradient_checkpointing: true
hybrid_configs:
  use_hybrid: true
```

**Command**:
```bash
python train_gear5.py --config-path configs/gear5/hybrid \\
  step=1 \\
  gear_checkpoint=configs/gear5/checkpoint_step40000_step2.pth \\
  load=configs/flashdepth/iter_43002.pth \\
  dataset.data_root=/path/to/data
```

#### Step 2: FFM 추가하여 전체 모델 학습

**목표**: Phase2 Step1 base에 Phase1 Step2의 FFM 추가

**Checkpoint Loading**:
1. Phase2 Step1 checkpoint에서 base 로드
2. Phase1 Step2 checkpoint에서 FFM만 추가 로드

**Trainable**:
- ✅ ForegroundOnlyModulationHead
- ✅ Mamba
- ✅ output_conv

**Frozen**:
- ❌ DINOv2 + DPT
- ❌ GlobalScalePredictorMultiLayer

**Config**:
```yaml
step: 2
gear_checkpoint: configs/gear5/hybrid/checkpoint_step40000_step1.pth  # Phase2 Step1
phase1_step2_checkpoint: configs/gear5/checkpoint_step40000_step2.pth  # Phase1 Step2 (for FFM)
training:
  fg_mod_lr: 1.0e-4
  mamba_lr: 5.0e-5
  output_lr: 5.0e-5
```

**Command**:
```bash
python train_gear5.py --config-path configs/gear5/hybrid \\
  step=2 \\
  gear_checkpoint=configs/gear5/hybrid/checkpoint_step40000_step1.pth \\
  phase1_step2_checkpoint=configs/gear5/checkpoint_step40000_step2.pth \\
  dataset.data_root=/path/to/data
```

---

## 손실 함수

### Log L1 Loss (Main Loss)

```python
# Inverse depth log L1 loss (같은 FlashDepth와 동일)
loss = torch.abs(torch.log(pred_inverse + 1e-3) - torch.log(gt_inverse + 1e-3))
loss = loss[valid_mask].mean()
```

**Valid Mask**:
- GT valid: Canonical 70m 이내 (inverse depth > 100/70)
- Pred outlier filtering: 200m 이내 (inverse depth > 100/200)
- Final mask: GT valid AND Pred not outlier

**Canonical Space**:
```python
# GT depth를 canonical space로 변환
depth_canonical = depth_actual × (CANONICAL_FX / fx_actual)
inverse_canonical = inverse_actual × (fx_actual / CANONICAL_FX)

# Training threshold
MIN_INVERSE_DEPTH = 100.0 / 70.0  # Canonical 70m
```

---

## 사용 방법

### Training

#### Phase 1

```bash
# Step 1: GSP 학습
python train_gear5.py --config-path configs/gear5 \\
  step=1 \\
  load=configs/flashdepth-l/iter_10001.pth \\
  dataset.data_root=/home/cvlab/hsy/Datasets \\
  training.wandb=true \\
  training.wandb_name=gear5_phase1_step1

# Step 2: FFM 학습
python train_gear5.py --config-path configs/gear5 \\
  step=2 \\
  gear_checkpoint=configs/gear5/checkpoint_step40000_step1.pth \\
  dataset.data_root=/home/cvlab/hsy/Datasets \\
  training.wandb=true \\
  training.wandb_name=gear5_phase1_step2
```

#### Phase 2

```bash
# Step 1: 2K에서 GSP 재학습 (FFM 제외)
python train_gear5.py --config-path configs/gear5/hybrid \\
  step=1 \\
  gear_checkpoint=configs/gear5/checkpoint_step40000_step2.pth \\
  load=configs/flashdepth/iter_43002.pth \\
  dataset.data_root=/home/cvlab/hsy/Datasets \\
  training.wandb=true \\
  training.wandb_name=gear5_phase2_step1

# Step 2: FFM 추가
python train_gear5.py --config-path configs/gear5/hybrid \\
  step=2 \\
  gear_checkpoint=configs/gear5/hybrid/checkpoint_step40000_step1.pth \\
  phase1_step2_checkpoint=configs/gear5/checkpoint_step40000_step2.pth \\
  dataset.data_root=/home/cvlab/hsy/Datasets \\
  training.wandb=true \\
  training.wandb_name=gear5_phase2_step2
```

---

## Dimension Flow

### Step 1: Global Scale Predictor

```
Input: Video [B=2, T=5, 3, 518, 518]
    ↓
Flatten to [B*T=10, 3, 518, 518]
    ↓
DINOv2 Encoder (frozen):
    intermediate_layer_idx = [4, 11, 17, 23]  # ViT-L
    encoder_features: List of [B*T=10, N=1370, embed_dim=1024]
        - Layer 4:  [10, 1370, 1024]
        - Layer 11: [10, 1370, 1024]
        - Layer 17: [10, 1370, 1024]
        - Layer 23: [10, 1370, 1024]

    CLS tokens extraction:
        cls_tokens_list: List of [B*T=10, 1024]
            - [10, 1024] × 4 layers
    ↓
GlobalScalePredictorMultiLayer:
    Input:  cls_tokens_list (4 tensors of [10, 1024])
    Fusion: torch.stack → [10, 4, 1024] → mean(dim=1) → [10, 1024]
    Output:
        scale: [10]  (positive)
        shift: [10]  (real)
    ↓
DPT Features (frozen):
    path_1: [10, 256, 37, 37]  (518/14 = 37)
    ↓
Global Modulation:
    scale: [10] → [10, 1, 1, 1]
    shift: [10] → [10, 1, 1, 1]
    path_1_global = path_1 * scale + shift  # [10, 256, 37, 37]
    ↓
Mamba Temporal Modeling:
    Input:  path_1_global [10, 256, 37, 37]
    Reshape: [B=2, T=5, 256, 37, 37]
    Mamba: Frame-by-frame processing
    Output: path_1_temporal [10, 256, 37, 37]
    ↓
DPT Output Head (trainable):
    output_conv1: [10, 256, 37, 37] → [10, 32, 37, 37]
    upsample: → [10, 32, 518, 518]
    output_conv2: → [10, 1, 518, 518]
    ↓
Metric Depth: [10, 1, 518, 518]
```

### Step 2: + Foreground-only Modulation

```
(... same as Step 1 until path_1_global ...)
    ↓
path_1_global: [10, 256, 37, 37]
    ↓
Multi-layer Attention Extraction:
    target_blocks = [11, 17]  # Middle 2 layers
    attention_weights:
        - Layer 11: [10, num_heads=16, 1370, 1370]
        - Layer 17: [10, 16, 1370, 1370]

    Fusion: torch.stack → [2, 10, 16, 1370, 1370] → mean(dim=0) → [10, 16, 1370, 1370]
    ↓
Importance Map:
    CLS→patch attention: [10, 16, 1370] → mean(heads) → [10, 1370]
    Reshape: [10, 1, 37, 37]
    Normalize: percentile(1-99) → [0, 1]
    ↓
FG Mask:
    Binary threshold (0.5): [10, 1, 37, 37] → {0, 1}
    ↓
Patch Tokens Extraction:
    patch_tokens: [10, 1370, 1024]  (from Layer 23)
    Remove CLS: [10, 1369, 1024]
    Apply FG mask → fg_patches: [10, N_fg, 1024]  (N_fg varies per sample)
    ↓
FG Feature Aggregation:
    fg_feature_mlp: [10, N_fg, 1024] → [10, N_fg, 256]
    Average pooling: → [10, 256]
    ↓
FG Modulation Prediction:
    fg_modulation_predictor: [10, 256] → [10, 512]
    Split: γ_fg [10, 256], β_fg [10, 256]
    ↓
Spatial Broadcast + Mask:
    γ_fg: [10, 256] → [10, 256, 1, 1] * fg_mask → [10, 256, 37, 37]
    β_fg: [10, 256] → [10, 256, 1, 1] * fg_mask → [10, 256, 37, 37]
    ↓
FG-only Modulation:
    path_1_fg = path_1_global + γ_fg * path_1_global + β_fg
    Result: [10, 256, 37, 37]
    ↓
(... same as Step 1: Mamba → output_conv → Metric Depth ...)
```

---

## Testing

### Test 구성

Gear5는 **4가지 테스트 스테이지** 지원:

1. **Phase1 Step1**: 518×518, ViT-L, GSP only
2. **Phase1 Step2**: 518×518, ViT-L, GSP + FFM
3. **Phase2 Step1**: 2K, ViT-S Hybrid, GSP only
4. **Phase2 Step2**: 2K, ViT-S Hybrid, GSP + FFM

### Testing Commands

```bash
# Phase1 Step1
python test_gear5.py --config-path configs/gear5 \\
  step=1 \\
  load=configs/gear5/checkpoint_step40000_step1.pth \\
  dataset.data_root=/home/cvlab/hsy/Datasets \\
  results_dir=test_results/phase1_step1

# Phase1 Step2
python test_gear5.py --config-path configs/gear5 \\
  step=2 \\
  load=configs/gear5/checkpoint_step40000_step2.pth \\
  dataset.data_root=/home/cvlab/hsy/Datasets \\
  results_dir=test_results/phase1_step2

# Phase2 Step1 (Hybrid)
python test_gear5.py --config-path configs/gear5/hybrid \\
  step=1 \\
  load=configs/gear5/hybrid/checkpoint_step40000_step1.pth \\
  dataset.data_root=/home/cvlab/hsy/Datasets \\
  results_dir=test_results/phase2_step1

# Phase2 Step2 (Hybrid)
python test_gear5.py --config-path configs/gear5/hybrid \\
  step=2 \\
  load=configs/gear5/hybrid/checkpoint_step40000_step2.pth \\
  dataset.data_root=/home/cvlab/hsy/Datasets \\
  results_dir=test_results/phase2_step2
```

### Sequence Normalization

**중요**: Gear5의 sequence.png 및 gif 시각화는 **GT의 vmin/vmax를 Pred에도 적용**하여 일관된 비교 제공

```python
# GT의 percentile 계산
gt_vmin = np.nanpercentile(gt_display, 2)
gt_vmax = np.nanpercentile(gt_display, 98)

# Pred도 GT의 range 사용 (NOT pred's own percentile!)
axes[1, col].imshow(pred_display, cmap=cmap_pred, vmin=gt_vmin, vmax=gt_vmax)
axes[2, col].imshow(gt_display, cmap=cmap_gt, vmin=gt_vmin, vmax=gt_vmax)
```

**장점**:
- ✅ Pred와 GT의 색상 범위가 동일 → 직접 비교 가능
- ✅ GT 기준 normalization → GT의 depth range에 맞춰 평가

---

## 기대 효과

### 1. 정확도 향상

**Multi-layer Fusion**:
- 4개 layer의 CLS tokens 융합 → 더 풍부한 global context
- 2개 layer의 attention weights 융합 → 더 robust한 FG detection

**Two-stage Learning**:
- Step 1: Global scale을 먼저 정확하게 학습 → metric depth의 전반적 정확도 ↑
- Step 2: FG-only refinement → 중요 영역(foreground)의 세밀한 개선

### 2. 효율성

**Foreground-only Modulation**:
- Background는 global만으로 충분
- Foreground에만 추가 modulation → 파라미터 효율 ↑

**파라미터 비교**:
- Gear3: ~9.2M (전체 feature modulation)
- Gear5 Step1: 4.86M (GSP + Mamba + output)
- Gear5 Step2: 7.5M (+ FFM, but FG-only)

### 3. 학습 안정성

**Uniform Weight Fusion**:
- Learnable weights 대신 uniform → 학습 안정성 ↑
- Overfitting 위험 ↓

**Residual Connection**:
- FG-only modulation이 residual로 추가 → gradient flow 개선

### 4. 고해상도 대응

**Phase 2 (2K Hybrid)**:
- ViT-S student + ViT-L teacher
- 고해상도에서 효율적 학습
- FlashDepth-hybrid의 이점 활용

---

## 참고사항

### ViT-S vs ViT-L Target Blocks

| Model | Step 1 (All) | Step 2 (Mid 2) |
|-------|-------------|----------------|
| **ViT-L** | [4, 11, 17, 23] | [11, 17] |
| **ViT-S** | [2, 5, 8, 11] | [5, 8] |

### Config Parameters

**Phase Detection**:
- config_dir에 'hybrid' 포함 → Phase 2
- 그 외 → Phase 1

**ViT Size**:
- Phase 1: config.model.vit_size 사용 (보통 'vitl')
- Phase 2: 강제로 'vits' (hybrid)

### Checkpoint 로딩 전략

**Phase1 Step1**:
- load: FlashDepth-L checkpoint (DINOv2 + DPT + Mamba + output)

**Phase1 Step2**:
- gear_checkpoint: Phase1 Step1 checkpoint (모든 파라미터)

**Phase2 Step1**:
- gear_checkpoint: Phase1 Step2 (GSP + Mamba + output만, FFM 제외)
- load: FlashDepth-hybrid (DINOv2 + DPT 덮어쓰기)

**Phase2 Step2**:
- gear_checkpoint: Phase2 Step1 (base model)
- phase1_step2_checkpoint: Phase1 Step2 (FFM만 추가)

---

## 문제 해결

### Training Issues

1. **OOM (Out of Memory)**:
   - Phase 2: gradient_checkpointing 활성화 필수
   - batch_size 감소 (2K: 3 → 2)
   - workers 감소 (8 → 4)

2. **Loss가 떨어지지 않음**:
   - Learning rate 확인 (Step 1: 1e-4, Step 2: 5e-5)
   - Valid mask 확인 (canonical 70m 이내)
   - Checkpoint 로딩 확인 (phase/step에 맞는지)

3. **Attention weights 없음**:
   - target_blocks가 올바르게 설정되었는지 확인
   - store_attn_weights = True 확인

### Testing Issues

1. **Phase/Step detection 오류**:
   - config_dir 확인 (hybrid 포함 여부)
   - config.step 확인 (1 or 2)

2. **Checkpoint 로딩 실패**:
   - strict=False로 로딩 (hybrid는 key 불일치 가능)
   - Missing/Unexpected keys 로그 확인

3. **Visualization 색상 이상**:
   - GT vmin/vmax 사용 확인
   - Valid mask 확인 (70m 이내)

---

**작성**: Claude (Anthropic)
**기반**: FlashDepth + Gear3
**목적**: Two-stage learning for improved metric depth estimation
