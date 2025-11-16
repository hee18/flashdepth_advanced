# GEAR5: Metric Depth Enhancement System

This document covers two distinct approaches for enhancing FlashDepth with metric depth capabilities:
- **Gear5**: GRU-based temporal scale/shift prediction (original)
- **Gear5 FiLM**: FiLM-style channel-wise feature modulation (variant)

Both share common components (CLS token extraction, importance map generation) but differ in their modulation strategies.

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Gear5 (Original): GRU-Based Temporal Modulation](#gear5-original-gru-based-temporal-modulation)
- [Gear5 FiLM: Channel-Wise Feature Modulation](#gear5-film-channel-wise-feature-modulation)
- [Comparison](#comparison)
- [Loss Functions](#loss-functions)
- [Training & Testing](#training--testing)
- [Configuration](#configuration)

---

## Architecture Overview

Both Gear5 variants extend FlashDepth with metric depth prediction capabilities by:
1. Extracting semantic features from multi-layer CLS tokens
2. Generating importance maps from attention weights for loss weighting
3. Applying learned transformations to enhance depth predictions

**Key Shared Components**:
- **Multi-layer CLS Token Extraction**: Layers [11, 23] for ViT-L, [5, 11] for ViT-S
- **ImportanceMapGenerator**: CLS-to-patch attention → importance map for loss weighting
- **Loss Functions**: Both support `log_l1` (standard) and `importance` (weighted) loss types
- **Canonical Space**: All training/inference uses canonical focal length (500.0 for 518×518, configurable via `canonical_focal_length`; on/off via `use_canonical_space`)

**Key Differences**:
| Component | Gear5 (Original) | Gear5 FiLM |
|-----------|------------------|------------|
| **Modulation Target** | Final relative depth map | DPT path_1 features (before Mamba) |
| **Modulation Method** | GRU-based temporal scale/shift | Channel-wise FiLM (gamma/beta) |
| **Temporal Modeling** | GRU inside head | Existing Mamba layers |
| **Trainable Params** | ~132K (head only) | ~1.03M (head + Mamba + output_conv2) |
| **Frozen Components** | Everything except Gear5Head | ViT + DPT + output_conv1 |

---

## Gear5 (Original): GRU-Based Temporal Modulation

### Architecture

Gear5 applies **temporal scale and shift** to the final relative depth output using a GRU-based predictor.

```
Video Input [B, T, 3, H, W]
    ↓
ViT Encoder (Frozen)
    ↓
CLS Tokens [Layers 11, 23] → ImportanceMapGenerator → Importance Map
    ↓                              ↓
TemporalScalePredictor          Loss Weighting
    ↓
Scale [B, T, 1, 1, 1], Shift [B, T, 1, 1, 1]
    ↓
Relative Depth × Scale + Shift = Metric Depth
```

### Key Components

#### 1. ImportanceMapGenerator
Generates spatial importance weights from CLS attention for loss weighting.

```python
class ImportanceMapGenerator(nn.Module):
    """
    Extract CLS-to-patch attention from multiple layers (11, 23)
    Average across layers → Reshape → Remove register token → Normalize [0,1]

    Input: List of [B*T, num_heads, N+1, N+1] attention weights
    Output: [B, T, patch_h, patch_w] importance map
    """
```

**Pipeline**:
1. Extract CLS-to-patch attention from each layer: `attn[:, :, 0, 1:]`
2. Average over heads: `[B, num_heads, N] → [B, N]`
3. Average across layers (11, 23)
4. Reshape to spatial: `[B, patch_h, patch_w]`
5. Remove register token (highest attention patch) via 3×3 inpainting
6. Percentile normalization (1-99 percentile → [0, 1])

#### 2. TemporalScalePredictor
Predicts scale and shift parameters using a GRU for temporal consistency.

```python
class TemporalScalePredictor(nn.Module):
    """
    CLS token [B, T, 1024] → Linear → GRU → Linear → Scale/Shift

    Architecture:
        - Input projection: 1024 → 256
        - Bi-directional GRU: hidden_dim=256, 2 layers
        - Output projection: 512 (bidirectional) → 2 (scale, shift)

    Trainable params: ~132K (only this module is trained)
    """
```

**Forward Flow**:
```python
cls_token: [B, T, 1024]  # Last layer CLS token (Layer 23)
    ↓ Linear(1024 → 256)
cls_features: [B, T, 256]
    ↓ Bi-GRU(2 layers, 256 hidden)
gru_output: [B, T, 512]  # 256 × 2 (bidirectional)
    ↓ Linear(512 → 2)
scale_shift: [B, T, 2]
    ↓ Split
scale: [B, T, 1, 1, 1] = exp(scale_raw)  # Ensures positive
shift: [B, T, 1, 1, 1] = shift_raw        # Real number
```

**Temporal Processing**: GRU processes sequence bidirectionally, ensuring each frame's scale/shift considers past and future context.

#### 3. Gear5Head (Main Module)

```python
class Gear5Head(nn.Module):
    def __init__(self, embed_dim=1024):
        self.temporal_scale_predictor = TemporalScalePredictor(embed_dim)
        self.importance_map_generator = ImportanceMapGenerator(num_layers=2)

    def forward(self, cls_tokens_multi_layer, attention_weights_list,
                relative_depth, patch_h, patch_w):
        """
        Args:
            cls_tokens_multi_layer: List of [B, T, 1024] from [Layer 11, 23]
            attention_weights_list: List of [B*T, heads, N+1, N+1] from [Layer 11, 23]
            relative_depth: [B, T, 1, H, W] normalized depth
            patch_h, patch_w: Spatial patch dimensions (e.g., 37×37 for 518×518)

        Returns:
            dict with:
                - metric_depth: [B, T, 1, H, W]
                - scale: [B, T, 1, 1, 1]
                - shift: [B, T, 1, 1, 1]
                - importance_map: [B, T, patch_h, patch_w]
        """
        # Use last layer CLS token for scale/shift prediction
        cls_token = cls_tokens_multi_layer[-1]  # [B, T, 1024]

        # Predict scale and shift
        scale, shift = self.temporal_scale_predictor(cls_token)

        # Apply to relative depth
        metric_depth = scale * relative_depth + shift

        # Generate importance map for loss
        importance_map = self.importance_map_generator(
            attention_weights_list, patch_h, patch_w
        )

        return {
            'metric_depth': metric_depth,
            'scale': scale,
            'shift': shift,
            'importance_map': importance_map
        }
```

### Dimension Flow

```
Input Video:        [B=2, T=5, C=3, H=518, W=518]
                             ↓
ViT Encoder (Frozen):
  - Patch embedding:     [B*T=10, N=1369, embed=1024]
  - Layer 11 output:     [B*T=10, N+1=1370, 1024]  (with CLS)
  - Layer 23 output:     [B*T=10, N+1=1370, 1024]  (with CLS)

CLS Token Extraction:
  - Layer 11 CLS:        [B*T=10, 1024] → Reshape → [B=2, T=5, 1024]
  - Layer 23 CLS:        [B*T=10, 1024] → Reshape → [B=2, T=5, 1024]
  - cls_tokens_multi_layer = [[B, T, 1024], [B, T, 1024]]

Attention Weights:
  - Layer 11 attn:       [B*T=10, heads=16, N+1=1370, N+1=1370]
  - Layer 23 attn:       [B*T=10, heads=16, N+1=1370, N+1=1370]

TemporalScalePredictor:
  cls_token (Layer 23):  [B=2, T=5, 1024]
       ↓ Linear(1024→256)
  cls_features:          [B=2, T=5, 256]
       ↓ Bi-GRU(2 layers, hidden=256)
  gru_output:            [B=2, T=5, 512]  # 256×2 bidirectional
       ↓ Linear(512→2)
  scale_shift:           [B=2, T=5, 2]
       ↓ Split & Reshape
  scale:                 [B=2, T=5, 1, 1, 1]
  shift:                 [B=2, T=5, 1, 1, 1]

ImportanceMapGenerator:
  attention_weights_list: [[B*T=10, 16, 1370, 1370], [B*T=10, 16, 1370, 1370]]
       ↓ Extract CLS-to-patch attn[:, :, 0, 1:]
  cls_to_patch:          [B*T=10, 16, N=1369]
       ↓ Mean over heads
  cls_attention:         [B*T=10, N=1369]
       ↓ Average layers 11, 23
  cls_avg:               [B*T=10, N=1369]
       ↓ Reshape to spatial
  importance_map:        [B*T=10, 1, patch_h=37, patch_w=37]
       ↓ Remove register token + normalize
       ↓ Reshape temporal
  importance_map:        [B=2, T=5, patch_h=37, patch_w=37]

Metric Depth:
  relative_depth:        [B=2, T=5, 1, H=518, W=518]
  metric_depth:          scale × relative_depth + shift
                         [B=2, T=5, 1, 518, 518]
```

### Parameter Count

**Frozen Components** (~340M params):
- ViT Encoder: ~304M
- DPT: ~35M
- Mamba: ~0.9M
- output_conv: ~0.6K

**Trainable Components** (~132K params):
- TemporalScalePredictor:
  - Input Linear: 1024×256 = 262K weights + 256 bias = 262,400
  - Bi-GRU (2 layers): ~66K
  - Output Linear: 512×2 = 1,024 + 2 bias = 1,026
- ImportanceMapGenerator: 0 (no trainable params)
- **Total trainable**: ~132K

### Training Strategy

**Freezing**:
```python
# Freeze entire FlashDepth model
for param in model.pretrained.parameters():
    param.requires_grad = False
for param in model.depth_head.parameters():
    param.requires_grad = False
if model.use_mamba:
    for param in model.mamba_temporal_modules.parameters():
        param.requires_grad = False

# Only train Gear5Head
for param in model.gear5_head.parameters():
    param.requires_grad = True
```

**Learning Rates**:
- `gear5_lr`: 1.0e-4 (TemporalScalePredictor)
- Weight decay: 1.0e-6

**Loss**: Log L1 in inverse depth space (100/m) with optional importance weighting.

---

## Gear5 FiLM: Channel-Wise Feature Modulation

### Architecture

Gear5 FiLM applies **channel-wise FiLM modulation** to DPT features before Mamba temporal modeling.

```
Video Input [B, T, 3, H, W]
    ↓
ViT Encoder (Frozen)
    ↓
CLS Tokens [Layers 11, 23] → GlobalFeatureNetwork → ModulationNetwork
    ↓                              ↓                      ↓
DPT (Frozen)                  Global Feature         Gamma, Beta
    ↓                              ↓                      ↓
path_1 features          SimpleFeatureModulator (Channel-wise)
    ↓
Modulated path_1 → Mamba (Trainable) → output_conv2 (Trainable) → Metric Depth
    ↓
ImportanceMapGenerator → Importance Map (Loss Weighting)
```

### Key Components

#### 1. ImportanceMapGenerator
Identical to Gear5 - generates spatial importance weights from CLS attention.

#### 2. GlobalFeatureNetwork
Extracts global semantic features from CLS token.

```python
class GlobalFeatureNetwork(nn.Module):
    """
    CLS token → Global semantic feature

    Architecture:
        - Linear(1024 → 512) → ReLU
        - Linear(512 → 256) → ReLU

    Processes each frame independently, handles temporal dimension.
    """

    def forward(self, cls_token):
        """
        Args:
            cls_token: [B, T, 1024] from Layer 23

        Returns:
            global_feature: [B, T, 256]
        """
```

#### 3. ModulationNetwork
Generates channel-wise gamma and beta for FiLM modulation.

```python
class ModulationNetwork(nn.Module):
    """
    Global feature → Gamma, Beta (channel-wise)

    Architecture:
        - Linear(256 → 512) → ReLU
        - Linear(512 → 512)  # First 256: gamma, Last 256: beta

    Each channel gets its own gamma/beta, applied uniformly across spatial locations.
    """

    def forward(self, global_feature):
        """
        Args:
            global_feature: [B, T, 256]

        Returns:
            gamma: [B, T, 256]  # Channel-wise scaling
            beta: [B, T, 256]   # Channel-wise shift
        """
```

**Key Concept - Channel-wise Modulation**:
- 256 DPT channels = 256 information types
- Each channel gets its own gamma/beta pair
- All spatial locations (H×W) within a channel share the same gamma/beta
- Formula: `modulated[b,t,c,x,y] = gamma[b,t,c] ⊙ feature[b,t,c,x,y] + beta[b,t,c]`

#### 4. SimpleFeatureModulator
Applies channel-wise FiLM modulation to DPT features.

```python
class SimpleFeatureModulator(nn.Module):
    """
    Apply FiLM-style modulation: feature * gamma + beta (channel-wise)
    """

    def forward(self, features, gamma, beta):
        """
        Args:
            features: [B*T, C=256, H, W] DPT path_1 features
            gamma: [B, T, 256] channel-wise scaling
            beta: [B, T, 256] channel-wise shift

        Returns:
            modulated_features: [B*T, 256, H, W]
        """
        # Reshape gamma/beta to [B*T, C, 1, 1]
        gamma_expanded = gamma.view(B*T, C, 1, 1)
        beta_expanded = beta.view(B*T, C, 1, 1)

        # Apply channel-wise modulation
        return gamma_expanded * features + beta_expanded
```

#### 5. Gear5FilmHead (Main Module)

```python
class Gear5FilmHead(nn.Module):
    def __init__(self, embed_dim=1024, dpt_dim=256):
        self.global_feature_net = GlobalFeatureNetwork(embed_dim, feature_dim=256)
        self.modulation_net = ModulationNetwork(feature_dim=256, dpt_dim=dpt_dim)
        self.feature_modulator = SimpleFeatureModulator()
        self.importance_map_generator = ImportanceMapGenerator(num_layers=2)

    def forward(self, cls_tokens_multi_layer, attention_weights_list,
                dpt_features, patch_h, patch_w):
        """
        Args:
            cls_tokens_multi_layer: List of [B, T, 1024] from [Layer 11, 23]
            attention_weights_list: List of [B*T, heads, N+1, N+1] from [Layer 11, 23]
            dpt_features: List of 4× [B*T, 256, H, W] DPT layer features
            patch_h, patch_w: Spatial patch dimensions

        Returns:
            dict with:
                - path_1_modulated: [B*T, 256, H, W]
                - gamma: [B, T, 256]
                - beta: [B, T, 256]
                - importance_map: [B, T, patch_h, patch_w]
        """
        # Use last layer CLS token (Layer 23)
        cls_token = cls_tokens_multi_layer[-1]  # [B, T, 1024]

        # Generate global semantic feature
        global_feature = self.global_feature_net(cls_token)  # [B, T, 256]

        # Get modulation parameters
        gamma, beta = self.modulation_net(global_feature)  # [B, T, 256] each

        # Modulate ONLY path_1 (last DPT layer features)
        path_1 = dpt_features[-1]  # [B*T, 256, H, W]
        path_1_modulated = self.feature_modulator(path_1, gamma, beta)

        # Generate importance map
        importance_map = self.importance_map_generator(
            attention_weights_list, patch_h, patch_w
        )  # [B, T, patch_h, patch_w]

        return {
            'path_1_modulated': path_1_modulated,
            'gamma': gamma,
            'beta': beta,
            'importance_map': importance_map
        }
```

### Dimension Flow

```
Input Video:        [B=2, T=5, C=3, H=518, W=518]
                             ↓
ViT Encoder (Frozen):
  - Layer 11 CLS:        [B*T=10, 1024] → Reshape → [B=2, T=5, 1024]
  - Layer 23 CLS:        [B*T=10, 1024] → Reshape → [B=2, T=5, 1024]

DPT (Frozen):
  - path_1 (Layer 23):   [B*T=10, C=256, H=65, W=65]
  - path_2 (Layer 11):   [B*T=10, C=512, H=65, W=65]
  - path_3 (Layer 5):    [B*T=10, C=256, H=65, W=65]
  - path_4 (Layer 2):    [B*T=10, C=256, H=65, W=65]

GlobalFeatureNetwork:
  cls_token (Layer 23):  [B=2, T=5, 1024]
       ↓ Reshape to [B*T=10, 1024]
       ↓ Linear(1024→512) → ReLU
       ↓ Linear(512→256) → ReLU
  global_feature:        [B*T=10, 256]
       ↓ Reshape to [B=2, T=5, 256]

ModulationNetwork:
  global_feature:        [B=2, T=5, 256]
       ↓ Reshape to [B*T=10, 256]
       ↓ Linear(256→512) → ReLU
       ↓ Linear(512→512)
  params:                [B*T=10, 512]
       ↓ Split into [B*T=10, 256] each
  gamma:                 [B*T=10, 256] → Reshape → [B=2, T=5, 256]
  beta:                  [B*T=10, 256] → Reshape → [B=2, T=5, 256]

SimpleFeatureModulator:
  path_1:                [B*T=10, C=256, H=65, W=65]
  gamma:                 [B=2, T=5, 256] → Reshape → [B*T=10, 256, 1, 1]
  beta:                  [B=2, T=5, 256] → Reshape → [B*T=10, 256, 1, 1]
       ↓ Broadcast to [B*T=10, 256, 65, 65]
  path_1_modulated:      gamma * path_1 + beta
                         [B*T=10, 256, 65, 65]

Mamba Temporal Processing (Trainable):
  path_1_modulated:      [B*T=10, 256, 65, 65]
       ↓ Reshape to [B=2, T=5, 256, 65, 65]
       ↓ Mamba layers (4 layers, d_state=256)
  temporal_features:     [B=2, T=5, 256, 65, 65]
       ↓ Flatten to [B*T=10, 256, 65, 65]

DPT Refinement Head (Frozen output_conv1 + Trainable output_conv2):
  temporal_features:     [B*T=10, 256, 65, 65]
       ↓ Upsample + fusion
  depth_pred:            [B*T=10, 1, 518, 518]
       ↓ Reshape temporal
  metric_depth:          [B=2, T=5, 1, 518, 518]

ImportanceMapGenerator:
  attention_weights:     [[B*T=10, 16, 1370, 1370], [B*T=10, 16, 1370, 1370]]
       ↓ CLS attention extraction + averaging
  importance_map:        [B=2, T=5, patch_h=37, patch_w=37]
```

### Parameter Count

**Frozen Components** (~339M params):
- ViT Encoder: ~304M
- DPT: ~35M
- output_conv1: ~0.3K

**Trainable Components** (~1.03M params):
- Gear5FilmHead:
  - GlobalFeatureNetwork: (1024×512 + 512×256) + bias ≈ 656K
  - ModulationNetwork: (256×512 + 512×512) + bias ≈ 394K
  - SimpleFeatureModulator: 0 (no params)
  - ImportanceMapGenerator: 0 (no params)
  - **Subtotal**: ~132K (per actual module implementation)
- Mamba modules: ~0.9M
- output_conv2: ~0.6K
- **Total trainable**: ~1.03M

### Training Strategy

**Freezing**:
```python
# Freeze ViT encoder
for param in model.pretrained.parameters():
    param.requires_grad = False

# Freeze DPT (except output_conv2)
for name, param in model.depth_head.named_parameters():
    if 'output_conv2' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# Train Mamba
if model.use_mamba:
    for param in model.mamba_temporal_modules.parameters():
        param.requires_grad = True

# Train Gear5FilmHead
for param in model.gear5_film_head.parameters():
    param.requires_grad = True
```

**Learning Rates**:
- `film_lr`: 1.0e-4 (Gear5FilmHead modules)
- `mamba_lr`: 1.0e-5 (Mamba temporal modules)
- `output_lr`: 1.0e-5 (output_conv2)
- Weight decay: 1.0e-6

**Loss**: Log L1 in inverse depth space (100/m) with optional importance weighting.

---

## Comparison

### Side-by-Side Comparison

| Feature | Gear5 (Original) | Gear5 FiLM |
|---------|------------------|------------|
| **Modulation Target** | Final relative depth map | DPT path_1 features |
| **Modulation Timing** | After all processing | Before Mamba temporal modeling |
| **Modulation Type** | Scalar scale/shift per frame | Channel-wise gamma/beta (256 channels) |
| **Temporal Modeling** | GRU inside head (bidirectional) | Existing Mamba layers (4 layers) |
| **CLS Token Usage** | Layer 23 only (for scale/shift) | Layer 23 only (for FiLM params) |
| **Attention Usage** | Layers [11, 23] for importance map | Layers [11, 23] for importance map |
| **Trainable Params** | ~132K (TemporalScalePredictor) | ~1.03M (head + Mamba + conv2) |
| **Frozen Components** | Everything except Gear5Head | ViT + DPT + output_conv1 |
| **Training Speed** | Faster (less trainable params) | Slower (more trainable params) |
| **Modulation Granularity** | Coarse (1 scale + 1 shift per frame) | Fine (256 gamma + 256 beta per frame) |
| **Loss Functions** | log_l1, importance | log_l1, importance |

### When to Use Which?

**Use Gear5 (Original)** when:
- You want minimal trainable parameters (~132K)
- You need faster training and inference
- You want explicit temporal consistency via GRU
- Simple global scale/shift is sufficient

**Use Gear5 FiLM** when:
- You want fine-grained channel-wise modulation
- You can afford more trainable parameters (~1.03M)
- You want to leverage and refine Mamba's temporal modeling
- You want modulation integrated early in the pipeline

---

## Loss Functions

Both Gear5 variants support two loss types: `log_l1` (default) and `importance` (weighted).

### Log L1 Loss (Standard)

```python
def log_l1_loss(pred, target, valid_mask):
    """
    Log L1 loss in inverse depth space (100/m)

    Formula: L = |log(100/pred) - log(100/target)|
           = |log(target) - log(pred)|  (after simplification)

    Args:
        pred: [B*T, 1, H, W] predicted metric depth (meters)
        target: [B*T, 1, H, W] ground truth metric depth (meters)
        valid_mask: [B*T, 1, H, W] boolean mask (True = valid pixel)

    Returns:
        Scalar loss (mean over valid pixels)
    """
    pred_inv = 100.0 / (pred + 1e-8)
    target_inv = 100.0 / (target + 1e-8)

    loss = torch.abs(torch.log(pred_inv + 1e-8) - torch.log(target_inv + 1e-8))

    return loss[valid_mask].mean()
```

**Usage**:
```bash
python train_gear5.py --config-path configs/gear5 +loss_type=log_l1
```

### Importance-Weighted Loss

```python
def importance_weighted_loss(pred, target, valid_mask, importance_map):
    """
    Importance-weighted Log L1 loss

    Formula: L_weighted = L × (1 + fg_ratio × importance)

    Where:
        - L: Standard Log L1 loss per pixel
        - importance: Spatial importance map from CLS attention [0, 1]
        - fg_ratio: Fraction of high-attention pixels (importance > mean)
        - Higher weights on semantically important regions

    Args:
        pred: [B*T, 1, H, W] predicted metric depth
        target: [B*T, 1, H, W] ground truth metric depth
        valid_mask: [B*T, 1, H, W] boolean mask
        importance_map: [B, T, patch_h, patch_w] attention-based importance

    Returns:
        Scalar loss (mean over valid pixels)
    """
    # Compute base loss
    pred_inv = 100.0 / (pred + 1e-8)
    target_inv = 100.0 / (target + 1e-8)
    loss = torch.abs(torch.log(pred_inv + 1e-8) - torch.log(target_inv + 1e-8))

    # Resize importance map to match depth resolution
    B, T, patch_h, patch_w = importance_map.shape
    importance_resized = F.interpolate(
        importance_map.view(B * T, 1, patch_h, patch_w),
        size=(H, W), mode='bilinear', align_corners=True
    )  # [B*T, 1, H, W]

    importance_flat = importance_resized.flatten()

    # Compute foreground ratio (pixels with importance > mean)
    importance_threshold = importance_flat.mean()
    fg_mask = (importance_flat > importance_threshold)
    fg_ratio = fg_mask.float().mean()

    # Apply importance weighting
    weighted_loss = loss * (1.0 + fg_ratio * importance_flat.float())

    return weighted_loss[valid_mask].mean()
```

**Usage**:
```bash
python train_gear5.py --config-path configs/gear5 +loss_type=importance
```

**Effect**: Higher loss weights on regions with high CLS attention (semantically important areas like objects), lower weights on background/less important regions.

---

## Training & Testing

### Training

#### Gear5 (Original)

**Single GPU**:
```bash
CUDA_VISIBLE_DEVICES=1 python train_gear5.py \
  --config-path configs/gear5 \
  training.iterations=40001 \
  dataset.data_root=/path/to/datasets \
  +loss_type=log_l1
```

**Multi-GPU (DDP)**:
```bash
./train_gear5_ddp.sh \
  --config-path configs/gear5 \
  training.iterations=40001 \
  dataset.data_root=/path/to/datasets \
  +loss_type=importance
```

**Key Parameters**:
- `load`: Path to FlashDepth-L checkpoint (configs/flashdepth-l/iter_10001.pth)
- `training.batch_size`: 20 per GPU (effective 40 with 2 GPUs in DDP)
- `training.gear5_lr`: 1.0e-4 (TemporalScalePredictor learning rate)
- `training.iterations`: 40001 (with save_freq=5000, val_freq=1000)
- `loss_type`: 'log_l1' or 'importance'

#### Gear5 FiLM

**Single GPU**:
```bash
CUDA_VISIBLE_DEVICES=1 python train_gear5_film.py \
  --config-path configs/gear5_film \
  training.iterations=40001 \
  dataset.data_root=/path/to/datasets \
  +loss_type=log_l1
```

**Multi-GPU (DDP)**:
```bash
./train_gear5_film_ddp.sh \
  --config-path configs/gear5_film \
  training.iterations=40001 \
  dataset.data_root=/path/to/datasets \
  +loss_type=importance
```

**Key Parameters**:
- `load`: Path to FlashDepth-L checkpoint (configs/flashdepth-l/iter_10001.pth)
- `training.batch_size`: 20 per GPU (effective 40 with 2 GPUs in DDP)
- `training.film_lr`: 1.0e-4 (Gear5FilmHead learning rate)
- `training.mamba_lr`: 1.0e-5 (Mamba temporal modules learning rate)
- `training.output_lr`: 1.0e-5 (output_conv2 learning rate)
- `training.iterations`: 40001
- `loss_type`: 'log_l1' or 'importance'

### Testing

#### Gear5 (Original)

```bash
CUDA_VISIBLE_DEVICES=1 python test_gear5.py \
  --config-path configs/gear5 \
  --checkpoint train_results/results_X/gear_5/phase_1/checkpoint_step40000.pth \
  --results-dir test_results/gear5_results \
  --gpu 1
```

#### Gear5 FiLM

```bash
CUDA_VISIBLE_DEVICES=1 python test_gear5_film.py \
  --config-path configs/gear5_film \
  --checkpoint train_results/results_X/gear_5_film/phase_1/checkpoint_step40000.pth \
  --results-dir test_results/gear5_film_results \
  --gpu 1
```

**Output**:
- Depth predictions (npy/png)
- Visualization grids (input, prediction, GT)
- Metrics JSON (per-sequence and averaged)
- Gamma/beta visualizations (FiLM only)
- Importance maps (if enabled)

### Docker Commands

**Build**:
```bash
./run_docker.sh build
```

**Training (Gear5)**:
```bash
./run_docker.sh train_gear5 \
  --loss importance \
  --batch-size 20 \
  --workers 8 \
  --gpu 1
```

**Training (Gear5 FiLM)**:
```bash
./run_docker.sh train_gear5_film \
  --loss log_l1 \
  --batch-size 20 \
  --workers 8 \
  --gpu 1
```

**Testing**:
```bash
./run_docker.sh test_gear5 --gpu 1
./run_docker.sh test_gear5_film --gpu 1
```

---

## Configuration

### Gear5 Config Example (`configs/gear5/config.yaml`)

```yaml
# General settings
config_dir: null
inference: false
load: configs/flashdepth-l/iter_10001.pth  # FlashDepth-L pretrained weights

# Canonical space settings
canonical_focal_length: 500.0  # Fixed canonical focal length for 518×518 resolution (configurable)
use_canonical_space: true  # Enable/disable canonicalization (on/off toggle)

# Loss function selection
# Options: 'log_l1' (default), 'importance' (importance-weighted)
loss_type: "log_l1"

# Dataset configuration
dataset:
  data_root: null
  resolution: 'base'  # 518x518
  video_length: 5
  train_datasets: [mvs-synth, dynamicreplica, tartanair, pointodyssey, spring]
  val_datasets: [sintel, waymo_seg]

# Training configuration
training:
  batch_size: 20  # Per GPU (effective 40 with 2 GPUs in DDP)
  workers: 8
  iterations: 40001
  save_freq: 5000
  val_freq: 1000
  log_freq: 100
  wandb: false
  wandb_name: "gear5_phase1"

  # Learning rates
  gear5_lr: 1.0e-4     # TemporalScalePredictor learning rate
  weight_decay: 1.0e-6

# Model configuration
model:
  vit_size: "vitl"
  patch_size: 14
  attn_class: "MemEffAttention"

  # Gear5 attention layer selection (2-layer CLS tokens)
  target_blocks: [11, 23]  # ViT-L: layers 11, 23
  target_blocks_s: [5, 11]  # ViT-S: layers 5, 11 (for Phase 2 hybrid)

  # Mamba configuration (frozen in Gear5)
  use_mamba: true
  mamba_type: "add"
  num_mamba_layers: 4
  downsample_mamba: [0.1]
  mamba_pos_embed: null
  mamba_in_dpt_layer: [1]
  mamba_d_conv: 4
  mamba_d_state: 256

# Evaluation configuration
eval:
  compile: false
  metrics: true
  save_grid: true
  outfolder: "test_gear5"
  test_datasets: [sintel]
  save_vis_map: true  # Save importance map visualizations
```

### Gear5 FiLM Config Example (`configs/gear5_film/config.yaml`)

```yaml
# (Similar structure to Gear5 config)

# Loss function selection (Gear5 FiLM style)
loss_type: "log_l1"

# Training configuration
training:
  batch_size: 20
  workers: 8
  iterations: 40001
  save_freq: 5000
  val_freq: 1000

  # Learning rates
  film_lr: 1.0e-4     # FiLM modules learning rate
  mamba_lr: 1.0e-5    # Mamba learning rate (lower than FiLM)
  output_lr: 1.0e-5   # output_conv2 learning rate
  weight_decay: 1.0e-6

# Model configuration
model:
  # (Same as Gear5)
  use_mamba: true  # Trainable in Gear5 FiLM
  # ... (Mamba configuration)
```

---

## Summary

**Gear5** provides lightweight, GRU-based temporal scale/shift prediction for metric depth, ideal for scenarios requiring minimal trainable parameters (~132K) and explicit temporal consistency.

**Gear5 FiLM** offers fine-grained channel-wise feature modulation integrated early in the pipeline, suitable for scenarios where you can afford more trainable parameters (~1.03M) and want to leverage Mamba's temporal modeling capabilities.

Both variants:
- Support canonical space normalization (focal length = 500.0 for 518×518, configurable)
- Generate importance maps for loss weighting
- Support `log_l1` and `importance` loss types
- Use multi-layer CLS tokens from DINOv2 [Layers 11, 23]
- Trained on video datasets with metric depth ground truth

Choose the variant that best fits your computational budget, modulation granularity needs, and training objectives.
