# Gear5 Bankai: Unified Mamba for Metric Depth Estimation

## 1. Overview

Gear5 Bankai is an advanced metric depth estimation system that unifies temporal processing and metric scale/shift prediction into a single Mamba2-based architecture.

### Key Features

1. **Unified Architecture**: Single Mamba2 replaces both F-Mamba (FlashDepth temporal) and T-Mamba (Gear5 scale predictor)
2. **Temporal Consistency**: Enhanced TAE (Temporal Alignment Error) through unified temporal processing
3. **Relative Depth Quality**: Preserves DAv2's DINOv2-DPT structure strength
4. **TGM Loss**: Temporal Gradient Matching loss from Video Depth Anything for improved temporal consistency
5. **Head-First Tuning**: Two-phase training strategy for stable optimization

---

## 2. Architecture

### 2.1 Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        DINOv2 (Frozen)                          │
│  ViT-L: CLS tokens [17, 23] → avg → Fused CLS-L [1, 1024]      │
│  ViT-S: CLS tokens [8, 11]  → avg → Fused CLS-S [1, 384]       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────┐
        │  [Hybrid only] CLS Fusion (FiLM/Bilinear)  │
        │  FiLM: γ(CLS-L) × CLS-S + β(CLS-L)         │
        └─────────────────────────────────────────────┘
                              ↓
              Fused CLS → Linear + ReLU → [1, C]

┌─────────────────────────────────────────────────────────────────┐
│                         DPT (Frozen)                            │
│  path_1 (148×148, C) → Downsample → Flatten → [h×w, C]         │
│                                                                 │
│  Downsample ratio:                                              │
│    - Large: ×0.1 (148→14, 196 tokens)                          │
│    - Small/Hybrid: ×0.05 (148→7, 49 tokens)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Concat → [h×w + 1, C]
                    (Spatial tokens + CLS at END)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Mamba (Trainable)                    │
│  - Per-frame processing + hidden state propagation              │
│  - Temporal consistency via hidden state                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
            ┌─────────────────┴─────────────────┐
            ↓                                   ↓
    Spatial [h×w, C]                     CLS [1, C]
            ↓                                   ↓
    Unflatten + Upsample                 MLP Head
    (Large: ×10, Small/Hybrid: ×20)          ↓
            ↓                           Scale (Softplus)
    + original_path_1                   Shift (Clamped)
            ↓
    output_conv (Head-First Tuning)
            ↓
      Relative Depth
            ↓
    Metric Depth = Scale × Relative + Shift
```

### 2.2 CLS Fusion Module (Hybrid Mode)

Hybrid 모드에서 Teacher(CLS-L)의 rich semantic information을 Student(CLS-S)에 전달하기 위한 fusion 모듈.

#### 왜 CrossAttention이 아닌가?

CLS 토큰끼리의 CrossAttention은 비효율적:
- N_k=1 (Key가 1개)일 때 `softmax([x]) = 1.0` (항상 고정)
- Query(CLS-S)가 무시되고 단순 Linear transformation으로 퇴화
- 파라미터 낭비 (~1.08M 중 W_q가 무의미)

```
CrossAttn with N_k=1:
    Attention_weights = softmax(Q @ K^T / √d) = [1.0]  # Query 무관
    Output = [1.0] @ V = V
    ≈ Linear(CLS-L)  # Query(CLS-S) 완전 무시
```

#### FiLM Fusion (Primary - 권장)

**Feature-wise Linear Modulation**: Teacher가 Student를 channel-wise로 modulate

```python
class FiLMFusion(nn.Module):
    """
    CLS-L이 CLS-S의 각 dimension을 개별적으로 조절.
    γ: scaling (feature 강조/억제)
    β: shifting (bias 조정)
    """
    def __init__(self, student_dim=384, teacher_dim=1024, hidden_dim=512):
        super().__init__()
        self.film_generator = nn.Sequential(
            nn.Linear(teacher_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, student_dim * 2)  # γ, β
        )
        # Identity initialization (γ=1, β=0에서 시작)
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        self.film_generator[-1].bias.data[:student_dim] = 1.0  # γ = 1

    def forward(self, cls_s, cls_l):
        # cls_s: [B, 384], cls_l: [B, 1024]
        film_params = self.film_generator(cls_l)     # [B, 768]
        gamma, beta = film_params.chunk(2, dim=-1)   # [B, 384] each

        fused = gamma * cls_s + beta  # Channel-wise affine transform
        return fused
```

**특징:**
- Teacher가 Student를 "가이드" (일방향 정보 흐름)
- Student identity 구조적 보존 (γx + β 형태)
- Identity mapping에서 시작 → 학습 안정성
- 파라미터: ~655K (1024×512 + 512×768)
- 검증된 방식: VQA, Visual Reasoning, Conditional Generation

#### Bilinear Fusion (Alternative - 선택 옵션)

**Low-rank Bilinear Pooling**: 두 feature 간 second-order interaction 캡처

```python
class BilinearFusion(nn.Module):
    """
    CLS-S와 CLS-L의 multiplicative interaction.
    cls_s[i] × cls_l[j] 형태의 cross-dimension 관계 학습.
    """
    def __init__(self, student_dim=384, teacher_dim=1024, out_dim=384, rank=64):
        super().__init__()
        self.U = nn.Linear(student_dim, rank, bias=False)   # Student → shared space
        self.V = nn.Linear(teacher_dim, rank, bias=False)   # Teacher → shared space
        self.P = nn.Linear(rank, out_dim)                   # Output projection

    def forward(self, cls_s, cls_l):
        # Project to shared low-rank space
        u = self.U(cls_s)  # [B, rank]
        v = self.V(cls_l)  # [B, rank]

        # Hadamard product = low-rank bilinear interaction
        z = u * v  # [B, rank]

        # Optional: signed sqrt + L2 norm
        z = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-8)
        z = F.normalize(z, p=2, dim=-1)

        fused = self.P(z)  # [B, out_dim]
        return fused
```

**특징:**
- 대칭적 정보 결합 (Student ↔ Teacher)
- Cross-dimension interaction 학습 가능
- 파라미터: ~450K (384×64 + 1024×64 + 64×384)
- Student identity가 섞여 사라질 수 있음
- 검증된 방식: VQA, Fine-grained Recognition

#### FiLM vs Bilinear 비교

| 측면 | FiLM | Bilinear |
|------|------|----------|
| **정보 흐름** | Teacher → Student (일방향) | 대칭적 융합 |
| **Student 보존** | ✅ 구조적 보존 (γx+β) | ⚠️ 섞임 |
| **Interaction** | Affine (1st order) | Multiplicative (2nd order) |
| **표현력** | 중간 | 높음 |
| **학습 안정성** | ✅ 높음 | ⚠️ 중간 |
| **파라미터** | ~655K | ~450K |
| **권장 상황** | Knowledge transfer | Information fusion |

#### 설정 방법

```yaml
# configs/gear5/config.yaml

# CLS Fusion 설정 (Hybrid mode에서만 사용)
cls_fusion_type: "film"    # "film" (default) or "bilinear"
cls_fusion_hidden: 512     # FiLM hidden dimension
cls_fusion_rank: 64        # Bilinear low-rank dimension
```

---

### 2.3 Key Components

#### UnifiedMamba

- Processes spatial tokens + CLS token through Mamba2
- CLS at END position to aggregate spatial information
- Hidden state propagation for temporal consistency
- Zero-init output projection for training stability

```python
# Key parameters (Large model)
- dpt_dim: 256
- cls_embed_dim: 1024
- num_mamba_layers: 4
- downsample_factor: 0.1
- Parameters: ~1.76M
```

#### BankaiMetricHead

- Wrapper for UnifiedMamba + Importance Map Generator
- Optional CLS Fusion (FiLM/Bilinear) for Hybrid mode
- Outputs: spatial_out, scale, shift, importance_map

### 2.4 Model Variants

| Variant | DPT Dim | CLS Dim | Downsample | Mamba Tokens |
|---------|---------|---------|------------|--------------|
| Large | 256 | 1024 | 0.1 | 196 + 1 |
| Small | 64 | 384 | 0.05 | 49 + 1 |
| Hybrid | 64 | 384 (fused) | 0.05 | 49 + 1 |

---

## 3. Training Strategy

### 3.1 Loss Function

```
L_total = L_depth + α × L_TGM

Where:
- L_depth: Log L1 Loss on inverse depth
- L_TGM: Temporal Gradient Matching Loss (Video Depth Anything)
- α: TGM weight (default: 0.3)
```

#### TGM Loss Details

- Multi-scale temporal gradients (stride=1, 2, 4, 8)
- Validity masking for stable regions
- Trimmed MAE for outlier robustness
- Exponential decay for longer temporal distances

### 3.2 Head-First Tuning (Two-Phase Training)

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| DINOv2 (encoder) | Frozen | Frozen |
| DPT (decoder) | Frozen | Frozen |
| UnifiedMamba | **Frozen** | **Trainable** |
| Metric Head (MLP) | **Trainable** | **Trainable** |
| output_conv | Frozen | **Trainable** |

**Phase 1**: ~5,000 steps
- Train only scale/shift MLPs
- UnifiedMamba uses FlashDepth pretrained weights
- Goal: Initialize scale/shift to reasonable range

**Phase 2**: ~35,000 steps
- Train UnifiedMamba + Metric Head + output_conv
- Learning rate: 0.1× of Phase 1
- Goal: Joint optimization for temporal consistency

### 3.3 Pretrained Weights

| Component | Source |
|-----------|--------|
| DINOv2 | FlashDepth pretrained |
| DPT | FlashDepth pretrained |
| UnifiedMamba | FlashDepth Mamba (F-Mamba) |
| Metric Head | Random init |
| output_conv | FlashDepth pretrained |

---

## 4. Usage

### 4.1 Training

```bash
# Phase 1: Train Metric Head only
torchrun --nproc_per_node=2 train_gear5.py \
    --config-path configs/gear5 \
    use_bankai=true \
    bankai_phase=1 \
    tgm_weight=0.3 \
    dataset.data_root=<path_to_data>

# Phase 2: Train UnifiedMamba + output_conv
torchrun --nproc_per_node=2 train_gear5.py \
    --config-path configs/gear5 \
    use_bankai=true \
    bankai_phase=2 \
    tgm_weight=0.3 \
    load=<phase1_checkpoint> \
    dataset.data_root=<path_to_data>
```

### 4.2 Testing

```bash
python test_gear5.py \
    --config-path configs/gear5 \
    use_bankai=true \
    load=<checkpoint_path> \
    results_dir=test_results/bankai
```

### 4.3 Configuration Options

```yaml
# configs/gear5/config_l.yaml (or config_s.yaml)

# Enable Bankai mode
use_bankai: true

# Training phase (1 or 2)
bankai_phase: 1

# TGM loss weight (0 to disable)
tgm_weight: 0.3

# Architecture settings
bankai_downsample: 0.1  # Large: 0.1, Small: 0.05
bankai_num_mamba_layers: 4

# CLS layers (3rd and 4th intermediate)
cls_layers: [3, 4]

# CLS Fusion settings (Hybrid mode only)
cls_fusion_type: "film"    # "film" (recommended) or "bilinear"
cls_fusion_hidden: 512     # FiLM: hidden dimension for γ,β generator
cls_fusion_rank: 64        # Bilinear: low-rank dimension
```

---

## 5. Evaluation

### 5.1 Metrics

| Category | Metric | Target |
|----------|--------|--------|
| Depth Quality | MAE, RMSE | Maintain or improve |
| Accuracy | δ1, δ2, δ3 | Maintain or improve |
| Temporal | **TAE** | **20%+ improvement** |
| Efficiency | FPS | <10% drop |

### 5.2 Datasets

- **Training**: TartanAir, MVS-Synth, DynamicReplica, PointOdyssey, Spring
- **Validation**: Sintel, Waymo_seg
- **Testing**: ETH3D, UrbanSyn, Unreal4K, Bonn

---

## 6. Implementation Files

### Modified Files

```
flashdepth/gear5_modules.py     # UnifiedMamba, BankaiMetricHead
utils/gear_losses.py            # TGMTemporalLoss, CombinedBankaiLoss
train_gear5.py                  # Bankai mode training
test_gear5.py                   # Bankai mode testing
configs/gear5/config_l.yaml     # Bankai config options
configs/gear5/config_s.yaml     # Bankai config options
```

### Key Classes

- `UnifiedMamba`: Unified temporal processing for spatial + CLS tokens
- `BankaiMetricHead`: Full metric head with UnifiedMamba
- `FiLMFusion`: CLS fusion via feature-wise linear modulation (primary)
- `BilinearFusion`: CLS fusion via low-rank bilinear pooling (alternative)
- `TGMTemporalLoss`: Temporal Gradient Matching loss
- `CombinedBankaiLoss`: Depth loss + TGM loss

---

## 7. References

- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) - TGM Loss
- [FlashDepth](https://github.com/xxx/FlashDepth) - Base architecture
- [Mamba2](https://github.com/state-spaces/mamba) - Temporal modeling

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-22 | v1.1 | CLS Fusion Module 추가 (FiLM primary, Bilinear alternative) |
| 2025-01-20 | v1.0 | Initial implementation |
