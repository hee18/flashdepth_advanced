# FlashDepth Gear3 vs 원본 FlashDepth 차이점 총정리

**작성일**: 2025-10-07
**비교 대상**:
- Gear3: `/home/cvlab/hsy/flashdepth_advanced/flashdepth_claude/`
- 원본: `/home/cvlab/hsy/flashdepth_advanced/FlashDepth/`

---

## 목차

1. [핵심 목적의 차이](#1-핵심-목적의-차이)
2. [학습률 (Learning Rate) 설정](#2-학습률-learning-rate-설정)
3. [Loss 함수](#3-loss-함수)
4. [모델 아키텍처](#4-모델-아키텍처)
5. [학습 파라미터 설정](#5-학습-파라미터-설정)
6. [Optimizer 및 Scheduler](#6-optimizer-및-scheduler)
7. [데이터 로딩 및 전처리](#7-데이터-로딩-및-전처리)
8. [Forward Pass 흐름](#8-forward-pass-흐름)
9. [Validation 및 Visualization](#9-validation-및-visualization)
10. [체크포인트 로딩 전략](#10-체크포인트-로딩-전략)
11. [Distributed Training 설정](#11-distributed-training-설정)
12. [기타 세부사항](#12-기타-세부사항)

---

## 1. 핵심 목적의 차이

### 원본 FlashDepth
- **목적**: Relative depth estimation (상대적 깊이 추정)
- **출력**: Normalized depth (0~1 범위)
- **특징**: Scale and Shift invariant loss 사용

### Gear3
- **목적**: **Metric depth estimation** (절대적 깊이 추정, 단위: meters)
- **출력**: Metric depth (미터 단위)
- **특징**: Feature-level FiLM modulation을 통한 metric depth 학습

---

## 2. 학습률 (Learning Rate) 설정

### 원본 FlashDepth (`configs/flashdepth-l/config.yaml`)

```yaml
training:
  lr:
    vit: 5.0e-6      # DINOv2 ViT encoder
    dpt: 5.0e-5      # DPT head
    head: 5.0e-5     # Output head (사용 안 함)
    mamba: 1.0e-4    # Mamba temporal module
    fusion: 1.0e-4   # Hybrid fusion (optional)
    warmup_steps: 1000
```

**특징**:
- 5개의 분리된 learning rate groups
- ViT는 매우 낮은 LR (fine-tuning)
- DPT는 중간 LR
- Warmup만 있고 decay 없음 (단순 warmup scheduler)

### Gear3 (`configs/gear3/config.yaml`)

```yaml
training:
  gear3_lr: 1.0e-4     # Gear3 modules (새로 추가된 모듈)
  mamba_lr: 1.0e-4     # Mamba (from scratch)
  weight_decay: 1.0e-6
```

**특징**:
- **2개의 learning rate groups만 존재**
- Gear3와 Mamba 모두 **같은 LR** (1e-4) 사용
- **ViT, DPT는 frozen** (학습 안 함, LR 없음)
- Cosine annealing with warmup 사용 (원본과 다름)

**주요 차이점**:
| 모듈 | 원본 FlashDepth | Gear3 |
|------|----------------|-------|
| ViT (DINOv2) | 5e-6 (fine-tune) | **Frozen** |
| DPT | 5e-5 (fine-tune) | **Frozen** |
| Mamba | 1e-4 | 1e-4 (from scratch) |
| Output conv | - | 1e-4 (Gear3 LR, from scratch) |
| Gear3 modules | - | **1e-4 (new)** |

---

## 3. Loss 함수

### 원본 FlashDepth

**Loss 종류**: `ScaleAndShiftInvariantLoss` (SSI Loss)

```python
# flashdepth/util/loss.py
class ScaleAndShiftInvariantLoss(nn.Module):
    def forward(self, prediction, target, mask=None):
        # 1. Compute optimal scale and shift
        scale, shift = compute_scale_and_shift(prediction, target, mask)

        # 2. Apply scale and shift
        scaled_prediction = scale * prediction + shift

        # 3. L1 loss on aligned prediction
        loss = F.l1_loss(scaled_prediction[mask], target[mask])
        return loss
```

**특징**:
- Scale and shift invariant (절대값 무관)
- Relative depth 학습에 최적화
- Metric 정보 학습 불가

### Gear3

**Loss 종류**: `LogL1Loss` (Inverse depth space)

```python
# train_gear3.py
class LogL1Loss(nn.Module):
    def forward(self, pred_inverse, gt_inverse, valid_mask=None):
        # Log L1 loss on inverse depth (100/m scale)
        loss = F.l1_loss(
            torch.log(pred_inverse + 1e-8),
            torch.log(gt_inverse + 1e-8),
            reduction='mean'
        )
        return loss
```

**특징**:
- **Inverse depth (100/m) 스케일**에서 직접 학습
- Log space에서 L1 loss → scale-sensitive
- **Metric 정보 직접 학습 가능**
- Valid mask 적용: `inverse_depth > 0.5` (depth < 200m)

**수식 비교**:
```
원본: L1(scale × pred + shift, GT)  → Scale/shift 자동 보정
Gear3: L1(log(100/pred), log(100/GT)) → Metric 직접 학습
```

---

## 4. 모델 아키텍처

### 원본 FlashDepth

```
Input → DINOv2 (trainable) → DPT (trainable) → Mamba (trainable) → Output
                                  ↓
                        Relative Depth (0~1)
```

**구성**:
- DINOv2: Fine-tuned (LR 5e-6)
- DPT: Fine-tuned (LR 5e-5)
- Mamba: Trainable (LR 1e-4)
- Output conv: Trainable

### Gear3

```
Input → DINOv2 (frozen) → DPT (frozen) → Gear3 Modulation → Mamba (scratch) → Output
              ↓                              ↑
    Last Block Attention          Importance Map + FG/BG Features
              ↓
      ImportancePredictor
      ForegroundBackgroundNetworks
      ModulationNetworks (×4 layers)
                ↓
         FiLM: γ, β per pixel
```

**Gear3 추가 모듈**:

1. **ImportancePredictor** (새로 추가)
   - Input: Attention weights [B, 16, 37, 37]
   - Output: Importance map [B, 1, 37, 37]
   - Zero initialization (마지막 conv layer)

2. **ForegroundBackgroundNetworks** (새로 추가)
   - Input: Patch tokens [B, 1369, 1024] + Attention weights
   - Attention-based pooling (median split)
   - Output: FG features [B, 256], BG features [B, 256]

3. **ModulationNetworks** (×4 DPT layers) (새로 추가)
   - Input: FG/BG features
   - Output: γ_fg, β_fg, γ_bg, β_bg [B, 256] per layer

4. **FeatureModulator** (새로 추가)
   - FiLM-style modulation: `modulated = γ ⊙ feature + β`
   - Spatial-adaptive: `γ[x,y] = importance[x,y] × γ_fg + (1-importance) × γ_bg`

**파라미터 수**:
| 모듈 | 원본 | Gear3 |
|------|------|-------|
| Trainable | ~320M | **~9.2M** |
| Frozen | 0 | ~311M (DINOv2 + DPT) |

---

## 5. 학습 파라미터 설정

### 원본 FlashDepth

```yaml
training:
  batch_size: 4              # Per GPU
  workers: 10
  gradient_checkpointing: true
  total_iters: 60001
  gradient_accumulation: 1
  save_freq: 1000
  val_freq: 1000
  vis_freq: 1000
  loss_type: "l1"            # SSI loss 사용
  start_with_val: false
```

### Gear3

```yaml
training:
  batch_size: 20             # Per GPU (5배 증가!)
  workers: 8
  iterations: 60001          # Same
  save_freq: 5000
  val_freq: 1000
  log_freq: 100
  # No gradient_checkpointing field
  # No gradient_accumulation
  # No vis_freq
```

**주요 차이**:
| 설정 | 원본 | Gear3 |
|------|------|-------|
| Batch size | 4 | **20** (5배 ↑) |
| Workers | 10 | 8 |
| Gradient checkpointing | Yes | No |
| Gradient accumulation | 1 | **No field** (암묵적 1) |
| Save freq | 1000 | 5000 |
| Vis freq | 1000 | **Custom** (steps 0,10,50,100, 250 간격) |
| Loss type | "l1" (SSI) | **LogL1 (inverse depth)** |

**왜 batch size가 크게 증가?**
- Frozen backbone → 메모리 절약 (~11GB attention weights 절약)
- BFloat16 사용
- Trainable params 감소 (~9M vs ~320M)

---

## 6. Optimizer 및 Scheduler

### 원본 FlashDepth

```python
# utils/init_setup.py
optimizer = torch.optim.AdamW(
    optim_list,  # [vit_params, dpt_params, mamba_params, ...]
    betas=[0.9, 0.95]
)

# Scheduler: Simple warmup only
warmup_lambda = get_warmup_lambda(cfg.training.lr.warmup_steps)
scheduler = LambdaLR(optimizer, lr_lambda=[warmup_lambda]*len(optim_list))
```

**특징**:
- AdamW optimizer
- Beta: [0.9, 0.95]
- Scheduler: **Warmup만 존재** (1000 steps)
- Warmup 후 constant LR

### Gear3

```python
# train_gear3.py
optimizer = torch.optim.Adam(  # AdamW 아님!
    param_groups,  # [gear3_params, mamba_params]
    weight_decay=1e-6
)

# Scheduler: Cosine annealing with warmup
def lr_lambda(step):
    if step < warmup_steps:              # 0-10%: Warmup
        return 0.1 + 0.9 * (step / warmup_steps)
    elif step < decay_start:             # 10-30%: Stable
        return 1.0
    else:                                 # 30-100%: Cosine decay
        progress = (step - decay_start) / (total_steps - decay_start)
        return 0.01 + 0.99 * 0.5 * (1 + cos(π × progress))

scheduler = LambdaLR(optimizer, lr_lambda)
```

**Scheduler 비교**:

| Phase | 원본 FlashDepth | Gear3 |
|-------|----------------|-------|
| 0-1k steps | Warmup (0 → 1.0) | Warmup (0.1 → 1.0) |
| 1k-60k steps | **Constant 1.0** | Stable (1.0) → Cosine decay (1.0 → 0.01) |

**그래프**:
```
원본:     0____/¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  (1.0 유지)
         0    1k                      60k

Gear3:    0.1__/¯¯¯¯¯¯¯¯¯¯¯¯¯\_____        (Cosine decay)
         0    6k      18k      60k
```

---

## 7. 데이터 로딩 및 전처리

### 원본 FlashDepth

```python
# train.py
dataloaders = get_all_dataloaders(cfg)  # Separate dataloaders per dataset
train_iterators = [iter(dataloader) for dataloader in dataloaders]
dataloader_cycle = cycle(range(len(dataloaders)))

# Cycling through dataloaders
current_loader_idx = next(dataloader_cycle)
batch = next(train_iterators[current_loader_idx])
```

**특징**:
- **Dataset별 독립적인 dataloader** 생성
- Cycle을 통해 번갈아가며 샘플링
- 각 dataset의 비율 조절 가능

### Gear3

```python
# train_gear3.py
train_dataset = CombinedDataset(
    root_dir=self.config.dataset.data_root,
    enable_dataset_flags=train_datasets,  # ['mvs-synth', 'pointodyssey', 'spring']
    resolution=self.config.dataset.resolution,
    split='train',
    video_length=5,
    color_aug=False  # No augmentation for metric training!
)

train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    sampler=train_sampler,  # DistributedSampler
    collate_fn=self.collate_fn  # Filter None values
)
```

**특징**:
- **Single CombinedDataset** 사용
- 모든 dataset이 하나로 통합
- **No color augmentation** (metric 학습에 중요!)
- Custom collate_fn으로 None 필터링

**비교**:
| 측면 | 원본 | Gear3 |
|------|------|-------|
| Dataset 구조 | 분리 (per-dataset loaders) | **통합 (CombinedDataset)** |
| Sampling | Cycle through dataloaders | Random from combined pool |
| Color aug | Yes (기본값) | **No** (metric 학습) |
| Collate fn | Default | **Custom** (None filtering) |

---

## 8. Forward Pass 흐름

### 원본 FlashDepth

```python
# flashdepth/model.py - train_sequence()
loss, grid = model.train_sequence((video, gt_depth),
                                   loss_type='l1',  # SSI loss
                                   timestep=train_step)

# Inside model:
# 1. Encoder (DINOv2) - trainable
encoder_features = self.pretrained.get_intermediate_layers(...)

# 2. DPT head - trainable
dpt_features = self.depth_head.get_forward_features(...)

# 3. Mamba temporal processing - trainable
if self.use_mamba:
    dpt_features = self.mamba(dpt_features, ...)

# 4. DPT refinement + output
depth = self.depth_head.forward_with_features(dpt_features)

# 5. Scale-shift invariant loss
loss = ScaleAndShiftInvariantLoss(depth, gt_depth)
```

### Gear3

```python
# train_gear3.py - train_step()
for t in range(T):
    img_t = images[:, t]
    gt_t = gt_depth_inverse[:, t]  # Already in 100/m scale

    # 1. Encoder (DINOv2) - frozen, no_grad
    with torch.no_grad():
        encoder_features = model.pretrained.get_intermediate_layers(...)
        attention_weights = last_block.attn.attn_weights
        patch_tokens = encoder_features[-1]

    # 2. DPT features - frozen, no_grad
    with torch.no_grad():
        dpt_features = model.depth_head.get_forward_features(...)

    # 3. Gear3 modulation - trainable
    modulated_dpt_features, importance_map = model.gear3_head(
        patch_tokens, attention_weights, dpt_features, patch_h, patch_w
    )

    # 4. Output conv - trainable
    path_1_modulated = modulated_dpt_features[-1]
    out = model.depth_head.scratch.output_conv1(path_1_modulated)
    out = F.interpolate(out, (h, w), mode="bilinear")
    out = model.depth_head.scratch.output_conv2(out)  # Softplus activation

    # 5. LogL1 loss on inverse depth
    pred_depth_inverse = out  # Already positive (Softplus)
    valid_mask = (gt_t > 0.5)  # depth < 200m
    loss_t = LogL1Loss(pred_depth_inverse, gt_t, valid_mask)
```

**핵심 차이**:

| 단계 | 원본 | Gear3 |
|------|------|-------|
| Encoder | Trainable (gradient 흐름) | **Frozen (no_grad)** |
| DPT | Trainable | **Frozen (no_grad)** |
| Modulation | 없음 | **Gear3 FiLM modulation** |
| Mamba | Trainable (relative) | **Trainable (metric, from scratch)** |
| Output | Relative depth | **Inverse depth (100/m)** |
| Loss | SSI loss | **LogL1 loss** |

---

## 9. Validation 및 Visualization

### 원본 FlashDepth

```python
# train.py - validation()
@torch.no_grad()
def validation(cfg, model, train_step, test_dataloader):
    model.eval()

    for batch in test_dataloader:
        video, gt_depth, dataset_name = batch

        # Forward with video output
        loss, grid = model(
            (video, gt_depth),
            use_mamba=True,
            out_mp4=True,
            gif_path=...,
            resolution=518
        )

        # Log to wandb
        if cfg.training.wandb:
            wandb.log({f"{dataset_name}/{k}": v for k, v in loss.items()})
            wandb.log({"vis_val": wandb.Video(grid['stacked_frames'])})
```

**특징**:
- Validation 시 video GIF/MP4 생성
- Dataset별 loss 분리 집계
- Wandb에 video 업로드

### Gear3

```python
# train_gear3.py - validate()
def validate(self):
    model.eval()

    for batch in val_loader:
        # Frame-by-frame processing
        for t in range(T):
            # Forward pass (same as training)
            pred_depth_inverse = ...
            gt_inverse = ...

            # Compute metrics (MAE, RMSE, AbsRel, δ1/δ2/δ3)
            metrics = MetricDepthMetrics.compute(pred, gt, valid_mask)

    # Visualization with Gear3Visualizer
    self.visualizer.create_validation_summary(
        sample_batch,
        model_outputs,  # pred_depth, importance_map
        step,
        prefix="validation"
    )
```

**특징**:
- Frame-by-frame metric 계산
- **Importance map visualization** 추가
- Metric depth metrics (MAE, RMSE, δ1 등)
- No video output (프레임별 분석)

**Visualization 차이**:

| 요소 | 원본 | Gear3 |
|------|------|-------|
| Video output | GIF/MP4 | **No** (이미지만) |
| Importance map | 없음 | **Yes** (FG/BG 분리) |
| Error map | 없음 | **Yes** (pixel-wise error) |
| Metrics | SSI loss only | **MAE, RMSE, AbsRel, δ1/δ2/δ3** |
| Distribution plot | 없음 | **Depth & Importance histogram** |

---

## 10. 체크포인트 로딩 전략

### 원본 FlashDepth

```python
# utils/init_setup.py - load_checkpoint()
def load_checkpoint(cfg, model, optimizer, lr_scheduler):
    checkpoint = torch.load(checkpoint_path)

    # Load full state dict
    model.load_state_dict(checkpoint['model'], strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['scheduler'])

    return checkpoint.get('train_step', 0)
```

**특징**:
- 전체 모델 가중치 로드
- Optimizer/scheduler state 복원
- Strict=False (일부 mismatch 허용)

### Gear3

```python
# train_gear3.py - _setup_model()
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['model']

# SELECTIVE LOADING: Only DINOv2 and DPT refinement
filtered_state_dict = {}
for name, param in state_dict.items():
    # Load: pretrained.*, depth_head.projects.*, depth_head.resize_layers.*,
    #       depth_head.scratch.refinenet.*
    # Exclude: mamba.*, depth_head.scratch.output_conv.*, gear3_head.*
    if any(keyword in name for keyword in
           ['pretrained.', 'depth_head.projects.', 'depth_head.resize_layers.',
            'depth_head.scratch.refinenet.']):
        filtered_state_dict[name] = param

model.load_state_dict(filtered_state_dict, strict=False)

# No optimizer/scheduler loading (train from scratch)
```

**특징**:
- **선택적 로딩** (DINOv2 + DPT refinement만)
- Mamba, output_conv, Gear3 제외 (from scratch)
- Optimizer/scheduler state 로드 안 함

**로딩 비교**:

| 모듈 | 원본 | Gear3 |
|------|------|-------|
| DINOv2 | ✅ Load | ✅ Load (frozen) |
| DPT projects/resize | ✅ Load | ✅ Load (frozen) |
| DPT refinenet | ✅ Load | ✅ Load (frozen) |
| DPT output_conv | ✅ Load | ❌ **Train from scratch** |
| Mamba | ✅ Load | ❌ **Train from scratch** |
| Gear3 modules | N/A | ❌ **New modules** |
| Optimizer | ✅ Restore | ❌ Fresh init |
| Scheduler | ✅ Restore | ❌ Fresh init |

---

## 11. Distributed Training 설정

### 원본 FlashDepth

```python
# utils/init_setup.py
def dist_init():
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=3600))

    # Seed for reproducibility
    seed = 42 + rank
    torch.manual_seed(seed)

    return dict(rank=rank, world_size=world_size, ...)

# DDP wrapping
model = DDP(
    model,
    device_ids=[local_rank],
    find_unused_parameters=True  # Allow unused params
)
```

### Gear3

```python
# train_gear3.py
def init_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl', timeout=timedelta(seconds=3600))
        dist.barrier()

    return rank, world_size, local_rank

# DDP wrapping (in Gear3Trainer)
self.model = DDP(
    self.model,
    device_ids=[local_rank],
    find_unused_parameters=False  # All params used!
)
```

**차이점**:

| 설정 | 원본 | Gear3 |
|------|------|-------|
| Single GPU fallback | No | **Yes** (world_size=1) |
| find_unused_parameters | True | **False** (모든 파라미터 사용) |
| Barrier | No | **Yes** (sync 보장) |
| Seed setting | Yes (42 + rank) | No |

---

## 12. 기타 세부사항

### 12.1 Attention Weights 저장

**원본**: 모든 block의 attention weights 저장 (메모리 낭비)

**Gear3**: Last block만 저장

```python
# flashdepth/dinov2_layers/attention.py
class MemEffAttention(Attention):
    def __init__(self):
        self.store_attn_weights = False  # Default: 저장 안 함

# train_gear3.py - 마지막 block만 활성화
for i, block in enumerate(model.pretrained.blocks):
    if i == len(model.pretrained.blocks) - 1:  # Last block
        block.attn.store_attn_weights = True
    else:
        block.attn.store_attn_weights = False
```

**메모리 절약**: 24 blocks × 11.5GB → 1 block × 0.5GB = **11GB 절약**

---

### 12.2 Canonical Space Normalization

**원본**: 없음

**Gear3**: 있음 (현재 비활성화)

```python
# train_gear3.py
class CanonicalSpaceNormalizer:
    def __init__(self, focal_canonical=1000.0, enable=True):
        self.focal_canonical = focal_canonical
        self.enable = enable  # Config: use_canonical_space=false

    def canonicalize_inverse(self, inverse_depth, focal_length):
        """
        inverse_canonical = inverse_depth / (focal_canonical / focal_actual)
        """
        if not self.enable:
            return inverse_depth

        scale_factor = self.focal_canonical / focal_length
        return inverse_depth / scale_factor
```

**현재 설정**: `use_canonical_space: false` (사용 안 함)

---

### 12.3 Valid Depth Range

**원본**: 전체 depth range 사용 (무한대 포함)

**Gear3**: 200m 이하만 학습

```python
# train_gear3.py
MIN_INVERSE_DEPTH = 0.5  # 100/200m = 0.5
valid_mask = (gt_inverse > MIN_INVERSE_DEPTH)  # depth < 200m

# utils/gear3_visualization.py
MAX_DEPTH = 200.0  # meters
gt_valid_mask = (gt_depth > 0) & (gt_depth < MAX_DEPTH)
```

**이유**:
- TartanAir, Spring 대부분 200m 이내
- Infinity depth (10억m) 제거
- 안정적인 학습

---

### 12.4 Gradient Clipping

**원본**: 없음

**Gear3**: Max norm 1.0

```python
# train_gear3.py - train_step()
loss.backward()
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
self.optimizer.step()
```

---

### 12.5 BFloat16 Autocast

**원본**: 사용 (validation만)

```python
# train.py
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    loss, grid = model(...)
```

**Gear3**: 사용 (training + validation)

```python
# train_gear3.py
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    # Forward pass
    pred_depth_inverse = ...

# Loss computation INSIDE autocast (BFloat16)
loss_t = self.loss_fn(pred_depth_inverse, gt_t, valid_mask)
```

**차이**: Gear3는 loss 계산도 BFloat16 내부에서 수행 (원본은 외부)

---

### 12.6 Training Mode 관리

**원본**: `model.train()` / `model.eval()` 단순 전환

**Gear3**: Selective training mode

```python
# train_gear3.py - _set_train_mode()
def _set_train_mode(self):
    """
    Set model to training mode, but keep frozen parts in eval mode.
    """
    self.model.train()

    # Keep frozen parts in eval mode
    for name, module in self.model.named_modules():
        if name == '':
            continue

        # Keep trainable parts in train mode
        if any(keyword in name for keyword in ['gear3_head', 'mamba', 'output_conv']):
            continue

        # Set frozen parts to eval mode
        module.eval()
```

**이유**: BatchNorm/Dropout in frozen parts 업데이트 방지

---

### 12.7 Phase-based Training

**원본**: 없음 (단일 학습 과정)

**Gear3**: 2-phase training

```yaml
# Phase 1: General datasets
train_datasets: [mvs-synth, pointodyssey, spring, tartanair, dynamicreplica]
val_datasets: [spring]

# Phase 2: nuScenes fine-tuning (planned)
train_datasets: [nuscenes]
val_datasets: [nuscenes]
```

---

## 요약 테이블

| 항목 | 원본 FlashDepth | Gear3 |
|------|----------------|-------|
| **목적** | Relative depth | **Metric depth** |
| **Loss** | ScaleAndShiftInvariant (SSI) | **LogL1 (inverse depth)** |
| **학습률** | 5개 그룹 (5e-6~1e-4) | **2개 그룹 (1e-4)** |
| **Frozen 모듈** | 없음 | **ViT, DPT** |
| **Trainable** | ~320M params | **~9.2M params** |
| **Batch size** | 4 | **20** |
| **Scheduler** | Warmup only | **Cosine annealing** |
| **Optimizer** | AdamW | **Adam** |
| **Gradient clip** | No | **Yes (1.0)** |
| **Valid range** | All | **< 200m** |
| **Attention 저장** | All blocks | **Last block만** |
| **Color aug** | Yes | **No** |
| **Modulation** | 없음 | **Gear3 FiLM** |
| **Output** | Relative depth | **Inverse depth (100/m)** |
| **Visualization** | Video GIF | **Image + metrics + importance** |

---

**마지막 업데이트**: 2025-10-07
