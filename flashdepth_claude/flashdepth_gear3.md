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

Gear3는 FlashDepth에 **Feature-level Metric Injection**을 적용하여 metric depth 추정 성능을 향상시킵니다. 기존 GSP(Global Scale Predictor)와 달리, **DPT feature에 직접 FiLM-style modulation**을 적용합니다.

### 기존 방식 vs Gear3

| 방식 | GSP (기존) | Gear3 (현재) |
|------|-----------|-------------|
| **Metric 주입** | Depth map에 scale/shift | **DPT features에 modulation** |
| **공간 변화** | 전역 균일 | **Importance map 기반 spatial modulation** |
| **FG/BG 구분** | 없음 | **Separate FG/BG modulation** |
| **학습 파라미터** | ~0.5M | **~9.2M** (Gear3 + Mamba + output_conv) |

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
         ImportancePredictor              ForegroundBackgroundNetworks
         (zero init last layer)                  (Simple GAP)
                    ↓                                   ↓
         Importance Map [B,1,H,W]          FG/BG Features [B,256]
                    ↓                                   ↓
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

**Simple Global Average Pooling (GAP)**:
```python
def forward(self, patch_tokens):
    global_features = patch_tokens.mean(dim=1)  # [B, embed_dim]
    fg_features = self.fg_net(global_features)  # [B, 256]
    bg_features = self.bg_net(global_features)  # [B, 256]
    return fg_features, bg_features
```

**Note**: Importance-weighted pooling 대신 simple GAP 사용 (gradient separation이 더 중요)

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
  batch_size: 22        # Per GPU (effective 44 with DDP)
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
    training.batch_size=22 \
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
  training.batch_size=22 \
  training.workers=8 \
  +results_dir=train_results/results_7 \
  load=configs/flashdepth-l/iter_10001.pth

# Docker
./run_docker.sh train_gear3_ddp \
  --batch-size 22 \
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

**Current (batch_size=22 per GPU)**:
- GPU 0: ~45GB / 48GB (**3GB 여유**)
- GPU 1: ~36GB / 48GB (12GB 여유)
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

**현재 (batch 22×2, DDP)**:
- Effective batch: **44** (5.5배!)
- Training time: **~25시간** (83% 감소)

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
│   └── config.yaml               # Gear3 설정 (batch 22, workers 8)
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
