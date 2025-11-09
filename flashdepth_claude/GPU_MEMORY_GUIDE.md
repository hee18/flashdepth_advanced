# GPU 메모리 사용 가이드: 딥러닝 학습과 추론

## 목차
1. [GPU 메모리 구성 요소](#gpu-메모리-구성-요소)
2. [메모리 계산 방법](#메모리-계산-방법)
3. [입력 크기와 메모리 관계](#입력-크기와-메모리-관계)
4. [분산 학습과 메모리](#분산-학습과-메모리)
5. [메모리 최적화 기법](#메모리-최적화-기법)
6. [실전 예제](#실전-예제)
7. [트러블슈팅](#트러블슈팅)

---

## GPU 메모리 구성 요소

딥러닝 학습 시 GPU 메모리는 다음과 같이 구성됩니다:

```
총 GPU 메모리 = 모델 파라미터 + Gradients + Optimizer States + Activations + 기타
```

### 1. 모델 파라미터 (Model Parameters)

**설명**: 신경망의 가중치(weights)와 편향(biases)

**메모리 계산**:
```
메모리 = 파라미터 개수 × 데이터 타입 크기

데이터 타입별 크기:
- FP32 (Float32): 4 bytes
- FP16 (Float16): 2 bytes
- BF16 (BFloat16): 2 bytes
- INT8: 1 byte
```

**예시**:
- ViT-Large (340M parameters, FP32): 340M × 4 = 1.36 GB
- ViT-Large (340M parameters, BF16): 340M × 2 = 680 MB

**특징**:
- 학습/추론 모두 필요
- 배치 크기에 무관 (상수)
- 모델 크기에만 비례

---

### 2. Gradients

**설명**: 역전파(backpropagation)에서 계산된 각 파라미터의 기울기

**메모리 계산**:
```
메모리 = 파라미터 개수 × 데이터 타입 크기
     = 모델 파라미터와 동일한 크기
```

**특징**:
- **학습 시에만 필요** (추론 시 0)
- Mixed precision 사용 시에도 FP32로 저장하는 경우 많음
- `torch.no_grad()` 또는 `model.eval()` 사용 시 메모리 절약

**예시**:
- ViT-Large (340M params): 1.36 GB (FP32 gradients)

---

### 3. Optimizer States

**설명**: Optimizer가 유지하는 추가 상태 정보 (momentum, variance 등)

**메모리 계산**:

#### Adam Optimizer (가장 일반적):
```
메모리 = 파라미터 개수 × 데이터 타입 크기 × 2
       (first moment + second moment)

예: ViT-Large + Adam (FP32)
= 340M × 4 × 2 = 2.72 GB
```

#### SGD with Momentum:
```
메모리 = 파라미터 개수 × 데이터 타입 크기 × 1

예: ViT-Large + SGD
= 340M × 4 = 1.36 GB
```

#### SGD without Momentum:
```
메모리 ≈ 0 (상태 저장 없음)
```

**특징**:
- **학습 시에만 필요**
- Adam이 SGD보다 2배 많은 메모리 사용
- 8-bit optimizer 사용 시 ~75% 절약 가능

---

### 4. Activations (Feature Maps)

**설명**: Forward pass에서 생성되는 중간 결과물 (역전파를 위해 저장)

**메모리 계산**:
```
메모리 = Σ(각 레이어의 activation 크기)
     = 배치 크기 × 시퀀스 길이 × 채널 수 × 높이 × 너비 × 데이터 타입 크기
```

**특징**:
- **가장 큰 메모리 소비원** (특히 고해상도/큰 배치)
- 배치 크기에 **선형 비례**
- 입력 해상도에 **제곱 비례** (이미지의 경우)
- Gradient checkpointing으로 크게 절약 가능

#### Vision Transformer (ViT) Activation 메모리:

```
주요 구성 요소:

1. Patch Embedding
   메모리 = B × (H/P × W/P) × D × 4 bytes

2. Self-Attention
   - Query, Key, Value: 각각 B × N × D × 4 bytes
   - Attention weights: B × num_heads × N × N × 4 bytes

3. MLP (Feed-Forward)
   - Hidden: B × N × (D × expansion) × 4 bytes

여기서:
- B: 배치 크기
- H, W: 입력 이미지 높이, 너비
- P: 패치 크기 (예: 14 또는 16)
- N: 토큰 개수 = (H/P) × (W/P) + 1
- D: Embedding 차원
```

**Attention Memory가 특히 큰 이유**:

```
Attention weights = B × num_heads × N × N

예시 (ViT-Large, 1920×1080 이미지, batch=1):
- Patch size: 14
- Tokens: (1920/14) × (1080/14) = 137 × 77 = 10,549
- Attention per head: 10,549² = 111,281,401 entries
- 16 heads: 1,780,502,416 entries
- BF16: 1,780,502,416 × 2 bytes = 3.56 GB per layer!
```

---

### 5. 기타 (임시 버퍼, CUDA context 등)

**구성 요소**:
- CUDA kernels
- cuDNN workspace
- PyTorch 내부 캐싱
- Fragmentation overhead

**메모리**: 보통 1-3 GB

---

## 메모리 계산 방법

### 학습(Training) 시 총 메모리

```
총 메모리 = 모델 파라미터 (P)
         + Gradients (P)
         + Optimizer States (2P for Adam, P for SGD)
         + Activations (배치/해상도 의존)
         + 기타 (1-3 GB)

Mixed Precision (BF16) 사용 시:
- 모델: P/2 (BF16)
- Gradients: P (FP32, master copy)
- Optimizer: 2P (FP32)
- Activations: depends on autocast scope

총 ≈ 3.5P + Activations + 기타
```

### 추론(Inference) 시 총 메모리

```
총 메모리 = 모델 파라미터 (P)
         + Activations (배치/해상도 의존)
         + 기타 (1-3 GB)

torch.no_grad() 사용 시:
- Gradients: 0
- Optimizer States: 0
- Activations: 크게 감소 (중간 결과 저장 불필요)

총 ≈ P + Activations(reduced) + 기타
```

---

## 입력 크기와 메모리 관계

### 배치 크기 (Batch Size)

```
메모리 증가율: 선형 (Linear)

batch_size = 2 → 메모리 2배
batch_size = 4 → 메모리 4배

주의: 모델 파라미터는 배치 크기에 무관!
→ Activation memory만 배수로 증가
```

**최적 배치 크기 찾기**:
```python
# Binary search로 최대 배치 크기 찾기
max_batch = 1
while True:
    try:
        # Test with batch_size = max_batch
        train_one_step(batch_size=max_batch)
        max_batch *= 2
    except torch.cuda.OutOfMemoryError:
        max_batch //= 2
        break
```

---

### 이미지 해상도 (Image Resolution)

```
메모리 증가율: 제곱 (Quadratic)

특히 Transformer/Attention의 경우:
- 토큰 개수: N = (H/P) × (W/P)
- Attention: O(N²)

해상도 2배 → 토큰 4배 → Attention memory 16배!
```

**예시**:

| 해상도 | 토큰 개수 (N) | Attention (N²) | 메모리 비율 |
|--------|--------------|---------------|------------|
| 518×518 | 1,369 | 1.9M | 1x (baseline) |
| 1024×1024 | 5,329 | 28.4M | 15x |
| 1920×1080 | 10,549 | 111M | 59x |
| 2048×2048 | 21,316 | 454M | 241x |

**해상도 최적화 전략**:
1. 학습: 낮은 해상도로 시작 → 점진적 증가
2. 추론: Multi-scale testing
3. Validation: 학습보다 낮은 해상도 사용 가능

---

### 시퀀스 길이 (Sequence Length / Video)

```
비디오 처리 시:
- 시퀀스 길이 T frames
- 메모리 ∝ T (일반적으로 선형)
- 단, Temporal attention 사용 시 O(T²)

예: video_length=5 → video_length=10
→ 2배 메모리 (linear attention)
→ 4배 메모리 (full temporal attention)
```

---

## 분산 학습과 메모리

### 1. Data Parallel (DP) - 단일 노드

```
메모리 사용: 각 GPU에 모델 전체 복제

GPU 0: 모델 + gradients + optimizer + activations
GPU 1: 모델 + gradients + optimizer + activations
GPU 2: 모델 + gradients + optimizer + activations

효과: 메모리 절약 없음 (배치만 분산)
장점: 구현 간단, 처리량 증가
단점: GPU 0에 병목 (gradient aggregation)
```

---

### 2. Distributed Data Parallel (DDP)

```
메모리 사용: 각 GPU에 모델 전체 복제

GPU 0: 모델 + gradients + optimizer + activations
GPU 1: 모델 + gradients + optimizer + activations
GPU 2: 모델 + gradients + optimizer + activations

효과: 메모리 절약 없음
장점:
  - DP보다 효율적 (all-reduce)
  - 다중 노드 지원
단점: 여전히 각 GPU가 모든 파라미터 저장
```

**코드 예시**:
```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(
    model,
    device_ids=[local_rank],
    find_unused_parameters=True
)
```

---

### 3. Fully Sharded Data Parallel (FSDP)

```
메모리 사용: 모델/gradients/optimizer를 샤딩

GPU 0: 모델(1/3) + gradients(1/3) + optimizer(1/3) + activations
GPU 1: 모델(1/3) + gradients(1/3) + optimizer(1/3) + activations
GPU 2: 모델(1/3) + gradients(1/3) + optimizer(1/3) + activations

효과: 파라미터 관련 메모리를 N개 GPU로 분산
절약량:
  - 모델: P → P/N
  - Gradients: P → P/N
  - Optimizer: 2P → 2P/N
  - 총: 4P → 4P/N

단점: Activations는 여전히 각 GPU에서 동일
```

**메모리 계산 (3 GPU, ViT-Large)**:
```
DDP:
- 각 GPU: 680MB(model) + 680MB(grad) + 1.36GB(opt) = 2.72GB
- 총: 8.16GB

FSDP:
- 각 GPU: 227MB + 227MB + 453MB = 907MB
- 총: 2.72GB
→ 파라미터 관련 5.44GB 절약
```

**코드 예시**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mixed_precision_policy,
)
```

---

### 4. DeepSpeed ZeRO

ZeRO는 3단계로 나뉩니다:

#### ZeRO Stage 1: Optimizer State Partitioning
```
효과: Optimizer states만 샤딩
절약: 2P → 2P/N
예: Adam states 2.72GB → 907MB (3 GPU)
```

#### ZeRO Stage 2: Gradient Partitioning
```
효과: Optimizer + Gradients 샤딩
절약: 3P → 3P/N
예: Opt(2.72GB) + Grad(1.36GB) → 1.36GB (3 GPU)
```

#### ZeRO Stage 3: Parameter Partitioning
```
효과: Model + Optimizer + Gradients 모두 샤딩
절약: 4P → 4P/N (FSDP와 유사)
추가 기능: CPU/NVMe offload 가능
```

**ZeRO-3 + Offload**:
```
GPU 메모리: P/N + Activations
CPU 메모리: Optimizer states (2P)
효과: 매우 큰 모델도 학습 가능 (단, 속도 저하)
```

**코드 예시**:
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    },
    "offload_param": {
      "device": "cpu"
    }
  }
}
```

---

### 5. Model Parallelism (Pipeline / Tensor)

#### Pipeline Parallelism:
```
레이어를 여러 GPU에 분산:

GPU 0: Layer 1-8
GPU 1: Layer 9-16
GPU 2: Layer 17-24

효과: 모델 크기 P → P/N
단점: Pipeline bubble (일부 GPU idle)
```

#### Tensor Parallelism:
```
레이어 내부를 분할:

GPU 0: Attention heads 1-8
GPU 1: Attention heads 9-16

효과: 레이어당 메모리 절약
단점: 통신 오버헤드 큼, 구현 복잡
```

---

### 분산 학습 방법 비교표

| 방법 | 모델 메모리 | Gradient | Optimizer | Activations | 구현 난이도 | 통신 비용 |
|------|------------|----------|-----------|-------------|------------|----------|
| **DP** | 복제 | 복제 | 복제 | 분산 | ⭐ | 높음 |
| **DDP** | 복제 | 복제 | 복제 | 분산 | ⭐⭐ | 중간 |
| **FSDP** | 분산 | 분산 | 분산 | 복제 | ⭐⭐⭐ | 중간 |
| **ZeRO-1** | 복제 | 복제 | 분산 | 복제 | ⭐⭐ | 낮음 |
| **ZeRO-2** | 복제 | 분산 | 분산 | 복제 | ⭐⭐ | 중간 |
| **ZeRO-3** | 분산 | 분산 | 분산 | 복제 | ⭐⭐⭐ | 높음 |
| **Pipeline** | 분산 | 분산 | 분산 | 분산 | ⭐⭐⭐⭐ | 낮음 |
| **Tensor** | 분산 | 분산 | 분산 | 분산 | ⭐⭐⭐⭐⭐ | 매우높음 |

---

## 메모리 최적화 기법

### 1. Gradient Checkpointing (Activation Checkpointing)

**원리**:
- Forward pass에서 일부 activations만 저장
- Backward pass에서 필요 시 재계산

```python
from torch.utils.checkpoint import checkpoint

# 일반적인 forward
output = layer(input)  # activation 저장

# Gradient checkpointing
output = checkpoint(layer, input)  # activation 저장 안 함, 역전파 시 재계산
```

**효과**:
```
메모리 절약: 50-80% (activation 부분)
계산 비용 증가: 30-50% (재계산 필요)

예: Transformer 24 layers
- 일반: 모든 layer의 activation 저장
- Checkpointing: 2-3개 layer만 저장, 나머지 재계산
```

**적용 전략**:
```python
# 모든 레이어에 적용
apply_activation_checkpointing(
    model,
    check_fn=lambda _: True
)

# 특정 레이어만 (예: Transformer blocks)
apply_activation_checkpointing(
    model,
    check_fn=lambda m: isinstance(m, TransformerBlock)
)
```

---

### 2. Mixed Precision Training

**FP16 / BF16 사용**:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in BF16
    with autocast(dtype=torch.bfloat16):
        output = model(input)
        loss = criterion(output, target)

    # Backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**효과**:
```
모델 메모리: 50% 절약 (FP32 → BF16)
Activations: 50% 절약
Gradients: 일부만 절약 (master weights는 FP32)

총 메모리 절약: 30-40%
```

**BF16 vs FP16**:
```
BF16 (BFloat16):
- 장점: FP32와 같은 range, overflow 없음
- 단점: 정밀도 약간 낮음
- 추천: 대부분의 경우

FP16 (Float16):
- 장점: 약간 더 빠름
- 단점: Gradient scaling 필요, overflow 위험
- 추천: 안정성이 검증된 경우
```

---

### 3. Gradient Accumulation

**원리**:
- 작은 배치로 여러 번 forward/backward
- Gradient를 누적한 후 한 번에 업데이트

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output, target)
    loss = loss / accumulation_steps  # 스케일 조정
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**효과**:
```
실제 배치 크기 = batch_size × accumulation_steps
메모리는 batch_size만큼만 사용

예:
- 목표: batch_size = 32
- OOM 발생 → batch_size = 8로 줄임
- accumulation_steps = 4 설정
- 효과: 메모리는 8, 효과는 32
```

**주의사항**:
- BatchNorm은 실제 배치 크기로 계산 (8)
- Learning rate 조정 필요할 수 있음
- 속도는 느려짐 (4배의 forward/backward)

---

### 4. Memory-Efficient Attention

**문제**: Standard attention은 O(N²) 메모리

```
Standard Attention:
Q, K, V = ..., ..., ...
scores = Q @ K.T  # [B, H, N, N] - 매우 큼!
attn = softmax(scores)
out = attn @ V
```

**해결책**:

#### Flash Attention:
```python
from flash_attn import flash_attn_func

# O(N) 메모리, O(N²) 계산
out = flash_attn_func(q, k, v, causal=False)
```

**효과**:
```
메모리: O(N²) → O(N)
속도: 2-4배 빠름
정확도: 동일

예: 10K tokens
- Standard: 100M entries (400MB)
- Flash Attn: 10K entries (40KB)
→ 10,000배 절약!
```

#### Memory-Efficient Attention (xFormers):
```python
from xformers.ops import memory_efficient_attention

out = memory_efficient_attention(q, k, v)
```

#### Linear Attention (근사):
```python
# Kernel trick으로 O(N) 복잡도
out = linear_attention(q, k, v)  # 정확도 약간 손실
```

---

### 5. Model Pruning & Quantization

#### Pruning (가지치기):
```python
import torch.nn.utils.prune as prune

# 가중치의 30%를 0으로
prune.l1_unstructured(module, name='weight', amount=0.3)

# 효과: 메모리 절약 (sparse 저장 시)
#       속도 향상 (0 계산 스킵)
```

#### Quantization (양자화):
```python
# FP32 → INT8
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 효과: 4배 메모리 절약, 2-4배 속도 향상
#       정확도 약간 손실 (1-2%)
```

---

### 6. 기타 최적화

#### CPU Offload:
```python
# 일부 레이어를 CPU에 저장
model.layer1.to('cpu')
model.layer2.to('cuda')

# Forward 시 자동으로 이동
# 효과: GPU 메모리 절약, 속도 저하
```

#### Empty Cache:
```python
# 사용하지 않는 캐시 정리
torch.cuda.empty_cache()

# 주의: 실제 메모리 해제는 안 됨, fragmentation만 정리
# Validation 전/후 사용 권장
```

#### Smaller Datatypes:
```python
# 입력 데이터도 작게
images = images.to(torch.bfloat16)

# 일부 변수는 FP16 사용
loss = loss.half()
```

---

## 실전 예제

### 예제 1: ViT-Large 학습 (단일 GPU)

**시나리오**:
- 모델: ViT-Large (340M params)
- 입력: 1920×1080 RGB 이미지
- GPU: A100 40GB
- 목표: OOM 없이 학습

**메모리 분석**:

```
1. 모델 파라미터 (FP32):
   340M × 4 bytes = 1.36 GB

2. Gradients (FP32):
   340M × 4 bytes = 1.36 GB

3. Optimizer (Adam, FP32):
   340M × 4 × 2 = 2.72 GB

4. Activations (1920×1080, batch=1):
   - Tokens: 137 × 77 = 10,549
   - Attention (24 layers, 16 heads):
     ≈ 24 × (10,549² × 16 × 2 bytes) = 101 GB
   - 다른 activations: ~5 GB
   총: ~106 GB (!!!)

총: 1.36 + 1.36 + 2.72 + 106 = 111.44 GB
→ 40GB GPU로는 불가능!
```

**해결 방법**:

```python
# 1단계: Mixed Precision (30-40% 절약)
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)

# 예상 절약: ~35 GB → 총 76 GB (여전히 부족)

# 2단계: Gradient Checkpointing (50% 절약)
from torch.distributed.fsdp.wrap import checkpoint_wrapper
model = checkpoint_wrapper(model)

# 예상 절약: ~53 GB → 총 23 GB (아직 부족)

# 3단계: Resolution 감소 (제곱 효과!)
# 1920×1080 → 1280×720 (0.67배)
# Activations: 106 GB × (0.67)² = 48 GB
# 절약: 58 GB → 총 53 GB → 24 GB (해결!)

# 또는 Flash Attention (가장 효과적)
from flash_attn import flash_attn_func
# Attention: 101 GB → ~1 GB
# 총: 111 - 100 = 11 GB (완전 해결!)
```

**최종 설정**:
```python
# 옵션 1: Flash Attention
model = ViTLarge(use_flash_attn=True)
# 메모리: ~11 GB
# 배치 크기: 4-8 가능

# 옵션 2: Mixed + Checkpointing + Resolution
model = ViTLarge()
with autocast(dtype=torch.bfloat16):
    with checkpoint_wrapper(model):
        output = model(resize(input, (1280, 720)))
# 메모리: ~24 GB
# 배치 크기: 1-2 가능
```

---

### 예제 2: DDP vs FSDP (8 GPU 학습)

**시나리오**:
- 모델: GPT-3 (175B params)
- GPU: 8× A100 80GB
- 비교: DDP vs FSDP

#### DDP:
```
각 GPU:
- 모델: 175B × 4 = 700 GB (!!!)
→ 불가능! 각 GPU가 전체 모델을 저장해야 함
```

#### FSDP:
```
각 GPU:
- 모델: 700 GB / 8 = 87.5 GB
- Gradients: 87.5 GB / 8 = 10.9 GB
- Optimizer: 175 GB / 8 = 21.9 GB
- Activations: ~10 GB (추정)
총: 130.3 GB

→ 여전히 불가능 (80 GB < 130 GB)
```

#### FSDP + CPU Offload:
```
각 GPU:
- 모델: 87.5 GB (필요 시 fetch)
- Activations: ~10 GB
총: ~15 GB (가능!)

CPU:
- Optimizer states: 175 GB (offload)
- 예비 파라미터: 700 GB (offload)
총: 875 GB CPU RAM 필요
```

**결론**: GPT-3 규모는 FSDP + CPU offload 필수

---

### 예제 3: Video Model 메모리 분석

**시나리오**:
- 모델: FlashDepth (340M params)
- 입력: 1920×1080, 5 frames (video)
- GPU: A100 48GB

**메모리 계산**:

```
1. 모델 관련 (BF16):
   - 파라미터: 680 MB
   - Gradients: 680 MB (FP32)
   - Optimizer: 1.36 GB (Adam, FP32)
   총: 2.72 GB

2. Activations (5 frames):
   - Per frame: 10,549 tokens
   - Total tokens: 5 × 10,549 = 52,745

   Spatial attention (per frame):
   - 10,549² × 16 heads × 24 layers × 2 bytes
   - ≈ 101 GB per frame
   - × 5 frames = 505 GB (!!!)

   Temporal attention (across frames):
   - 52,745² × 8 heads × 4 layers × 2 bytes
   - ≈ 177 GB (!!!)

   총: 682 GB (완전히 불가능)

3. 해결책: Mamba (Linear RNN)
   - Spatial attention: 101 GB → 유지 필요
   - Temporal: 177 GB → ~1 GB (recurrent)
   절약: 176 GB

   총: 2.72 + 101 × 5 + 1 = 508 GB
   (여전히 너무 큼)

4. 추가 해결: Flash Attention + 낮은 해상도
   - Flash Attn: 101 GB → 1 GB per frame
   - 5 frames: 5 GB
   - Resolution 1280×720: 5 GB × 0.44 = 2.2 GB

   총: 2.72 + 2.2 + 1 = 5.92 GB (해결!)
```

**최종 구성**:
```python
model = FlashDepth(
    use_flash_attn=True,
    use_mamba=True,  # Temporal modeling
    resolution=(1280, 720),
    video_length=5
)

# 메모리: ~6 GB
# 가능 배치 크기: 4-6
```

---

## 트러블슈팅

### OOM 발생 시 체크리스트

#### 1단계: 문제 파악
```bash
# GPU 메모리 모니터링
nvidia-smi -l 1

# PyTorch 메모리 상세
python -c "
import torch
print(torch.cuda.memory_summary())
"
```

#### 2단계: 즉시 시도할 것들
```python
# 1. 배치 크기 줄이기
batch_size = 1  # 최소값

# 2. 캐시 정리
torch.cuda.empty_cache()

# 3. Gradient 저장 끄기 (추론 시)
with torch.no_grad():
    output = model(input)

# 4. 해상도 줄이기
input = F.interpolate(input, size=(512, 512))
```

#### 3단계: 체계적 최적화
```python
# 1. Mixed Precision
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)

# 2. Gradient Checkpointing
from torch.utils.checkpoint import checkpoint
model.layer = checkpoint(model.layer)

# 3. Gradient Accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch) / accum_steps
    loss.backward()
    if (i+1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 4단계: 고급 기법
```python
# 1. FSDP (다중 GPU)
from torch.distributed.fsdp import FSDP
model = FSDP(model)

# 2. Flash Attention
from flash_attn import flash_attn_func

# 3. CPU Offload
model.layer1.to('cpu')
```

---

### 일반적인 OOM 시나리오

#### Scenario 1: Validation 중 OOM
```python
# 원인: Validation 데이터가 더 큰 해상도
# 해결:
def validate():
    model.eval()
    with torch.no_grad():  # Gradient 저장 안 함
        for batch in val_loader:
            # 해상도 줄이기
            batch = F.interpolate(batch, size=(512, 512))
            output = model(batch)

    # Validation 후 캐시 정리
    torch.cuda.empty_cache()
```

#### Scenario 2: 첫 iteration은 성공, 이후 OOM
```python
# 원인: Memory fragmentation
# 해결:
# 1. 주기적으로 캐시 정리
if step % 100 == 0:
    torch.cuda.empty_cache()

# 2. PyTorch 설정
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

#### Scenario 3: Multi-GPU에서 GPU 0만 OOM
```python
# 원인: DP 사용 중 (GPU 0에 병목)
# 해결: DDP로 전환
# Before (DP):
model = nn.DataParallel(model)

# After (DDP):
model = DDP(model, device_ids=[local_rank])
```

---

## 메모리 프로파일링 도구

### 1. PyTorch Profiler
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_memory_usage"))
```

### 2. nvidia-smi
```bash
# 실시간 모니터링
watch -n 1 nvidia-smi

# 로그 저장
nvidia-smi --query-gpu=timestamp,memory.used,memory.free \
           --format=csv -l 1 > gpu_mem.log
```

### 3. torch.cuda.memory_summary()
```python
import torch

# 현재 메모리 상태
print(torch.cuda.memory_summary())

# 메모리 스냅샷
snapshot = torch.cuda.memory_snapshot()

# 할당 기록
print(torch.cuda.memory_stats())
```

---

## 참고 자료

### 논문
- FlashAttention: [Dao et al., 2022]
- ZeRO: [Rajbhandari et al., 2020]
- Gradient Checkpointing: [Chen et al., 2016]

### 공식 문서
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- DeepSpeed: https://www.deepspeed.ai/
- Mixed Precision: https://pytorch.org/docs/stable/amp.html

### 도구
- DeepSpeed: https://github.com/microsoft/DeepSpeed
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- xFormers: https://github.com/facebookresearch/xformers

---

## 요약: 메모리 절약 효과 한눈에 보기

| 기법 | 메모리 절약 | 속도 영향 | 구현 난이도 | 추천도 |
|-----|-----------|---------|-----------|--------|
| 배치 크기 ↓ | 선형 | 느려짐 | ⭐ | ⭐⭐⭐⭐⭐ |
| 해상도 ↓ | 제곱 | 빨라짐 | ⭐ | ⭐⭐⭐⭐⭐ |
| Mixed Precision | 30-40% | 빨라짐 | ⭐ | ⭐⭐⭐⭐⭐ |
| Gradient Checkpoint | 50-80% | 느려짐 | ⭐⭐ | ⭐⭐⭐⭐ |
| Flash Attention | 90%+ (attn) | 빨라짐 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| FSDP | 75% (params) | 약간느림 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| ZeRO-3 | 75%+ | 느려짐 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| CPU Offload | 크게 절약 | 매우느림 | ⭐⭐ | ⭐⭐⭐ |
| Gradient Accum | 활용도↑ | 느려짐 | ⭐ | ⭐⭐⭐⭐ |

**빠른 결정 가이드**:

1. **단일 GPU OOM**: 해상도 ↓ + Mixed Precision + Gradient Checkpoint
2. **Attention 메모리 문제**: Flash Attention
3. **다중 GPU**: FSDP 또는 ZeRO-2
4. **거대 모델**: ZeRO-3 + CPU Offload
5. **배치 크기 제약**: Gradient Accumulation

---

*최종 수정: 2025-11-07*
