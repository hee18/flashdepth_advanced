# Mamba2 구조 및 시간적 모델링

## 개요

Mamba는 **State Space Model (SSM)** 기반의 시퀀스 모델링 아키텍처로, Transformer의 대안으로 설계되었습니다. FlashDepth에서는 **비디오 프레임 간 시간적 일관성**을 유지하기 위해 사용됩니다.

**핵심 특징:**
- O(N) 복잡도 (Transformer는 O(N²))
- 긴 시퀀스에 효율적 (비디오 처리에 최적)
- Selective state space: 입력에 따라 동적으로 변화하는 hidden state

---

## Mamba vs Transformer

### Complexity 비교

| 모델 | Time Complexity | Space Complexity | Long Context |
|------|----------------|------------------|--------------|
| **Transformer** | O(N²) | O(N²) | ❌ (메모리 폭발) |
| **Mamba** | O(N) | O(N) | ✅ (효율적) |

### 왜 비디오에 Mamba를 사용하나?

```
비디오 시퀀스 예시:
- 프레임 수 T = 50
- 각 프레임의 패치 수 = 37×37 = 1369

Transformer Self-Attention:
- Sequence length = T × patches = 50 × 1369 = 68,450
- Attention matrix = 68,450 × 68,450 ≈ 4.7억 elements
- Memory: 4.7억 × 4 bytes ≈ 1.9 GB (float32 기준)
❌ 메모리 폭발!

Mamba:
- Sequence length = 50 (프레임만 처리)
- State size = 256 (고정)
- Memory: O(50 × 256) ≈ 12.8 KB
✅ 효율적!
```

---

## State Space Model (SSM) 기초

### 고전적 SSM (LTI - Linear Time-Invariant)

```
Continuous-time:
    dx/dt = A·x(t) + B·u(t)
    y(t)  = C·x(t) + D·u(t)

Discrete-time (컴퓨터 구현):
    x[t] = A·x[t-1] + B·u[t]
    y[t] = C·x[t]    + D·u[t]

where:
    x[t]: Hidden state (d_state 차원)
    u[t]: Input (입력)
    y[t]: Output (출력)
    A, B, C, D: 학습 가능한 행렬
```

**문제점:**
- A, B, C가 **시간에 무관하게 고정** (time-invariant)
- 입력에 따라 동적으로 변하지 않음
- 표현력 제한

### Selective SSM (Mamba의 핵심)

```
x[t] = A(u[t])·x[t-1] + B(u[t])·u[t]
y[t] = C(u[t])·x[t]

A, B, C가 입력 u[t]에 의존! (Input-dependent)
```

**장점:**
- 중요한 정보는 오래 기억 (큰 A)
- 덜 중요한 정보는 빨리 잊음 (작은 A)
- **Content-aware filtering**

---

## Mamba2 아키텍처

### MambaBlock 구조

```python
# flashdepth/mamba.py:31-80
class MambaBlock(nn.Module):
    def __init__(self, d_model, layer_idx, expand, d_state, d_conv, headdim):
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,      # 256 (DPT feature dimension)
            d_state=d_state,      # 256 (SSM state dimension)
            d_conv=d_conv,        # 4 (local convolution width)
            expand=expand,        # 2 (expansion factor)
            headdim=headdim       # 64 (head dimension)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, inference_params=None):
        # SSM + Residual
        residual = x
        x = self.norm1(x)
        x = self.mamba(x, inference_params=inference_params)
        x = residual + x

        # MLP + Residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x
```

**구조:**
```
Input [B, L, d_model]
  ↓ LayerNorm
  ↓ Mamba2 (SSM)
  ↓ Residual Add
  ↓ LayerNorm
  ↓ MLP (4× expansion)
  ↓ Residual Add
Output [B, L, d_model]
```

**Transformer Block과 비교:**
| Component | Transformer | Mamba |
|-----------|-------------|-------|
| **주 연산** | Multi-Head Attention | Selective SSM |
| **Complexity** | O(L²) | O(L) |
| **Position** | Positional Encoding | Implicit (convolution) |
| **Memory** | Key-Value Cache (크다) | State (작다) |

---

## Mamba2 내부 구조

### Mamba2 Forward Pass

```python
# mamba_ssm/modules/mamba2.py (simplified)
class Mamba2(nn.Module):
    def forward(self, u, inference_params=None):
        # u: [B, L, d_model] (예: [1, 1369, 256])

        # 1. Expand to internal dimension
        zxbcdt = self.in_proj(u)  # [B, L, d_inner * 2 + 2*ngroups*d_state + nheads]

        # Split into components
        z, xBC, dt = torch.split(zxbcdt, [d_inner, d_inner + 2*ngroups*d_state, nheads], dim=-1)

        # 2. SiLU activation
        z = F.silu(z)  # Gate

        # 3. Short convolution (local context)
        xBC = self.conv1d(xBC)  # Causal conv, kernel=4

        # 4. Split x, B, C
        x, B, C = torch.split(xBC, [d_inner, ngroups*d_state, ngroups*d_state], dim=-1)

        # 5. SSM operation (핵심!)
        y = selective_scan(x, dt, A, B, C, D)
        #   x: input [B, L, d_inner]
        #   dt: delta (time step) [B, L, nheads]
        #   A: state transition [nheads, d_state]
        #   B, C: input/output projection [B, L, ngroups*d_state]

        # 6. Gating and output projection
        y = y * F.silu(z)
        out = self.out_proj(y)  # [B, L, d_model]

        return out
```

### Selective Scan 연산

```python
# 핵심 알고리즘 (simplified)
def selective_scan(x, dt, A, B, C, D):
    B, L, d_inner = x.shape
    d_state = A.shape[-1]

    # Initialize state
    h = torch.zeros(B, d_state)  # Hidden state

    outputs = []
    for t in range(L):
        # Input-dependent discretization
        A_t = torch.exp(dt[:, t] * A)  # [B, d_state]
        B_t = B[:, t]  # [B, d_state]

        # State update (selective!)
        h = A_t * h + B_t * x[:, t]  # [B, d_state]

        # Output
        y_t = C[:, t] @ h  # [B, d_inner]
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)  # [B, L, d_inner]
    return y + D * x  # Skip connection
```

**핵심 아이디어:**
1. **dt (delta)**: 각 입력마다 다른 time step
   - 중요한 정보: 큰 dt (느리게 변화)
   - 덜 중요한 정보: 작은 dt (빠르게 변화)

2. **A, B, C가 입력 의존적**
   - 각 프레임마다 다른 transition matrix
   - Content-aware state update

3. **State는 작다** (d_state = 256)
   - Transformer의 Key-Value cache보다 훨씬 작음
   - 긴 시퀀스도 효율적으로 처리

---

## FlashDepth에서의 Mamba 사용

### MambaModel 구조

```python
# flashdepth/mamba.py:86-154
class MambaModel(nn.Module):
    def __init__(self, dpt_dim, mamba_type, num_mamba_layers, batch_size):
        # num_mamba_layers = 4 (기본값, ViT-L)
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=dpt_dim,      # 256
                expand=2,             # Internal dimension = 512
                d_state=256,          # State size
                d_conv=4,             # Conv kernel size
                layer_idx=i,
                headdim=64
            ) for i in range(num_mamba_layers)
        ])

        self.mamba_type = mamba_type  # 'add' or 'modulation'
        self.inference_params = InferenceParams(...)  # State 저장용
```

### 두 가지 통합 방식

#### 1. Addition Mode (`mamba_type='add'`)

```python
# flashdepth/mamba.py:197-199
if self.mamba_type == 'add':
    x = self.final_layer(x)
    x = x + frame  # Residual add
```

**수식:**
```
output = Mamba(input) + input
```

**특징:**
- 단순한 residual connection
- 안정적인 학습
- **FlashDepth-L 기본 설정**

#### 2. Modulation Mode (`mamba_type='modulation'`)

```python
# flashdepth/mamba.py:192-195
if self.mamba_type == 'modulation':
    x = self.final_layer(x)
    scale, shift = x.chunk(2, dim=-1)
    x = (1 + scale) * frame + shift
```

**수식:**
```
scale, shift = Linear(Mamba(input))
output = (1 + scale) ⊙ input + shift
```

**특징:**
- FiLM-style modulation
- 더 강한 표현력
- 학습이 약간 불안정할 수 있음

---

### 비디오 처리 흐름

#### 시퀀스 시작

```python
# flashdepth/model.py:844
self.mamba.start_new_sequence()
```

```python
# flashdepth/mamba.py:155-160
def start_new_sequence(self):
    """Reset for new video sequence"""
    self.inference_params = InferenceParams(
        max_seqlen=60000,
        max_batch_size=self.max_batch_size
    )
```

**목적:**
- 새 비디오 시작 시 hidden state 초기화
- 이전 비디오 정보가 섞이는 것 방지

#### 프레임 단위 처리

```python
# flashdepth/model.py:416-438 (train_sequence)
B, T, C, H, W = video.shape  # [1, 5, 3, 518, 518]
video = rearrange(video, 'b t c h w -> (b t) c h w')  # [5, 3, 518, 518]

dpt_features = self.get_dpt_features(video, input_shape=(B, T, C, H, W))
# Mamba는 내부에서 프레임별로 처리
```

#### Mamba 호출 (DPT 내부)

```python
# flashdepth/model.py:107-157 (dpt_features_to_mamba)
def dpt_features_to_mamba(self, input_shape, dpt_features, in_dpt_layer):
    B, T, C, H, W = input_shape  # [1, 5, 3, 518, 518]
    BT, c, h, w = dpt_features.shape  # [5, 256, 37, 37]

    # Reshape to sequence: [BT, c, h, w] → [B, T, h*w, c]
    dpt_features = rearrange(dpt_features, '(b t) c h w -> b t (h w) c',
                            b=B, t=T)  # [1, 5, 1369, 256]

    mamba_out = []
    for i in range(T):
        # 프레임별 처리 (시간 순서대로)
        frame = dpt_features[:, i, ...]  # [1, 1369, 256]
        out = self.mamba.forward_single_frame(frame)  # [1, 1369, 256]
        mamba_out.append(out)

    # Reshape back to spatial
    mamba_out = torch.stack(mamba_out, dim=1)  # [1, 5, 1369, 256]
    mamba_out = rearrange(mamba_out, 'b t (h w) c -> (b t) c h w',
                         h=h, w=w)  # [5, 256, 37, 37]

    return mamba_out
```

#### forward_single_frame (핵심!)

```python
# flashdepth/mamba.py:163-202
def forward_single_frame(self, frame, **kwargs):
    """
    프레임 단위 처리 + state 업데이트

    Args:
        frame: [B, L, d_model] 예: [1, 1369, 256]

    Returns:
        output: [B, L, d_model]
    """
    # 1. Mamba blocks 통과
    x = frame.clone()
    for block in self.blocks:
        x = block(x, inference_params=self.inference_params)
        # inference_params: 이전 프레임의 state 저장

    # 2. Final layer (add or modulation)
    if self.mamba_type == 'add':
        x = self.final_layer(x)
        x = x + frame

    # 3. Update sequence offset
    self.inference_params.seqlen_offset += frame.shape[1]
    # 다음 프레임 처리 시 continuation

    return x
```

---

## InferenceParams (State 관리)

### 구조

```python
# flashdepth/mamba.py:10-27
@dataclass
class InferenceParams:
    max_seqlen: int           # 최대 시퀀스 길이 (60000)
    max_batch_size: int       # 배치 크기 (20)
    seqlen_offset: int = 0    # 현재까지 처리한 프레임 수
    key_value_memory_dict: dict = field(default_factory=dict)  # State 저장소
    seq_idx_dict: dict = field(default_factory=dict)
```

### State 저장 흐름

```
Frame 0:
    seqlen_offset = 0
    process frame 0 → update state
    seqlen_offset = 1

Frame 1:
    seqlen_offset = 1
    process frame 1 (using state from frame 0) → update state
    seqlen_offset = 2

Frame 2:
    seqlen_offset = 2
    process frame 2 (using state from frame 1) → update state
    seqlen_offset = 3

...
```

**장점:**
- 이전 프레임 정보를 효율적으로 저장
- O(d_state) 메모리만 사용 (Transformer의 O(T × d_model)보다 작음)
- 긴 비디오도 처리 가능

---

## DPT Layer 삽입 위치

### FlashDepth-L 설정

```python
# configs/flashdepth-l/config.yaml
mamba_in_dpt_layer: [1]  # DPT Layer 1 (path_3)
```

### 4가지 가능한 위치

```python
# flashdepth/original_dpt.py:187-197
path_4 = self.scratch.refinenet4(...)
if 0 in temporal_layer:  # DPT Layer 0
    path_4 = mamba_fn(path_4)

path_3 = self.scratch.refinenet3(path_4, layer_3_rn, ...)
if 1 in temporal_layer:  # DPT Layer 1 ← FlashDepth-L 사용
    path_3 = mamba_fn(path_3)

path_2 = self.scratch.refinenet2(path_3, layer_2_rn, ...)
if 2 in temporal_layer:  # DPT Layer 2
    path_2 = mamba_fn(path_2)

path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
if 3 in temporal_layer:  # DPT Layer 3
    path_1 = mamba_fn(path_1)
```

### 왜 Layer 1 (path_3)?

| DPT Layer | 해상도 | 특징 수준 | Mamba 효과 | 선택 이유 |
|-----------|--------|----------|-----------|----------|
| 0 (path_4) | 18×18 | 최고수준 | 전역 시간 일관성 | 너무 추상적 |
| **1 (path_3)** | **37×37** | **중간-고수준** | **의미+구조 일관성** | **✅ 최적** |
| 2 (path_2) | 74×74 | 중간-저수준 | 지역 시간 일관성 | 계산량 많음 |
| 3 (path_1) | 148×148 | 저수준 | 디테일 일관성 | 계산량 매우 많음 |

**경험적 발견:**
- Layer 1 (path_3): **의미적 일관성** + **구조적 일관성** 균형
- 37×37 해상도: 계산 효율적 (1369 tokens)
- 중간 수준 특징: 물체 형태, 표면 구조 등 시간적으로 안정적인 정보

---

## Mamba의 효과

### 1. 시간적 일관성 (Temporal Consistency)

**문제:**
```
Frame t:   Depth[car] = 10.5m
Frame t+1: Depth[car] = 11.2m
Frame t+2: Depth[car] = 10.8m

→ Flickering (깜빡임 현상)
```

**Mamba 적용 후:**
```
Frame t:   Depth[car] = 10.5m
Frame t+1: Depth[car] = 10.6m  ← Smoothed
Frame t+2: Depth[car] = 10.7m  ← Smoothed

→ 부드러운 변화
```

### 2. 메모리 효율

```python
# Transformer (가정)
Key-Value cache = T × num_layers × d_model × num_heads
                = 50 × 24 × 1024 × 16
                ≈ 19.7M elements ≈ 78.8 MB

# Mamba
State = d_state × num_mamba_layers
      = 256 × 4
      = 1024 elements ≈ 4 KB

Mamba가 약 20,000배 메모리 효율적!
```

### 3. 긴 시퀀스 처리

```python
# test_gear3.py: video_length = 50 frames
# Transformer: O(50²) = 2500 operations
# Mamba: O(50) = 50 operations

50배 효율적!
```

---

## 학습 안정성 (Zero Initialization)

### Final Layer 초기화

```python
# flashdepth/mamba.py:127-134
if mamba_type == 'add':
    self.final_layer = nn.Sequential(
        nn.GELU(),
        nn.Linear(dpt_dim, dpt_dim),
    )
    nn.init.zeros_(self.final_layer[1].weight)
    nn.init.zeros_(self.final_layer[1].bias)
```

**왜 Zero Init?**

```
초기 상태 (weight=0, bias=0):
    output = Mamba(input) + 0 * Linear(input) + 0
           = Mamba(input) + 0
           = Mamba(input)

학습 초기:
    - Mamba의 출력이 0에 가까움 (SSM 초기화)
    - Final layer도 0
    → output ≈ input (identity mapping)

학습 진행:
    - Mamba가 점진적으로 시간 정보 학습
    - Final layer도 점진적으로 조정
    → 안정적인 학습
```

**장점:**
- 학습 초기에 baseline (Mamba 없는 모델)과 동일한 성능
- Gradient flow 안정적
- Residual connection 효과적

---

## Gear3에서의 Mamba

### Gear3 vs FlashDepth Mamba 사용

| 모델 | Mamba 사용 위치 | 목적 |
|------|----------------|------|
| **FlashDepth** | DPT Layer 1 (path_3) | 시간적 일관성 |
| **Gear3** | DPT Layer 1 (path_3) | 시간적 일관성 (동일) |

### Gear3 학습 설정

```python
# train_gear3.py:637-638
elif 'mamba' in name:
    param.requires_grad = True  # Mamba는 처음부터 학습
    mamba_params += param.numel()
```

**차이점:**
- **FlashDepth**: Mamba를 처음부터 학습
- **Gear3**: Mamba를 **처음부터 다시 학습** (pre-trained 없음)
  - 이유: Modulated features를 받기 때문에 입력 분포가 다름
  - Path_1이 modulate되면 path_3의 입력도 영향 받음

---

## 요약

### Mamba의 핵심 특징

1. **State Space Model (SSM) 기반**
   - Selective SSM: 입력 의존적 state transition
   - O(N) 복잡도 (Transformer의 O(N²)보다 효율적)

2. **FlashDepth 통합**
   - DPT Layer 1 (path_3)에 삽입
   - 프레임 단위 처리 (forward_single_frame)
   - InferenceParams로 state 관리

3. **두 가지 모드**
   - Add mode: Residual connection (안정적)
   - Modulation mode: FiLM-style (표현력 강함)

4. **효율성**
   - 메모리: O(d_state) vs Transformer O(T × d_model)
   - 계산: O(T) vs Transformer O(T²)
   - 긴 비디오 처리에 최적

5. **시간적 일관성**
   - Depth flickering 감소
   - 부드러운 프레임 전환
   - 장기 의존성 학습

### Gear3에서의 역할

- Modulated features의 시간적 일관성 유지
- 처음부터 재학습 (modulation과 함께)
- DPT Layer 1에서 의미+구조 일관성 확보

---

## 참고 자료

- [Mamba Paper (Mamba: Linear-Time Sequence Modeling)](https://arxiv.org/abs/2312.00752)
- [Mamba2 Paper](https://arxiv.org/abs/2405.21060)
- [FlashDepth 구현](../flashdepth/mamba.py)
- [Mamba2 SSM 구현](../mamba/mamba_ssm/modules/mamba2.py)
