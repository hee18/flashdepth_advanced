# FlashDepth Gear3: Feature-level Metric Depth Learning

**작성일**: 2025-10-02
**브랜치**: gear3
**목적**: 순수한 특징 기반(feature-based) metric depth 학습을 통한 교통 참여자 깊이 추정 개선

---

## 두 가지 버전

Gear3는 두 가지 버전으로 제공됩니다:

1. **Gear3 (F-L Only)**: 518x518 해상도, FlashDepth-L 기반
   - 파일: `train_gear3.py`, `configs/gear3/`
   - 체크포인트: `configs/flashdepth-l/iter_10001.pth`
   - 사용 케이스: 저해상도, 빠른 학습

2. **Gear3 Hybrid (F-Full)**: 2K 해상도, Teacher-Student Fusion
   - 파일: `train_gear3_hybrid.py`, `configs/gear3-hybrid/`
   - 체크포인트: `configs/flashdepth/iter_43002.pth`
   - 사용 케이스: 고해상도, 최고 정확도

---

## 목차

1. [개요](#개요)
2. [핵심 아이디어](#핵심-아이디어)
3. [아키텍처 설계](#아키텍처-설계)
4. [학습 전략](#학습-전략)
5. [Canonical Space 정규화](#canonical-space-정규화)
6. [손실 함수](#손실-함수)
7. [사용 방법](#사용-방법)
8. [기대 효과](#기대-효과)

---

## 개요

Gear3는 FlashDepth의 metric depth 추정 성능을 향상시키기 위한 새로운 접근법입니다. 기존의 Global Scale Predictor (GSP)가 depth map에 직접 scale/shift를 적용하는 방식과 달리, **DPT feature level에서 metric 정보를 주입**하는 방식을 채택합니다.

### 기존 GSP 방식의 한계

```python
# 기존 GSP 방식
depth_metric = scale * depth_relative + shift
```

- **전역적 보정(global correction)**: 모든 픽셀에 동일한 scale/shift 적용
- 교통 참여자와 배경의 깊이 특성이 다름에도 불구하고 균일하게 처리
- 지역적 보정(local correction)을 시도하면 주변 픽셀과의 관계가 깨질 위험

### Gear3의 접근법

**Feature-wise Linear Modulation (FiLM)**을 사용하여 DPT feature에 직접 metric 정보를 주입:

```python
# Gear3 방식
modulated_feature = gamma ⊙ feature + beta
depth_metric = DPT_Head(modulated_feature)  # scale/shift 없이 직접 출력
```

---

## 핵심 아이디어

### 1. Feature-level Metric Injection

깊이 맵 수준이 아닌 **feature 수준**에서 metric 정보를 주입하면:

- 네트워크가 feature representation 자체를 metric-aware하게 학습
- 교통 참여자와 배경에 대해 서로 다른 modulation 적용 가능
- 주변 픽셀과의 관계를 유지하면서 지역적 보정 가능

### 2. Importance-based Spatial Modulation

모든 픽셀을 동일하게 처리하지 않고, **importance map**을 사용해 foreground/background를 구분:

```python
gamma[x,y] = importance[x,y] × γ_fg + (1 - importance[x,y]) × γ_bg
beta[x,y] = importance[x,y] × β_fg + (1 - importance[x,y]) × β_bg
```

- `importance[x,y] ≈ 1`: 전경(교통 참여자) → FG modulation 강하게 적용
- `importance[x,y] ≈ 0`: 배경 → BG modulation 강하게 적용

### 3. Hierarchical Modulation

DPT의 4개 layer 각각에 대해 독립적인 modulation 적용:

```
layer_1 (high-res) → modulation_1 → path_1
layer_2 (mid-res)  → modulation_2 → path_2
layer_3 (mid-res)  → modulation_3 → path_3
layer_4 (low-res)  → modulation_4 → path_4
```

---

## 아키텍처 설계

### 전체 파이프라인

#### Gear3 (F-L Only) - 518x518 해상도

```
Video Frame → DINOv2-L (frozen) → Patch Tokens + Attention Weights
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
         ImportancePredictor              ForegroundBackgroundNetworks
                    ↓                                   ↓
         Importance Map                    FG Features, BG Features
              [B,1,H,W]                         [B,256]
                    ↓                                   ↓
                    └─────────────────┬─────────────────┘
                                      ↓
                            ModulationNetworks (×4 layers)
                                      ↓
                          γ_fg, β_fg, γ_bg, β_bg
                                      ↓
         DPT-L Features → FeatureModulator → Modulated Features
                                      ↓
                            DPT Refinement + Mamba
                                      ↓
                            Metric Depth (직접 출력)
```

#### Gear3 Hybrid (F-Full) - 2K 해상도

```
2K Image ──────┬──── Downsample (518x518) ──→ Teacher (DINOv2-L + DPT-L, frozen)
               │                                           ↓
               │                                    Teacher path_4
               │                                           ↓
               └──────────────→ Student (DINOv2-S) → Student path_4
                                      ↓                    ↓
                          Patch Tokens + Attention    Cross-Attention
                                      ↓                  (frozen)
                          ImportancePredictor              ↓
                                      ↓               Fused path_4
                              Importance Map              ↓
                                      ↓                    ↓
                          FG/BG Networks ←────────────────┘
                                      ↓
                          ModulationNetworks (×4 layers)
                                      ↓
                          γ_fg, β_fg, γ_bg, β_bg
                                      ↓
         DPT-S Features → FeatureModulator → Modulated Features
                                      ↓
                            DPT Refinement + Mamba
                                      ↓
                            Metric Depth (직접 출력)
```

**Hybrid 핵심 차이점**:
1. **Teacher (F-L)**: 518x518 downsampled image 처리, path_4만 추출, **frozen**
2. **Student (F-S)**: 2K 원본 image 처리, **frozen encoder, trainable head**
3. **Cross-Attention Fusion**: Teacher/Student path_4 결합, **frozen** (사전학습 활용)
4. **Gear3 Modulation**: Fused features에 적용 (fusion 이후 단계)

### 주요 모듈 설명

#### 1. ImportancePredictor

**입력**: DINOv2 attention weights `[B, num_heads, num_patches+1, num_patches+1]`
**출력**: Importance map `[B, 1, patch_h, patch_w]` (0~1 범위)

```python
class ImportancePredictor(nn.Module):
    def __init__(self, num_heads=16, hidden_dim=128):
        self.conv1 = nn.Conv2d(num_heads, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim//2, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, attention_weights, patch_h, patch_w):
        # CLS token의 patch attention 추출
        cls_to_patches = attention_weights[:, :, 0, 1:]  # [B, num_heads, num_patches]
        attn_spatial = cls_to_patches.reshape(B, num_heads, patch_h, patch_w)

        # Conv layers로 importance 예측
        x = self.conv1(attn_spatial)
        x = self.conv2(x)
        importance = self.sigmoid(self.conv3(x))  # [B, 1, patch_h, patch_w]
        return importance
```

**역할**: Attention weights로부터 어떤 영역이 중요한지(전경인지) 학습

#### 2. ForegroundBackgroundNetworks

**입력**: Patch tokens `[B, num_patches, embed_dim]`
**출력**: FG features `[B, 256]`, BG features `[B, 256]`

```python
class ForegroundBackgroundNetworks(nn.Module):
    def __init__(self, embed_dim=1024, feature_dim=256):
        self.fg_net = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        self.bg_net = nn.Sequential(...)  # 동일한 구조

    def forward(self, patch_tokens):
        global_features = patch_tokens.mean(dim=1)  # [B, embed_dim]
        fg_features = self.fg_net(global_features)
        bg_features = self.bg_net(global_features)
        return fg_features, bg_features
```

**역할**: 전경과 배경에 대한 semantic features 생성

#### 3. ModulationNetworks

**입력**: FG/BG features `[B, 256]`, layer_idx (0~3)
**출력**: γ_fg, β_fg, γ_bg, β_bg `[B, dpt_dim]`

```python
class ModulationNetworks(nn.Module):
    def __init__(self, feature_dim=256, dpt_dim=256, num_dpt_layers=4):
        # 각 DPT layer마다 독립적인 modulation network
        self.fg_modulation = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, dpt_dim * 2),
                nn.ReLU(),
                nn.Linear(dpt_dim * 2, dpt_dim * 2)
            ) for _ in range(num_dpt_layers)
        ])
        self.bg_modulation = nn.ModuleList([...])  # 동일

    def forward(self, fg_features, bg_features, layer_idx):
        fg_params = self.fg_modulation[layer_idx](fg_features)  # [B, dpt_dim*2]
        fg_gamma = fg_params[:, :dpt_dim]
        fg_beta = fg_params[:, dpt_dim:]

        bg_params = self.bg_modulation[layer_idx](bg_features)
        bg_gamma = bg_params[:, :dpt_dim]
        bg_beta = bg_params[:, dpt_dim:]

        return fg_gamma, fg_beta, bg_gamma, bg_beta
```

**역할**: Layer별, FG/BG별로 다른 modulation parameters 생성

#### 4. FeatureModulator

**입력**:
- DPT features `[B, C, H, W]`
- Importance map `[B, 1, H', W']`
- γ_fg, β_fg, γ_bg, β_bg `[B, C]`

**출력**: Modulated features `[B, C, H, W]`

```python
class FeatureModulator(nn.Module):
    def forward(self, features, importance_map, fg_gamma, fg_beta, bg_gamma, bg_beta):
        # Importance map을 feature 크기에 맞게 리사이즈
        importance_map = F.interpolate(importance_map, size=features.shape[2:])

        # Spatial-varying modulation parameters
        gamma = importance_map * fg_gamma.view(B,C,1,1) + \
                (1 - importance_map) * bg_gamma.view(B,C,1,1)
        beta = importance_map * fg_beta.view(B,C,1,1) + \
               (1 - importance_map) * bg_beta.view(B,C,1,1)

        # FiLM modulation
        modulated = gamma * features + beta
        return modulated
```

**역할**: 픽셀별로 다른 modulation 적용 (importance 기반)

---

## 학습 전략

### 파라미터 설정

| 모듈 | 학습 여부 | Learning Rate | 파라미터 수 | 비고 |
|------|----------|---------------|------------|------|
| DINOv2 Encoder | ❌ Frozen | - | ~300M | ✓ 사전학습 로드 |
| DPT projects/resize | ❌ Frozen | - | ~5M | ✓ 사전학습 로드 |
| DPT refinenet | ❌ Frozen | - | ~15M | ✓ 사전학습 로드 |
| DPT output_conv1/2 | ✅ **Train from scratch** | 1e-4 | ~0.1M | ❌ 로드 안함 (modulated features 입력) |
| Mamba | ✅ **Train from scratch** | 1e-4 | ~21M | ❌ 로드 안함 (modulated input) |
| ImportancePredictor | ✅ Train | 1e-4 | ~0.3M | 신규 모듈 |
| FG/BG Networks | ✅ Train | 1e-4 | ~0.5M | 신규 모듈 |
| Modulation Networks | ✅ Train | 1e-4 | ~0.5M | 신규 모듈 |

**총 학습 가능 파라미터**: ~22.4M
**새로 추가된 파라미터**: ~1.3M

**핵심 설계 원칙**:
- **Modulated features의 영향을 받는 모듈**은 사전학습 가중치 사용 안 함:
  - `Mamba`: Modulated features를 입력으로 받음
  - `output_conv1/2`: Modulated path_1을 입력으로 받음
- **Modulation 이전 단계**는 사전학습 가중치 활용:
  - `DINOv2`: Feature extraction (modulation 무관)
  - `DPT refinenet`: Skip connection으로만 결합 (modulation 영향 최소)

### 2단계 학습 (Two-Phase Training)

#### Phase 1: 다양한 데이터셋으로 기본 학습

```bash
python train_gear3.py \
  --config-path configs/gear3 \
  phase=1 \
  load=configs/flashdepth-l/iter_60001.pth \
  dataset.data_root=/path/to/data
```

**사용 데이터셋**:
- MVS-Synth
- PointOdyssey
- Spring
- TartanAir
- DynamicReplica

**목적**: 일반적인 feature modulation 능력 학습

#### Phase 2: nuScenes로 자율주행 특화 학습

```bash
python train_gear3.py \
  --config-path configs/gear3 \
  phase=2 \
  load=configs/gear3/best_phase1.pth \
  dataset.data_root=/path/to/data
```

**사용 데이터셋**:
- nuScenes only

**목적**: 교통 참여자에 대한 metric depth 정확도 향상

### 학습률 스케줄링

**Cosine Annealing with Warmup**:

```
Warmup (0-10%): 0.1x → 1.0x (선형 증가)
Stable (10-30%): 1.0x 유지
Decay (30-100%): 1.0x → 0.01x (cosine 감소)
```

**총 반복 횟수**: 60,001 iterations (원본 FlashDepth와 동일, Mamba 처음부터 학습 필요)

---

## Canonical Space 정규화

### 개념

서로 다른 카메라의 focal length를 고려하여 depth를 정규화:

```python
depth_canonical = depth_actual × (focal_canonical / focal_actual)
```

- `focal_canonical = 1000` (고정값)
- `focal_actual`: 각 카메라의 실제 focal length

### 구현

```python
class CanonicalSpaceNormalizer:
    def __init__(self, focal_canonical=1000.0, enable=True):
        self.focal_canonical = focal_canonical
        self.enable = enable

    def canonicalize(self, depth, focal_length):
        """GT depth를 canonical space로 변환 (학습 시)"""
        scale = self.focal_canonical / focal_length
        return depth * scale.view(-1, 1, 1, 1)

    def decanonicalize(self, depth_canonical, focal_length):
        """예측 depth를 실제 metric으로 복원 (평가 시)"""
        scale = focal_length / self.focal_canonical
        return depth_canonical * scale.view(-1, 1, 1, 1)
```

### 학습 파이프라인

```python
# 1. GT를 canonical space로 변환
gt_canonical = canonicalize(gt_depth, focal_length)
gt_metric = 1.0 / (gt_canonical + 1e-8)  # inverse → metric

# 2. 모델 예측 (canonical space에서)
pred_metric_canonical = model(image)

# 3. Loss 계산 (canonical space에서)
loss = InverseDepthLoss(pred_metric_canonical, gt_metric)

# 4. 평가 시 실제 metric으로 복원
pred_metric_actual = decanonicalize(pred_metric_canonical, focal_length)
```

### 장점

1. **스케일 일관성**: 서로 다른 카메라 간 학습 안정성 향상
2. **일반화 성능**: 새로운 카메라 설정에 대한 robustness
3. **토글 가능**: `use_canonical_space: true/false`로 쉽게 활성화/비활성화

---

## 손실 함수

### Inverse Depth Loss

기존 GSP는 depth map에 scale/shift를 적용했지만, Gear3는 feature modulation을 통해 직접 metric depth를 예측하므로 **inverse depth loss**를 사용:

```python
class InverseDepthLoss(nn.Module):
    def __init__(self, inverse_scale=100.0):
        self.inverse_scale = inverse_scale

    def forward(self, pred_depth, gt_depth, valid_mask):
        # Inverse depth로 변환
        pred_inverse = self.inverse_scale / (pred_depth + 1e-8)
        gt_inverse = self.inverse_scale / (gt_depth + 1e-8)

        # L1 loss
        loss = F.l1_loss(pred_inverse, gt_inverse, reduction='none')
        loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        return loss
```

### 왜 Inverse Depth인가?

1. **스케일 불변성(scale-invariance)**: 가까운 물체와 먼 물체의 오차를 공평하게 처리
2. **수치 안정성**: 먼 거리(큰 depth 값)에서 gradient 소실 방지
3. **기존 방식과의 일관성**: FlashDepth의 inverse depth representation과 통일

### 손실 함수 흐름

```
GT Depth (meter) → Canonicalize → GT Metric Canonical
                                        ↓
                                   1.0 / depth
                                        ↓
                               GT Inverse Canonical
                                        ↓
Pred Depth (model output) → 100 / depth → Pred Inverse
                                        ↓
                              L1(pred_inverse, gt_inverse)
```

---

## 사용 방법

### 버전 1: Gear3 (F-L Only) - 518x518

#### Phase 1 학습

```bash
# FlashDepth-L 체크포인트에서 DINOv2 + DPT만 로드
# Mamba, output_conv는 처음부터 학습 (modulated features에 맞게)
python train_gear3.py \
  --config-path configs/gear3 \
  phase=1 \
  load=configs/flashdepth-l/iter_10001.pth \
  dataset.data_root=/data \
  training.batch_size=12 \
  training.iterations=60001
```

#### Phase 2 학습 (nuScenes)

```bash
# Phase 1 best checkpoint에서 시작
python train_gear3.py \
  --config-path configs/gear3 \
  phase=2 \
  load=configs/gear3/best_phase1.pth \
  dataset.data_root=/data
```

### 버전 2: Gear3 Hybrid (F-Full) - 2K

#### Phase 1 학습

```bash
# FlashDepth Full 체크포인트에서 로드
# Teacher, Student, Hybrid Fusion은 freeze
# Mamba, output_conv만 처음부터 학습
python train_gear3_hybrid.py \
  --config-path configs/gear3-hybrid \
  phase=1 \
  load=configs/flashdepth/iter_43002.pth \
  dataset.data_root=/data \
  training.batch_size=4 \
  training.iterations=60001
```

#### Phase 2 학습 (nuScenes)

```bash
# Phase 1 best checkpoint에서 시작
python train_gear3_hybrid.py \
  --config-path configs/gear3-hybrid \
  phase=2 \
  load=configs/gear3-hybrid/best_phase1.pth \
  dataset.data_root=/data
```

### 테스트

```bash
python test_gear3.py \
  --config-path configs/gear3 \
  load=configs/gear3/final_phase2.pth \
  dataset.data_root=/data \
  eval.test_datasets=[tartanair,nuscenes]
```

### 시각화 출력

테스트 시 각 sequence마다 다음을 포함한 시각화 생성:

```
Row 1: Input RGB frames
Row 2: Predicted metric depth (meters)
Row 3: Ground truth metric depth (meters)
Row 4: Importance map (0~1, 전경/배경 구분)
```

저장 위치: `configs/gear3/test_gear3/sequence_XXXX.png`

### 메트릭

테스트 결과는 `test_results.json`으로 저장:

```json
{
  "tae": 0.0234,           // Temporal Alignment Error
  "abs_rel": 0.0567,       // Absolute Relative Error
  "sq_rel": 0.0234,        // Squared Relative Error
  "rmse": 0.234,           // Root Mean Squared Error
  "rmse_log": 0.0456,      // RMSE in log space
  "delta_1": 0.945,        // δ < 1.25
  "delta_2": 0.987,        // δ < 1.25²
  "delta_3": 0.995         // δ < 1.25³
}
```

---

## 기대 효과

### 1. 교통 참여자 깊이 정확도 향상

- **FG/BG 분리**: Importance map 기반으로 전경(차량, 보행자)과 배경을 다르게 처리
- **객체 특화 modulation**: 교통 참여자에 대해 더 정확한 metric scale 학습

### 2. Temporal Consistency 개선

- **Mamba fine-tuning**: 시간적 일관성 유지 능력 향상
- **TAE 감소**: 프레임 간 깊이 변화의 일관성 향상

### 3. 일반화 성능

- **Canonical space**: 다양한 카메라 설정에 대한 robustness
- **Feature-level learning**: Scale/shift보다 더 표현력이 풍부한 학습

### 4. 효율성

- **최소한의 추가 파라미터**: ~1.3M (전체의 0.4%)
- **기존 FlashDepth 재사용**: DINOv2 + DPT는 동결

---

## 파일 구조

```
flashdepth_claude/
├── flashdepth/
│   ├── model.py                  # 기존 FlashDepth 모델
│   ├── gear3_modules.py          # 새로운 Gear3 모듈들
│   └── ...
├── dataloaders/
│   ├── nuscenes_dataset.py       # 새로 추가된 nuScenes 로더
│   └── ...
├── configs/
│   └── gear3/
│       └── config.yaml           # Gear3 설정 파일
├── train_gear3.py                # Gear3 학습 스크립트
├── test_gear3.py                 # Gear3 테스트 스크립트
└── flashdepth_gear3.md           # 이 문서
```

---

## 주요 차이점 요약

| 특성 | 기존 GSP | Gear3 |
|------|---------|-------|
| **Metric 주입 위치** | Depth map (후처리) | DPT features (중간 단계) |
| **보정 방식** | Global scale/shift | Feature-level modulation |
| **공간적 변화** | 없음 (균일) | Importance map 기반 |
| **Foreground/Background** | 구분 없음 | 별도 처리 |
| **학습 가능 파라미터** | ~0.5M (GSP head만) | ~1.3M (Gear3 modules) |
| **Mamba 학습** | Frozen | **Train from scratch (1e-4 LR)** |
| **Canonical space** | 없음 | 지원 (f=1000) |
| **손실 함수** | Depth에 scale/shift 후 loss | Inverse depth loss 직접 |
| **Relative depth** | 생성 (시각화용) | 생성 안함 (feature modulation) |

---

## 디버깅 체크리스트

### 학습 시 확인 사항

1. **파라미터 동결 확인**:
   ```
   Frozen parameters: XXX,XXX,XXX
   Mamba fine-tune parameters: 21,XXX,XXX
   Gear3 trainable parameters: 1,XXX,XXX
   ```

2. **Loss 범위**:
   - 초기 loss: 0.5 ~ 2.0 (정상)
   - 학습 후 loss: 0.1 ~ 0.5 (목표)
   - NaN 발생 시: gradient clipping 확인

3. **Learning rate 스케줄**:
   - Warmup 구간에서 loss 급격히 감소
   - Stable 구간에서 안정적 학습
   - Decay 구간에서 미세 조정

### 테스트 시 확인 사항

1. **Importance map 시각화**:
   - 전경(차량, 보행자)에서 높은 값 (0.7~1.0)
   - 배경(도로, 건물)에서 낮은 값 (0.0~0.3)
   - 경계가 명확하지 않으면 ImportancePredictor 재학습 필요

2. **Metric depth 범위**:
   - 자율주행: 1m ~ 80m (일반적 범위)
   - 이상치 (>200m 또는 <0.1m) 비율 < 1%
   - De-canonicalization이 제대로 적용되었는지 확인

3. **TAE (Temporal Alignment Error)**:
   - Mamba가 제대로 작동하면 TAE < 0.05
   - TAE > 0.1이면 Mamba fine-tuning LR 조정 필요

---

## 향후 개선 방향

### 1. Multi-scale Importance

현재는 단일 importance map을 모든 layer에 사용하지만, layer별로 다른 importance map 사용 가능:

```python
importance_maps = [importance_predictor(attn, layer_idx) for layer_idx in range(4)]
```

### 2. Attention-based Modulation

Importance map 대신 cross-attention으로 직접 modulation:

```python
modulated = CrossAttention(query=dpt_features, key=fg_bg_features, value=fg_bg_features)
```

### 3. 3D Consistency Loss

인접 프레임 간 3D geometric consistency 강제:

```python
loss_3d = GeometricConsistencyLoss(depth_t, depth_t+1, camera_motion)
```

---

## 참고 문헌

1. **FiLM (2018)**: "FiLM: Visual Reasoning with a General Conditioning Layer"
2. **Vanishing Depth (2025)**: Positional Depth Encoding (PDE) 제안
3. **CoL3D (2025)**: Camera intrinsics 기반 feature modulation
4. **FlashDepth (2024)**: 기본 아키텍처 (DINOv2 + DPT + Mamba)

---

## 연락처 및 기여

- **Branch**: gear3
- **개발자**: hsy
- **이슈 리포트**: GitHub Issues 또는 직접 연락

---

**마지막 업데이트**: 2025-10-02
