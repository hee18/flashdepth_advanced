# GSP Fusion Strategy: Concat vs Mix

**핵심 발견:** GSP는 concat 사용, FFM은 mix 사용 → 일관성 없음!

**제안:** GSP도 FFM처럼 mix로 통일

---

## Current Implementation

### GSP: Concatenation
```python
# flashdepth/gear5_modules.py:88
cls_concat = torch.cat(cls_tokens_list, dim=-1)  # [B, 4*1024] = [B, 4096]

# MLP
Linear(4096 → 1024)  # 4,194,304 params
Linear(1024 → 256)   #   262,144 params
Linear(256 → 2)      #       512 params
# Total: 4,456,960 params
```

### FFM: Weighted Average (Mix)
```python
# flashdepth/gear3_upgrade_modules.py:252
importance_stack = torch.stack(importance_maps, dim=1)  # [B, 4, H, W]
weights_norm = torch.softmax(fusion_weights, dim=0)     # [4] learnable
importance_fused = (importance_stack * weights_norm.view(1, -1, 1, 1)).sum(dim=1)

# Or uniform weights (Gear5 default):
weights = torch.ones(4) / 4  # [0.25, 0.25, 0.25, 0.25]
```

**문제:** 왜 GSP는 concat, FFM은 mix? 🤔

---

## Proposed: GSP with Mix (동일 비율)

### Option A: Uniform Weights (가장 단순)

```python
class GlobalScalePredictorUniformMix(nn.Module):
    def __init__(self, embed_dim=1024, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Uniform weights: [0.25, 0.25, 0.25, 0.25]
        uniform_weights = torch.ones(num_layers) / num_layers
        self.register_buffer('fusion_weights', uniform_weights)

        # MLP: 훨씬 작아짐!
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, 512),      # 1024→512 (524,288 params)
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),            # 512→256 (131,072 params)
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)               # 256→2 (512 params)
        )
        # Total: 655,872 params (-85.3% from concat!)

    def forward(self, cls_tokens_list):
        # Stack: [B, num_layers, embed_dim]
        cls_stack = torch.stack(cls_tokens_list, dim=1)  # [B, 4, 1024]

        # Weighted average: [B, 4, 1024] → [B, 1024]
        cls_fused = (cls_stack * self.fusion_weights.view(1, -1, 1)).sum(dim=1)

        # Predict scale and shift
        params = self.predictor(cls_fused)  # [B, 2]
        scale = F.softplus(params[:, 0])
        shift = params[:, 1]

        return scale, shift
```

### Option B: Learnable Weights

```python
class GlobalScalePredictorLearnableMix(nn.Module):
    def __init__(self, embed_dim=1024, num_layers=4):
        super().__init__()

        # Learnable fusion weights (like FFM)
        init_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Favor later layers
        self.fusion_weights = nn.Parameter(init_weights)

        # Same MLP as Option A
        self.predictor = nn.Sequential(...)
        # Total: 655,876 params (+4 for fusion weights)

    def forward(self, cls_tokens_list):
        cls_stack = torch.stack(cls_tokens_list, dim=1)

        # Normalize weights with softmax
        weights_norm = torch.softmax(self.fusion_weights, dim=0)

        # Weighted average
        cls_fused = (cls_stack * weights_norm.view(1, -1, 1)).sum(dim=1)

        params = self.predictor(cls_fused)
        scale = F.softplus(params[:, 0])
        shift = params[:, 1]

        return scale, shift
```

---

## Comparison Table

| Method | Input Dim | Params | Reduction | Fusion | Consistency |
|--------|-----------|--------|-----------|--------|-------------|
| **Current (Concat)** | 4096 | 4.46M | Baseline | Concat | ❌ (FFM과 다름) |
| **Option A (Uniform Mix)** | 1024 | 0.66M | **-85.3%** | Average | ✅ (FFM과 동일) |
| **Option B (Learnable Mix)** | 1024 | 0.66M | **-85.3%** | Weighted | ✅ (FFM과 동일) |

---

## Detailed Analysis

### 1. Parameter Reduction

**Current (Concat):**
```
4096 × 1024 = 4,194,304  (first layer)
1024 × 256  =   262,144  (second layer)
256  × 2    =       512  (output)
──────────────────────
Total       = 4,456,960
```

**Proposed (Mix):**
```
1024 × 512  =   524,288  (first layer)
512  × 256  =   131,072  (second layer)
256  × 2    =       512  (output)
──────────────────────
Total       =   655,872  (-85.3%!)
Learnable   =         4  (fusion weights, optional)
```

**Savings: 3.8M parameters!**

### 2. Conceptual Difference

**Concatenation:**
```
[CLS_4, CLS_11, CLS_17, CLS_23] → [1024, 1024, 1024, 1024] → concat → [4096]
```
- 각 layer의 정보를 **모두 유지**
- MLP가 layer 간 관계를 학습해야 함
- 첫 layer가 매우 무거움 (4096→1024)

**Mix (Weighted Average):**
```
[CLS_4, CLS_11, CLS_17, CLS_23] → weighted sum → [1024]
```
- 각 layer의 정보를 **융합**
- Layer importance는 weights로 표현
- 차원 유지 (1024 → 1024)

### 3. Information Preservation

**Question:** Mix하면 정보가 손실되지 않나?

**Answer:** Scale/shift prediction에서는 문제 없음!

**이유:**
1. **Global statistics 추정**: 각 CLS token은 이미 global scene info 포함
   - CLS_4: Edge density, texture patterns
   - CLS_11: Object parts, local geometry
   - CLS_17: Object instances, relative depth
   - CLS_23: Scene layout, absolute scale

2. **Complementary information**: 각 layer가 다른 측면 포착
   - Mix = "모든 layer의 종합적 판단"
   - Concat = "모든 layer를 나열하고 MLP가 선택"

3. **Empirical evidence**: FFM에서 mix가 잘 작동
   - Importance map fusion (4 layers → 1 map)
   - Uniform weights도 충분히 좋은 성능

**결론:** Scale/shift는 global prediction이므로 mix가 더 적합!

### 4. Training Dynamics

**Concat (Current):**
- ❌ 매우 큰 첫 layer (4096→1024)
- ❌ Gradient bottleneck 가능
- ❌ Overfitting 위험 (많은 params)
- ❌ 초기화 중요 (큰 matrix)

**Mix (Proposed):**
- ✅ 균형잡힌 layer sizes
- ✅ Stable gradients
- ✅ Less overfitting (적은 params)
- ✅ 초기화 덜 중요

### 5. Speed Impact

**FLOPs Comparison:**

Current (Concat):
```
Concat: 0 FLOPs (just memory copy)
Linear(4096→1024): 4096 × 1024 × B = 4.19M × B
Linear(1024→256):  1024 × 256 × B  = 0.26M × B
Linear(256→2):     256 × 2 × B     = 0.0005M × B
────────────────────────────────────────────
Total: 4.45M × B FLOPs
```

Proposed (Mix):
```
Stack: 0 FLOPs (view operation)
Weighted sum: 4 × 1024 × B = 0.004M × B
Linear(1024→512):  1024 × 512 × B = 0.52M × B
Linear(512→256):   512 × 256 × B  = 0.13M × B
Linear(256→2):     256 × 2 × B    = 0.0005M × B
────────────────────────────────────────────
Total: 0.65M × B FLOPs
```

**FLOPs Reduction: -85.4%!**

**실제 속도:**
- GSP는 전체 inference의 <5%
- -85% FLOPs → 실제 FPS 변화: ~+1 FPS
- **예상**: 38.5 → 39.5 FPS

---

## Expected Performance

### Will mix lose accuracy?

**No! 오히려 향상될 가능성:**

**1. Regularization effect**
- 적은 params → overfitting 감소
- Validation generalization 향상

**2. Inductive bias**
- Mix = "모든 layer를 동등하게 고려"
- Concat = "MLP가 알아서 선택" (더 어려움)

**3. Empirical evidence from FFM**
```python
# FFM에서 uniform weights (Gear5 default):
uniform_weights = torch.ones(4) / 4  # 동일 비율

# 이미 잘 작동하고 있음!
# → GSP에도 적용 가능
```

### Expected Metric Change

**Conservative estimate (no performance loss):**
- AbsRel: ±0% (동일)
- δ1: ±0% (동일)
- RMSE: ±0% (동일)

**Optimistic estimate (regularization benefit):**
- AbsRel: -2~5% (향상)
- δ1: +0.5~1% (향상)
- RMSE: -1~3% (향상)

**Worst case (information loss):**
- AbsRel: +3~5% (약간 저하)
- δ1: -1~2% (약간 저하)
- RMSE: +2~4% (약간 저하)

**Most likely: Conservative estimate** (성능 유지)

---

## Recommendation

### ⭐ **Strongly Recommend: Option A (Uniform Mix)**

**Reasons:**
1. ✅ **FFM과 일관성**: 같은 fusion strategy
2. ✅ **파라미터 85% 감소**: 4.46M → 0.66M
3. ✅ **속도 향상**: +1 FPS 예상
4. ✅ **단순성**: No hyperparameters to tune
5. ✅ **안정성**: 검증된 방법 (FFM에서 사용)
6. ✅ **성능 유지 예상**: Scale/shift는 global prediction

### Implementation Priority

**Immediate action:**
1. Implement Option A (Uniform Mix)
2. Train for 5K steps (quick validation)
3. Compare with current concat version

**If Option A works well:**
- Use as new baseline
- Document 85% param reduction
- Mention in paper

**If Option A shows performance drop:**
- Try Option B (Learnable Mix)
- If still worse, revert to concat

---

## Code Implementation

### New Module

```python
# flashdepth/gear5_modules.py

class GlobalScalePredictorUniformMix(nn.Module):
    """
    Predict global scale and shift from multi-layer CLS tokens using uniform mix.

    This is more consistent with FFM (which uses weighted average)
    and much more parameter-efficient than concatenation.

    Architecture:
        Stack [CLS_4, CLS_11, CLS_17, CLS_23] → [B, 4, 1024]
        ↓
        Uniform Average → [B, 1024]
        ↓
        MLP: 1024 → 512 → 256 → 2
        ↓
        [scale (Softplus), shift]
    """
    def __init__(self, embed_dim=1024, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Uniform fusion weights (fixed, non-trainable)
        uniform_weights = torch.ones(num_layers) / num_layers
        self.register_buffer('fusion_weights', uniform_weights)

        # Lightweight MLP
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)  # [scale, shift]
        )

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"GlobalScalePredictorUniformMix: {total_params:,} parameters")
        logging.info(f"  Input: {num_layers} CLS tokens × {embed_dim} (uniform mix)")
        logging.info(f"  Reduction: -85.3% vs. concat version")

    def forward(self, cls_tokens_list):
        """
        Args:
            cls_tokens_list: List of [B, embed_dim] CLS tokens
                            [CLS_4, CLS_11, CLS_17, CLS_23]

        Returns:
            scale: [B] - positive scale factor
            shift: [B] - shift value (any)
        """
        # Stack CLS tokens: [B, num_layers, embed_dim]
        cls_stack = torch.stack(cls_tokens_list, dim=1)  # [B, 4, 1024]

        # Uniform weighted average: [B, 4, 1024] → [B, 1024]
        cls_fused = (cls_stack * self.fusion_weights.view(1, -1, 1)).sum(dim=1)

        # Predict scale and shift
        params = self.predictor(cls_fused)  # [B, 2]

        # Ensure positive scale with Softplus
        scale = F.softplus(params[:, 0])  # [B]
        shift = params[:, 1]  # [B]

        return scale, shift
```

### Update Gear5MetricHead

```python
class Gear5MetricHead(nn.Module):
    def __init__(self, embed_dim=1024, dpt_dim=256, use_mix_fusion=True):
        super().__init__()

        if use_mix_fusion:
            # New: Uniform mix (85% fewer params)
            self.global_gsp = GlobalScalePredictorUniformMix(
                embed_dim=embed_dim, num_layers=4
            )
        else:
            # Legacy: Concat (for comparison)
            self.global_gsp = GlobalScalePredictorMultiLayer(
                embed_dim=embed_dim, num_layers=4
            )

        # Rest of the code unchanged...
```

### Config Update

```yaml
# configs/gear5/config.yaml

model:
  use_mix_fusion: true  # true: uniform mix (default), false: concat (legacy)
```

---

## Ablation Study

### Comparison Matrix

| Variant | Params | Expected Perf | Training Time | Use Case |
|---------|--------|--------------|---------------|----------|
| **Concat (Current)** | 4.46M | Baseline | Baseline | Baseline |
| **Uniform Mix** | 0.66M (-85%) | ±0% | -10% | **Recommended** |
| **Learnable Mix** | 0.66M (-85%) | +1~2% | -10% | If uniform fails |
| **Attention** | 6.5M (+46%) | +15~25% | +11% | Max performance |

### Test Protocol

```bash
# 1. Train uniform mix for 5K steps
CUDA_VISIBLE_DEVICES=0 python train_gear5.py \
  --config configs/gear5/ablation_uniform_mix.yaml \
  training.iterations=5000 \
  model.use_mix_fusion=true

# 2. Compare with current concat (5K steps from checkpoint)
CUDA_VISIBLE_DEVICES=0 python train_gear5.py \
  --config configs/gear5/ablation_concat.yaml \
  training.iterations=5000 \
  model.use_mix_fusion=false

# 3. Test both on validation sets
python test_gear5.py --checkpoint ablation/uniform_mix/best.pth
python test_gear5.py --checkpoint ablation/concat/best.pth

# 4. Compare metrics
python compare_results.py \
  --baseline ablation/concat/test_results.json \
  --variant ablation/uniform_mix/test_results.json
```

---

## Conclusion

### 핵심 발견
- GSP는 concat 사용 (4.46M params)
- FFM은 mix 사용 (일관성 없음!)

### 제안
**GSP도 uniform mix로 변경**

### 이유
1. ✅ FFM과 일관성
2. ✅ 85% 파라미터 감소
3. ✅ 속도 향상
4. ✅ 성능 유지 예상
5. ✅ 단순성

### Next Steps
1. Implement uniform mix version
2. Quick validation (5K steps)
3. If successful → adopt as default
4. If issues → try learnable mix

---

**Document Version**: 1.0
**Date**: 2025-11-10
**Status**: Ready for implementation
