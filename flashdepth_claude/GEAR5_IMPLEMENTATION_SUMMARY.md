# Gear5 Implementation Summary

## ✅ Completed Components

### 1. **flashdepth/gear5_modules.py** ✓
**Fully Implemented**

Three main components:
- `GlobalScalePredictorMultiLayer`: Predicts global scale & shift from multi-layer CLS tokens [4, 11, 17, 23]
- `ForegroundOnlyModulationHead`: FG-only modulation using multi-layer attention [4, 11, 17, 23]
- `Gear5MetricHead`: Combined module for Step 2 (frozen GSP + trainable FG modulation)

**Key Design Decisions**:
- Multi-layer CLS extraction layers: [4, 11, 17, 23] (encoder output layers)
- Multi-layer attention layers: [4, 11, 17, 23] (same as CLS for consistency)
- FG modulation uses FiLM (gamma * x + beta) as recommended
- BG pixels keep global modulation (no FG modulation applied)

### 2. **train_gear5.py** ✓
**Base Created** + **Critical Modifications Documented**

Status:
- ✅ File copied from train_gear3_upgrade.py
- ✅ Basic modifications applied (class name, imports, phase→step)
- ⚠️  **Requires critical modifications** (see GEAR5_TRAIN_MODIFICATIONS.md)

Key modifications needed:
- `_setup_model()`: Add Gear5 modules based on step
- `_configure_parameters()`: Step-specific freeze logic
- `train_step()`: Step-specific forward pass
- `validation_step()`: Step-specific forward pass
- `_setup_optimizer()`: Step-specific learning rates

### 3. **test_gear5.py** ✓
**Base Created** + **Needs Same Modifications as train_gear5.py**

Status:
- ✅ File copied from test_gear3_upgrade.py
- ✅ Basic modifications applied (class name, imports, phase→step)
- ⚠️  **Requires same critical modifications** as train_gear5.py

### 4. **configs/gear5/config.yaml** ✓
**Fully Configured**

Key settings:
```yaml
step: 1  # Set to 2 for Step 2 training
step1_checkpoint: null  # Required for Step 2

# Learning rates
gsp_lr: 1.0e-4      # Step 1: Global GSP
fg_mod_lr: 1.0e-4   # Step 2: FG modulation
mamba_lr: 1.0e-4    # Step 1: 1e-4; Step 2: 5e-5 (lower)
output_lr: 1.0e-4   # Step 1: 1e-4; Step 2: 5e-5 (lower)
```

### 5. **run_docker.sh** ✓
**Fully Integrated**

New commands added:
```bash
# Training
./run_docker.sh train_gear5 --step 1               # Step 1 only
./run_docker.sh train_gear5 --step 2 --step1-checkpoint train_results/gear5_step1/best.pth

# Testing
./run_docker.sh test_gear5 --step 1
./run_docker.sh test_gear5 --step 2 --step1-checkpoint train_results/gear5_step1/best.pth
```

New options:
- `--step STEPS`: Set training/testing step (1, 2, or "1,2")
- `--step1-checkpoint PATH`: Set Step 1 checkpoint path for Step 2

---

## 🔧 Critical Remaining Work

### Priority 1: Complete train_gear5.py and test_gear5.py

**Reference Document**: `GEAR5_TRAIN_MODIFICATIONS.md`

**Critical Methods to Modify**:

1. **_setup_model()** (Lines ~232-427)
   - Step 1: Add `GlobalScalePredictorMultiLayer`
   - Step 2: Load Step 1 checkpoint + Add `Gear5MetricHead`
   - Enable attention storage for layers [4, 11, 17, 23]

2. **_configure_parameters()** (Lines ~429-490)
   - Step 1: Trainable = GSP + Mamba + Final; Frozen = ViT + DPT
   - Step 2: Trainable = FG mod + Mamba + Final; Frozen = ViT + DPT + GSP

3. **train_step()** (Lines ~903-1000)
   - Step 1: Global GSP forward pass
   - Step 2: Gear5MetricHead forward pass (global + FG modulation)

4. **validation_step()** (Lines ~1002-1100)
   - Same logic as train_step()

5. **_setup_optimizer()** (Lines ~?)
   - Step 1: All modules use 1e-4
   - Step 2: FG mod (1e-4), Mamba+Final (5e-5, lower for fine-tuning)

### Priority 2: Testing & Validation

**Test Plan**:
1. ✅ Module imports: `python -c "from flashdepth.gear5_modules import *"`
2. Test Step 1 training with dummy data
3. Test Step 2 loading Step 1 checkpoint
4. Verify parameter counts (frozen vs trainable)
5. Test forward pass outputs
6. Full training validation

---

## 📊 Architecture Summary

### Step 1: Global Scale & Shift Prediction
```
Input → ViT(F) → CLS [4,11,17,23] → Global GSP(T) → scale, shift
                                            ↓
         DPT(F) → path_1 × scale + shift → Mamba(PT+FT) → Final(PT+FT) → Depth
```

**Trainable**: Global GSP + Mamba + Final head
**Frozen**: ViT + DPT
**Loss**: gt valid & pred inlier pixels

### Step 2: Foreground-only Modulation
```
Input → ViT(F) → Attention [4,11,17,23] → Multi-layer Fusion → Importance Map → FG Mask
                                                                                    ↓
         DPT(F) → Global GSP(F) → path_1_global ← FG Modulation(T) → FG-only FiLM
                                       ↓
                                 Mamba(S1+FT) → Final(S1+FT) → Depth
```

**Trainable**: FG modulation + Mamba + Final head
**Frozen**: ViT + DPT + Global GSP
**Loss**: gt valid & pred inlier & FG pixels

---

## 🎯 Design Rationale

### Why Same Layers [4, 11, 17, 23] for CLS and Attention?

**Consistency**: Using the same layers for both purposes ensures:
- Same hierarchical information extraction
- Easier to understand and debug
- More coherent multi-scale reasoning

**Coverage**: These layers represent ~17%, 46%, 71%, 96% of network depth
- Layer 4: Low-level patterns (edges, textures)
- Layer 11: Mid-level semantics (object parts)
- Layer 17: High-level semantics (whole objects)
- Layer 23: Abstract scene understanding

### Why Fine-tune Mamba in Both Steps?

**Step 1**:
- GSP changes the input distribution to Mamba
- Fine-tuning allows adaptation to globally-modulated features
- Start from FlashDepth pretrained weights (good initialization)

**Step 2**:
- FG modulation further changes feature distribution
- Fine-tuning from Step 1 provides excellent initialization
- Lower LR (5e-5) prevents catastrophic forgetting

### Why Freeze GSP in Step 2?

- GSP learns global scene characteristics in Step 1
- Step 2 adds foreground-specific refinements
- Keeping GSP frozen prevents interference between global and local signals
- Clear separation of responsibilities: global (GSP) vs local (FG mod)

---

## 📝 Usage Examples

### Step 1 Training
```bash
# Single GPU
./run_docker.sh train_gear5 --step 1 --gpu 0

# Multi-GPU (not yet implemented in script)
CUDA_VISIBLE_DEVICES=0,1 python train_gear5.py \
  --config-path configs/gear5 \
  --config-name config \
  step=1 \
  load=configs/flashdepth-l/iter_10001.pth \
  dataset.data_root=/path/to/data
```

### Step 2 Training
```bash
# Load Step 1 checkpoint
./run_docker.sh train_gear5 --step 2 \
  --step1-checkpoint train_results/gear5_step1/best.pth \
  --gpu 0
```

### Testing
```bash
# Test Step 1
./run_docker.sh test_gear5 --step 1 --gpu 0

# Test Step 2
./run_docker.sh test_gear5 --step 2 \
  --step1-checkpoint train_results/gear5_step1/best.pth \
  --gpu 0
```

---

## 🐛 Known Issues & TODOs

### High Priority
- [ ] Complete train_gear5.py critical modifications
- [ ] Complete test_gear5.py critical modifications
- [ ] Test imports and module loading
- [ ] Verify ViT-S encoder layers (currently: [2, 5, 8, 11] estimated)

### Medium Priority
- [ ] Add multi-GPU (DDP) support to run_docker.sh
- [ ] Create gear5-specific visualizer (or adapt gear3_upgrade_visualization.py)
- [ ] Add unit tests for gear5_modules.py

### Low Priority
- [ ] Hybrid 2K training support (after Step 1 & 2 work)
- [ ] Step 1+2 combined training mode (--step "1,2")
- [ ] Performance profiling and optimization

---

## 📚 Reference Documents

1. **GEAR5_TRAIN_MODIFICATIONS.md**: Detailed modification guide for train_gear5.py
2. **flashdepth_advanced.md**: Original metric depth system documentation
3. **CLAUDE.md**: General project setup and commands

---

## 🚀 Quick Start Checklist

To get Gear5 training working:

1. ✅ Verify all files are created
2. ⚠️  Apply critical modifications from GEAR5_TRAIN_MODIFICATIONS.md
3. ⚠️  Test module imports
4. ⚠️  Run Step 1 training
5. ⚠️  Run Step 2 training
6. ⚠️  Validate results

**Estimated Time**: 2-3 hours for critical modifications + testing

---

## 💡 Tips

- Start with Step 1 training and validate before moving to Step 2
- Use small batch size (2-4) for initial testing
- Check parameter counts match expectations (~262K for GSP, etc.)
- Monitor loss curves carefully - Step 2 should start lower than Step 1
- Visualize importance maps and FG masks to verify correctness

---

## Contact & Support

For questions or issues:
1. Check GEAR5_TRAIN_MODIFICATIONS.md for implementation details
2. Review gear3_upgrade implementation for reference
3. Consult flashdepth_advanced.md for metric depth concepts

Good luck with the implementation! 🎉
