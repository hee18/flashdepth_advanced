# Remaining Modifications TODO

This file tracks the remaining modifications needed to align with original FlashDepth training approach.

## Completed ✅
1. ✅ `dataloaders/nuscenes_dataset.py` - Fixed invalid pixel marking (-1)
2. ✅ `dataloaders/combined_dataset.py` - Added fx_actual_original return value
3. ✅ `train_gear5.py` - Updated batch unpacking and valid mask logic

## Remaining Tasks

### 1. train_gear2.py - Batch Unpacking and Valid Mask

**Locations to modify**: Lines 663, 821, 1018 (batch unpacking)

#### A. Batch Unpacking (3 locations)

**Line 663 (visualization section)**:
```python
# BEFORE
images, gt_depth, focal_lengths, dataset_idx = batch

# AFTER
images, gt_depth, focal_lengths_canonical, focal_lengths_actual, actual_valid_masks, dataset_idx = batch
```

**Line 821 (train_step method)**:
```python
# BEFORE
images, gt_depth, focal_lengths, dataset_idx = batch
images = images.to(self.device)
gt_depth = gt_depth.to(self.device)
focal_lengths = focal_lengths.to(self.device)

# AFTER
images, gt_depth, focal_lengths_canonical, focal_lengths_actual, actual_valid_masks, dataset_idx = batch
images = images.to(self.device)
gt_depth = gt_depth.to(self.device)
focal_lengths_canonical = focal_lengths_canonical.to(self.device)
focal_lengths_actual = focal_lengths_actual.to(self.device)
actual_valid_masks = actual_valid_masks.to(self.device)
```

**Line 1018 (validation section)**:
```python
# BEFORE
images, gt_depth, focal_lengths, dataset_idx = batch

# AFTER
images, gt_depth, focal_lengths_canonical, focal_lengths_actual, actual_valid_masks, dataset_idx = batch
```

#### B. Train Step Valid Mask Logic

**Search for**: "Compute valid mask" or "gt_valid_mask" in train_step method

**BEFORE** (likely around line 850-870):
```python
# GT valid mask: Use actual space mask from dataloader (<70m in actual space)
gt_valid_mask = actual_valid_mask.view(B * T, H, W).bool()

# Pred outlier mask: filter extreme predictions
MAX_DEPTH_OUTLIER = 200.0
MIN_INVERSE_OUTLIER = 100.0 / MAX_DEPTH_OUTLIER
pred_outlier_mask = (pred_depth_inverse_flat > MIN_INVERSE_OUTLIER)

# Final mask
valid_mask = gt_valid_mask & pred_outlier_mask
```

**AFTER**:
```python
# Follow original FlashDepth: use ONLY GT valid mask (no 70m threshold, no pred check)
# GT depth is already scaled to 100/m (inverse depth: 100/m)
# Invalid pixels are marked as -1, which becomes -100 after scaling
valid_mask = (gt_depth_inverse_flat >= 0)  # Exclude only invalid pixels (originally -1)
```

#### C. Validation Valid Mask Logic

**Search for**: "Compute loss for entire sequence" or validation valid_mask computation

**BEFORE** (likely around line 1100-1120):
```python
# GT valid mask: Use actual space mask
H_gt, W_gt = gt_shape
gt_valid_mask = actual_valid_mask.view(B_orig * T_orig, 1, H_gt, W_gt).bool()

# Pred valid mask
MIN_INVERSE_DEPTH = 100.0 / 70.0
pred_valid_mask = (pred_depth_inverse >= MIN_INVERSE_DEPTH)

# Final mask
valid_mask = (gt_valid_mask & pred_valid_mask).float()
```

**AFTER**:
```python
# Follow original FlashDepth: use ONLY GT valid mask
valid_mask = (gt_depth_inverse_flat >= 0).float()  # Exclude only invalid pixels (originally -1)
```

#### D. Update References to focal_lengths

**Search and replace** throughout the file:
- References to `focal_lengths` in logging → update to `focal_lengths_canonical`
- Add logging for `focal_lengths_actual` where useful (optional)

---

### 2. train_gear3.py - Same Changes as train_gear2.py

**Apply identical modifications as train_gear2.py**

**Batch unpacking locations** (use Grep to find):
```bash
grep -n "images, gt_depth, focal_lengths.*= batch" train_gear3.py
```

**Follow the same pattern**:
1. Update batch unpacking (3+ locations)
2. Update train_step valid_mask logic
3. Update validation valid_mask logic
4. Update focal_lengths references

---

### 3. utils/gear3_visualization.py - CANONICAL_FX and Display

#### A. Fix CANONICAL_FX Hard-coded Value

**Line 401**:
```python
# BEFORE
CANONICAL_FX = 1000.0  # ❌ Wrong value

# AFTER
CANONICAL_FX = 500.0  # ✅ Correct value (matches dataset_intrinsics.py)
```

#### B. Update Canon Valid Display

**Lines 392-410** (search for "resized_fx"):

**BEFORE**:
```python
# Actual space equivalent: depth_actual = 70 × (fx_actual / CANONICAL_FX)
valid_gt_max_actual = 70.0 * (fx_value / CANONICAL_FX)
ax9.text(0.05, y_pos,
        f'resized_fx: {fx_value:.1f}, valid: 70.0m (canon={valid_gt_max_actual:.1f}m actual)',
        fontsize=10, transform=ax9.transAxes,
        bbox=dict(boxstyle="round", facecolor='lightyellow'))
```

**AFTER**:
```python
# Get original fx_actual from model_outputs
if 'fx_actual_original' in model_outputs and torch.is_tensor(model_outputs['fx_actual_original']):
    fx_actual_original = model_outputs['fx_actual_original'][0, 0].item()
else:
    fx_actual_original = fx_value  # Fallback to canonical fx

# Canon space equivalent: depth_canon = 70 × (CANONICAL_FX / fx_actual_original)
# Example: fx_actual=1000 → actual 70m = canon 35m
canon_valid_max = 70.0 * (CANONICAL_FX / fx_actual_original)
ax9.text(0.05, y_pos,
        f'resized_fx: {fx_value:.1f}, canon_valid={canon_valid_max:.1f}m (actual 70m)',
        fontsize=10, transform=ax9.transAxes,
        bbox=dict(boxstyle="round", facecolor='lightyellow'))
```

#### C. Pass fx_actual_original to Visualizer

**In train_gear5.py, train_gear2.py, train_gear3.py** - when calling visualizer:

**Add to model_outputs dict**:
```python
model_outputs = {
    'pred_depth': pred_depth,
    'gt_depth': gt_depth,
    'importance': importance,
    'focal_lengths': focal_lengths_canonical,
    'fx_actual_original': focal_lengths_actual,  # NEW
    # ... other outputs
}
```

**Note**: Visualization calls might be in different formats. Search for:
- `train_visualizer.visualize_batch`
- `val_visualizer.visualize_batch`
- `self.train_visualizer`
- `self.val_visualizer`

#### D. Optional: Update Valid Mask Computation in Visualization

**Lines 110-195** (search for "canonical_gt_valid"):

This section computes valid masks for visualization. Consider updating to match new philosophy:

**CURRENT**: Uses 70m threshold from `canonical_gt_valid` / `canonical_pred_valid`

**OPTION 1** (Keep as-is): Use visualization masks passed from training (actual_valid_mask at 70m)
**OPTION 2** (Simplify): Remove 70m threshold, only filter invalid pixels

**Recommendation**: Keep current approach but ensure masks come from actual_valid_mask (70m in actual space).

---

### 4. train_gear3_upgrade.py - Canonical Focal Length

#### A. Fix _get_canonical_focal_length Method

**Lines 200-205** (search for "_get_canonical_focal_length"):

**BEFORE**:
```python
def _get_canonical_focal_length(self):
    """
    Get canonical focal length (fixed at 1000.0 for all resolutions).

    Returns:
        float: Canonical focal length (always 1000.0)
    """
    return 1000.0  # ❌ Inconsistent with gear5
```

**AFTER**:
```python
def _get_canonical_focal_length(self):
    """
    Get canonical focal length (fixed at 500.0 for all resolutions).
    Matches gear5 and dataset_intrinsics.py CANONICAL_FOCAL_LENGTH.

    Returns:
        float: Canonical focal length (always 500.0)
    """
    return 500.0  # ✅ Consistent with gear5
```

#### B. Update Any Hard-coded 1000.0 References

**Search for**: `1000.0` or `CANONICAL_FX.*1000`

**Replace with**: `500.0` or use `self._get_canonical_focal_length()`

---

## Verification Checklist

After completing all modifications:

### 1. Grep Checks
```bash
# Check for old batch unpacking (should find none in modified files)
grep -n "images, gt_depth, focal_lengths, dataset_idx = batch" train_gear2.py train_gear3.py

# Check for 70m threshold in training (should find none in train_step)
grep -n "actual_valid_mask.*70" train_gear2.py train_gear3.py

# Check for canonical FX consistency
grep -n "CANONICAL_FX.*1000" train_gear3_upgrade.py utils/gear3_visualization.py
```

### 2. Manual Verification

**train_gear2.py**:
- [ ] All batch unpacking updated (3+ locations)
- [ ] train_step uses `valid_mask = (gt_depth_inverse_flat >= 0)`
- [ ] validation uses `valid_mask = (gt_depth_inverse_flat >= 0).float()`
- [ ] focal_lengths → focal_lengths_canonical references updated

**train_gear3.py**:
- [ ] Same checks as train_gear2.py

**utils/gear3_visualization.py**:
- [ ] CANONICAL_FX = 500.0
- [ ] Display shows `canon_valid=XXm (actual 70m)`
- [ ] Uses `fx_actual_original` from model_outputs

**train_gear3_upgrade.py**:
- [ ] `_get_canonical_focal_length()` returns 500.0
- [ ] No hard-coded 1000.0 references

### 3. Test Run

```bash
# Test train_gear2
CUDA_VISIBLE_DEVICES=1 python train_gear2.py --config-path configs/gear2 \
  training.iterations=1 dataset.data_root=/home/cvlab/hsy/Datasets

# Test train_gear3
CUDA_VISIBLE_DEVICES=1 python train_gear3.py --config-path configs/gear3 \
  training.iterations=1 dataset.data_root=/home/cvlab/hsy/Datasets
```

**Expected outcome**:
- No batch unpacking errors
- No shape mismatch errors
- Validation completes without errors
- Visualizations display correct focal length info

---

## Summary of Changes

### Philosophy Change
- **OLD**: Training uses 70m threshold (actual space) + pred outlier mask (200m canonical)
- **NEW**: Training uses ONLY GT valid mask (invalid pixels = -1)
- **Reason**: Match original FlashDepth behavior

### Key Points
1. **Dataloader** already computes actual_valid_mask (70m actual space) - keep for testing
2. **Training** now ignores actual_valid_mask, only checks `gt >= 0`
3. **Validation** same as training - only checks `gt >= 0`
4. **Testing** (test_gear5.py) still uses actual_valid_mask - no changes needed
5. **Visualization** shows fx_actual_original and canon_valid for reference

### Benefits
- ✅ More training data (includes >70m pixels)
- ✅ Simpler logic (no complex mask combinations)
- ✅ Matches original FlashDepth exactly
- ✅ Testing still evaluates at 70m actual space (consistent metrics)

---

## Notes for Claude

When you resume this task:

1. Read this TODO.md file
2. Start with `train_gear2.py` (most similar to gear5)
3. Then do `train_gear3.py` (same pattern)
4. Then `utils/gear3_visualization.py` (requires careful testing)
5. Finally `train_gear3_upgrade.py` (simple change)

Use `mcp__serena__find_symbol` and `mcp__serena__search_for_pattern` to locate exact positions before editing.

For each file:
- Read the relevant sections first
- Understand the current structure
- Apply modifications carefully
- Update todo list as you progress

Good luck! 🚀
