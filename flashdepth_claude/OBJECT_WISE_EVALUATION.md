# Object-wise Depth Evaluation

This document describes the object-wise depth evaluation system for demonstrating improvements in depth estimation accuracy on specific object types (e.g., vehicles, pedestrians, cyclists).

## Overview

The object-wise evaluation system computes depth metrics separately for each segmentation class, allowing you to:

1. **Identify which objects benefit most** from Gear3's attention mechanisms
2. **Compare models per object type** (e.g., Gear3 vs Baseline on cars, pedestrians, etc.)
3. **Generate detailed reports** showing per-class performance improvements
4. **Visualize object-specific improvements** in depth estimation quality

## Components

### 1. `utils/object_wise_evaluation.py`

Core metrics calculation module that computes depth accuracy per segmentation class.

**Features**:
- Supports multiple datasets: KITTI, Cityscapes, NYU Depth V2, ScanNet, VKITTI2
- Computes standard metrics per class: MAE, RMSE, AbsRel, δ1/δ2/δ3
- Aggregates metrics across video sequences
- Compares models to show percentage improvements
- Saves results to JSON files

**Usage Example**:
```python
from utils.object_wise_evaluation import ObjectWiseMetrics

# Initialize for KITTI
evaluator = ObjectWiseMetrics(dataset_type='kitti')

# Compute metrics for one frame
class_metrics = evaluator.compute_metrics_per_class(
    pred_depth,   # (H, W) predicted depth
    gt_depth,     # (H, W) ground truth depth
    seg_mask,     # (H, W) segmentation mask
    min_pixels=100  # Skip classes with fewer pixels
)

# Aggregate across multiple frames
all_metrics = [class_metrics_frame1, class_metrics_frame2, ...]
aggregated = evaluator.aggregate_metrics(all_metrics)

# Compare two models
comparison = evaluator.compare_models(
    baseline_metrics, gear3_metrics,
    model_a_name="Baseline", model_b_name="Gear3"
)

# Print and save
evaluator.print_summary(aggregated, comparison=comparison)
evaluator.save_results(aggregated, output_path, comparison=comparison)
```

### 2. `test_object_wise.py`

End-to-end evaluation script for running object-wise metrics on datasets.

**Command Line Usage**:
```bash
# Evaluate single model
python test_object_wise.py \
    --model-checkpoint train_results/results_14/gear_3/best_checkpoint.pth \
    --config-path configs/gear3 \
    --dataset kitti \
    --data-root /home/cvlab/hsy/Datasets/KITTI \
    --results-dir test_results/object_wise/gear3_kitti \
    --gpu 0

# Compare two models
python test_object_wise.py \
    --model-checkpoint train_results/results_14/gear_3/best_checkpoint.pth \
    --baseline-checkpoint train_results/baseline/best_checkpoint.pth \
    --config-path configs/gear3 \
    --dataset kitti \
    --data-root /home/cvlab/hsy/Datasets/KITTI \
    --results-dir test_results/object_wise/comparison \
    --gpu 0 \
    --max-sequences 50
```

**Arguments**:
- `--model-checkpoint`: Path to Gear3 checkpoint
- `--baseline-checkpoint`: (Optional) Path to baseline checkpoint for comparison
- `--config-path`: Model config directory
- `--dataset`: Dataset type (`kitti`, `cityscapes`, `nyu`, `vkitti2`)
- `--data-root`: Root directory of dataset
- `--results-dir`: Output directory for results
- `--gpu`: GPU device ID
- `--max-sequences`: Maximum sequences to evaluate (default: all)

### 3. `dataloaders/kitti_segmentation_dataset.py`

Reference dataset loader for KITTI with segmentation support.

**Features**:
- Loads RGB images, LiDAR depth, and segmentation masks
- Handles temporal sequences (video frames)
- Resizes to consistent resolution (518x518)
- Maps instance IDs to class IDs
- Custom collate function for batching

**Dataset Structure Required**:
```
KITTI/
├── raw/                              # RGB images
│   └── 2011_09_26/
│       └── 2011_09_26_drive_0001_sync/
│           └── image_02/data/
│               ├── 0000000000.png
│               ├── 0000000001.png
│               └── ...
├── depth/                            # LiDAR depth maps
│   └── 2011_09_26_drive_0001_sync/
│       └── proj_depth/groundtruth/image_02/
│           ├── 0000000000.png
│           ├── 0000000001.png
│           └── ...
└── segmentation/                     # Segmentation masks
    └── 2011_09_26_drive_0001_sync/
        └── image_02/
            ├── 0000000000.png
            ├── 0000000001.png
            └── ...
```

## Supported Datasets

### 1. KITTI (Implemented)

**Classes**: Background, Car, Pedestrian, Cyclist (instance segmentation)

**Getting Segmentation**:
- Option A: Download KITTI instance segmentation labels
- Option B: Use semantic KITTI labels
- Option C: Generate with Segment Anything Model (SAM)

**Data**: LiDAR depth (sparse, up to 80m)

### 2. Cityscapes (TODO)

**Classes**: 19 semantic classes (road, sidewalk, building, car, person, etc.)

**Getting Segmentation**: Download Cityscapes semantic/instance segmentation

**Data**: Stereo depth (dense, up to 50m)

### 3. NYU Depth V2 (TODO)

**Classes**: 40 indoor classes (bed, chair, table, wall, floor, etc.)

**Getting Segmentation**: Included in NYU Depth V2 dataset

**Data**: Kinect RGB-D (dense, indoor scenes)

### 4. ScanNet (TODO)

**Classes**: 20 indoor classes (wall, floor, cabinet, bed, chair, etc.)

**Getting Segmentation**: Included in ScanNet dataset

**Data**: RGB-D reconstruction (dense, indoor)

### 5. VKITTI2 (TODO)

**Classes**: 13 classes (terrain, tree, building, road, car, van, truck, etc.)

**Getting Segmentation**: Perfect synthetic segmentation included

**Data**: Perfect synthetic depth (dense, up to 100m)

## Preparing Datasets

### KITTI Setup

1. **Download KITTI Raw Data** (RGB images):
   - http://www.cvlibs.net/datasets/kitti/raw_data.php
   - Extract to `KITTI/raw/`

2. **Download KITTI Depth** (LiDAR ground truth):
   - http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
   - Extract to `KITTI/depth/`

3. **Get Segmentation Masks** (3 options):

   **Option A: KITTI Instance Segmentation**
   - Download from KITTI instance segmentation benchmark
   - Extract to `KITTI/segmentation/`

   **Option B: Semantic KITTI**
   - Download from http://semantic-kitti.org/
   - Map semantic labels to instance classes
   - Extract to `KITTI/segmentation/`

   **Option C: Generate with SAM**
   ```bash
   # Install Segment Anything
   pip install segment-anything

   # Run SAM on KITTI images
   python scripts/generate_sam_masks.py \
       --images KITTI/raw/ \
       --output KITTI/segmentation/ \
       --model-type vit_h \
       --checkpoint sam_vit_h.pth
   ```

### Cityscapes Setup (TODO)

1. Download Cityscapes dataset: https://www.cityscapes-dataset.com/
2. Extract leftImg8bit (RGB), disparity (depth), and gtFine (segmentation)
3. Implement `dataloaders/cityscapes_segmentation_dataset.py`

### NYU Depth V2 Setup (TODO)

1. Download NYU Depth V2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
2. Extract RGB, depth, and segmentation labels
3. Implement `dataloaders/nyu_segmentation_dataset.py`

### VKITTI2 Setup (TODO)

1. Download VKITTI2: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
2. Extract RGB, depth, and segmentation
3. Implement `dataloaders/vkitti2_segmentation_dataset.py`

## Running Evaluations

### Step 1: Prepare Dataset

Follow dataset setup instructions above to get RGB, depth, and segmentation data.

### Step 2: Run Evaluation

```bash
# Single model evaluation
python test_object_wise.py \
    --model-checkpoint train_results/results_14/gear_3/best_checkpoint.pth \
    --config-path configs/gear3 \
    --dataset kitti \
    --data-root /home/cvlab/hsy/Datasets/KITTI \
    --results-dir test_results/object_wise/gear3 \
    --gpu 0
```

### Step 3: Compare Models

```bash
# Baseline vs Gear3 comparison
python test_object_wise.py \
    --model-checkpoint train_results/results_14/gear_3/best_checkpoint.pth \
    --baseline-checkpoint train_results/baseline/best_checkpoint.pth \
    --config-path configs/gear3 \
    --dataset kitti \
    --data-root /home/cvlab/hsy/Datasets/KITTI \
    --results-dir test_results/object_wise/gear3_vs_baseline \
    --gpu 0
```

### Step 4: Analyze Results

Results are saved to `{results_dir}/{dataset}_object_wise_results.json`:

```json
{
  "dataset_type": "kitti",
  "per_class_metrics": {
    "car": {
      "mae": 2.345,
      "rmse": 3.456,
      "abs_rel": 0.123,
      "a1": 0.875,
      "num_pixels": 125000,
      "num_frames": 50
    },
    "pedestrian": {
      "mae": 1.234,
      "rmse": 2.345,
      "abs_rel": 0.098,
      "a1": 0.912,
      "num_pixels": 45000,
      "num_frames": 50
    }
  },
  "model_comparison": {
    "car": {
      "Baseline_mae": 2.567,
      "Gear3_mae": 2.345,
      "mae_improvement": 8.65,  // % improvement
      "Baseline_a1": 0.845,
      "Gear3_a1": 0.875,
      "a1_improvement": 3.55    // % improvement
    }
  }
}
```

## Expected Improvements with Gear3

Based on Gear3's attention-based foreground/background separation:

### High Improvement Expected:
- **Cars** (moving objects with spatial attention)
- **Pedestrians** (salient objects in foreground)
- **Cyclists** (moving foreground objects)

### Moderate Improvement Expected:
- **Buildings** (static background, depends on scene)
- **Road** (usually flat, less benefit from attention)

### Low Improvement Expected:
- **Sky** (no depth, often masked out)
- **Background vegetation** (static, less attention)

## Generating SAM Masks (Optional)

If your dataset doesn't have segmentation labels, you can generate them using Segment Anything Model:

```python
# scripts/generate_sam_masks.py
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import torch

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
sam.to(device="cuda")
mask_generator = SamAutomaticMaskGenerator(sam)

# Process images
for image_path in image_paths:
    image = np.array(Image.open(image_path))
    masks = mask_generator.generate(image)

    # Save masks (instance IDs)
    seg_mask = np.zeros(image.shape[:2], dtype=np.uint16)
    for i, mask in enumerate(masks):
        seg_mask[mask['segmentation']] = i + 1

    # Save
    output_path = output_dir / image_path.name
    Image.fromarray(seg_mask).save(output_path)
```

## Troubleshooting

### Issue: "Segmentation not found for sequence"

**Solution**: Make sure segmentation directory structure matches KITTI format:
```
KITTI/segmentation/{sequence_name}/image_02/{frame_id}.png
```

### Issue: "Too few pixels for class"

**Solution**: Lower `min_pixels` threshold in `compute_metrics_per_class()` or increase sequence count.

### Issue: "No sequences found"

**Solution**: Verify dataset paths and structure. Check that depth and segmentation exist for same sequences.

### Issue: Memory error during evaluation

**Solution**: Reduce `--batch-size` to 1 or use `--max-sequences` to limit evaluation size.

## Citation

If you use this object-wise evaluation system, please cite:

```bibtex
@article{flashdepth2024,
  title={FlashDepth: Real-time Monocular Depth Estimation with Temporal Processing},
  author={...},
  year={2024}
}
```

## Future Enhancements

- [ ] Implement Cityscapes dataset loader
- [ ] Implement NYU Depth V2 dataset loader
- [ ] Implement ScanNet dataset loader
- [ ] Implement VKITTI2 dataset loader
- [ ] Add SAM mask generation script
- [ ] Add visualization of per-class improvements
- [ ] Add spatial heatmaps showing where improvements occur
- [ ] Support multi-GPU evaluation
- [ ] Add temporal consistency metrics per object class
