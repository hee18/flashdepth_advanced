# Waymo Segmentation Dataset Preprocessing Plan

## Overview

This document contains the complete plan for preprocessing Waymo Open Dataset v2.0 with segmentation data to create a unified `waymo_processed` dataset that supports both depth and segmentation for training/validation/testing and object-wise evaluation.

**Current Status**: ⏸️ Waiting for Waymo Training Set download

**Prerequisites**:
- Waymo Open Dataset v2.0 **training set** with segmentation data
- Location: `/home/cvlab/hsy/Datasets/waymo_seg/waymo/2.0.1/train/`
- Expected files:
  - `camera_image/*.parquet` (RGB images)
  - `camera_segmentation/*.parquet` (Panoptic segmentation)
  - `lidar_camera_projection/*.parquet` (LiDAR data for depth)

---

## Current Situation

### Existing Datasets

| Dataset | Location | Depth | Segmentation | Usage |
|---------|----------|-------|--------------|-------|
| `waymo` | `/Datasets/waymo/val/` | ✅ Sparse .npy | ❌ None | Validation only |
| `waymo_seg` | `/Datasets/waymo_seg/waymo/2.0.1/val/` | ✅ Parquet (raw) | ❌ Empty (val set) | Cannot use for training |

### Problem

- **waymo**: Has depth but no segmentation → Cannot do object-wise evaluation
- **waymo_seg (val)**: Raw parquet format + empty segmentation → Cannot use directly
- **Goal**: Create unified `waymo_processed` with both depth AND segmentation

---

## Output Structure

### Target Directory

```
/home/cvlab/hsy/Datasets/waymo_processed/
├── train/
│   ├── segment-{context_name}/
│   │   └── FRONT/
│   │       ├── rgb/original/
│   │       │   ├── 0000.jpg
│   │       │   ├── 0001.jpg
│   │       │   └── ... (198 frames)
│   │       ├── depth/
│   │       │   ├── 0000.npy  # Sparse format (N, 3): [x_pixel, y_pixel, depth_meters]
│   │       │   ├── 0001.npy
│   │       │   └── ...
│   │       └── segmentation/
│   │           ├── 0000.png  # Semantic class (0-18), uint8
│   │           ├── 0001.png
│   │           └── ...
│   └── ... (more sequences)
└── val/  # Optional: Copy from existing waymo/val/ and add zero-filled segmentation
```

### File Formats

**RGB Images** (`.jpg`):
- Resolution: 1920×1280 pixels
- Quality: 95 (JPEG compression)
- Naming: 4-digit zero-padded (0000.jpg, 0001.jpg, ...)

**Depth Maps** (`.npy`):
- Format: NumPy array, dtype=float32
- Shape: (N, 3) - Sparse point cloud
- Columns:
  - [0]: X pixel coordinate (0-1919)
  - [1]: Y pixel coordinate (0-1279)
  - [2]: Depth in meters (typically 2.99-3888m range)
- Same format as existing `waymo` dataset

**Segmentation Maps** (`.png`):
- Format: Grayscale PNG, 8-bit
- Values: 0-18 (semantic class ID)
- Resolution: 1920×1280 pixels
- Class mapping (Waymo v2.0):
  ```
  0: undefined       7: construction_cone  14: road
  1: vehicle         8: bicycle            15: lane_marker
  2: pedestrian      9: motorcycle         16: other_ground
  3: sign           10: building           17: walkable
  4: cyclist        11: vegetation         18: sidewalk
  5: traffic_light  12: tree_trunk
  6: pole           13: curb
  ```

---

## Implementation: `scripts/preprocess_waymo_seg.py`

### Script Overview

**Purpose**: Convert Waymo Open Dataset v2.0 parquet files to FlashDepth-compatible format

**Input**:
- `--input-dir`: Directory containing parquet files (e.g., `/Datasets/waymo_seg/waymo/2.0.1/train/`)
- `--output-dir`: Output directory (e.g., `/Datasets/waymo_processed/train/`)
- `--camera`: Camera to process (default: `FRONT`)
- `--num-workers`: Parallel processing workers (default: 8)
- `--max-sequences`: Limit number of sequences (optional, for testing)

**Output**: Preprocessed dataset in target structure

### Key Processing Steps

#### Step 1: Load Parquet Files

```python
import pyarrow.parquet as pq
from pathlib import Path

# Find all unique context names
image_files = sorted(Path(input_dir).glob('camera_image/*.parquet'))
context_names = [f.stem for f in image_files]

# For each context
for context_name in context_names:
    # Load tables
    img_table = pq.read_table(f'{input_dir}/camera_image/{context_name}.parquet')
    seg_table = pq.read_table(f'{input_dir}/camera_segmentation/{context_name}.parquet')
    lidar_table = pq.read_table(f'{input_dir}/lidar_camera_projection/{context_name}.parquet')

    # Convert to DataFrames
    img_df = img_table.to_pandas()
    seg_df = seg_table.to_pandas()
    lidar_df = lidar_table.to_pandas()
```

#### Step 2: Filter by Camera

```python
# Camera name mapping (Waymo v2.0)
CAMERA_NAMES = {
    'FRONT': 1,
    'FRONT_LEFT': 2,
    'FRONT_RIGHT': 3,
    'SIDE_LEFT': 4,
    'SIDE_RIGHT': 5
}

camera_id = CAMERA_NAMES[camera]

# Filter DataFrames
img_camera_df = img_df[img_df['key.camera_name'] == camera_id].reset_index(drop=True)
seg_camera_df = seg_df[seg_df['key.camera_name'] == camera_id].reset_index(drop=True) if len(seg_df) > 0 else None
```

#### Step 3: Extract and Save RGB Images

```python
import io
from PIL import Image

output_rgb_dir = output_dir / f'segment-{context_name}' / camera / 'rgb' / 'original'
output_rgb_dir.mkdir(parents=True, exist_ok=True)

for frame_idx, row in img_camera_df.iterrows():
    # Decode JPEG from bytes
    img_bytes = row['[CameraImageComponent].image']
    image = Image.open(io.BytesIO(img_bytes))

    # Save as JPEG
    output_path = output_rgb_dir / f'{frame_idx:04d}.jpg'
    image.save(output_path, 'JPEG', quality=95)
```

#### Step 4: Extract and Save Segmentation

```python
output_seg_dir = output_dir / f'segment-{context_name}' / camera / 'segmentation'
output_seg_dir.mkdir(parents=True, exist_ok=True)

if seg_camera_df is not None and len(seg_camera_df) > 0:
    for frame_idx, row in seg_camera_df.iterrows():
        # Decode panoptic label
        seg_bytes = row['[CameraSegmentationLabelComponent].panoptic_label']
        divisor = row['[CameraSegmentationLabelComponent].panoptic_label_divisor']

        # Load as 32-bit integer image
        seg_img = Image.open(io.BytesIO(seg_bytes))
        panoptic_label = np.array(seg_img).astype(np.int64)

        # Extract semantic class: semantic_class = panoptic_label // divisor
        semantic_class = panoptic_label // divisor

        # Convert to uint8 (0-18 range)
        semantic_class = semantic_class.astype(np.uint8)

        # Save as PNG
        output_path = output_seg_dir / f'{frame_idx:04d}.png'
        Image.fromarray(semantic_class).save(output_path)
else:
    # Empty segmentation: create zero-filled images
    print(f"Warning: No segmentation data for {context_name}, creating zero-filled images")
    for frame_idx in range(len(img_camera_df)):
        zero_seg = np.zeros((1280, 1920), dtype=np.uint8)
        output_path = output_seg_dir / f'{frame_idx:04d}.png'
        Image.fromarray(zero_seg).save(output_path)
```

#### Step 5: Project LiDAR to Camera and Save Depth

**Note**: This is the most complex part. Waymo v2.0 uses different API than v1.x.

**Reference**: Adapt from `scripts/preprocess_waymo.py` lines 92-165

```python
import numpy as np
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2

# This part requires Waymo Open Dataset SDK and is complex
# Key steps:
# 1. Parse range images from lidar_camera_projection
# 2. Convert range images to 3D point clouds
# 3. Project 3D points to camera coordinates
# 4. Filter points within image bounds
# 5. Save as sparse depth (N, 3) format

# Pseudocode:
output_depth_dir = output_dir / f'segment-{context_name}' / camera / 'depth'
output_depth_dir.mkdir(parents=True, exist_ok=True)

for frame_idx in range(len(img_camera_df)):
    # Project LiDAR for this frame
    sparse_depth = project_lidar_to_camera(
        lidar_df,
        img_camera_df,
        frame_idx,
        camera
    )

    # sparse_depth shape: (N, 3) - [x_pixel, y_pixel, depth_meters]

    # Save as .npy
    output_path = output_depth_dir / f'{frame_idx:04d}.npy'
    np.save(output_path, sparse_depth.astype(np.float32))
```

**Critical Function** (needs adaptation from v1.x):

```python
def project_lidar_to_camera(lidar_df, img_df, frame_idx, camera):
    """
    Project LiDAR points to camera coordinates.

    Args:
        lidar_df: DataFrame with lidar_camera_projection data
        img_df: DataFrame with camera_image data
        frame_idx: Frame index
        camera: Camera name (e.g., 'FRONT')

    Returns:
        sparse_depth: (N, 3) array of [x_pixel, y_pixel, depth_meters]
    """
    # TODO: Implement using Waymo Open Dataset SDK v2.0
    # Reference: scripts/preprocess_waymo.py:build_camera_depth_from_lidar()

    # Key differences from v1.x:
    # - v2.0 stores data in parquet, not tfrecord
    # - Range image format may differ
    # - Need to adapt frame_utils functions

    pass  # Placeholder
```

### Error Handling

```python
def process_sequence(context_name, input_dir, output_dir, camera):
    """Process a single sequence with error handling."""
    try:
        # Processing steps...
        print(f"✓ Successfully processed {context_name}")
        return True
    except Exception as e:
        print(f"✗ Error processing {context_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
```

### Parallel Processing

```python
from multiprocessing import Pool

def main():
    # Parse arguments...

    # Get list of sequences
    context_names = [f.stem for f in image_files]

    if max_sequences:
        context_names = context_names[:max_sequences]

    # Process in parallel
    with Pool(num_workers) as pool:
        results = pool.starmap(
            process_sequence,
            [(name, input_dir, output_dir, camera) for name in context_names]
        )

    # Summary
    success_count = sum(results)
    print(f"\nProcessed {success_count}/{len(context_names)} sequences successfully")
```

---

## Dataset Loader Modifications

### Option A: Modify `waymo_dataset.py` to Support Segmentation

**File**: `dataloaders/waymo_dataset.py`

**Changes**:

1. Add optional segmentation loading in `depth_read()` method
2. Return dict format when segmentation is loaded
3. Maintain backward compatibility

```python
class WaymoDepth(BaseDatasetPairs):
    def __init__(self, root_dir, split, load_cache=None, use_segmentation=False, return_dict=False, **kwargs):
        """
        Args:
            use_segmentation: If True, load segmentation from {scene}/FRONT/segmentation/
            return_dict: If True, return dict format; else return tuple
        """
        self.use_segmentation = use_segmentation
        self.return_dict = return_dict

        self.root_dir = os.path.join(root_dir, 'waymo/val')
        super().__init__(dataset_name='waymo', root_dir=self.root_dir, split=split, load_cache=load_cache)
        self.reshape_list['resolution'] = (1920, 1280)

    def depth_read(self, path):
        """Load depth and optionally segmentation."""
        # Load sparse depth (existing code)
        depth_points = np.load(path)
        depth_map = np.full((1280, 1920), -1.0, dtype=np.float32)

        if len(depth_points) > 0:
            x_coords = depth_points[:, 0].astype(np.int32)
            y_coords = depth_points[:, 1].astype(np.int32)
            depths = depth_points[:, 2]

            valid_mask = (x_coords >= 0) & (x_coords < 1920) & (y_coords >= 0) & (y_coords < 1280)
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            depths = depths[valid_mask]

            depth_map[y_coords, x_coords] = depths

        inverse_depth = np.where(depth_map > 0, 1.0 / depth_map, -1.0)

        # Load segmentation if requested
        if self.use_segmentation:
            seg_path = path.replace('/depth/', '/segmentation/').replace('.npy', '.png')
            if os.path.exists(seg_path):
                seg_mask = np.array(Image.open(seg_path))
            else:
                seg_mask = np.zeros((1280, 1920), dtype=np.uint8)

            if self.return_dict:
                return {'depth': inverse_depth, 'segmentation': seg_mask}
            else:
                return inverse_depth, seg_mask

        return inverse_depth
```

### Option B: Use `waymo_segmentation_dataset.py` with Depth Enabled

**File**: `dataloaders/waymo_segmentation_dataset.py`

**Current Status**: Already supports both depth and segmentation!

**Required Changes**: None (already implemented)

**Usage**:
```python
dataset = WaymoSegmentationDataset(
    data_root='/Datasets/waymo_processed',
    split='train',
    use_depth=True,
    depth_root='/Datasets/waymo_processed/train'  # Same as data_root in this case
)
```

**Recommendation**: ✅ **Use Option B** - Already implemented and tested!

---

## Configuration Updates

### After Preprocessing Completes

Update all config files:

```yaml
# configs/gear2/config*.yaml
# configs/gear3/config*.yaml
# configs/gear3_upgrade/config*.yaml

dataset:
  # Before
  val_datasets: [sintel_seg, waymo]
  test_datasets: [sintel_seg]

  # After
  val_datasets: [sintel_seg, waymo_processed]
  test_datasets: [sintel_seg, waymo_processed]

# Object-wise evaluation
object_wise:
  enabled: false
  dataset: waymo_processed  # Was: waymo_seg
```

### Docker Volume Mount

Ensure `/Datasets/waymo_processed` is accessible in Docker:

**File**: `docker-compose.yml`

```yaml
volumes:
  - /home/cvlab/hsy/Datasets:/data/datasets:ro
```

This already mounts the entire Datasets directory, so `waymo_processed` will be accessible at `/data/datasets/waymo_processed`.

---

## Testing Plan

### Phase 1: Preprocessing Validation

```bash
# Test on a few sequences first
python scripts/preprocess_waymo_seg.py \
  --input-dir /home/cvlab/hsy/Datasets/waymo_seg/waymo/2.0.1/train \
  --output-dir /home/cvlab/hsy/Datasets/waymo_processed/train \
  --camera FRONT \
  --max-sequences 5 \
  --num-workers 1

# Check output structure
tree /home/cvlab/hsy/Datasets/waymo_processed/train/ -L 4
```

Expected output:
```
train/
├── segment-{name1}/
│   └── FRONT/
│       ├── rgb/original/ (198 .jpg files)
│       ├── depth/ (198 .npy files)
│       └── segmentation/ (198 .png files)
├── segment-{name2}/
...
```

### Phase 2: Data Integrity Checks

```python
# scripts/verify_waymo_processed.py

import numpy as np
from PIL import Image
from pathlib import Path

def verify_sequence(seq_dir):
    """Verify a processed sequence."""
    rgb_files = sorted((seq_dir / 'FRONT/rgb/original').glob('*.jpg'))
    depth_files = sorted((seq_dir / 'FRONT/depth').glob('*.npy'))
    seg_files = sorted((seq_dir / 'FRONT/segmentation').glob('*.png'))

    assert len(rgb_files) == len(depth_files) == len(seg_files), "Frame count mismatch"

    for i, (rgb_f, depth_f, seg_f) in enumerate(zip(rgb_files, depth_files, seg_files)):
        # Check RGB
        img = Image.open(rgb_f)
        assert img.size == (1920, 1280), f"RGB size mismatch: {rgb_f}"

        # Check depth
        depth = np.load(depth_f)
        assert depth.shape[1] == 3, f"Depth format mismatch: {depth_f}"
        assert depth.dtype == np.float32, f"Depth dtype mismatch: {depth_f}"

        # Check segmentation
        seg = np.array(Image.open(seg_f))
        assert seg.shape == (1280, 1920), f"Seg size mismatch: {seg_f}"
        assert seg.dtype == np.uint8, f"Seg dtype mismatch: {seg_f}"
        assert seg.max() <= 18, f"Seg class out of range: {seg_f}"

    print(f"✓ {seq_dir.name}: {len(rgb_files)} frames verified")

# Run verification
processed_dir = Path('/home/cvlab/hsy/Datasets/waymo_processed/train')
for seq_dir in sorted(processed_dir.glob('segment-*')):
    verify_sequence(seq_dir)
```

### Phase 3: Dataset Loader Testing

```python
# Test WaymoSegmentationDataset with depth enabled
from dataloaders.waymo_segmentation_dataset import WaymoSegmentationDataset

dataset = WaymoSegmentationDataset(
    data_root='/home/cvlab/hsy/Datasets/waymo_processed',
    split='train',
    video_length=5,
    resolution=518,
    use_depth=True,
    depth_root='/home/cvlab/hsy/Datasets/waymo_processed/train'
)

print(f"Dataset size: {len(dataset)}")

# Load first sample
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")  # (5, 3, 518, 518)
print(f"Depth shape: {sample['depth'].shape}")  # (5, 518, 518)
print(f"Segmentation shape: {sample['segmentation'].shape}")  # (518, 518)
print(f"Sequence name: {sample['sequence_name']}")

# Check depth is not all zeros
assert sample['depth'].sum() > 0, "Depth is all zeros!"
print("✓ Depth loaded successfully")
```

### Phase 4: Training Test

```bash
# Update config to use waymo_processed
# Then start training for a few iterations
./run_docker.sh train_gear2_ddp --config-variant l \
  --results-dir train_results/test_waymo_processed/ \
  --epochs 100

# Check logs for:
# - "loading cache for waymo_processed" or similar
# - Validation metrics on waymo_processed
# - No crashes or data loading errors
```

### Phase 5: Object-wise Evaluation Test

```bash
# Test object-wise evaluation with waymo_processed
./run_docker.sh test_gear3_objwise \
  --dataset waymo_processed \
  --gpu 2 \
  --results-dir test_results/waymo_processed_objwise/
```

---

## Estimated Storage and Time

### Storage Requirements

**Training Set** (estimated):
- Sequences: ~1000 sequences (assuming similar to validation)
- RGB: 1000 × 198 frames × ~300KB = ~59GB
- Depth: 1000 × 198 frames × ~50KB = ~10GB
- Segmentation: 1000 × 198 frames × ~10KB = ~2GB
- **Total**: ~71GB

### Processing Time

**Bottleneck**: LiDAR projection (most CPU-intensive)

Estimated per sequence:
- RGB extraction: ~10 seconds
- Segmentation extraction: ~5 seconds
- LiDAR projection: ~120 seconds
- **Total per sequence**: ~135 seconds

With 8 workers:
- 1000 sequences / 8 = 125 batches
- 125 × 135 seconds = ~17000 seconds = **~4.7 hours**

---

## Troubleshooting

### Issue: Waymo Open Dataset SDK Version Mismatch

**Symptom**: Import errors or API differences

**Solution**: Check installed version and update
```bash
pip show waymo-open-dataset
pip install --upgrade waymo-open-dataset-tf-2-11-0  # Or appropriate version
```

### Issue: PyArrow Not Installed

**Symptom**: `ModuleNotFoundError: No module named 'pyarrow'`

**Solution**:
```bash
pip install pyarrow
```

### Issue: Empty Segmentation Parquet

**Symptom**: `len(seg_df) == 0` for training set

**Solution**: This should not happen for training set. Check:
1. Downloaded correct split (train, not val)
2. File is not corrupted
3. Context name matches between files

### Issue: Memory Errors During Processing

**Symptom**: `MemoryError` or OOM killer

**Solution**:
1. Reduce `--num-workers`
2. Process in smaller batches with `--max-sequences`
3. Add memory monitoring:
   ```python
   import psutil
   print(f"Memory usage: {psutil.virtual_memory().percent}%")
   ```

### Issue: LiDAR Projection Produces No Points

**Symptom**: Depth arrays are all empty

**Solution**:
1. Check camera calibration parameters
2. Verify coordinate transformation matrices
3. Check if camera name filtering is correct
4. Debug with single frame visualization

---

## Next Steps

### Immediate Actions (After Downloading Waymo Training Set)

1. ✅ **Verify download**: Check that segmentation data is present
   ```bash
   ls -lh /home/cvlab/hsy/Datasets/waymo_seg/waymo/2.0.1/train/camera_segmentation/
   # Should show many .parquet files with non-zero size
   ```

2. ✅ **Create preprocessing script**: `scripts/preprocess_waymo_seg.py`
   - Use this document as reference
   - Adapt LiDAR projection from `scripts/preprocess_waymo.py`
   - Test on 5 sequences first

3. ✅ **Run preprocessing**:
   ```bash
   python scripts/preprocess_waymo_seg.py \
     --input-dir /home/cvlab/hsy/Datasets/waymo_seg/waymo/2.0.1/train \
     --output-dir /home/cvlab/hsy/Datasets/waymo_processed/train \
     --camera FRONT \
     --num-workers 8
   ```

4. ✅ **Verify output**: Run `scripts/verify_waymo_processed.py`

5. ✅ **Update configs**: Replace `waymo` with `waymo_processed` in all config files

6. ✅ **Test training**: Short training run to verify data loading

7. ✅ **Test object-wise evaluation**: Verify segmentation is loaded correctly

8. ✅ **Clean up old datasets**: (Optional) Remove duplicate `waymo` and `waymo_seg/waymo/2.0.1`

### Future Improvements

- **Validation set**: Optionally preprocess val set and add zero-filled segmentation for consistency
- **Multi-camera**: Extend to all 5 cameras instead of just FRONT
- **Compression**: Explore better compression for depth files (e.g., compressed numpy)
- **Caching**: Add caching mechanism similar to other datasets

---

## References

### Code Files to Review

1. **Existing preprocessing**: `scripts/preprocess_waymo.py`
   - LiDAR projection logic (lines 92-165)
   - TFRecord parsing (adapt to parquet)
   - Multiprocessing setup

2. **Dataset loaders**:
   - `dataloaders/waymo_dataset.py` - Original waymo loader
   - `dataloaders/waymo_segmentation_dataset.py` - Segmentation loader (use this!)
   - `dataloaders/base_dataset_pairs.py` - Dataset factory

3. **Object-wise evaluation**:
   - `utils/object_wise_evaluation.py` - Metrics computation
   - `test_gear3.py` - Test script with object-wise mode

### External Documentation

- [Waymo Open Dataset v2.0 Documentation](https://waymo.com/open/data/perception/)
- [Waymo Open Dataset SDK GitHub](https://github.com/waymo-research/waymo-open-dataset)
- [PyArrow Documentation](https://arrow.apache.org/docs/python/)

---

## Summary

This plan provides a complete roadmap for preprocessing Waymo Open Dataset v2.0 with segmentation data. The key points:

1. ✅ **Clear goal**: Unified `waymo_processed` dataset with depth + segmentation
2. ✅ **Detailed structure**: Exact directory layout and file formats specified
3. ✅ **Implementation guide**: Step-by-step processing logic with code examples
4. ✅ **Testing plan**: Comprehensive verification from data integrity to training
5. ✅ **Troubleshooting**: Common issues and solutions documented
6. ✅ **Next steps**: Clear action items for when training data arrives

**Status**: Ready to implement once Waymo training set is downloaded!

---

*Generated: 2025-10-30*
*Author: Claude (via flashdepth_claude project)*
