# Canonicalization Test Results

Test date: 2025-11-13

## Dataset Overview

| Dataset | Original Resolution (W×H) | Original Focal Length (fx) |
|---------|---------------------------|----------------------------|
| dynamicreplica (train-base) | 1280×720 | 1244.44 |
| mvs-synth (train-base) | 1920×1080 | 1157.85 |
| pointodyssey (train-base) | 960×540 | 800.00 |
| spring (train-base) | 1920×1080 | 2181.82 |
| tartanair (train-base) | 640×480 | 320.00 |
| eth3d (val-base) | 6220×4141 | 3527.82 |
| sintel (val-base) | 1024×436 | 688.00 |
| unreal4k (val-base) | 3840×2160 | 1920.00 |
| urbansyn (val-base) | 2048×1024 | 1730.21 |
| waymo_seg (val-base) | 1920×1280 | 2059.61 |
| dynamicreplica (train-2k) | 1280×720 | 1244.44 |
| mvs-synth (train-2k) | 1920×1080 | 1157.85 |
| pointodyssey (train-2k) | 960×540 | 800.00 |
| spring (train-2k) | 1920×1080 | 2181.82 |
| tartanair (train-2k) | 640×480 | 320.00 |
| eth3d (val-2k) | 6220×4141 | 3527.82 |
| sintel (val-2k) | 1024×436 | 688.00 |
| unreal4k (val-2k) | 3840×2160 | 1920.00 |
| urbansyn (val-2k) | 2048×1024 | 1730.21 |
| waymo_seg (val-2k) | 1920×1280 | 2059.61 |

---

## Canonicalization Results by Dataset

### Formulas

```
pre_h = original_h × resize_factor
pre_w = original_w × resize_factor
target_w, target_h = target_resolution  # FIXED: target_resolution is (W, H)
small_resize_ratio = max(target_w / pre_w, target_h / pre_h)  # W→W, H→H
fx_ratio = 500.0 / fx_actual
total_resize_ratio = resize_factor × small_resize_ratio
inverse_depth_correction_ratio = total_resize_ratio / fx_ratio
normal_depth_correction_ratio = fx_ratio / total_resize_ratio
```

---

## Base Resolution - Training Datasets

### dynamicreplica (train-base)

**Input:**

- Original resolution: 1280×720 (W×H)
- Original focal length: fx_actual = 1244.44
- Target resolution: (518, 518)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/dynamicreplica/train/00e2a3-3_obj_source_left/images/00e2a3-3_obj_source_left-0000.png`

**Calculation:**

```
pre_h = 720 × 1 = 720
pre_w = 1280 × 1 = 1280
target_w, target_h = (518, 518)
target_w = 518
target_h = 518
small_resize_ratio = max(518/1280, 518/720)
                   = max(0.404687, 0.719444)
                   = 0.719444
```

**Results:**

```
fx_ratio = 500.0 / 1244.44 = 0.401786
total_resize_ratio = 1 × 0.719444 = 0.719444
inverse_depth_correction_ratio = 0.719444 / 0.401786 = 1.790617
normal_depth_correction_ratio = 0.401786 / 0.719444 = 0.558467
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 2.488889
- Actual inverse_depth_correction: 1.790617
- Error: 0.698272
- ⚠️ **Error is significant** - possible bug in calculation

---

### mvs-synth (train-base)

**Input:**

- Original resolution: 1920×1080 (W×H)
- Original focal length: fx_actual = 1157.85
- Target resolution: (518, 518)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/mvs-synth/GTAV_1080/0000/images/0000.png`

**Calculation:**

```
pre_h = 1080 × 1 = 1080
pre_w = 1920 × 1 = 1920
target_w, target_h = (518, 518)
target_w = 518
target_h = 518
small_resize_ratio = max(518/1920, 518/1080)
                   = max(0.269792, 0.479630)
                   = 0.479630
```

**Results:**

```
fx_ratio = 500.0 / 1157.85 = 0.431836
total_resize_ratio = 1 × 0.479630 = 0.479630
inverse_depth_correction_ratio = 0.479630 / 0.431836 = 1.110676
normal_depth_correction_ratio = 0.431836 / 0.479630 = 0.900353
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 2.315695
- Actual inverse_depth_correction: 1.110676
- Error: 1.205019
- ⚠️ **Significant downsampling** (total_resize_ratio = 0.480)

---

### pointodyssey (train-base)

**Input:**

- Original resolution: 960×540 (W×H)
- Original focal length: fx_actual = 800.00
- Target resolution: (518, 518)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/pointodyssey/train/cnb_dlab_0215_3rd/rgbs/rgb_00000.jpg`

**Calculation:**

```
pre_h = 540 × 1 = 540
pre_w = 960 × 1 = 960
target_w, target_h = (518, 518)
target_w = 518
target_h = 518
small_resize_ratio = max(518/960, 518/540)
                   = max(0.539583, 0.959259)
                   = 0.959259
```

**Results:**

```
fx_ratio = 500.0 / 800.00 = 0.625000
total_resize_ratio = 1 × 0.959259 = 0.959259
inverse_depth_correction_ratio = 0.959259 / 0.625000 = 1.534815
normal_depth_correction_ratio = 0.625000 / 0.959259 = 0.651544
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 1.600000
- Actual inverse_depth_correction: 1.534815
- Error: 0.065185
- ⚠️ **Error is significant** - possible bug in calculation

---

### spring (train-base)

**Input:**

- Original resolution: 1920×1080 (W×H)
- Original focal length: fx_actual = 2181.82
- Target resolution: (518, 518)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/spring/train/0001/frame_left/frame_left_0001.png`

**Calculation:**

```
pre_h = 1080 × 1 = 1080
pre_w = 1920 × 1 = 1920
target_w, target_h = (518, 518)
target_w = 518
target_h = 518
small_resize_ratio = max(518/1920, 518/1080)
                   = max(0.269792, 0.479630)
                   = 0.479630
```

**Results:**

```
fx_ratio = 500.0 / 2181.82 = 0.229167
total_resize_ratio = 1 × 0.479630 = 0.479630
inverse_depth_correction_ratio = 0.479630 / 0.229167 = 2.092929
normal_depth_correction_ratio = 0.229167 / 0.479630 = 0.477799
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 4.363636
- Actual inverse_depth_correction: 2.092929
- Error: 2.270707
- ⚠️ **Significant downsampling** (total_resize_ratio = 0.480)

---

### tartanair (train-base)

**Input:**

- Original resolution: 640×480 (W×H)
- Original focal length: fx_actual = 320.00
- Target resolution: (518, 518)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/tartanair/abandonedfactory/Easy/P000/image_left/000000_left.png`

**Calculation:**

```
pre_h = 480 × 1 = 480
pre_w = 640 × 1 = 640
target_w, target_h = (518, 518)
target_w = 518
target_h = 518
small_resize_ratio = max(518/640, 518/480)
                   = max(0.809375, 1.079167)
                   = 1.079167
```

**Results:**

```
fx_ratio = 500.0 / 320.00 = 1.562500
total_resize_ratio = 1 × 1.079167 = 1.079167
inverse_depth_correction_ratio = 1.079167 / 1.562500 = 0.690667
normal_depth_correction_ratio = 1.562500 / 1.079167 = 1.447876
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 0.640000
- Actual inverse_depth_correction: 0.690667
- Error: 0.050667
- ⚠️ **Error is significant** - possible bug in calculation

---


## Base Resolution - Validation Datasets

### eth3d (val-base)

**Input:**

- Original resolution: 6220×4141 (W×H)
- Original focal length: fx_actual = 3527.82
- Target resolution: (784, 518)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/eth3d/pipes/images/dslr_images/DSC_0634.JPG`

**Calculation:**

```
pre_h = 4141 × 1 = 4141
pre_w = 6220 × 1 = 6220
target_w, target_h = (784, 518)
target_w = 784
target_h = 518
small_resize_ratio = max(784/6220, 518/4141)
                   = max(0.126045, 0.125091)
                   = 0.126045
```

**Results:**

```
fx_ratio = 500.0 / 3527.82 = 0.141730
total_resize_ratio = 1 × 0.126045 = 0.126045
inverse_depth_correction_ratio = 0.126045 / 0.141730 = 0.889329
normal_depth_correction_ratio = 0.141730 / 0.126045 = 1.124443
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 7.055648
- Actual inverse_depth_correction: 0.889329
- Error: 6.166319
- ⚠️ **Significant downsampling** (total_resize_ratio = 0.126)

---

### sintel (val-base)

**Input:**

- Original resolution: 1024×436 (W×H)
- Original focal length: fx_actual = 688.00
- Target resolution: (1022, 434)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/sintel/images/training/clean/alley_1/frame_0001.png`

**Calculation:**

```
pre_h = 436 × 1 = 436
pre_w = 1024 × 1 = 1024
target_w, target_h = (1022, 434)
target_w = 1022
target_h = 434
small_resize_ratio = max(1022/1024, 434/436)
                   = max(0.998047, 0.995413)
                   = 0.998047
```

**Results:**

```
fx_ratio = 500.0 / 688.00 = 0.726744
total_resize_ratio = 1 × 0.998047 = 0.998047
inverse_depth_correction_ratio = 0.998047 / 0.726744 = 1.373313
normal_depth_correction_ratio = 0.726744 / 0.998047 = 0.728166
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 1.376000
- Actual inverse_depth_correction: 1.373313
- Error: 0.002688
- ✅ **Close to expected value** (resize is minimal)

---

### unreal4k (val-base)

**Input:**

- Original resolution: 3840×2160 (W×H)
- Original focal length: fx_actual = 1920.00
- Target resolution: (924, 518)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/unreal4k/UnrealStereo4K_00000/Image0/00000.png`

**Calculation:**

```
pre_h = 2160 × 1 = 2160
pre_w = 3840 × 1 = 3840
target_w, target_h = (924, 518)
target_w = 924
target_h = 518
small_resize_ratio = max(924/3840, 518/2160)
                   = max(0.240625, 0.239815)
                   = 0.240625
```

**Results:**

```
fx_ratio = 500.0 / 1920.00 = 0.260417
total_resize_ratio = 1 × 0.240625 = 0.240625
inverse_depth_correction_ratio = 0.240625 / 0.260417 = 0.924000
normal_depth_correction_ratio = 0.260417 / 0.240625 = 1.082251
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 3.840000
- Actual inverse_depth_correction: 0.924000
- Error: 2.916000
- ⚠️ **Significant downsampling** (total_resize_ratio = 0.241)

---

### urbansyn (val-base)

**Input:**

- Original resolution: 2048×1024 (W×H)
- Original focal length: fx_actual = 1730.21
- Target resolution: (1036, 518)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/urbansyn/rgb/rgb_0001.png`

**Calculation:**

```
pre_h = 1024 × 1 = 1024
pre_w = 2048 × 1 = 2048
target_w, target_h = (1036, 518)
target_w = 1036
target_h = 518
small_resize_ratio = max(1036/2048, 518/1024)
                   = max(0.505859, 0.505859)
                   = 0.505859
```

**Results:**

```
fx_ratio = 500.0 / 1730.21 = 0.288983
total_resize_ratio = 1 × 0.505859 = 0.505859
inverse_depth_correction_ratio = 0.505859 / 0.288983 = 1.750483
normal_depth_correction_ratio = 0.288983 / 0.505859 = 0.571271
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 3.460414
- Actual inverse_depth_correction: 1.750483
- Error: 1.709931
- ⚠️ **Significant downsampling** (total_resize_ratio = 0.506)

---

### waymo_seg (val-base)

**Input:**

- Original resolution: 1920×1280 (W×H)
- Original focal length: fx_actual = 2059.61
- Target resolution: (784, 518)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/waymo_seg/val/segment-10017090168044687777_6380_000_6400_000/FRONT/rgb/original/0000.jpg`

**Calculation:**

```
pre_h = 1280 × 1 = 1280
pre_w = 1920 × 1 = 1920
target_w, target_h = (784, 518)
target_w = 784
target_h = 518
small_resize_ratio = max(784/1920, 518/1280)
                   = max(0.408333, 0.404687)
                   = 0.408333
```

**Results:**

```
fx_ratio = 500.0 / 2059.61 = 0.242764
total_resize_ratio = 1 × 0.408333 = 0.408333
inverse_depth_correction_ratio = 0.408333 / 0.242764 = 1.682016
normal_depth_correction_ratio = 0.242764 / 0.408333 = 0.594524
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 4.119224
- Actual inverse_depth_correction: 1.682016
- Error: 2.437208
- ⚠️ **Significant downsampling** (total_resize_ratio = 0.408)

---


## 2K Resolution - Training Datasets

### dynamicreplica (train-2k)

**Input:**

- Original resolution: 1280×720 (W×H)
- Original focal length: fx_actual = 1244.44
- Target resolution: (1918, 1078)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/dynamicreplica/train/00e2a3-3_obj_source_left/images/00e2a3-3_obj_source_left-0000.png`

**Calculation:**

```
pre_h = 720 × 1 = 720
pre_w = 1280 × 1 = 1280
target_w, target_h = (1918, 1078)
target_w = 1918
target_h = 1078
small_resize_ratio = max(1918/1280, 1078/720)
                   = max(1.498438, 1.497222)
                   = 1.498438
```

**Results:**

```
fx_ratio = 500.0 / 1244.44 = 0.401786
total_resize_ratio = 1 × 1.498438 = 1.498438
inverse_depth_correction_ratio = 1.498438 / 0.401786 = 3.729444
normal_depth_correction_ratio = 0.401786 / 1.498438 = 0.268136
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 2.488889
- Actual inverse_depth_correction: 3.729444
- Error: 1.240556
- ⚠️ **Error is significant** - possible bug in calculation

---

### mvs-synth (train-2k)

**Input:**

- Original resolution: 1920×1080 (W×H)
- Original focal length: fx_actual = 1157.85
- Target resolution: (1918, 1078)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/mvs-synth/GTAV_1080/0000/images/0000.png`

**Calculation:**

```
pre_h = 1080 × 1 = 1080
pre_w = 1920 × 1 = 1920
target_w, target_h = (1918, 1078)
target_w = 1918
target_h = 1078
small_resize_ratio = max(1918/1920, 1078/1080)
                   = max(0.998958, 0.998148)
                   = 0.998958
```

**Results:**

```
fx_ratio = 500.0 / 1157.85 = 0.431836
total_resize_ratio = 1 × 0.998958 = 0.998958
inverse_depth_correction_ratio = 0.998958 / 0.431836 = 2.313283
normal_depth_correction_ratio = 0.431836 / 0.998958 = 0.432286
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 2.315695
- Actual inverse_depth_correction: 2.313283
- Error: 0.002412
- ✅ **Close to expected value** (resize is minimal)

---

### pointodyssey (train-2k)

**Input:**

- Original resolution: 960×540 (W×H)
- Original focal length: fx_actual = 800.00
- Target resolution: (1918, 1078)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/pointodyssey/train/cnb_dlab_0215_3rd/rgbs/rgb_00000.jpg`

**Calculation:**

```
pre_h = 540 × 1 = 540
pre_w = 960 × 1 = 960
target_w, target_h = (1918, 1078)
target_w = 1918
target_h = 1078
small_resize_ratio = max(1918/960, 1078/540)
                   = max(1.997917, 1.996296)
                   = 1.997917
```

**Results:**

```
fx_ratio = 500.0 / 800.00 = 0.625000
total_resize_ratio = 1 × 1.997917 = 1.997917
inverse_depth_correction_ratio = 1.997917 / 0.625000 = 3.196667
normal_depth_correction_ratio = 0.625000 / 1.997917 = 0.312826
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 1.600000
- Actual inverse_depth_correction: 3.196667
- Error: 1.596667
- ⚠️ **Significant upsampling** (total_resize_ratio = 1.998)

---

### spring (train-2k)

**Input:**

- Original resolution: 1920×1080 (W×H)
- Original focal length: fx_actual = 2181.82
- Target resolution: (1918, 1078)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/spring/train/0001/frame_left/frame_left_0001.png`

**Calculation:**

```
pre_h = 1080 × 1 = 1080
pre_w = 1920 × 1 = 1920
target_w, target_h = (1918, 1078)
target_w = 1918
target_h = 1078
small_resize_ratio = max(1918/1920, 1078/1080)
                   = max(0.998958, 0.998148)
                   = 0.998958
```

**Results:**

```
fx_ratio = 500.0 / 2181.82 = 0.229167
total_resize_ratio = 1 × 0.998958 = 0.998958
inverse_depth_correction_ratio = 0.998958 / 0.229167 = 4.359091
normal_depth_correction_ratio = 0.229167 / 0.998958 = 0.229406
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 4.363636
- Actual inverse_depth_correction: 4.359091
- Error: 0.004545
- ✅ **Close to expected value** (resize is minimal)

---

### tartanair (train-2k)

**Input:**

- Original resolution: 640×480 (W×H)
- Original focal length: fx_actual = 320.00
- Target resolution: (1918, 1078)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/tartanair/abandonedfactory/Easy/P000/image_left/000000_left.png`

**Calculation:**

```
pre_h = 480 × 1 = 480
pre_w = 640 × 1 = 640
target_w, target_h = (1918, 1078)
target_w = 1918
target_h = 1078
small_resize_ratio = max(1918/640, 1078/480)
                   = max(2.996875, 2.245833)
                   = 2.996875
```

**Results:**

```
fx_ratio = 500.0 / 320.00 = 1.562500
total_resize_ratio = 1 × 2.996875 = 2.996875
inverse_depth_correction_ratio = 2.996875 / 1.562500 = 1.918000
normal_depth_correction_ratio = 1.562500 / 2.996875 = 0.521376
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 0.640000
- Actual inverse_depth_correction: 1.918000
- Error: 1.278000
- ⚠️ **Significant upsampling** (total_resize_ratio = 2.997)

---


## 2K Resolution - Validation Datasets

### eth3d (val-2k)

**Input:**

- Original resolution: 6220×4141 (W×H)
- Original focal length: fx_actual = 3527.82
- Target resolution: (1918, 1274)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/eth3d/pipes/images/dslr_images/DSC_0634.JPG`

**Calculation:**

```
pre_h = 4141 × 1 = 4141
pre_w = 6220 × 1 = 6220
target_w, target_h = (1918, 1274)
target_w = 1918
target_h = 1274
small_resize_ratio = max(1918/6220, 1274/4141)
                   = max(0.308360, 0.307655)
                   = 0.308360
```

**Results:**

```
fx_ratio = 500.0 / 3527.82 = 0.141730
total_resize_ratio = 1 × 0.308360 = 0.308360
inverse_depth_correction_ratio = 0.308360 / 0.141730 = 2.175681
normal_depth_correction_ratio = 0.141730 / 0.308360 = 0.459626
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 7.055648
- Actual inverse_depth_correction: 2.175681
- Error: 4.879967
- ⚠️ **Significant downsampling** (total_resize_ratio = 0.308)

---

### sintel (val-2k)

**Input:**

- Original resolution: 1024×436 (W×H)
- Original focal length: fx_actual = 688.00
- Target resolution: (1022, 434)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/sintel/images/training/clean/alley_1/frame_0001.png`

**Calculation:**

```
pre_h = 436 × 1 = 436
pre_w = 1024 × 1 = 1024
target_w, target_h = (1022, 434)
target_w = 1022
target_h = 434
small_resize_ratio = max(1022/1024, 434/436)
                   = max(0.998047, 0.995413)
                   = 0.998047
```

**Results:**

```
fx_ratio = 500.0 / 688.00 = 0.726744
total_resize_ratio = 1 × 0.998047 = 0.998047
inverse_depth_correction_ratio = 0.998047 / 0.726744 = 1.373313
normal_depth_correction_ratio = 0.726744 / 0.998047 = 0.728166
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 1.376000
- Actual inverse_depth_correction: 1.373313
- Error: 0.002688
- ✅ **Close to expected value** (resize is minimal)

---

### unreal4k (val-2k)

**Input:**

- Original resolution: 3840×2160 (W×H)
- Original focal length: fx_actual = 1920.00
- Target resolution: (2044, 1148)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/unreal4k/UnrealStereo4K_00000/Image0/00000.png`

**Calculation:**

```
pre_h = 2160 × 1 = 2160
pre_w = 3840 × 1 = 3840
target_w, target_h = (2044, 1148)
target_w = 2044
target_h = 1148
small_resize_ratio = max(2044/3840, 1148/2160)
                   = max(0.532292, 0.531481)
                   = 0.532292
```

**Results:**

```
fx_ratio = 500.0 / 1920.00 = 0.260417
total_resize_ratio = 1 × 0.532292 = 0.532292
inverse_depth_correction_ratio = 0.532292 / 0.260417 = 2.044000
normal_depth_correction_ratio = 0.260417 / 0.532292 = 0.489237
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 3.840000
- Actual inverse_depth_correction: 2.044000
- Error: 1.796000
- ⚠️ **Significant downsampling** (total_resize_ratio = 0.532)

---

### urbansyn (val-2k)

**Input:**

- Original resolution: 2048×1024 (W×H)
- Original focal length: fx_actual = 1730.21
- Target resolution: (2044, 1022)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/urbansyn/rgb/rgb_0001.png`

**Calculation:**

```
pre_h = 1024 × 1 = 1024
pre_w = 2048 × 1 = 2048
target_w, target_h = (2044, 1022)
target_w = 2044
target_h = 1022
small_resize_ratio = max(2044/2048, 1022/1024)
                   = max(0.998047, 0.998047)
                   = 0.998047
```

**Results:**

```
fx_ratio = 500.0 / 1730.21 = 0.288983
total_resize_ratio = 1 × 0.998047 = 0.998047
inverse_depth_correction_ratio = 0.998047 / 0.288983 = 3.453656
normal_depth_correction_ratio = 0.288983 / 0.998047 = 0.289548
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 3.460414
- Actual inverse_depth_correction: 3.453656
- Error: 0.006759
- ✅ **Close to expected value** (resize is minimal)

---

### waymo_seg (val-2k)

**Input:**

- Original resolution: 1920×1280 (W×H)
- Original focal length: fx_actual = 2059.61
- Target resolution: (1918, 1274)
- Resize factor: 1
- First frame: `/home/cvlab/hsy/Datasets/waymo_seg/val/segment-10017090168044687777_6380_000_6400_000/FRONT/rgb/original/0000.jpg`

**Calculation:**

```
pre_h = 1280 × 1 = 1280
pre_w = 1920 × 1 = 1920
target_w, target_h = (1918, 1274)
target_w = 1918
target_h = 1274
small_resize_ratio = max(1918/1920, 1274/1280)
                   = max(0.998958, 0.995313)
                   = 0.998958
```

**Results:**

```
fx_ratio = 500.0 / 2059.61 = 0.242764
total_resize_ratio = 1 × 0.998958 = 0.998958
inverse_depth_correction_ratio = 0.998958 / 0.242764 = 4.114933
normal_depth_correction_ratio = 0.242764 / 0.998958 = 0.243017
```

**Interpretation:**

- Expected inverse_depth_correction (for minimal resize): 1/fx_ratio = 4.119224
- Actual inverse_depth_correction: 4.114933
- Error: 0.004291
- ✅ **Close to expected value** (resize is minimal)

---
