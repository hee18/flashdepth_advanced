# Model Processing Resolutions by Dataset

각 데이터셋별로 각 depth estimation 모델이 실제로 처리하는 해상도를 정리한 문서입니다.

---

## ETH3D

**Native Resolution**: 6048×4032 (24.4M pixels)

| Model | Internal Processing Resolution | Downscale | Output Resolution |
|-------|-------------------------------|-----------|-------------------|
| **DepthAnythingV2** | 518×518 | 11.7× | 6048×4032 |
| **Metric3D-v2 (ViT-L)** | ~1596×1064 (padded to 616×1064) | 3.8× | 6048×4032 |
| **Metric3D-v1 (ConvNeXt-L)** | ~1824×1216 (padded to 544×1216) | 3.3× | 6048×4032 |
| **UniDepth-v1** | 6048×4032 (원본 그대로) | 1.0× | 6048×4032 |
| **UniDepth-v2** | ~952×644 (adaptive, 600k pixels) | 6.4× | 6048×4032 |
| **ZoeDepth** | Unknown (model-internal) | - | 6048×4032 |
| **DepthPro** | 1536×1536 | 10.4× | 6048×4032 |
| **CUT3R** | 512×512 | 11.8× | 6048×4032 |

**특징**:
- ETH3D의 매우 높은 해상도로 인해 모든 모델이 대폭 다운샘플링
- UniDepth-v1만 원본 해상도 그대로 처리 (메모리 사용량 극대)
- CUT3R, DepthAnythingV2가 가장 많이 다운샘플 (512², 518²)

---

## Sintel

**Native Resolution**: 1024×436 (446k pixels, aspect ratio 2.35:1)

| Model | Internal Processing Resolution | Aspect Distortion | Output Resolution |
|-------|-------------------------------|-------------------|-------------------|
| **DepthAnythingV2** | 518×518 | ⚠️ 2.35:1 → 1:1 | 1024×436 |
| **Metric3D-v2 (ViT-L)** | ~1024×436 (padded to 616×1064) | ✅ Preserved | 1024×436 |
| **Metric3D-v1 (ConvNeXt-L)** | ~1024×436 (padded to 544×1216) | ✅ Preserved | 1024×436 |
| **UniDepth-v1** | 1024×436 (원본 그대로) | ✅ Preserved | 1024×436 |
| **UniDepth-v2** | ~1022×434 (adaptive, ~446k pixels) | ✅ Preserved | 1024×436 |
| **ZoeDepth** | Unknown (model-internal) | Likely preserved | 1024×436 |
| **DepthPro** | 1536×1536 | ⚠️ 2.35:1 → 1:1 | 1024×436 |
| **CUT3R** | 512×512 | ⚠️ 2.35:1 → 1:1 | 1024×436 |

**특징**:
- **극단적 wide aspect ratio (2.35:1)로 square resize 모델들이 심각한 왜곡**
- DepthAnythingV2, DepthPro, CUT3R: 가로를 세로 길이만큼 찌그러뜨림
- Metric3D, UniDepth: aspect ratio 보존으로 기하학적 정확도 유지
- UniDepth-v2가 원본과 거의 동일한 해상도로 처리 (~1022×434)

---

## Waymo

**Native Resolution**: 1920×1280 (2.46M pixels, aspect ratio 3:2)

| Model | Internal Processing Resolution | Downscale | Output Resolution |
|-------|-------------------------------|-----------|-------------------|
| **DepthAnythingV2** | 518×518 | 4.8× | 1920×1280 |
| **Metric3D-v2 (ViT-L)** | ~1596×1064 (padded to 616×1064) | 1.2× | 1920×1280 |
| **Metric3D-v1 (ConvNeXt-L)** | ~1824×1216 (padded to 544×1216) | 1.1× | 1920×1280 |
| **UniDepth-v1** | 1920×1280 (원본 그대로) | 1.0× | 1920×1280 |
| **UniDepth-v2** | ~952×630 (adaptive, 600k pixels) | 2.0× | 1920×1280 |
| **ZoeDepth** | Unknown (model-internal) | - | 1920×1280 |
| **DepthPro** | 1536×1536 | 1.3× | 1920×1280 |
| **CUT3R** | 512×512 | 3.8× | 1920×1280 |

**특징**:
- 비교적 표준적인 해상도로 대부분의 모델이 무난하게 처리
- Metric3D-v1이 가장 높은 처리 해상도 유지 (~1824×1216)
- DepthPro도 1536²로 비교적 고해상도 처리

---

## UnrealStereo4K

**Native Resolution**: 3840×2160 (8.29M pixels, 4K)

| Model | Internal Processing Resolution | Downscale | Output Resolution |
|-------|-------------------------------|-----------|-------------------|
| **DepthAnythingV2** | 518×518 | 16.0× | 3840×2160 |
| **Metric3D-v2 (ViT-L)** | ~1892×1064 (padded to 616×1064) | 2.0× | 3840×2160 |
| **Metric3D-v1 (ConvNeXt-L)** | ~2164×1216 (padded to 544×1216) | 1.8× | 3840×2160 |
| **UniDepth-v1** | 3840×2160 (원본 그대로) | 1.0× | 3840×2160 |
| **UniDepth-v2** | ~1036×574 (adaptive, 600k pixels) | 3.6× | 3840×2160 |
| **ZoeDepth** | Unknown (model-internal) | - | 3840×2160 |
| **DepthPro** | 1536×1536 | 2.5× | 3840×2160 |
| **CUT3R** | 512×512 | 7.5× | 3840×2160 |

**특징**:
- 4K 해상도로 인해 모든 모델이 상당히 다운샘플링
- Metric3D-v1이 가장 높은 해상도 유지 (~2164×1216, 2.6M pixels)
- DepthAnythingV2, CUT3R은 16배, 7.5배 다운샘플로 디테일 손실 우려

---

## VKITTI

**Native Resolution**: 1242×375 (466k pixels, aspect ratio 3.31:1)

| Model | Internal Processing Resolution | Aspect Distortion | Output Resolution |
|-------|-------------------------------|-------------------|-------------------|
| **DepthAnythingV2** | 518×518 | ⚠️ 3.31:1 → 1:1 | 1242×375 |
| **Metric3D-v2 (ViT-L)** | ~1242×375 (padded to 616×1064) | ✅ Preserved (heavy padding) | 1242×375 |
| **Metric3D-v1 (ConvNeXt-L)** | ~1242×375 (padded to 544×1216) | ✅ Preserved (heavy padding) | 1242×375 |
| **UniDepth-v1** | 1242×375 (원본 그대로) | ✅ Preserved | 1242×375 |
| **UniDepth-v2** | ~1232×364 (adaptive, ~466k pixels) | ✅ Preserved | 1242×375 |
| **ZoeDepth** | Unknown (model-internal) | Likely preserved | 1242×375 |
| **DepthPro** | 1536×1536 | ⚠️ 3.31:1 → 1:1 | 1242×375 |
| **CUT3R** | 512×512 | ⚠️ 3.31:1 → 1:1 | 1242×375 |

**특징**:
- **가장 극단적인 aspect ratio (3.31:1)**
- Square resize 모델들이 극심한 왜곡: 가로를 3배 이상 압축
- Metric3D: padding이 많이 추가됨 (height 375 → 1064/1216)
- UniDepth-v2가 거의 원본 해상도로 처리 (~1232×364)

---

## 전체 비교 요약

### 처리 해상도 한눈에 보기

| Dataset | Native | DAv2 | M3D-v2 | M3D-v1 | UD-v1 | UD-v2 | ZoeD | DPro | CUT3R |
|---------|--------|------|--------|--------|-------|-------|------|------|-------|
| **ETH3D** | 6048×4032 | 518² | 1596×1064 | 1824×1216 | **원본** | 952×644 | ? | 1536² | 512² |
| **Sintel** | 1024×436 | 518² | 1024×436 | 1024×436 | **원본** | 1022×434 | ? | 1536² | 512² |
| **Waymo** | 1920×1280 | 518² | 1596×1064 | 1824×1216 | **원본** | 952×630 | ? | 1536² | 512² |
| **Unreal4K** | 3840×2160 | 518² | 1892×1064 | 2164×1216 | **원본** | 1036×574 | ? | 1536² | 512² |
| **VKITTI** | 1242×375 | 518² | 1242×375 | 1242×375 | **원본** | 1232×364 | ? | 1536² | 512² |

**범례**:
- `²`: Square resize (aspect ratio 왜곡)
- `원본`: 원본 해상도 그대로 처리
- `?`: Model-internal (알 수 없음)
- 숫자: Aspect-preserving resize (with padding for Metric3D)

---

## 모델별 특징 정리

### 1. DepthAnythingV2
- **전략**: 고정 518×518 square resize
- **장점**: 빠른 추론 속도, 일관된 성능
- **단점**: Aspect ratio 왜곡 (특히 Sintel, VKITTI에서 심각)
- **추천**: 표준 aspect ratio 이미지, 속도 우선

### 2. Metric3D
- **전략**: Aspect-preserving resize + padding
- **v2 (ViT-L)**: 616×1064로 패딩 (1.7M pixels 처리)
- **v1 (ConvNeXt-L)**: 544×1216로 패딩 (2.2M pixels 처리)
- **장점**: 기하학적 정확도 보존, 안정적
- **단점**: Padding overhead
- **추천**: 정확도 우선, 극단적 aspect ratio

### 3. UniDepth-v1
- **전략**: 원본 해상도 그대로 처리
- **장점**: 최대 디테일 보존
- **단점**: 메모리 사용량 극대 (특히 ETH3D, Unreal4K)
- **추천**: 충분한 VRAM, 최고 품질 필요 시

### 4. UniDepth-v2
- **전략**: Adaptive resolution (200k~600k pixels)
- **계산 방식**:
  ```
  target_pixels = clip(native_pixels, 200k, 600k)
  resize_factor = sqrt(target_pixels / native_pixels)
  new_size = (H * factor, W * factor) rounded to 14의 배수
  ```
- **장점**: 자동 최적화, aspect ratio 보존
- **단점**: 예측 불가능한 처리 크기
- **추천**: 범용 사용, 다양한 해상도

### 5. ZoeDepth
- **전략**: Model-internal preprocessing
- **장점**: 모델 최적화된 처리
- **단점**: 실제 처리 해상도 불명확
- **추천**: 표준 사용 케이스

### 6. DepthPro
- **전략**: 고정 1536×1536 square resize
- **장점**: 비교적 높은 해상도 처리
- **단점**: Aspect ratio 왜곡, 느린 추론
- **추천**: 고해상도 필요하지만 속도 덜 중요

### 7. CUT3R
- **전략**: 고정 512×512 square resize
- **장점**: 매우 빠른 추론
- **단점**: 심각한 다운샘플링 + aspect ratio 왜곡
- **추천**: 실시간 처리, 저사양 환경

---

## Aspect Ratio 왜곡 정도

### Sintel (2.35:1)
- **심각**: DAv2 (2.35→1), DPro (2.35→1), CUT3R (2.35→1)
- **없음**: M3D-v1/v2, UD-v1/v2, ZoeD

### VKITTI (3.31:1)
- **매우 심각**: DAv2 (3.31→1), DPro (3.31→1), CUT3R (3.31→1)
- **없음**: M3D-v1/v2, UD-v1/v2, ZoeD

### 표준 해상도 (Waymo 3:2, ETH3D/Unreal4K ~3:2)
- **중간**: DAv2, DPro, CUT3R (여전히 왜곡 있음)
- **없음**: M3D-v1/v2, UD-v1/v2, ZoeD

---

## 추론 속도 vs 정확도 Trade-off

### 속도 우선 (빠름 → 느림)
1. **CUT3R** (512²) - 가장 빠름
2. **DepthAnythingV2** (518²)
3. **UniDepth-v2** (adaptive 200k~600k)
4. **ZoeDepth** (?)
5. **Metric3D-v2** (616×1064)
6. **DepthPro** (1536²)
7. **Metric3D-v1** (544×1216)
8. **UniDepth-v1** (원본) - 가장 느림

### 정확도 우선 (낮음 → 높음, 일반적 경향)
1. **CUT3R** (512², aspect 왜곡)
2. **DepthAnythingV2** (518², aspect 왜곡)
3. **UniDepth-v2** (adaptive, aspect 보존)
4. **DepthPro** (1536², aspect 왜곡)
5. **Metric3D-v2** (1.7M pixels, aspect 보존)
6. **ZoeDepth** (?)
7. **Metric3D-v1** (2.2M pixels, aspect 보존)
8. **UniDepth-v1** (원본 해상도, 최고 디테일)

*주의: 실제 정확도는 모델 아키텍처, 학습 데이터, 최적화 등에 크게 의존*

---

## UniDepth v1 vs v2 상세 비교

### UniDepth-v1
```json
{
  "data": {
    "image_shape": [462, 616]  // 실제로는 사용 안함
  }
}
```
- **실제 동작**: 입력 이미지를 원본 해상도 그대로 처리
- **메모리**: 입력 크기에 비례 (ETH3D: 24M pixels)
- **속도**: 느림 (특히 고해상도)
- **품질**: 최고 (디테일 보존)

### UniDepth-v2
```json
{
  "data": {
    "augmentations": {
      "shape_constraints": {
        "ratio_bounds": [0.5, 2.5],
        "pixels_max": 600000,
        "pixels_min": 200000,
        "shape_mult": 14
      }
    }
  }
}
```
- **실제 동작**:
  1. Aspect ratio를 0.5~2.5 범위로 제한 (padding)
  2. 픽셀 수를 200k~600k로 조정 (resize)
  3. 14의 배수로 맞춤 (model requirement)
- **메모리**: 최대 600k pixels로 제한
- **속도**: 빠름 (일정한 메모리 사용)
- **품질**: 우수 (adaptive로 최적화)

### 처리 해상도 비교 예시

| 입력 해상도 | Native Pixels | v1 처리 | v2 처리 | v2 다운샘플 |
|------------|--------------|---------|---------|-----------|
| 640×480 | 307k | 640×480 | ~616×462 | 0.99× (거의 원본) |
| 1920×1080 | 2.07M | 1920×1080 | ~896×504 | 2.3× |
| 3840×2160 | 8.29M | 3840×2160 | ~1036×574 | 3.7× |
| 6048×4032 | 24.4M | 6048×4032 | ~952×644 | 6.4× |

**결론**:
- v1은 고해상도에서 메모리 부족 위험
- v2는 자동으로 적절한 크기로 조정 (실용적)

---

## 실제 값 확인 방법

### UniDepth v2 처리 해상도 로깅

`refer_test/UniDepth/unidepth/models/unidepthv2/unidepthv2.py` 수정:

```python
# Line 284-286 근처에 추가
resize_factor, (new_H, new_W) = get_resize_factor(
    (padded_H, padded_W), pixels_bounds
)
print(f"[UniDepth-v2] Input: {H}×{W} → Padded: {padded_H}×{padded_W} → Processing: {new_H}×{new_W} ({new_H*new_W:,} pixels)")
```

### Adapter에서 확인

`adapters/unidepth_adapter.py`의 `inference` 메서드에 추가:

```python
def inference(self, image, intrinsics=None):
    ...
    # Run inference
    with torch.no_grad():
        predictions = self.model.infer(rgb_torch, camera)

    # 로그 추가
    depth = predictions["depth"].squeeze()
    print(f"[UniDepth Adapter] Input: {image.shape[2:]} → Output: {depth.shape}")
    ...
```

### 테스트 실행

```bash
# 각 데이터셋에서 실제 해상도 확인
python test_comparison.py --method unidepth --version v2 --dataset eth3d --gpu 0
python test_comparison.py --method unidepth --version v2 --dataset sintel --gpu 0
python test_comparison.py --method unidepth --version v2 --dataset waymo --gpu 0
```

---

## 테스트 명령어

```bash
# ETH3D
python test_comparison.py --method depthanythingv2 --dataset eth3d --gpu 0
python test_comparison.py --method metric3d --version v2 --dataset eth3d --gpu 0
python test_comparison.py --method unidepth --version v2 --dataset eth3d --gpu 0
python test_comparison.py --method depthpro --dataset eth3d --gpu 0

# Sintel
python test_comparison.py --method depthanythingv2 --dataset sintel --gpu 0
python test_comparison.py --method metric3d --version v2 --dataset sintel --gpu 0
python test_comparison.py --method unidepth --version v2 --dataset sintel --gpu 0

# Waymo
python test_comparison.py --method depthanythingv2 --dataset waymo --gpu 0
python test_comparison.py --method metric3d --version v2 --dataset waymo --gpu 0
python test_comparison.py --method unidepth --version v2 --dataset waymo --gpu 0

# UnrealStereo4K (single sequence)
python test_comparison.py --method depthanythingv2 --dataset unrealstereo4k --seq 0 --gpu 0
python test_comparison.py --method depthpro --dataset unrealstereo4k --seq 0 --gpu 0

# VKITTI (clone only)
python test_comparison.py --method depthanythingv2 --dataset vkitti --only-clone true --gpu 0
python test_comparison.py --method unidepth --version v2 --dataset vkitti --only-clone true --gpu 0
```

---

## 참고 자료

### 코드 위치
- Dataset native resolutions: `dataloaders/comparison_dataset.py`
- Model adapters: `adapters/*_adapter.py`
- Test framework: `test_comparison.py`

### 해상도 정보 출처
- **DepthAnythingV2**: `adapters/depth_anything_v2_adapter.py:63`
- **Metric3D**: `adapters/metric3d_adapter.py:100-106, 124-210`
- **UniDepth-v1**: 원본 해상도 사용 (no resize)
- **UniDepth-v2**: `refer_test/UniDepth/unidepth/models/unidepthv2/unidepthv2.py:61-85, 247-286`
  - Config: `refer_test/UniDepth/configs/config_v2_*.json`
- **ZoeDepth**: Internal preprocessing (미공개)
- **DepthPro**: `adapters/depthpro_adapter.py:102-171`
- **CUT3R**: `adapters/cut3r_adapter.py:26, 123`

### 중요 사항

1. **모든 모델의 최종 출력은 원본 해상도로 업샘플링됨**
2. **내부 처리 해상도가 작을수록**:
   - ✅ 빠른 추론, 적은 메모리
   - ❌ 디테일 손실, 낮은 정확도 가능성
3. **Aspect ratio 왜곡은 기하학적 정확도에 영향**
4. **UniDepth-v2의 실제 처리 해상도는 입력에 따라 가변적** (예상값 제공)
