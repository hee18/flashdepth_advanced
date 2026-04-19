# Phase 2: TensorRT FP16 Optimization Guide (A to Z)

## 목차
1. [전체 구조 개요](#1-전체-구조-개요)
2. [사전 준비](#2-사전-준비)
3. [YOLO TensorRT 변환](#3-yolo-tensorrt-변환)
4. [Onepiece TensorRT 변환 (Hybrid 방식)](#4-onepiece-tensorrt-변환-hybrid-방식)
5. [CUDA 멀티스트림 파이프라인](#5-cuda-멀티스트림-파이프라인)
6. [벤치마크](#6-벤치마크)
7. [Jetson Orin 배포](#7-jetson-orin-배포)
8. [트러블슈팅](#8-트러블슈팅)

---

## 1. 전체 구조 개요

### Phase 1 (현재) — 순차 실행
```
Frame → resize 756x518
  → YOLO 추론 (PyTorch) ──── 순차 ────→ Onepiece 추론 (PyTorch)
  → 후처리 (mask + depth → 궤적)
```

### Phase 2 (목표) — CUDA 멀티스트림 병렬 실행
```
Frame → resize 756x518 → GPU 메모리에 올림 (1회)
  ├── Stream A: YOLO TRT FP16 ──────────┐
  ├── Stream B: Onepiece Hybrid TRT ────┤ (병렬)
  └── synchronize() ────────────────────┘
  → 후처리 (mask + depth → 궤적)
```

### 왜 병렬화가 가능한가?
YOLO와 Depth 추정은 같은 프레임을 입력으로 받지만 서로의 결과에 의존하지 않는다.
두 모델의 출력이 모두 완료된 후에야 "마스크 × depth map" 후처리가 시작된다.

### Onepiece 모델의 TRT 변환 전략

Onepiece 모델의 추론 흐름은 8단계로 구성된다:

```
Step 1: DINOv2 인코더 (Frozen)          ← TRT 변환 가능
Step 2: Scene Cut 감지                   ← 단순 연산, PyTorch 유지
Step 3: CLS Projection (Linear)          ← 경량, PyTorch 유지
Step 4: DPT Head 디코더 (Frozen)         ← TRT 변환 가능
Step 5: SpatialMamba (Stateful)          ← TRT 변환 불가 (hidden state)
Step 6: CLSMetricHead (MLP)              ← 경량, PyTorch 유지
Step 7: Final Head (Conv layers, Frozen) ← TRT 변환 가능
Step 8: Metric 변환 (산술 연산)           ← PyTorch 유지
```

**Hybrid 전략**: Step 1 + 4 + 7 (연산량의 ~80%)을 TRT로 변환하고,
Step 5 (SpatialMamba)는 PyTorch로 유지하여 hidden state를 관리한다.

#### 각 모듈의 연산량 비중 (ViT-L 기준)

| 모듈 | 파라미터 수 | 연산 비중 | TRT 변환 |
|------|-----------|----------|---------|
| DINOv2 (ViT-L) | ~304M | **~70%** | O |
| DPT Head | ~20M | ~10% | O |
| Final Head | ~0.1M | ~2% | O |
| SpatialMamba | ~8M | ~15% | X (stateful) |
| CLSMetricHead | ~0.02M | <1% | X (경량) |
| CLS Projection | ~0.26M | <1% | X (경량) |

**결론**: 전체 연산의 ~82%를 TRT로 가속할 수 있다.

---

## 2. 사전 준비

### 2.1 필요 패키지

```bash
# PC 환경 (Docker 내)
pip install tensorrt>=8.6
pip install onnx onnxruntime-gpu
pip install pycuda

# 버전 확인
python -c "import tensorrt; print(tensorrt.__version__)"
python -c "import onnx; print(onnx.__version__)"
```

### 2.2 디렉토리 구조

```
pedestrian_tracker/
├── trt/
│   ├── export_yolo.py             # YOLO → TRT 변환 스크립트
│   ├── export_depth_backbone.py   # Onepiece backbone → ONNX → TRT
│   ├── trt_depth_estimator.py     # TRT backbone + PyTorch Mamba hybrid 추론기
│   ├── trt_pipeline.py            # CUDA 멀티스트림 통합 파이프라인
│   └── benchmark.py               # PyTorch vs TRT 성능 비교
├── models/                         # 변환된 엔진 파일 저장
│   ├── yolo11m-seg_fp16.engine
│   ├── dinov2_dpt_fp16.engine
│   └── final_head_fp16.engine
```

---

## 3. YOLO TensorRT 변환

YOLO는 Ultralytics에서 TRT export를 공식 지원한다. 가장 간단한 단계.

### 3.1 변환 스크립트 (`trt/export_yolo.py`)

```python
from ultralytics import YOLO

def export_yolo_trt(model_path='yolo11m-seg.pt', imgsz=640, half=True):
    """
    YOLO 모델을 TensorRT FP16 엔진으로 변환.
    
    Args:
        model_path: YOLO .pt 가중치 경로
        imgsz: 추론 해상도 (YOLO는 stride 32의 배수 필요)
        half: FP16 사용 여부
    """
    model = YOLO(model_path)
    
    # export()가 .engine 파일을 자동 생성
    engine_path = model.export(
        format='engine',      # TensorRT
        half=half,            # FP16
        imgsz=imgsz,          # 입력 해상도
        device=0,             # GPU ID
        simplify=True,        # ONNX simplify
        workspace=4,          # TRT workspace (GB)
    )
    print(f"TRT engine saved: {engine_path}")
    return engine_path

if __name__ == '__main__':
    export_yolo_trt()
```

### 3.2 실행

```bash
# Docker 내에서
python pedestrian_tracker/trt/export_yolo.py
# 결과: yolo11m-seg.engine 생성
```

### 3.3 TRT 엔진으로 추론

```python
from ultralytics import YOLO

# .engine 파일을 직접 로드하면 TRT로 추론
model = YOLO('yolo11m-seg.engine')
results = model.track(frame, persist=True, classes=[0])
```

Ultralytics가 TRT 추론을 내부적으로 처리하므로 API 변경이 거의 없다.

### 3.4 주의사항
- **TRT 엔진은 GPU 아키텍처에 종속적**: PC에서 빌드한 엔진은 Jetson Orin에서 동작하지 않음
- **입력 해상도 고정**: export 시 지정한 `imgsz`와 다른 크기를 넣으면 에러
- **dynamic shape**를 원하면 `dynamic=True` 옵션 사용 (속도가 약간 느려짐)

---

## 4. Onepiece TensorRT 변환 (Hybrid 방식)

### 4.1 왜 전체 모델을 한 번에 TRT로 변환할 수 없는가?

SpatialMamba (Mamba2)가 **recurrent hidden state**를 유지한다:

```python
# SpatialMamba.forward_single_frame() 내부
for block in self.blocks:
    x = block(x, inference_params=self.inference_params)  # hidden state 참조/갱신
self.inference_params.seqlen_offset += x.shape[1]         # 시퀀스 위치 추적
```

TensorRT/ONNX는 이런 가변 상태를 지원하지 않는다.
따라서 **stateless 부분만 TRT로 변환**하고, stateful 부분은 PyTorch로 유지한다.

### 4.2 변환 대상 분리

**TRT 엔진 A: DINOv2 + DPT (backbone_dpt)**
```
Input:  frame [1, 3, 518, 756]
Output: dpt_features [1, 256, 148, 216]
        cls_token [1, 1024]
```

**PyTorch 유지: SpatialMamba + CLS Projection + CLSMetricHead**
```
Input:  dpt_features [1, 256, 148, 216], cls_token [1, 1024]
Output: post_mamba [1, 256, 148, 216], scale [1], shift [1]
```

**TRT 엔진 B: Final Head**
```
Input:  post_mamba [1, 256, 148, 216]
Output: relative_depth [1, 518, 756]
```

### 4.3 ONNX Export 스크립트 (`trt/export_depth_backbone.py`)

```python
"""
Onepiece backbone (DINOv2 + DPT) 및 Final Head를 ONNX로 변환.
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path

FLASHDEPTH_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(FLASHDEPTH_ROOT))

from flashdepth.model import FlashDepth


class BackboneDPT(nn.Module):
    """DINOv2 인코더 + DPT 디코더를 하나의 모듈로 래핑."""
    
    def __init__(self, model):
        super().__init__()
        self.pretrained = model.pretrained       # DINOv2
        self.depth_head = model.depth_head       # DPT
        self.encoder = model.encoder
        self.intermediate_layer_idx = model.intermediate_layer_idx
        self.cls_layer_indices = model.cls_layer_indices
        self.patch_size = model.patch_size
    
    def forward(self, x):
        B, C, H, W = x.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        # DINOv2 forward
        features, cls_token = self._get_features_and_cls(x)
        
        # DPT forward
        dpt_features = self.depth_head(features, patch_h, patch_w)
        
        return dpt_features, cls_token
    
    def _get_features_and_cls(self, x):
        layer_indices = self.intermediate_layer_idx[self.encoder]
        raw_outputs = self.pretrained._get_intermediate_layers_not_chunked(x, layer_indices)
        normed = [self.pretrained.norm(out) for out in raw_outputs]
        
        # CLS token (average of selected layers)
        selected_cls = [normed[idx][:, 0] for idx in self.cls_layer_indices]
        cls_token = torch.stack(selected_cls, dim=0).mean(dim=0)
        
        # Patch features (exclude CLS)
        features = [out[:, 1:] for out in normed]
        
        return features, cls_token


class FinalHead(nn.Module):
    """Final head: post_mamba → relative_depth."""
    
    def __init__(self, model):
        super().__init__()
        self.output_conv1 = model.depth_head.scratch.output_conv1
        self.output_conv2 = model.depth_head.scratch.output_conv2
        self.patch_size = model.patch_size
    
    def forward(self, post_mamba, patch_h, patch_w):
        out = self.output_conv1(post_mamba)
        target_h = patch_h * self.patch_size
        target_w = patch_w * self.patch_size
        out = torch.nn.functional.interpolate(
            out, (target_h, target_w), mode='bilinear', align_corners=True
        )
        out = self.output_conv2(out)
        return torch.nn.functional.relu(out).squeeze(1)


def export_backbone_dpt(model, input_shape, output_path):
    """BackboneDPT를 ONNX로 export."""
    backbone = BackboneDPT(model).eval().cuda()
    dummy = torch.randn(*input_shape, device='cuda')
    
    torch.onnx.export(
        backbone, dummy,
        str(output_path),
        input_names=['input'],
        output_names=['dpt_features', 'cls_token'],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported BackboneDPT: {output_path}")


def export_final_head(model, dpt_shape, patch_h, patch_w, output_path):
    """FinalHead를 ONNX로 export."""
    head = FinalHead(model).eval().cuda()
    dummy_post_mamba = torch.randn(*dpt_shape, device='cuda')
    # patch_h, patch_w는 상수이므로 trace 시 고정됨
    
    # trace-friendly wrapper
    class FinalHeadFixed(nn.Module):
        def __init__(self, head, ph, pw):
            super().__init__()
            self.head = head
            self.ph = ph
            self.pw = pw
        def forward(self, x):
            return self.head(x, self.ph, self.pw)
    
    fixed = FinalHeadFixed(head, patch_h, patch_w).eval().cuda()
    
    torch.onnx.export(
        fixed, dummy_post_mamba,
        str(output_path),
        input_names=['post_mamba'],
        output_names=['relative_depth'],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported FinalHead: {output_path}")
```

### 4.4 ONNX → TensorRT 변환

```bash
# BackboneDPT
trtexec \
    --onnx=models/backbone_dpt.onnx \
    --saveEngine=models/backbone_dpt_fp16.engine \
    --fp16 \
    --workspace=4096 \
    --verbose

# FinalHead
trtexec \
    --onnx=models/final_head.onnx \
    --saveEngine=models/final_head_fp16.engine \
    --fp16 \
    --workspace=2048 \
    --verbose
```

### 4.5 Hybrid 추론기 (`trt/trt_depth_estimator.py`)

```python
"""
TRT backbone + PyTorch Mamba hybrid 추론기.

추론 흐름:
  Frame → [TRT: BackboneDPT] → dpt_features, cls_token
        → [PyTorch: SpatialMamba] → post_mamba, cls_output
        → [PyTorch: CLSMetricHead] → scale, shift
        → [TRT: FinalHead] → relative_depth
        → metric_depth = scale * (100 / relative_depth) + shift
"""
import tensorrt as trt
import torch
import numpy as np
import pycuda.driver as cuda

class TRTEngine:
    """TensorRT 엔진 래퍼."""
    
    def __init__(self, engine_path, stream=None):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = stream or cuda.Stream()
        
        # 입출력 버퍼 사전할당
        self.inputs = {}
        self.outputs = {}
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = {'mem': device_mem, 'shape': shape, 'dtype': dtype}
            else:
                self.outputs[name] = {'mem': device_mem, 'shape': shape, 'dtype': dtype}
    
    def infer(self, input_dict):
        """
        Args:
            input_dict: {name: numpy_array} 입력 데이터
        Returns:
            {name: numpy_array} 출력 데이터
        """
        # 입력 복사 (Host → Device)
        for name, arr in input_dict.items():
            cuda.memcpy_htod_async(self.inputs[name]['mem'], arr.ravel(), self.stream)
        
        # 추론 실행
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        
        # 출력 복사 (Device → Host)
        results = {}
        for name, info in self.outputs.items():
            host_arr = np.empty(info['shape'], dtype=info['dtype'])
            cuda.memcpy_dtoh_async(host_arr, info['mem'], self.stream)
            results[name] = host_arr
        
        self.stream.synchronize()
        return results


class HybridDepthEstimator:
    """
    TRT BackboneDPT + PyTorch SpatialMamba + TRT FinalHead.
    """
    
    def __init__(self, backbone_engine_path, final_head_engine_path,
                 onepiece_checkpoint_path, config_path, device='cuda'):
        
        # TRT 엔진 로드
        self.backbone_engine = TRTEngine(backbone_engine_path)
        self.final_head_engine = TRTEngine(final_head_engine_path)
        
        # PyTorch 모듈 로드 (SpatialMamba, CLS Projection, CLSMetricHead만)
        # ... (기존 OnepieceDepthEstimator에서 해당 모듈만 추출)
        self.device = device
        self._load_pytorch_modules(onepiece_checkpoint_path, config_path)
        
        self.prev_cls = None
    
    def _load_pytorch_modules(self, checkpoint_path, config_path):
        """SpatialMamba, CLS Projection, CLSMetricHead만 로드."""
        # 전체 모델 로드 후 필요한 모듈만 추출
        import yaml
        from omegaconf import OmegaConf
        from flashdepth.model import FlashDepth
        
        with open(config_path) as f:
            config = OmegaConf.create(yaml.safe_load(f))
        
        model_config = dict(config.model)
        model_config['batch_size'] = 1
        model_config['use_metric_head'] = False
        model_config['use_onepiece'] = True
        model_config['spatial_mamba_layers'] = config.model.get('spatial_mamba_layers', 4)
        model_config['spatial_mamba_d_state'] = config.model.get('spatial_mamba_d_state', 256)
        model_config['spatial_mamba_d_conv'] = config.model.get('spatial_mamba_d_conv', 4)
        model_config['spatial_mamba_downsample'] = config.model.get('spatial_mamba_downsample', 0.1)
        model_config['onepiece_train_mode'] = config.get('train_mode', 'metric')
        model_config['hybrid_configs'] = config.get('hybrid_configs', None)
        scene_cut = config.get('scene_cut', {})
        model_config['scene_cut_tau'] = scene_cut.get('tau', 0.05) if scene_cut else 0.05
        
        full_model = FlashDepth(**model_config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        full_model.load_state_dict(state_dict, strict=False)
        
        # 필요한 모듈만 추출
        self.spatial_mamba = full_model.spatial_mamba.to(self.device).eval()
        self.cls_projection = full_model.cls_projection.to(self.device).eval()
        self.metric_head = full_model.onepiece_metric_head.to(self.device).eval()
        self.scene_cut_tau = full_model.scene_cut_detector.tau
        self.train_mode = full_model.onepiece_train_mode
        
        # DINOv2, DPT, final_head는 TRT가 대체하므로 삭제 (메모리 절약)
        del full_model
        torch.cuda.empty_cache()
    
    def reset(self):
        self.spatial_mamba.start_new_sequence()
        self.prev_cls = None
    
    @torch.no_grad()
    def estimate(self, frame_np):
        """
        Args:
            frame_np: [H, W, 3] uint8 BGR numpy (이미 리사이즈됨)
        Returns:
            metric_depth: [H, W] numpy (meters)
        """
        import cv2
        
        # 전처리: BGR→RGB, normalize, NCHW
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame_norm = (frame_rgb - mean) / std
        frame_nchw = np.transpose(frame_norm, (2, 0, 1))[np.newaxis]  # [1, 3, H, W]
        
        # ── Step 1: TRT BackboneDPT ──
        trt_out = self.backbone_engine.infer({'input': frame_nchw.astype(np.float16)})
        dpt_features = torch.from_numpy(trt_out['dpt_features']).to(self.device)
        cls_token = torch.from_numpy(trt_out['cls_token']).to(self.device)
        
        # ── Step 2: Scene Cut Detection (PyTorch) ──
        if self.prev_cls is not None:
            import torch.nn.functional as F
            cos_sim = F.cosine_similarity(
                F.normalize(cls_token, dim=-1),
                F.normalize(self.prev_cls, dim=-1), dim=-1
            )
            if (1.0 - cos_sim.mean()) > self.scene_cut_tau:
                self.spatial_mamba.start_new_sequence()
        self.prev_cls = cls_token
        
        # ── Step 3: CLS Projection (PyTorch) ──
        cls_projected = self.cls_projection(cls_token)
        
        # ── Step 4: SpatialMamba (PyTorch) ──
        post_mamba, cls_output = self.spatial_mamba.forward_single_frame(
            dpt_features.float(), cls_projected=cls_projected.float()
        )
        
        # ── Step 5: CLSMetricHead (PyTorch) ──
        scale, shift = self.metric_head(cls_output)
        
        # ── Step 6: TRT FinalHead ──
        post_mamba_np = post_mamba.cpu().numpy().astype(np.float16)
        head_out = self.final_head_engine.infer({'post_mamba': post_mamba_np})
        relative_depth = torch.from_numpy(head_out['relative_depth']).to(self.device)
        
        # ── Step 7: Metric Conversion (PyTorch) ──
        if self.train_mode == 'inverse':
            metric_depth = scale.unsqueeze(-1) * relative_depth + shift.unsqueeze(-1)
        else:
            depth_meters = 100.0 / (relative_depth.float() + 1e-8)
            metric_depth = scale.unsqueeze(-1) * depth_meters + shift.unsqueeze(-1)
        
        return metric_depth[0].cpu().numpy()
```

### 4.6 핵심 텐서 형상 요약

518x756 입력 기준 (ViT-L, patch_size=14):

```
patch_h = 518 // 14 = 37
patch_w = 756 // 14 = 54

Step 1 DINOv2 출력:
  encoder_features[i]: [1, 37*54=1998, 1024]
  cls_token:           [1, 1024]

Step 4 DPT 출력:
  dpt_features:        [1, 256, 148, 216]
  (h = 37*4 = 148, w = 54*4 = 216)

Step 5 SpatialMamba (downsample=0.1):
  downsampled:         [1, 256, 15, 22]  (148*0.1, 216*0.1)
  mamba input:         [1, 331, 256]     (1 + 15*22 = 331 tokens)
  post_mamba:          [1, 256, 148, 216]
  cls_output:          [1, 256]

Step 6 CLSMetricHead:
  scale:               [1, 1]
  shift:               [1, 1]

Step 7 FinalHead:
  relative_depth:      [1, 518, 756]

Step 8 최종:
  metric_depth:        [1, 518, 756]
```

---

## 5. CUDA 멀티스트림 파이프라인

### 5.1 핵심 개념

CUDA 스트림(Stream)은 GPU 작업의 **비동기 실행 큐**이다.
서로 다른 스트림에 제출된 작업은 **동시에 실행**될 수 있다.

```python
stream_a = torch.cuda.Stream()
stream_b = torch.cuda.Stream()

# Stream A에서 YOLO 실행 (비동기)
with torch.cuda.stream(stream_a):
    yolo_result = yolo_engine(frame)

# Stream B에서 Depth 실행 (동시)
with torch.cuda.stream(stream_b):
    depth_result = depth_engine(frame)

# 두 스트림 모두 완료될 때까지 대기
torch.cuda.synchronize()

# 후처리 (기본 스트림에서)
postprocess(yolo_result, depth_result)
```

### 5.2 구현 (`trt/trt_pipeline.py`)

```python
"""
CUDA 멀티스트림 TRT 파이프라인.

Stream A: YOLO TRT (세그멘테이션 + 트래킹)
Stream B: Onepiece Hybrid TRT (깊이 추정)
동기화 후: 마스크 × depth → 궤적 업데이트
"""
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO

# 기존 모듈 재사용
from tracker.trajectory import TrajectoryManager
from tracker.visualization import colorize_depth, draw_detections, plot_trajectories
from ped_utils.camera import calculate_lateral_position
from ped_utils.mask_ops import (create_mask_from_polygon, erode_mask,
                                 get_mask_center, extract_depth_from_mask)


class TRTPipeline:
    def __init__(self, config):
        # CUDA 스트림 생성
        self.stream_yolo = torch.cuda.Stream()
        self.stream_depth = torch.cuda.Stream()
        
        # YOLO TRT 모델
        self.yolo = YOLO(config['paths']['yolo_trt_engine'])  # .engine 파일
        
        # Onepiece Hybrid TRT 모델
        from trt.trt_depth_estimator import HybridDepthEstimator
        self.depth_estimator = HybridDepthEstimator(
            backbone_engine_path=config['paths']['backbone_trt_engine'],
            final_head_engine_path=config['paths']['final_head_trt_engine'],
            onepiece_checkpoint_path=config['paths']['onepiece_checkpoint'],
            config_path=config['paths']['onepiece_config'],
        )
        
        self.trajectory_mgr = TrajectoryManager(
            ema_alpha=config['trajectory']['ema_alpha'],
        )
        
        # 카메라/설정
        self.config = config
    
    def process_frame(self, frame, frame_idx):
        """
        멀티스트림 병렬 처리.
        
        Args:
            frame: [H, W, 3] BGR uint8 (이미 리사이즈됨)
            frame_idx: 프레임 인덱스
        Returns:
            vis_frame, detections_info
        """
        # ── 결과 저장용 컨테이너 ──
        yolo_result = [None]
        depth_result = [None]
        
        # ── Stream A: YOLO 추론 ──
        with torch.cuda.stream(self.stream_yolo):
            results = self.yolo.track(
                frame, persist=True, classes=[0],
                conf=self.config['yolo']['confidence'],
                verbose=False,
            )
            yolo_result[0] = results
        
        # ── Stream B: Depth 추론 (동시 실행) ──
        with torch.cuda.stream(self.stream_depth):
            depth_map = self.depth_estimator.estimate(frame)
            depth_result[0] = depth_map
        
        # ── 동기화: 두 스트림 모두 완료 대기 ──
        torch.cuda.synchronize()
        
        # ── 후처리 (기본 스트림) ──
        detections = self._parse_yolo_results(yolo_result[0])
        depth_map = depth_result[0]
        
        detections_info = self._process_detections(
            detections, depth_map, frame_idx
        )
        
        return depth_map, detections_info
    
    def _parse_yolo_results(self, results):
        """YOLO 결과를 detection dict 리스트로 변환."""
        result = results[0]
        if result.boxes.id is None or result.masks is None:
            return []
        
        detections = []
        for tid, cls, box, conf, mask_pts in zip(
            result.boxes.id.int().cpu().tolist(),
            result.boxes.cls.int().cpu().tolist(),
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.masks.xy
        ):
            detections.append({
                'track_id': tid, 'class_id': cls,
                'bbox': box, 'mask_points': mask_pts,
                'confidence': float(conf),
            })
        return detections
    
    def _process_detections(self, detections, depth_map, frame_idx):
        """마스크 기반 depth 추출 + 궤적 업데이트."""
        h, w = depth_map.shape
        fx = self.config['camera']['fx']
        cx = self.config['camera']['cx']
        max_depth = self.config['depth']['max_depth']
        
        detections_info = []
        for det in detections:
            mask = create_mask_from_polygon(det['mask_points'], (h, w))
            center = get_mask_center(mask)
            if center is None:
                continue
            cx_px, cy_px = center
            
            raw_depth = extract_depth_from_mask(depth_map, mask)
            if raw_depth is None or raw_depth <= 0 or raw_depth > max_depth:
                raw_depth = self.trajectory_mgr.get_last_depth(det['track_id'])
                if raw_depth is None:
                    continue
            
            lateral = calculate_lateral_position(raw_depth, cx_px, fx, cx)
            self.trajectory_mgr.update(det['track_id'], raw_depth, lateral, frame_idx)
            smoothed = self.trajectory_mgr.get_last_depth(det['track_id'])
            lat_smooth = calculate_lateral_position(smoothed, cx_px, fx, cx)
            
            detections_info.append({
                'track_id': det['track_id'],
                'bbox': det['bbox'],
                'depth': smoothed,
                'lateral': lat_smooth,
            })
        
        return detections_info
```

### 5.3 멀티스트림의 실제 병렬화 조건

**주의**: 단순히 다른 스트림에 넣었다고 무조건 병렬 실행되는 것은 아니다.

병렬화가 실제로 일어나려면:
1. **GPU 리소스에 여유가 있어야 함** — 한 모델이 GPU SM을 100% 점유하면 다른 스트림은 대기
2. **메모리 전송과 연산이 겹쳐야 함** — H2D 전송과 커널 실행이 동시에 가능
3. **모델 크기가 적당해야 함** — Orin의 SM 수가 제한적이므로 큰 모델 2개는 직렬화될 수 있음

**현실적 기대치**:
- PC (RTX A6000, 48GB): YOLO와 Depth가 실제 병렬 실행될 가능성 높음
- Jetson Orin AGX (2048 CUDA cores): 부분 병렬화 예상, ~20-40% 속도 향상
- Jetson Orin NX (1024 CUDA cores): 거의 직렬화될 수 있음

**검증 방법**: `nsys profile`로 실제 커널 실행 타이밍을 확인
```bash
nsys profile python trt/trt_pipeline.py --video test.mp4
nsys stats report.nsys-rep
```

---

## 6. 벤치마크

### 6.1 벤치마크 스크립트 (`trt/benchmark.py`)

```python
"""
PyTorch vs TRT 성능 비교.

측정 항목:
- FPS (프레임/초)
- Latency (ms/프레임)
- GPU 메모리 사용량
- Depth 출력 오차 (TRT vs PyTorch)
"""
import time
import torch
import numpy as np


def benchmark_model(model_fn, frame, num_warmup=10, num_iter=100):
    """모델 추론 속도 측정."""
    # Warmup
    for _ in range(num_warmup):
        model_fn(frame)
    torch.cuda.synchronize()
    
    # 측정
    start = time.time()
    for _ in range(num_iter):
        model_fn(frame)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    fps = num_iter / elapsed
    latency = elapsed / num_iter * 1000  # ms
    
    return {'fps': fps, 'latency_ms': latency}


def compare_outputs(pytorch_depth, trt_depth):
    """PyTorch vs TRT 출력 비교."""
    diff = np.abs(pytorch_depth - trt_depth)
    return {
        'max_diff': float(diff.max()),
        'mean_diff': float(diff.mean()),
        'relative_error': float(diff.mean() / (np.abs(pytorch_depth).mean() + 1e-8)),
    }
```

### 6.2 예상 성능

| 구성 | YOLO (ms) | Depth (ms) | 합계 (ms) | FPS |
|------|-----------|------------|-----------|-----|
| PyTorch 순차 | ~30 | ~130 | ~160 | ~6 |
| TRT FP16 순차 | ~8 | ~45 | ~53 | ~19 |
| TRT FP16 멀티스트림 | max(8, 45) | — | ~45 | ~22 |

---

## 7. Jetson Orin 배포

### 7.1 왜 반드시 Orin에서 빌드해야 하는가?

TensorRT 엔진(.engine)은 다음 조건에 종속적이다:
- **GPU 아키텍처** (SM version): PC의 sm_86/89 ≠ Orin의 sm_87
- **TensorRT 버전**: PC와 Orin의 TRT 버전이 다를 수 있음
- **CUDA 버전**: JetPack에 포함된 CUDA와 PC CUDA가 다름

따라서:
- **ONNX 파일**: PC에서 생성 가능 (플랫폼 독립적)
- **TRT 엔진**: 반드시 Orin에서 `trtexec`로 빌드

### 7.2 PC에서 준비할 것

```bash
# 1. ONNX 모델 생성
python trt/export_depth_backbone.py  # → models/backbone_dpt.onnx
python trt/export_depth_backbone.py --final-head  # → models/final_head.onnx

# 2. 전송할 파일 목록
pedestrian_tracker/          # 프로젝트 코드 전체
models/backbone_dpt.onnx     # Onepiece backbone ONNX
models/final_head.onnx       # Onepiece final head ONNX
yolo11m-seg.pt               # YOLO 가중치 (Orin에서 TRT 변환)
checkpoints/onepiece.pth     # Mamba 가중치 (PyTorch용)
mamba/                       # Mamba 소스 (Orin에서 빌드)
```

### 7.3 Orin 환경 세팅 (단계별)

#### Step 1: JetPack 확인
```bash
# Orin에서 실행
cat /etc/nv_tegra_release    # L4T 버전 확인
dpkg -l | grep tensorrt      # TRT 버전 확인
nvcc --version               # CUDA 버전 확인

# 필요 시 JetPack 업데이트
sudo apt-get update && sudo apt-get install nvidia-jetpack
```

#### Step 2: Python 환경
```bash
# venv 사용 (conda는 ARM64에서 불안정)
python3 -m venv ~/ped_env
source ~/ped_env/bin/activate

# Jetson용 PyTorch (반드시 NVIDIA 공식 빌드)
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/
pip install torch-2.1.0a0+...nv...-cp310-linux_aarch64.whl
pip install torchvision-0.16.1a0+...-cp310-linux_aarch64.whl

# 기타
pip install ultralytics opencv-python numpy matplotlib pyyaml omegaconf
pip install pycuda
```

#### Step 3: Mamba2 빌드 (Orin ARM64에서)
```bash
# causal-conv1d (CUDA 커널 포함, 소스 빌드 필수)
pip install causal-conv1d --no-binary :all:

# mamba-ssm
cd mamba/
MAMBA_FORCE_BUILD=TRUE pip install -e . --no-build-isolation

# 빌드 실패 시 체크:
# - CUDA toolkit이 PATH에 있는지: which nvcc
# - torch.cuda.is_available() == True 인지
# - gcc/g++ 버전이 호환되는지
```

#### Step 4: 파일 전송 (PC → Orin)
```bash
# PC에서 실행
scp -r pedestrian_tracker/ user@orin-ip:~/ped_tracker/
scp models/*.onnx user@orin-ip:~/ped_tracker/models/
scp yolo11m-seg.pt user@orin-ip:~/ped_tracker/
scp -r mamba/ user@orin-ip:~/ped_tracker/mamba/
scp checkpoints/onepiece.pth user@orin-ip:~/ped_tracker/checkpoints/
```

#### Step 5: Orin에서 TRT 엔진 빌드
```bash
# YOLO
python -c "
from ultralytics import YOLO
model = YOLO('yolo11m-seg.pt')
model.export(format='engine', half=True, imgsz=640, device=0)
"

# Onepiece BackboneDPT
trtexec \
    --onnx=models/backbone_dpt.onnx \
    --saveEngine=models/backbone_dpt_fp16.engine \
    --fp16 \
    --workspace=2048    # Orin 메모리 제한 고려

# Onepiece FinalHead
trtexec \
    --onnx=models/final_head.onnx \
    --saveEngine=models/final_head_fp16.engine \
    --fp16 \
    --workspace=1024
```

#### Step 6: 실행 및 검증
```bash
# PyTorch 모드 (기능 확인)
python run_tracker.py --video test.mp4

# TRT 모드
python run_tracker.py --video test.mp4 --mode trt

# 벤치마크
python trt/benchmark.py --compare
```

#### Step 7: 성능 프로파일링
```bash
# tegrastats로 GPU/CPU/메모리 모니터링
sudo tegrastats --interval 1000

# nsys로 CUDA 커널 프로파일링
nsys profile -o ped_tracker_profile python run_tracker.py --mode trt --max-frames 50
nsys stats ped_tracker_profile.nsys-rep
```

### 7.4 Orin 성능 예상치

| 모델 | Orin NX (8GB) | Orin AGX (32GB) |
|------|--------------|-----------------|
| YOLO 11m TRT FP16 | ~15 FPS | ~25 FPS |
| Onepiece Hybrid TRT | ~8 FPS | ~15 FPS |
| 멀티스트림 병렬 | ~10 FPS | ~18 FPS |

### 7.5 Orin 메모리 최적화 팁

```bash
# 최대 성능 모드
sudo nvpmodel -m 0        # MAXN 모드
sudo jetson_clocks         # 클럭 최대화

# GPU 메모리 사용 확인
python -c "import torch; print(torch.cuda.memory_summary())"
```

---

## 8. 트러블슈팅

### 8.1 ONNX Export 실패

**증상**: `torch.onnx.export()` 시 `Unsupported operator` 에러

**원인**: DINOv2의 일부 연산 (xformers memory efficient attention 등)이 ONNX 미지원

**해결**:
```python
# xformers 대신 기본 attention 사용
# dinov2.py의 MemEffAttention을 Attention으로 교체
model.pretrained.blocks[i].attn = StandardAttention(...)
```
또는:
```python
# torch.onnx.export에 custom opset 등록
torch.onnx.export(..., opset_version=17, 
                   custom_opsets={"custom_domain": 1})
```

### 8.2 TRT 엔진 빌드 실패

**증상**: `trtexec`에서 `Unsupported layer type` 에러

**해결**: ONNX simplify 적용
```bash
pip install onnx-simplifier
python -m onnxsim models/backbone_dpt.onnx models/backbone_dpt_sim.onnx
trtexec --onnx=models/backbone_dpt_sim.onnx ...
```

### 8.3 FP16 정밀도 문제

**증상**: TRT FP16 출력이 PyTorch FP32와 크게 다름

**해결**: 특정 레이어를 FP32로 유지
```bash
trtexec --onnx=model.onnx --fp16 \
    --layerPrecisions=*/LayerNorm:fp32,*/Softmax:fp32
```

### 8.4 Mamba2 Orin 빌드 실패

**증상**: `causal-conv1d` 또는 `mamba-ssm` 컴파일 에러

**해결**:
```bash
# CUDA 아키텍처 명시
export TORCH_CUDA_ARCH_LIST="8.7"  # Orin의 SM 87

# gcc 버전 확인 (JetPack 6.x는 gcc-11 권장)
gcc --version

# ninja 사용 비활성화 (빌드 문제 시)
MAX_JOBS=1 pip install causal-conv1d --no-binary :all:
```

### 8.5 CUDA 멀티스트림이 실제로 병렬화되지 않음

**증상**: nsys 프로파일에서 두 스트림이 직렬 실행됨

**원인**: GPU SM이 하나의 모델에 의해 100% 점유됨

**해결**:
- 모델 크기를 줄임 (YOLO 11m → 11s, ViT-L → ViT-S)
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`로 SM 할당 비율 제어:
```bash
# MPS 서버 시작
nvidia-cuda-mps-control -d
echo "set_active_thread_percentage 50" | nvidia-cuda-mps-control
```

---

## 전체 구현 순서 요약

```
1. YOLO TRT 변환
   └── ultralytics export → .engine (30분)

2. Onepiece ONNX Export
   ├── BackboneDPT → .onnx
   └── FinalHead → .onnx (1-2시간, 디버깅 포함)

3. ONNX → TRT 변환
   └── trtexec → .engine (각 10-30분)

4. Hybrid 추론기 구현
   └── trt_depth_estimator.py (2-3시간)

5. CUDA 멀티스트림 파이프라인
   └── trt_pipeline.py (2-3시간)

6. 벤치마크 & 검증
   └── PyTorch vs TRT 비교 (1-2시간)

7. Jetson Orin 배포
   ├── 환경 세팅 (2-4시간)
   ├── Mamba2 빌드 (1-3시간, 트러블슈팅 포함)
   ├── TRT 엔진 빌드 (30분-1시간)
   └── 테스트 & 프로파일링 (2-3시간)
```
