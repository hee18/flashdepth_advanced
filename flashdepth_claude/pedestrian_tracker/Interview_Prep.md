# PT 직무면접 대비: 보행자 경로 추정 + TensorRT 최적화

## 1. 프로젝트 개요

### 한 줄 요약
"YOLO 11m 인스턴스 세그멘테이션 + Mamba2 기반 temporal metric depth 모델(Onepiece)을 결합하여 단안 카메라로 보행자의 3D 경로를 추정하고, TensorRT FP16 + CUDA 멀티스트림으로 Jetson Orin에 최적화한 프로젝트"

### 파이프라인
```
단안 카메라 영상
  → resize (756×518, 1회)
  ├── YOLO 11m-seg: 보행자 검출 + 세그멘테이션 + 트래킹
  └── Onepiece: temporal metric depth 추정
  → 마스크 × depth → 보행자별 metric depth 추출
  → 카메라 intrinsic으로 lateral position 계산
  → EMA smoothing → 3D 궤적 (Z, X)
```

---

## 2. 구현 시 어려웠던 점과 해결 방법

### 2.1 Mamba2의 TensorRT 변환 불가
**문제**: Onepiece 모델의 핵심 모듈인 SpatialMamba(Mamba2 기반)는 recurrent hidden state를 프레임 간 유지한다. TensorRT/ONNX는 이런 가변 상태를 지원하지 않아 전체 모델을 한 번에 TRT로 변환할 수 없다.

**해결 (Hybrid 방식)**:
- DINOv2 backbone + DPT decoder (연산의 ~82%, stateless) → TRT 변환
- SpatialMamba (~15%, stateful) → PyTorch 유지
- 이렇게 하면 재학습 없이 대부분의 연산을 TRT로 가속 가능

**면접 포인트**: "전체를 변환하려다 실패하는 것보다, 연산량 비중을 분석하여 stateless/stateful을 분리하는 판단이 중요했습니다."

### 2.2 해상도 통합 (YOLO vs Depth)
**문제**: YOLO는 내부적으로 640×640, Onepiece는 518×518으로 처리하여 각각 리사이즈 + 결과를 원본 해상도로 다시 매핑해야 했다.

**해결**: 공통 해상도 756×518로 한 번만 리사이즈하여 두 모델에 동시 전달.
- 518 = 37 × 14 (DINOv2 patch_size의 배수)
- 756 = 54 × 14 (patch_size 배수 + 원본 종횡비 유지)
- YOLO는 내부적으로 stride-32 배수로 자동 패딩

**면접 포인트**: "리사이즈 연산 자체는 가벼우나, 업스케일 시 depth 정보 손실이 발생한다. 공통 해상도로 처리하면 이중 리사이즈와 정보 손실 모두 제거된다."

### 2.3 xformers MemEffAttention의 ONNX 미지원
**문제**: DINOv2가 xformers의 `memory_efficient_attention`을 사용하는데, 이 연산자는 ONNX opset에 없다.

**해결**: export 시 `XFORMERS_AVAILABLE = False` 플래그로 표준 `torch.nn.functional.scaled_dot_product_attention`으로 fallback. 코드 수정 없이 런타임 플래그만 변경.

### 2.4 카메라 intrinsic 스케일링
**문제**: 원본 해상도(1600×1100)에서 calibration된 카메라 intrinsic(fx, fy, cx, cy)을 처리 해상도(756×518)에서 그대로 사용하면 lateral position 계산이 틀어진다.

**해결**: 리사이즈 비율에 맞게 intrinsic도 스케일링:
```
fx_new = fx_orig × (target_w / orig_w)
cx_new = cx_orig × (target_w / orig_w)
```

---

## 3. 구현 방식 선택 이유

### 3.1 왜 Onepiece(Mamba2)인가? (vs DAv2, vs Transformer)
| 방식 | 장점 | 단점 |
|------|------|------|
| DAv2 (단일 프레임) | 빠름, TRT 변환 쉬움 | 프레임 간 depth flickering 심함 |
| Transformer (self-attention) | 강력한 temporal 모델링 | O(T²) 메모리, TRT 가능하나 느림 |
| **Mamba2 (SSM)** | **O(T) 메모리, 실시간 스트리밍** | **TRT 변환 어려움 (hidden state)** |

Mamba2를 선택한 이유:
- **선형 시간 복잡도**: 프레임 수에 비례하여 메모리 증가 (Transformer는 제곱)
- **스트리밍 추론**: 프레임 단위로 처리 가능 (전체 시퀀스 필요 없음)
- **Scene Cut 대응**: CLS 토큰 코사인 거리로 장면 전환 자동 감지 → Mamba state reset

### 3.2 왜 FP16인가? (vs INT8)
- **FP16**: 정밀도 손실 거의 없음 (depth 추정에서 정밀도가 중요)
- **INT8**: calibration dataset 필요, depth의 미세한 차이가 궤적 오차로 누적됨
- Orin의 FP16 Tensor Core가 충분히 빠름

### 3.3 왜 CUDA 멀티스트림인가?
- YOLO와 Depth는 같은 프레임을 입력으로 받지만 서로의 결과에 의존하지 않음 → **독립적이므로 병렬 실행 가능**
- 순차 실행 시: latency = YOLO + Depth
- 병렬 실행 시: latency = max(YOLO, Depth) → **이론적 최대 2배 속도 향상**
- 실제로는 GPU SM 공유 때문에 20-40% 향상이 현실적

### 3.4 왜 Hybrid TRT인가? (backbone TRT + Mamba PyTorch)
- DINOv2 backbone이 전체 연산의 ~70%를 차지 → 이것만 TRT화해도 큰 효과
- SpatialMamba는 10% spatial downsample (해상도 1/10)에서 동작 → 연산량 자체가 작음
- 전체를 TRT로 변환하려면 Mamba를 Attention으로 대체해야 하는데, 이는 **모델 재학습이 필요**

---

## 4. YOLO 사이즈별 차이점

### YOLO 11 모델 비교
| 모델 | 파라미터 | mAPbox | mAPmask | 속도 (T4 TRT) | 용도 |
|------|---------|--------|---------|-------------|------|
| YOLO11n-seg | 2.9M | 38.9 | 32.0 | 1.8ms | 극한 경량 (IoT) |
| YOLO11s-seg | 10.1M | 46.6 | 37.8 | 2.9ms | 경량 (모바일) |
| **YOLO11m-seg** | **22.4M** | **50.0** | **41.5** | **5.3ms** | **균형 (Orin)** |
| YOLO11l-seg | 27.6M | 52.4 | 42.9 | 6.4ms | 고정밀 |
| YOLO11x-seg | 62.1M | 54.7 | 43.8 | 11.1ms | 최고 정밀 (서버) |

### 왜 YOLO 11m을 선택했는가?
- Orin에서 **실시간 처리 가능**한 최대 정밀도 모델
- n/s는 보행자 detection에서 재현율(recall)이 낮아 트래킹 끊김 발생
- l/x는 Orin에서 FPS가 너무 낮아 실시간 처리 어려움
- **m이 정밀도-속도 트레이드오프의 sweet spot**

### YOLO 세그멘테이션 vs 바운딩 박스
- 세그멘테이션 마스크를 사용하면 **보행자 영역의 depth만 정확히 추출** 가능
- 바운딩 박스만 사용하면 배경 depth가 섞여 depth 추정 오차 증가
- 마스크 erosion을 추가로 적용하여 가장자리 노이즈 제거

---

## 5. TensorRT 최적화 원리

### 5.1 TensorRT가 빠른 이유
1. **Layer Fusion**: Conv + BN + ReLU → 하나의 커널로 합침 (메모리 접근 감소)
2. **Kernel Auto-Tuning**: GPU 아키텍처별 최적 커널 자동 선택
3. **Precision Calibration**: FP32 → FP16/INT8 변환 (Tensor Core 활용)
4. **Memory Optimization**: 중간 텐서 메모리 재사용, 불필요한 복사 제거
5. **Static Graph Optimization**: 동적 분기 제거, 상수 폴딩

### 5.2 FP16 vs INT8 vs FP32

| 정밀도 | 비트 | 속도 배수 | 정밀도 손실 | 추가 작업 |
|--------|-----|----------|-----------|----------|
| FP32 | 32 | 1× | 없음 | 없음 |
| FP16 | 16 | ~2× | 매우 적음 | 없음 |
| INT8 | 8 | ~4× | 중간 | Calibration dataset 필요 |

Depth estimation에서 INT8을 안 쓰는 이유: 깊이 값의 **절대 정밀도**가 중요하고, scale/shift 변환에서 미세한 양자화 오차가 누적되어 lateral position 오차로 확대됨.

### 5.3 ONNX의 역할
```
PyTorch 모델 → ONNX (중간 표현) → TensorRT 엔진

ONNX: 플랫폼 독립적인 모델 그래프 포맷
  - PC에서 생성 가능 (GPU 아키텍처 무관)
  - 어느 하드웨어에서든 TRT로 변환 가능

TRT Engine: GPU 아키텍처 종속적
  - 반드시 타겟 GPU에서 빌드해야 함
  - PC의 .engine 파일은 Orin에서 동작 안 함
```

---

## 6. Jetson Orin 아키텍처 특성

### 6.1 Orin vs PC GPU 비교
| 항목 | RTX A6000 | Orin AGX (64GB) | Orin NX (16GB) |
|------|-----------|-----------------|----------------|
| CUDA Cores | 10,752 | 2,048 | 1,024 |
| Tensor Cores | 336 | 64 | 32 |
| 메모리 | 48GB GDDR6 | 64GB LPDDR5 | 16GB LPDDR5 |
| TDP | 300W | 60W | 25W |
| 아키텍처 | Ampere (sm_86) | Ampere (sm_87) | Ampere (sm_87) |

### 6.2 Orin 개발 시 주의사항
- **JetPack SDK**: CUDA, cuDNN, TensorRT가 JetPack으로 통합 관리됨. 버전 불일치 주의
- **ARM64 아키텍처**: x86용 pip 패키지가 안 되는 경우 많음. 소스 빌드 필수
- **PyTorch**: 반드시 NVIDIA가 Jetson용으로 빌드한 wheel 사용
- **전력 모드**: `nvpmodel`로 성능/전력 트레이드오프 조절 가능

### 6.3 왜 TRT 엔진을 Orin에서 빌드해야 하는가?
TensorRT 엔진은 빌드 시점에 **해당 GPU의 SM 버전, 사용 가능한 Tensor Core 수, 메모리 대역폭**을 고려하여 최적 커널을 선택한다. PC(sm_86)와 Orin(sm_87)은 SM 버전이 다르므로 선택되는 커널이 완전히 다르다.

---

## 7. 전체 파이프라인 지연시간 분석

### Phase 1 (PyTorch, PC)
```
프레임 처리 1회 = ~160ms (6.3 FPS)
  ├── 프레임 리사이즈:    ~1ms
  ├── YOLO 추론:         ~30ms
  ├── Onepiece 추론:     ~120ms
  │   ├── DINOv2 인코더:  ~85ms  (70%)
  │   ├── DPT 디코더:     ~15ms  (12%)
  │   ├── SpatialMamba:   ~15ms  (12%)
  │   └── FinalHead:      ~5ms   (4%)
  └── 후처리:             ~5ms
```

### Phase 2 (TRT FP16, 멀티스트림, PC 예상)
```
프레임 처리 1회 = ~45ms (22 FPS)
  ├── 프레임 리사이즈 + GPU 전송: ~2ms
  ├── [병렬] YOLO TRT:           ~8ms  ┐
  ├── [병렬] Onepiece Hybrid:    ~40ms ┤ max = 40ms
  │   ├── DINOv2+DPT TRT:       ~25ms │
  │   ├── SpatialMamba PyTorch:  ~10ms │
  │   └── FinalHead TRT:        ~2ms  │
  └── 후처리:                    ~3ms  ┘
```

### Phase 2 (TRT FP16, Orin AGX 예상)
```
프레임 처리 1회 = ~80ms (12 FPS)
  ├── [병렬] YOLO TRT:          ~25ms  ┐
  ├── [병렬] Onepiece Hybrid:   ~70ms  ┤ max = 70ms
  └── 후처리:                   ~5ms   ┘
```

---

## 8. 예상 면접 질문 & 답변

### Q1: "Mamba2를 TRT로 변환할 수 없다면, 왜 Transformer 대신 Mamba를 쓰나요?"
**A**: Transformer의 self-attention은 O(T²) 메모리를 사용하여 긴 시퀀스에서 메모리 폭발이 발생합니다. Mamba2는 O(T) 선형 메모리로 스트리밍 처리가 가능하고, hidden state만 유지하면 되므로 실시간 시스템에 적합합니다. TRT 변환이 안 되는 부분은 전체 연산의 ~15%에 불과하고, backbone(~82%)만 TRT로 가속해도 충분한 성능 향상을 얻었습니다.

### Q2: "INT8 양자화를 왜 사용하지 않았나요?"
**A**: Depth estimation은 연속적인 실수 값을 출력하므로 양자화 오차에 민감합니다. 특히 Onepiece의 scale/shift 변환에서 INT8의 양자화 오차가 누적되면 lateral position 계산에서 수십 cm 단위 오차가 발생할 수 있습니다. FP16은 정밀도 손실이 거의 없으면서 Tensor Core를 활용하여 ~2배 속도 향상을 제공합니다.

### Q3: "CUDA 멀티스트림이 실제로 병렬화되나요?"
**A**: GPU SM(Streaming Multiprocessor) 리소스에 여유가 있어야 실제 병렬화됩니다. PC급 GPU(RTX A6000, 10,752 CUDA cores)에서는 두 모델이 동시에 실행될 가능성이 높지만, Orin(2,048 cores)에서는 큰 모델이 SM을 대부분 점유하여 부분 병렬화될 수 있습니다. nsys 프로파일링으로 실제 커널 실행 타이밍을 확인하여 검증합니다.

### Q4: "Scene cut detection이 왜 필요한가요?"
**A**: Mamba2는 이전 프레임의 hidden state를 기반으로 현재 프레임을 처리합니다. 장면이 급격히 바뀌면(카메라 전환, 급정거 등) 이전 hidden state가 오히려 방해가 됩니다. CLS 토큰 간 코사인 거리로 장면 변화를 감지하고, threshold를 초과하면 Mamba state를 초기화합니다.

### Q5: "세그멘테이션 마스크 대신 바운딩 박스로 depth를 추출하면 안 되나요?"
**A**: 바운딩 박스 영역에는 보행자 뒤의 배경 depth가 포함됩니다. 예를 들어 보행자가 5m에 있고 배경이 30m이면, 박스 내 평균 depth는 실제보다 훨씬 크게 산출됩니다. 세그멘테이션 마스크를 사용하면 보행자 픽셀만 정확히 추출할 수 있고, 추가로 mask erosion을 적용하면 가장자리 노이즈도 제거됩니다.

### Q6: "EMA smoothing의 alpha 값은 어떻게 정했나요?"
**A**: alpha=0.3으로 설정했습니다. alpha가 1에 가까우면 노이즈가 그대로 전달되고, 0에 가까우면 급격한 변화를 놓칩니다. 차량 전방 카메라에서 보행자는 급격히 움직이지 않으므로 0.3 정도의 smoothing이 적절합니다. 실제로는 비디오 FPS와 보행자 이동 속도에 따라 튜닝이 필요합니다.
