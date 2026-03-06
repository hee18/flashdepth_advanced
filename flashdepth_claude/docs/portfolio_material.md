# FlashDepth / Onepiece — 포트폴리오 & 자소서 소재

## 프로젝트 한 줄 요약

> DINOv2 Vision Transformer + Mamba2 State Space Model 기반의 **실시간 비디오 Metric Depth Estimation** 시스템 설계 및 구현. 5개 학습 데이터셋, 6개 평가 데이터셋, 11개 메트릭으로 검증.

---

## 1. 프로젝트 개요

### 1.1 문제 정의

단안(monocular) 비디오에서 각 픽셀의 **실제 거리(미터 단위)**를 추정하는 문제.

기존 방법의 한계:
- **Single-frame 모델** (Depth Anything V2, DepthPro 등): 프레임 간 depth가 불연속 (flickering)
- **Video 모델** (Video-Depth-Anything, DepthCrafter 등): Temporal consistency는 좋으나, relative depth만 출력하여 실제 미터 단위 변환 불가
- **Metric depth 모델**: Scale/shift를 global token으로만 예측하여 spatial 정보 부족

### 1.2 제안 방법 — Onepiece V3

FlashDepth의 relative depth 모델 위에 **Spatial Mamba + Dual-Stream Architecture**를 추가하여, relative depth와 metric depth를 동시에 생성하는 구조 설계.

**핵심 아이디어**: DPT feature를 1/10로 downsample한 뒤 Mamba2로 temporal processing → 하나의 Mamba 출력이 relative stream (depth quality 향상)과 metric stream (scale/shift 예측) 양쪽에 기여.

### 1.3 아키텍처 반복 설계 과정

| Version | 설계 | 결과 | 교훈 |
|---------|------|------|------|
| **V1** | CLS+GAP → Global Mamba → FiLM + MetricHead | Baseline 대비 개선 | Global token만으로는 spatial 정보 부족 |
| **V2** | GAP+GStdP → 공유 modulated features | V1보다 성능 하락 | Relative/Metric stream 간 optimization conflict 발생 |
| **V3** | DPT → Spatial Mamba → Dual-Stream | *학습 진행 중* | Zero-init으로 pretrained 보존 + two-phase로 안정적 수렴 |

---

## 2. 기술 상세

### 2.1 모델 아키텍처

```
Input Video [B, 8, 3, 518, 518]
    │
    ▼
┌──────────────────────────────────┐
│  DINOv2 ViT-L (Frozen, 304M)    │  ← Pretrained backbone
│  24 layers, embed_dim=1024       │
│  4 intermediate features 추출    │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│  DPT Head (30.6M)               │  ← Dense Prediction Transformer
│  4-level feature pyramid         │
│  Output: [B×8, 256, 37, 37]     │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  SpatialMamba (Trainable)                                │
│                                                          │
│  1. Downsample: [B×8, 256, 37, 37] → [B×8, 256, 4, 4]  │
│  2. Reshape:    [B, 8, 16, 256] (spatial tokens/frame)   │
│  3. 4-layer Mamba2 blocks (d_state=256, hidden state)    │
│  4. final_layer: GELU + Linear (ZERO-INIT)               │
│  5. Upsample + Residual ADD                              │
└──────────────────────────────────────────────────────────┘
    │                              │
    ▼                              ▼
┌─────────────────┐    ┌──────────────────────┐
│ Relative Stream │    │ Metric Stream        │
│                 │    │                      │
│ output_conv     │    │ ConvMetricHead       │
│ → rel_depth     │    │ Conv(256,64,1)→ReLU  │
│   [B×8, H, W]  │    │ →Conv(64,2,1)→GAP   │
│                 │    │ → scale, shift       │
└─────────────────┘    └──────────────────────┘
    │                              │
    ▼                              ▼
┌──────────────────────────────────────┐
│ metric_depth = scale × 100/(rel+ε)  │
│                + shift               │
│ Output: [B, 8, 518, 518] in meters  │
└──────────────────────────────────────┘
```

### 2.2 모델 규모

| Component | ViT-L | ViT-S |
|-----------|------:|------:|
| DINOv2 Encoder (Frozen) | 304.4M | ~22M |
| DPT Head | 30.6M | ~1.5M |
| output_conv | 0.33M | ~0.5M |
| SpatialMamba (4 layers) | ~5.3M | ~5.3M |
| ConvMetricHead | ~33K | ~4K |
| **Total** | **~340M** | **~29M** |

### 2.3 학습 안정성 엔지니어링

#### Zero-Initialization 전략

SpatialMamba의 `final_layer`(GELU + Linear)를 weight=0, bias=0으로 초기화.

**효과**: 학습 시작 시 Mamba 출력 = 0 → post_mamba = 원본 DPT features (identity).
pretrained FlashDepth의 relative depth 품질을 **완전히 보존**한 상태에서 학습 시작.

#### Two-Phase Training

| | Phase 1 (Step 0→1,500) | Phase 2 (Step 1,500→40,001) |
|---|---|---|
| **목표** | Scale/shift 수렴 | 전체 depth quality 향상 |
| **Trainable** | ConvMetricHead only (33K) | SpatialMamba + ConvMetricHead + DPT + output_conv |
| **Frozen** | ViT + DPT + SpatialMamba + output_conv | ViT only |
| **Loss** | LogL1 + TGM | LogL1 + TGM + OFC |
| **LR** | 1e-4 | 1e-4 (Mamba/Head), 1e-5 (DPT, 1/10) |

Phase 2 전환 시 DPT에 500-step warmup (LR 0 → 1e-5) 적용하여 급격한 변화 방지.

### 2.4 Loss 설계 — Multi-Objective Optimization

```
L_total = 1.0 × L_LogL1 + 1.0 × L_TGM + 0.01 × L_OFC
```

| Loss | 역할 | 수식 | 적용 대상 |
|------|------|------|-----------|
| **LogL1** | Per-frame depth 정확도 | `\|log(pred_inv) - log(gt_inv)\|` | Phase 1, 2 |
| **TGM** | Temporal gradient matching | `\|Δpred(t,t-1) - Δgt(t,t-1)\|` | Phase 1, 2 |
| **OFC** | Feature-level temporal consistency | `conf × \|\|feat_t - warp(feat_{t-1})\|\|²` | Phase 2 only |

**OFC (Optical Flow Consistency) 상세**:
1. Sea-RAFT (frozen)로 인접 프레임 간 optical flow + confidence 추정
2. post_mamba features를 1/4 해상도로 downsample (효율성)
3. Flow로 frame t-1의 feature를 frame t 위치로 warp
4. Confidence-weighted L2 loss: occlusion 영역은 자동으로 무시
5. Gradient가 DPT와 SpatialMamba 양쪽으로 역전파 → 두 모듈 동시 최적화

**OFC weight가 0.01인 이유**: Raw L2 스케일이 ~14로 크기 때문에 0.01로 보정.

**Validation loss에서 OFC 제외**: OFC는 학습 수단(auxiliary regularization)이지 평가 목표가 아님. Best model 선정은 LogL1 + TGM 기준.

### 2.5 Streaming Inference & Scene Cut Detection

추론 시 frame-by-frame streaming으로 동작:
- Mamba2의 hidden state를 유지하며 프레임 순차 처리
- DINOv2 CLS token (마지막 레이어)의 cosine distance로 scene 전환 감지
- `D_cls = 1 - cos_sim(CLS_t, CLS_{t-1}) > 0.05` → Mamba state 리셋
- 같은 장면 내에서는 hidden state 축적 → temporal consistency 향상

---

## 3. 인프라 & 엔지니어링

### 3.1 코드베이스 규모

| 항목 | 수치 |
|------|------|
| Python 파일 수 (자체 코드) | **1,036개** |
| 총 코드 라인 수 | **236,430줄** |
| 핵심 모델 코드 (`flashdepth/`) | 47,001줄 |
| 유틸리티 (`utils/`) | 13,883줄 |
| 데이터 로더 (`dataloaders/`) | 6,691줄 |
| 학습 스크립트 (`train_onepiece.py`) | 1,158줄 |
| 평가 스크립트 4종 합계 | 9,145줄 |
| 설정 변형 (config dirs) | 9종 (flashdepth, flashdepth-l/s, gear2-5, onepiece) |

### 3.2 학습 환경

| 항목 | 상세 |
|------|------|
| **Docker Image** | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` |
| **Python** | 3.11 |
| **PyTorch** | 2.4.0 + CUDA 12.4 |
| **핵심 라이브러리** | Mamba2 (local build), flash-attn, xformers v0.0.27 |
| **GPU** | Multi-GPU DDP (torchrun, NCCL backend) |
| **Mixed Precision** | BFloat16 autocast (forward), Float32 (loss) |
| **Gradient Checkpointing** | Enabled (ViT + DPT) |
| **Configuration** | Hydra (YAML-based, CLI override 지원) |
| **Logging** | Weights & Biases (WandB) |

### 3.3 학습 설정

| 항목 | ViT-L | ViT-S |
|------|------:|------:|
| Batch size / GPU | 3 (= 24 frames) | 8 (= 64 frames) |
| 입력 해상도 | 518×518 | 518×518 |
| Video length (학습) | 8 frames | 8 frames |
| Video length (평가) | 50 frames | 50 frames |
| Total iterations | 40,001 steps | 40,001 steps |
| Optimizer | AdamW (β=[0.9, 0.95]) | AdamW (β=[0.9, 0.95]) |
| Weight decay | 1e-6 | 1e-6 |
| Gradient clipping | max_norm=1.0 | max_norm=1.0 |
| Warmup | 500 steps | 500 steps |
| DataLoader workers | 8 | 8 |

### 3.4 데이터셋 파이프라인

**21개 데이터셋 로더** 구현 (synthetic + real indoor/outdoor):

| 용도 | 데이터셋 | 유형 |
|------|----------|------|
| **학습 (5개)** | MVS-Synth, DynamicReplica, TartanAir, PointOdyssey, Spring | Synthetic + Real |
| **검증 (2개)** | Sintel, Waymo Segmentation | Synthetic + Real outdoor |
| **평가 (6개)** | Sintel, Waymo Seg, ETH3D, UrbanSyn, Unreal4K, Bonn | 실내/실외/합성 혼합 |

추가 지원: KITTI, ScanNet, NYU, NuScenes, Virtual KITTI 등.

각 데이터셋별 depth format 대응:
- `.npy` (TartanAir), `.exr` (MVS-Synth), `.dsp5` (Spring), `.png` (PointOdyssey, DynamicReplica)
- Sparse depth 처리 (Waymo LiDAR), segmentation mask 연동

### 3.5 평가 체계

**11개 메트릭** 자동 계산:

| 카테고리 | 메트릭 | 설명 |
|----------|--------|------|
| 정확도 | MAE, RMSE, AbsRel, SqRel | 절대/상대 depth 오차 |
| 임계 정확도 | δ1, δ2, δ3 | max(pred/gt, gt/pred) < 1.25^n 비율 |
| 시간 일관성 | TAE (reprojection) | Camera intrinsic 활용 reprojection 기반 |
| 시간 일관성 | rTC (flow-based) | Optical flow warp + depth ratio consistency |
| 성능 | FPS | 추론 속도 |

**Depth range별 분석**: 0-10m (근거리), 10-30m (중거리), 30-80m (원거리) 세분화.

**출력 자동화**:
- `test_results.json` — 전체 평균 메트릭
- `per_sequence_results.json` — 시퀀스별 상세 메트릭
- `temporal_analysis.json` — 프레임별 TAE, scene cut reset 이력
- 시각화: per-frame depth PNG, error heatmap, best/worst frame 3×3 grid, GIF/MP4

**비교 평가 인프라** (4개 test script):
- `test_onepiece.py` — Onepiece 모델 전용 (Hydra, streaming inference)
- `test_gear5.py` — Gear5 모델 전용 (Hydra, object-wise evaluation)
- `test_comparison.py` — Image 모델 비교 (Metric3D, UniDepth, ZoeDepth, DepthPro, CUT3R 등)
- `test_video_comparison.py` — Video 모델 비교 (Video-Depth-Anything, DepthCrafter)

Shell script (`run_comparison.sh`, `run_video_comparison.sh`)로 conda 환경 자동 전환 + 인자 pass-through.

---

## 4. 자소서 서술 프레임

### 4.1 마스터 자소서 — 연구 경험 단락

```
[문제]
단안 비디오에서 metric depth estimation은 (1) 프레임 간 temporal consistency와
(2) 실제 미터 단위 scale 정확도를 동시에 달성해야 하는 multi-objective 문제입니다.
기존 방법들은 global token 기반 scale 예측으로 spatial 정보가 부족하거나,
relative depth만 출력하여 metric 변환이 불가능했습니다.

[접근]
DINOv2 + DPT feature에 Mamba2 State Space Model을 적용하여 spatial-temporal
정보를 통합하는 Dual-Stream Architecture를 설계했습니다. 하나의 Mamba 출력이
relative depth quality 향상과 metric scale/shift 예측에 동시 기여하는 구조입니다.

[핵심 기여]
1. Spatial Mamba: DPT feature를 1/10 downsample 후 per-frame Mamba2 처리.
   Zero-initialization으로 pretrained 모델 품질을 100% 보존한 채 학습 시작.
2. Two-phase training: Phase 1에서 33K parameter만 학습 (scale/shift 수렴),
   Phase 2에서 SpatialMamba + DPT 전체 fine-tuning (500-step warmup).
3. Optical Flow Consistency Loss: Sea-RAFT의 flow + confidence로 feature-level
   temporal consistency 강제. Occlusion 영역 자동 무시.

[엔지니어링]
236K줄 Python 코드베이스, 21개 데이터셋 로더, 11개 평가 메트릭 자동화,
Docker + DDP multi-GPU 학습, WandB 실험 관리, 6개 데이터셋 × 4개 비교 모델
자동 벤치마크 파이프라인 구축.

[배운 점]
"모델을 만드는 것"보다 "학습이 수렴하게 만드는 것"이 더 어려웠습니다.
Zero-init, phase 전략, loss weight 조정 등 학습 안정성 엔지니어링의 중요성을
체감했고, 3번의 아키텍처 반복(V1→V2→V3)을 통해 가설 → 실험 → 분석의
연구 사이클을 경험했습니다.
```

### 4.2 회사 유형별 변형 포인트

| 회사 유형 | 강조할 키워드 | 구체적으로 부각할 내용 |
|-----------|-------------|---------------------|
| **자율주행** (현대모비스, 네이버랩스, 42dot) | Metric depth, LiDAR sparse GT, 실시간 | Waymo 데이터 처리, streaming inference, 80m 범위 depth |
| **AI 플랫폼** (카카오, 네이버, 라인) | 대규모 학습, MLOps, 평가 자동화 | Docker, DDP, WandB, 21개 데이터셋, 자동 벤치마크 |
| **로보틱스** (삼성리서치, LG, Boston Dynamics) | Temporal consistency, 실시간, scene understanding | Mamba streaming, scene cut detection, rTC 메트릭 |
| **반도체/엣지** (삼성 시스템LSI, 퀄컴) | 모델 경량화, 효율성 | ViT-S variant, Mamba vs Transformer 효율, bf16 |
| **연구소** (ETRI, KIST, KAIST) | 연구 방법론, ablation, novelty | V1→V2→V3 반복, loss 설계 근거, zero-init 전략 |
| **스타트업** (3D/AR/VR 관련) | End-to-end, 빠른 프로토타이핑 | 전체 파이프라인 단독 구현, Docker 배포, 비교 평가 |

---

## 5. 포트폴리오 시각 자료 체크리스트

### 5.1 필수 자료 (마스터 포폴)

| # | 자료 | 내용 | 용도 |
|---|------|------|------|
| 1 | **Architecture Diagram** | V3 전체 파이프라인 block diagram (위 ASCII 참고) | 첫 페이지, 기술 요약 |
| 2 | **Qualitative Comparison** | 3열 비교 (Input / GT / Pred) × 3 데이터셋 (Sintel, Waymo, ETH3D) | 성능 직관적 전달 |
| 3 | **V1→V3 Evolution** | 3개 버전 구조도 + 각각의 한계/해결 화살표 | 연구 사고력 증명 |
| 4 | **Training Curve** | Loss plot with Phase 1→2 전환점 annotation | 학습 안정성 증명 |
| 5 | **Temporal Consistency** | 연속 5-10 프레임 depth strip 또는 t/t+1 difference map | Flickering 없음 증명 |
| 6 | **Metric Table** | 6개 데이터셋 × 주요 메트릭 (AbsRel, δ1, MAE, rTC) | 정량 성능 |

### 5.2 차별화 자료 (선택)

| # | 자료 | 내용 | 용도 |
|---|------|------|------|
| 7 | **Comparison Table** | Ours vs DepthAnything V2 vs Video-Depth-Anything vs DepthPro | 경쟁력 입증 |
| 8 | **FPS 비교** | 모델별 추론 속도 bar chart | 실시간성 어필 |
| 9 | **Ablation Table** | OFC 유/무, Phase 1 길이, Zero-init 유/무 | 설계 근거 정량화 |
| 10 | **Error Heatmap** | Spatial error 분포 (근거리 vs 원거리) | 분석 능력 |
| 11 | **Scale/Shift Plot** | Predicted vs optimal scale trajectory over frames | Metric head 동작 확인 |
| 12 | **Depth Range Bar Chart** | 0-10m / 10-30m / 30-80m AbsRel 비교 | 거리별 성능 분석 |
| 13 | **Scene Cut Visualization** | Reset 전후 depth 변화 | SCD 동작 증명 |
| 14 | **코드베이스 구조도** | Directory tree + 파일 수/줄 수 요약 | 엔지니어링 규모 전달 |

### 5.3 시각 자료 생성 방법

```bash
# 1. 학습 완료 후 test 실행 → qualitative results 자동 생성
./run_docker.sh test_onepiece --config-variant l \
    --gear-checkpoint train_results/results_32/onepiece/large/best.pth \
    --gpu 0 --frame-interval 2

# 2. 비교 모델 실행
./run_comparison.sh depthanythingv2 --dataset sintel --gpu 0
./run_video_comparison.sh vda --dataset sintel --gpu 1

# 3. 결과 디렉토리에서 자동 생성된 시각 자료 수집
# - frames/seq*/frame_*.png        → Qualitative comparison
# - best_frames/*.png              → Best frame 3×3 grid
# - error_heatmaps/seq*/*.png      → Error heatmap
# - test_results.json              → Metric table 소스
# - per_sequence_results.json      → Per-dataset breakdown

# 4. WandB에서 training curve export
# → wandb.ai → project → runs → loss plot download
```

---

## 6. 면접 예상 질문 & 답변 소재

### 6.1 아키텍처 설계

**Q: 왜 Mamba2를 선택했나? Transformer가 아닌 이유는?**
- Mamba2는 sequence length에 대해 선형 복잡도 (O(L)) vs Transformer의 O(L²)
- Video depth에서는 프레임이 길어질수록 (50+) 차이가 커짐
- Hidden state 기반 streaming inference 가능 → frame-by-frame 실시간 처리
- FlashDepth 원본이 이미 Mamba2 사용 → 동일 인프라 재사용

**Q: V2에서 왜 V1보다 성능이 하락했나?**
- V2는 GAP+GStdP를 concat한 shared feature로 FiLM + MetricHead를 동시 학습
- Relative stream(FiLM)과 Metric stream(scale/shift)의 gradient가 동일 feature를 통해 충돌
- V3에서는 SpatialMamba 출력을 분기하되, 각 stream의 입력이 다름 (post_mamba vs mamba_raw)

**Q: Zero-init이 왜 중요한가?**
- Pretrained FlashDepth는 이미 높은 품질의 relative depth 생성
- 새 모듈(SpatialMamba)을 추가하면 random init으로 인해 기존 성능 즉시 파괴
- Zero-init → Mamba output = 0 → post_mamba = original DPT features → 초기 성능 보존
- Phase 1에서 MetricHead만 학습하는 동안 relative depth 품질 유지 보장

### 6.2 Loss 설계

**Q: OFC weight가 0.01로 작은 이유는?**
- OFC의 raw L2 스케일이 ~14로 LogL1(~0.5), TGM(~0.5)보다 훨씬 큼
- 0.01 × 14 ≈ 0.14로 보정하여 다른 loss와 비슷한 스케일로 맞춤
- 너무 크면 feature consistency에 overfitting되어 depth accuracy 하락

**Q: Validation에서 OFC를 제외한 이유는?**
- OFC는 학습 수단(auxiliary regularization)이지 평가 목표가 아님
- Best model은 최종 목표(depth 정확도 + temporal consistency)에 가까운 메트릭으로 선정해야 함
- LogL1 = per-frame accuracy, TGM = temporal consistency → 직접 측정
- OFC = feature space consistency → 간접 수단. 포함 시 model 선정 왜곡 가능

**Q: OFC에서 feature를 1/4로 downsample해도 되는 이유는?**
- OFC의 목적은 pixel-level alignment가 아닌 semantic feature consistency
- DPT + Mamba를 거친 high-level feature는 spatial redundancy가 높음
- Flow 자체도 pixel-perfect하지 않아서, downsample이 noise 평균화 효과
- `adaptive_avg_pool2d`는 미분 가능 → gradient 역전파에 영향 없음

### 6.3 엔지니어링

**Q: 대규모 코드베이스를 어떻게 관리했나?**
- Hydra config: 9개 모델 변형을 YAML override로 관리 (코드 분기 최소화)
- Docker: 재현 가능한 환경 (PyTorch 2.4.0, CUDA 12.4, Mamba2 local build)
- 평가 자동화: 4개 test script가 동일 메트릭/JSON 패턴 공유
- Shell script: conda 환경 자동 전환 + 인자 pass-through로 비교 모델 실행 통일

**Q: Mixed precision에서 주의할 점은?**
- Forward: BFloat16 (메모리 절약 + 속도), Loss: Float32 (수치 안정성)
- BFloat16은 FP16 대비 dynamic range가 넓어 gradient underflow 적음
- Mamba2의 SSM scan은 bf16 호환, 단 loss 계산은 반드시 float32로 upcast

---

## 7. 프로젝트 실수치 요약표

| 항목 | 수치 |
|------|------|
| 코드베이스 | 1,036 Python 파일, 236K줄 |
| 모델 파라미터 (ViT-L) | 340M 총, 304M frozen, ~5.3M SpatialMamba |
| Phase 1 학습 파라미터 | 33K (ConvMetricHead only) |
| 학습 데이터셋 | 5개 (synthetic + real) |
| 평가 데이터셋 | 6개 (실내/실외/합성 혼합) |
| 데이터 로더 | 21개 데이터셋 대응 |
| 평가 메트릭 | 11개 |
| 비교 모델 | 8개+ (DepthAnything V2, DepthPro, Metric3D, UniDepth, ZoeDepth, CUT3R, VDA, DepthCrafter) |
| 학습 해상도 | 518×518, 8 frames/batch |
| 평가 해상도 | 518×518, 50 frames/sequence |
| 학습 iterations | 40,001 steps (Phase 1: 1,500 + Phase 2: 38,501) |
| Loss 구성 | 3개 (LogL1 + TGM + OFC, weight 1:1:0.01) |
| Docker | PyTorch 2.4.0, CUDA 12.4, Python 3.11 |
| 학습 인프라 | Multi-GPU DDP, bf16, gradient checkpointing, WandB |
| Config 변형 | 9종 (flashdepth, flashdepth-l/s, gear2-5, onepiece) |
| Mamba2 설정 | 4 layers, d_state=256, d_conv=4, expand=2 (ViT-L) |
