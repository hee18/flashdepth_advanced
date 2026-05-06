# FSDP2 마이그레이션 플랜 — Onepiece Hybrid 고해상도 학습용

> 목적: 기존 DDP 기반 `train_onepiece.py`를 건드리지 않고, Hybrid 모델 2K 해상도 학습에서
> per-GPU 메모리 한계를 돌파하기 위한 **FSDP2 전용 학습 스크립트**를 별도로 마련한다.
> 작성일: 2026-04-19

---

## 0. 적용 대상

- **주 타겟**: `configs/onepiece/config_hybrid.yaml` (ViT-S student + ViT-L teacher, 2K 해상도, video_length=5)
- **부 타겟**: 동일 스크립트로 `config.yaml` (non-hybrid, 518×518)도 실행 가능하도록 호환 유지
- 실행 환경: `flashdepth:latest` 도커 이미지 (torch 2.4.0, CUDA 12.4, FSDP2 `_composable` 경로 사용 가능 확인 완료)

---

## 1. 현재 상태 스냅샷

| 항목 | 현재 (`train_onepiece.py`) | FSDP2 버전 목표 |
|---|---|---|
| 병렬화 | `DDP(find_unused_parameters=True)` | `fully_shard` (FSDP2, per-param) |
| Mixed Precision | `torch.amp.autocast(bf16)` | FSDP `MixedPrecisionPolicy(bf16/fp32)` |
| Gradient Checkpointing | `apply_activation_checkpointing` on DINOv2 + DPT | 동일 유지 (FSDP wrap 이전 적용) |
| GradScaler | 미사용 (bf16이라 불필요) | 미사용 |
| Optimizer | AdamW, 2-phase 재구성 | 동일 |
| Checkpoint I/O | `model.state_dict()` (DDP unwrap) | `get_model_state_dict(full_state_dict=True, cpu_offload=True)` |

---

## 2. 환경 확인 (완료)

```
# flashdepth:latest 컨테이너 기준
torch: 2.4.0
CUDA: 12.4
FSDP2 (_composable path): AVAILABLE
FSDP1: AVAILABLE
```

- Import 경로: `from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy`
- Torch 2.4 제약 (Mamba2) 충족, stable public path는 2.6부터라 `_composable` prototype을 씀 (기능 완전 동작)

---

## 3. 파일 구조

**신규 생성**

| 경로 | 역할 |
|---|---|
| `train_onepiece_fsdp.py` | FSDP2 버전 학습 스크립트 (원본 복제 후 병렬화/MP/ckpt I/O 교체) |
| `configs/onepiece/config_hybrid_fsdp.yaml` | Hybrid 2K용 FSDP 전용 config |
| `configs/onepiece/config_fsdp.yaml` | 518 non-hybrid용 FSDP config (smoke test용) |

**수정 없음**: `train_onepiece.py`, `flashdepth/model.py`, 기존 config 4종, loss/dataloader 등.

---

## 4. FSDP2 래핑 전략

### 4.1 래핑 단위 (inner → outer 순서로 `fully_shard` 호출)

```
for block in model.pretrained.blocks:       # DINOv2 student transformer blocks
    fully_shard(block, mp_policy=POL)
if hybrid:
    for block in model.teacher_model.pretrained.blocks:
        fully_shard(block, mp_policy=POL)    # teacher도 샤드 (2K라 teacher도 무거움)
for refine in depth_head.scratch.refinenet[1:5]:
    fully_shard(refine, mp_policy=POL)
fully_shard(model.spatial_mamba, mp_policy=POL, reshard_after_forward=False)
fully_shard(model.hybrid_fusion, mp_policy=POL)   # hybrid일 때만
fully_shard(model, mp_policy=POL)            # root
```

**근거**:
- Block 단위 래핑이 communication ↔ compute overlap을 극대화
- `spatial_mamba`는 프레임 루프에서 반복 호출 → `reshard_after_forward=False`로 매 프레임 all-gather 재실행 방지
- Root wrap은 FSDP2에서도 필요 (메타데이터 초기화 + 미래 포괄 래핑용)

### 4.2 Hybrid teacher 처리

- 2K에서는 teacher(ViT-L) 파라미터도 800MB+ → **frozen이지만 샤드 필요**
- `requires_grad=False`여도 FSDP2는 메모리 이득을 줌 (파라미터 샤딩만, gradient 샤드는 0 크기)
- 단, teacher forward는 `torch.no_grad()` 안에서 돌아야 함 (기존 코드가 이미 그렇게 되어있는지 확인 필수)

### 4.3 완전 제외 (`ignored_modules` 상당)

- **SEA-RAFT flow estimator**: OFC loss 전용, 학습 대상 아님 → FSDP 밖에 두고 `torch.no_grad()` 그대로
- GSP head가 쓰일 경우도 동일 (이번 Hybrid에선 미사용)

---

## 5. Mixed Precision 정책

### 5.1 FSDP MP 사용, autocast 제거

```python
POL = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,   # all-reduce/reduce-scatter는 fp32 (수치 안정)
    output_dtype=torch.bfloat16,
    cast_forward_inputs=True,
)
```

- 기존 `with torch.amp.autocast('cuda', dtype=torch.bfloat16):` 블록은 **전부 삭제**
- Loss 블록은 `.float()` 명시 유지 → fp32 계산
- Master weight fp32 유지 → 체크포인트도 fp32로 저장됨 → 추후 Orin fp16 엔진 빌드 시 정밀도 여유

### 5.2 버퍼 dtype 주의

- DINOv2 positional embedding buffer, SpatialMamba 내부 상수 등은 fp32 유지가 안전
- `MixedPrecisionPolicy`의 buffer cast는 **기본적으로 fp32 유지** 정책 사용 (bf16 캐스트 X)

---

## 6. Activation Checkpointing

**현행 유지** + Hybrid 2K 대응으로 범위 확장.

```python
# FSDP2 wrap 이전에 먼저 AC 적용
apply_activation_checkpointing(model.pretrained, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda _: True)
apply_activation_checkpointing(model.depth_head, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda _: True)
if hybrid:
    apply_activation_checkpointing(model.teacher_model.pretrained, ...)  # teacher도 AC
    apply_activation_checkpointing(model.hybrid_fusion, ...)
# 2K에서 SpatialMamba 메모리 부담 크면:
apply_activation_checkpointing(model.spatial_mamba, ...)   # 옵션
```

**순서**: `AC wrap → fully_shard wrap` (역순은 FSDP2가 AC를 인식 못 함)

---

## 7. Phase Transition (1 → 2)

- FSDP2는 per-param 샤드라서 `requires_grad` 런타임 변경에 **강건** (FSDP1의 flat_param 재구성 불필요)
- 기존 `_configure_parameters_phase2()` → `_setup_optimizer()` → `_setup_scheduler()` 플로우 **그대로 재사용 가능**
- 단, optimizer 재생성 시 `model.parameters()`가 FSDP 래핑된 param을 반환하는지 sanity check 필요 (FSDP2는 투명하게 노출됨)
- Hybrid에서 `fusion` param group도 phase 2에서 추가되는 로직 보존

---

## 8. 체크포인트 I/O

### 8.1 저장 (rank 0 full fp32 consolidation)

```python
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_optimizer_state_dict, StateDictOptions
)

opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
model_sd = get_model_state_dict(self.model, options=opts)
optim_sd = get_optimizer_state_dict(self.model, self.optimizer, options=opts)
if self.rank == 0:
    torch.save({
        'global_step': self.global_step,
        'model': model_sd,                 # fp32 consolidated
        'optimizer': optim_sd,
        'scheduler': self.scheduler.state_dict(),
        ...                                # 나머지 메타는 동일
    }, checkpoint_path)
```

- **포맷 호환**: 키 이름/구조 원본과 동일하게 유지 → `train_onepiece.py`로도 재개 가능
- `full_state_dict=True` + `cpu_offload=True` → 대용량(hybrid 2K)에서도 rank 0 OOM 방지

### 8.2 로드 (FSDP ↔ DDP 양방향)

```python
from torch.distributed.checkpoint.state_dict import set_model_state_dict, set_optimizer_state_dict

# load_state_dict(strict=False) 대신 FSDP-aware set_* 사용
set_model_state_dict(self.model, ckpt['model'], options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
set_optimizer_state_dict(self.model, self.optimizer, ckpt['optimizer'], options=...)
```

- `broadcast_from_rank0=True`로 rank 0만 디스크 I/O
- 기존 DDP 체크포인트(`configs/flashdepth/iter_43002.pth`)도 같은 API로 load 가능 (fp32 full state dict 형태)

---

## 9. Gradient Accumulation (2K에서 필요 시)

- DDP의 `no_sync()` 대응: FSDP2는 **`model.set_requires_gradient_sync(False)`**
- 마지막 마이크로배치에서 True로 복구
- 현재 config `gradient_accumulation: 1`이라 당장은 불필요하지만, 2K batch_size=1 강제시 대비해 래퍼 유틸 하나 준비

---

## 10. Hybrid 2K 특이사항

### 10.1 메모리 예상 (ViT-S student + ViT-L teacher, 2K, video_length=5)

- 파라미터: 약 360M (ViT-L 300M + ViT-S 22M + DPT + fusion + mamba) → bf16 shard 시 GPU당 ~ (360M × 2B) / world_size
- Activation이 지배적 (2K × 5 frames × bf16) → FSDP만으론 부족, AC 필수 유지

### 10.2 Teacher forward의 no_grad 경로 확인

- `flashdepth/model.py:245` 등에서 teacher 호출이 `torch.no_grad()` 안인지 점검 예정
- FSDP2는 no_grad 경로에서도 all-gather를 하므로 teacher block들을 샤드하되 `reshard_after_forward=True`로 즉시 해제해 peak 완화

### 10.3 해상도 할당 (main_x / teacher)

- `teacher_resolution: 490` vs student 2K → 두 경로 별도 해상도
- FSDP는 해상도에 무관하지만, 프레임 루프 구조 때문에 **student block 샤드가 더 자주 unshard** 됨 → student에 `reshard_after_forward=False` 고려 (메모리 vs 속도 트레이드오프, 프로파일로 결정)

### 10.4 DistributedSampler × video sequence

- 현행 dataloader가 video 단위 샘플링 → FSDP에서도 동일하게 작동
- 단, `drop_last=True` 확실히 해야 phase transition 시 rank간 step 어긋남 방지

---

## 11. 리스크 체크리스트

| 리스크 | 완화책 |
|---|---|
| Mamba2 커스텀 커널이 FSDP unshard된 bf16 param에서 dtype mismatch | `selective_scan_cuda` 입력을 `.contiguous()`로 보장, 실패 시 해당 블록만 fp32 wrap |
| `_composable` prototype API 변경 | torch 2.4로 핀 고정, 업그레이드 시 재검증 |
| `find_unused_parameters` 부재로 phase 1 미사용 param이 에러 | Phase 1에서도 forward가 모든 trainable param을 경유하도록 zero-init + grad 통로 유지 (기존 구조가 이미 이렇게 설계됨 — `spatial_mamba_downsample=0.1`로 zero-init) |
| Rank 0 checkpoint OOM (hybrid 360M fp32) | `cpu_offload=True` 필수 |
| Activation checkpointing과 FSDP non-reentrant 충돌 | `checkpoint_wrapper`의 `CheckpointImpl.NO_REENTRANT` 사용 확인 |
| Teacher model의 `eval()` 상태 유지 | Phase transition 후에도 teacher는 항상 `.eval()` 강제 (기존 로직 재검증) |
| SEA-RAFT가 FSDP 프로세스 그룹을 건드릴 위험 | flow estimator는 main model과 완전 분리, FSDP 밖에서 로드 |

---

## 12. 구현 단계 (실제 작업 시)

1. **환경 셋업** (5분)
   - `flashdepth:latest` 도커 진입, 멀티 GPU 가시성 확인 (`nvidia-smi`, `NVIDIA_VISIBLE_DEVICES=0,1,2`)

2. **`train_onepiece_fsdp.py` 생성** (1~2시간)
   - 원본 복제
   - `init_distributed()` 그대로
   - `_setup_model()` 에서 AC 적용 → 블록별 `fully_shard` → root `fully_shard`
   - 모든 `autocast` 블록 제거
   - `save_checkpoint` / `load_checkpoint` FSDP API로 교체
   - DDP wrap 제거

3. **`config_hybrid_fsdp.yaml` 작성** (10분)
   - `config_hybrid.yaml` 복제
   - FSDP 전용 하이퍼 추가: `fsdp.cpu_offload`, `fsdp.reshard_after_forward`, `fsdp.buffer_dtype` 등 스위치

4. **Smoke test 1: non-hybrid 518 / 2 GPU** (1시간)
   - 1000 step 학습 → loss 정상 감소 확인, 메모리/throughput 프로파일

5. **Smoke test 2: hybrid 2K / 2 GPU** (2시간)
   - 500 step → OOM 없는지, teacher no_grad 경로 정상 작동
   - AC 범위 조정 (mamba 포함/제외)

6. **체크포인트 왕복 검증** (30분)
   - FSDP 저장 → DDP 로드 → FSDP 로드 세 경로 모두 테스트

7. **Full run** (며칠)
   - Hybrid 2K 40K step 학습

8. **문서화**
   - `changelog.md`에 날짜별 변경 기록 (신규 파일 생성, 코드 없음)
   - `Onepiece.md`에 FSDP 실행법 섹션 추가 (학습 코드 변경은 아니지만 운영 문서이므로)

---

## 13. 실행 방법 (예정)

```bash
# 도커 진입
./run_docker.sh shell

# 2 GPU hybrid 2K FSDP 학습
torchrun --nproc_per_node=2 train_onepiece_fsdp.py \
  --config-path configs/onepiece --config-name config_hybrid_fsdp \
  dataset.data_root=/data/datasets \
  training.batch_size=1 \
  results_dir=train_results/results_fsdp_hybrid_01

# 8 GPU 풀 스케일
torchrun --nproc_per_node=8 train_onepiece_fsdp.py ...
```

---

## 14. 검증 체크리스트 (smoke test 통과 기준)

- [ ] 2 GPU 학습 시작, 첫 스텝 loss 정상 (NaN 아님)
- [ ] 메모리 사용량 DDP 대비 축소 확인 (nvidia-smi 스냅샷)
- [ ] Throughput 비교 (steps/sec, DDP 대비 ±20% 이내면 OK)
- [ ] Phase transition 정상 (step 1500에서 crash 없음, loss 재감소)
- [ ] Validation loss DDP 버전과 동등 (±3%)
- [ ] `last.pth` 저장 → 원본 `train_onepiece.py`로 재개 가능
- [ ] Hybrid 2K에서 teacher forward OOM 없음

---

## 15. 롤백 전략

- 문제 발생 시 `train_onepiece.py` (DDP 원본)로 즉시 복귀 가능 — FSDP 스크립트/config는 신규 파일이라 삭제만 하면 끝
- 체크포인트 포맷 호환 유지가 핵심 (fp32 full state dict)

---

## 16. 향후 확장 (선택, 이 플랜 범위 밖)

- `CPUOffloadPolicy(offload_params=True)` → 극한 메모리 부족 시 (8K 해상도 등)
- HSDP (Hybrid Sharded) — 노드 내 shard, 노드 간 replicate — 멀티 노드 확장 시 고려
- `torch.compile` 조합 — FSDP2 + compile 호환성 torch 2.4에선 불안정, 2.6+ 대기 권장
