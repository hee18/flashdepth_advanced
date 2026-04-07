# WandB (Weights & Biases) 사용 가이드

FlashDepth GEAR 모델 학습 시 WandB를 사용하여 실시간으로 메트릭을 모니터링하는 방법입니다.

## 📋 목차
- [WandB란?](#wandb란)
- [초기 설정](#초기-설정)
- [학습 시작하기](#학습-시작하기)
- [웹 대시보드 사용법](#웹-대시보드-사용법)
- [로깅되는 메트릭](#로깅되는-메트릭)
- [오프라인 모드](#오프라인-모드)
- [여러 실험 비교](#여러-실험-비교)

---

## WandB란?

**Weights & Biases (wandb)**는 머신러닝 실험 추적 도구입니다.

### 작동 방식
```
[서버 터미널] --인터넷--> [WandB 클라우드] <--인터넷-- [브라우저]
     ↓                         ↓                       ↓
 wandb.log()              데이터 저장            실시간 확인
```

- **터미널**: 학습 중 메트릭을 WandB 클라우드로 전송
- **브라우저**: wandb.ai 웹사이트에서 실시간 그래프 확인
- **클라우드**: 데이터 저장 및 중계 (터미널-브라우저 직접 연결 아님)

### 주요 장점
- ✅ **실시간 모니터링**: 학습 진행 상황을 웹에서 실시간 확인
- ✅ **자동 로깅**: Loss, LR, GPU 사용률 등 자동 기록
- ✅ **실험 비교**: 여러 실험을 한눈에 비교
- ✅ **협업**: 링크 공유로 팀원과 결과 공유
- ✅ **영구 저장**: 실험 기록이 클라우드에 보관됨

---

## 초기 설정

### 1. WandB 설치

```bash
# WandB 설치 (conda 환경에서)
pip install wandb
```

### 2. WandB 로그인 (최초 1회만)

```bash
wandb login
```

실행 시 다음 메시지가 나타납니다:
```
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter:
```

**로그인 절차**:
1. 브라우저에서 https://wandb.ai/authorize 열기
2. **계정 없으면**: GitHub/Google 계정으로 간편 가입
3. **API Key 복사**: 웹페이지에 표시된 key (예: `1a2b3c4d5e6f...`)
4. **터미널에 붙여넣기** → Enter

> **참고**: 한번 로그인하면 `~/.netrc` 파일에 인증 정보가 저장되어 다시 로그인 불필요

---

## 학습 시작하기

### 방법 1: CLI 옵션으로 활성화

```bash
# Gear2 학습 예시
python train_gear2.py \
  --config-path configs/gear2 \
  training.wandb=true \
  training.wandb_name="gear2_baseline_v1" \
  dataset.data_root=/home/cvlab/hsy/Datasets

# Gear4 학습 예시
python train_gear4.py \
  --config-path configs/gear4 \
  training.wandb=true \
  training.wandb_name="gear4_exp_depth_edge" \
  dataset.data_root=/home/cvlab/hsy/Datasets
```

### 방법 2: Config 파일 수정

`configs/gear2/config.yaml` (또는 해당 모델의 config):
```yaml
training:
  wandb: true
  wandb_name: "gear2_baseline_v1"  # 실험 이름 (자유롭게 설정)
  iterations: 150000
  val_freq: 1000
  # ... 기타 설정
```

그 후 평소처럼 실행:
```bash
python train_gear2.py --config-path configs/gear2 dataset.data_root=/home/cvlab/hsy/Datasets
```

### 학습 시작 시 출력 예시

```
wandb: Currently logged in as: your-username
wandb: Tracking run with wandb version 0.16.0
wandb: Run data is saved locally in /path/to/wandb/run-20250115_082030-abc123xyz
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run gear2_phase1_baseline_v1
wandb: ⭐️ View project at https://wandb.ai/your-username/flashdepth-gear2
wandb: 🚀 View run at https://wandb.ai/your-username/flashdepth-gear2/runs/abc123xyz
```

**🚀 View run 링크를 복사해서 브라우저에서 열면 됩니다!**

---

## 웹 대시보드 사용법

브라우저에서 WandB 링크를 열면:

### 왼쪽 사이드바
- **Overview**: 실험 요약, 하이퍼파라미터, 시스템 정보
- **Charts**: 모든 메트릭 그래프 (가장 중요!)
- **System**: GPU/CPU/메모리 사용량 (자동 수집)
- **Logs**: 터미널 출력 (stdout)
- **Files**: 저장된 모델 체크포인트 등
- **Artifacts**: 데이터셋, 모델 버전 관리

### Charts 탭 - 메트릭 그래프

기본적으로 다음 그래프들이 자동 생성됩니다:

#### **Training Metrics**
- `loss`: 전체 loss
- `depth_loss`: Depth estimation loss (Gear2, Gear5에만 있음)
- `edge_loss`: Edge-aware loss (Gear5에만 있음)
- `grad_norm`: Gradient norm (학습 안정성 확인)
- `lr_gear2` / `lr_gear3` / etc.: Gear 모듈 learning rate
- `lr_mamba`: Mamba 모듈 learning rate

#### **Validation Metrics**
- `val/loss`: 전체 validation loss
- `val/sintel_loss`: Sintel dataset 개별 loss
- `val/waymo_seg_loss`: Waymo dataset 개별 loss
- `val/eth3d_loss`: ETH3D dataset 개별 loss
- `val/urbansyn_loss`: UrbanSyn dataset 개별 loss
- ... (config에 설정한 dataset별로 자동 생성)
- `val/{dataset}_sequences`: 각 dataset별 검증 sequence 개수

#### **System Metrics** (자동 수집)
- `system.gpu.0.gpu`: GPU 사용률 (%)
- `system.gpu.0.memory`: GPU 메모리 사용량 (GB)
- `system.cpu`: CPU 사용률 (%)
- `system.memory`: RAM 사용량 (%)
- `system.disk`: 디스크 사용량

### 그래프 조작법

#### 확대/축소
- **마우스 드래그**: X축 범위 선택
- **더블클릭**: 줌 리셋
- **휠**: Y축 범위 조절

#### Y축 스케일 변경
1. 그래프 클릭
2. 우측 상단 **Edit** 버튼
3. **Y Axis** → `Linear` / `Log` 선택
4. Loss가 크게 변할 때는 `Log` 추천

#### 그래프 다운로드
- 그래프 우클릭 → **Download PNG**
- 논문/발표 자료에 활용

#### Smoothing 조절
- 우측 상단 슬라이더로 smoothing 강도 조절
- 0: 원본 데이터
- 0.9: 부드러운 곡선 (트렌드 파악용)

---

## 로깅되는 메트릭

### 모든 모델 공통

| 메트릭 | 설명 | 업데이트 주기 |
|--------|------|---------------|
| `loss` | 전체 training loss | 매 step |
| `grad_norm` | Gradient L2 norm | 매 step |
| `lr_gear*` | Gear 모듈 learning rate | 매 step |
| `lr_mamba` | Mamba 모듈 learning rate | 매 step |
| `val/loss` | Validation loss | val_freq마다 (기본 1000 step) |
| `val/{dataset}_loss` | Dataset별 validation loss | val_freq마다 |
| `val/{dataset}_sequences` | Dataset별 sequence 개수 | val_freq마다 |

### 모델별 추가 메트릭

#### Gear2
```python
'loss': total_loss
'depth_loss': depth_loss
'grad_norm': gradient_norm
```

#### Gear3
```python
'loss': total_loss
'grad_norm': gradient_norm
```

#### Gear4
```python
'loss': total_loss
'grad_norm': gradient_norm
```

#### Gear5-FILM
```python
'loss': total_loss
'depth_loss': depth_loss  # 있는 경우
'edge_loss': edge_loss    # 있는 경우
'grad_norm': gradient_norm
```

---

## 오프라인 모드

인터넷이 안 되는 서버에서 학습할 때:

### 1. 오프라인 모드 활성화

```bash
# 오프라인 모드 설정
wandb offline

# 학습 진행 (로컬에만 저장)
python train_gear2.py --config-path configs/gear2 training.wandb=true
```

오프라인 모드에서는:
- 메트릭이 로컬 디렉토리에만 저장 (`wandb/run-xxx/`)
- 실시간 웹 대시보드 사용 불가
- 나중에 인터넷 연결 시 동기화 가능

### 2. 나중에 동기화

인터넷 연결된 환경에서:

```bash
# 특정 run 동기화
wandb sync wandb/run-20250115_082030-abc123xyz

# 모든 오프라인 run 동기화
wandb sync --sync-all
```

동기화 후 웹 대시보드에서 확인 가능!

---

## 여러 실험 비교

### 1. Workspace에서 여러 Run 선택

1. WandB 프로젝트 페이지 (예: `flashdepth-gear2`)
2. 왼쪽 **Workspace** 탭
3. 비교할 실험들 체크박스 선택
4. **Compare** 버튼 클릭

### 2. 그래프에서 비교

- 모든 메트릭이 하나의 그래프에 오버레이됨
- 각 실험은 다른 색상으로 표시
- 범례에서 특정 실험 on/off 가능

### 3. Parallel Coordinates Plot

1. **Charts** 탭
2. **Add Panel** → **Parallel Coordinates**
3. X축: 하이퍼파라미터 (lr, batch_size 등)
4. 색상: 최종 성능 (val_loss 등)
5. 어떤 설정이 좋은지 한눈에 파악

### 4. Table View

- **Table** 탭에서 모든 실험을 표 형식으로 비교
- 정렬, 필터링 가능
- Best val_loss를 가진 실험 찾기 등

---

## 유용한 팁

### 1. 실험 이름 짓기 규칙

```bash
# 좋은 예
training.wandb_name="gear2_lr1e4_bs8_sintel_waymo"
training.wandb_name="gear4_edge_loss_0.1_v2"
training.wandb_name="gear5_film_baseline_100k"

# 나쁜 예 (나중에 구분이 안됨)
training.wandb_name="test1"
training.wandb_name="experiment"
```

규칙:
- 모델 이름 포함 (gear2, gear4 등)
- 주요 하이퍼파라미터 포함 (lr, batch size 등)
- 실험 목적 포함 (baseline, ablation 등)
- 버전 번호 (v1, v2 등)

### 2. Tags 활용

WandB에서 실험에 태그를 추가하면 나중에 필터링 가능:

```python
# train_gear2.py 수정 예시 (line 183)
wandb.init(
    project="flashdepth-gear2",
    name=f"gear2_phase{self.phase}_{config.training.get('wandb_name', 'experiment')}",
    config=dict(config),
    tags=["baseline", "phase1", "sintel"]  # 태그 추가
)
```

### 3. Alert 설정

WandB 웹에서 alert 설정 가능:
1. Run 페이지에서 **Alerts** 클릭
2. **Add Alert** 선택
3. 조건 설정 (예: `val_loss < 0.5`)
4. 알림 방법 (이메일, Slack 등)

학습이 특정 성능에 도달하면 자동 알림!

### 4. Notes 작성

각 실험에 메모 작성:
1. Run 페이지 상단 **Notes** 탭
2. 실험 목적, 결과 분석 등 자유롭게 작성
3. Markdown 지원

나중에 논문 쓸 때 유용!

### 5. Sweeps (하이퍼파라미터 튜닝)

WandB Sweeps로 자동 하이퍼파라미터 탐색:

```yaml
# sweep.yaml
program: train_gear2.py
method: bayes  # 또는 grid, random
parameters:
  training.learning_rate:
    min: 1e-5
    max: 1e-3
  training.batch_size:
    values: [4, 8, 16]
```

```bash
wandb sweep sweep.yaml
wandb agent your-username/flashdepth-gear2/sweep-id
```

자동으로 최적 하이퍼파라미터 찾기!

---

## 문제 해결

### 1. "wandb: ERROR Not logged in"

```bash
# 다시 로그인
wandb login
```

### 2. "wandb: ERROR API key is invalid"

- API key가 만료되었거나 잘못됨
- https://wandb.ai/authorize 에서 새 key 받기
- `~/.netrc` 파일 삭제 후 재로그인

### 3. 네트워크 오류

```bash
# 오프라인 모드로 전환
wandb offline

# 나중에 동기화
wandb sync wandb/run-xxx
```

### 4. 너무 많은 데이터 로깅으로 느려짐

```python
# 로깅 빈도 줄이기 (매 step이 아닌 매 10 step)
if step % 10 == 0:
    wandb.log(wandb_dict, step=step)
```

### 5. GPU 메트릭이 안 보임

```bash
# nvidia-smi가 작동하는지 확인
nvidia-smi

# WandB 재설치
pip uninstall wandb
pip install wandb
```

---

## 요약

### 기본 사용 흐름

```bash
# 1. 최초 설정 (1회만)
pip install wandb
wandb login

# 2. 학습 시작
python train_gear2.py --config-path configs/gear2 \
  training.wandb=true \
  training.wandb_name="my_experiment" \
  dataset.data_root=/path/to/data

# 3. 브라우저에서 링크 열기
# 터미널에 나온 "View run at https://wandb.ai/..." 링크

# 4. 실시간으로 메트릭 확인
# Charts 탭에서 loss, lr, grad_norm 등 확인
```

### 확인할 주요 메트릭

- **Loss 감소 추세**: `loss`, `val/loss`
- **Learning rate 스케줄**: `lr_gear*`, `lr_mamba`
- **학습 안정성**: `grad_norm` (너무 크면 gradient explosion)
- **Dataset별 성능**: `val/sintel_loss`, `val/waymo_seg_loss` 등
- **시스템 상태**: GPU 사용률, 메모리

### 유용한 링크

- WandB 공식 문서: https://docs.wandb.ai/
- WandB 대시보드: https://wandb.ai/
- API Reference: https://docs.wandb.ai/ref/python

---

**Happy Training! 🚀**
