모든 테스트들(test_gear5, test_comparison, test_video_comparison)의 모든 모델들에서 unreal4k만 
AbsRel이 10단위나 100단위로 매우 큰데 delta_1은 0.3정도로 그정도로 나쁘지 않은 경우가 많아. 왜 
그럴까? 코드나 데이터셋의 문제인지 아니면 예측 아웃라이어 문제인지 검토해줄 수 있을까?

● 문제의 근본 원인 발견! 🎯

  디버그 로그에서 명확하게 확인했습니다:

  핵심 문제

  GT valid range: [0.03, 70.00]  (즉, 3cm ~ 70m)
  Pred valid range: [6.07, 69.87]

  AbsRel stats: mean=2.4767, median=0.6317, max=2679.6826

  Extreme AbsRel pixels (>10): 898 / 612958 (0.1%)
    Their GT range: [0.0258, 0.0355]  <- 2.5~3.5cm!
    Their Pred range: [15.0865, 69.8653]

  GT bin distribution:
    [0, 1m): 898 pixels (0.1%)  <- 문제의 원인!
    [1, 5m): 0 pixels
    [5, 10m): 0 pixels
    [10, 20m): 0 pixels
    [20, 50m): 326695 pixels (53.3%)
    [50, 70m): 285365 pixels (46.6%)

  왜 AbsRel이 10~100이 되는가?

  AbsRel = |pred - gt| / gt 공식에서:

  - GT = 0.03m (3cm), Pred = 15m인 픽셀: |15 - 0.03| / 0.03 = **500**
  - GT = 0.03m (3cm), Pred = 70m인 픽셀: |70 - 0.03| / 0.03 = **2333** (max=2679와 일치!)

  이런 극소수(0.1%)의 극단값이 평균을 2.4767로 끌어올립니다!

  왜 Delta_1은 괜찮은가?

  Delta_1 = max(gt/pred, pred/gt) < 1.25 공식:
  - 대부분의 픽셀(99.9%)은 20~70m 범위에 있고, 예측도 비슷한 범위
  - Median AbsRel = 0.6317 (괜찮음!)
  - 극소수의 극단값이 평균만 왜곡, threshold 기반 메트릭은 영향 적음

  해결 방안

  Option 1: MIN_DEPTH threshold 추가 (권장)
  # test_gear5.py:1128-1131
  MAX_DEPTH = 70.0
  MIN_DEPTH = 1.0  # 1m 미만 필터링

  gt_valid_mask = (gt_frame > MIN_DEPTH) & (gt_frame < MAX_DEPTH)
  pred_valid_mask = (pred_frame > MIN_DEPTH) & (pred_frame < MAX_DEPTH)

  Option 2: Median 기반 평가 추가
  - Mean AbsRel 대신 Median AbsRel을 주요 메트릭으로 사용
  - 극단값에 robust함

  Option 3: 데이터셋별 threshold
  if dataset_name == 'unreal4k':
      MIN_DEPTH = 1.0  # 실외 장면
  else:
      MIN_DEPTH = 0.1  # 기본값

  어떤 해결책을 적용해드릴까요?
