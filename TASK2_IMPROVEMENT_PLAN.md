# Task 2 v5 개선안과 실행 방법

구현 확인일: 2026-11-01.

**Kaggle·Colab 통합 노트북에 soft-BCE + hard-IoU, 해소창 보강, 모델 선택·임계값 선택 기간 분리를 반영했습니다.** 새 기본 tag는 `conditional-soft-hard-clear-v5`입니다. 이 설정의 성능 향상은 실제 학습 결과로 판단해야 합니다.

## 1. 제공 자료에서 확인한 병목

사용자 제공 8개 패널 NPZ, 80개 창, 480개 lead와 기존 CSV를 대조했습니다. 아래 수치는 기존 모델의 관측 기반 대리 지표입니다.

| 항목 | 결과 | 반영한 변경 |
|---|---|---|
| 전역 임계값 0.6 → 0.7 | 전체 IoU 0.75048 → 0.75283 | 임계값 조합 확대보다 학습 목표를 개선합니다. |
| onset / ongoing IoU | 0.66159 / 0.84408 | 조건별 학습량을 유지하고 진단은 분리합니다. |
| 정답 1~9셀인 onset 15개 창 | 평균 IoU 0.46032, onset FP의 64.9% | 창 단위 IoU 목표를 고정 hard 라벨로 계산합니다. |
| onset FP 114개 | 정답 없는 링크 68개, 시작 이전 43개 | 새 링크·시작 이전 오탐을 따로 저장합니다. |
| ongoing FP 115개 | 해소 이후 88개 | 확인 가능한 해소창을 보강합니다. |
| ongoing 적격 1→0 변화 46쌍 | 같은 경계에서 정확히 전환한 예측 6쌍 | 전환 검출 수와 해소 이후 오탐을 함께 기록합니다. |

NPZ에는 실제 입력 이력과 fit 표본 분포가 없으므로 원인을 확정하지 않습니다. 새로운 NPZ에는 링크 ID와 이력·prior logit을 저장해 후속 분석의 근거를 보강했습니다.

## 2. 기본 레시피

| 설정 | 기본값 | 동작 |
|---|---|---|
| `QUEUE_LABEL_MODE` | `"soft"` | 임계속도 ±5km/h에서 연속 라벨을 BCE에 사용합니다. |
| `QUEUE_IOU_TARGET` | `"hard"` | IoU는 적격 관측소 중 하나라도 임계속도 이하인 원래 라벨로 계산합니다. |
| `QUEUE_CLEAR_FRACTION` | `0.5` | ongoing 풀에서 확인된 해소창 비중이 낮으면 보강합니다. |
| `QUEUE_SELECTION_SPLIT` | `True` | 1월 전반에서 epoch, 후반에서 임계값을 선택합니다. |
| `QUEUE_VALIDATION_SEED` | `42` | 학습 seed를 바꾸어도 검증 창을 유지합니다. |
| `QUEUE_POLICY_SCOPE` | `"global"` | 전역 임계값 하나만 비교합니다. |
| `VALIDATION_WINDOWS` | `10` | 각 검증 기간의 패널·조건별 최대 창 수입니다. |

```text
y_soft = clip(0.5 + (v_cut - observed_speed) / 10, 0, 1)
y_hard = any eligible station speed <= v_cut
L = BCEWithLogits(logits, y_soft)
    + 0.2 * (1 - soft_IoU(sigmoid(logits), y_hard))
    + 0.1 * Huber(predicted_speed, observed_speed)
```

대상·적격 마스크, 조건별 출력층, 일반 도로 그래프, 미래 시점의 달력·fit 과거 평균 경로를 유지합니다. 실제 미래 관측을 입력에 추가하지 않습니다. Wavelet은 기존 선택 기능으로 남으며 기본으로 켜지 않습니다.

해소창은 같은 대상 링크의 연속 미래 세 시점 모두 적격이고 라벨이 `1,0,0`인 경우입니다. 해소창이 부족하면 ongoing quota의 50%를 목표로 추출하되, 보강한 창 하나는 최대 두 번만 사용합니다. 부족분은 다른 ongoing 창으로 채웁니다. 반복 제한으로 기존 추출의 기대 해소 비중보다 낮아지면 같은 난수 상태로 원래 추출을 유지합니다. 해소창이 이미 충분하거나 한쪽 풀이 비어도 기존 추출을 사용합니다. 실제 비중은 목표보다 낮을 수 있습니다.

패널×condition quota와 optimizer 업데이트 단위는 유지합니다. v5는 그룹별 난수를 분리하므로 해소창 비율만 바꾸면 onset의 표본 구성이 바뀌지 않습니다. 미래 라벨의 해소 여부는 fit 표본 분류에만 사용하며 모델 입력에는 넣지 않습니다.

## 3. 학습·선택·재개

| 용도 | 기간 |
|---|---|
| 가중치 학습 | 2030-06~12 |
| epoch 선택 | 2031-01-01~15 |
| 전역 임계값 선택 | 2031-01-16~31 |
| 선택값을 고정한 추가 비교 | 2031-02 |

검증의 공개 조건과 날짜 분산을 유지합니다. 창이 부족하면 실제 개수를 기록하고 조건을 완화하거나 중복 생성하지 않습니다. 체크포인트별 확률 캐시와 별도로 `processed_v2/queue-windows-<hash>.json`에 공통 날짜·origin·조건·링크·라벨·마스크를 저장하고 모델 간 일치를 확인합니다. epoch 검증 manifest의 식별자는 날짜 분산과 공개 조건을 포함하며, 공개 선정 조건을 끈 경우에는 sampling 버전·관측률·창 간격·일일 창 수도 구분합니다. 같은 설정에서는 창·라벨 불일치 검사를 유지합니다.

기존 모델과 이번 설계는 이미 1월 분석의 영향을 받았고 2월도 개발에 사용한 이력이 있습니다. 특히 전체 1월에서 선택한 기존 체크포인트로 초기화하면 과거 선택의 영향이 없어지지 않습니다. 이 비교는 새로운 독립 holdout 검증이 아닙니다.

Queue만 새로 학습하려면 설정 셀에서 다음을 사용합니다.

```python
TASK2_ONLY = True
RUN_MODE = "train"
RESUME = False
QUEUE_TAG = "conditional-soft-hard-clear-v5"
QUEUE_INIT_CHECKPOINT = None  # 처음부터 학습
```

기존 모델을 초기값으로 쓰려면 `QUEUE_INIT_CHECKPOINT`에 pooled-mlp 또는 conditional Queue의 `queue.pt`를 지정합니다. 기존 feature·encoder 계약이 맞아야 하며 원본과 출력 경로가 같으면 중단합니다. 이 경우 학습률은 `1e-4`로 시작하고 기존 optimizer 상태는 가져오지 않습니다.

`RESUME=True`는 저장된 레시피·optimizer·scheduler를 이어갑니다. 구형 모델에 v5 손실과 표본 분포를 덮어쓰지 않습니다. 새 레시피를 적용하려면 별도 tag에서 초기화 학습을 시작합니다. 구형 모델의 전체 1월 검증·기존 정책 비교와 기존 `selection.json` 읽기도 유지합니다.

같은 초기값과 예산에서 손실·표본 효과를 나누어 비교하려면 tag를 각각 바꿉니다.

| 후보 | `QUEUE_LABEL_MODE` | `QUEUE_IOU_TARGET` | `QUEUE_CLEAR_FRACTION` |
|---|---|---|---:|
| A: 기존 손실 대조군 | `"sample"` | `"train"` | `0.0` |
| C: 목표 분리 | `"soft"` | `"hard"` | `0.0` |
| D: 새 기본값 | `"soft"` | `"hard"` | `0.5` |

세 후보 모두 v5의 같은 검증 기간·seed·임계값 후보군을 사용합니다. 먼저 seed 42에서 비교하고, 통과 후보와 대조군만 seed 43으로 반복합니다. 학습 seed만 바꾸고 검증 seed는 고정합니다.

## 4. 저장 결과와 채택 판단

- `queue.epoch<N>.json`: 패널·조건별 학습량, 후보/해소 후보 수, 추출 횟수, 고유 창·날짜 수, 실제 해소창 노출량입니다.
- `calibration-public-v2/selection.json`: 체크포인트 hash, 검증 설정, 기간, 선택한 정책과 대리 지표입니다.
- `january_policy_comparison.csv`: 새 모델은 전역 임계값 후보만 비교합니다. 구형 모델은 저장된 정책 범위를 따릅니다.
- `january_{saved,selected}_{windows,leads}.csv`: TP/FP/FN, 정체 크기, 새 링크·시작 이전·해소 이후·중간 공백 FP와 정확한 시작/해소 전환 수입니다. 전환은 +5~+30분 안의 서로 인접한 두 적격 시점만 셉니다.
- `probabilities/<period>/<hash>/*.npz`: 확률, 대리 정답·마스크, 날짜·origin·조건·링크 ID와 `queue_logit`, `history_logit`, `prior_logit`입니다.
- `january_prior_sensitivity.csv`: 선택한 임계값을 고정하고 prior 기여도를 0/0.5/1로 바꾼 민감도 비교입니다. 자동으로 모델이나 정책을 변경하지 않습니다.
- `february_fixed.json` 및 CSV: `RUN_FEBRUARY=True`일 때 1월 선택값을 고정한 추가 비교입니다.
- `queue-validation.csv`, `queue-private.csv`: 선택한 정책의 제출용 Queue CSV입니다. `TASK2_ONLY=True`일 때 전체 제출 병합은 하지 않습니다.

전체 대리 IoU와 함께 ongoing 저하 여부, 작은 onset 창의 IoU·FP·FN, 해소 전환 검출과 해소 이후 FP를 비교합니다. FP 감소가 FN 증가만으로 생기지 않았는지 확인합니다. `prior` 약화·새 gate·전환 보조 손실·wavelet은 이 결과를 근거로 별도 비교할 후속 후보입니다.

로컬 확인 범위는 두 노트북의 구문·내장 SHA-256·공통 소스 일치, 호출 관계·마스크·기간·재개 설정의 정적 검토입니다. 회귀 검사는 `tests/test_task2_v5.py`와 기존 통합 검사에 추가했습니다.
