# 2026 IEEE Big Data — TrafficFlowBench 대회 안내

**고속도로의 결측 교통 상태를 복원하고, 정체를 예측하며, 교통량의 물리적 일관성과 통행 수요까지 함께 평가하는 대회입니다.** 속도 예측 하나로 순위를 정하지 않고, 네 과제의 점수를 합산합니다.

> 확인일: **2026-09-29, 한국시간(KST)**. Kaggle 현행 화면과 주최 측 공개 저장소를 조사한 참가 안내입니다. 변경 가능한 일정·규정은 제출 전에 다시 확인해 주세요. 대회 데이터는 2026-10-01에 내려받아 Git에서 제외된 `data/`에 저장했으며, 모델 초기 구현은 11절에 안내합니다.

| 항목 | 내용 |
| --- | --- |
| 공식 명칭 | 2026 IEEE Big Data-Traffic Flow Bench |
| 소속 행사 | 2026 IEEE Big Data Cup |
| 참가 플랫폼 | Kaggle Community Prediction Competition |
| 평가 구조 | State 35% + Queue 30% + Physics 15% + ODME 20% |
| 순위 결정 | 단일 리더보드, 최종 순위는 private 평가 기준 |
| 데이터 | 5개 도로군 × 양방향 = 10개 패널, 5분 간격, 총 11개월 |
| 팀 규모 | 최대 4명 |
| 상금 | 총 USD 3,500 |
| Kaggle 포인트·메달 | 지급 대상 아님 |

공식 확인 경로: [대회 Overview](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview), [Rules](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/rules), [주최 측 GitHub](https://github.com/jacky850/trafficflowbench-public).

## 1. 일정·등록·상금

| 일정 | 공식 안내 |
| --- | --- |
| 데이터 공개일 | 2026-06-01 — Overview 본문 기재 |
| 평가 데이터 재생성 | 2026-09-11 — validation·private 재배포 |
| 이메일 참가 등록 | **2026-10-25까지** |
| 최종 제출 마감 | 안내 본문: **2026-11-06** |
| Kaggle 화면의 종료 시각 | 확인 당시 **2026-11-07 15:55 KST / 06:55 UTC** |
| 학회·결과 워크숍 | 2026-12-14~17, 미국 애리조나주 Phoenix |

**마감 안내 본문과 Kaggle 종료 타이머의 날짜가 다릅니다.** 위 표는 둘을 구분해 기록한 것이며, 늦은 시각까지 제출할 수 있다고 단정하지 않습니다. 실제 제출 전 [Timeline and Registration](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview/timeline-and-registration) 및 운영진 공지를 확인해 주세요.

이메일 등록은 `trafficflowbench@gmail.com`으로 제목을 **`Traffic Flow Bench`**로 하여 팀명·소속·모든 팀원의 이름과 이메일을 보내는 방식입니다. 학회·워크숍 참가팀을 기록하는 절차이며, 데이터 접근을 위한 사전 등록은 아닙니다. Kaggle 가입·대회 규칙 동의는 별도로 진행합니다. [등록 안내](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview/timeline-and-registration)

| 수상 구분 | 상금 |
| --- | ---: |
| Gold | USD 1,500 |
| Silver | USD 1,000 |
| Bronze | USD 500 |
| Student / Newcomer | USD 500 |

학생상은 **2026-11-06 기준 팀원 전원이 고등학생 또는 학부 재학생**이어야 합니다. 대학원생은 제외되며, Kaggle 계정 나이·참가 횟수는 기준이 아닙니다. Gold·Silver·Bronze를 받지 않은 적격 팀 중 private 순위가 가장 높은 팀이 대상입니다. 증빙·재현성 심사를 거칩니다. [상금](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview/prizes), [학생상 규정](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/rules)

인증서·등록 지원·발표 기회는 추가 제공 가능 사항으로 안내되어 있으며, 모든 참가자에게 확정된 혜택은 아닙니다. [공식 안내](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview/prizes)

## 2. 무엇을 풀어야 하나요?

여기서 **링크(link)**는 도로의 한 구간, **패널(panel)**은 한 도로군의 한 방향, **램프(ramp)**는 고속도로 진입·진출 연결로를 뜻합니다. **OD(Origin–Destination)**는 출발지–도착지 쌍이고, **ODME**는 관측 교통량으로 이 통행 수요를 추정하는 문제입니다.

| 과제 | 질문 | 제출값 | 비중 |
| --- | --- | --- | ---: |
| Task 1 · State | 센서 관측이 가려진 구간·시각의 속도와 교통량은 얼마인가요? | `speed_kmh`, `flow_vph` | 35% |
| Task 2 · Queue | 예측 시점 이후 30분 동안 어느 구간이 정체되나요? | `queue_pred` = 0 또는 1 | 30% |
| Task 3 · Physics | 복원 결과가 교통류 관계와 차량 수 보존을 만족하나요? | **별도 제출 없음**, Task 1 결과 재사용 | 15% |
| Task 4 · ODME | 도로에서 센 차량들이 어떤 경로로 이동했나요? | 경로별 `path_flow` | 20% |

Task 1·3·4는 공개된 평가 기간 전체를 보는 **오프라인 추정**입니다. Task 2는 시점 `T`까지의 정보로 미래를 맞히는 **온라인 예측**입니다. 따라서 전체 기간의 미래 관측을 사용한 Task 1 복원 결과를 Task 2에 그대로 넣으면 시간 제약을 위반할 수 있습니다. [과제 설명](https://github.com/jacky850/trafficflowbench-public#why-these-four-tasks), [시간 사용 규정](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/rules)

## 3. 데이터 구성

### 데이터의 성격과 분할

실제 검지기에서 추정한 교통 특성으로 보정한 시뮬레이션의 **합성 데이터**입니다. 실제 도로의 관측 행을 복사한 자료가 아니며, 아래 2030~2031년 날짜도 합성 달력입니다. 현실의 같은 날짜 자료와 연결해 정답을 찾을 수 없습니다. [데이터 출처 설명](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/DATA.md#provenance)

| 분할 | 합성 날짜 | 길이 | 공개 범위·용도 |
| --- | --- | ---: | --- |
| `train` | 2030-06-01~2031-02-28 | 273일, 9개월 | Task 1용 마스킹 전 관측 공개, 학습·로컬 검증 |
| `validation` | 2031-03 | 31일 | 마스킹된 입력 공개, public 리더보드 |
| `private` | 2031-04 | 30일 | 마스킹된 입력 공개, 최종 순위 |

`train`에도 **모든 과제의 정답이 공개되는 것은 아닙니다.** 특히 Task 2 정체 정답은 train에도 없습니다. 두 평가 월은 독립적으로 생성되었고 돌발상황도 달라, public 점수와 private 점수가 같다고 기대하면 안 됩니다. [분할·로컬 평가 범위](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/DATA.md)

### 도로군과 주요 변수

| 도로군 | 방향별 패널 |
| --- | --- |
| `D7_I10` | `D7_I10_E`, `D7_I10_W` |
| `D7_I210` | `D7_I210_E`, `D7_I210_W` |
| `D7_I405` | `D7_I405_N`, `D7_I405_S` |
| `D12_I5` | `D12_I5_N`, `D12_I5_S` |
| `D12_I405` | `D12_I405_N`, `D12_I405_S` |

5개 도로군은 5개의 서로 다른 고속도로 번호를 의미하지 않습니다. I-405가 두 district에 나뉘어 포함됩니다. [패널 설정](https://github.com/jacky850/trafficflowbench-public/blob/main/config/corridors.json)

- 본선 관측: 속도(km/h), 교통량(veh/h), 점유율, 품질 정보.
- 램프 관측: 진입·진출 교통량과 품질 정보.
- 도로망: 길이·차로 수·자유류 속도·용량·상하류 연결 관계.
- 수요 추정: 경로 목록, 경로–링크 행렬, 링크 교통량, 약한 경로별 사전 추정치(prior).

`pct_observed >= 75`이며 필요한 원래 값이 있는 셀이 평가 적격입니다. 품질이 낮거나 없는 램프 관측은 **교통량 0으로 관측된 값과 구분**해야 합니다. [품질 기준](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/DATA.md#the-quality-rule)

### 배포 구조

Kaggle 화면 기준 **9,688개 파일·12.39 GB**, 주최 측 README는 압축 해제 규모를 약 13 GB로 설명합니다. 아래 구조의 `kaggle_public/` 자체가 스크립트의 `--release-root`입니다. [Data](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data), [실행 안내](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/RUN_LOCAL.md)

```text
kaggle_public/
├── config/
├── corridors/<PANEL>/
│   ├── network/                  # 링크·차로·물리 파라미터·경로·위상
│   ├── train/
│   │   ├── mainline_states/      # 마스킹 전 관측: train에만 존재
│   │   ├── mainline_states_masked/
│   │   └── ramp_states/
│   ├── validation/              # 마스킹된 본선 관측 + 램프 관측
│   └── private/
├── task1/<PANEL>/<split>/        # 상태 복원 템플릿
├── task2/<PANEL>/<split>/        # 예측 창·과거 관측·정체 템플릿
├── task4/<PANEL>/<split>/        # 링크 교통량·prior·경로 교통량 템플릿
├── sample_submission.csv        # 최종 업로드 형식
└── submission_key.csv           # 행 ID와 실제 예측 대상의 대응표
```

상세 파일 목록은 [공식 데이터 문서](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/DATA.md#layout)를 참조해 주세요.

## 4. 평가 방식

```text
S_total = 0.35 × S_state
        + 0.30 × S_queue
        + 0.15 × S_physics
        + 0.20 × S_ODME
```

각 점수는 0~1이며 높을수록 좋습니다. 방향별 평가 후 같은 도로군의 두 방향을 평균하고, 도로군끼리 다시 동일 가중 평균합니다. Task 2는 `D12_I405_N/S`를 제외한 대상 패널에서 평가합니다. [점수 명세](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SCORING_SPEC.md)

### Task 1: 가려진 속도·교통량 복원

마스킹은 R1=20%, R2=30%, R3=50%입니다. **하루는 정확히 하나의 regime에만 속하며**, 세 regime 점수를 같은 비중으로 평균합니다. 모든 결측값이 예측 대상은 아닙니다. Task 2 미래 구간을 숨기기 위한 결측도 있으므로 반드시 `sample_submission_state.csv`의 행을 사용합니다. [마스킹 명세](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/MASK_SPEC.md)

```text
S_speed(r) = max(0, 1 - RMSE_speed(r) / 25)
S_flow(r)  = max(0, 1 - RMSE_flow_per_lane(r) / 600)
S_state(r) = 0.54 × S_speed(r) + 0.46 × S_flow(r)
S_state    = mean(S_state(R1), S_state(R2), S_state(R3))
```

제출하는 `flow_vph`는 총 교통량이며, 평가기가 차로 수로 나누어 오차를 계산합니다. 밀도는 제출 대상이 아닙니다. [Task 1 점수식](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SCORING_SPEC.md#task-1-state-reconstruction)

### Task 2: 향후 정체 여부

제공되는 기본 창은 과거 60분이고 예측 대상은 `T+5, +10, +15, +20, +25, +30분`입니다. 정체 판정 기준은 `속도 <= 자유류 속도의 60%`이며, 정답은 잡음이 섞인 관측이 아닌 내부 상태로 만듭니다.

```text
IoU = 예측과 정답이 모두 정체인 링크-시각 수
      / 예측 또는 정답이 정체인 링크-시각 수
```

정체가 새로 생기는 `queue_onset`과 이미 진행 중인 `queue_ongoing`을 나누어 평가합니다. 각 대상 패널·split마다 각각 5개 창이 있고, 모든 창의 미래에 실제 정체가 있습니다. Task 2는 `D12_I405_N/S`를 제외한 **8개 패널**이 대상입니다. 2026-10-25 확인한 공식 명세는 기본 history 외에 공개된 `T` 시점의 관측 사용도 허용합니다. [Task 2 명세](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SCORING_SPEC.md#task-2-queue-forecasting)

### Task 3: 물리적 일관성

```text
S_physics = (1/3) × S_FD + (2/3) × S_LWR
```

`S_FD`는 속도·밀도·교통량 사이의 기본 관계(Fundamental Diagram), `S_LWR`는 구간 내 차량 수 변화와 유입·유출의 일치 여부를 봅니다. 밀도 `k=q/v`, 차량 수 `N=k×구간 길이`를 Task 1 예측에서 도출합니다.

```text
다음 시점 차량 수 - 현재 차량 수
= 시간 간격 × (본선 유입 + 램프 유입 - 본선 유출 - 램프 유출)
```

현재 10개 패널은 모두 램프 관측을 사용하는 Mode A로 고정되어 있습니다. 램프 결측 처리와 보존식 계산 구간은 공식 명세를 따라야 합니다. [물리 점수](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SCORING_SPEC.md#task-3-physical-consistency), [Coverage modes](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/TASK3_LWR_COVERAGE_MODES.md)

### Task 4: 경로별 통행량 추정

```text
S_ODME = 0.45 × S_od + 0.25 × S_link + 0.15 × S_dev + 0.15 × S_attr
```

| 항목 | 평가 내용 |
| --- | --- |
| `S_od` | 제출 경로 교통량과 숨겨진 기준값의 차이 |
| `S_link` | 경로 교통량을 링크에 합산한 `A × f`와 공개 링크 교통량의 차이 |
| `S_dev` | 공개 prior로부터 이동한 정도가 기준값과 얼마나 비슷한지 |
| `S_attr` | 도착지별 통행량 분포가 얼마나 비슷한지 |

공개 prior는 **해당 split의** `task4/<PANEL>/<split>/synthetic_weak_prior.csv`를 사용합니다. 링크 교통량만 잘 맞춘다고 전체 ODME 점수가 보장되지는 않습니다. [Task 4 명세](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SCORING_SPEC.md#task-4-od-and-path-flow-estimation), [공식 구현](https://github.com/jacky850/trafficflowbench-public/blob/main/src/task4/build_task4_odme_artifacts.py)

## 5. 제출 파일과 실수하기 쉬운 부분

Kaggle에는 **CSV 한 개**를 업로드합니다. 전체 템플릿은 **validation과 private 양쪽 예측**을 요구합니다.

```csv
submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow
```

| `task` | 채울 열 | 나머지 예측 열 |
| --- | --- | --- |
| `state` | `speed_kmh`, `flow_vph` | 0 |
| `queue` | `queue_pred`: 0 또는 1 | 0 |
| `odme` | `path_flow`: 유한한 0 이상 수 | 0 |

`submission_key.csv`로 행의 split·패널·시각·링크 등을 찾고, 결과는 `sample_submission.csv` 형식으로 돌려놓습니다. 대응표 자체는 업로드하지 않습니다. [제출 안내](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview/submission)

- `submission_id`를 바꾸거나 중복·알 수 없는 ID를 넣지 않습니다.
- 사용하지 않는 열도 빈칸 대신 0을 유지합니다.
- **Kaggle 현행 기준으로 필수 행 하나라도 누락된 과제는 해당 과제 전체가 0점**입니다.
- Task 3용 행·밀도·별도 물리 제출 파일은 만들지 않습니다.
- `departure_time`은 일반 날짜가 아닌 **수요 기간 식별자**이므로 템플릿 값을 유지합니다.
- 각종 ID와 UTC 시각 표기는 제공 형식을 그대로 보존합니다.

일부 GitHub 문서는 누락 행을 0 예측으로 처리한다고 설명하지만, 실제 참가에는 더 엄격한 [Kaggle Evaluation](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview/evaluation) 기준을 적용해야 합니다. 로컬 병합 도구도 매칭 실패를 0으로 채우므로, 파일이 생성됐다는 사실만으로 예측이 완성된 것은 아닙니다. [스키마](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SUBMISSION_SCHEMAS.md), [병합 구현](https://github.com/jacky850/trafficflowbench-public/blob/main/src/merge_submissions.py)

## 6. 베이스라인과 로컬 검증의 한계

| 과제 | 주최 측 베이스라인 | 공개 자료만으로 검증 가능한 범위 |
| --- | --- | --- |
| Task 1 | 링크별 요일·시간대 과거 평균 | `train` 정답과 비교 가능 |
| Task 2 | `T`의 정체 상태를 미래 6개 시점에 반복 | 정체 정답 비공개로 공식 점수 재현 불가 |
| Task 3 | Task 1 결과를 그대로 평가 | 운영진 경계 유량 비공개로 공식 점수 재현 불가 |
| Task 4 | prior 정규화가 있는 비음수 최소제곱 | 공개 링크 교통량에 대한 **`S_link`만** 계산 가능 |

Task 4 베이스라인은 `f >= 0` 제약에서 `||Af-c||² + λ||f-b||²`를 최소화하며, 공개 구현의 `λ`는 0.05입니다. [베이스라인 설명](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/BASELINES.md)

주최 측 `BASELINES.md`에 기재된 validation 참고값은 다음과 같습니다. **이 프로젝트에서 실행·재현한 점수나 현재 리더보드 기록이 아닙니다.** 데이터 재배포와 문서 버전에 따라 과거 수치와 달라질 수 있습니다.

| 지표 | 공개 문서의 베이스라인 점수 |
| --- | ---: |
| `S_state` | 0.6929 |
| `S_queue` | 0.3017 |
| `S_physics` | 0.3549 |
| `S_ODME` | 0.8359 |
| `S_total` | **0.5534** |

출처: [확인한 커밋의 베이스라인 표](https://github.com/jacky850/trafficflowbench-public/blob/c88cddf533bbf0afa4ff1fc6c031d760a08e7a31/docs/BASELINES.md#what-the-reference-baselines-score).

`score_task3.py`가 수치를 출력해도 공개 데이터에서 얻는 대체 계산값을 공식 물리 점수로 해석하면 안 됩니다. Task 4도 로컬에서 생성한 reference를 정답으로 삼아 1점을 얻는 방식은 실제 순위를 검증하지 못합니다. [로컬 실행의 한계](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/RUN_LOCAL.md), [현재 Task 4 평가 코드](https://github.com/jacky850/trafficflowbench-public/blob/main/src/task4/score_task4.py)

## 7. 시작하는 방법

1. Kaggle에서 참가 규칙에 동의하고 최신 데이터 패키지를 받습니다.
2. 아래처럼 공식 코드를 설치하고 작은 패널 하나로 Task 1 입출력을 확인합니다.
3. train 내부에 별도 검증 기간을 정해 개선 효과를 비교합니다.
4. 전체 제출용으로는 validation·private 양쪽의 모든 대상에 대해 추론합니다.
5. CSV 형식·누락·병합 경고를 확인하고 Kaggle에서 제출합니다.

아래 공식 저장소 예제는 **실행 방법 안내**이며, 이 명령과 대회 제출은 실행하지 않았습니다. 11절의 소규모 모델 학습은 별도로 실행했습니다. 공식 의존성은 pandas·NumPy·PyArrow·SciPy 등입니다. [실행 가이드](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/RUN_LOCAL.md), [requirements.txt](https://github.com/jacky850/trafficflowbench-public/blob/main/requirements.txt)

```bash
git clone https://github.com/jacky850/trafficflowbench-public.git
cd trafficflowbench-public
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 내려받은 데이터의 실제 절대 경로로 바꿔 주세요.
REL="/absolute/path/to/kaggle_public"

python src/task1/build_task1_baseline_submission.py \
  --release-root "$REL" --split train --panel D12_I5_N \
  --output state_train_D12_I5_N.csv

python src/task1/score_task1.py \
  --submission state_train_D12_I5_N.csv \
  --release-root "$REL" --split train --panel D12_I5_N
```

위 결과는 **train 한 패널의 검증용**이며 업로드용 전체 예측이 아닙니다. 제출 단계에서는 validation·private 결과를 모두 담은 과제별 파일을 준비한 다음 병합합니다.

```bash
# 전제: 아래 세 입력 CSV에 validation·private의 전체 예측이 들어 있습니다.
python src/merge_submissions.py \
  --state state_submission.csv \
  --queue queue_submission.csv \
  --odme path_flow_submission.csv \
  --key "$REL/submission_key.csv" \
  --output submission.csv
```

`--key`는 현재 공개 구현의 **필수 인자**입니다. 각 과제별 파일 형식은 [SUBMISSION_SCHEMAS.md](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SUBMISSION_SCHEMAS.md)를 따릅니다.

## 8. 참가 규칙 핵심

- **Task 2 시간 제한:** 각 창의 `T` 이하 공개 자료만 사용합니다. 앞선 날짜·train 전체는 허용하지만, 미래 램프 관측이나 마스킹 버퍼 이후 관측도 금지입니다.
- **외부 자료:** 공개 데이터·사전학습 모델·API 등은 적법한 라이선스와 다른 참가자의 무료·저비용 접근 가능성이 전제입니다.
- **계정·협업:** 1인 1계정·1팀이며, 팀 밖 비공개 코드·데이터 공유는 금지입니다.
- **수상 검증:** 코드, 버전, 전처리, seed, 외부 자료·라이선스, 재현 명령 및 실행 환경을 준비해야 합니다.
- **재배포:** 9월 11일 재구축 후 제출 재개 시점 이후 제출만 수상 대상입니다. 이전 데이터 사용자는 재다운로드·재제출해야 하며, 이전 점수와 새 점수는 직접 비교하지 않습니다.

이 항목의 근거는 [현행 Competition Rules](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/rules)입니다. 일일 제출 횟수·최종 선택 가능 제출 수·팀 병합 마감은 이번 조사에서 확정하지 못했으므로 참가 계정의 대회 설정에서 확인해야 합니다.

## 9. 자료 간 차이와 해석 기준

| 항목 | 발견한 차이 | 이 문서의 기준 |
| --- | --- | --- |
| 대회 구성 | IEEE 소개는 Open/Expert·3개 트랙·Codabench/EvalAI를 서술 | Kaggle의 4개 과제·단일 리더보드 |
| 데이터 | IEEE 소개에 실제 연도·다른 규모의 기획 설명이 남아 있음 | 현재 Kaggle 합성 데이터 패키지 |
| 마감 | 본문 11/6과 화면 타이머 11/7 KST가 다름 | 둘 다 기록, 제출 전 재확인 |
| 누락 행 | 일부 GitHub 로컬 설명은 0 예측 처리 | Kaggle의 과제 전체 0점 기준 |
| 제출 병합 명령 | Kaggle의 짧은 예시에 `--key`가 빠져 있음 | 실제 공개 코드의 필수 인자 포함 |

IEEE 페이지는 대회 제안서를 요약한 소개이며, 최종 확인은 각 대회 안내를 따르도록 명시합니다. 규정·일정은 [Kaggle](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview), 구현은 [공개 저장소](https://github.com/jacky850/trafficflowbench-public)를 대조했습니다. [IEEE 소개 원문](https://bigdataieee.org/BigData2026/cup/)

## 10. 운영진·문의·출처

현재 Kaggle 소개에는 IEEE ITS Society의 Travel Information and Traffic Management 기술위원회와 RERITE가 주최 측으로, Xuesong (Simon) Zhou·Ruolin Li가 공동 의장으로, Cathy Wu·Yudai Honma가 조직위원으로 기재되어 있습니다. [Organizers and Judging](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview/organizers-and-judging)

데이터·평가 질문은 [Kaggle Discussion](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion), 코드·문서 결함은 [GitHub Issues](https://github.com/jacky850/trafficflowbench-public/issues)를 이용합니다. [공식 문의 안내](https://github.com/jacky850/trafficflowbench-public#rules-licence-contact)

| 자료 | 용도 |
| --- | --- |
| [Kaggle Overview](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview) | 개요·일정·평가·상금 |
| [Kaggle Rules](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/rules) | 참가·시간 제한·재현성·수상 규정 |
| [Kaggle Data](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data) | 데이터와 제출 템플릿 |
| [공식 GitHub](https://github.com/jacky850/trafficflowbench-public) | 베이스라인·평가 코드·명세 |
| [IEEE Big Data Cup 2026](https://bigdataieee.org/BigData2026/cup/) | 전체 대회 행사 안내 |
| [CITATION.cff](https://github.com/jacky850/trafficflowbench-public/blob/main/CITATION.cff) | 연구 결과에서 벤치마크를 인용할 때 사용할 정보 |

코드 라이선스는 [MIT](https://github.com/jacky850/trafficflowbench-public/blob/main/LICENSE)이며, 데이터는 Kaggle의 **Subject to Competition Rules** 조건을 따릅니다. 두 조건을 동일한 것으로 간주하지 않습니다.

조사 시 확인한 GitHub 커밋: [`c88cddf533bbf0afa4ff1fc6c031d760a08e7a31`](https://github.com/jacky850/trafficflowbench-public/commit/c88cddf533bbf0afa4ff1fc6c031d760a08e7a31). 문서 검증 범위는 공식 페이지·공개 코드 대조와 README 형식 검사입니다. 데이터 다운로드와 샘플 스키마 확인, 소규모 학습 실행은 2026-10-01에 수행했습니다. 이후 체크포인트 재평가와 GPU 측정 결과는 [2026-10-03 구현 점검](MODEL_AUDIT.md)에 정리했습니다. Kaggle 점수 재현은 수행하지 않았습니다.

## 11. 모델 학습과 검증 사용법

2026-10-04부터 기본 모델은 **공식 Mamba-2를 사용하는 v2**입니다. 31개 공개 입력 특징, 학습 가능한 노드·시간 임베딩, FFN·dropout·skip 출력층을 추가했습니다. `--graph`와 Physics 보조 손실은 비교 실험 옵션입니다. FAST의 시간 Attention–공간 Mamba–시간 Attention 순서를 유지한 대회용 적응 모델이며, 원논문 Mamba-1의 성능 재현을 뜻하지 않습니다. [구현 점검과 남은 실험](MODEL_AUDIT.md#2026-10-04-구현-반영)

### 데이터와 WSL 환경

Kaggle 규칙 동의와 API 인증 후 기존 다운로드 명령을 사용합니다. 캐시는 Git에서 제외한 `data/kagglehub_cache/`에 저장하고, 실제 경로는 `data/release_root.txt`에 기록합니다. [kagglehub 인증](https://github.com/Kaggle/kagglehub#authenticate)

```powershell
.venv\Scripts\python.exe src\download_data.py
Get-Content data\release_root.txt
```

WSL 프로젝트는 `/home/parkj/projects/trafficflowbench`, 데이터는 그 아래 `data/kaggle_public`에 있습니다. Windows 폴더와 별도 복사본이므로 소스 변경은 자동 동기화되지 않습니다. 현재 WSL 환경에는 공식 Mamba-2와 CUDA PyTorch를 설치했습니다. 환경 재생성은 WSL에서 다음 순서로 진행합니다.

```bash
cd /home/parkj/projects/trafficflowbench
sudo apt-get update
sudo apt-get install -y python3.10-dev
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements-model.txt setuptools wheel packaging
uv pip install --python .venv/bin/python --no-build-isolation -r requirements-mamba.txt
.venv/bin/python -c "import torch; from mamba_ssm import Mamba2; print('CUDA:', torch.cuda.is_available())"
```

`requirements-mamba.txt`는 실제 실행한 공식 소스 커밋과 import 호환 버전을 고정합니다. `requirements-wsl.lock`은 검증 환경의 전체 설치 목록입니다. 기본 `mamba2`는 PyTorch convolution과 공식 Triton SSD scan을 사용합니다. `mamba1`은 별도 `selective_scan_cuda` 확장이 필요하며 이 환경에서는 실행을 검증하지 않았습니다. [고정한 공식 Mamba 소스](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa)

### 학습 시작과 진행 확인

```bash
cd /home/parkj/projects/trafficflowbench
mkdir -p artifacts/v2
nohup bash train_wsl.sh --epochs 75 >> artifacts/v2/train.log 2>&1 < /dev/null &
tail -f artifacts/v2/train.log
```

State를 먼저 학습하고, 최선 State 인코더로 Queue를 초기화합니다. 기본값은 **최대 75 epoch**, 폭 64, 블록 2개, head 4개, dropout 0.1입니다. 1월 선택 지표가 5 epoch 동안 개선되지 않으면 조기 종료합니다. 반드시 75 epoch를 실행하려면 `--patience 0`을 추가합니다. 최초 실행에는 학습 기간 평균·정확한 마스크 대상의 캐시 생성과 Triton 커널 컴파일 시간이 필요합니다.

새 파일은 `artifacts/v2/state.pt`·`queue.pt`에 저장하며, 이전 `artifacts/state.pt`·`queue.pt`와 구분합니다. `.pt`는 최선 모델, `.last.pt`는 최신 epoch 재개용입니다. `.jsonl`에는 진행과 학습 지표, `.epochN.json`에는 regime별 RMSE·Queue 조건별 IoU·비교 baseline·일자별 기록이 저장됩니다. `tail -f`의 `Ctrl+C`는 로그 보기만 종료합니다.

GPU에서는 지원 시 BF16을 자동 적용하고 fused AdamW·고정 메모리·비동기 전송·다음 날짜 선처리를 사용합니다. State 기본 배치는 128링크 이하 8, 그 외 2이며, Queue는 각각 48·12입니다. Queue 창은 여러 날짜를 모아 배치를 채웁니다. `--precision fp32`, `--batch-size 4`, `--panel-batch D7_I405_N=2`, `--workers 0`으로 조정할 수 있습니다. 이전 9채널 모델의 GPU 비교 배율을 v2 성능으로 해석하지 마세요. [기존 측정 조건](MODEL_AUDIT.md#3-gpu-처리량을-더-높일-수-있는가)

개별 실행과 재개 예시입니다. `--epochs`는 재개 전 epoch를 포함한 총 예산입니다. 데이터·모델·optimizer·scheduler·난수 상태를 복원하며 같은 학습 패널과 데이터를 요구합니다.

```bash
.venv/bin/python -u src/run.py train-state --release-root data/kaggle_public \
  --checkpoint artifacts/v2/state.pt --epochs 75
.venv/bin/python -u src/run.py train-queue --release-root data/kaggle_public \
  --checkpoint artifacts/v2/queue.pt --init-state artifacts/v2/state.pt --epochs 75
.venv/bin/python -u src/run.py train-state --release-root data/kaggle_public \
  --checkpoint artifacts/v2/state.pt --resume artifacts/v2/state.last.pt --epochs 75
```

`train_wsl.sh`에는 `--resume`을 전달하지 않습니다. State·Queue 재개는 해당 명령에서 각각 실행합니다. 그래프·Physics·불확실성 옵션의 예시는 다음과 같습니다. 계수는 실험 시작값이며 공식 점수 개선을 검증한 값이 아닙니다.

```bash
.venv/bin/python -u src/run.py train-state --release-root data/kaggle_public \
  --checkpoint artifacts/v2-physics/state.pt --graph \
  --lambda-n 0.05 --lambda-dn 0.05 --lambda-fd 0.01
.venv/bin/python -u src/run.py train-queue --release-root data/kaggle_public \
  --checkpoint artifacts/v2-physics/queue.pt --init-state artifacts/v2-physics/state.pt \
  --label-temperature 3 --uncertainty-margin 5
```

2026-10-04 추가한 결측 처리·학습 비중 조정은 다음 옵션으로 비교합니다. 새 학습은 사용자가 실행하며, 기존 점수보다 개선되는지는 아직 검증하지 않았습니다.

```bash
cd /home/parkj/projects/trafficflowbench
mkdir -p artifacts/v2-improved
nohup env CHECKPOINT_DIR=artifacts/v2-improved \
  bash train_wsl.sh --epochs 10 --update-windows 0 \
  --missing-aware --graph --graph-mode reliability \
  >> artifacts/v2-improved/train.log 2>&1 < /dev/null &
tail -f artifacts/v2-improved/train.log
```

- `--update-windows 0`: State는 8개, Queue는 48개 창의 기울기를 모아 한 번 업데이트합니다. GPU에 올리는 배치는 기존 `--batch-size`·`--panel-batch`를 따릅니다. 양수를 주면 과제에 관계없이 그 수를 사용하고, 옵션을 생략하면 기존 배치마다 업데이트합니다. epoch 끝의 남은 창도 실제 대상 수로 정규화해 업데이트하며 `updates_by_panel`에 횟수를 기록합니다. 패널의 창 수가 같을 때 배치 크기로 인한 업데이트 횟수 차이를 줄이는 기능이며, regime별 표본 비중까지 맞추지는 않습니다.
- `--missing-aware`: 속도·교통량·점유율의 결측 위치에 각각 학습 가능한 임베딩을 넣고, 결측으로 표시된 채널의 수치는 입력 투영에서 제외합니다. 관측된 0은 관측값으로 유지합니다.
- `--graph --graph-mode reliability`: 공개 물리 간선에서 채널별 관측 여부·관측률·결측 경과 시간과 양쪽 노드의 특징으로 집계 가중치를 학습합니다. 관측 채널이 전혀 없거나 관측률이 0인 이웃의 직접 메시지는 차단합니다. 데이터에서 새로운 간선을 학습하거나 MagiNet 전체 구조를 재현하는 옵션은 아닙니다. `--graph`만 쓰면 기존 평균 집계를 유지합니다.

옵션 효과는 `--update-windows 0`만 적용한 실험과 결측·그래프 옵션까지 적용한 실험을 같은 seed·데이터·epoch 예산에서 비교해 확인합니다. 손실 누적 시 State는 전체 업데이트의 대상 셀 수, Queue는 전체 업데이트의 조건별 창 수로 정규화합니다. Physics 보조 항은 작은 배치별 값을 창 수 비율로 누적하므로 모든 창을 한 배치로 계산한 Physics 손실과 같다고 보장하지 않습니다.

기존 v2 체크포인트는 계속 로드할 수 있습니다. `--resume`은 저장된 모델 구조와 `--update-windows` 설정을 복원하고, `--init-state`는 저장된 인코더 구조를 Queue에 전달합니다. 기존 체크포인트에 새 결측·그래프 구조를 켜도록 요청하면 오류로 안내하므로 새 구조는 별도 경로에서 학습합니다. 위 실험은 State → Queue 순서이며 각 작업 최대 10 epoch입니다. 충분한 비교에는 `--epochs 75`를 사용합니다. 실행 중인 모델은 이미 메모리에 올라간 구조를 계속 사용합니다.

`train_wsl.sh`는 `flock`으로 같은 저장 폴더의 스크립트 중복 실행을 차단합니다. State와 Queue가 끝나면 잠금이 자동으로 해제됩니다. 수정 전에 시작된 작업과 `src/run.py`를 직접 실행한 작업에는 이 잠금이 적용되지 않습니다. 백그라운드 로그는 `>>`로 이어 써서 재실행 시 기존 내용을 지우지 않습니다.

### Station 예측층·학습형 그래프·Wavelet 실험

2026-10-05에 다음 네 옵션을 추가했습니다. 기존 기본값과 체크포인트는 유지하며, 새 구조는 **별도 경로에서 처음부터 학습**합니다. 기존 실행이 끝난 뒤 아래 명령을 사용합니다. State 다음 Queue를 자동으로 학습하며 각 작업의 최대 예산은 75 epoch, 기본 early stopping patience는 5입니다.

```bash
cd /home/parkj/projects/trafficflowbench
mkdir -p artifacts/v2-wavelet
nohup env CHECKPOINT_DIR=artifacts/v2-wavelet \
  bash train_wsl.sh --epochs 75 --update-windows 0 \
  --missing-aware --graph --graph-mode reliability \
  --station-decoder gated --adaptive-graph-k 4 \
  --wavelet-levels 2 --wavelet-input embedded --state-regime-balance \
  >> artifacts/v2-wavelet/train.log 2>&1 < /dev/null &
tail -f artifacts/v2-wavelet/train.log
```

| 옵션 | 반영한 동작 |
| --- | --- |
| `--station-decoder gated` | State station 문맥을 3→20채널로 확장합니다. 채널별 앞뒤 관측값·거리·유효성·현재 관측 여부·관측 품질·fit 평균을 작은 MLP와 게이트에 넣어 보간값의 보정량을 학습합니다. 처음 출력은 기존 보간값입니다. |
| `--adaptive-graph-k 4` | 현재 패널의 노드 임베딩에서 방향별 관계를 학습하고 노드마다 자기 자신을 제외한 상위 4개 이웃만 집계합니다. 물리 간선과 별도 경로이므로 연결 성분 사이에도 정보를 전달합니다. 관측 품질을 반영하되 현재 결측인 노드의 시간 문맥도 사용할 수 있습니다. `0`은 비활성화, 활성화 값은 2 이상이며 실제 이웃 수는 노드 수−1 이하입니다. |
| `--wavelet-levels 2` | Haar DWT 2단계로 추세와 두 변동 대역을 나누고 대역별 temporal convolution 후 inverse DWT로 합칩니다. `0`은 비활성화하며 1~3단계를 지원합니다. 위의 `--wavelet-input embedded`는 최초 실험 구조를 명시하며, 후속 관측 신호 경로는 아래에서 설명합니다. |
| `--state-regime-balance` | 선택 구간을 사용하지 않고 fit 기간의 regime별 대상 셀 수로 SSE 가중치를 계산합니다. 패널 내 R1/R2/R3 MSE의 동일 비중을 목표로 학습하며, 검증 지표·검증 손실·템플릿은 바꾸지 않습니다. 없는 regime은 학습 가중치 계산에서 제외합니다. |

Queue에는 새 인코더의 wavelet·물리 그래프·학습형 그래프를 전이합니다. station 예측층과 State regime 손실은 Queue에 적용하지 않습니다. Queue wavelet은 허용된 과거 12칸 안에서만 계산하며 forecast origin과 미래 관측을 입력하지 않습니다. 재개 시 모델 구조·regime 가중치 설정을 복원하고, State 제출 추론에서도 동일한 20채널 station 문맥을 생성합니다.

[Graph WaveNet](https://arxiv.org/abs/1906.00121)의 학습형 관계와 [WavGCRN](https://arxiv.org/abs/2401.06040)의 주파수 대역 분리를 참고한 대회용 적용입니다. 두 논문의 전체 모델 재현은 아닙니다. Haar 변환은 PyTorch 연산으로 구현해 추가 패키지가 필요 없습니다. 학습형 그래프의 관계 점수는 패널 노드 수의 제곱에 비례해 계산하지만 시계열 메시지는 선택한 이웃만 집계합니다.

위 명령은 실행 안내이며 전체 재학습을 수행했다는 뜻이 아닙니다. 점수·전체 epoch 시간·GPU 사용률 개선은 미검증입니다. 같은 seed·분할·업데이트 설정에서 옵션을 하나씩 추가한 비교와 최종 조합을 함께 평가합니다. [구현·검증 기록](MODEL_AUDIT.md#2026-10-05-station-예측층학습형-그래프wavelet-추가)

### 진단 후 수정: 관측 신호 Wavelet·채널별 예측층

2026-10-05의 [성능 진단](PERFORMANCE_DIAGNOSIS.md)에 따라 세 가지를 수정했습니다. 새 구조는 별도 저장 폴더에서 처음부터 학습합니다. 현재 실행 중인 Queue 작업은 시작할 때 읽은 기존 구조로 계속 동작합니다.

| 수정 | 동작 |
| --- | --- |
| `--wavelet-input signals` | 정규화된 속도·차로당 교통량·점유율 3채널에만 DWT를 적용합니다. 결측은 해당 채널의 fit 평균으로 채우며 관측된 0은 유지합니다. 복원 후 은닉 차원으로 투영해 합칩니다. 시각·요일·노드·상대 시점 임베딩과 결측 토큰은 기존 주 경로에 유지하고 DWT에는 넣지 않습니다. |
| `--station-decoder split` | State 속도와 교통량에 각각 독립 MLP·보정 게이트를 사용합니다. 공통 인코더·20채널 station 문맥·처음 출력이 보간값인 성질은 유지합니다. `gated`는 공유 MLP 비교용으로 남겼습니다. 게이트 값을 강제로 높이거나 교통량 보정을 제거하지 않습니다. |
| station 전처리 배열화 | station별 Python 반복을 제거하고 `[시간, station]` 배열에서 앞뒤 관측 위치·보간값·시간 거리·유효성을 함께 계산합니다. 관측 창 밖 값을 참조하지 않으며 기존 3채널과 20채널 문맥 모두 같은 계약을 유지합니다. |

다음 명령은 **터미널에 진행 로그를 표시하면서 파일에도 저장**합니다. State 이후 새 State 인코더를 전달한 Queue를 순서대로 학습합니다. 각 작업은 최대 75 epoch이며 기본 patience는 5입니다. 전면 실행이므로 `Ctrl+C`는 이 학습을 중단합니다.

```bash
cd /home/parkj/projects/trafficflowbench
mkdir -p artifacts/v2-signal-split
env CHECKPOINT_DIR=artifacts/v2-signal-split \
  bash train_wsl.sh --epochs 75 --seed 42 --update-windows 0 \
  --missing-aware --graph --graph-mode reliability \
  --station-decoder split --adaptive-graph-k 4 \
  --wavelet-levels 2 --wavelet-input signals --state-regime-balance \
  2>&1 | tee -a artifacts/v2-signal-split/train.log
```

새 학습에서 `--wavelet-input`을 생략하면 `signals`를 사용합니다. `--resume`·`--init-state`에서는 체크포인트 설정을 그대로 상속하며, 이전 체크포인트에 해당 필드가 없으면 `embedded`로 복원합니다. 구조가 다른 옵션을 명시하면 오류로 차단합니다. 새 구조로 바꾸기 위해 기존 체크포인트를 그대로 재개할 수는 없습니다. State station 예측층과 regime 손실은 Queue에 적용하지 않습니다.

공유 예측층과 regime 손실 중 무엇이 교통량에 영향을 주는지 확인하려면 나머지 설정·분할·seed·업데이트 수를 고정하고, 각 행을 다른 저장 폴더에서 비교합니다. 단독 State 비교는 위 옵션을 `src/run.py train-state`에 전달하면 됩니다.

| 비교 | `--station-decoder` | `--wavelet-input` | `--state-regime-balance` |
| --- | --- | --- | --- |
| 기존 구조 | `gated` | `embedded` | 사용 |
| Wavelet 위치만 변경 | `gated` | `signals` | 사용 |
| 예측층만 분리 | `split` | `embedded` | 사용 |
| 두 구조 변경 | `split` | `signals` | 사용 |
| regime 손실 영향 | `split` | `signals` | 생략 |

전처리는 위 비교 모두에 적용합니다. 실제 공개 2개 패널의 하루 24개 창에서 기존·새 특징과 문맥이 정확히 같았고, 20채널 전처리 중앙값은 81-station에서 **7.60→2.85ms**, 352-station에서 **31.53→9.32ms**였습니다. 같은 입력을 메모리에 둔 11회 교대 측정이며 기존 Queue 학습이 함께 실행 중이었습니다. 파일 읽기·전체 epoch 시간·점수 개선을 측정한 값은 아닙니다. [측정 결과](artifacts/signal-split-check/results.json), [재현 스크립트](artifacts/signal-split-check/verify.py), [검증 기록](MODEL_AUDIT.md#2026-10-05-진단-후-세-가지-수정)

### Task 2: 입력·표본·출력층 비교

2026-10-25에 Queue 전용 실험 옵션을 추가했습니다. 옵션을 생략하면 기존 31특징·12시점·선형 출력층과 표본 선택을 유지합니다. 기본 손실 `BCE + 0.2 × soft-IoU + 0.1 × 미래 속도 Smooth L1`은 바꾸지 않습니다. 아래 구조의 점수 개선은 전체 학습 비교로 확인해야 합니다.

| 옵션 | 변경 내용 |
| --- | --- |
| `--queue-origin` | 공개 `mainline_states_masked`의 `T` 행을 추가해 13시점으로 입력합니다. train에서도 masked 원본을 사용하며, 가려진 채널·없는 링크·없는 시각은 결측으로 유지합니다. `T` 이후 값으로 보간하지 않습니다. |
| `--queue-resample` | 학습 창을 epoch별 seed로 재추출합니다. 검증 창은 epoch와 무관하게 고정합니다. |
| `--queue-condition-gap` | 학습 창의 기본 360분 간격을 onset·ongoing·normal 각각에 적용합니다. |
| `--queue-head pooled-mlp` | 마지막 시점과 시간 평균 표현을 연결해 `128 → 64 → 12` MLP로 미래 여섯 시점을 동시에 예측합니다. 차원은 기본 폭 64 기준입니다. |
| `--queue-changes` | `속도 - v_cut`과 속도·차로당 교통량·점유율의 5·15분 변화량 및 유효 마스크를 추가합니다. 총 44특징이며, 변화량은 입력 창 내 양 끝 관측과 정확한 시간 간격이 있을 때만 계산합니다. |
| `--queue-validation-spread --validation-windows 20` | 조건별로 날짜를 고르게 방문하고 하루 최대 한 창씩, 패널·조건별 최대 20개를 평가합니다. 검증의 360분 간격은 조건별로 적용합니다. 조건이 부족하면 실제 창 수가 더 적습니다. |

먼저 같은 seed·날짜·검증 설정·epoch 예산에서 기본형과 각 옵션을 하나씩 비교합니다. 아래 명령은 WSL 프로젝트 루트에서 실행하는 MLP 단독 비교 예시입니다. 입력·표본 옵션은 생략하고, 다른 후보에도 같은 `--queue-validation-spread --validation-windows 20`을 전달합니다.

```bash
cd /home/parkj/projects/trafficflowbench
.venv/bin/python -u src/run.py train-queue \
  --release-root data/kaggle_public --checkpoint artifacts/task2/pooled/queue.pt \
  --epochs 75 --seed 42 --precision bf16 \
  --queue-validation-spread --validation-windows 20 --queue-head pooled-mlp
```

`--queue-head pooled-mlp`를 생략한 기본형을 별도 폴더에 저장하고, `--queue-origin`, `--queue-resample`, `--queue-condition-gap`, `--queue-changes`도 각각 별도 폴더에서 비교합니다. 초기 실행 검사에는 `--panel D12_I5_S --max-days 3 --epochs 2`를 사용합니다. 전체 기간의 비교와 실행 검사를 구분합니다.

기존 모델도 저장된 입력·출력층·판정 임계값을 유지하면서 더 많은 고정 창에서 재평가할 수 있습니다.

```bash
.venv/bin/python src/run.py evaluate --release-root data/kaggle_public \
  --checkpoint artifacts/v2/queue.pt --period select \
  --queue-validation-spread --validation-windows 20 \
  --output reports/task2-baseline-spread.json
```

보고서의 `window_signature`가 같아야 동일 창 비교입니다. `proxy_IoU`, `onset`, `ongoing`, `normal_false_positive`, 패널별 결과를 함께 확인합니다. 학습 로그의 `fit_windows`와 `optimizer_updates`로 표본 수·업데이트 비용도 확인합니다. 현재는 1월에서 모델과 판정 임계값을 선택하고, 2월 평가에는 저장된 임계값을 고정합니다. 공식 정체 정답이 없는 상태의 대리 검증입니다.

새 옵션은 체크포인트에 저장되며 재개·평가·제출 추론에도 같은 설정을 사용합니다. 기존 체크포인트를 `--resume`하면서 입력·표본·출력층 옵션을 새로 켜면 오류로 안내합니다. 새 후보는 새 저장 경로에서 학습해 주세요. `--init-state`는 FAST v2 State 인코더 전용이며, 변화량 특징을 추가하면 새 입력 열의 전이 가중치를 0으로 시작합니다. `T` 입력을 쓴 모델은 재개 시 해당 학습용 masked 파일의 변경도 검사합니다.

### 선택 구간과 2월 잠금 평가

학습은 2030년 6~12월, 선택은 2031년 1월입니다. State는 배포된 `mainline_states_masked`와 Task 1 train 템플릿의 **정확한 대상 셀**을 사용합니다. 선택 지표는 RMSE 기반 State를 regime·방향·도로군으로 집계합니다. Queue는 공개 관측으로 만든 대리 라벨에서 onset·ongoing을 패널·도로군별로 집계하며, 정상 유지 창의 오탐과 미래 속도 RMSE도 기록합니다.

```bash
.venv/bin/python src/run.py evaluate --release-root data/kaggle_public \
  --checkpoint artifacts/v2/state.pt --period holdout \
  --output reports/state-february.json --bootstrap 1000
.venv/bin/python src/run.py evaluate --release-root data/kaggle_public \
  --checkpoint artifacts/v2/queue.pt --period holdout \
  --output reports/queue-february.json --bootstrap 1000
.venv/bin/python src/run.py validate-odme --release-root data/kaggle_public \
  --split validation --lambdas 0.01 0.05 0.2 --output reports/odme-lambda.json
```

`holdout`에서는 저장된 Queue 임계값을 고정하고 재선택하지 않습니다. bootstrap은 같은 일자를 재추출한 모델–baseline 차이의 구간입니다. State baseline은 같은 대상의 시간 보간·fit 구간 링크별 요일/시간 평균이며, 공식 station별 과거 평균 baseline과 완전히 같은 구현은 아닙니다. `physics_aux_normalized_mse`는 N·dN·FD 학습 항의 진단이며 공식 Physics 점수가 아닙니다. ODME의 숨긴 관측 링크 RMSE도 공개 데이터로 만든 대리 지표입니다.

### Task 4: 가중 Prior NNLS

Task 4는 해당 split의 관측 링크만 희소 행렬로 읽고 다음 문제를 풉니다. 기본값 `p=0`, `λ=0.05`는 기존 정규화 NNLS와 같은 목적함수입니다. 데이터 다운로드나 신경망 학습은 수행하지 않습니다.

```text
minimize over f >= 0: ||A_measured f - c_measured||² + λ sum_i (f_i-b_i)² / v_i
v_i = (b_i + τ)^p / median_k((b_k + τ)^p), p in {0, 1, 2}
τ = prior_floor_ratio * median(positive b), default prior_floor_ratio = 0.1
```

`p=0`은 같은 절대오차, `p=1`은 Prior에 비례하는 오차 분산, `p=2`는 비슷한 상대오차를 가정합니다. Prior가 모두 0이면 양수 중앙값 대신 1을 사용하며, `v_i`도 모두 1입니다. Prior가 0인 경로도 양수로 수정할 수 있습니다. 우선 `p=0`에서 λ를 비교하고 이후 아래처럼 오차 가정을 확장합니다.

```bash
.venv/bin/python src/run.py validate-odme --release-root data/kaggle_public \
  --split validation --lambdas 0.01 0.05 0.2 --prior-powers 0 1 2 \
  --prior-floor-ratio 0.1 --repeats 3 --seed 42 \
  --output reports/odme-weighted.json
```

매 반복에서 관측 링크의 약 20%를 숨기고 모든 후보에 같은 분할을 적용합니다. Prior 그대로와 `λ=0.05, p=0`은 항상 비교에 포함합니다. 보고서에는 숨긴 링크의 정규화 L1 오차·RMSE, Prior 이동량 L1·L2, Prior 대비 목적지 분포 변화, 0으로 추정한 경로 비율을 기록합니다. 반복 평균을 패널별로 계산한 뒤 방향·도로군을 동일 가중으로 집계하며, 반복별 L1 오차의 표준편차와 최악 패널도 표시합니다. 이 표준편차는 신뢰구간이 아닙니다.

`structure`에는 관측되지 않는 경로, 관측 링크 조합이 같은 경로, OD당 경로 수를 기록합니다. 동일 관측 조합이 없더라도 모든 경로가 식별된다는 뜻은 아닙니다. `destination_distribution_l1_from_prior`는 Prior와의 변화량이며 정답 기반 `S_attr`가 아닙니다. 총교통량이 0인 분포의 비교는 `null`입니다. `best_proxy_candidate`는 숨긴 링크 L1 오차가 가장 작은 후보로, 전체 `S_ODME`의 최선이나 성능 개선을 보장하지 않습니다. 보고서의 `proxy_only`는 항상 `true`이며, 선택 결과를 제출 설정에 자동 적용하지 않습니다.

Task 4 예측만 저장하려면 아래 명령을 사용합니다. `--prior-power`와 `--odme-lambda`는 직접 선택한 값으로 바꾸며, 전체 관측 링크를 사용해 다시 풉니다. `--panel`은 작은 패널만 확인할 때 반복 지정할 수 있습니다. Prior 자체의 CSV는 `--prior-only`로 저장합니다.

```bash
.venv/bin/python src/run.py predict-odme --release-root data/kaggle_public \
  --split validation --odme-lambda 0.05 --prior-power 0 \
  --output reports/odme-validation.csv
.venv/bin/python src/run.py predict-odme --release-root data/kaggle_public \
  --split private --odme-lambda 0.05 --prior-power 0 \
  --output reports/odme-private.csv
```

출력은 `sample_submission_path_flow.csv`의 키·구역·행 순서를 유지하는 과제별 CSV입니다. Kaggle 업로드 전에는 다른 과제와 함께 병합해야 합니다. 전체 `submit` 명령에서도 `--odme-lambda`, `--odme-prior-power`, `--odme-prior-floor-ratio`로 같은 설정을 전달할 수 있습니다. 가중 모델의 실제 데이터 성능과 실행 시간은 미측정입니다.

### 제출과 실행 검사

공식 저장소는 외부 체크아웃으로 사용합니다. 체크포인트를 생략하면 공식 State 과거 평균·Queue persistence와 NNLS를 조합합니다. 전체 대상 패널을 학습한 후 모델 체크포인트를 지정합니다.

```bash
.venv/bin/python src/run.py submit --official-root /path/to/trafficflowbench-public \
  --release-root data/kaggle_public --output-dir reports/model \
  --state-checkpoint artifacts/v2/state.pt --queue-checkpoint artifacts/v2/queue.pt
```

`--state-checkpoint`·`--queue-checkpoint`를 반복하면 State 연속값·Queue 확률을 각각 평균합니다. 여러 seed는 `--seed`와 별도 저장 경로로 학습합니다. 단일 패널 실행 점검 모델로 전체 제출을 만들지 않습니다. 공식 병합의 누락 경고가 없어야 최종 `submission.csv`를 확정합니다.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

이전 9채널 체크포인트의 1월 재평가 값은 State 정규화 MSE 0.009547, Queue 대리 IoU 0.4548입니다. 새 v2와 다른 마스크·라벨·집계 조건이므로 직접 성능 비교하지 않습니다. 전체 v2 학습·여러 seed 성능 비교·Kaggle 점수 재현은 아직 수행하지 않았습니다. [검증 기록](MODEL_AUDIT.md)

## 12. MagiNet을 로컬에서 실행하기

`src/maginet.py`는 `notebooks/task13_maginet_colab.ipynb`의 AMSTenc·MASTdec·양방향 GRU·보간 잔차 출력과 State·N·ΔN 손실을 옮긴 로컬 실행본입니다. 노트북은 Colab 실험 기록으로 유지합니다. 로컬에서는 Drive 마운트·Jupyter·IPython·Matplotlib·Mamba 패키지 없이 기존 `requirements-model.txt`의 패키지를 사용합니다. 그래프 이미지 대신 점수·학습 기록을 CSV로 저장합니다.

기본 실험은 노트북 설정과 같은 `temporal_direct`입니다. 기존 `baseline`, `dn_003`, `dn_zero`, `mask_resample`, `learned_graph`와 아래의 시간 경로를 유지하는 분리 실험을 `--experiment`로 선택합니다. 새 설정의 성능 개선은 별도 학습·평가로 확인해야 합니다.

| 경로·옵션 | 동작 |
| --- | --- |
| `data/release_root.txt` | 기존 로컬 데이터 경로를 읽습니다. `--release-root`로 직접 지정할 수도 있습니다. |
| `data/maginet_cache/` | 날짜별 NPZ, fit 과거 평균과 정규화 통계를 계속 보관합니다. `--cache-dir`로 변경합니다. |
| `reports/maginet/<실험명>/` | 기본 결과 폴더입니다. `--output-dir`로 별도 실행 폴더를 지정합니다. |
| `--cache-mode build` | 기본값입니다. 없거나 원본이 바뀐 캐시를 생성·갱신하고, 검증된 캐시는 재사용합니다. |
| `--cache-mode reuse` | 필요한 캐시가 없거나 원본 변경·출처 기록 누락을 발견하면 중단합니다. |
| `--device auto --precision auto` | CUDA가 있으면 지원되는 BF16 또는 FP16을 사용합니다. CPU에서는 FP32로 실행합니다. |

점검용·전체 학습용 캐시는 fit 날짜·과거 평균·도로 정보에 따른 별도 경로로 저장합니다. 모델 실험만 바뀌고 입력이 같으면 같은 캐시를 사용합니다. 데이터·캐시·결과는 기존 Git 제외 경로 안에 있습니다.

프로필과 일별 NPZ 옆의 `.sources.json`에 실제 입력 파일의 내용 SHA-256을 기록합니다. fit 원본·도로 CSV, 날짜별 원본·masked·램프 파일과 템플릿 대상이 검증에 포함됩니다. 파일 수정 시각이 같아도 내용 변경을 감지하며, 절대 경로를 제외하므로 같은 데이터를 다른 위치로 옮겨도 검증할 수 있습니다.

이전 캐시는 출처 기록이 없으므로 `prepare --cache-mode build`로 원본과 한 번 대조해야 합니다. 계산된 배열이 같으면 기존 NPZ를 다시 쓰지 않고 출처 기록만 추가해 기존 재개 서명을 유지합니다. 변경된 입력은 필요한 캐시를 갱신하며, 학습 입력이 달라진 기존 실행의 재개는 거부합니다. 이후에도 입력 내용을 해시로 확인하는 읽기 비용은 발생합니다.

### 작은 실행 점검과 전체 학습

다음은 Windows PowerShell에서 프로젝트 루트를 기준으로 실행하는 명령입니다. 첫 명령은 한 패널에서 fit·early·blend의 regime별 하루, 2 epoch, regime별 업데이트 1회로 전체 흐름을 확인합니다. 기본 5 epoch의 물리 손실 준비 기간을 유지하므로 이 점검 학습은 State만 사용합니다. 전체 데이터 성능을 판단하는 실행이 아닙니다.

```powershell
.venv\Scripts\python.exe -u src\maginet.py smoke --panel D12_I405_S --precision bf16
.venv\Scripts\python.exe -u src\maginet.py prepare --experiment temporal_direct
.venv\Scripts\python.exe -u src\maginet.py train --experiment temporal_direct --epochs 50
```

`prepare`는 입력 준비만 수행합니다. `train`도 필요한 입력을 준비하므로 캐시가 이미 있으면 바로 `train`을 실행해도 됩니다. `--panel`을 생략하면 10개 패널을 공동 학습합니다. `--days-per-regime N`으로 기간·regime별 날짜를 줄일 수 있으며, fit 과거 평균도 제한한 fit 날짜로만 계산합니다.

기본 배치는 2, 폭 64, 블록 3개, 최대 50 epoch입니다. `--batch-size`, `--width`, `--blocks`, `--heads`, `--updates-per-group`, `--lr`, `--patience`로 조정합니다. epoch별 로그를 출력하며 `full/best_maginet.pt`, `full/last_maginet.pt`, `full/history.csv`를 저장합니다. 기존 체크포인트가 있으면 새 학습으로 덮어쓰지 않습니다.

```powershell
.venv\Scripts\python.exe -u src\maginet.py train --experiment temporal_direct --resume --epochs 75
```

`--epochs`는 재개 이전을 포함한 총 epoch 수입니다. 설정·입력 캐시 내용·도로 정보가 같아야 재개하며 optimizer·scheduler·난수 상태도 복원합니다. 데이터·캐시·실행 폴더의 절대 경로는 서명에서 제외하므로 동일 내용을 다른 위치로 옮길 수 있습니다. 장치·PyTorch·정밀도·CPU 스레드 수가 바뀌면 수치적으로 같은 학습을 보장하지 않습니다. 이 실행 환경 정보도 체크포인트에 기록합니다.

### 시간 경로를 유지하는 분리 실험과 패널별 배치

2026-10-23에 WSL의 기존 10개 패널·50 epoch 실행을 확인했습니다. 1월 후반 State는 시간 보간 0.906766, `temporal_direct` 0.926482였으며 30개 패널×regime 모두 개선됐습니다. 공식 비공개 점수가 아닌 로컬 선택 구간 결과입니다. 이후 비교는 학습 기간 연장보다 아래 손실·구조를 각각 바꾸는 실험으로 진행합니다.

| 실험 | 시간 경로 | N 가중치 | ΔN 가중치 | 학습형 공간 경로 |
| --- | --- | ---: | ---: | --- |
| `temporal_direct` | 유지 | 0.05 | 0.10 | 없음 |
| `temporal_direct_dn003` | 유지 | 0.05 | 0.03 | 없음 |
| `temporal_direct_dnzero` | 유지 | 0.05 | 0 | 없음 |
| `temporal_direct_learned_graph` | 유지 | 0.05 | 0.10 | 추가 |
| `temporal_direct_dn003_learned_graph` | 유지 | 0.05 | 0.03 | 추가 |

기존 `dn_003`, `dn_zero`, `learned_graph`는 시간 경로가 꺼진 설정입니다. `temporal_direct_dn003_learned_graph`는 `temporal_direct_dn003`에서 학습형 공간 경로만 추가합니다. 나머지 위 분리 실험은 `temporal_direct`에서 한 요소만 바꾸며, fit 과거 평균·정규화·캐시·평가 기간을 공유합니다. `temporal_direct_dnzero`도 N 손실은 유지합니다.

다음은 WSL 프로젝트 루트에서 실행하는 전체 데이터 비교 명령입니다. 원래 실행과 같은 배치 2·최대 50 epoch를 사용하고 각 실험의 새 기본 결과 폴더에 저장합니다. 기존 50 epoch 실행에 `--resume`으로 설정을 바꾸지 않습니다.

```bash
.venv/bin/python -u src/maginet.py train --release-root data/kaggle_public --experiment temporal_direct_dn003 --cache-mode reuse --epochs 50 --precision bf16
.venv/bin/python -u src/maginet.py train --release-root data/kaggle_public --experiment temporal_direct_dnzero --cache-mode reuse --epochs 50 --precision bf16
.venv/bin/python -u src/maginet.py train --release-root data/kaggle_public --experiment temporal_direct_learned_graph --cache-mode reuse --epochs 50 --precision bf16
```

처리량 비교는 별도 실행에서 `--batch-policy adaptive`로 선택합니다. 관측소가 128개 이하이면 배치 8, 초과하면 4를 사용합니다. `--panel-batch PANEL=SIZE`를 반복해서 패널별로 재지정할 수 있으며 policy보다 우선합니다. 기본은 이전과 같은 `fixed`·배치 2입니다. `--batch-size`는 `fixed`에서 사용합니다.

```bash
.venv/bin/python -u src/maginet.py train --release-root data/kaggle_public --experiment temporal_direct --batch-policy adaptive --panel-batch D7_I405_N=4 --cache-mode reuse --epochs 50 --precision bf16 --output-dir reports/maginet/temporal_direct_adaptive
```

패널별 배치는 config·체크포인트에 저장하고 재개·평가·추론에도 적용합니다. 같은 실행을 재개할 때 옵션을 다시 입력하지 않아도 되며, 배치 설정이 달라지면 재개를 거부합니다. 이전 로컬 체크포인트에는 기본 `fixed` 설정을 적용합니다. 평가·추론 명령에서 `--batch-policy`·`--panel-batch`를 다시 지정하지 않습니다. `evaluate`·`predict --batch-size N`은 해당 실행에서 모든 패널의 배치를 N으로 덮어쓰며 체크포인트를 변경하지 않습니다. `--width`, `--lr`, `--input-normalization` 등 학습 설정은 평가·추론에서 지정하면 오류로 안내합니다.

`history.csv`에는 `optimizer_updates`와 `train_windows`를 추가로 기록합니다. adaptive도 패널×regime당 update 수는 같아서 한 epoch의 학습 창 수가 늘어납니다. 손실·그래프 효과의 분리 비교에는 같은 fixed 배치를 사용하고, adaptive의 속도·성능은 창 수·update 수·시간을 함께 비교해 주세요. 위 배치 크기는 처리량 측정에서 정한 시작값이며 전체 학습의 성능 향상을 보장하지 않습니다.

### 평가와 Task 1·3 예측

학습은 6~12월, 체크포인트 선택은 1월 전반, 보간 혼합 비율 선택은 1월 후반을 사용합니다. 학습 완료 후 공통 `alpha`를 자동 선택해 `selected_alpha.json`에 저장합니다. 기존 실험의 기본 선택 기준은 노트북의 `0.70 State + 0.10 FD_proxy + 0.20 LWR_proxy`이며 공식 Physics 점수가 아닙니다. 아래 새 문맥·채널 공간 실험은 기본으로 State를 사용합니다. 체크포인트·scheduler·조기 종료·공통 `alpha`에 같은 선택 기준을 적용하고 Physics 대리 점수도 계속 기록합니다.

```powershell
.venv\Scripts\python.exe -u src\maginet.py evaluate --checkpoint reports\maginet\temporal_direct\full\best_maginet.pt --period blend --select-alpha
.venv\Scripts\python.exe -u src\maginet.py evaluate --checkpoint reports\maginet\temporal_direct\full\best_maginet.pt --period february
.venv\Scripts\python.exe -u src\maginet.py predict --checkpoint reports\maginet\temporal_direct\full\best_maginet.pt --split both
```

2월은 저장된 `alpha`를 고정한 추가 진단입니다. 이전 실험에서 사용한 달이므로 독립적인 최종 평가라고 부르지 않습니다. `--alpha`를 명시하면 해당 비율과 보간을 비교할 수 있습니다. 저장된 선택값은 체크포인트 내용·평가 패널과 대조하며, 체크포인트가 바뀌면 1월 후반에서 다시 선택합니다.

체크포인트에는 입력 정규화 통계·관측소 순서·도로 정보·fit 과거 평균을 함께 저장합니다. `predict`에는 공개 validation/private 입력과 템플릿만 필요하며 학습용 원본·fit NPZ를 다시 읽지 않습니다. 패널별 CSV는 `submissions/task1-<패널>-<split>.csv`에 저장하고 관측소 총 교통량으로 환산합니다. 이 파일이 Task 3 입력도 됩니다. 기본 출력 폴더는 지정한 체크포인트의 실행 폴더이며 `--output-dir`로 변경할 수 있습니다.

체크포인트가 `full/` 안에 있으면 그 위 실행 폴더에서 `selected_alpha.json`을 찾고, 다른 위치의 체크포인트는 같은 폴더에서 찾습니다. 따라서 `portable/best.pt`에 대해 `evaluate --select-alpha`로 만든 혼합 선택 파일을 그대로 `predict`에서 사용할 수 있습니다. 선택 파일을 별도 출력 폴더에 저장했다면 `--selection`으로 지정합니다.

### 과거 구간 검색을 추가하는 WSL 실행

기존 신경망을 고정하고 fit 기간의 과거 패턴을 보조 예측으로 섞을 수 있습니다. 신경망 재학습·`--resume`은 필요하지 않습니다. 이 옵션에는 체크포인트와 같은 fit NPZ·학습 원본이 추가로 필요합니다. 기존 입력 캐시를 읽어 메모리에 검색 자료를 구성하며 별도 검색 캐시 파일은 만들지 않습니다.

`--select-retrieval`은 1월 전반에서 하루 전체 검색, 3시간 구간 검색, 신뢰도를 반영한 구간 검색을 비교하고 State 기준으로 방법 하나와 속도·교통량별 혼합 비율 두 개를 선택합니다. 구간 검색은 90분 간격으로 겹치며 과거 패턴을 ±10분 정렬합니다. 같은 과거 날짜가 여러 검색 이웃 자리를 차지하지 않게 하고, 관측과 잘 맞지 않거나 후보 간 차이가 크면 신뢰도를 낮춥니다. 검색 거리·신뢰도에는 공개된 관측만 사용합니다.

다음 명령은 WSL의 `/home/parkj/projects/trafficflowbench`에서 실행합니다. 입력 캐시가 준비된 기존 `temporal_direct_dn003` 체크포인트를 사용하며 선택 결과는 `reports/maginet/temporal_direct_dn003/retrieval/selected_retrieval.json`에 저장합니다. 두 번째 명령은 비율을 고정해 1월 후반을 평가합니다.

```bash
cd /home/parkj/projects/trafficflowbench
.venv/bin/python -u src/maginet.py evaluate --release-root data/kaggle_public --checkpoint reports/maginet/temporal_direct_dn003/full/best_maginet.pt --cache-mode reuse --precision bf16 --period early --select-retrieval
.venv/bin/python -u src/maginet.py evaluate --release-root data/kaggle_public --checkpoint reports/maginet/temporal_direct_dn003/full/best_maginet.pt --cache-mode reuse --precision bf16 --period blend --retrieval-selection reports/maginet/temporal_direct_dn003/retrieval/selected_retrieval.json
```

선택 구간의 방법별 점수는 `retrieval_early_comparison.csv`, 후반 점수는 `blend_macro.csv`, 패널별 점수는 `blend_panel_regime.csv`에 저장합니다. `alpha_0.0`은 시간 보간, `alpha_1.0`은 검색을 섞은 신경망 결과입니다. 검색 옵션을 지정하면 기본 출력은 기존 실행 폴더 아래 `retrieval/`이며 원래 평가 파일과 선택한 `alpha`를 덮어쓰지 않습니다. 검색은 신경망 예측에 혼합하도록 선택하므로 기존 보간 `alpha` 선택과 함께 지정하지 않습니다.

validation/private 예측 명령입니다. 선택 파일과 체크포인트·fit 자료의 해시를 대조한 후 고정된 방법과 비율만 적용합니다. 출력은 `retrieval/submissions/task1-<패널>-<split>.csv`이며 Task 3 입력으로도 사용할 수 있습니다. 1월 전반·후반과 2월은 반복 사용한 개발 구간이므로 공식 점수나 새로운 독립 시험으로 해석하지 않습니다.

```bash
.venv/bin/python -u src/maginet.py predict --release-root data/kaggle_public --checkpoint reports/maginet/temporal_direct_dn003/full/best_maginet.pt --cache-mode reuse --precision bf16 --retrieval-selection reports/maginet/temporal_direct_dn003/retrieval/selected_retrieval.json --split both
```

다음은 `dN=0.03`을 유지하고 학습형 공간 경로만 추가하는 별도 학습입니다. 새 기본 폴더 `reports/maginet/temporal_direct_dn003_learned_graph/`에 처음부터 학습하며 기존 `.03` 실행에 `--resume`으로 구조를 끼워 넣지 않습니다. 새 조합의 전체 데이터 성능은 별도 학습으로 확인해야 합니다.

```bash
.venv/bin/python -u src/maginet.py train --release-root data/kaggle_public --experiment temporal_direct_dn003_learned_graph --cache-mode reuse --epochs 50 --precision bf16
```

학습 완료 후에는 다음 두 명령으로 새 모델의 검색 비율을 선택하고 평가합니다. WSL 프로젝트 루트에서 실행하며 셸 변수 없이 각각 실행할 수 있습니다. 새 모델이므로 기존 `.03` 모델의 검색 선택 파일을 재사용하지 않습니다. `/full/best_maginet.pt`처럼 모델 이름이 사라진 경로가 오류에 나오면 비어 있는 셸 변수가 명령에 들어갔는지 확인해 주세요.

```bash
.venv/bin/python -u src/maginet.py evaluate --release-root data/kaggle_public --checkpoint reports/maginet/temporal_direct_dn003_learned_graph/full/best_maginet.pt --cache-mode reuse --precision bf16 --period early --select-retrieval
.venv/bin/python -u src/maginet.py evaluate --release-root data/kaggle_public --checkpoint reports/maginet/temporal_direct_dn003_learned_graph/full/best_maginet.pt --cache-mode reuse --precision bf16 --period blend --retrieval-selection reports/maginet/temporal_direct_dn003_learned_graph/retrieval/selected_retrieval.json
```

### 관측 문맥과 채널별 공간 보정

2026-10-24 문헌 검토의 첫 개발 단계인 관측 신뢰도 결합과 채널별 공간 잔차를 구현했습니다. 기존 fit 입력·캐시·손실 계수 `N=0.05`, `dN=0.03`을 사용하며 새 외부 패키지는 필요하지 않습니다. 두 실험은 처음부터 학습하는 별도 후보입니다. 기존 실행을 `--resume`해서 구조를 바꾸지 않습니다.

| 실험 | 추가 공간 경로 | 기본 선택 기준 |
| --- | --- | --- |
| `temporal_direct_dn003_context_graph` | 기존 블록의 스칼라 게이트에 관측소·시각별 문맥 게이트를 추가합니다. | State |
| `temporal_direct_dn003_channel_graph` | 블록의 추가 학습형 그래프를 출력 쪽으로 옮겨 속도·교통량별 공간 잔차와 게이트를 둡니다. | State |

게이트에는 공개된 속도·교통량·occupancy 관측 여부, 같은 시각에 속도·교통량이 모두 관측된 물리 이웃 수·비율, 물리 이웃 존재 여부, 좌우 관측까지의 시간 간격을 사용합니다. 이웃 집계는 자신을 제외합니다. 마스크를 반영한 기존 공간 Attention과 집계를 재사용하며, 관측소 전체가 비어 있는 시각에는 채널 공간 경로가 bias만으로 보정하지 않습니다. 채널별 공간 보정은 0에서 시작하고 학습으로 켜집니다. 공유 backbone도 함께 학습하므로 기존 교통량 출력의 보존을 보장하는 구성은 아닙니다.

다음 명령은 WSL의 `/home/parkj/projects/trafficflowbench`에서 실행합니다. 권장 첫 후보는 `channel_graph`입니다. 전체 50 epoch 학습과 목표 0.95 달성 여부는 별도 실측이 필요합니다.

```bash
cd /home/parkj/projects/trafficflowbench
.venv/bin/python -u src/maginet.py train \
  --release-root data/kaggle_public \
  --experiment temporal_direct_dn003_channel_graph \
  --cache-mode reuse --epochs 50 --precision bf16
```

결과는 `reports/maginet/temporal_direct_dn003_channel_graph/`에 저장합니다. `full/best_maginet.pt`와 `full/best_state_maginet.pt`는 1월 전반 State가 가장 높은 같은 모델입니다. `full/last_maginet.pt`는 optimizer·scheduler·난수 상태를 포함한 재개용이며, `history.csv`에는 State·Physics 복합 대리 점수·실제 선택 점수 `early_checkpoint_score`를 함께 기록합니다. 자동 보간 선택은 별도 `selected_alpha.json`에 저장합니다.

학습 후 새 모델로 검색 비율을 선택하고, 고정한 비율로 평가합니다. 셸 변수 없이 각각 실행할 수 있습니다. 기존 모델의 검색 선택 파일은 새 체크포인트와 맞지 않으므로 새로 선택합니다.

```bash
.venv/bin/python -u src/maginet.py evaluate \
  --release-root data/kaggle_public \
  --checkpoint reports/maginet/temporal_direct_dn003_channel_graph/full/best_maginet.pt \
  --cache-mode reuse --precision bf16 --period early --select-retrieval
.venv/bin/python -u src/maginet.py evaluate \
  --release-root data/kaggle_public \
  --checkpoint reports/maginet/temporal_direct_dn003_channel_graph/full/best_maginet.pt \
  --cache-mode reuse --precision bf16 --period blend \
  --retrieval-selection reports/maginet/temporal_direct_dn003_channel_graph/retrieval/selected_retrieval.json
.venv/bin/python -u src/maginet.py predict \
  --release-root data/kaggle_public \
  --checkpoint reports/maginet/temporal_direct_dn003_channel_graph/full/best_maginet.pt \
  --cache-mode reuse --precision bf16 --split both \
  --retrieval-selection reports/maginet/temporal_direct_dn003_channel_graph/retrieval/selected_retrieval.json
```

검색을 적용한 평가는 `retrieval/blend_macro.csv`, 제출 CSV는 `retrieval/submissions/`에 저장합니다. 검색 없는 모델 단독 비교는 검색 옵션 없이 `evaluate --period blend --alpha 1`로 실행합니다. 기존 `.03`·이전 graph와 구조 자체를 비교할 때에는 같은 State 선택·학습량의 별도 실행도 준비해 주세요.

```bash
.venv/bin/python -u src/maginet.py train \
  --release-root data/kaggle_public --experiment temporal_direct_dn003_context_graph \
  --cache-mode reuse --epochs 50 --precision bf16
.venv/bin/python -u src/maginet.py train \
  --release-root data/kaggle_public --experiment temporal_direct_dn003 \
  --checkpoint-metric state --output-dir reports/maginet/temporal_direct_dn003_state \
  --cache-mode reuse --epochs 50 --precision bf16
```

`--checkpoint-metric state|proxy`로 새 학습의 선택 기준을 지정할 수 있습니다. 재개·평가·추론은 저장한 구조와 선택 기준을 복원합니다. 재개 시 기준을 바꾸면 설정 불일치로 중단합니다. 새 동일 실행을 총 75 epoch까지 이어갈 경우에만 다음 명령을 사용합니다.

```bash
.venv/bin/python -u src/maginet.py train \
  --release-root data/kaggle_public --experiment temporal_direct_dn003_channel_graph \
  --cache-mode reuse --precision bf16 --resume --epochs 75
```

이번 단계에는 방향·시간차 메시지, hypergraph, 주파수 분해를 포함하지 않았습니다. 문맥·채널별 결합의 실익을 확인한 뒤 별도 구조 실험으로 비교합니다. 이전 모델도 계속 저장한 설정으로 평가·추론할 수 있습니다.

### Colab 캐시·체크포인트 가져오기

Colab에서 내려받은 `full/config.json`, `best_maginet.pt`, 선택적인 `last_maginet.pt`와 fit 과거 평균을 별도 폴더에 보관합니다. 기존 NPZ ZIP이 있다면 다음처럼 가져올 수 있습니다. 경로는 실제 로컬 파일로 바꿔 주세요.

```powershell
.venv\Scripts\python.exe -u src\maginet.py import-colab --source-run C:\models\maginet-colab --cache-zip C:\models\preprocessed.zip --output-dir reports\maginet\imported
```

ZIP 없이 이미 푼 `<패널>/fit|early|blend/YYYY-MM-DD.npz` 폴더를 `--cache-dir`로 지정할 수도 있습니다. 과거 평균이 다른 위치에 있으면 `--legacy-profile-dir`로 이전 `full/` 폴더를 지정합니다. 출처 기록이 없는 이전 NPZ·과거 평균은 기본 `build` 모드에서 실제 원본과 한 번 대조합니다. NPZ가 없으면 같은 fit 기간으로 로컬 입력을 생성합니다. `mask_resample`은 노트북과 마찬가지로 대상 occupancy 원본이 없는 캐시로 학습할 수 없습니다.

가져오기는 원래 설정과 체크포인트 서명, 관측소·기간·fit 과거 평균·입력 정규화가 맞는지 확인하고 별도 로컬 체크포인트를 만듭니다. 원본 Colab 파일은 수정하지 않습니다. `last_maginet.pt`까지 가져온 경우 같은 `--output-dir`에서 `train --resume`으로 재개할 수 있습니다. 기존 보간 선택값은 새 로컬 체크포인트에 그대로 연결하지 않으므로 `evaluate --period blend --select-alpha`를 실행해 다시 선택합니다.

### 다른 과제와 최종 병합

`src/run.py submit --state-csv`는 MagiNet CSV를 받아 기존 Queue·ODME와 공식 병합기로 연결합니다. 10개 패널의 validation/private 예측이 모두 있어야 하며 누락·중복·음수·비유한 값·템플릿 밖 키는 거부합니다. `--state-checkpoint`와 함께 지정하지 않습니다.

```powershell
$release = (Get-Content data\release_root.txt -Raw).Trim()
$stateFiles = Get-ChildItem reports\maginet\temporal_direct\submissions\task1-*.csv |
  ForEach-Object { @('--state-csv', $_.FullName) }
.venv\Scripts\python.exe src\run.py submit --official-root C:\path\trafficflowbench-public --release-root $release --output-dir reports\maginet\submission @stateFiles
```

`--official-root`는 별도로 준비한 주최 측 저장소입니다. Queue 체크포인트를 지정하지 않으면 공식 persistence를 사용하고 Task 4는 기존 NNLS 설정을 사용합니다. 단일 패널 점검 모델의 CSV는 전체 제출로 사용할 수 없습니다.

2026-10-22에 Windows RTX 5060 Ti에서 `D12_I405_S`의 regime별 하루로 BF16 학습을 시작하고 6 epoch까지 재개해 N·ΔN 보조 손실이 켜지는 것을 확인했습니다. validation 93,718행·private 107,416행 CSV의 템플릿 키·중복·값도 검사했습니다. 이는 한 패널의 작은 실행 점검이며 성능 비교 결과가 아닙니다. 합성 검사에는 노트북 자체 검사, 같은 CPU 조건에서 경로 이동 전후 재개 가중치의 일치, fit 캐시 없는 평가·추론, Colab 형식 가져오기와 제출 키 검사를 포함합니다. 2026-10-23에는 기존 WSL 전체 학습 결과를 별도로 분석했습니다. 실제 Colab 저장 파일은 제공되지 않아 가져오기는 합성 체크포인트로 검사했습니다. 새 분리 실험의 전체 데이터 성능·공식 점수·최종 공식 병합은 아직 검증하지 않았습니다.

2026-10-23의 추가 검사에서는 새 실험 3종을 WSL의 44·352개 관측소 두 패널에서 BF16으로 각각 6 epoch 실행했습니다. N·ΔN 손실 구간 진입, adaptive 배치 8·4, 실험 사이 18개 일별 캐시 재사용을 확인했습니다. 이는 regime별 하루의 실행 검사입니다. 기존 50 epoch 모델의 재개 서명과 30개 패널·날짜 표본의 예측 점수도 그대로였으며, 기존 체크포인트·혼합 선택 파일의 해시를 유지했습니다. 합성 검사는 44개 중 42개 통과·Windows에서 실행할 수 없는 2개 건너뛰기였습니다.

이후 원본 변경 감지·이동한 체크포인트의 혼합 파일 경로·추론 배치 적용에 대한 회귀 검사를 추가해 46개 중 44개 통과·2개 건너뛰기를 확인했습니다. WSL의 전체 학습용 2,450일 캐시는 원본 대조 후 출처 기록을 추가했고, 기존 NPZ를 다시 쓰지 않고 재개 서명을 유지했습니다. `full/` 밖으로 옮긴 체크포인트에서 혼합 선택 파일을 읽고 실제 배치 8로 validation 93,718행을 추론하는 것도 확인했습니다. 기존 50 epoch 모델·혼합 선택·학습 기록 파일은 보존했습니다.

2026-10-23 추가 검증에서는 CUDA FP32 학습의 난수 상태를 AMP 사용 여부와 관계없이 저장·복원하도록 수정했습니다. 합성 데이터에서 연속 학습과 중단 후 재개의 가중치가 허용 오차 안에서 일치했으며, WSL에서 회귀 검사 47개가 모두 통과했습니다. `temporal_direct_dnzero`·`temporal_direct_learned_graph`를 실제 44·352개 관측소 두 패널에서 고정 배치 2·BF16으로 6 epoch 학습한 뒤 7 epoch까지 재개했습니다. ΔN 0 모델의 validation 93,718행·private 107,416행 예측도 템플릿 키·유한값·음수 여부 검사를 통과했습니다. 기존 전체 모델·혼합 선택·학습 기록의 해시와 공유 NPZ 2,757개의 수정 시각을 유지했습니다. 이는 소규모 실행 검증이며 새 두 실험의 전체 학습 성능을 검증한 결과는 아닙니다.
