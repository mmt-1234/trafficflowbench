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

정체가 새로 생기는 `queue_onset`과 이미 진행 중인 `queue_ongoing`을 나누어 평가합니다. 각 대상 패널·split마다 각각 5개 창이 있고, 모든 창의 미래에 실제 정체가 있습니다. [Task 2 명세](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SCORING_SPEC.md#task-2-queue-forecasting)

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

[task4_odme_colab.ipynb](notebooks/task4_odme_colab.ipynb)는 같은 기능의 독립형 Colab 노트북입니다. CPU 런타임에서 Drive를 연결하고 `RELEASE_ROOT`를 지정한 뒤 위에서부터 실행합니다. 함수는 모두 셀 안에 있으며, 후보 비교표·그래프와 validation·private의 Task 4 CSV를 저장합니다. CSV 생성 기본값은 기존 baseline이고, 후보 비교 결과는 예측 설정에 자동 적용하지 않습니다. 예측은 실행마다 `predictions/run-<id>/`에 저장하고, 양쪽 split·합본·설정이 모두 완성된 경우에만 `predictions/latest.json`을 갱신합니다. 최신 합본 경로는 이 파일의 `combined_path`입니다. 중단된 `.pending-<id>/` 폴더는 미완료 결과이며 이전 완료 파일은 유지됩니다. 생성 파일은 Task 4 전용이므로 최종 Kaggle 제출 전 다른 과제와 병합해야 합니다.

Task 4는 해당 split의 관측 링크만 희소 행렬로 읽고 다음 문제를 풉니다. 기본값 `p=0`, `λ=0.05`는 기존 정규화 NNLS와 같은 목적함수입니다. 데이터 다운로드나 신경망 학습은 수행하지 않습니다.

솔버는 Prior와 관측 교통량을 같은 배율로 정규화해 계산하고 결과를 원래 단위로 복원합니다. 목적함수와 해의 의미는 유지하며, 큰 수요값에서의 수치 오차를 줄입니다.

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

## 12. Colab: Task 1·2·3 통합 노트북

[task123_colab.ipynb](notebooks/task123_colab.ipynb)는 [Kaggle용 task123 노트북](notebooks/task123_kaggle.ipynb)의 모델·학습·평가·제출 병합을 유지한 Colab용 파일입니다. GPU 런타임에서 Drive를 연결하고, 첫 설정 셀의 `RELEASE_ROOT`를 기존 데이터 폴더로 지정해 주세요. 해당 폴더 또는 바로 아래 `kaggle_public/`를 인식합니다. 기본 `RUN_MODE="smoke"`로 한 패널을 점검한 뒤 전체 학습은 `RUN_MODE="train"`, `PANELS=None`으로 실행합니다.

캐시와 입력 연결은 임시 디스크의 `WORK_ROOT=/content/trafficflow_task123`에 두고, 모델·보고서·예측은 Drive의 `SAVE_ROOT=/content/drive/MyDrive/dataset/trafficflow_task123`에 바로 저장합니다. 최종 제출은 `SAVE_ROOT/submission.csv`입니다. 같은 결과를 재개할 때는 `RESUME=True`로 설정하고, 기존 Kaggle Output을 가져올 때는 Drive에 올린 `trafficflow_task123/` 폴더를 `RESTORE_ROOT`로 지정합니다. 복원은 임시 폴더에 완료한 뒤 확정하므로 중단된 복사본이 재시도를 막지 않습니다. 복원 후에는 `RESTORE_ROOT=None`으로 사용합니다. Queue 재개는 원본 파일 크기·수정 시각도 검사하므로 데이터 사본의 메타데이터를 유지해야 합니다. 저장된 BF16과 GPU의 호환성은 긴 학습 전에 확인하고, 평가·추론은 미지원 장치에서 FP32를 사용합니다.

기존 Task 4 결과가 있으면 `TASK4_FILES`에 validation·private를 모두 포함하는 CSV 경로를 지정해 주세요. 없으면 기존 NNLS baseline을 사용합니다. 기본 Mamba2 backend는 패키지 설치 후 GPU에서 순전파·역전파를 먼저 확인합니다.

Task 3은 기존 기본 손실 `N=0.05, dN=0.03`을 `current003`으로 보존하고, `baseline010=(0.05, 0.10)`, `dn020=(0.05, 0.20)`, `both020=(0.10, 0.20)`을 비교합니다. 새 학습 기본값 `PHYSICS_PRESET="dn020"`은 성능을 검증할 후보이며, 설정별 결과 폴더를 분리합니다. 공유 `AMSTenc` 파라미터의 State/N/dN 기울기 크기·방향, 가중 물리/State 비율, `dN` 정규화 분모·표본 수를 `gradient_diagnostics.csv`에 기록합니다. `GRADIENT_EVERY=0`이면 이 진단을 끕니다. 자동 가중치 조절과 손실 정규화 변경은 진단 후 별도 비교 대상으로 둡니다.

Task 2의 새 학습은 검증 구간에 관측 링크 기준 과거 관측률 70% 이상, 미래 정체 존재, persistence IoU 0.9 이하, 조건별 최소 간격 360분을 적용합니다. 비공개 정답을 사용하는 공식 평가와 달리 공개 관측 속도로 만든 대리 검증입니다. `CALIBRATE_QUEUE=True`이면 고정된 1월 검증 구간의 확률을 캐시하고 원래 임계값, 전부 0인 window의 최대 확률 한 곳만 1로 만드는 후보, 전역·조건별 임계값을 비교합니다. 선택값은 체크포인트 해시에 연결하며 `RUN_FEBRUARY=True`일 때 2월에 고정 적용합니다. [공식 구간 선정 설정](https://github.com/jacky850/trafficflowbench-public/blob/main/config/synthetic_release_v1.json), [공식 평가 명세](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SCORING_SPEC.md), 확인일: 2026-10-28.

기존 모델로 재학습 없이 비교하려면 `RUN_MODE="predict"`, `PHYSICS_PRESET=None`, 원래 `MAGINET_EXPERIMENT`·`QUEUE_TAG`·`QUEUE_BACKEND`를 지정합니다. 새 Queue 학습의 기본 tag는 `public-validation-v1`이며, 기존 `enhanced` 결과와 분리합니다. `CALIBRATE_QUEUE=False`이면 저장된 선택값을 사용하고 없으면 원래 임계값을 사용합니다. 선택에 필요한 1월 원본과 train masked 자료는 유지해 주세요.

Queue 결과 폴더의 `calibration-public-v1/`에는 정책 비교표, 고정 검증 목록·확률, window별·5~30분별 TP/FP/FN과 정체 개수, 1월 선택값과 2월 고정 비교를 저장합니다. validation/private 추론은 선택된 0/1 CSV와 함께 `-baseline.csv`, `-probabilities.csv`를 저장합니다. 확률 파일의 값 열은 `queue_probability`이며 제출 파일이 아닙니다. `QUEUE_REFERENCE_SUBMISSION`에 이전 전체 제출 CSV를 지정하면 원래 임계값 출력과의 차이를 기록합니다. 새 후보의 점수 개선 여부는 동일 조건의 실제 데이터 비교로 판단해야 합니다.
