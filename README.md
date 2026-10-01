# 2026 IEEE Big Data — TrafficFlowBench 대회 안내

**고속도로의 결측 교통 상태를 복원하고, 정체를 예측하며, 교통량의 물리적 일관성과 통행 수요까지 함께 평가하는 대회입니다.** 속도 예측 하나로 순위를 정하지 않고, 네 과제의 점수를 합산합니다.

> 확인일: **2026-09-29, 한국시간(KST)**. Kaggle 현행 화면과 주최 측 공개 저장소를 조사한 참가 안내입니다. 변경 가능한 일정·규정은 제출 전에 다시 확인해 주세요. 이 폴더에는 대회 설명만 정리했으며 데이터와 모델 코드는 포함하지 않았습니다.

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

아래는 **실행 방법 안내**이며, 이 문서를 작성하면서 설치·학습·대회 제출을 수행한 것은 아닙니다. 공식 의존성은 pandas·NumPy·PyArrow·SciPy 등입니다. [실행 가이드](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/RUN_LOCAL.md), [requirements.txt](https://github.com/jacky850/trafficflowbench-public/blob/main/requirements.txt)

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

조사 시 확인한 GitHub 커밋: [`c88cddf533bbf0afa4ff1fc6c031d760a08e7a31`](https://github.com/jacky850/trafficflowbench-public/commit/c88cddf533bbf0afa4ff1fc6c031d760a08e7a31). 문서 검증 범위는 공식 페이지·공개 코드 대조와 README 형식 검사입니다. 대회 데이터 다운로드, 모델 실행, Kaggle 점수 재현은 수행하지 않았습니다.
