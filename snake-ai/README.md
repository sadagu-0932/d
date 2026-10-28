# Snake AI (유전 알고리즘)

Pygame으로 만든 스네이크 게임을, 신경망 개체들의 집단(population)을 유전
알고리즘(genetic algorithm)으로 진화시켜 점점 잘 플레이하게 만드는 프로젝트입니다.

## 프로젝트 구조

```
snake-ai/
├── game.py           # 20x20 grid 스네이크 게임 (Gym 스타일 reset/step 인터페이스)
├── model.py           # 신경망 구조만 정의 (Linear_QNet) — 가중치 "학습"은 agent.py가 담당
├── agent.py           # 유전 알고리즘: Population(개체 집단), crossover/mutation
├── train.py           # 학습 루프 + 실시간 학습 곡선(matplotlib) + 체크포인트/CSV 로그 저장
├── play.py            # 학습된 모델(best.pth)로 실제 플레이 화면 보여주기
├── requirements.txt   # 필요 패키지 목록
├── model/             # 학습 체크포인트 저장 위치 (best.pth)
└── README.md
```

## 설치

```bash
cd snake-ai
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 사용법

### 1. 사람이 직접 플레이 (기본 게임 동작 확인)

```bash
python game.py
```

방향키(↑ ↓ ← →)로 조작합니다. 게임오버 시 자동으로 새 게임이 시작됩니다.

### 2. AI 학습시키기

```bash
python train.py
```

- 매 세대(generation)마다 `POPULATION_SIZE`(기본 50)개체 각각을
  `GAMES_PER_INDIVIDUAL`(기본 3)판씩 플레이시킵니다 — 매 판마다
  `SnakeGameAI.reset(randomize=True)`로 뱀의 길이와 장애물 배치를 무작위로 바꿔서
  "한 맵만이 아니라 다양한 필드에서" 평가합니다. 각 판의 누적 reward 평균이 그
  개체의 적합도(fitness)가 됩니다(점수 자체는 Record/체크포인트 저장 기준으로
  따로 씁니다).
- 적합도 상위 `NUM_PARENTS`(기본 4)개체를 부모로 선정합니다. 이 부모들은 그대로
  다음 세대에 보존되고(엘리티즘), 나머지 자리는 부모 중 무작위로 2명을 골라
  가중치를 섞고(crossover) 살짝 변형(mutation)해서 채웁니다. **경사하강/역전파는
  전혀 쓰지 않습니다.**
- 매 세대마다 `Gen N | Best | Avg | Worst | Record` 형태로 콘솔에 출력됩니다.
- matplotlib 창에 세대별 최고/평균/최저 적합도 그래프가 실시간으로 갱신됩니다.
- 지금까지 한 판에서 나온 최고 점수(record)를 세우면 그 개체의 가중치를
  `model/best.pth`로 저장합니다.
- 매 세대마다 `generation, best_score, avg_score, worst_score, record` 다섯 컬럼을
  `training_log.csv`(실행할 때마다 새로 덮어씀)에 기록합니다.
- 학습을 더 빠르게 돌리고 싶다면 `train.py` 상단의 `RENDER_WHILE_TRAINING = False`
  (기본값)를 유지하세요 — 한 세대에 `POPULATION_SIZE * GAMES_PER_INDIVIDUAL`판을
  전부 플레이해야 해서, 화면 렌더링을 켜면 훨씬 느려집니다.
- 대략 수십 세대가 지나면 최고/평균 적합도가 눈에 띄게 오르는 것을 확인할 수
  있습니다. (게임을 종료하려면 콘솔에서 Ctrl+C)

### 3. 학습된 모델로 플레이 보기

```bash
python play.py                       # model/best.pth를 FPS 20으로 재생
python play.py --fps 40              # 더 빠르게 재생
python play.py --episodes 5          # 5판만 플레이하고 종료
python play.py --model model/best.pth
```

⚠️ **`model/best.pth`는 현재 신경망 구조와 호환되지 않습니다.** state/action 설계가
여러 차례 바뀌어서(최종적으로 14차원 입력, 2차원 출력 — 아래 참고) 지금의 신경망과
shape이 맞지 않습니다. `python play.py`로 이 파일을 불러오려 하면 (원인을 명확히
알려주는) `RuntimeError`가 발생합니다 — `train.py`를 실행해서 처음부터 새로
학습시켜 주세요. 새로 학습된 체크포인트가 `model/best.pth`를 덮어씁니다.

## 설계 개요

### 안전한 순환 경로(Hamiltonian cycle) + 지름길 결정 — "같은 자리를 맴도는" 문제 해법

반응형(reactive) state(머리 바로 옆 1칸 위험 + 먹이 방향)만 보고 신경망이 매 스텝
3방향 이동을 전부 직접 고르면, 특정 상황이 반복될 때 똑같은 행동을 또 선택하게
되어 **같은 자리를 계속 맴도는 제자리 루프**에 쉽게 갇힙니다. 이를 근본적으로
막기 위해, 보드의 모든 칸을 정확히 한 번씩 방문하고 처음 자리로 되돌아오는 고정
순환 경로(`game.py`의 `HAMILTONIAN_CYCLE`)를 미리 계산해 둡니다. 이 경로를 그대로
따라가기만 하면(몸이 몇 칸이든) **자기 몸통과 절대 부딪히지 않습니다** — 몸은
항상 이 경로상 "가장 최근 지나온 N칸"이라서, 다음 칸은 항상 그보다 훨씬 앞쪽에
있기 때문입니다.

신경망의 역할은 이제 3방향을 직접 고르는 게 아니라, 매 스텝 **"이 안전한 경로를
그대로 따라갈지, 아니면 경로를 벗어나 먹이로 직행하는 지름길을 탈지"** 2-way
결정뿐입니다(`agent.py`의 `greedy_action`). 그 결정을 실제 3-way 이동 action으로
바꾸는 건 `resolve_action()`이 담당하며, `SnakeGameAI.cycle_action()`(순환 경로를
따르되, 몸이 아직 경로와 안 맞는 에피소드 초반 등 당장 위험하면 1칸 앞만 봐도
안전한 쪽으로 즉시 피함)과 `SnakeGameAI.shortcut_action()`(1칸 앞이 위험하지 않은
후보 중 먹이까지 맨해튼 거리가 가장 줄어드는 쪽을 고름 — 없으면 지름길 포기)을
그대로 사용합니다. 지름길은 1칸 앞만 보고 안전해 보여도, 나중에(특히 몸이 길어졌을
때) 꼬리에 막혀 죽을 수 있는 함정까지는 걸러내지 못합니다 — 그래도 먹이 보상이
사망 페널티보다 훨씬 크므로(아래 Reward 참고), 적절히 지름길을 타는 쪽이
fitness에 유리합니다.

장애물(학습 시 환경 다양화로 깔리는)이 순환 경로를 가끔 막을 수도 있는데, 그
경우도 `cycle_action()`이 1칸 앞만 봐도 안전한 다른 방향으로 즉시 피하도록
처리되어 있습니다(다만 그 즉흥적인 회피까지 완벽하게 안전함을 보장하진 않습니다).

### State (14차원 벡터, `STATE_SIZE`)

| 인덱스 | 의미 |
|---|---|
| 0 | 직진 방향으로 머리 바로 옆 1칸(주변 블럭)의 위험(충돌) 여부 |
| 1 | 우회전 방향으로 머리 바로 옆 1칸의 위험 여부 |
| 2 | 좌회전 방향으로 머리 바로 옆 1칸의 위험 여부 |
| 3~6 | 현재 이동 방향 one-hot (상/하/좌/우) |
| 7~10 | 먹이의 머리 기준 상대 방향 (상/하/좌/우, boolean) |
| 11 | 지금 안전한 지름길(`shortcut_action`)이 실제로 있는지 |
| 12 | 순환 경로를 그대로 따라갈 때 먹이까지 남은 거리 (전체 경로 길이로 정규화, 0~1) |
| 13 | 몸이 보드 전체 칸 중 몇 %를 채웠는지 (`len(snake) / 전체 칸 수`, 0~1) |

[0:3]는 "머리 바로 옆 블럭이 막혔는지"만 보는 가장 단순한 위험 감지입니다(한동안
3·2칸 lookahead와 flood-fill 기반 reachability까지 늘렸다가 "모든 규칙을
초기화하고 주변 블럭만 감지하자"는 요청으로 되돌린 것). [11:14]는 바로 위 "안전한
순환 경로 + 지름길 결정"에 필요한 신호입니다.

행동(action)은 여전히 절대 방향이 아니라 **머리 기준 상대 방향**([직진, 우회전,
좌회전])의 3가지 one-hot으로 정의되고, `SnakeGameAI.step(action)`도 이전과 동일한
인터페이스입니다 — 바뀐 건 "그 action을 누가/어떻게 정하느냐"뿐입니다. 사람
플레이(방향키, 절대 방향)는 여전히 `SnakeGameAI.direction_to_action()`으로 변환해
같은 `step(action)`을 쓰고, AI는 신경망의 2-way 결정을 `agent.py`의
`resolve_action()`이 3-way action으로 바꿔서 넘깁니다.

### Reward

딱 두 가지 사건에만 반응합니다 — **먹이(사과)를 먹으면 점수를 얻고, 죽으면
점수를 잃습니다.** 그 외 조건(회전 페널티, self-trap 전용 페널티, 빈칸 페널티,
사망 시 별도 보너스, 생존 시간에 비례한 가산점 등)은 전혀 없습니다. 예전에는
"몸이 길수록 더 유리하다"는 신호로 생존 스텝마다 소량의 길이 보너스
(`LENGTH_BONUS_COEF`)를 얹었지만, "생존 시간 보너스를 삭제하자"는 요청에 따라
완전히 제거했습니다.

| 상수 | 값 | 적용 시점 |
|---|---|---|
| `REWARD_FOOD` | `+30` | 먹이를 먹은 스텝 ("사과를 먹을 때의 보상을 늘리자" 요청에 따라 `10` -> `30`으로 상향) |
| `REWARD_DEATH` | `-10` | 사망한 스텝 (벽/몸통/장애물 충돌, 타임아웃 불문 — **항상 이 고정값**, 길이/score와 무관) |

유전 알고리즘의 적합도(fitness)는 한 판이 끝났을 때의 **판별 누적 reward의
평균**입니다(`Population.evaluate`) — 이렇게 해야 위 `REWARD_FOOD`/`REWARD_DEATH`
값을 조정했을 때 실제로 어느 개체가 부모로 선택되는지에 반영됩니다. (예전엔 raw
score의 평균을 그대로 fitness로 썼는데, 그러면 reward 상수를 바꿔도 선택 압력이
전혀 달라지지 않는 문제가 있었습니다.) 다만 화면에 표시되는 **Record**와
`model/best.pth` 체크포인트 저장 기준은 reward가 아니라, reward 가중치와 무관한
실제 **점수(score, 먹은 먹이 수)**를 그대로 씁니다.

먹이를 못 먹었을 때 게임을 강제 종료시키는 타임아웃 기준은 `TIMEOUT_STEPS_PER_SEGMENT`(기본 150)로,
`game.py` 상단에서 조절할 수 있습니다. 몸이 길어질수록 허용 스텝도 함께 늘어나므로,
사실상 "최근에 먹이를 못 먹은 시간"과 비슷하게 동작합니다. 타임아웃으로 죽든 충돌로 죽든
reward는 동일하게 `REWARD_DEATH`입니다.

### 신경망 구조

`state(14) → Linear(256) → ReLU → Linear(2)` (`model.py`의 `Linear_QNet`).
출력 2개는 위 "안전한 순환 경로 + 지름길 결정"의 **[순환 경로 유지, 지름길]**
2-way 결정이고, 실제 3-way 이동은 그 결정을 바탕으로 `game.py`가 결정론적으로
계산합니다. 이 구조(레이어 수·뉴런 수)는 우리가 미리 고정해 둔 값입니다 — **AI가
스스로 뉴런 개수나 레이어 구조를 바꾸지는 않습니다.** 신경망의 토폴로지 자체를
탐색하는 것(NEAT류 알고리즘)은 훨씬 큰 별도 작업이라 이번 범위에는 포함하지
않았습니다. 유전 알고리즘이 "학습"하는 대상은 이 고정된 구조 안의 **가중치**뿐입니다.

### 학습 방식 — 유전 알고리즘 (초기화됨: 경사하강 대신)

예전에는 Double DQN + Experience Replay + epsilon-greedy로 가중치를 역전파
학습시켰는데, "세트 내 가장 우수한 모델을 토대로 4개를 정해서 부모로 삼고, 섞고
살짝씩 바꿔가며 학습하자"는 요청에 따라 경사하강을 전혀 쓰지 않는 유전
알고리즘으로 교체했습니다 (`agent.py`의 `Population`).

- **Population**: 신경망 `POPULATION_SIZE`(기본 50)개를 한 세대(generation)로
  묶어서 관리합니다.
- **다양한 필드에서 평가 (`GAMES_PER_INDIVIDUAL`, 기본 3)**: 각 개체를 매번
  다르게 무작위화된 맵(길이·장애물이 다른 "필드") 여러 판 플레이시켜, 판별
  누적 reward의 평균을 그 개체의 적합도(fitness)로 삼습니다(위 Reward 섹션
  참고 — Record/체크포인트 저장 기준은 reward가 아니라 실제 점수입니다). 한
  맵만 보면 운에 좌우되는 평가가 되는 문제를 줄이기 위함입니다.
- **Selection + Elitism**: 적합도 상위 `NUM_PARENTS`(기본 4)개체를 부모로
  선정합니다. 이 개체들은 그대로 다음 세대에 보존되어(엘리티즘) 지금까지 찾은
  가장 좋은 개체를 잃지 않습니다.
- **Crossover + Mutation**: 나머지 자리는 부모 중 무작위로 2명을 골라, 가중치
  각 성분마다 50% 확률로 둘 중 하나를 물려받도록 섞고(`crossover`), 그 중
  `MUTATION_RATE`(기본 10%)의 가중치에 평균 0·표준편차 `MUTATION_STRENGTH`
  (기본 0.1)인 가우시안 노이즈를 더해(`mutate`) 살짝 변형해서 채웁니다.
- **기록 저장**: 어느 개체든 한 판에서 역대 최고 점수(`record`)를 세우면 그
  가중치를 `model/best.pth`로 저장합니다(`Population.save_best_if_new_record`).

## 진행 순서 (개발 시 권장)

1. `python game.py`로 방향키 기본 플레이 확인
2. state/reward 설계 확인 (`SnakeGameAI._get_state`, `SnakeGameAI.step`)
3. `agent.py`의 `Population`으로 `train.py` 실행 → 세대별 적합도 곡선으로 성능 확인
4. `python play.py`로 학습된 모델의 실제 플레이 성능 확인
