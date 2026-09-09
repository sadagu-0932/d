# Snake AI (DQN)

Pygame으로 만든 스네이크 게임을, PyTorch 기반 DQN(Deep Q-Network) 에이전트가
스스로 플레이하며 점점 잘하도록 학습하는 프로젝트입니다.

## 프로젝트 구조

```
snake-ai/
├── game.py           # 20x20 grid 스네이크 게임 (Gym 스타일 reset/step 인터페이스)
├── model.py           # Q-Network(신경망) + QTrainer(학습 로직, Double DQN + 타겟 네트워크)
├── agent.py           # DQN 에이전트 (Experience Replay, epsilon-greedy, 에피소드별 loss 집계)
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

- 매 에피소드(한 판)가 끝날 때마다 `Game N | Score | Record | Epsilon | Avg Loss` 형태로
  콘솔에 출력됩니다.
- matplotlib 창에 에피소드별 점수와 누적 평균 점수 그래프가 실시간으로 갱신됩니다.
- 최고 점수를 갱신할 때마다 `model/best.pth`에 체크포인트가 저장됩니다.
- 매 에피소드마다 `episode, score, mean_score, avg_loss` 네 컬럼을 `training_log.csv`
  (실행할 때마다 새로 덮어씀)에 기록합니다. `avg_loss`는 그 에피소드 동안의 모든 학습
  스텝(`QTrainer.train_step`) loss의 평균으로, 점수가 주기적으로 진동하는 구간과 loss가
  튀는 구간이 겹치는지 나중에 분석하는 데 씁니다.
- 학습을 더 빠르게 돌리고 싶다면 `train.py` 상단의 `RENDER_WHILE_TRAINING = False`로
  바꿔 게임 화면 렌더링을 끄세요 (matplotlib 학습 곡선은 계속 표시됩니다).
- 대략 수십~수백 에피소드가 지나면 평균 점수가 눈에 띄게 오르는 것을 확인할 수 있습니다.
  (게임을 종료하려면 콘솔에서 Ctrl+C)

### 3. 학습된 모델로 플레이 보기

```bash
python play.py                       # model/best.pth를 FPS 20으로 재생
python play.py --fps 40              # 더 빠르게 재생
python play.py --episodes 5          # 5판만 플레이하고 종료
python play.py --model model/best.pth
```

⚠️ **`model/best.pth`는 현재 신경망 구조와 호환되지 않습니다.** 이 체크포인트는
state가 15차원이던 시절(flood-fill 기반 reachability feature 추가 이전)에 학습된
것이라, 지금의 21차원 입력 신경망과 shape이 맞지 않습니다. `python play.py`로 이
파일을 불러오려 하면 (원인을 명확히 알려주는) `RuntimeError`가 발생합니다 —
`train.py`를 실행해서 처음부터 새로 학습시켜 주세요. 새로 학습된 체크포인트가
`model/best.pth`를 덮어씁니다.

## 설계 개요

### State (21차원 벡터, `STATE_SIZE`)

| 인덱스 | 의미 |
|---|---|
| 0~2 | 직진 방향으로 1~3칸 앞의 충돌(위험) 여부 (`FORWARD_LOOKAHEAD=3`) |
| 3~4 | 우회전 방향으로 1~2칸 앞의 위험 여부 (`SIDE_LOOKAHEAD=2`) |
| 5~6 | 좌회전 방향으로 1~2칸 앞의 위험 여부 (`SIDE_LOOKAHEAD=2`) |
| 7~10 | 현재 이동 방향 one-hot (상/하/좌/우) |
| 11~14 | 먹이의 머리 기준 상대 방향 (상/하/좌/우, boolean) |
| 15~17 | 직진/우회전/좌회전 각 후보 방향의 정규화된 flood-fill 도달 가능 면적 (0~1) |
| 18~20 | 직진/우회전/좌회전 각 후보 방향에서 꼬리 칸에 도달 가능한지 여부 (boolean) |

원래는 직진/좌/우 모두 딱 1칸 앞의 위험만 봤는데, 그러면 코앞에 닥쳐야만 위험을
인지할 수 있어 미리 대비하기 어려웠습니다. 그래서 직진 방향은 더 멀리(3칸), 좌우는
그보다 조금 덜 멀리(2칸)까지 내다보도록 확장해서, 한발 앞서 판단할 수 있는 정보를
줍니다. `FORWARD_LOOKAHEAD` / `SIDE_LOOKAHEAD`는 `game.py` 상단에서 조절할 수 있고,
이 값을 바꾸면 `STATE_SIZE`(따라서 신경망 입력 크기)도 자동으로 함께 바뀝니다.

lookahead는 "바로 근처가 막혔는지"만 보는 국소적인 정보라, 당장은 안 막혀 보여도
사실 좁은 구석에 갇히는 길일 수 있습니다. 그래서 직진/우회전/좌회전 각 후보 행동에
대해 BFS(flood-fill, `_flood_fill_from`)로 그 방향이 실제로 얼마나 넓은 공간으로
이어지는지(`reachable_area`)와, 그 안에서 최소한의 순환/탈출 경로가 있는지를
어림하는 `can_reach_tail`(꼬리 칸에 도달 가능한지)까지 함께 봅니다. BFS의 장애물
판정에서 몸통은 막힌 칸으로 취급하되, 그 후보 행동이 먹이를 먹는 행동이 아니라면
꼬리는 다음 스텝에 비워지므로 장애물에서 제외합니다(먹이를 먹는 행동이라면 꼬리도
그대로 남으므로 포함).

행동(action)은 절대 방향이 아니라 **머리 기준 상대 방향**([직진, 우회전, 좌회전])의
3가지 one-hot으로 정의합니다. 사람 플레이(방향키, 절대 방향)는
`SnakeGameAI.direction_to_action()`이 내부적으로 상대 action으로 변환해
동일한 `step(action)`을 사용하도록 통일되어 있습니다.

### Reward

두 가지 사건에만 반응합니다 — **먹이(사과)를 먹으면 점수를 얻고, 죽으면 점수를
잃습니다.** 그 위에 "몸이 길수록 더 유리하다"는 신호를 주는 길이 보너스 하나만
얹었습니다. 그 외 조건(회전 페널티, self-trap 전용 페널티, 빈칸 페널티, 사망 시
별도 보너스 등)은 없습니다.

| 상수 | 값 | 적용 시점 |
|---|---|---|
| `REWARD_FOOD` | `+10` | 먹이를 먹은 스텝 |
| `REWARD_DEATH` | `-10` | 사망한 스텝 (벽/몸통 충돌, 타임아웃 불문 — **항상 이 고정값**, 길이/score와 무관) |
| `LENGTH_BONUS_COEF` | `0.01` | **사망이 아닌 모든 생존 스텝마다**: `+LENGTH_BONUS_COEF * len(snake)` (먹이 섭취 스텝 포함) |

먹이를 못 먹었을 때 게임을 강제 종료시키는 타임아웃 기준은 `TIMEOUT_STEPS_PER_SEGMENT`(기본 150)로,
`game.py` 상단에서 조절할 수 있습니다. 몸이 길어질수록 허용 스텝도 함께 늘어나므로,
사실상 "최근에 먹이를 못 먹은 시간"과 비슷하게 동작합니다. 타임아웃으로 죽든 충돌로 죽든
reward는 동일하게 `REWARD_DEATH`입니다.

### 에이전트 / 신경망

- 신경망: `state(21) → Linear(256) → ReLU → Linear(3)` (`model.py`의 `Linear_QNet`)
- Experience Replay Buffer: `deque(maxlen=100_000)`, 배치 크기 1,000
- epsilon-greedy 탐험: 게임 수(`n_games`)가 늘수록 `epsilon`이 300게임에 걸쳐 1.0 → 0.05로
  선형 감소 (`agent.py`의 `EPS_DECAY_GAMES`/`EPS_END`). 예전 값(150게임, 0.02)에서는 평균
  점수가 오르기 시작하는 시점(100~200게임)에 탐험이 사실상 끝나버려서, 그 이후 수백~수천
  게임 동안 점수가 정체(plateau)되는 현상이 실측 학습 로그에서 확인됨 — 탐험을 더 오래
  유지해 정체 시점을 늦추도록 조정
- 학습 안정화: 벨만 타겟 계산에 별도의 **타겟 네트워크**를 사용하고,
  일정 스텝(기본 100)마다 policy 네트워크의 가중치로 동기화(hard update)합니다.
- **Double DQN**: 다음 상태(s')에서의 행동 선택은 online 네트워크(`argmax_a' Q_online(s', a')`)로,
  그 행동의 가치 평가는 target 네트워크(`Q_target(s', a*)`)로 분리해서 계산합니다
  (`model.py`의 `QTrainer.train_step`). vanilla DQN처럼 target 네트워크 하나로 선택과
  평가를 모두 처리하면 Q-value가 과대추정(overestimation)되는 경향이 있는데, Double DQN은
  이를 완화해 학습을 더 안정적으로 수렴시킵니다.
- 매 스텝 즉시 학습(`train_short_memory`) + 매 에피소드 종료 시 리플레이 버퍼에서
  샘플링한 배치로 추가 학습(`train_long_memory`)을 병행합니다.
- **챔피언 세트 기반 추가 학습**: `SET_SIZE`(기본 100)게임을 '1세트'로 묶습니다. 세트가
  끝날 때마다 그 세트의 평균 점수를, 지금까지 채택하고 있던 **챔피언 세트**의 평균 점수와
  비교해서 — 더 높으면 챔피언을 이번 세트로 교체하고, 더 낮으면 기존 챔피언을 그대로
  유지합니다. 매 에피소드가 끝나면 `train_long_memory()` 뒤에, 이 챔피언 세트 안에서
  가장 점수가 높았던 한 판의 transition만 따로 뽑아 한 번 더 학습합니다
  (`train_from_best_episode`). 정책을 통째로 그 판으로 덮어쓰는 게 아니라, 평소와 같은
  방식(경사하강)으로 지금까지 가장 안정적으로 잘했던 구간의 플레이 쪽을 조금씩 더
  강화하는 형태입니다. `train.py` 콘솔에 세트가 끝날 때마다 평균/최고 점수와 챔피언
  교체 여부가 출력됩니다.

## 진행 순서 (개발 시 권장)

1. `python game.py`로 방향키 기본 플레이 확인
2. state/reward 설계 확인 (`SnakeGameAI._get_state`, `SnakeGameAI.step`)
3. `agent.py`의 DQN 에이전트로 `train.py` 실행 → 학습 곡선으로 성능 확인
4. `python play.py`로 학습된 모델의 실제 플레이 성능 확인
