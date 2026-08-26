# Snake AI (DQN)

Pygame으로 만든 스네이크 게임을, PyTorch 기반 DQN(Deep Q-Network) 에이전트가
스스로 플레이하며 점점 잘하도록 학습하는 프로젝트입니다.

## 프로젝트 구조

```
snake-ai/
├── game.py           # 20x20 grid 스네이크 게임 (Gym 스타일 reset/step 인터페이스)
├── model.py           # Q-Network(신경망) + QTrainer(학습 로직, 타겟 네트워크 포함)
├── agent.py           # DQN 에이전트 (Experience Replay, epsilon-greedy)
├── train.py           # 학습 루프 + 실시간 학습 곡선(matplotlib) + 체크포인트 저장
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

- 매 에피소드(한 판)가 끝날 때마다 `Game N | Score | Record | Epsilon` 형태로 콘솔에 출력됩니다.
- matplotlib 창에 에피소드별 점수와 누적 평균 점수 그래프가 실시간으로 갱신됩니다.
- 최고 점수를 갱신할 때마다 `model/best.pth`에 체크포인트가 저장됩니다.
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

이 저장소에는 미리 학습된 `model/best.pth`가 포함되어 있어서, `train.py`를 직접
돌리지 않아도 `python play.py`만으로 바로 플레이를 볼 수 있습니다. 총 2,500 에피소드
학습 후 탐험 없이(순수 정책) 20판을 평가했을 때 평균 26.65점(최저 11 / 최고 47,
0점으로 죽은 판 없음)을 기록한 체크포인트입니다. 직접 처음부터 학습시키고 싶다면
`train.py`를 실행하면 이 파일이 새 신기록으로 덮어써집니다.

## 설계 개요

### State (15차원 벡터, `STATE_SIZE`)

| 인덱스 | 의미 |
|---|---|
| 0~2 | 직진 방향으로 1~3칸 앞의 충돌(위험) 여부 (`FORWARD_LOOKAHEAD=3`) |
| 3~4 | 우회전 방향으로 1~2칸 앞의 위험 여부 (`SIDE_LOOKAHEAD=2`) |
| 5~6 | 좌회전 방향으로 1~2칸 앞의 위험 여부 (`SIDE_LOOKAHEAD=2`) |
| 7~10 | 현재 이동 방향 one-hot (상/하/좌/우) |
| 11~14 | 먹이의 머리 기준 상대 방향 (상/하/좌/우, boolean) |

원래는 직진/좌/우 모두 딱 1칸 앞의 위험만 봤는데, 그러면 코앞에 닥쳐야만 위험을
인지할 수 있어 미리 대비하기 어려웠습니다. 그래서 직진 방향은 더 멀리(3칸), 좌우는
그보다 조금 덜 멀리(2칸)까지 내다보도록 확장해서, 한발 앞서 판단할 수 있는 정보를
줍니다. `FORWARD_LOOKAHEAD` / `SIDE_LOOKAHEAD`는 `game.py` 상단에서 조절할 수 있고,
이 값을 바꾸면 `STATE_SIZE`(따라서 신경망 입력 크기)도 자동으로 함께 바뀝니다.

행동(action)은 절대 방향이 아니라 **머리 기준 상대 방향**([직진, 우회전, 좌회전])의
3가지 one-hot으로 정의합니다. 사람 플레이(방향키, 절대 방향)는
`SnakeGameAI.direction_to_action()`이 내부적으로 상대 action으로 변환해
동일한 `step(action)`을 사용하도록 통일되어 있습니다.

### Reward

- 먹이 섭취: **+10**
- 충돌/게임오버(타임아웃 포함): **-20 + 몸길이 보너스(최대 +15)** — 죽는 것에 대한 기본 페널티는 크게 주되,
  지금까지 먹은 먹이 개수(= 시작 길이를 초과한 몸길이)만큼 **+1씩** 더해줘서 오래 살아남아 몸을
  키운 뒤 죽는 것이 초반에 바로 죽는 것보다 덜 아프도록 합니다. 단, 이 보너스는
  `MAX_LENGTH_BONUS`로 상한이 걸려 있어서 **사망 시 reward는 아무리 몸이 길어도 항상 최소 -5는
  남습니다** — 즉 죽는 것 자체가 이득이 되는 경우는 없습니다.
- 그 외 이동(생존): **+0.1** — 죽지 않고 살아있는 것 자체에 소량의 보상을 줘서 무작정 몸을 던지는 행동을 억제
  (단, 먹지 않고 계속 맴돌기만 하면 `frame_iteration > 150 * len(snake)` 타임아웃에 걸려 결국 -20을 받으므로,
  이 소량 보상을 노린 맴돌기 꼼수는 통하지 않음)
- (옵션) `use_distance_reward=True`로 켜면, 먹이를 먹지 않은 스텝에 한해
  먹이와의 맨해튼 거리가 가까워지면 +1, 멀어지면 -1의 소량 보상을 추가로 줍니다.
- **완전 자기 감금 페널티**: 죽기 직전 머리 기준 상하좌우 네 방향이 전부(벽이 아니라)
  자기 몸통으로 막혀 있었다면(`REWARD_SELF_TRAP_PENALTY = -30`), 그 판에 한해 추가로
  크게 감점합니다. 이 상황은 어떤 행동을 했어도 피할 수 없었던 죽음이므로,
  스스로를 그 상태로 몰아넣은 것 자체가 명백한 실수였다고 보고 위의 -5 하한과
  무관하게 훨씬 낮은 reward를 줍니다.

먹이를 못 먹었을 때 게임을 강제 종료시키는 타임아웃 기준은 `TIMEOUT_STEPS_PER_SEGMENT`(기본 150)로,
`game.py` 상단에서 조절할 수 있습니다. 몸이 길어질수록 허용 스텝도 함께 늘어나므로,
사실상 "최근에 먹이를 못 먹은 시간"과 비슷하게 동작합니다.

### 에이전트 / 신경망

- 신경망: `state(15) → Linear(256) → ReLU → Linear(3)` (`model.py`의 `Linear_QNet`)
- Experience Replay Buffer: `deque(maxlen=100_000)`, 배치 크기 1,000
- epsilon-greedy 탐험: 게임 수(`n_games`)가 늘수록 `epsilon`이 1.0 → 0.02로 선형 감소
- 학습 안정화: 벨만 타겟 계산에 별도의 **타겟 네트워크**를 사용하고,
  일정 스텝(기본 100)마다 policy 네트워크의 가중치로 동기화(hard update)합니다.
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
