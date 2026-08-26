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

## 설계 개요

### State (11차원 벡터)

| 인덱스 | 의미 |
|---|---|
| 0 | 직진 시 충돌(위험) 여부 |
| 1 | 우회전 시 충돌(위험) 여부 |
| 2 | 좌회전 시 충돌(위험) 여부 |
| 3~6 | 현재 이동 방향 one-hot (상/하/좌/우) |
| 7~10 | 먹이의 머리 기준 상대 방향 (상/하/좌/우, boolean) |

행동(action)은 절대 방향이 아니라 **머리 기준 상대 방향**([직진, 우회전, 좌회전])의
3가지 one-hot으로 정의합니다. 사람 플레이(방향키, 절대 방향)는
`SnakeGameAI.direction_to_action()`이 내부적으로 상대 action으로 변환해
동일한 `step(action)`을 사용하도록 통일되어 있습니다.

### Reward

- 먹이 섭취: **+10**
- 충돌/게임오버(타임아웃 포함): **-20 + 몸길이 보너스** — 죽는 것에 대한 기본 페널티는 크게 주되,
  지금까지 먹은 먹이 개수(= 시작 길이를 초과한 몸길이)만큼 **+1씩** 더해줘서 오래 살아남아 몸을
  키운 뒤 죽는 것이 초반에 바로 죽는 것보다 덜 아프도록(경우에 따라 거의 상쇄되도록) 합니다.
- 그 외 이동(생존): **+0.1** — 죽지 않고 살아있는 것 자체에 소량의 보상을 줘서 무작정 몸을 던지는 행동을 억제
  (단, 먹지 않고 계속 맴돌기만 하면 `frame_iteration > 150 * len(snake)` 타임아웃에 걸려 결국 -20을 받으므로,
  이 소량 보상을 노린 맴돌기 꼼수는 통하지 않음)
- (옵션) `use_distance_reward=True`로 켜면, 먹이를 먹지 않은 스텝에 한해
  먹이와의 맨해튼 거리가 가까워지면 +1, 멀어지면 -1의 소량 보상을 추가로 줍니다.

먹이를 못 먹었을 때 게임을 강제 종료시키는 타임아웃 기준은 `TIMEOUT_STEPS_PER_SEGMENT`(기본 150)로,
`game.py` 상단에서 조절할 수 있습니다. 몸이 길어질수록 허용 스텝도 함께 늘어나므로,
사실상 "최근에 먹이를 못 먹은 시간"과 비슷하게 동작합니다.

### 에이전트 / 신경망

- 신경망: `state(11) → Linear(256) → ReLU → Linear(3)` (`model.py`의 `Linear_QNet`)
- Experience Replay Buffer: `deque(maxlen=100_000)`, 배치 크기 1,000
- epsilon-greedy 탐험: 게임 수(`n_games`)가 늘수록 `epsilon`이 1.0 → 0.02로 선형 감소
- 학습 안정화: 벨만 타겟 계산에 별도의 **타겟 네트워크**를 사용하고,
  일정 스텝(기본 100)마다 policy 네트워크의 가중치로 동기화(hard update)합니다.
- 매 스텝 즉시 학습(`train_short_memory`) + 매 에피소드 종료 시 리플레이 버퍼에서
  샘플링한 배치로 추가 학습(`train_long_memory`)을 병행합니다.

## 진행 순서 (개발 시 권장)

1. `python game.py`로 방향키 기본 플레이 확인
2. state/reward 설계 확인 (`SnakeGameAI._get_state`, `SnakeGameAI.step`)
3. `agent.py`의 DQN 에이전트로 `train.py` 실행 → 학습 곡선으로 성능 확인
4. `python play.py`로 학습된 모델의 실제 플레이 성능 확인
