"""
agent.py
--------
DQN 에이전트: 게임(game.py)이 반환한 state를 받아 행동을 선택하고,
Experience Replay Buffer를 이용해 신경망(model.py)을 학습시킵니다.
"""

import random
from collections import deque

import torch

from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000   # 리플레이 버퍼 최대 크기 (deque라서 꽉 차면 오래된 것부터 자동 삭제)
BATCH_SIZE = 1_000     # 장기 기억(long memory) 학습 시 한 번에 샘플링할 transition 개수
LR = 0.001             # 학습률(learning rate)

# epsilon-greedy 탐험 파라미터
# 학습 초반에는 무작위 행동(탐험) 비중을 높게, 게임 수가 늘수록 점점 줄여 나감(감쇠).
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY_GAMES = 150  # 이 정도 게임 수가 지나면 epsilon이 EPS_END 근처까지 감소


class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9  # 미래 보상 할인율 (벨만 방정식에 사용)
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience Replay Buffer

        # 지금까지 본 것 중 "가장 점수가 높았던 한 판(에피소드)"의 transition들만 따로 보관.
        # 매 에피소드마다 여기서 추가로 배치를 뽑아 학습시켜서, 잘한 플레이를 조금씩 더
        # 강화하는 방향으로 학습이 흘러가게 한다 (정책을 통째로 덮어쓰는 게 아니라
        # 평소처럼 작은 학습률의 경사하강 스텝을 몇 번 더 밟는 것뿐).
        self.best_episode = []
        self.best_episode_score = -1

        self.model = Linear_QNet(11, 256, 3)  # state(11) -> hidden(256) -> action(3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # ----------------------------------------------------------------
    def get_epsilon(self) -> float:
        """진행된 게임 수(n_games)에 따라 선형으로 감소하는 epsilon 값을 반환."""
        ratio = min(self.n_games / EPS_DECAY_GAMES, 1.0)
        return EPS_START + ratio * (EPS_END - EPS_START)

    def get_action(self, state, explore: bool = True):
        """
        epsilon-greedy 정책으로 행동을 선택.

        explore=True  : 확률 epsilon으로 무작위 행동(탐험), 그 외에는 Q값이 가장 큰 행동(활용)
        explore=False : 항상 신경망이 예측한 최선의 행동만 선택 (play.py에서 사용)
        """
        final_move = [0, 0, 0]

        if explore and random.random() < self.get_epsilon():
            move = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move

    # ----------------------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        """transition 하나를 리플레이 버퍼에 저장."""
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        """방금 겪은 transition 하나로 즉시(단기) 학습 -> 최신 경험을 빠르게 반영."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        """리플레이 버퍼에서 배치를 무작위로 샘플링해 학습 (경험 재사용 + 데이터 상관관계 완화)."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # ----------------------------------------------------------------
    # "가장 점수가 높았던 플레이" 기반 추가 학습
    # ----------------------------------------------------------------
    def update_best_episode(self, episode_transitions, score):
        """방금 끝난 에피소드가 지금까지의 최고 기록이면, 그 판의 transition들을 통째로 저장."""
        if score > self.best_episode_score:
            self.best_episode_score = score
            self.best_episode = list(episode_transitions)

    def train_from_best_episode(self):
        """
        저장해 둔 '최고 기록 에피소드'에서 배치를 뽑아 추가로 학습.

        train_long_memory()와 똑같은 방식(무작위 샘플 + 일반 배치 학습)이지만,
        표본을 전체 리플레이 버퍼가 아니라 최고 점수 에피소드 하나로 한정한다.
        매 에피소드마다 이 함수를 한 번 더 호출해주면, 신경망이 잘 풀렸던 판의
        상태->행동 대응을 조금씩 더 강하게 기억하게 된다.
        """
        if not self.best_episode:
            return

        sample_size = min(len(self.best_episode), BATCH_SIZE)
        mini_sample = random.sample(self.best_episode, sample_size)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
