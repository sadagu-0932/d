"""
agent.py
--------
DQN 에이전트: 게임(game.py)이 반환한 state를 받아 행동을 선택하고,
Experience Replay Buffer를 이용해 신경망(model.py)을 학습시킵니다.

또한 매 학습 스텝(QTrainer.train_step)마다 나오는 loss를 누적해뒀다가
에피소드 단위 평균으로 뽑아낼 수 있게 합니다 (pop_episode_avg_loss,
train.py의 CSV 로깅용). 점수 진동과 loss 스파이크가 겹치는지 나중에
확인하기 위한 용도입니다.
"""

import random
from collections import deque

import torch

from game import STATE_SIZE
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000   # 리플레이 버퍼 최대 크기 (deque라서 꽉 차면 오래된 것부터 자동 삭제)
BATCH_SIZE = 1_000     # 장기 기억(long memory) 학습 시 한 번에 샘플링할 transition 개수
LR = 0.001             # 학습률(learning rate)

# epsilon-greedy 탐험 파라미터
# 학습 초반에는 무작위 행동(탐험) 비중을 높게, 게임 수가 늘수록 점점 줄여 나감(감쇠).
# 예전 값(EPS_DECAY_GAMES=150, EPS_END=0.02)에서는 점수가 오르기 시작하는 바로 그
# 시점(100~200게임)에서 탐험이 사실상 끝나버려서, 그 이후 수백~수천 게임 동안 평균
# 점수가 전혀 개선되지 않고 정체(plateau)되는 현상이 실제로 관찰됨.
#
# 처음엔 EPS_END도 0.05로 같이 올렸는데, 직접 head-to-head 비교 실험(웹 데모 JS로
# decay/floor 값만 바꿔가며 600게임씩 4가지 조합을 실전 학습)을 해보니 이게 오히려
# 역효과였음: floor를 올리면 학습이 다 끝난 뒤에도 "매 스텝 5% 확률로 완전 무작위
# 행동"이 영원히 남는데, 뱀이 길어질수록 좁은 공간에서 무작위 행동 한 번이 곧바로
# 죽음으로 이어지기 쉬워서, 최종적으로 도달 가능한 점수 자체가 예전(EPS_END=0.02)
# 보다 훨씬 낮은 수준에서 다시 정체됨 (실측: 600게임 기준 record 76 -> 48로 하락).
# 반면 EPS_DECAY_GAMES만 늘리고 EPS_END는 예전 값(0.02)으로 유지한 조합은 정체
# 시점만 늦추면서 최종 도달 점수는 기존과 동등하거나 더 나았음 (실측: record 75,
# 기존과 사실상 동일). 그래서 EPS_DECAY_GAMES만 늘리고 EPS_END는 원래 값으로 되돌림.
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY_GAMES = 300  # 이 정도 게임 수가 지나면 epsilon이 EPS_END 근처까지 감소

SET_SIZE = 100  # 몇 게임을 '1세트'로 묶어서 세트별 최고 기록을 다시 집계할지


class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9  # 미래 보상 할인율 (벨만 방정식에 사용)
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience Replay Buffer

        # SET_SIZE(기본 100)게임을 '1세트'로 묶어서 세트별 평균 점수를 비교한다.
        # - 진행 중인 세트 안에서의 최고 기록 에피소드는 current_set_*에 임시로 담아둔다.
        # - 세트가 끝나면, 그 세트의 평균 점수를 지금까지의 '챔피언 세트'(champion_set_avg)와
        #   비교해서 더 높으면 챔피언을 교체한다 (아니면 기존 챔피언을 그대로 유지).
        # - train_from_best_episode()는 항상 이 챔피언 세트의 최고 기록 에피소드(self.best_episode)를
        #   기반으로 추가 학습한다. 즉 "최근 세트가 지금까지의 최고 세트보다 나으면 그걸로 교체하고,
        #   아니면 계속 기존 최고 세트를 쓴다"는 규칙을 매 세트마다 반복한다.
        self.games_in_set = 0
        self.set_scores = []                       # 현재 세트에서 나온 점수들 (세트 평균 계산용)
        self.current_set_best_episode = []          # 현재 세트 안에서의 최고 기록 에피소드
        self.current_set_best_episode_score = -1

        self.champion_set_avg = None    # 지금까지 채택된 챔피언 세트의 평균 점수 (아직 없으면 None)
        self.best_episode = []          # 챔피언 세트의 최고 기록 에피소드 (train_from_best_episode가 사용)
        self.best_episode_score = -1
        self.set_history = []           # 완료된 세트들의 요약 [{"set_index","avg_score","best_score","is_champion"}]

        # 이번 에피소드 동안 발생한 학습 loss를 누적해뒀다가, 에피소드가 끝나면
        # pop_episode_avg_loss()로 평균을 꺼내 쓴다 (train.py의 CSV 로깅용).
        self.episode_loss_sum = 0.0
        self.episode_loss_count = 0

        self.model = Linear_QNet(STATE_SIZE, 256, 3)  # state(STATE_SIZE) -> hidden(256) -> action(3)
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
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        self._record_loss(loss)

    def train_long_memory(self):
        """리플레이 버퍼에서 배치를 무작위로 샘플링해 학습 (경험 재사용 + 데이터 상관관계 완화)."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        self._record_loss(loss)

    # ----------------------------------------------------------------
    # loss 로깅: 매 학습 스텝(train_step)마다 나오는 loss를 누적해뒀다가,
    # 에피소드가 끝나면 평균을 뽑아 CSV에 기록한다 (train.py에서 사용).
    # ----------------------------------------------------------------
    def _record_loss(self, loss: float):
        self.episode_loss_sum += loss
        self.episode_loss_count += 1

    def pop_episode_avg_loss(self) -> float:
        """이번 에피소드 동안 기록된 loss들의 평균을 반환하고, 다음 에피소드를 위해 리셋."""
        if self.episode_loss_count == 0:
            avg_loss = 0.0
        else:
            avg_loss = self.episode_loss_sum / self.episode_loss_count

        self.episode_loss_sum = 0.0
        self.episode_loss_count = 0
        return avg_loss

    # ----------------------------------------------------------------
    # "세트(SET_SIZE게임)별 평균 점수를 비교해 더 나은 세트의 플레이" 기반 추가 학습
    # ----------------------------------------------------------------
    def update_best_episode(self, episode_transitions, score):
        """
        방금 끝난 에피소드를 현재 진행 중인 세트에 반영하고, 세트가 다 찼으면 마감한다.

        1) 이번 에피소드가 현재 세트 안에서 최고 기록이면 current_set_best_episode에 저장.
        2) 세트가 SET_SIZE게임에 도달했으면:
           - 이 세트의 평균 점수를 계산해서 지금까지의 챔피언 세트 평균과 비교.
           - 이번 세트가 더 높으면(또는 아직 챔피언이 없으면) 챔피언을 이 세트로 교체.
             더 낮으면 기존 챔피언을 그대로 유지한다.
           - 다음 세트를 위해 진행 중 상태를 초기화한다.
        """
        self.games_in_set += 1
        self.set_scores.append(score)
        if score > self.current_set_best_episode_score:
            self.current_set_best_episode_score = score
            self.current_set_best_episode = list(episode_transitions)

        if self.games_in_set >= SET_SIZE:
            avg_score = sum(self.set_scores) / len(self.set_scores)
            is_champion = self.champion_set_avg is None or avg_score > self.champion_set_avg

            if is_champion:
                self.champion_set_avg = avg_score
                self.best_episode = self.current_set_best_episode
                self.best_episode_score = self.current_set_best_episode_score

            self.set_history.append({
                "set_index": len(self.set_history) + 1,
                "avg_score": avg_score,
                "best_score": self.current_set_best_episode_score,
                "is_champion": is_champion,
            })

            # 다음 세트를 위해 진행 중 상태만 초기화 (챔피언 정보는 그대로 유지)
            self.games_in_set = 0
            self.set_scores = []
            self.current_set_best_episode = []
            self.current_set_best_episode_score = -1

    def train_from_best_episode(self):
        """
        저장해 둔 '챔피언 세트(지금까지 세트 평균이 가장 높았던 세트)의 최고 기록
        에피소드'에서 배치를 뽑아 추가로 학습.

        train_long_memory()와 똑같은 방식(무작위 샘플 + 일반 배치 학습)이지만,
        표본을 전체 리플레이 버퍼가 아니라 챔피언 세트의 에피소드 하나로 한정한다.
        매 에피소드마다 이 함수를 한 번 더 호출해주면, 신경망이 지금까지 가장 안정적으로
        잘했던 구간의 상태->행동 대응을 조금씩 더 강하게 기억하게 된다.
        """
        if not self.best_episode:
            return

        sample_size = min(len(self.best_episode), BATCH_SIZE)
        mini_sample = random.sample(self.best_episode, sample_size)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        self._record_loss(loss)
