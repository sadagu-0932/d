"""
agent.py
--------
유전 알고리즘(genetic algorithm)으로 스네이크 신경망을 학습시킵니다.

- Agent      : 학습된 가중치 하나를 불러와 플레이만 하는 가벼운 래퍼 (play.py에서 사용).
- Population : 신경망 여러 개(개체 집단)를 관리하며 세대를 진화시키는 클래스.
               매 세대마다 각 개체를 여러 판(다양하게 무작위화된 맵)에 플레이시켜
               점수를 매기고, 가장 우수한 4개체를 부모로 선정해 — 그 부모들의
               가중치를 섞고(crossover) 살짝 변형(mutation)해서 — 다음 세대를
               만듭니다. 경사하강(역전파)은 전혀 쓰지 않습니다.

신경망 구조(Linear_QNet, 11 -> 256 -> 3)는 model.py에 고정돼 있고, 여기서 바뀌는
건 그 가중치뿐입니다 — "뉴런 개수나 레이어 구조를 AI가 알아서 정하는" 것은 아니고,
구조는 우리가 미리 정해둔 채로 가중치만 진화시킵니다.
"""

import copy
import random

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from game import STATE_SIZE
from model import Linear_QNet

HIDDEN_SIZE = 256
OUTPUT_SIZE = 3

# ---- 유전 알고리즘 파라미터 -------------------------------------------------
POPULATION_SIZE = 50     # 한 세대에 몇 개체(신경망)를 둘지
NUM_PARENTS = 4          # 매 세대마다 부모로 뽑는 상위 개체 수 (요청대로 4개)
# 다양한 필드에서 평가하기 위해, 개체 한 명당 여러 판(각각 다른 무작위 맵)을 플레이시켜
# 평균 점수를 그 개체의 적합도(fitness)로 쓴다. 한 판만 보면 운 좋게/나쁘게 걸린
# 맵 하나로 전체를 평가하게 되는 문제가 있어서, GAMES_PER_INDIVIDUAL판을 평균낸다.
GAMES_PER_INDIVIDUAL = 3
MUTATION_RATE = 0.1      # 자식 개체의 가중치 각각이 변이(mutation)를 겪을 확률
MUTATION_STRENGTH = 0.1  # 변이가 일어날 때 더해지는 가우시안 노이즈의 표준편차


def greedy_action(model: Linear_QNet, state):
    """신경망 하나로부터 결정적으로(탐험 없이) 행동을 뽑는다 — 선호도 점수가 가장 높은 쪽."""
    state_tensor = torch.tensor(state, dtype=torch.float)
    with torch.no_grad():
        prediction = model(state_tensor)
    move = torch.argmax(prediction).item()

    final_move = [0, 0, 0]
    final_move[move] = 1
    return final_move


class Agent:
    """학습된 가중치 하나를 불러와 플레이만 하는 가벼운 래퍼 (play.py에서 사용)."""

    def __init__(self):
        self.model = Linear_QNet(STATE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    def get_action(self, state, explore: bool = False):
        """
        신경망이 예측한 선호도 점수가 가장 높은 행동을 그대로 선택.

        explore 인자는 과거 epsilon-greedy DQN 시절의 play.py 호출 형태와
        호환되도록 남겨뒀지만, 유전 알고리즘으로 학습된 개체는 항상 결정적으로
        (탐험 없이) 플레이하므로 실제로는 아무 효과가 없다.
        """
        return greedy_action(self.model, state)


# ----------------------------------------------------------------------------
# 유전 연산 (crossover / mutation) — 신경망의 모든 가중치를 1차원 벡터로 펴서
# 다루면 레이어 구조를 몰라도 되는 범용 연산이 된다.
# ----------------------------------------------------------------------------
def _new_model() -> Linear_QNet:
    return Linear_QNet(STATE_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)


def crossover(parent_a: Linear_QNet, parent_b: Linear_QNet) -> Linear_QNet:
    """
    두 부모의 가중치를 섞어(uniform crossover) 자식 신경망 하나를 만든다.
    가중치 각 성분마다 50% 확률로 parent_a 또는 parent_b의 값을 물려받는다.
    """
    vec_a = parameters_to_vector(parent_a.parameters())
    vec_b = parameters_to_vector(parent_b.parameters())

    mask = torch.rand_like(vec_a) < 0.5
    child_vec = torch.where(mask, vec_a, vec_b)

    child = _new_model()
    vector_to_parameters(child_vec, child.parameters())
    return child


def mutate(model: Linear_QNet, rate: float = MUTATION_RATE, strength: float = MUTATION_STRENGTH) -> None:
    """
    model의 가중치 각 성분마다 rate 확률로 평균 0, 표준편차 strength인 가우시안
    노이즈를 더해서 살짝 변형한다 (in-place).
    """
    with torch.no_grad():
        vec = parameters_to_vector(model.parameters())
        mutate_mask = torch.rand_like(vec) < rate
        noise = torch.randn_like(vec) * strength
        vec = torch.where(mutate_mask, vec + noise, vec)
        vector_to_parameters(vec, model.parameters())


def clone_model(model: Linear_QNet) -> Linear_QNet:
    """가중치까지 완전히 복제한 새 신경망을 반환 (부모를 다음 세대로 그대로 넘길 때 사용)."""
    return copy.deepcopy(model)


class Population:
    """
    POPULATION_SIZE개의 신경망(개체)을 관리하며 세대를 진화시키는 클래스.

    evaluate(game_factory)로 현재 세대 전체의 적합도(fitness)를 계산한 뒤,
    evolve()를 호출하면:
      1. 적합도 상위 NUM_PARENTS(4)개체를 부모로 선정하고, 그대로 다음 세대에
         엘리트로 보존한다(엘리티즘 — 지금까지 찾은 가장 좋은 개체를 잃지 않음).
      2. 나머지 자리는 그 4명의 부모 중 무작위로 2명을 골라 crossover + mutate해서
         채운다.
    """

    def __init__(self, population_size: int = POPULATION_SIZE):
        self.population_size = population_size
        self.individuals = [_new_model() for _ in range(population_size)]
        self.generation = 0

        self.fitnesses = [0.0] * population_size   # 가장 최근 evaluate() 결과
        self.best_scores = [0] * population_size    # 개체별, 여러 판 중 최고 단일 점수

        self.record = 0            # 지금까지 한 판에서 나온 최고 점수
        self.best_model_state = None  # 그 기록을 세운 신경망의 가중치(state_dict)
        self._last_saved_record = -1  # save_best_if_new_record()가 중복 저장을 피하려고 기억하는 값

    def evaluate(self, play_episode):
        """
        현재 세대의 모든 개체를 평가한다.

        play_episode(model) : 신경망 하나를 받아 "다양한 필드(서로 다르게
        무작위화된 맵)"에서 GAMES_PER_INDIVIDUAL판을 플레이시키고, 각 판의
        (score, total_reward) 튜플 리스트를 반환하는 콜백(train.py에서 정의).

        - fitness(선택 압력에 쓰이는 값)는 total_reward(판마다 step()이 돌려주는
          reward를 다 더한 값)의 평균이다. game.py의 REWARD_FOOD/REWARD_DEATH
          보상 설계를 그대로 선택 압력에 반영하기 위함 — 예전처럼 raw score만
          쓰면 이 보상 상수들을 바꿔도 어느 개체가 부모가 될지는 전혀 달라지지
          않는다.
        - best_scores/record(화면에 보여주고 체크포인트를 저장하는 기준)는 reward
          가중치와 무관하게, 실제로 "몇 개 먹었는지"(score) 그 자체를 쓴다.

        반환값: 이번 세대의 (최고 fitness, 평균 fitness, 최저 fitness)
        """
        for i, model in enumerate(self.individuals):
            episodes = play_episode(model)
            scores = [score for score, _ in episodes]
            rewards = [reward for _, reward in episodes]
            self.fitnesses[i] = sum(rewards) / len(rewards)
            self.best_scores[i] = max(scores)

            if self.best_scores[i] > self.record:
                self.record = self.best_scores[i]
                self.best_model_state = copy.deepcopy(model.state_dict())

        return max(self.fitnesses), sum(self.fitnesses) / len(self.fitnesses), min(self.fitnesses)

    def top_parents(self, n: int = NUM_PARENTS):
        """fitness 기준 상위 n개체(신경망)를 반환."""
        ranked = sorted(range(self.population_size), key=lambda i: self.fitnesses[i], reverse=True)
        return [self.individuals[i] for i in ranked[:n]]

    def evolve(self):
        """
        가장 최근 evaluate() 결과를 바탕으로 다음 세대를 만든다.
        (엘리트 NUM_PARENTS개체는 그대로 보존 + 나머지는 그 부모들을 섞고
        변형해서 채움)
        """
        parents = self.top_parents(NUM_PARENTS)

        next_generation = [clone_model(p) for p in parents]  # 엘리티즘: 부모 원본 보존
        while len(next_generation) < self.population_size:
            mother, father = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
            child = crossover(mother, father)
            mutate(child)
            next_generation.append(child)

        self.individuals = next_generation
        self.generation += 1

    def save_best_if_new_record(self) -> bool:
        """
        record가 마지막 저장 이후 갱신됐으면 그 가중치를 model/best.pth로 저장하고
        True를 반환 (train.py 콘솔 로그용). 갱신되지 않았으면 아무 것도 안 하고 False.
        """
        if self.best_model_state is None or self.record <= self._last_saved_record:
            return False

        self._last_saved_record = self.record
        best_model = _new_model()
        best_model.load_state_dict(self.best_model_state)
        best_model.save("best.pth")
        return True
