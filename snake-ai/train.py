"""
train.py
--------
유전 알고리즘(genetic algorithm)으로 스네이크 신경망 집단(population)을 진화시키는
메인 학습 루프.

실행 : python train.py
- 매 세대(generation)마다 POPULATION_SIZE(기본 50)개체 각각을 GAMES_PER_INDIVIDUAL
  (기본 3)판씩 플레이시킵니다 — 매 판마다 SnakeGameAI.reset(randomize=True)로 뱀의
  길이와 장애물 배치를 무작위로 바꿔서, "한 맵만이 아니라 다양한 필드에서" 평가합니다.
  그 점수들의 평균이 그 개체의 적합도(fitness)가 됩니다.
- 적합도 상위 NUM_PARENTS(기본 4)개체를 부모로 선정합니다. 이 부모들은 그대로
  다음 세대에 보존되고(엘리티즘), 나머지 자리는 부모 중 무작위로 2명을 골라
  가중치를 섞고(crossover) 살짝 변형(mutation)해서 채웁니다 (agent.py의
  Population.evolve 참고). 경사하강/역전파는 전혀 쓰지 않습니다.
- 지금까지 한 판에서 나온 최고 점수(record)를 세우면 그 개체의 가중치를
  model/best.pth로 저장합니다. play.py로 바로 재생할 수 있습니다.
- 매 세대마다 (generation, best_score, avg_score, worst_score, record)를
  TRAINING_LOG_PATH(기본: training_log.csv)에 한 줄씩 기록합니다.
- matplotlib 창에 세대별 최고/평균/최저 적합도 그래프가 실시간으로 갱신됩니다.
"""

import csv

import matplotlib.pyplot as plt

from agent import GAMES_PER_INDIVIDUAL, POPULATION_SIZE, Population, greedy_action
from game import SnakeGameAI

# 세대당 POPULATION_SIZE * GAMES_PER_INDIVIDUAL판을 전부 플레이해야 해서 DQN 때보다
# 한 세대에 훨씬 많은 판을 소화해야 한다. 화면 렌더링을 켜면 그만큼 느려지므로 기본은 끔
# — 학습된 결과는 play.py로 재생해서 확인하세요.
RENDER_WHILE_TRAINING = False
GAME_SPEED = 80  # 렌더링을 켰을 때의 FPS

TRAINING_LOG_PATH = "training_log.csv"


plt.ion()  # matplotlib 대화형(interactive) 모드 -> 창을 새로 띄우지 않고 실시간 갱신


def plot(best_scores, avg_scores, worst_scores):
    """지금까지의 세대별 (최고/평균/최저 적합도) 그래프를 실시간으로 갱신."""
    plt.clf()
    plt.title("Snake AI Genetic Training")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (avg score per individual)")
    plt.plot(best_scores, label="Best")
    plt.plot(avg_scores, label="Average")
    plt.plot(worst_scores, label="Worst")
    plt.ylim(ymin=0)
    plt.legend(loc="upper left")
    if best_scores:
        plt.text(len(best_scores) - 1, best_scores[-1], f"{best_scores[-1]:.2f}")
    plt.pause(0.001)


def train():
    game = SnakeGameAI(render=RENDER_WHILE_TRAINING, speed=GAME_SPEED)
    population = Population(POPULATION_SIZE)

    def play_episode(model):
        """
        model(신경망) 하나를 GAMES_PER_INDIVIDUAL판 플레이시켜 (score, total_reward)
        튜플 리스트를 반환. 매 판마다 reset(randomize=True)로 뱀 길이·장애물을 새로
        무작위화해서 "다양한 필드"에서 평가한다. total_reward는 그 판의 매 스텝
        step()이 돌려주는 reward를 전부 더한 값 — Population.evaluate()가 이걸
        fitness(선택 압력)로 쓰고, score는 record/체크포인트 기준으로 따로 쓴다.
        """
        episodes = []
        for _ in range(GAMES_PER_INDIVIDUAL):
            state = game.reset(randomize=True)
            done = False
            total_reward = 0.0
            while not done:
                action = greedy_action(model, state)
                state, reward, done, score = game.step(action)
                total_reward += reward
            episodes.append((score, total_reward))
        return episodes

    best_history, avg_history, worst_history = [], [], []

    log_file = open(TRAINING_LOG_PATH, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["generation", "best_score", "avg_score", "worst_score", "record"])
    log_file.flush()

    while True:
        best_fit, avg_fit, worst_fit = population.evaluate(play_episode)

        if population.save_best_if_new_record():
            print(f"[체크포인트 저장] Generation {population.generation} 신기록 갱신: {population.record}")

        print(
            f"Gen {population.generation:>5} | Best {best_fit:>6.2f} | Avg {avg_fit:>6.2f} "
            f"| Worst {worst_fit:>6.2f} | Record {population.record:>3}"
        )

        best_history.append(best_fit)
        avg_history.append(avg_fit)
        worst_history.append(worst_fit)
        plot(best_history, avg_history, worst_history)

        log_writer.writerow([population.generation, round(best_fit, 4), round(avg_fit, 4),
                              round(worst_fit, 4), population.record])
        log_file.flush()

        population.evolve()


if __name__ == "__main__":
    train()
