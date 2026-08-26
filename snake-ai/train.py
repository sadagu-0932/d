"""
train.py
--------
DQN 에이전트를 스네이크 게임 환경에서 반복 학습시키는 메인 학습 루프.

실행 : python train.py
- 매 에피소드(한 판)가 끝날 때마다 점수 / 평균 점수를 콘솔에 출력하고,
  matplotlib 창에 학습 곡선을 실시간으로 갱신합니다.
- 최고 점수를 갱신하면 model/best.pth 로 체크포인트를 저장합니다.
"""

import matplotlib.pyplot as plt

from agent import Agent
from game import SnakeGameAI

# 학습 중 화면 렌더링 여부. True면 게임 화면을 직접 볼 수 있지만, 그만큼 느려짐.
# 최대한 빠르게 많은 에피소드를 돌리고 싶다면 False로 바꾸세요.
RENDER_WHILE_TRAINING = True
GAME_SPEED = 80  # 렌더링을 켰을 때의 FPS (높을수록 화면이 빨리 진행됨)


plt.ion()  # matplotlib 대화형(interactive) 모드 -> 창을 새로 띄우지 않고 실시간 갱신


def plot(scores, mean_scores):
    """지금까지의 (에피소드별 점수, 누적 평균 점수) 그래프를 실시간으로 갱신."""
    plt.clf()
    plt.title("Snake AI Training")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")
    plt.ylim(ymin=0)
    plt.legend(loc="upper left")
    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
    plt.pause(0.001)


def train():
    scores = []        # 에피소드별 점수
    mean_scores = []   # 에피소드별 누적 평균 점수
    total_score = 0
    record = 0          # 지금까지의 최고 점수

    agent = Agent()
    game = SnakeGameAI(render=RENDER_WHILE_TRAINING, speed=GAME_SPEED)
    state_old = game.reset()

    while True:
        # 1) 현재 state로부터 행동 선택 (epsilon-greedy)
        action = agent.get_action(state_old)

        # 2) 환경에 행동을 적용하고 결과 관찰 (Gym 스타일 step)
        state_new, reward, done, score = game.step(action)

        # 3) 방금 겪은 transition으로 즉시(단기) 학습
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # 4) 리플레이 버퍼에 transition 저장
        agent.remember(state_old, action, reward, state_new, done)

        state_old = state_new

        if done:
            # 한 판이 끝났으므로 환경을 초기화하고, 리플레이 버퍼로 장기 학습 수행
            state_old = game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save("best.pth")  # model/best.pth 로 체크포인트 저장
                print(f"[체크포인트 저장] Game {agent.n_games} 신기록 갱신: {record}")

            print(
                f"Game {agent.n_games:>5} | Score: {score:>3} | Record: {record:>3} "
                f"| Epsilon: {agent.get_epsilon():.3f}"
            )

            scores.append(score)
            total_score += score
            mean_scores.append(total_score / agent.n_games)
            plot(scores, mean_scores)


if __name__ == "__main__":
    train()
