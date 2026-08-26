"""
train.py
--------
DQN 에이전트를 스네이크 게임 환경에서 반복 학습시키는 메인 학습 루프.

실행 : python train.py
- 매 에피소드(한 판)가 끝날 때마다 점수 / 평균 점수를 콘솔에 출력하고,
  matplotlib 창에 학습 곡선을 실시간으로 갱신합니다.
- 최고 점수를 갱신하면 model/best.pth 로 체크포인트를 저장합니다.
- 매 에피소드가 끝나면, 리플레이 버퍼 전체에서 뽑은 배치로 학습(train_long_memory)한
  뒤에 "100게임을 1세트로 묶었을 때, 현재 세트에서 가장 점수가 높았던 한 판"의
  transition들만 따로 다시 샘플링해서 한 번 더 학습(train_from_best_episode)합니다.
  즉, 최근 세트에서 잘 풀렸던 플레이를 기반으로 조금씩(작은 학습률의 경사하강 스텝)
  정책을 다듬어 나가는 효과를 냅니다. 세트가 다 차면(SET_SIZE게임) 그 세트의
  평균/최고 점수를 콘솔에 요약 출력하고 다음 세트를 새로 시작합니다.
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
    episode_transitions = []  # 현재 진행 중인 한 판의 transition들을 순서대로 모아둠

    while True:
        # 1) 현재 state로부터 행동 선택 (epsilon-greedy)
        action = agent.get_action(state_old)

        # 2) 환경에 행동을 적용하고 결과 관찰 (Gym 스타일 step)
        state_new, reward, done, score = game.step(action)

        # 3) 방금 겪은 transition으로 즉시(단기) 학습
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # 4) 리플레이 버퍼에 transition 저장 + 이번 판의 transition 목록에도 기록
        agent.remember(state_old, action, reward, state_new, done)
        episode_transitions.append((state_old, action, reward, state_new, done))

        state_old = state_new

        if done:
            # 한 판이 끝났으므로 환경을 초기화하고, 리플레이 버퍼로 장기 학습 수행
            state_old = game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # 이번 판이 (현재 세트 안에서) 최고 기록이면 최고 기록 에피소드로
            # 저장해두고, 그 에피소드를 기반으로 조금 더 학습한다.
            sets_completed_before = len(agent.set_history)
            agent.update_best_episode(episode_transitions, score)
            agent.train_from_best_episode()
            episode_transitions = []

            # 방금 이 호출로 한 세트(SET_SIZE게임)가 마감됐다면 요약을 출력
            if len(agent.set_history) > sets_completed_before:
                finished_set = agent.set_history[-1]
                champion_note = "★ 새 챔피언 세트!" if finished_set["is_champion"] else "(기존 챔피언 유지)"
                print(
                    f"=== Set {finished_set['set_index']} 완료 (게임 {agent.n_games - 1}까지) "
                    f"| 평균 {finished_set['avg_score']:.2f} | 세트 내 최고 {finished_set['best_score']} "
                    f"| {champion_note} (챔피언 평균: {agent.champion_set_avg:.2f}) ==="
                )

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
