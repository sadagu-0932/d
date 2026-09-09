"""
play.py
-------
학습이 끝난 모델(model/best.pth)을 불러와 AI가 직접 플레이하는 모습을 보여줍니다.

실행 예:
    python play.py                     # 기본 FPS(20)로 재생
    python play.py --fps 40            # 더 빠르게 재생
    python play.py --model model/best.pth --episodes 5
"""

import argparse
import os

from agent import Agent
from game import SnakeGameAI


def parse_args():
    parser = argparse.ArgumentParser(description="학습된 스네이크 AI 플레이")
    parser.add_argument("--model", type=str, default="model/best.pth",
                         help="불러올 체크포인트 경로 (기본: model/best.pth)")
    parser.add_argument("--fps", type=int, default=20,
                         help="게임 진행 속도 = 초당 프레임 수 (기본: 20)")
    parser.add_argument("--episodes", type=int, default=0,
                         help="플레이할 게임 수. 0이면 무한 반복 (기본: 0)")
    return parser.parse_args()


def play(model_path: str, fps: int, episodes: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"'{model_path}' 를 찾을 수 없습니다. 먼저 train.py를 실행해 모델을 학습시켜 주세요."
        )

    agent = Agent()
    agent.model.load(model_path)  # 학습된 가중치 로드 (내부에서 eval() 모드로 전환)

    game = SnakeGameAI(render=True, speed=fps)
    state = game.reset()

    games_played = 0
    while episodes == 0 or games_played < episodes:
        # explore=False -> epsilon-greedy 탐험 없이 항상 신경망이 예측한 최선의 행동만 선택
        action = agent.get_action(state, explore=False)
        state, _, done, score = game.step(action)

        if done:
            games_played += 1
            print(f"[Game {games_played}] Score: {score}")
            state = game.reset()


if __name__ == "__main__":
    args = parse_args()
    play(args.model, args.fps, args.episodes)
