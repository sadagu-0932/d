"""
game.py
-------
Pygame 기반 20x20 grid 스네이크 게임 환경.

- Gym 스타일 인터페이스: reset() / step(action) -> (state, reward, done, score)
- 사람이 방향키로 직접 플레이하는 모드와, AI가 상대적 행동
  ([직진, 우회전, 좌회전])으로 제어하는 모드를 모두 지원합니다.
  (사람의 절대 방향 입력도 내부적으로는 동일한 step(action) 인터페이스를 사용)

직접 실행하면(`python game.py`) 방향키로 플레이할 수 있는 human-play 모드가 실행됩니다.
"""

import random
from collections import namedtuple
from enum import Enum

import numpy as np
import pygame

pygame.init()
FONT = pygame.font.SysFont("arial", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", ["x", "y"])

# ---- 색상 / 그리드 설정 -------------------------------------------------
BLOCK_SIZE = 20          # 한 칸(셀)의 픽셀 크기
GRID_SIZE = 20           # 가로/세로 칸 수 -> 20 x 20 grid
WIDTH = HEIGHT = BLOCK_SIZE * GRID_SIZE  # 400 x 400

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# ---- 보상 설계 -----------------------------------------------------------
REWARD_FOOD = 10          # 먹이 섭취
REWARD_COLLISION = -20    # 충돌/게임오버 (죽는 것에 대한 페널티를 크게)
REWARD_MOVE = 0.1         # 그냥 이동(생존)했을 때 주는 소량의 보상
REWARD_LENGTH_BONUS = 1   # 사망 시, 시작 길이를 초과한 몸길이 1칸당 얹어주는 보너스
INITIAL_SNAKE_LENGTH = 3  # 게임 시작 시 뱀의 길이

# length_bonus가 아무리 커져도 사망 시 reward가 -5보다 좋아지지(0에 가까워지지) 않도록 하는 상한.
# REWARD_COLLISION + MAX_LENGTH_BONUS == -5 가 항상 성립 (REWARD_COLLISION 값이 바뀌어도 동일).
MAX_LENGTH_BONUS = REWARD_COLLISION * -1 - 5

# 먹이를 못 먹고 맴돌기만 할 때 게임을 강제 종료시키는 기준.
# frame_iteration(누적 스텝 수)이 TIMEOUT_STEPS_PER_SEGMENT * len(snake)를 넘으면 종료.
# (몸이 길어질수록 허용 스텝도 늘어나므로, 결과적으로 '최근 먹이를 못 먹은 시간'과 비슷하게 동작)
TIMEOUT_STEPS_PER_SEGMENT = 150

# 시계 방향으로 방향을 나열해두면, 오른쪽으로 90도 회전 = 다음 인덱스,
# 왼쪽으로 90도 회전 = 이전 인덱스로 아주 간단하게 계산할 수 있음.
CLOCK_WISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


class SnakeGameAI:
    """
    20x20 grid 스네이크 게임 환경 (Gym 스타일).

    Parameters
    ----------
    w, h : 게임 화면 크기(px). 기본값은 20x20 grid에 맞춘 400x400.
    render : True면 pygame 창을 띄워 화면을 그림. 학습 속도를 높이고 싶다면 False.
    speed : 초당 프레임 수(FPS). render=True일 때만 의미가 있음.
    use_distance_reward : True면 먹이와의 거리 변화에 따른 소량의 +/-1 보상을
        (먹이 섭취/충돌 보상과 겹치지 않을 때) 추가로 부여.
    """

    def __init__(self, w=WIDTH, h=HEIGHT, render=True, speed=40,
                 use_distance_reward=False):
        self.w = w
        self.h = h
        self.render_enabled = render
        self.speed = speed
        self.use_distance_reward = use_distance_reward

        if self.render_enabled:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake AI")
        else:
            self.display = None
        self.clock = pygame.time.Clock()

        self.reset()

    # ------------------------------------------------------------------
    # Gym 스타일 인터페이스
    # ------------------------------------------------------------------
    def reset(self):
        """게임 상태를 초기화하고 초기 state를 반환."""
        self.direction = Direction.RIGHT

        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            Point(self.head.x - i * BLOCK_SIZE, self.head.y)
            for i in range(INITIAL_SNAKE_LENGTH)
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        return self._get_state()

    def step(self, action):
        """
        action : 길이 3의 one-hot 리스트/배열 [직진, 우회전, 좌회전]

        Returns
        -------
        state : np.ndarray, shape (11,)
        reward : float
        done : bool
        score : int
        """
        self.frame_iteration += 1

        # 창을 닫으면 즉시 종료할 수 있도록 이벤트 처리
        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        prev_distance = self._food_distance()

        # 1) 행동에 따라 이동
        self._move(action)
        self.snake.insert(0, self.head)

        # 2) 보상 / 종료 판정
        reward = 0
        game_over = False

        # 충돌하거나, 너무 오랫동안 먹이를 못 먹으면(맴돌기 방지) 게임 종료
        # -> 맴돌기로 REWARD_MOVE를 계속 챙기는 꼼수도 결국 이 페널티로 막힘
        if self.is_collision() or self.frame_iteration > TIMEOUT_STEPS_PER_SEGMENT * len(self.snake):
            game_over = True
            # 시작 길이를 초과한 몸길이(= 지금까지 먹은 먹이 개수 = score)만큼 보너스를
            # 더해, 오래 살아남아 몸을 키운 뒤 죽는 것이 초반에 바로 죽는 것보다
            # 덜 아프도록 함. 단, 보너스가 아무리 커져도 사망 reward는 항상 최소
            # -5는 남도록(= 죽는 것 자체는 절대 이득이 되지 않도록) MAX_LENGTH_BONUS로 캡을 씌운다.
            # 주의: 이 시점의 self.snake는 방금 insert()로 새 머리가 추가된 직후라
            # len(self.snake)를 그대로 쓰면 1칸 더 많게 계산되므로, 대신 먹은 먹이
            # 개수인 self.score를 사용한다 (score만큼만 몸이 늘어났으므로 동일한 값).
            length_bonus = min(REWARD_LENGTH_BONUS * self.score, MAX_LENGTH_BONUS)
            reward = REWARD_COLLISION + length_bonus
            return self._get_state(), reward, game_over, self.score

        ate_food = self.head == self.food
        if ate_food:
            self.score += 1
            reward = REWARD_FOOD
            self._place_food()
        else:
            self.snake.pop()  # 먹이를 못 먹었으면 꼬리를 잘라 길이를 유지
            reward = REWARD_MOVE  # 죽지 않고 이동한 것 자체에 소량의 보상

        # 3) (옵션) 거리 기반 보조 보상 - 먹이를 먹은 스텝이 아닐 때만 추가로 적용
        if self.use_distance_reward and not ate_food:
            new_distance = self._food_distance()
            reward += 1 if new_distance < prev_distance else -1

        # 4) 렌더링
        if self.render_enabled:
            self._update_ui()
            self.clock.tick(self.speed)

        return self._get_state(), reward, game_over, self.score

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:  # 뱀 몸통 위에 놓였다면 다시 뽑기
            self._place_food()

    def _food_distance(self):
        """머리와 먹이 사이의 맨해튼 거리."""
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # 벽 충돌
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # 몸통 충돌 (머리 자기 자신은 제외)
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        """
        상대적 action([직진, 우회전, 좌회전])을 받아 self.direction / self.head를 갱신.
        action: [1,0,0]=직진, [0,1,0]=우회전, [0,0,1]=좌회전
        """
        idx = CLOCK_WISE.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = CLOCK_WISE[idx]              # 직진: 방향 유지
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = CLOCK_WISE[(idx + 1) % 4]     # 우회전 (시계 방향)
        else:  # [0, 0, 1]
            new_dir = CLOCK_WISE[(idx - 1) % 4]     # 좌회전 (반시계 방향)

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _get_state(self):
        """
        11차원 state 벡터를 계산해서 반환.

        [0] 직진 방향 위험(충돌) 여부
        [1] 우회전 방향 위험 여부
        [2] 좌회전 방향 위험 여부
        [3~6] 현재 이동 방향 one-hot (상, 하, 좌, 우)
        [7~10] 먹이의 상대적 방향 (상, 하, 좌, 우) - boolean
        """
        head = self.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # 현재 진행 방향을 기준으로 '직진/우회전/좌회전'했을 때 부딪히는지 확인
        danger_straight = (
            (dir_r and self.is_collision(point_r))
            or (dir_l and self.is_collision(point_l))
            or (dir_u and self.is_collision(point_u))
            or (dir_d and self.is_collision(point_d))
        )
        danger_right = (
            (dir_u and self.is_collision(point_r))
            or (dir_d and self.is_collision(point_l))
            or (dir_l and self.is_collision(point_u))
            or (dir_r and self.is_collision(point_d))
        )
        danger_left = (
            (dir_d and self.is_collision(point_r))
            or (dir_u and self.is_collision(point_l))
            or (dir_r and self.is_collision(point_u))
            or (dir_l and self.is_collision(point_d))
        )

        state = [
            danger_straight,
            danger_right,
            danger_left,

            dir_u,
            dir_d,
            dir_l,
            dir_r,

            self.food.y < head.y,  # 먹이가 위쪽
            self.food.y > head.y,  # 먹이가 아래쪽
            self.food.x < head.x,  # 먹이가 왼쪽
            self.food.x > head.x,  # 먹이가 오른쪽
        ]
        return np.array(state, dtype=int)

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = FONT.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    # ------------------------------------------------------------------
    # 사람 플레이용 헬퍼
    # 방향키의 절대 방향(상하좌우)을 step()이 요구하는 상대 action으로 변환한다.
    # ------------------------------------------------------------------
    def direction_to_action(self, desired_direction):
        """
        방향키로 입력받은 절대 방향(desired_direction)을 현재 진행 방향 기준
        상대 action([직진, 우회전, 좌회전])으로 변환.
        (뒤로 즉시 반전하는 입력은 무시하고 직진으로 처리 -> 자기 몸통에 즉사 방지)
        """
        idx = CLOCK_WISE.index(self.direction)
        desired_idx = CLOCK_WISE.index(desired_direction)

        if desired_idx == idx:
            return [1, 0, 0]
        elif desired_idx == (idx + 1) % 4:
            return [0, 1, 0]
        elif desired_idx == (idx - 1) % 4:
            return [0, 0, 1]
        else:
            # 정반대 방향(180도 반전) 입력 -> 직진 유지
            return [1, 0, 0]


# ==========================================================================
# 사람이 방향키로 직접 플레이하는 모드
# `python game.py` 로 바로 실행 가능
# ==========================================================================
def play_human():
    game = SnakeGameAI(render=True, speed=10)  # 사람이 반응할 수 있도록 느린 속도
    pending_direction = game.direction

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    pending_direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    pending_direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    pending_direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    pending_direction = Direction.DOWN

        action = game.direction_to_action(pending_direction)
        _, _, done, score = game.step(action)

        if done:
            print(f"게임 종료! 최종 점수: {score}")
            game.reset()
            pending_direction = game.direction


if __name__ == "__main__":
    play_human()
