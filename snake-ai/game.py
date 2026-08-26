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
from collections import deque, namedtuple
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

# 방향을 꺾을 때(직진이 아닌 우회전/좌회전)마다 주는 소량의 페널티.
# 많이 꺾을수록(지그재그로 움직일수록) 누적 감점이 커지도록 해서, 불필요하게
# 자주 방향을 트는 경로를 억제한다. 특히 좁은 공간에서 계속 꺾으며 이동하면
# 자기 몸통으로 'ㄷ'자(U자) 모양을 만들다가 그 안에 스스로 갇히는 경우가 많은데,
# 꺾는 행동 자체에 비용을 매겨 이런 경로를 덜 선호하게 만드는 것이 목적이다.
# (먹이를 먹거나 죽음을 피하기 위해 꼭 필요한 회전은 그 보상/페널티가 훨씬 크므로
# 여전히 선택된다 — 이 페널티는 '불필요한' 지그재그만 줄이는 정도의 크기로 잡는다)
REWARD_TURN_PENALTY = -0.2

# length_bonus가 아무리 커져도 사망 시 reward가 -5보다 좋아지지(0에 가까워지지) 않도록 하는 상한.
# REWARD_COLLISION + MAX_LENGTH_BONUS == -5 가 항상 성립 (REWARD_COLLISION 값이 바뀌어도 동일).
MAX_LENGTH_BONUS = REWARD_COLLISION * -1 - 5

# 죽기 직전, 머리 기준 상하좌우 네 방향이 전부(벽이 아니라) 자기 몸통으로 막혀 있었다면
# -> 어떤 행동을 했어도 피할 수 없었던 '완전 자기 감금' 죽음이므로 훨씬 크게 감점한다.
# (length_bonus의 -5 하한과는 별개로 적용되어, 이 경우엔 reward가 -5보다 훨씬 낮아질 수 있다)
REWARD_SELF_TRAP_PENALTY = -30

# 먹이를 못 먹고 맴돌기만 할 때 게임을 강제 종료시키는 기준.
# frame_iteration(누적 스텝 수)이 TIMEOUT_STEPS_PER_SEGMENT * len(snake)를 넘으면 종료.
# (몸이 길어질수록 허용 스텝도 늘어나므로, 결과적으로 '최근 먹이를 못 먹은 시간'과 비슷하게 동작)
TIMEOUT_STEPS_PER_SEGMENT = 150

# 시계 방향으로 방향을 나열해두면, 오른쪽으로 90도 회전 = 다음 인덱스,
# 왼쪽으로 90도 회전 = 이전 인덱스로 아주 간단하게 계산할 수 있음.
CLOCK_WISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

# Direction -> (dx, dy) 단위 이동량. lookahead/reachability 계산에서 매 스텝
# 반복 재사용하므로, 호출할 때마다 dict를 새로 만들지 않도록 모듈 상수로 미리 정의.
DIRECTION_DELTA = {
    Direction.RIGHT: (1, 0),
    Direction.LEFT: (-1, 0),
    Direction.DOWN: (0, 1),
    Direction.UP: (0, -1),
}

# ---- state에서 미리 내다볼(lookahead) 위험 감지 범위 ----------------------
# 기존에는 직진/좌/우 방향으로 딱 1칸 앞의 위험만 봤는데, 그러면 곧바로 옆이
# 막혀야만 '위험'을 인지해서 미리 대비하기 어려웠다. 그래서 직진 방향은 더 멀리,
# 좌우 방향은 그보다 조금 덜 멀리 내다보게 해서 한발 앞서 생각할 수 있게 한다.
FORWARD_LOOKAHEAD = 3  # 직진 방향으로 몇 칸 앞까지 위험을 확인할지
SIDE_LOOKAHEAD = 2     # 좌/우 방향으로 몇 칸 앞까지 위험을 확인할지

# ---- state에서 쓸 flood-fill 기반 전역 reachability feature ---------------
# lookahead(위 FORWARD_LOOKAHEAD/SIDE_LOOKAHEAD)는 "바로 근처가 막혔는지"만 보는
# 국소적인 정보라, 안쪽이 넓게 뚫려 보여도 사실은 자기 몸통에 둘러싸여 갇힌
# 구석일 수 있다. 그래서 직진/우회전/좌회전 각 후보 행동에 대해 BFS(flood-fill)로
# "그 방향으로 가면 실제로 얼마나 넓은 공간에 접근할 수 있는지"까지 함께 본다.
REACHABILITY_FEATURES = 6  # (reachable_area 3개 + can_reach_tail 3개)

# state 벡터 총 차원 =
#   (직진 위험 3 + 좌 위험 2 + 우 위험 2) + 이동방향 one-hot 4 + 먹이방향 4
#   + reachability feature 6 (직진/우/좌 각각의 reachable_area, can_reach_tail)
STATE_SIZE = FORWARD_LOOKAHEAD + SIDE_LOOKAHEAD * 2 + 4 + 4 + REACHABILITY_FEATURES


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
        # 이동하기 '전' 시점 기준으로 완전 자기 감금 상태였는지 미리 확인해둔다.
        # (이동/insert 이후에는 self.snake[0]이 새 머리로 바뀌어 버려서 판단할 수 없음)
        was_boxed_in = self._is_boxed_in_by_own_body()

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

            # 완전 자기 감금(사방이 전부 자기 몸통) 상태에서 죽은 것이라면, 어떤 행동을
            # 했어도 피할 수 없었던 명백히 예측 가능한 실수였으므로 훨씬 크게 감점한다.
            # (위의 -5 하한과는 별개로 적용되어 reward가 그보다 훨씬 낮아질 수 있다)
            if was_boxed_in:
                reward += REWARD_SELF_TRAP_PENALTY

            return self._get_state(), reward, game_over, self.score

        ate_food = self.head == self.food
        if ate_food:
            self.score += 1
            reward = REWARD_FOOD
            self._place_food()
        else:
            self.snake.pop()  # 먹이를 못 먹었으면 꼬리를 잘라 길이를 유지
            reward = REWARD_MOVE  # 죽지 않고 이동한 것 자체에 소량의 보상

        # 2.5) 이번 행동이 (직진이 아니라) 방향을 꺾은 것이었다면 소량의 페널티를 추가.
        # 먹이를 먹었어도 예외 없이 적용 — REWARD_FOOD(10)에 비하면 무시할 수준이라
        # 필요한 회전(먹이 쪽으로 꺾기)을 막지는 않으면서, 불필요하게 자주 꺾는
        # 지그재그 경로만 누적 페널티로 억제한다.
        if not np.array_equal(action, [1, 0, 0]):
            reward += REWARD_TURN_PENALTY

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

    def _is_boxed_in_by_own_body(self):
        """
        (이동하기 전) 머리 기준 상하좌우 네 칸이 전부 '벽이 아니라 자기 몸통'인지 확인.

        네 칸 중 하나라도 벽(그리드 밖)이면 몸통으로 완전히 둘러싸인 게 아니므로 False.
        이 조건이 True라면, 이번에 어떤 방향으로 움직였어도 피할 수 없었던
        '완전 자기 감금' 상태였다는 뜻이다.
        """
        head = self.snake[0]
        neighbors = [
            Point(head.x, head.y - BLOCK_SIZE),
            Point(head.x, head.y + BLOCK_SIZE),
            Point(head.x - BLOCK_SIZE, head.y),
            Point(head.x + BLOCK_SIZE, head.y),
        ]
        for pt in neighbors:
            if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
                return False  # 벽 밖 -> 자기 몸통이 아니므로 완전 감금이 아님
            if pt not in self.snake[1:]:
                return False  # 몸통이 아닌 빈 칸(또는 머리 자신) -> 완전 감금이 아님
        return True

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

    def _relative_direction(self, turn):
        """turn: 0=직진, +1=우회전, -1=좌회전 (CLOCK_WISE 기준) -> 절대 Direction 반환."""
        idx = CLOCK_WISE.index(self.direction)
        return CLOCK_WISE[(idx + turn) % 4]

    def _lookahead_dangers(self, turn, steps):
        """
        현재 진행 방향 기준 turn(0=직진, +1=우회전, -1=좌회전) 쪽으로, 머리에서
        1칸, 2칸, ..., steps칸 떨어진 지점이 각각 위험(벽 또는 몸통)한지 리스트로 반환.
        """
        direction = self._relative_direction(turn)
        dx, dy = DIRECTION_DELTA[direction]

        head = self.snake[0]
        return [
            self.is_collision(
                Point(head.x + dx * BLOCK_SIZE * step, head.y + dy * BLOCK_SIZE * step)
            )
            for step in range(1, steps + 1)
        ]

    def _flood_fill_from(self, point, obstacle_grid):
        """
        point 칸에서 BFS(flood-fill)로 도달 가능한 빈 칸 개수와, 그 과정에서
        뱀의 꼬리 칸에 도달했는지 여부를 계산해서 (reachable_count, reached_tail)로 반환한다.

        obstacle_grid: 길이 GRID_SIZE*GRID_SIZE인 bytearray. 인덱스는
        `row * GRID_SIZE + col`이고, 값이 1이면 그 칸은 막힌 칸(장애물)이다.
        호출부(_reachability_features)에서 '이번에 먹이를 먹는 후보 행동인지'에
        따라 꼬리 포함 여부를 다르게 구성해서 넘겨준다.

        시작 칸 자체가 그리드 밖이거나 obstacle_grid에서 막혀 있으면(=그 후보
        행동 자체가 즉시 충돌) 도달 가능 칸 0, 꼬리 도달 False로 바로 반환한다.

        3000+ steps/sec 환경에서 매 스텝(후보 방향당 1번, 최대 3번) 호출되는 걸
        감안해서, Point/튜플 기반 set 대신 정수 인덱스 + bytearray로 가볍게 구현했다
        (장애물 grid 자체는 이 메서드 밖에서 한 번만 만들어 재사용됨).
        """
        col, row = point.x // BLOCK_SIZE, point.y // BLOCK_SIZE
        if not (0 <= col < GRID_SIZE and 0 <= row < GRID_SIZE):
            return 0, False
        start_idx = row * GRID_SIZE + col
        if obstacle_grid[start_idx]:
            return 0, False

        tail = self.snake[-1]
        tail_idx = (tail.y // BLOCK_SIZE) * GRID_SIZE + (tail.x // BLOCK_SIZE)

        visited = bytearray(GRID_SIZE * GRID_SIZE)
        visited[start_idx] = 1
        queue = deque((start_idx,))
        reachable_count = 1
        reached_tail = start_idx == tail_idx

        while queue:
            idx = queue.popleft()
            r, c = divmod(idx, GRID_SIZE)

            if c > 0:
                nidx = idx - 1
                if not visited[nidx] and not obstacle_grid[nidx]:
                    visited[nidx] = 1
                    reachable_count += 1
                    reached_tail = reached_tail or nidx == tail_idx
                    queue.append(nidx)
            if c < GRID_SIZE - 1:
                nidx = idx + 1
                if not visited[nidx] and not obstacle_grid[nidx]:
                    visited[nidx] = 1
                    reachable_count += 1
                    reached_tail = reached_tail or nidx == tail_idx
                    queue.append(nidx)
            if r > 0:
                nidx = idx - GRID_SIZE
                if not visited[nidx] and not obstacle_grid[nidx]:
                    visited[nidx] = 1
                    reachable_count += 1
                    reached_tail = reached_tail or nidx == tail_idx
                    queue.append(nidx)
            if r < GRID_SIZE - 1:
                nidx = idx + GRID_SIZE
                if not visited[nidx] and not obstacle_grid[nidx]:
                    visited[nidx] = 1
                    reachable_count += 1
                    reached_tail = reached_tail or nidx == tail_idx
                    queue.append(nidx)

        return reachable_count, reached_tail

    def _reachability_features(self):
        """
        직진/우회전/좌회전 각 후보 행동에 대해 flood-fill 기반 (정규화된 도달
        가능 면적, 꼬리 도달 가능 여부)를 계산해서
        (area_straight, area_right, area_left, tail_straight, tail_right, tail_left)
        6개 값을 튜플로 반환한다.

        세 후보가 대부분 같은 장애물 배치(몸통, 꼬리는 제외)를 공유하므로,
        obstacle_grid를 매 후보마다 새로 만들지 않고 여기서 딱 한 번(먹이를
        먹는 후보가 있을 때만 꼬리 포함 버전을 하나 더, 지연 생성)만 만들어
        재사용한다 -> 매 스텝(3000+ steps/sec) 호출되는 이 계산의 비용을 크게 줄인다.

        주의: 벽 충돌로 게임이 끝나는 바로 그 스텝에서는 self.snake[0](머리)이
        그리드 밖 좌표를 가질 수 있다 (step()이 game_over 판정 뒤에도 next_state로
        _get_state()를 호출하기 때문). 그 next_state는 done=True라 학습에 실제로
        쓰이진 않지만, 크래시가 나면 안 되므로 그리드 밖 세그먼트는 조용히 건너뛴다.
        """
        grid_no_tail = bytearray(GRID_SIZE * GRID_SIZE)
        for p in self.snake[:-1]:  # 꼬리는 다음 스텝에 비워질 예정이므로 제외
            col, row = p.x // BLOCK_SIZE, p.y // BLOCK_SIZE
            if 0 <= col < GRID_SIZE and 0 <= row < GRID_SIZE:
                grid_no_tail[row * GRID_SIZE + col] = 1

        grid_with_tail = None  # 먹이를 먹는 후보가 나올 때만 지연 생성 (복사 1회)
        cell_total = GRID_SIZE * GRID_SIZE
        results = []

        for turn in (0, 1, -1):  # 직진, 우회전, 좌회전
            dx, dy = DIRECTION_DELTA[self._relative_direction(turn)]
            head = self.snake[0]
            candidate = Point(head.x + dx * BLOCK_SIZE, head.y + dy * BLOCK_SIZE)

            if candidate == self.food:
                # 먹이를 먹는 행동이라 꼬리가 이번 스텝엔 안 빠짐 -> 꼬리도 장애물에 포함
                if grid_with_tail is None:
                    grid_with_tail = bytearray(grid_no_tail)
                    tail = self.snake[-1]
                    tail_col, tail_row = tail.x // BLOCK_SIZE, tail.y // BLOCK_SIZE
                    if 0 <= tail_col < GRID_SIZE and 0 <= tail_row < GRID_SIZE:
                        grid_with_tail[tail_row * GRID_SIZE + tail_col] = 1
                grid = grid_with_tail
            else:
                grid = grid_no_tail

            count, reached_tail = self._flood_fill_from(candidate, grid)
            results.append((count / cell_total, reached_tail))

        (area_straight, tail_straight), (area_right, tail_right), (area_left, tail_left) = results
        return area_straight, area_right, area_left, tail_straight, tail_right, tail_left

    def _get_state(self):
        """
        STATE_SIZE(기본 21)차원 state 벡터를 계산해서 반환.

        [0:3]   직진 방향으로 1~3칸 앞의 위험(충돌) 여부 (FORWARD_LOOKAHEAD)
        [3:5]   우회전 방향으로 1~2칸 앞의 위험 여부 (SIDE_LOOKAHEAD)
        [5:7]   좌회전 방향으로 1~2칸 앞의 위험 여부 (SIDE_LOOKAHEAD)
        [7:11]  현재 이동 방향 one-hot (상, 하, 좌, 우)
        [11:15] 먹이의 상대적 방향 (상, 하, 좌, 우) - boolean
        [15:18] 직진/우회전/좌회전 각 방향의 정규화된 flood-fill 도달 가능 면적 (0~1)
        [18:21] 직진/우회전/좌회전 각 방향에서 꼬리 칸에 도달 가능한지 여부 (boolean)

        lookahead([0:7])는 "바로 근처가 막혔는지"만 보는 국소적인 정보라, 당장은
        안 막혀 보여도 사실 좁은 구석에 갇히는 길일 수 있다. [15:21]의 flood-fill
        기반 feature는 각 후보 방향이 실제로 얼마나 넓은 공간으로 이어지는지,
        그리고 최소한의 탈출/순환 경로(꼬리에 닿을 수 있는지)가 있는지를 알려줘서
        더 멀리 내다본 판단을 돕는다.
        """
        danger_straight = self._lookahead_dangers(0, FORWARD_LOOKAHEAD)
        danger_right = self._lookahead_dangers(1, SIDE_LOOKAHEAD)
        danger_left = self._lookahead_dangers(-1, SIDE_LOOKAHEAD)

        head = self.snake[0]
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        area_straight, area_right, area_left, tail_straight, tail_right, tail_left = (
            self._reachability_features()
        )

        state = (
            danger_straight
            + danger_right
            + danger_left
            + [dir_u, dir_d, dir_l, dir_r]
            + [
                self.food.y < head.y,  # 먹이가 위쪽
                self.food.y > head.y,  # 먹이가 아래쪽
                self.food.x < head.x,  # 먹이가 왼쪽
                self.food.x > head.x,  # 먹이가 오른쪽
            ]
            + [area_straight, area_right, area_left, tail_straight, tail_right, tail_left]
        )
        # reachable_area가 0~1 사이의 float이라, 나머지(위험/방향/꼬리 도달) boolean까지
        # 전부 float로 반환한다 (dtype=int로 하면 area 값이 0으로 잘려버림에 주의).
        return np.array(state, dtype=np.float32)

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
