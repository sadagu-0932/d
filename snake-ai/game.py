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
GRAY = (120, 120, 120)  # 장애물 색

# ---- 보상 설계 (먹이/사망 두 가지 사건에만 반응) ---------------------------
# 딱 두 가지 사건에만 반응한다: 먹이(사과)를 먹으면 점수를 얻고, 죽으면 점수를
# 잃는다. 그 외 조건(회전 페널티, 빈칸 페널티, self-trap 전용 페널티, 사망 시
# 별도 보너스, 생존 시간에 비례한 가산점 등)은 전혀 없다. 예전에는 몸이 길수록
# 유리하다는 신호로 생존 스텝마다 소량의 길이 보너스(LENGTH_BONUS_COEF)를 얹었지만,
# "생존 시간 보너스를 삭제하자"는 요청에 따라 완전히 제거했다 — 이제 보상은 오직
# 먹이를 먹었는지/죽었는지로만 결정된다.
REWARD_FOOD = 30    # 먹이(사과) 섭취 — "사과를 먹을 때의 보상을 늘리자" 요청에 따라 10 -> 30으로 상향
REWARD_DEATH = -10  # 사망 — 원인(벽/몸통/장애물 충돌, 타임아웃)과 현재 길이에 상관없이 항상 이 고정값

INITIAL_SNAKE_LENGTH = 3  # 게임 시작 시 뱀의 길이 (사람 플레이 / 감상 모드의 표준값)

# 먹이를 못 먹고 맴돌기만 할 때 게임을 강제 종료시키는 기준.
# frame_iteration(누적 스텝 수)이 TIMEOUT_STEPS_PER_SEGMENT * len(snake)를 넘으면 종료.
# (몸이 길어질수록 허용 스텝도 늘어나므로, 결과적으로 '최근 먹이를 못 먹은 시간'과 비슷하게 동작)
# 타임아웃으로 죽든 충돌로 죽든 reward는 동일하게 REWARD_DEATH.
TIMEOUT_STEPS_PER_SEGMENT = 150

# 시계 방향으로 방향을 나열해두면, 오른쪽으로 90도 회전 = 다음 인덱스,
# 왼쪽으로 90도 회전 = 이전 인덱스로 아주 간단하게 계산할 수 있음.
CLOCK_WISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

# Direction -> (dx, dy) 단위 이동량.
DIRECTION_DELTA = {
    Direction.RIGHT: (1, 0),
    Direction.LEFT: (-1, 0),
    Direction.DOWN: (0, 1),
    Direction.UP: (0, -1),
}

# ---- state 설계 (주변 블럭 감지 + 안전한 순환 경로 관련 신호) ----------------
# 한동안 직진 3칸/좌우 2칸 lookahead + flood-fill 기반 reachable_area/can_reach_tail
# (총 21차원)까지 늘렸었는데, "모든 규칙을 초기화하고 주변 블럭만 감지해서 위험/안전을
# 구분하자"는 요청에 따라 전부 걷어내고 가장 단순한 형태로 되돌린 적이 있다: 직진/
# 우회전/좌회전 방향으로 딱 1칸 앞(머리에 바로 붙어있는 블럭)만 보고 위험 여부를
# 판단한다. 그런데 이 반응형(reactive) state만으로는 "같은 자리를 계속 맴도는"
# 제자리 루프에 잘 갇힌다 — 아래 "안전한 순환 경로(Hamiltonian cycle)" 섹션이 그
# 근본 해법이고, 그에 필요한 신호 3개(지름길 존재 여부/순환 경로상 먹이까지 남은
# 거리/몸이 보드를 얼마나 채웠는지)를 state에 추가했다.
# state 벡터 총 차원 = 주변 위험 3(직진/우/좌) + 이동방향 one-hot 4 + 먹이방향 4
#                    + 지름길 존재 1 + 순환경로상 먹이까지 거리(정규화) 1 + 보드 채움률 1
STATE_SIZE = 3 + 4 + 4 + 3

# ---- 학습 시 환경 다양화 (domain randomization) ----------------------------
# reset(randomize=True)에서만 적용됨 — train.py의 학습 루프가 이 값으로 호출하고,
# 사람 플레이(game.py)·play.py(감상)는 항상 randomize=False(표준 보드: 길이 3,
# 장애물 없음)로 호출한다. 매번 똑같이 쉬운 상황(길이 3, 빈 보드)에서만 시작하면
# 위험한 상황(긴 꼬리로 좁은 곳을 헤쳐 나가기, 장애물 피하기)에 실제로 놓이는 경험이
# 적어서 안전하게 대처하는 법을 배우기 어렵다. 그래서 학습 중에는 꼬리 길이를
# 무작위로 주고, 가끔 무작위 장애물도 깔아서 위험 상황에 대처하는 경험 자체를 더
# 자주 겪게 한다.
RANDOM_LENGTH_MIN = 3
RANDOM_LENGTH_MAX = 10   # 머리(그리드 중심, 가로 인덱스 10)에서 왼쪽으로 늘어놓으므로
                         # 그리드 밖으로 나가지 않도록 10을 넘지 않게 제한
OBSTACLE_CHANCE = 0.3    # 학습용 reset마다 이 확률로 장애물을 깔아놓음
OBSTACLE_COUNT_MIN = 3
OBSTACLE_COUNT_MAX = 15

# ---- 안전한 순환 경로 (Hamiltonian cycle) — "같은 자리를 맴도는" 문제의 근본 해법 ----
# 보드의 모든 칸을 정확히 한 번씩 방문하고 처음 자리로 되돌아오는 고정된 순환 경로를
# 미리 하나 계산해 둔다. 이 경로를 그대로 따라가기만 하면(몸이 몇 칸이든) 자기
# 몸통과 절대 부딪히지 않는다 — 몸은 늘 이 순환 경로상 "가장 최근 N칸"이기 때문에,
# 다음 칸은 몸이 차지한 칸들보다 항상 훨씬 앞쪽(순환 경로 기준)에 있다.
# 신경망은 매 스텝 "이 안전한 경로를 그대로 따라갈지, 아니면 경로를 벗어나 먹이로
# 직행하는 지름길을 탈지"만 결정한다(agent.py의 resolve_action 참고) — 지름길은
# 1칸 앞만 보고 당장은 안전해 보여도, 나중에 꼬리에 막혀 죽는 함정을 못 거를 수
# 있다는 위험을 감수하는 선택이다.
def _build_hamiltonian_cycle(grid_size):
    """
    grid_size x grid_size 보드 전체를 한 번씩 방문하는 고정 순환 경로를 칸 좌표
    (col, row) 리스트로 반환한다 (리스트의 마지막 칸 다음은 다시 첫 칸으로 이어짐).
    grid_size는 짝수여야 하는 구성법이다 — 1열은 위에서 아래로 내려오는 "복귀
    통로"로 쓰고, 나머지 열들은 한 칸씩 지그재그로 오르내리며 보드를 채운다.
    """
    w = h = grid_size
    cells = [(x, 0) for x in range(w)]            # 0행: 왼쪽 -> 오른쪽
    cells += [(w - 1, y) for y in range(1, h)]     # 마지막 열: 위 -> 아래
    for c in range(w - 2, 0, -1):                  # 나머지 열(w-2 ~ 1)을 지그재그로
        rows = range(h - 1, 0, -1) if c % 2 == 0 else range(1, h)
        cells += [(c, y) for y in rows]
    cells += [(0, y) for y in range(h - 1, 0, -1)]  # 0열: 아래 -> 위 (처음 칸으로 복귀)
    return cells


_CYCLE_CELLS = _build_hamiltonian_cycle(GRID_SIZE)
HAMILTONIAN_CYCLE = [Point(cx * BLOCK_SIZE, cy * BLOCK_SIZE) for cx, cy in _CYCLE_CELLS]
CYCLE_INDEX = {pt: i for i, pt in enumerate(HAMILTONIAN_CYCLE)}
CYCLE_NEXT = {pt: HAMILTONIAN_CYCLE[(i + 1) % len(HAMILTONIAN_CYCLE)] for i, pt in enumerate(HAMILTONIAN_CYCLE)}


class SnakeGameAI:
    """
    20x20 grid 스네이크 게임 환경 (Gym 스타일).

    Parameters
    ----------
    w, h : 게임 화면 크기(px). 기본값은 20x20 grid에 맞춘 400x400.
    render : True면 pygame 창을 띄워 화면을 그림. 학습 속도를 높이고 싶다면 False.
    speed : 초당 프레임 수(FPS). render=True일 때만 의미가 있음.
    """

    def __init__(self, w=WIDTH, h=HEIGHT, render=True, speed=40):
        self.w = w
        self.h = h
        self.render_enabled = render
        self.speed = speed

        if self.render_enabled:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake AI")
        else:
            self.display = None
        self.clock = pygame.time.Clock()

        self.obstacles = []  # reset()에서 매번 다시 채워지지만, 최초 reset() 전에도 존재해야 함
        self.reset()

    # ------------------------------------------------------------------
    # Gym 스타일 인터페이스
    # ------------------------------------------------------------------
    def reset(self, randomize=False):
        """
        게임 상태를 초기화하고 초기 state를 반환.

        randomize=True면(학습 중에만 사용) 꼬리 길이를 무작위로 주고, 일정 확률로
        무작위 장애물도 깔아서 위험한 상황에 더 자주 놓이게 한다. randomize=False
        (기본값, 사람 플레이/감상 모드)면 항상 길이 3·장애물 없는 표준 보드로 시작한다.
        """
        self.direction = Direction.RIGHT

        length = INITIAL_SNAKE_LENGTH
        if randomize:
            length = random.randint(RANDOM_LENGTH_MIN, RANDOM_LENGTH_MAX)

        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            Point(self.head.x - i * BLOCK_SIZE, self.head.y)
            for i in range(length)
        ]

        self.score = 0
        self.obstacles = []
        if randomize and random.random() < OBSTACLE_CHANCE:
            self._place_obstacles()

        self.food = None
        self._place_food()
        self.frame_iteration = 0

        return self._get_state()

    def step(self, action):
        """
        action : 길이 3의 one-hot 리스트/배열 [직진, 우회전, 좌회전]

        Returns
        -------
        state : np.ndarray, shape (STATE_SIZE,)
        reward : float
        done : bool
        score : int

        reward 구성 (딱 두 가지 사건에만 반응 — 생존 시간 보너스 없음):
          - 먹이(사과) 섭취: REWARD_FOOD
          - 사망(충돌/타임아웃 불문): REWARD_DEATH(고정, 길이/score와 무관)
          - 그 외 그냥 이동한 스텝: 0
        """
        self.frame_iteration += 1

        # 창을 닫으면 즉시 종료할 수 있도록 이벤트 처리
        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 1) 행동에 따라 이동
        self._move(action)
        self.snake.insert(0, self.head)

        # 2) 종료 판정 — 충돌하거나, 너무 오랫동안 먹이를 못 먹으면(맴돌기 방지) 게임 종료.
        # 원인(충돌/타임아웃)이나 현재 길이와 상관없이 reward는 항상 REWARD_DEATH로 고정.
        game_over = False
        if self.is_collision() or self.frame_iteration > TIMEOUT_STEPS_PER_SEGMENT * len(self.snake):
            game_over = True
            reward = REWARD_DEATH
            return self._get_state(), reward, game_over, self.score

        # 3) 먹이 섭취 여부에 따른 보상
        ate_food = self.head == self.food
        if ate_food:
            self.score += 1
            reward = REWARD_FOOD
            self._place_food()
        else:
            self.snake.pop()  # 먹이를 못 먹었으면 꼬리를 잘라 길이를 유지
            reward = 0

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
        if self.food in self.snake or self.food in self.obstacles:  # 뱀 몸통/장애물 위에 놓였다면 다시 뽑기
            self._place_food()

    def _place_obstacles(self):
        """학습용 랜덤 장애물을 뱀 몸통과 겹치지 않게 깔아놓는다 (self.obstacles에 저장)."""
        count = random.randint(OBSTACLE_COUNT_MIN, OBSTACLE_COUNT_MAX)
        max_x = (self.w - BLOCK_SIZE) // BLOCK_SIZE
        max_y = (self.h - BLOCK_SIZE) // BLOCK_SIZE
        attempts = 0
        while len(self.obstacles) < count and attempts < count * 20:
            attempts += 1
            pt = Point(random.randint(0, max_x) * BLOCK_SIZE, random.randint(0, max_y) * BLOCK_SIZE)
            if pt in self.snake or pt in self.obstacles:
                continue
            self.obstacles.append(pt)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # 벽 충돌
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # 몸통 충돌 (머리 자기 자신은 제외)
        if pt in self.snake[1:]:
            return True
        # 장애물 충돌 (학습 중 randomize=True일 때만 존재)
        if pt in self.obstacles:
            return True
        return False

    def _action_to_direction(self, action):
        """상대적 action([직진, 우회전, 좌회전])을 현재 방향 기준 절대 Direction으로 변환."""
        idx = CLOCK_WISE.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            return CLOCK_WISE[idx]              # 직진: 방향 유지
        elif np.array_equal(action, [0, 1, 0]):
            return CLOCK_WISE[(idx + 1) % 4]     # 우회전 (시계 방향)
        else:  # [0, 0, 1]
            return CLOCK_WISE[(idx - 1) % 4]     # 좌회전 (반시계 방향)

    def _next_point(self, action):
        """지금 머리 위치에서, action을 수행했을 때(이동시키지 않고) 도달할 지점만 계산."""
        dx, dy = DIRECTION_DELTA[self._action_to_direction(action)]
        return Point(self.head.x + dx * BLOCK_SIZE, self.head.y + dy * BLOCK_SIZE)

    def _direction_to(self, point):
        """머리 바로 옆(1칸) point로 가는 절대 Direction을 반환 (인접하지 않으면 에러)."""
        dx = (point.x - self.head.x) // BLOCK_SIZE
        dy = (point.y - self.head.y) // BLOCK_SIZE
        for d, delta in DIRECTION_DELTA.items():
            if delta == (dx, dy):
                return d
        raise ValueError(f"{point}는 머리({self.head})와 인접하지 않습니다")

    def _move(self, action):
        """상대적 action([직진, 우회전, 좌회전])을 받아 self.direction / self.head를 갱신."""
        self.direction = self._action_to_direction(action)
        dx, dy = DIRECTION_DELTA[self.direction]
        self.head = Point(self.head.x + dx * BLOCK_SIZE, self.head.y + dy * BLOCK_SIZE)

    def cycle_action(self):
        """
        안전한 순환 경로(HAMILTONIAN_CYCLE)상 다음 칸으로 가는 상대 action을 반환.
        몸이 아직 순환 경로와 완전히 겹치지 않은 상태(에피소드 시작 직후 등)라
        그 다음 칸이 당장 위험하면, 경로를 고집하지 않고 1칸 앞만 봐도 안전한
        다른 방향으로 즉시 피한다 — 몇 스텝만 지나면 몸 전체가 자연히 순환
        경로와 다시 일치하게 된다(죽는 것보다 잠깐 경로를 벗어나는 게 낫다).
        """
        action = self.direction_to_action(self._direction_to(CYCLE_NEXT[self.head]))
        if not self.is_collision(self._next_point(action)):
            return action
        for fallback in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            if not self.is_collision(self._next_point(fallback)):
                return fallback
        return action  # 셋 다 위험하면(사실상 사망 확정) 그냥 원래 action 유지

    def shortcut_action(self):
        """
        순환 경로를 벗어나 먹이로 더 가까워지는 지름길을 찾는다. 직진/우회전/좌회전
        중 "1칸 앞이 당장 위험하지 않은" 후보들 가운데 먹이까지 맨해튼 거리가 가장
        많이 줄어드는 쪽을 고른다. 그런 후보가 없으면 (None, False)를 반환한다 —
        호출부(agent.py의 resolve_action)는 이 경우 cycle_action()으로 폴백해야 한다.

        주의: "당장 위험하지 않음"은 1칸 앞만 보는 판정이라, 지금은 안전해 보여도
        나중에(특히 몸이 길어졌을 때) 꼬리에 막혀 죽을 수 있는 함정까지는 걸러내지
        못한다 — 지름길에는 그 위험이 실제로 남아있다.
        """
        current_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        candidates = []
        for action in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            pt = self._next_point(action)
            if self.is_collision(pt):
                continue
            dist = abs(pt.x - self.food.x) + abs(pt.y - self.food.y)
            candidates.append((dist, action))

        if not candidates:
            return None, False

        candidates.sort(key=lambda c: c[0])
        best_dist, best_action = candidates[0]
        if best_dist >= current_dist:
            return None, False
        return best_action, True

    def _relative_direction(self, turn):
        """turn: 0=직진, +1=우회전, -1=좌회전 (CLOCK_WISE 기준) -> 절대 Direction 반환."""
        idx = CLOCK_WISE.index(self.direction)
        return CLOCK_WISE[(idx + turn) % 4]

    def _danger_in_direction(self, turn):
        """
        현재 진행 방향 기준 turn(0=직진, +1=우회전, -1=좌회전) 쪽으로, 머리에 바로
        붙어있는 1칸(주변 블럭)이 위험(벽/몸통/장애물)한지 여부.
        """
        dx, dy = DIRECTION_DELTA[self._relative_direction(turn)]
        head = self.snake[0]
        return self.is_collision(Point(head.x + dx * BLOCK_SIZE, head.y + dy * BLOCK_SIZE))

    def _get_state(self):
        """
        STATE_SIZE(14)차원 state 벡터를 계산해서 반환.

        [0]    직진 방향으로 바로 1칸 앞(주변 블럭)이 위험(충돌)한지
        [1]    우회전 방향으로 바로 1칸 옆이 위험한지
        [2]    좌회전 방향으로 바로 1칸 옆이 위험한지
        [3:7]  현재 이동 방향 one-hot (상, 하, 좌, 우)
        [7:11] 먹이의 상대적 방향 (상, 하, 좌, 우) - boolean
        [11]   지금 안전한 지름길(shortcut_action)이 실제로 있는지
        [12]   순환 경로(HAMILTONIAN_CYCLE)를 그대로 따라갈 때 먹이까지 남은 거리
               (전체 경로 길이로 정규화, 0~1)
        [13]   몸이 보드 전체 칸 중 몇 %를 채웠는지 (len(snake) / 전체 칸 수, 0~1)

        [0:3]는 "머리 바로 옆 블럭이 막혔는지"만 보는 가장 단순한 위험 감지다.
        [11:14]은 순환 경로 기반 지름길 판단에 필요한 신호 — 지름길이 없으면 애초에
        탈 수 없고, 순환 경로로 먹이까지 한참 남았을수록/보드가 꽉 찰수록 지름길의
        가치(또는 위험)가 달라진다는 걸 신경망이 학습할 수 있게 한다.
        """
        danger_straight = self._danger_in_direction(0)
        danger_right = self._danger_in_direction(1)
        danger_left = self._danger_in_direction(-1)

        head = self.snake[0]
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        _, shortcut_exists = self.shortcut_action()
        cycle_len = len(HAMILTONIAN_CYCLE)
        # 벽 충돌로 죽은 바로 다음 순간엔 head가 보드 밖으로 나가 있을 수 있다(step()이
        # 사망 처리 직전에 한 번 더 _get_state()를 호출함) — 그 경우 순환 경로상의
        # 위치가 없으므로 의미 없는 값(1.0)으로 둔다. 어차피 종료 상태라 학습에 영향 없음.
        head_cycle_idx = CYCLE_INDEX.get(head)
        food_cycle_idx = CYCLE_INDEX.get(self.food)
        if head_cycle_idx is None or food_cycle_idx is None:
            cycle_gap_norm = 1.0
        else:
            cycle_gap_norm = ((food_cycle_idx - head_cycle_idx) % cycle_len) / cycle_len
        fill_ratio = len(self.snake) / (GRID_SIZE * GRID_SIZE)

        state = [
            danger_straight, danger_right, danger_left,
            dir_u, dir_d, dir_l, dir_r,
            self.food.y < head.y,  # 먹이가 위쪽
            self.food.y > head.y,  # 먹이가 아래쪽
            self.food.x < head.x,  # 먹이가 왼쪽
            self.food.x > head.x,  # 먹이가 오른쪽
            shortcut_exists,
            cycle_gap_norm,
            fill_ratio,
        ]
        return np.array(state, dtype=np.float32)

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.obstacles:
            pygame.draw.rect(self.display, GRAY, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

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
