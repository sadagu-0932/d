"""
model.py
--------
DQN에 사용되는 신경망(Q-Network)과 학습기(Trainer)를 정의합니다.

- Linear_QNet : state(game.py의 STATE_SIZE) -> hidden(256) -> action(3) 구조의 간단한 MLP.
- QTrainer    : 리플레이 버퍼에서 뽑은 (s, a, r, s', done) 배치로
                벨만 방정식(Bellman equation) 기반 손실을 계산하고 역전파합니다.
                학습을 안정화하기 위해 일정 주기로 가중치를 복사해오는
                '타겟 네트워크(target network)'를 사용합니다.
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Linear_QNet(nn.Module):
    """state -> 각 action의 Q값을 예측하는 완전연결 신경망 (입력 크기는 호출부에서 지정)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)  # 출력층은 활성화 함수 없이 Q값을 그대로 반환
        return x

    def save(self, file_name: str = "best.pth", folder: str = "model"):
        """현재 신경망 가중치를 <folder>/<file_name> (기본: model/best.pth)로 저장."""
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, file_name)
        torch.save(self.state_dict(), file_path)

    def load(self, file_path: str, device: str = "cpu"):
        """
        저장된 가중치를 불러와 현재 모델에 적용하고 평가 모드로 전환.

        state 설계(game.py의 STATE_SIZE)가 바뀌면 신경망 입력 크기도 함께 바뀌므로,
        예전 STATE_SIZE로 학습된 체크포인트는 더 이상 그대로 불러올 수 없다.
        PyTorch의 원본 에러(size mismatch ...)만으로는 원인이 뭔지 알기 어려우니,
        여기서 잡아서 "체크포인트가 지금 state 설계와 안 맞으니 새로 학습해야 한다"는
        걸 명확히 알려주는 메시지로 바꿔서 다시 던진다.
        """
        state_dict = torch.load(file_path, map_location=device)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            raise RuntimeError(
                f"'{file_path}' 체크포인트를 불러오지 못했습니다: 신경망 입력 크기가 "
                "지금 코드의 state 설계(game.py의 STATE_SIZE)와 맞지 않습니다. "
                "state를 확장/변경한 뒤 예전 체크포인트를 그대로 불러오려는 경우일 "
                "가능성이 높습니다 — 이 체크포인트는 더 이상 호환되지 않으니, "
                "train.py를 다시 실행해서 처음부터 새로 학습시켜 주세요.\n"
                f"(원본 에러: {e})"
            ) from e
        self.eval()


class QTrainer:
    """
    Q-Learning(DQN) 파라미터 업데이트를 담당하는 클래스.

    - self.model        : 실제로 행동을 선택하는 데 사용되며 계속 학습되는 네트워크(policy network)
    - self.target_model : self.model의 가중치를 일정 주기로 복사해오는 '고정된' 네트워크

    타겟 Q값 계산에 target_model을 쓰면, 학습 도중 타겟 값 자체가 매 스텝
    흔들리는 현상(moving target problem)이 줄어들어 학습이 더 안정적으로 수렴합니다.
    """

    def __init__(self, model: nn.Module, lr: float, gamma: float,
                 target_update_freq: int = 100):
        self.lr = lr
        self.gamma = gamma  # 미래 보상 할인율

        self.model = model
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()

        self.target_update_freq = target_update_freq
        self._train_step_count = 0

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        """target_model의 가중치를 policy 모델(self.model)과 동일하게 하드 카피."""
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        """
        하나의 배치(또는 단일 transition)에 대해 한 번의 경사하강 업데이트를 수행.

        state / next_state : (batch, 11) 또는 (11,)
        action              : (batch, 3) 또는 (3,)  - one-hot
        reward              : (batch,) 또는 스칼라
        done                : (batch,) 또는 bool
        """
        # 배치 학습 시 (numpy array들의) tuple을 그대로 torch.tensor()에 넘기면
        # 매우 느리다는 경고가 뜨므로, 먼저 np.array로 한 번에 합친 뒤 변환한다.
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # 단일 transition(1차원)이면 배치 차원을 추가해 (1, N) 형태로 맞춤
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1) 현재 상태에서 policy 네트워크가 예측한 Q값
        pred = self.model(state)
        target = pred.clone()

        # 2) 벨만 방정식으로 타겟 Q값 계산: Q_new = r + gamma * max(Q_target(s'))
        #    (단, 게임이 끝났다면(done=True) 미래 보상 없이 r만 사용)
        with torch.no_grad():
            next_q = self.target_model(next_state)

        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(next_q[idx])

            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = q_new

        # 3) 손실 계산 및 역전파
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        # 4) 일정 스텝마다 target 네트워크를 policy 네트워크로 동기화 (학습 안정화)
        self._train_step_count += 1
        if self._train_step_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()
