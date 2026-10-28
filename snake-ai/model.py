"""
model.py
--------
신경망(정책 네트워크) 구조만 정의합니다.

- Linear_QNet : state(game.py의 STATE_SIZE) -> hidden(256) -> action(3) 구조의 간단한 MLP.
  출력은 Q값이 아니라 "각 행동(직진/우회전/좌회전)의 선호도 점수"로 쓰입니다 — 가장
  점수가 높은 행동을 그대로 선택(argmax)합니다.

이전에는 이 구조를 경사하강(Double DQN, QTrainer)으로 학습시켰지만, "4개 부모를
뽑아 섞고 살짝 변형해가며 학습하자"는 요청에 따라 학습 방식을 유전 알고리즘으로
교체했습니다 (agent.py의 Population 참고). 그래서 역전파용 QTrainer는 더 이상
쓰이지 않아 제거했고, 신경망 구조 자체(Linear_QNet)만 그대로 남아있습니다 — 가중치를
"학습"하는 방법이 바뀐 것이고, 신경망의 생김새(레이어 수·뉴런 수)는 여전히 우리가
고정해 둔 값입니다(11 -> 256 -> 3). 신경망 구조 자체를 AI가 알아서 바꾸는 것(토폴로지
탐색/NEAT류)은 하지 않습니다 — 훨씬 큰 별도 작업입니다.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    """state -> 각 action의 선호도 점수를 출력하는 완전연결 신경망 (입력 크기는 호출부에서 지정)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)  # 출력층은 활성화 함수 없이 선호도 점수를 그대로 반환
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
