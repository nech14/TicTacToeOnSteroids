
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Вход 9 (3x3 клетки), скрытый слой на 128
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, input_size)  # Выход 9 (Q-значения для всех 9 клеток)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

