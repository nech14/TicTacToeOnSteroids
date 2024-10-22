
from src.DQN import DQN
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(self, player, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.player = player
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model_81 = DQN(81).to(device)  # Модель для первого этапа (81 позиция)
        self.model_9 = DQN(9).to(device)  # Модель для последующих этапов (9 позиций)
        self.optimizer_81 = optim.Adam(self.model_81.parameters(), lr=lr)
        self.optimizer_9 = optim.Adam(self.model_9.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state, available_actions, stage='81_positions'):
        if np.random.rand() < self.epsilon:
            return random.choice(available_actions)  # Exploration
        state = torch.FloatTensor(state).to(device)
        if stage == '81_positions':
            q_values = self.model_81(state)
        else:
            q_values = self.model_9(state)

        q_values = q_values.cpu().detach().numpy()
        q_values = np.where(np.isin(range(len(q_values)), available_actions, invert=True), -np.inf, q_values)
        return np.argmax(q_values)

    def update_q_value(self, state, action, reward, next_state, done, stage='81_positions'):
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)

        # Преобразуем целевое значение в Float
        target = torch.tensor(reward, dtype=torch.float32).to(device)

        if not done:
            if stage == '81_positions':
                target += self.gamma * torch.max(self.model_81(next_state)).item()
            else:
                target += self.gamma * torch.max(self.model_9(next_state)).item()

        if stage == '81_positions':
            current_q = self.model_81(state)[action]
            loss = self.loss_fn(current_q, target)  # Приведение loss к float
            self.optimizer_81.zero_grad()
        else:
            current_q = self.model_9(state)[action]
            loss = self.loss_fn(current_q, target)  # Приведение loss к float
            self.optimizer_9.zero_grad()

        loss.backward()

        if stage == '81_positions':
            self.optimizer_81.step()
        else:
            self.optimizer_9.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
