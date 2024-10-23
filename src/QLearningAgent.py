
import numpy as np
import random
from collections import defaultdict
import concurrent.futures
from concurrent.futures import as_completed
import pickle

from src.logic_game import TicTacToe


class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = defaultdict(float)  # Q-values for state-action pairs
        self.alpha = learning_rate         # Learning rate
        self.gamma = discount_factor       # Discount factor
        self.epsilon = epsilon             # Exploration rate

    def choose_action(self, state, available_actions):
        if np.random.rand() < self.epsilon:
            return random.choice(available_actions)  # Random action
        q_values = [self.q_table[(state, a)] for a in available_actions]
        return available_actions[np.argmax(q_values)]  # Greedy action

    def update_q_value(self, state, action, reward, next_state, next_actions):
        future_q = max([self.q_table[(next_state, a)] for a in next_actions]) if next_actions else 0
        current_q = self.q_table[(state, action)]
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * future_q - current_q)

    # Метод для сохранения агента в файл
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    # Метод для загрузки агента из файла
    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def train_single_game(agent1, agent2):
    game = TicTacToe()
    state = tuple(game.reset())
    done = False
    current_agent = agent1
    opponent_agent = agent2
    player = 1

    while not done:
        available_actions = game.available_actions()
        action = current_agent.choose_action(state, available_actions)
        done, next_state = game.step(action, player)
        next_state = tuple(next_state)

        if done:
            winner = game.get_winner()
            reward = 1 if winner == player else -1 if winner != 0 else 0
            current_agent.update_q_value(state, action, reward, next_state, [])
            opponent_agent.update_q_value(state, action, -reward, next_state, [])
        else:
            next_actions = game.available_actions()
            current_agent.update_q_value(state, action, 0, next_state, next_actions)
            state = next_state

        current_agent, opponent_agent = opponent_agent, current_agent
        player = 3 - player

    with open(f"log.txt", 'a') as f:  # Логи для каждого агента
        f.write(f"Player: {player}, Winner: {game.get_winner()}\n")

    return agent1, agent2


def parallel_training(episodes=10000, num_workers=4):
    agent1 = QLearningAgent()
    agent2 = QLearningAgent()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(train_single_game, agent1, agent2) for _ in range(episodes)]
        for i, future in enumerate(as_completed(futures), start=1):
            agent1, agent2 = future.result()
            if i % 100000 == 0:
                agent1.save(f"agent1_{i}.pkl")
                agent2.save(f"agent2_{i}.pkl")

    return agent1, agent2


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    agent1, agent2 = parallel_training(episodes=100000, num_workers=4)

    # game = TicTacToe()
    # state = tuple(game.reset())
    # done = False
    #
    # current_agent = QLearningAgent.load("agent1_10000.pkl")
    # player = 1
    #
    # while not done:
    #     print(game.available_actions())
    #     game.print_board()
    #
    #     if player == 1:
    #         available_actions = game.available_actions()
    #         action = current_agent.choose_action(state, available_actions)
    #         done, next_state = game.step(action, player)
    #
    #     else:
    #         x, y = input().split(' ')
    #         done = game.human_step(int(y), int(x), player)[0]
    #
    #     print(f"end game: {done}")
    #
    #
    #     player = 3 - player
    #
    # print(f"Winner: {game.get_winner()}")