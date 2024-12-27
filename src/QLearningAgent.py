import heapq

import numpy as np
import random
from collections import defaultdict
import concurrent.futures
from concurrent.futures import as_completed
import pickle
import bz2
import sys

from src.logic_game import TicTacToe


class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.8, max_memory_mb=6*1024):
        self.q_table = defaultdict(float)  # Q-values for state-action pairs
        self.alpha = learning_rate         # Learning rate
        self.gamma = discount_factor       # Discount factor
        self.state_count = defaultdict(int)  # Счетчик состояний
        self.epsilon = epsilon             # Exploration rate
        self.max_memory_bytes = max_memory_mb * 1024 * 1024  # Max memory in bytes
        self.current_memory_usage = 0  # Кэширование размера памяти

    def choose_action(self, state, available_actions):
        # Уменьшение epsilon
        self.epsilon = max(0.01, self.epsilon * 0.99)

        # if np.random.rand() < self.epsilon:
        #     return random.choice(available_actions)  # Random action
        # q_values = [self.q_table[(state, a)] for a in available_actions]
        # return available_actions[np.argmax(q_values)]  # Greedy action

        if np.random.rand() < self.epsilon:
            return random.choice(available_actions)  # Random action

            # Boltzmann Sampling вместо жадного выбора
        q_values = [self.q_table[(state, a)] for a in available_actions]
        probs = np.exp(q_values) / np.sum(np.exp(q_values))  # Softmax
        return np.random.choice(available_actions, p=probs)


    # def update_q_value(self, state, action, reward, next_state, next_actions):
    #     future_q = max([self.q_table[(next_state, a)] for a in next_actions], default=0)
    #     current_q = self.q_table[(state, action)]
    #     self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * future_q - current_q)

    # def update_q_value(self, state, action, reward, next_state, next_actions):
    #     future_q = max([self.q_table[(next_state, a)] for a in next_actions], default=0)
    #     current_q = self.q_table[(state, action)]
    #     self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * future_q - current_q)
    #
    #     # Уменьшение Q-значений при частом посещении
    #     self.state_count[(state, action)] += 1
    #     decay_factor = 1 / (1 + self.state_count[(state, action)])
    #     self.q_table[(state, action)] *= decay_factor
    #
    #     # Check memory usage and prune if necessary
    #     self._check_memory_limit()


    def update_q_value(self, state, action, reward, next_state, next_actions):
        future_q = max((self.q_table[(next_state, a)] for a in next_actions), default=0)
        current_q = self.q_table[(state, action)]
        updated_q = current_q + self.alpha * (reward + self.gamma * future_q - current_q)

        # Уменьшение Q-значений при частом посещении
        self.state_count[(state, action)] += 1
        decay_factor = 1 / (1 + self.state_count[(state, action)])
        self.q_table[(state, action)] = updated_q * decay_factor

        # Обновляем размер памяти
        self._update_memory_usage(state, action, updated_q)
        self._check_memory_limit()

    def _update_memory_usage(self, state, action, new_value):
        """Обновляет общий размер памяти для Q-таблицы."""
        key_size = sys.getsizeof((state, action))
        value_size = sys.getsizeof(new_value)
        old_value_size = sys.getsizeof(self.q_table[(state, action)])
        self.current_memory_usage += key_size + value_size - old_value_size

    def _check_memory_limit(self):
        """Проверяет лимит памяти и удаляет наименее значимые элементы."""
        if self.current_memory_usage > self.max_memory_bytes:
            # Создаем кучу на основе абсолютного значения Q
            q_items = [(abs(v), k) for k, v in self.q_table.items()]
            heapq.heapify(q_items)

            while self.current_memory_usage > self.max_memory_bytes and q_items:
                _, key_to_remove = heapq.heappop(q_items)
                key_size = sys.getsizeof(key_to_remove)
                value_size = sys.getsizeof(self.q_table[key_to_remove])
                self.current_memory_usage -= key_size + value_size
                del self.q_table[key_to_remove]


    def save(self, file_path):
        with bz2.BZ2File(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with bz2.BZ2File(file_path, 'rb') as f:
            return pickle.load(f)


    def merge_agents(*agents):
        merged_q_table = defaultdict(float)
        counts = defaultdict(int)  # Счетчик для усреднения значений

        # Проходимся по каждому агенту и их Q-таблицам
        for agent in agents:
            for (state, action), q_value in agent.q_table.items():
                merged_q_table[(state, action)] += q_value
                counts[(state, action)] += 1

        # Усредняем Q-значения
        for key in merged_q_table:
            merged_q_table[key] /= counts[key]

        # Создаем нового агента с объединенной Q-таблицей
        merged_agent = QLearningAgent()
        merged_agent.q_table = merged_q_table
        return merged_agent


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
        next_state = (tuple(next_state), tuple(game.won_fields))

        if done:
            winner = game.get_winner()
            reward = 5 if winner == player else -10 if winner != 0 else -0.5
            reward += sum(1 for x in game.won_fields if x == player) * 2
            reward -= sum(1 for x in game.won_fields if x == 3-player) * 1
            current_agent.update_q_value(state, action, reward, next_state, [])

            player = 3 - player
            reward = 5 if winner == player else -10 if winner != 0 else -0.5
            reward += sum(1 for x in game.won_fields if x == player) * 2
            reward -= sum(1 for x in game.won_fields if x == 3-player) * 1
            opponent_agent.update_q_value(state, action, reward, next_state, [])
        else:
            next_actions = game.available_actions()
            current_agent.update_q_value(state, action, 0, next_state, next_actions)
            state = next_state

        current_agent, opponent_agent = opponent_agent, current_agent
        player = 3 - player

    with open(f"log1.txt", 'a') as f:  # Логи для каждого агента
        f.write(f"Player: {player}, Winner: {game.get_winner()}\n")

    return agent1, agent2


# def parallel_training(episodes=10000, num_workers=4, agent1=QLearningAgent(), agent2=QLearningAgent()):
#
#     with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
#         futures = [executor.submit(train_single_game, agent1, agent2) for _ in range(episodes)]
#         for i, future in enumerate(as_completed(futures), start=1):
#             agent1, agent2 = future.result()
#             if i % 100000 == 0:
#                 agent1.save(f"agent1_{600000+i}.pkl")
#                 agent2.save(f"agent2_{600000+i}.pkl")
#
#     # Объединяем знания агентов после завершения всех эпизодов
#     merged_agent = QLearningAgent.merge_agents(agent1, agent2)
#     merged_agent.save("merged_agent.pkl")
#
#     return merged_agent


def sequential_training(episodes=10000, agent1=QLearningAgent(), agent2=QLearningAgent(), start_i=0):
    print("start")
    for episode in range(episodes):
        agent1, agent2 = train_single_game(agent1, agent2)

        if episode % 1000 == 0 and episodes != 0:
            agent1.save(fr"../AI/agent1_{episode+start_i}.pkl")
            agent2.save(f"../AI/agent2_{episode+start_i}.pkl")
            print(f"episode: {episode}")

    merged_agent = QLearningAgent.merge_agents(agent1, agent2)
    merged_agent.save("merged_agent.pkl")

    agent1.save(fr"../AI/agent1_{episodes + start_i}.pkl")
    agent2.save(f"../AI/agent2_{episodes + start_i}.pkl")

    return merged_agent



if __name__ == '__main__':
    # from multiprocessing import freeze_support
    #
    # freeze_support()

    # agent1 = QLearningAgent()
    # agent2 = QLearningAgent()
    agent1 = QLearningAgent.load("../AI/agent1_7000.pkl")
    agent2 = QLearningAgent.load("../AI/agent2_7000.pkl")

    # agent = parallel_training(episodes=1000000, num_workers=4, agent1=current_agent1, agent2=current_agent2)
    agent = sequential_training(episodes=10000, agent1=agent1, agent2=agent2, start_i=7000)

    # merged_agent = QLearningAgent.merge_agents(current_agent1, current_agent2)
    # merged_agent.save("merged_agent.pkl")

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