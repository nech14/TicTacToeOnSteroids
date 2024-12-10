
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
        next_state = tuple(next_state)

        if done:
            winner = game.get_winner()
            reward = 1 if winner == player else -1.5 if winner != 0 else -1
            reward += sum(1 for x in game.won_fields if x == player) * 0.2
            reward -= sum(1 for x in game.won_fields if x != player) * 0.1
            current_agent.update_q_value(state, action, reward, next_state, [])

            player = 3 - player
            reward = 1 if winner == player else -1.5 if winner != 0 else -1
            reward += sum(1 for x in game.won_fields if x == player) * 0.2
            reward -= sum(1 for x in game.won_fields if x != player) * 0.1
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
    for episode in range(episodes):
        agent1, agent2 = train_single_game(agent1, agent2)

        if episode % 1000 == 0:
            agent1.save(fr"../AI/agent1_{episode+start_i}.pkl")
            agent2.save(f"../AI/agent2_{episode+start_i}.pkl")
            print(f"episode: {episode}")

    merged_agent = QLearningAgent.merge_agents(agent1, agent2)
    merged_agent.save("merged_agent.pkl")
    return merged_agent



if __name__ == '__main__':
    # from multiprocessing import freeze_support
    #
    # freeze_support()

    agent1 = QLearningAgent()
    agent2 = QLearningAgent()
    # agent1 = QLearningAgent.load("../AI/agent1_9000.pkl")
    # agent2 = QLearningAgent.load("../AI/agent2_9000.pkl")

    # agent = parallel_training(episodes=1000000, num_workers=4, agent1=current_agent1, agent2=current_agent2)
    agent = sequential_training(episodes=10000, agent1=agent1, agent2=agent2, start_i=0)

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