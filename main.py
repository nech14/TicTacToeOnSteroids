import pygame

from src.DQNAgent import DQNAgent
from src.QLearningAgent import QLearningAgent
from src.gui import TicTacToeGUI, font, BLACK, SCREEN_SIZE, screen
from src.logic_game import TicTacToe
import torch
import multiprocessing

# Сохранение модели
def save_models(agent_X, agent_O):
    torch.save(agent_X.model_81.state_dict(), 'agent_X_model_81.pth')
    torch.save(agent_X.model_9.state_dict(), 'agent_X_model_9.pth')
    torch.save(agent_O.model_81.state_dict(), 'agent_O_model_81.pth')
    torch.save(agent_O.model_9.state_dict(), 'agent_O_model_9.pth')

# Загрузка модели
def load_models(agent_X, agent_O):
    agent_X.model_81.load_state_dict(torch.load('agent_X_model_81.pth'))
    agent_X.model_9.load_state_dict(torch.load('agent_X_model_9.pth'))
    agent_O.model_81.load_state_dict(torch.load('agent_O_model_81.pth'))
    agent_O.model_9.load_state_dict(torch.load('agent_O_model_9.pth'))


def train(agent_id="agen_1", episodes=1000000):
    game = TicTacToe()
    agent_X = DQNAgent(player=1)
    agent_O = DQNAgent(player=2)

    player1 = agent_X
    player2 = agent_O

    for episode in range(episodes):
        state = game.reset()
        done = False

        while not done:
            # Ход агента X
            available_actions = game.available_actions()
            action_X = player1.choose_action(state, available_actions)
            done, new_state = game.step(action_X, player1.player)
            winner = game.get_winner() if done else 0
            reward_X = 1 if winner == player1.player else -1 if winner == agent_O.player else 0

            player1.update_q_value(state, action_X, reward_X, new_state, done)
            state = new_state
            if done:
                break

            # Ход агента O
            available_actions = game.available_actions()
            action_O = player2.choose_action(state, available_actions)
            done, new_state = game.step(action_O, player2.player)
            winner = game.get_winner() if done else 0
            reward_O = 1 if winner == player2.player else -1 if winner == agent_X.player else 0
            player2.update_q_value(state, action_O, reward_O, new_state, done)
            state = new_state

        with open(f"log_{agent_id}.txt", 'a') as f:  # Логи для каждого агента
            f.write(f"Episode: {episode}, Winner: {game.get_winner()}\n")

        if episode % 100 == 0:
            print(f"Agent {agent_id}: Episode {episode}, Epsilon X: {player1.epsilon}, Epsilon O: {player2.epsilon}")

        if episode % 2 == 0:
            player1 = agent_X
            player2 = agent_O
        else:
            player2 = agent_X
            player1 = agent_O


    save_models(agent_X, agent_O)


#
# if __name__ == '__main__':
#     print("start")
#     num_agents = 4  # Количество агентов для параллельного обучения
#     episodes_per_agent = 2500  # Количество эпизодов для каждого агента
#
#     processes = []
#     for i in range(num_agents):
#         p = multiprocessing.Process(target=train, args=(i, episodes_per_agent))
#         processes.append(p)
#         p.start()
#
#     for p in processes:
#         p.join()  # Ждем завершения всех процессов
#
#     print("end")

# print("start")
# train()
# print("end")


# game = TicTacToe()
#
# agent_X = DQNAgent(player=1)
# agent_O = DQNAgent(player=2)
#
# load_models(agent_X, agent_O)
#
# player = [1, 2]
# n = 0
#
# state = game.reset()
# result = False
# while not result:
#     print(game.available_actions())
#     game.print_board()
#
#
#     if n%2 == 1:
#         x, y = input().split(' ')
#         result = game.human_step(int(y), int(x), player[n%2])[0]
#
#     else:
#         # Ход агента X
#         available_actions = game.available_actions()
#         action_X = agent_X.choose_action(state, available_actions)
#         result, new_state = game.step(action_X, agent_X.player)
#         winner = game.get_winner() if result else 0
#         reward_X = 1 if winner == agent_X.player else -1 if winner == agent_O.player else 0
#
#         agent_X.update_q_value(state, action_X, reward_X, new_state, result)
#         state = new_state
#
#     print(f"end game: {result}")
#     n+=1
#
# print(f"Winner: {game.get_winner()}")


def main():
    play_with_bot = True
    current_agent = QLearningAgent.load("src/merged_agent.pkl")

    ttt_gui = TicTacToeGUI()
    state = tuple(ttt_gui.game.reset())

    running = True
    while running:
        ttt_gui.draw_board()

        turn = ttt_gui.current_player

        if play_with_bot and turn == 1:
            available_actions = ttt_gui.game.available_actions()
            action = current_agent.choose_action(state, available_actions)
            done, next_state = ttt_gui.game.step(action, 1)
            ttt_gui.current_player = 3 - ttt_gui.current_player
            ttt_gui.check_winner()
            ttt_gui.selected_board = ttt_gui.game.active_desk
            ttt_gui.wins_boards = ttt_gui.game.won_fields
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        x, y = pygame.mouse.get_pos()
                        ttt_gui.handle_click(x, y)

        if ttt_gui.game_over:
            ttt_gui.draw_board()
            if ttt_gui.winner == 0:
                winner_text = f"Draw in the game!"
            else:
                winner_text = f"Player {ttt_gui.winner} wins!"
            text = font.render(winner_text, True, BLACK)
            screen.blit(text, (SCREEN_SIZE // 2.5, SCREEN_SIZE // 2))
            pygame.display.flip()
            pygame.time.wait(3000)
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()



