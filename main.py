import multiprocessing

from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import sys

from src.QLearningAgent import QLearningAgent
from src.gui import TicTacToeGUI, initialize_pygame
from src.minimax import find_best_move, find_best_move_CPU


SCREEN_SIZE = 800

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

font = None
screen = None

game_mode = None
player_start = None

def init_pygame():
    """Инициализирует Pygame и возвращает screen и font."""
    pygame.init()
    global screen, font
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Game Mode Selection")
    font = pygame.font.Font(None, 48)


def draw_button(text, x, y, w, h):
    pygame.draw.rect(screen, BLUE, (x, y, w, h))
    label = font.render(text, True, WHITE)
    screen.blit(label, (x + 20, y + 10))

def mode_selection():
    global game_mode
    while game_mode is None:
        screen.fill(WHITE)
        draw_button("Mode 1", 300, 250, 200, 60)
        draw_button("Mode 2", 300, 350, 200, 60)
        draw_button("Mode 3", 300, 450, 200, 60)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if 300 <= x <= 500:
                    if 250 <= y <= 310:
                        game_mode = 1
                    elif 350 <= y <= 410:
                        game_mode = 2
                    elif 450 <= y <= 510:
                        game_mode = 3


def main():
    play_with_bot = True
    # current_agent = QLearningAgent.load("src/agent1_9000.pkl")
    current_agent = QLearningAgent.load("AI/agent1_17000.pkl")
    # current_agent = 1
    # current_agent = QLearningAgent.load("src/merged_agent.pkl")

    ttt_gui = TicTacToeGUI()
    state = ttt_gui.game.reset()

    running = True

    while running:
        ttt_gui.draw_board()

        turn = ttt_gui.current_player

        if play_with_bot and turn == 1:
            available_actions = ttt_gui.game.available_actions()
            action = current_agent.choose_action((tuple(ttt_gui.game.get_all_board()), tuple(ttt_gui.game.won_fields)), available_actions)
            done, next_state = ttt_gui.game.step(action, 1)
            ttt_gui.current_player = 3 - ttt_gui.current_player
            ttt_gui.check_winner()
            ttt_gui.selected_board = ttt_gui.game.active_desk
            ttt_gui.wins_boards = ttt_gui.game.won_fields
            ttt_gui.last_turn = [action//9, action%9%3, action%9//3]

        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        x, y = pygame.mouse.get_pos()
                        next_state = ttt_gui.handle_click(x, y)


        if ttt_gui.game_over:
            ttt_gui.draw_board()
            if ttt_gui.winner == 0:
                winner_text = f"Draw in the game!"
            else:
                winner_text = f"Player {ttt_gui.winner} wins!"

            print(winner_text)
            text = font.render(winner_text, True, BLACK)
            screen.blit(text, (SCREEN_SIZE // 2.5, SCREEN_SIZE // 2))
            pygame.display.flip()
            pygame.time.wait(5000)
            running = False


    pygame.quit()


def player_order_selection():
    global player_start
    while player_start is None:
        screen.fill(WHITE)
        draw_button("Player 1 Starts", 250, 250, 300, 60)
        draw_button("Player 2 Starts", 250, 350, 300, 60)
        draw_button("Two players", 250, 450, 300, 60)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if 250 <= x <= 550:
                    if 250 <= y <= 310:
                        player_start = 1
                    elif 350 <= y <= 410:
                        player_start = 2
                    elif 450 <= y <= 510:
                        player_start = 3


def game_thiw_MM(gamemode=1, player=2):
    ttt_gui = TicTacToeGUI(screen, font, gamemode)
    game = ttt_gui.game
    game.reset()

    depth = 7 # Глубина поиска

    AI_player = 3 - player

    running = True
    text_displayed = False

    while running:
        ttt_gui.draw_board()

        turn = ttt_gui.current_player
        # print(ttt_gui.game.won_fields)

        if not ttt_gui.game_over:
            if turn == AI_player:
                # best_move = find_best_move(game, depth, turn)
                best_move = find_best_move_CPU(game, depth, turn)
                # best_move = find_best_move_CPU()
                is_done, board = game.step(best_move, turn)
                ttt_gui.current_player = 3 - ttt_gui.current_player
                ttt_gui.check_winner()
                ttt_gui.selected_board = ttt_gui.game.active_desk
                ttt_gui.wins_boards = ttt_gui.game.won_fields
                # print("p1", best_move)
                ttt_gui.last_turn = [best_move // 9, best_move % 9 % 3, best_move % 9 // 3]


            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            x, y = pygame.mouse.get_pos()
                            next_state = ttt_gui.handle_click(x, y)
                # best_move = find_best_move(game, depth, 3-player)
                # is_done, board = game.step(best_move, 3-player)
                # ttt_gui.current_player = 3 - ttt_gui.current_player
                # ttt_gui.check_winner()
                # ttt_gui.selected_board = ttt_gui.game.active_desk
                # ttt_gui.wins_boards = ttt_gui.game.won_fields
                # print("p2", best_move)
                # ttt_gui.last_turn = [best_move // 9, best_move % 9 % 3, best_move % 9 // 3]

            ttt_gui.check_winner()


        elif ttt_gui.game_over and not text_displayed:
            ttt_gui.draw_board()
            if ttt_gui.winner == 0:
                winner_text = f"Draw in the game!"
            else:
                winner_text = f"Player {ttt_gui.winner} wins!"

            print(winner_text)
            text = font.render(winner_text, True, BLACK)
            screen.blit(text, (SCREEN_SIZE // 2.5, SCREEN_SIZE // 2))
            pygame.display.flip()

            pygame.time.wait(500)


        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    running = False  # Закрыть игру при нажатии Enter

    pygame.quit()

#
# if __name__ == "__main__":
#     # main()
#     game_thiw_MM()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)  # Перенос в основную функцию
    screen, font = initialize_pygame()

    mode_selection()
    player_order_selection()


    print(game_mode, player_start)
    game_thiw_MM(game_mode, player_start)

