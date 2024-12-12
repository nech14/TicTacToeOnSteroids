import pygame

from src.QLearningAgent import QLearningAgent
from src.gui import TicTacToeGUI, font, BLACK, SCREEN_SIZE, screen
from src.minimax import find_best_move


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


def game_thiw_MM():
    ttt_gui = TicTacToeGUI()
    game = ttt_gui.game
    game.reset()

    player = 2  # Игрок 1 (ИИ)
    depth = 4  # Глубина поиска


    running = True

    while running:
        ttt_gui.draw_board()

        turn = ttt_gui.current_player

        if turn == player:
            best_move = find_best_move(game, depth, player)
            is_done, board = game.step(best_move, player)
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


if __name__ == "__main__":
    # main()
    game_thiw_MM()



