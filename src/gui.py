
import pygame
import numpy as np

from src.logic_game import TicTacToe

# Инициализация Pygame
pygame.init()

# Настройки экрана
SCREEN_SIZE = 800
MARGIN = 10
BOARD_SIZE = SCREEN_SIZE // 3 - MARGIN
CELL_SIZE = BOARD_SIZE // 3 - MARGIN
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Tic Tac Toe on Steroids")

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Шрифт
font = pygame.font.SysFont(None, 40)

# Класс игры
class TicTacToeGUI:
    def __init__(self):
        self.game = TicTacToe()
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.selected_board = 0
        self.mode = 1
        self.wins_boards = np.zeros(9, dtype=int)

    def draw_board(self):
        screen.fill(WHITE)

        # # Рисуем внешнюю доску
        # for i in range(1, 3):
        #     pygame.draw.line(screen, BLACK, (i * BOARD_SIZE, 0), (i * BOARD_SIZE, SCREEN_SIZE), 3)
        #     pygame.draw.line(screen, BLACK, (0, i * BOARD_SIZE), (SCREEN_SIZE, i * BOARD_SIZE), 3)

        # Рисуем внутренние доски и метки X/O
        for i in range(9):
            board_x = (i % 3) * (BOARD_SIZE + MARGIN) + MARGIN
            board_y = (i // 3) * (BOARD_SIZE + MARGIN) + MARGIN

            if self.wins_boards[i] == 1:
                pygame.draw.rect(
                    screen, RED,
                    (board_x - 1.5 * MARGIN, board_y - 1.5 * MARGIN, BOARD_SIZE, BOARD_SIZE),
                    10
                )
            elif self.wins_boards[i] == 2:
                pygame.draw.rect(
                    screen, BLUE,
                    (board_x - 1.5 * MARGIN, board_y - 1.5 * MARGIN, BOARD_SIZE, BOARD_SIZE),
                    10
                )

            # Проверка, если доска выбрана, обводим ее желтой рамкой
            if i == self.selected_board and not self.game_over:
                pygame.draw.rect(screen, YELLOW, (board_x-1.5*MARGIN, board_y-1.5*MARGIN, BOARD_SIZE, BOARD_SIZE), 10)




            for y in range(3):
                for x in range(3):
                    value = self.game.board[i][y][x]
                    cell_x = board_x + x * CELL_SIZE
                    cell_y = board_y + y * CELL_SIZE

                    pygame.draw.rect(screen, WHITE, (cell_x, cell_y, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(screen, BLACK, (cell_x, cell_y, CELL_SIZE, CELL_SIZE), 1)

                    if value == 1:
                        text = font.render('X', True, RED)
                        screen.blit(text, (cell_x + CELL_SIZE//3, cell_y + CELL_SIZE//3))
                    elif value == 2:
                        text = font.render('O', True, BLUE)
                        screen.blit(text, (cell_x + CELL_SIZE//3, cell_y + CELL_SIZE//3))


        pygame.display.flip()

    def check_winner(self):
        # Проверка победителя
        if self.mode == 0:
            if self.game.get_winner() != 0:
                self.winner = self.game.get_winner()
                self.game_over = True
        elif self.mode == 1:
            if len(self.game.available_actions()) == 0:
                self.winner = self.game.get_winner()
                self.game_over = True

    def handle_click(self, x, y):
        if not self.game_over:
            # Находим индекс большой доски с учетом отступов
            board_x = (x - MARGIN) // (BOARD_SIZE + MARGIN)
            board_y = (y - MARGIN) // (BOARD_SIZE + MARGIN)

            # Проверяем, что клик попал в пределах доски
            if 0 <= board_x < 3 and 0 <= board_y < 3:
                # Индекс выбранной большой доски от 0 до 8
                board_index = board_y * 3 + board_x
                self.selected_board = board_index

                # Определяем координаты верхнего левого угла большой доски
                board_start_x = board_x * (BOARD_SIZE + MARGIN) + MARGIN
                board_start_y = board_y * (BOARD_SIZE + MARGIN) + MARGIN

                # Находим индекс ячейки внутри большой доски (с учетом положения внутри этой доски)
                cell_x = (x - board_start_x) // CELL_SIZE
                cell_y = (y - board_start_y) // CELL_SIZE

                # Проверяем, что клик попал в пределах ячейки
                if 0 <= cell_x < 3 and 0 <= cell_y < 3:
                    # Индекс ячейки внутри большой доски
                    cell_index = cell_y * 3 + cell_x

                    # Вычисляем глобальный индекс ячейки на сетке 9x9
                    global_cell_index = board_index * 9 + cell_index
                    print(f"Нажата глобальная ячейка: {global_cell_index}")  # Выводим индекс

                    result_step, _ = self.game.human_step(cell_x, cell_y, self.current_player)
                    if result_step is None:
                        print("Bad turn!")
                        pass
                    else:
                        self.current_player = 3 - self.current_player
                        self.check_winner()
                        self.selected_board = self.game.active_desk
                        self.wins_boards = self.game.won_fields




def main():
    ttt_gui = TicTacToeGUI()

    play_with_bot = False
    turn = 1

    running = True
    while running:
        turn = 3 - turn
        ttt_gui.draw_board()

        if play_with_bot and turn == 1:
            pass
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

# if __name__ == "__main__":
#     main()
#

