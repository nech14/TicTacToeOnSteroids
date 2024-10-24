
import pygame
import numpy as np

from src.logic_game import TicTacToe

# Инициализация Pygame
pygame.init()

# Настройки экрана
SCREEN_SIZE = 600
BOARD_SIZE = SCREEN_SIZE // 3
CELL_SIZE = BOARD_SIZE // 3
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Tic Tac Toe on Steroids")

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Шрифт
font = pygame.font.SysFont(None, 40)

# Класс игры
class TicTacToeGUI:
    def __init__(self):
        self.game = TicTacToe()
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def draw_board(self):
        screen.fill(WHITE)

        # Рисуем внешнюю доску
        for i in range(1, 3):
            pygame.draw.line(screen, BLACK, (i * BOARD_SIZE, 0), (i * BOARD_SIZE, SCREEN_SIZE), 3)
            pygame.draw.line(screen, BLACK, (0, i * BOARD_SIZE), (SCREEN_SIZE, i * BOARD_SIZE), 3)

        # Рисуем внутренние доски и метки X/O
        for i in range(9):
            board_x = (i % 3) * BOARD_SIZE
            board_y = (i // 3) * BOARD_SIZE
            for y in range(3):
                for x in range(3):
                    value = self.game.board[i][y][x]
                    cell_x = board_x + x * CELL_SIZE
                    cell_y = board_y + y * CELL_SIZE

                    pygame.draw.rect(screen, WHITE, (cell_x, cell_y, CELL_SIZE, CELL_SIZE))
                    pygame.draw.rect(screen, BLACK, (cell_x, cell_y, CELL_SIZE, CELL_SIZE), 1)

                    if value == 1:
                        text = font.render('X', True, RED)
                        screen.blit(text, (cell_x + 10, cell_y + 10))
                    elif value == 2:
                        text = font.render('O', True, BLUE)
                        screen.blit(text, (cell_x + 10, cell_y + 10))

        pygame.display.flip()

    def check_winner(self):
        # Проверка победителя
        if self.game.get_winner() != 0:
            self.winner = self.game.get_winner()
            self.game_over = True

    def handle_click(self, x, y):
        if not self.game_over:
            global_x = x // CELL_SIZE
            global_y = y // CELL_SIZE
            board_x = global_x // 3
            board_y = global_y // 3
            cell_x = global_x % 3
            cell_y = global_y % 3
            if self.game.human_step(cell_x, cell_y, self.current_player):
                self.current_player = 3 - self.current_player
                self.check_winner()

def main():
    ttt_gui = TicTacToeGUI()

    running = True
    while running:
        ttt_gui.draw_board()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = pygame.mouse.get_pos()
                    print(x, y)
                    ttt_gui.handle_click(x, y)

        if ttt_gui.game_over:
            winner_text = f"Player {ttt_gui.winner} wins!"
            text = font.render(winner_text, True, BLACK)
            screen.blit(text, (SCREEN_SIZE // 4, SCREEN_SIZE // 2))
            pygame.display.flip()
            pygame.time.wait(3000)
            running = False

    pygame.quit()

if __name__ == "__main__":
    main()


