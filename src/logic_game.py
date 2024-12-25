from tkinter.ttk import Treeview

import numpy as np
from pyopencl.cache import retrieve_from_cache



class TicTacToe():

    last_turn = [-1,0,0]
    active_desk = 0


    def __init__(self, gamemode=1):
        self.board = np.zeros((9,3,3), dtype=int)
        self.last_turn = [-1, 0, 0]
        self.won_fields = np.zeros(9, dtype=int)
        self.history = []
        self.history_won_fields = []
        self.gamemode = gamemode
        self.status = False


    def reset(self):
        self.board = np.zeros((9,3,3), dtype=int)
        self.won_fields = np.zeros(9, dtype=int)
        self.last_turn = [-1, 0, 0]
        self.history = []
        self.history_won_fields = []
        self.history_won_fields.append(self.won_fields)
        return self.board.flatten()


    def available_actions(self):
        if np.all(self.won_fields != 0):
            return []

        if self.last_turn[0] == -1:
            return [i for i in range(9*3*3) if self.board.flatten()[i] == 0]

        d = self.last_turn[1]*3 + self.last_turn[2]
        self.active_desk = d

        return [i for i in range(d*9, d*9+9) if self.board.flatten()[i] == 0]


    def get_all_board(self):
        return self.board.flatten()


    def check_end(self):
        if self.gamemode == 1 and np.any(self.won_fields != 0):
            # print("gggg, end game!!!!")
            self.status = True
            return True

        if np.all(self.won_fields != 0):
            self.status = True
            return True

        if len(self.available_actions()) == 0:
            self.status = True
            return True

        return False


    def step(self, action, player):
        if action in self.available_actions():
            d = action // 9
            self.active_desk = d
            self.board[d, (action - d * 9) // 3, action % 3] = player
            self.history.append(self.last_turn)
            self.history_won_fields.append(self.won_fields.copy())
            self.last_turn = [d, (action - d * 9) // 3, action % 3]

            self.won_fields[d] = self.check_winner_field(d)


            return self.check_end(), self.get_all_board()

        return None, self.get_all_board()


    def human_step(self, x ,y, player):

        d = self.last_turn[1]*3 + self.last_turn[2]

        if self.board[d][y][x] == 0:
            self.board[d][y][x] = player
            self.history.append(self.last_turn)
            self.history_won_fields.append(self.won_fields.copy())
            self.last_turn = [d, y, x]

            self.won_fields[d] = self.check_winner_field(d)

            return self.check_end(), self.get_all_board()

        return None, self.get_all_board()


    def undo_move(self, action):
        # Определяем малое поле, строку и столбец
        d = action // 9
        row = (action - d * 9) // 3
        col = action % 3

        # Сбрасываем значение ячейки
        self.board[d, row, col] = 0

        # Восстанавливаем last_turn из истории
        if self.history:
            self.last_turn = self.history.pop()
        else:
            self.last_turn = [-1, 0, 0]
            self.active_desk = 0

        # print(self.history_won_fields)
        if self.history_won_fields:
            self.won_fields = self.history_won_fields.pop().copy()
        else:
            self.won_fields = np.zeros(9, dtype=int)

        if self.status:
            # print("gg")
            self.status = False

        # Обновляем active_desk
        if self.last_turn[0] != -1:
            self.active_desk = self.last_turn[1] * 3 + self.last_turn[2]
        else:
            self.active_desk = next((i for i, x in enumerate(self.won_fields) if x == 0), 0)


    def print_board(self):
        for i in range(0, 9, 3):
            for j in range(3):
                print(self.board[i][j], " ", self.board[i+1][j], " ", self.board[i+2][j])
            print()
        print(f"field: {self.last_turn[1]*3 + self.last_turn[2]}")
        print(f"last turn: {self.last_turn}")
        print(f"won_fields: {self.won_fields}")


    def check_winner_field(self, field, players=None):

        if self.won_fields[field] != 0:
            return self.won_fields[field]

        if players is None:
            players = [1, 2]

        for player in players:
            if np.any(np.all(self.board[field] == player, axis=0)) or \
                    np.any(np.all(self.board[field] == player, axis=1)) or \
                    np.all(np.diag(self.board[field]) == player) or \
                    np.all(np.diag(np.fliplr(self.board[field])) == player):
                return player
            if np.all(self.board[field] != 0):
                return -1  # Ничья

        return 0



    def get_winner(self):

        if self.gamemode == 3:
            win_player = 3 - (len(self.history)%2 + 1)
            return win_player

        player1 = np.count_nonzero(self.won_fields == 1)
        player2 = np.count_nonzero(self.won_fields == 2)

        if player1 == player2:
            return 0
        elif player1 > player2:
            return 1
        else:
            return 2


    def copy(self):
        # Создаём копию текущего состояния игры
        new_game = TicTacToe()
        new_game.board = np.copy(self.board)
        new_game.last_turn = self.last_turn.copy()
        new_game.won_fields = self.won_fields.copy()
        return new_game




