import numpy as np
from src.logic_game import TicTacToe

#
# def evaluate_state(game, player):
#     opponent = 2 if player == 1 else 1
#     score = 0
#
#     # Оцениваем метаполе
#     for d in range(9):
#         if game.won_fields[d] == player:
#             score += 1000  # Выигранное поле
#         elif game.won_fields[d] == opponent:
#             score -= 1000  # Проигранное поле
#         elif game.won_fields[d] == -1:
#             score += 10  # Ничья, небольшое преимущество
#
#     return score


def evaluate_state(game, player):
    opponent = 2 if player == 1 else 1
    score = 0

    # Оцениваем метаполе
    for d in range(9):
        if game.won_fields[d] == player:
            score += 1000
        elif game.won_fields[d] == opponent:
            score -= 1000
        elif game.won_fields[d] == -1:
            score += 10

        # Центральные позиции на уровне метаполя
        if game.board[d][1, 1] == player:
            score += 15
        elif game.board[d][1, 1] == opponent:
            score -= 15

        # Угрозы в метаполях
        if game.won_fields[d] == 0:
            for line in np.vstack([game.board[d], game.board[d].T, [game.board[d].diagonal()], [np.fliplr(game.board[d]).diagonal()]]):
                if np.count_nonzero(line == player) > 0 and np.count_nonzero(line == opponent) == 0:
                    score += 20
                elif np.count_nonzero(line == opponent) > 0 and np.count_nonzero(line == player) == 0:
                    score -= 20

    # Учитываем текущее активное поле
    if game.last_turn[0] != -1:
        active_desk = game.last_turn[1] * 3 + game.last_turn[2]
        if game.won_fields[active_desk] == player:
            score += 30
        elif game.won_fields[active_desk] == opponent:
            score -= 30

    # Усиление оценки для завершающихся игр
    if np.count_nonzero(game.won_fields == 0) <= 3:
        score *= 1.5

    return score



def minimax_alpha_beta(game, depth, alpha, beta, maximizing_player, player):
    if (len(game.available_actions()) == 1):
        return game.available_actions()[0]

    opponent = 2 if player == 1 else 1

    # Проверяем окончание игры
    if depth == 0 or np.all(game.won_fields != 0):
        return evaluate_state(game, player)

    if maximizing_player:
        max_eval = float('-inf')
        for action in game.available_actions():
            # Создаём копию игры
            game_copy = game.copy()
            game_copy.step(action, player)
            eval = minimax_alpha_beta(game_copy, depth - 1, alpha, beta, False, player)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Отсечение
        return max_eval
    else:
        min_eval = float('inf')
        for action in game.available_actions():
            # Создаём копию игры
            game_copy = game.copy()
            game_copy.step(action, opponent)
            eval = minimax_alpha_beta(game_copy, depth - 1, alpha, beta, True, player)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Отсечение
        return min_eval


def find_best_move(game: TicTacToe, depth, player):
    best_move = None
    best_value = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    actions = game.available_actions()

    if (len(actions) == 1):
        return actions[0]
    for action in actions:
        # Создаём копию игры
        game_copy = game.copy()
        game_copy.step(action, player)
        move_value = minimax_alpha_beta(game_copy, depth - 1, alpha, beta, False, player)

        if move_value > best_value:
            best_value = move_value
            best_move = action

    return best_move
