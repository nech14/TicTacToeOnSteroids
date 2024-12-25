import numpy as np
from src.logic_game import TicTacToe
import time
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

# def evaluate_state(game, player):
#     opponent = 2 if player == 1 else 1
#     score = 0
#
#     # Проверка на победу немедленно
#     if game.status:
#         if game.get_winner() == opponent:
#             print("bad")
#             return float("-inf")
#         elif game.get_winner() == player:
#             print("good")
#             return float("inf")
#
#     # Проверка на победу немедленно
#     if not game.available_actions():
#         return float('inf') if player == (1 if len(game.history) % 2 == 0 else 2) else float('-inf')
#
#     # Оцениваем метаполе
#     for d in range(9):
#         if game.won_fields[d] == player:
#             score += 1000
#         elif game.won_fields[d] == opponent:
#             score -= 1000
#         elif game.won_fields[d] == -1:
#             score += 10
#
#         # Центральные позиции на уровне метаполя
#         if game.board[d][1, 1] == player:
#             score += 15
#         elif game.board[d][1, 1] == opponent:
#             score -= 15
#
#         # Угрозы в метаполях
#         if game.won_fields[d] == 0:
#             for line in np.vstack([game.board[d], game.board[d].T, [game.board[d].diagonal()], [np.fliplr(game.board[d]).diagonal()]]):
#                 if np.count_nonzero(line == player) > 0 and np.count_nonzero(line == opponent) == 0:
#                     score += 20
#                 elif np.count_nonzero(line == opponent) > 0 and np.count_nonzero(line == player) == 0:
#                     score -= 20
#
#     # Учитываем текущее активное поле
#     if game.last_turn[0] != -1:
#         active_desk = game.last_turn[1] * 3 + game.last_turn[2]
#         if game.won_fields[active_desk] == player:
#             score += 30
#         elif game.won_fields[active_desk] == opponent:
#             score -= 30
#
#     # Усиление оценки для завершающихся игр
#     if np.count_nonzero(game.won_fields == 0) <= 3:
#         score *= 1.5
#
#     return score



def evaluate_state(game, player):
    opponent = 2 if player == 1 else 1
    score = 0

    # Проверка на победу немедленно
    if game.status:
        # print(player)
        if game.get_winner() == opponent:
            # print("bad")
            return float("-inf")
        elif game.get_winner() == player:
            # print("good")
            return float("inf")

    # Оценка победы в метаполях
    for d in range(9):
        if game.won_fields[d] == opponent:
            score -= 1000  # Противник захватил поле
        elif game.won_fields[d] == player:
            score += 1000  # AI захватил поле
        elif game.won_fields[d] == -1:
            score += 10  # Ничья на поле


    # 2. Оценка состояния метаполя
    for d in range(9):
        if game.won_fields[d] == player:
            score += 1000
        elif game.won_fields[d] == opponent:
            score -= 1000
        elif game.won_fields[d] == -1:
            score += 50  # Ничья в поле менее важна, но всё же учитывается

        # 3. Центральные позиции в малых полях
        if game.board[d][1, 1] == player:
            score += 30
        elif game.board[d][1, 1] == opponent:
            score -= 30

        # 4. Углы и стороны
        for x, y in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            if game.board[d][x, y] == player:
                score += 10
            elif game.board[d][x, y] == opponent:
                score -= 10

    # 5. Угрозы в активном поле
    if game.last_turn[0] != -1:
        active_desk = game.last_turn[1] * 3 + game.last_turn[2]
        for line in np.vstack([game.board[active_desk], game.board[active_desk].T]):
            if np.count_nonzero(line == player) == 2 and np.count_nonzero(line == 0) == 1:
                score += 40
            if np.count_nonzero(line == opponent) == 2 and np.count_nonzero(line == 0) == 1:
                score -= 40

    # 6. Завершающая игра (если осталось менее 3 полей)
    if np.count_nonzero(game.won_fields == 0) <= 3:
        score *= 1.5

    return score


    return score




def minimax_alpha_beta(game, depth, alpha, beta, maximizing_player, player):
    opponent = 2 if player == 1 else 1
    # print(f"opponent: {opponent} | player: {player}")
    # time.sleep(0.5)
    # print(game.print_board())
    # Проверка окончания игры
    if depth == 0 or game.status:
        if not maximizing_player:
            opponent, player = player, opponent
        evaluation = evaluate_state(game, player)
        # print(f"end: {evaluation}, status: {game.status}")
        return evaluation

    if maximizing_player:
        max_eval = float('-inf')
        for action in game.available_actions():
            game.step(action, player)
            eval = minimax_alpha_beta(game, depth - 1, alpha, beta, False, opponent)
            game.undo_move(action)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for action in game.available_actions():
            game.step(action, player)
            eval = minimax_alpha_beta(game, depth - 1, alpha, beta, True, opponent)
            game.undo_move(action)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def find_best_move(game: TicTacToe, depth, player):
    best_move = None
    best_value = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    actions = game.available_actions()

    best_move = actions[0]

    for action in actions:
        # print(action)
        game.step(action, player)
        move_value = minimax_alpha_beta(game, depth - 1, alpha, beta, False, 3-player)
        game.undo_move(action)

        if move_value > best_value:
            best_value = move_value
            # print(best_value)
            best_move = action

    return best_move