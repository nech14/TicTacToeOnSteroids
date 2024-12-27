import multiprocessing
import os
import sys

import numpy as np
import pygame

from src.gui import TicTacToeGUI
from src.logic_game import TicTacToe
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import copy
import graphviz

# Глобальный массив для хранения состояний игры
state_storage = []
#
# def save_board_image(game, filename, screen=None, gui=None):
#     """
#     Сохраняет текущее состояние игры как изображение.
#     Эта функция должна вызываться только из основного процесса.
#     """
#     if screen is None or gui is None:
#         raise ValueError("Screen and GUI must be provided in multiprocessing mode.")
#     gui.game = game
#     gui.draw_board()
#     pygame.image.save(screen, filename)


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


#
# def minimax_alpha_beta(game, depth, alpha, beta, maximizing_player, player, tree=None, parent_id=None):
#     opponent = 2 if player == 1 else 1
#
#     # Проверка окончания игры
#     if depth == 0 or game.status:
#         if not maximizing_player:
#             opponent, player = player, opponent
#         evaluation = evaluate_state(game, player)
#
#         if not tree is None and not parent_id is None:
#             leaf_id = f"Leaf_{parent_id}_{depth}"
#             tree.node(leaf_id, label=f"Score: {evaluation}", shape="ellipse")
#             tree.edge(parent_id, leaf_id)
#
#         return evaluation
#
#     if maximizing_player:
#         max_eval = float('-inf')
#         for action in game.available_actions():
#             game.step(action, player)
#
#             if not tree is None and not parent_id is None:
#                 # Сохранение изображения для узла
#                 child_id = f"Action_{action}_{depth}"
#                 state_image = f"state_{child_id}.png"
#                 save_board_image(game, state_image)
#
#                 # Создание узла дерева
#                 tree.node(child_id, label="", image=state_image, shape="plaintext")
#                 tree.edge(parent_id, child_id)
#
#             eval = minimax_alpha_beta(game, depth - 1, alpha, beta, False, opponent)
#             game.undo_move(action)
#             max_eval = max(max_eval, eval)
#             alpha = max(alpha, eval)
#             if beta <= alpha:
#                 break
#         return max_eval
#     else:
#         min_eval = float('inf')
#         for action in game.available_actions():
#             game.step(action, player)
#
#             if not tree is None and not parent_id is None:
#                 # Сохранение изображения для узла
#                 child_id = f"Action_{action}_{depth}"
#                 state_image = f"state_{child_id}.png"
#                 save_board_image(game, state_image)
#
#                 # Создание узла дерева
#                 tree.node(child_id, label="", image=state_image, shape="plaintext")
#                 tree.edge(parent_id, child_id)
#
#             eval = minimax_alpha_beta(game, depth - 1, alpha, beta, True, opponent)
#             game.undo_move(action)
#             min_eval = min(min_eval, eval)
#             beta = min(beta, eval)
#             if beta <= alpha:
#                 break
#         return min_eval




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

#
# def evaluate_action():
#     return 0, 0

# def evaluate_action(action, game, depth, alpha, beta, player, tree=None, parent_id=None):
#     game_copy = game.copy()
#     game_copy.step(action, player)
#
#     if not tree is None and not parent_id is None:
#         # # Сохранение изображения для узла
#         # node_id = f"Move_{action}"
#         # state_image = f"state_{node_id}.png"
#         # save_board_image(game_copy, state_image)
#         #
#         # # Создаем узел дерева с изображением
#         # tree.node(node_id, label="", image=state_image, shape="plaintext")
#         # tree.edge(parent_id, node_id)
#
#         move_value = minimax_alpha_beta(game_copy, depth - 1, alpha, beta, False, 3 - player, tree, parent_id)
#     else:
#         move_value = minimax_alpha_beta(game_copy, depth - 1, alpha, beta, False, 3-player)
#
#     pygame.quit()
#     return action, move_value


#
# # Параллельная версия find_best_move
# def find_best_move_CPU(game: TicTacToe, depth, player, create_tree=True):
#
#     best_move = None
#     alpha = float('-inf')
#     beta = float('inf')
#     actions = game.available_actions()
#     copy_game = copy.deepcopy(game)
#
#     multiprocessing.set_start_method('spawn', force=True)
#
#     if create_tree:
#         # Создаем дерево решений
#         tree = graphviz.Digraph(format="png")
#         root_id = "Root"
#         tree.node(root_id, label="Start", shape="ellipse")
#
#         # Оценка всех действий
#         with Pool(cpu_count()) as pool:
#             results = pool.map(
#                 partial(evaluate_action, game=copy_game, depth=depth, alpha=alpha, beta=beta, player=player, tree=tree,
#                         parent_id=root_id), actions)
#
#         # Генерация дерева с изображениями в основном процессе
#         for action, move_value in results:
#             game.step(action, player)
#             state_image = f"state_{action}.png"
#             save_board_image(game, state_image)  # Создание изображения в основном процессе
#             tree.node(f"Move_{action}", label=f"Value: {move_value}", image=state_image, shape="plaintext")
#             tree.edge(root_id, f"Move_{action}")
#             game.undo_move(action)
#
#     else:
#         with Pool(cpu_count()) as pool:
#             results = pool.map(partial(evaluate_action, game=copy_game, depth=depth, alpha=alpha, beta=beta,
#                                        player=player), actions)
#
#     best_move, _ = max(results, key=lambda x: x[1])
#
#     if best_move is None:
#         best_move = np.random.choice(actions)
#
#     if create_tree:
#         # Сохраняем дерево решений
#         output_file = "decision_tree"
#         tree.render(output_file, cleanup=True)
#         print(f"Дерево решений сохранено в {output_file}.png")
#
#     return best_move



def minimax_alpha_beta(game, depth, alpha, beta, maximizing_player, player, tree=None, parent_id=None):
    opponent = 2 if player == 1 else 1

    if depth == 0 or game.status:
        evaluation = evaluate_state(game, player if maximizing_player else opponent)

        if tree is not None and parent_id is not None:
            leaf_id = f"Leaf_{parent_id}_{depth}"
            tree.node(leaf_id, label=f"Score: {evaluation}", shape="ellipse")
            tree.edge(parent_id, leaf_id)

        return evaluation, []

    nodes = []

    if maximizing_player:
        max_eval = float('-inf')
        for action in game.available_actions():
            game.step(action, player)
            child_id = f"Node_{parent_id}_Action_{action}"

            if tree is not None and parent_id is not None:
                state_copy = copy.deepcopy(game)
                nodes.append((child_id, state_copy, parent_id))
                tree.node(child_id, label=f"Move: {action}", shape="plaintext")
                tree.edge(parent_id, child_id)

            eval, child_nodes = minimax_alpha_beta(game, depth - 1, alpha, beta, False, opponent, tree, child_id)
            nodes.extend(child_nodes)
            game.undo_move(action)

            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, nodes
    else:
        min_eval = float('inf')
        for action in game.available_actions():
            game.step(action, player)
            child_id = f"Node_{parent_id}_Action_{action}"

            if tree is not None and parent_id is not None:
                state_copy = copy.deepcopy(game)
                nodes.append((child_id, state_copy, parent_id))
                tree.node(child_id, label=f"Move: {action}", shape="plaintext")
                tree.edge(parent_id, child_id)

            eval, child_nodes = minimax_alpha_beta(game, depth - 1, alpha, beta, True, opponent, tree, child_id)
            nodes.extend(child_nodes)
            game.undo_move(action)

            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, nodes




def save_board_image(game, filename, screen=None, gui=None):
    """
    Сохраняет текущее состояние игры как изображение.
    Эта функция должна вызываться только из основного процесса.
    """
    if screen is None or gui is None:
        raise ValueError("Screen and GUI must be provided in multiprocessing mode.")
    gui.game = game
    gui.draw_board()
    pygame.image.save(screen, filename)


def evaluate_action(action, game, depth, alpha, beta, player, tree=None, parent_id=None):
    game_copy = game.copy()
    game_copy.step(action, player)

    move_value, nodes = minimax_alpha_beta(game_copy, depth - 1, alpha, beta, False, 3 - player, tree, parent_id)

    return action, move_value, nodes

def find_best_move_CPU(game, depth, player, create_tree=True):
    best_move = None
    alpha = float('-inf')
    beta = float('inf')
    actions = game.available_actions()
    copy_game = copy.deepcopy(game)
    results = []

    if create_tree:
        tree = graphviz.Digraph(format="svg")
        tree.attr(rankdir="TB", nodesep="1", ranksep="1")
        root_id = "Root"

        pygame.init()
        screen = pygame.Surface((800, 800))
        gui = TicTacToeGUI(screen, pygame.font.Font(None, 48))

        # Сохранение стартового игрового поля в корне
        filename = f"state_{root_id}.png"
        save_board_image(game, filename, screen, gui)
        tree.node(root_id, label="", image=filename, shape="plaintext")

        state_storage = []
        temp_files = [filename]

        # Перебор первого уровня
        for action in actions:
            action_id = f"Root_Action_{action}"
            game.step(action, player)
            state_copy = copy.deepcopy(game)
            filename = f"state_{action_id}.png"
            save_board_image(state_copy, filename, screen, gui)
            temp_files.append(filename)

            tree.node(action_id, label="", image=filename, shape="plaintext")
            tree.edge(root_id, action_id)
            _, child_nodes = minimax_alpha_beta(game, depth - 1, alpha, beta, False, 3 - player, tree, action_id)
            state_storage.extend(child_nodes)
            game.undo_move(action)

        # Сохранение состояний второго и третьего уровня
        for node_id, state, parent_id in state_storage:
            filename = f"state_{node_id}.png"
            save_board_image(state, filename, screen, gui)
            temp_files.append(filename)
            tree.node(node_id, label="", image=filename, shape="plaintext")

        output_file = "decision_tree"
        tree.render(output_file, cleanup=True)
        print(f"Дерево решений сохранено в {output_file}.svg")

        # Удаляем временные файлы
        for file in temp_files:
            os.remove(file)
    else:
        with Pool(cpu_count()) as pool:
            results = pool.map(
                partial(evaluate_action, game=copy_game, depth=depth, alpha=alpha, beta=beta, player=player),
                actions
            )

    # best_move, _ = max(((action, value) for action, value, _ in results), key=lambda x: x[1])
    #
    # if best_move is None:
    #     best_move = np.random.choice(actions)

    if results:
        best_move, _ = max(((action, value) for action, value, _ in results), key=lambda x: x[1])

    if best_move is None:
        best_move = np.random.choice(actions)

    return best_move



# # Параллельная версия find_best_move
# def find_best_move_CPU():
#
#     best_move = None
#     alpha = float('-inf')
#     beta = float('inf')
#     # actions = game.available_actions()
#     # copy_game = copy.deepcopy(game)
#     actions = [1, 2, 3, 4]
#
#     # if multiprocessing.current_process().name == 'MainProcess':
#     #     init_pygame()
#
#     multiprocessing.set_start_method('spawn', force=True)
#     with Pool(cpu_count()) as pool:
#     #     print("Initializing Pygame in the main process...")
#         # results = pool.map(partial(evaluate_action, game=copy_game, depth=depth, alpha=alpha, beta=beta, player=player), actions)
#         results = pool.map(partial(evaluate_action), actions)
#     # results = [(1,2)]
#     best_move, _ = max(results, key=lambda x: x[1])
#
#     if best_move is None:
#         return np.random.choice(actions)
#
#     return best_move
#
