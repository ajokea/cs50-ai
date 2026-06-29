"""
Tic Tac Toe Player
"""

import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    empty_cells = 0
    for row in board:
        for col in row:
            if col == EMPTY:
                empty_cells += 1

    return O if empty_cells % 2 == 0 else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    empty_cells = set()
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == EMPTY:
                empty_cells.add((row, col))

    return empty_cells


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    row, col = action
    if not 0 <= row < len(board) or not 0 <= col < len(board[0]):
        raise Exception("Index out of bounds")
    if board[row][col] != EMPTY:
        raise Exception(f"{row}, {col} already taken")

    resulting_board = copy.deepcopy(board)
    resulting_board[row][col] = player(board)
    return resulting_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    win_states = [
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        [(0, 0), (1, 1), (2, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        [(0, 2), (1, 1), (2, 0)]
    ]

    for state in win_states:
        tic, tac, toe = state

        if board[tic[0]][tic[1]] == board[tac[0]][tac[1]] == board[toe[0]][toe[1]] != EMPTY:
            return board[tic[0]][tic[1]]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is None:
        for row in board:
            for col in row:
                if col == EMPTY:
                    return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    game_winner = winner(board)

    if game_winner == X:
        return 1
    elif game_winner == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    def max_value(state, a, b):
        if terminal(state):
            return utility(state)
        val = -math.inf
        for act in actions(state):
            val = max(val, min_value(result(state, act), a, b))
            a = max(a, val)
            if a >= b:  # prune when optimal alpha (a) and beta (b) are reached
                break
        return val

    def min_value(state, a, b):
        if terminal(state):
            return utility(state)
        val = math.inf
        for act in actions(state):
            val = min(val, max_value(result(state, act), a, b))
            b = min(b, val)
            if a >= b:  # prune when optimal alpha (a) and beta (b) are reached
                break
        return val

    alpha = -math.inf
    beta = math.inf
    optimal_action = None
    if player(board) == X:
        best_value = -math.inf
        for action in actions(board):
            # finding the MAX of the min values of opponent
            value = min_value(result(board, action), alpha, beta)
            if value > best_value:
                best_value = value
                optimal_action = action
                alpha = max(alpha, best_value)
    else:
        best_value = math.inf
        for action in actions(board):
            # finding the MIN of the max values of opponent
            value = max_value(result(board, action), alpha, beta)
            if value < best_value:
                best_value = value
                optimal_action = action
                beta = min(beta, best_value)

    return optimal_action