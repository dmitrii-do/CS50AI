"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

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
    # raise NotImplementedError
    cnt_x = cnt_o = 0
    for cel in board:
        if cel == X:
            cnt_x += 1
        elif cel == O:
            cnt_o += 1
    if cnt_x > cnt_o:
        return O
    return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # raise NotImplementedError
    free_cells = []
    size = len(board)
    for i in range(size):
        for j in range(size):
            if board[i][j] is None:
                free_cells.append((i, j))
    return free_cells


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # raise NotImplementedError
    if action not in actions(board):
        raise ValueError("The action is not a valid action for the board.")
    return deepcopy(board)


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # raise NotImplementedError
    n = len(board)
    # Check rows and columns
    for i in range(n):
        row = board[i]
        if is_three_in_row(X, row):
            return X
        if is_three_in_row(O, row):
            return O
        column = [line[i] for line in board]
        if is_three_in_row(X, column):
            return X
        if is_three_in_row(O, column):
            return O
        
    # Check diagonals
    diagonal_left = [board[i][i] for i in range(n)]
    diagonal_right = [board[i][n-1-i] for i in range(n)]
    if is_three_in_row(X, diagonal_left) or is_three_in_row(X, diagonal_right):
        return X
    if is_three_in_row(O, diagonal_left) or is_three_in_row(O, diagonal_right):
        return O
    return


# Check if three in row
def is_three_in_row(name, row):
    return all(value == name for value in row)

        
def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    raise NotImplementedError


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    raise NotImplementedError


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    raise NotImplementedError
