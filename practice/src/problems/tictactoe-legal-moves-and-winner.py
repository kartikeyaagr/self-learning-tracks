import numpy as np

META = dict(title='Tic-tac-toe rules: legal moves and the winner', tags=['Reinforcement Learning'], diff='Easy', runtime='python')

STATEMENT = r"""
A tic-tac-toe board is a 9-character string, read row by row, using `X`, `O` and `.`. X always moves first.

Write `game_state(board)` returning a dict:

- `to_move`: `"X"` if both players have made the same number of moves, otherwise `"O"`.
- `winner`: `"X"` or `"O"` if that player has three in a row (any row, column or diagonal), otherwise `None`.
- `legal`: the indices of empty cells in increasing order, or `[]` if the game has a winner.
- `draw`: true when there is no winner and no empty cell.

Boards in the tests are always reachable in a real game.
"""

STARTER = '''def game_state(board):
    """Whose turn it is, the winner if any, legal moves and whether it is a draw."""
    legal = [i for i, ch in enumerate(board) if ch == '.']
    return {"to_move": "X", "winner": None, "legal": legal, "draw": not legal}
'''

SOLUTION = '''LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

def game_state(board):
    to_move = "X" if board.count("X") == board.count("O") else "O"
    winner = None
    for a, b, c in LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            winner = board[a]
            break
    legal = [] if winner else [i for i, ch in enumerate(board) if ch == "."]
    return {"to_move": to_move, "winner": winner, "legal": legal, "draw": winner is None and "." not in board}
'''

HINTS = [
    'List the eight winning lines once as index triples.',
    'A finished game has no legal moves even if empty cells remain; search code relies on that to stop.',
]

TESTS = [
    (".........",),
    ("X........",),
    ("XXXOO....",),
    ("XOXXOOOXX",),
    ("XOXXOOXXO",),
    ("OXX.O.X.O",),
    ("XO.XO....",),
]

EXPLANATION = r"""
## Rules as an interface
AlphaZero needs only three things from a game: whose turn it is, which moves are legal, and whether the game is over and who won. Everything else (the search, the network, the training) is game-agnostic. Writing the rules carefully pays off because the search will visit every corner case thousands of times.

## Perspective
Search code usually stores values from the point of view of the player to move. `to_move` is what lets the search flip signs correctly as it goes up the tree.

## Interview follow-ups
- What changes in the interface for a game with chance (dice) or hidden information (cards)?
"""
