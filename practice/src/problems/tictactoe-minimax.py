import numpy as np

META = dict(title='Solve a tic-tac-toe position with minimax', tags=['Reinforcement Learning'], diff='Medium', runtime='python',
            related=['tictactoe-legal-moves-and-winner'])

STATEMENT = r"""
Write `solve(board)` for a reachable tic-tac-toe position (same encoding as the rules step), using plain negamax: no pruning and no memoisation.

- The value of a finished position is from the point of view of the player to move: `-1` if the other player has just won, `0` for a draw.
- Otherwise the value is the maximum over legal moves of minus the value of the resulting position.

Return a dict with `value` (int), `best_moves` (every legal move that achieves the value, increasing; `[]` if the game is over) and `nodes` (the number of positions evaluated, counting the root and every position the recursion visits).
"""

STARTER = '''def solve(board):
    """Negamax value, optimal moves and node count for a tic-tac-toe position."""
    moves = [i for i, ch in enumerate(board) if ch == "."]
    return {"value": 0, "best_moves": moves[:1], "nodes": 1}
'''

SOLUTION = '''LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

def _winner(b):
    for x, y, z in LINES:
        if b[x] != "." and b[x] == b[y] == b[z]:
            return b[x]
    return None

def _negamax(b, count):
    count[0] += 1
    if _winner(b):
        return -1, []
    moves = [i for i, ch in enumerate(b) if ch == "."]
    if not moves:
        return 0, []
    me = "X" if b.count("X") == b.count("O") else "O"
    scores = []
    for m in moves:
        v, _ = _negamax(b[:m] + me + b[m + 1:], count)
        scores.append(-v)
    best = max(scores)
    return best, [m for m, s in zip(moves, scores) if s == best]

def solve(board):
    count = [0]
    v, best = _negamax(board, count)
    return {"value": v, "best_moves": best, "nodes": count[0]}
'''

HINTS = [
    'Negamax: a position’s value for me is the best of minus my opponent’s value after each of my moves.',
    'If a winner exists, the player who just moved won, so the player to move scores -1.',
    'Count a node every time the function is entered, including terminal positions.',
]

TESTS = [
    ("XX.OO....",),
    ("XO.XO....",),
    ("XOXXOOOXX",),
    ("XOX.O....",),
    ("X...O....",),
    ("XXXOO....",),
]

EXPLANATION = r"""
## Exact search
Minimax explores every line of play and backs up the result, so on a game as small as tic-tac-toe it gives the exact value: the empty board is a draw, and plain negamax visits 549,946 positions to prove it.

## Why it does not scale
The node count grows exponentially with depth. Chess and Go have far too many positions for this, even with alpha-beta pruning, which is why AlphaZero replaces exhaustive search with a guided search (PUCT) and replaces exact terminal values deep in the tree with a learned value estimate.

## A useful oracle
On small games minimax is ground truth. Comparing an MCTS agent's chosen moves against `best_moves` is the simplest way to test a search implementation before trusting it on a game you cannot solve.

## Interview follow-ups
- How much does alpha-beta pruning reduce the node count with perfect move ordering, and how does a transposition table help in tic-tac-toe specifically?
"""
