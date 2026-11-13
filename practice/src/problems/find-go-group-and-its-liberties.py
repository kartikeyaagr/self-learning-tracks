import numpy as np

META = dict(title='Find a Go group and its liberties', tags=['Reinforcement Learning'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:07.01'],
            related=['score-terminal-position-with-tromp-taylor-rules', 'build-legal-go-action-mask'])

STATEMENT = r"""
Every Go rule (capture, suicide, ko) is phrased in terms of **groups** and **liberties**:

- A group is a maximal set of same-coloured stones connected through orthogonal neighbours (up, down, left, right; diagonals do not connect).
- A liberty of a group is an **empty** point orthogonally adjacent to at least one of its stones. A group with zero liberties is captured.

The board is an `(N, N)` integer array: `0` empty, `1` black, `-1` white.

Write `group_and_liberties(board, row, col)` that returns `(stones, liberties)` for the group containing the stone at `(row, col)`.

## Output
- `stones`: list of `(r, c)` tuples of Python ints, every stone in the group, sorted ascending (row first, then column).
- `liberties`: list of `(r, c)` tuples, every distinct liberty of the group, sorted the same way. A liberty shared by several stones appears once.
- If `(row, col)` is empty, return `([], [])`.

The board edge is not a liberty: points outside the board simply do not exist.
"""

STARTER = '''import numpy as np

def group_and_liberties(board, row, col):
    """board: (N, N) ints in {-1, 0, 1}. Return (sorted stones, sorted liberties) as lists of (r, c)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def group_and_liberties(board, row, col):
    board = np.asarray(board)
    n = board.shape[0]
    color = board[row, col]
    if color == 0:
        return [], []
    stones, liberties = {(row, col)}, set()
    stack = [(row, col)]
    while stack:                                    # iterative flood fill: no recursion limit on big groups
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            v = board[nr, nc]
            if v == 0:
                liberties.add((nr, nc))
            elif v == color and (nr, nc) not in stones:
                stones.add((nr, nc))
                stack.append((nr, nc))
    fix = lambda pts: sorted((int(r), int(c)) for r, c in pts)
    return fix(stones), fix(liberties)
'''

HINTS = [
    'This is a flood fill (BFS or DFS) from `(row, col)` that only walks onto stones of the same colour.',
    'While exploring, every empty neighbour you meet is a liberty. Store both stones and liberties in sets so shared points are counted once.',
    'Keep a stack of points to visit and a `seen` set of stones; for each popped point check its 4 in-bounds neighbours: empty → add to liberties, same colour and unseen → push. Sort both sets at the end.',
]

_rng = np.random.default_rng(7)
_rand = _rng.integers(-1, 2, size=(7, 7))

TESTS = [
    (np.array([[0, 0, 0, 0, 0],
               [0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]), 1, 1),
    (np.array([[1, -1, 0],
               [-1, 0, 0],
               [0, 0, 0]]), 0, 0),
    (np.array([[1, -1, 0],
               [-1, 0, 0],
               [0, 0, 0]]), 1, 1),
    (np.array([[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]), 2, 2),
    (np.array([[0, -1, 1, 0, 0],
               [-1, -1, 1, 0, 0],
               [1, -1, -1, 1, 0],
               [0, 1, -1, 1, 0],
               [0, 0, 1, 0, 0]]), 1, 1),
    (np.array([[1, 0, 1, 0],
               [1, 0, 1, 0],
               [1, 1, 1, 0],
               [0, 0, 0, -1]]), 0, 2),
    (_rand, 3, 3 if _rand[3, 3] != 0 else 4),
    (_rand, 0, 0),
]

EXPLANATION = r"""
## Why a flood fill
Connectivity is transitive, so the group is exactly the connected component of `(row, col)` in the graph whose vertices are same-coloured stones and whose edges are orthogonal adjacency. Any graph search (stack, queue, recursion) visits each stone of the component once. Liberties fall out for free: they are the empty vertices on the component's boundary.

## Complexity
Each stone in the group is pushed once and looks at 4 neighbours, so $O(|\text{group}|)$ time plus the final sort, at most $O(N^2 \log N)$.

## Common bugs
- Counting a liberty once per adjacent stone instead of once per point (use a set).
- Treating diagonals as connected.
- Negative indices: `board[-1, c]` is valid NumPy and silently wraps to the last row, so an explicit bounds check is required.
- Recursive DFS on a 19x19 board can hit Python's recursion limit for a large group.

## Interview follow-ups
- Engines call this millions of times per second. How would you keep groups and liberty counts incrementally (union-find with liberty sets, or Zobrist-hashed group IDs) so a move costs $O(1)$ amortised?
- How would you find *all* groups and their liberty counts for the whole board in one pass (label connected components, e.g. with `scipy.ndimage.label`)?
"""
