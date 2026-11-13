import numpy as np

META = dict(title='Legal-move mask for Go', tags=['Reinforcement Learning'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:07.01'],
            related=['find-go-group-and-its-liberties', 'apply-go-action-with-captures-and-repetition',
                     'initialize-legal-mcts-edge-statistics'])

STATEMENT = r"""
A policy network outputs a score for every action, but search and sampling must only ever pick legal ones. You will build the boolean mask that filters them.

The board is an `(N, N)` integer array: `0` empty, `1` black, `-1` white. There are $A = N^2 + 1$ actions: action `r * N + c` places a stone at `(r, c)`, and action `N * N` is **pass**.

Placing a stone of colour `to_play` at an empty point works in this order:

1. Put the stone down.
2. Every **opponent** group orthogonally adjacent to it that now has zero liberties is captured (removed).
3. If the group containing the new stone now has zero liberties, the move is **suicide**.

A placement at `(r, c)` is legal iff all of the following hold:

- the point is empty;
- it is not suicide (step 3, judged *after* the captures of step 2, so a move that captures is never suicide);
- its action index is not `ko_point`. `ko_point` is an action index forbidden by the simple ko rule, or `-1` for none.

Pass is always legal.

Write `legal_mask(board, to_play, ko_point)` returning a boolean array of shape `(N * N + 1,)`.

## Notes
- A group is a maximal set of same-coloured stones connected orthogonally; a liberty is an empty point orthogonally adjacent to the group.
- Do not modify `board`.
"""

STARTER = '''import numpy as np

def legal_mask(board, to_play, ko_point):
    """board: (N, N) ints in {-1, 0, 1}; to_play: 1 or -1; ko_point: action index or -1.
    Return a bool array of shape (N*N + 1,), pass last."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _neighbors(r, c, n):
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < n and 0 <= nc < n:
            yield nr, nc

def _has_liberty(board, r, c):
    """True if the group containing (r, c) touches an empty point."""
    n, color = board.shape[0], board[r, c]
    stack, seen = [(r, c)], {(r, c)}
    while stack:
        y, x = stack.pop()
        for ny, nx in _neighbors(y, x, n):
            if board[ny, nx] == 0:
                return True                         # one liberty is enough: stop early
            if board[ny, nx] == color and (ny, nx) not in seen:
                seen.add((ny, nx))
                stack.append((ny, nx))
    return False

def legal_mask(board, to_play, ko_point):
    board = np.asarray(board)
    n = board.shape[0]
    mask = np.zeros(n * n + 1, dtype=bool)
    mask[n * n] = True                              # pass
    for r in range(n):
        for c in range(n):
            a = r * n + c
            if board[r, c] != 0 or a == ko_point:
                continue
            nb = [(y, x) for y, x in _neighbors(r, c, n)]
            if any(board[y, x] == 0 for y, x in nb):
                mask[a] = True                      # an empty neighbour is already a liberty
                continue
            trial = board.copy()
            trial[r, c] = to_play
            # Capturing any adjacent opponent group frees a liberty, so the move is legal.
            if any(trial[y, x] == -to_play and not _has_liberty(trial, y, x) for y, x in nb):
                mask[a] = True
                continue
            # No capture: legal iff the merged friendly group still has a liberty.
            mask[a] = _has_liberty(trial, r, c)
    return mask
'''

HINTS = [
    'Pass is always legal and occupied points never are. The interesting points are empty ones whose four neighbours are all occupied.',
    'If an empty point has an empty neighbour, the new stone has a liberty: legal (unless it is the ko point). Otherwise place the stone on a copy and ask: does it capture an adjacent opponent group (that group has no liberties left)? If yes, legal.',
    'If it captures nothing, the move is legal exactly when the friendly group containing the new stone still has a liberty somewhere. Write one `_has_liberty(board, r, c)` flood fill and use it for both checks; the ko point is masked out at the very start.',
]

_rng = np.random.default_rng(23)
_rand = _rng.integers(-1, 2, size=(7, 7))

TESTS = [
    (np.array([[0, 1, 0],
               [1, 0, 0],
               [0, 0, 0]]), -1, -1),
    (np.array([[0, -1, 1, 0],
               [-1, 1, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]), 1, 0),
    (np.array([[0, -1, 1, 0],
               [-1, 1, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]), 1, -1),
    (np.array([[0, 1, -1, 0],
               [1, -1, 0, -1],
               [0, 1, -1, 0],
               [0, 0, 0, 0]]), -1, -1),
    (np.array([[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]]), -1, -1),
    (np.array([[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]]), 1, -1),
    (np.array([[0, 1, -1, 0, -1],
               [1, 1, -1, -1, 0],
               [-1, -1, 0, -1, -1],
               [0, -1, -1, 1, 0],
               [-1, 0, -1, 1, 1]]), 1, -1),
    (_rand, 1, -1),
    (_rand, -1, 10),
]

EXPLANATION = r"""
## Why the order of checks matters
Capture is resolved **before** suicide. A stone dropped into a point with no liberties is still legal if it removes an adjacent opponent group, because that removal creates a liberty right next to it. Test 2 vs test 3 is the classic ko shape: black captures one white stone, and the only thing that forbids it in test 2 is the ko point.

The shortcuts are exact, not heuristics:
- an empty neighbour is a liberty of the new stone, so the move cannot be suicide;
- a capture empties a neighbour, so again the new group has a liberty;
- otherwise nothing is removed, and the new stone merges with adjacent friendly groups; the move is legal iff that merged group touches an empty point somewhere.

## Complexity
Each candidate point may flood-fill a few groups: $O(N^2)$ per point and $O(N^4)$ worst case for the whole mask, in practice far less because most points have an empty neighbour. Real engines track liberties incrementally to make the mask $O(N^2)$.

## Common bugs
- Checking suicide before captures (forbids legal capturing moves).
- Forgetting that the new stone **joins** friendly neighbours: a point with no empty neighbours is fine if a connected friendly group has a liberty elsewhere.
- Mutating the input board while testing a move.
- Forgetting the pass action, or putting it first instead of last.

## Interview follow-ups
- The simple ko rule only forbids immediately retaking one stone. Which repetitions does it miss, and how does positional superko with Zobrist hashing catch them?
- In MCTS, why mask illegal actions in the prior (renormalising) rather than letting the search discover they are bad?
"""
