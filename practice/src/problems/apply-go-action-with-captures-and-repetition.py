import numpy as np

META = dict(title='Play a Go move: captures, ko and superko', tags=['Reinforcement Learning'], diff='Hard',
            runtime='numpy', lessons=['air-fanout-v1:07.01'],
            related=['build-legal-go-action-mask', 'find-go-group-and-its-liberties',
                     'encode-alphago-zero-board-history'])

STATEMENT = r"""
This is the environment's `step` function for Go. It applies one action and reports what happened, including the information the next turn's legality check needs.

The board is an `(N, N)` integer array: `0` empty, `1` black, `-1` white. Action `r * N + c` places a stone at `(r, c)`; action `N * N` is pass.

Write `play(board, to_play, action, history)`.

## Placing a stone
1. The point must be empty, else the move is illegal.
2. Put a `to_play` stone there. Remove every adjacent **opponent** group left with zero liberties; `captured` is the total number of stones removed.
3. If the new stone's group has zero liberties after step 2, the move is suicide: illegal.
4. **Positional superko**: if the resulting board is equal (every point) to any board in `history`, the move is illegal. `history` is a list of `(N, N)` arrays, the positions that have already occurred in this game (it may be empty).

## Ko point
After a legal placement, compute `ko_point`: if exactly one stone was captured, **and** the new stone's group is that single stone, **and** that stone has exactly one liberty, then `ko_point` is the action index of the captured point (the opponent may not retake there immediately). Otherwise `ko_point = -1`.

## Return
- Illegal move: return `None`.
- Legal placement: `(new_board, captured, ko_point)`, with `new_board` a new int array (do not modify the input).
- Pass: always legal, never checked against `history`; return `(copy of board, 0, -1)`.
"""

STARTER = '''import numpy as np

def play(board, to_play, action, history):
    """Return None if illegal, else (new_board, captured, ko_point)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _group(board, r, c):
    """(stones, liberties) sets of the group containing (r, c)."""
    n, color = board.shape[0], board[r, c]
    stones, libs, stack = {(r, c)}, set(), [(r, c)]
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < n and 0 <= nx < n:
                if board[ny, nx] == 0:
                    libs.add((ny, nx))
                elif board[ny, nx] == color and (ny, nx) not in stones:
                    stones.add((ny, nx))
                    stack.append((ny, nx))
    return stones, libs

def play(board, to_play, action, history):
    board = np.asarray(board)
    n = board.shape[0]
    if action == n * n:
        return board.copy(), 0, -1
    r, c = divmod(action, n)
    if board[r, c] != 0:
        return None
    new = board.copy()
    new[r, c] = to_play
    captured_pts = set()
    for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= y < n and 0 <= x < n and new[y, x] == -to_play and (y, x) not in captured_pts:
            stones, libs = _group(new, y, x)
            if not libs:
                captured_pts |= stones
    for y, x in captured_pts:                       # remove only after every neighbour group was judged
        new[y, x] = 0
    stones, libs = _group(new, r, c)
    if not libs:
        return None                                 # suicide
    if any(np.array_equal(new, h) for h in history):
        return None                                 # positional superko
    ko = -1
    if len(captured_pts) == 1 and len(stones) == 1 and len(libs) == 1:
        (ky, kx), = captured_pts
        ko = ky * n + kx
    return new, len(captured_pts), ko
'''

HINTS = [
    'Work on a copy. Pass is a special case that returns immediately; everything else starts with "is the point empty?".',
    'After placing the stone, flood-fill each adjacent opponent group; collect the stones of groups with no liberties into one set (two neighbours can belong to the same group, so a set avoids double counting), then clear them.',
    'Then flood-fill your own new group: no liberties means suicide. Compare against every board in `history` with `np.array_equal`. Finally the ko point exists only when `len(captured) == 1`, your group is one stone, and it has exactly one liberty: the captured point.',
]

_ko = np.array([[0, -1, 1, 0],
                [-1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])
_ko_after = np.array([[1, 0, 1, 0],
                      [-1, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
_big = np.array([[0, -1, -1, 1, 0],
                 [-1, 1, 1, -1, 0],
                 [0, -1, -1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]])
_rng = np.random.default_rng(31)
_rand = _rng.integers(-1, 2, size=(6, 6))

TESTS = [
    (np.array([[0, 1, 0],
               [1, -1, 1],
               [0, 0, 0]]), 1, 7, []),
    (_ko, 1, 0, []),
    (_ko, 1, 0, [_ko_after]),
    (np.array([[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]]), -1, 4, []),
    (_ko, -1, 16, [_ko]),
    (_ko, 1, 1, []),
    (_big, -1, 0, [np.zeros((5, 5), dtype=int)]),
    (_big, 1, 0, []),
    (np.array([[0, -1, 1],
               [-1, 1, 1],
               [1, 1, 0]]), -1, 8, []),
    (_rand, 1, int(np.flatnonzero(_rand.ravel() == 0)[0]), []),
]

EXPLANATION = r"""
## Order of resolution
Captures first, suicide second, repetition last. Captures must be judged against the board **with** the new stone but **before** any removal, otherwise removing one group could change whether a neighbouring group counts as captured. Collecting all captured stones into a set first and clearing them afterwards avoids that and handles two neighbours that belong to the same group.

## Ko point vs superko
The ko point is a cheap local rule: the only immediate repetition in Go is a single stone capturing a single stone and being instantly recapturable. Superko compares the whole board against every previous position, which also forbids longer cycles (triple ko, sending-two-returning-one). Self-play engines usually track both: the ko point feeds the fast legal-move mask, and Zobrist hashes of past positions catch rare long cycles.

Test 3 replays test 2 with the resulting position already in `history`: the same capture becomes illegal.

## Complexity
Up to 5 flood fills of $O(N^2)$ plus the history check, $O(|\text{history}| \cdot N^2)$. With 64-bit Zobrist hashes kept in a set, the history check drops to $O(1)$.

## Common bugs
- Removing an opponent group before judging the next neighbour.
- Counting a captured stone twice when two neighbours share a group.
- Setting a ko point after a multi-stone capture, or after a single-stone capture by a stone that joins a bigger group.
- Checking passes against history: a pass repeats the position by definition and is always legal.

## Interview follow-ups
- How do Zobrist hashes let you update the position hash in $O(\text{stones changed})$ per move?
- Positional vs situational superko: what is the difference, and why does it matter for a pass-heavy endgame?
"""
