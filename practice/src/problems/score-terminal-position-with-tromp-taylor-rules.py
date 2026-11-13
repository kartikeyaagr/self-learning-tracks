import numpy as np

META = dict(title='Area-score a finished Go board (Tromp-Taylor)', tags=['Reinforcement Learning'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:07.01'],
            related=['find-go-group-and-its-liberties', 'apply-go-action-with-captures-and-repetition'])

STATEMENT = r"""
Self-play needs an unambiguous winner for every finished game, with no human judging which stones are "dead". Tromp-Taylor area scoring gives one:

- A player's score is the number of points **of their colour**, plus the number of **empty points that reach only their colour**.
- An empty point *reaches* a colour if a path of orthogonally adjacent empty points leads from it to a point next to a stone of that colour. Equivalently: split the empty points into connected regions; a region that borders only black stones counts for black, one that borders only white stones counts for white, and a region bordering both colours, or no stones at all, counts for nobody.

The board is an `(N, N)` integer array: `0` empty, `1` black, `-1` white. Score it exactly as it stands (no stones are removed).

Write `tromp_taylor_score(board, komi)` returning `(black, white, margin)`:

- `black`, `white`: Python ints, the two area scores.
- `margin`: Python float, `black - white - komi`. Positive means black wins; `komi` compensates white for moving second.
"""

STARTER = '''import numpy as np

def tromp_taylor_score(board, komi):
    """board: (N, N) ints in {-1, 0, 1}. Return (black_area, white_area, black - white - komi)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def tromp_taylor_score(board, komi):
    board = np.asarray(board)
    n = board.shape[0]
    score = {1: int((board == 1).sum()), -1: int((board == -1).sum())}
    seen = np.zeros(board.shape, dtype=bool)
    for r0 in range(n):
        for c0 in range(n):
            if board[r0, c0] != 0 or seen[r0, c0]:
                continue
            # flood-fill one empty region, recording which colours touch it
            seen[r0, c0] = True
            stack, size, borders = [(r0, c0)], 0, set()
            while stack:
                r, c = stack.pop()
                size += 1
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < n and 0 <= nc < n:
                        v = int(board[nr, nc])
                        if v != 0:
                            borders.add(v)
                        elif not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            if len(borders) == 1:                       # territory only if exactly one colour touches it
                score[borders.pop()] += size
    return score[1], score[-1], float(score[1] - score[-1] - komi)
'''

HINTS = [
    'Stones count directly. The only work is deciding who owns each connected region of empty points.',
    'Flood-fill each unvisited empty point. While filling, collect the set of non-empty colours adjacent to the region.',
    'After the fill, if that set is exactly `{1}` add the region size to black, if exactly `{-1}` add it to white, otherwise (both, or empty board) add it to nobody. Mark visited points so each region is filled once.',
]

_rng = np.random.default_rng(11)
_rand = _rng.integers(-1, 2, size=(9, 9)) * (_rng.integers(0, 3, size=(9, 9)) > 0)

TESTS = [
    (np.array([[0, 1, -1, 0],
               [0, 1, -1, 0],
               [0, 1, -1, 0],
               [0, 1, -1, 0]]), 0.5),
    (np.array([[0, 1, 0],
               [1, 0, -1],
               [0, -1, 0]]), 0.0),
    (np.zeros((5, 5), dtype=int), 7.5),
    (np.array([[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]]), 6.5),
    (np.array([[0, 0, 1, -1, 0],
               [0, 0, 1, -1, 0],
               [1, 1, 1, -1, 0],
               [-1, -1, -1, -1, 0],
               [0, 0, 0, 0, 0]]), 2.5),
    (np.array([[0, 1, 0, -1, 0],
               [1, 1, 1, -1, -1],
               [0, 1, -1, 0, -1],
               [1, 1, -1, -1, 0],
               [0, 1, -1, 0, 1]]), 0.5),
    (_rand, 7.5),
]

EXPLANATION = r"""
## Why area scoring suits self-play
Territory scoring needs agreement on which stones are dead, which is a judgement call. Tromp-Taylor needs none: the agents simply keep playing until dead stones are actually captured, and the board then scores itself. With area scoring, filling your own territory costs nothing, so playing on never changes the result, which is what makes this rule safe for an agent that does not know when to stop.

## Correctness
Every empty point lies in exactly one connected empty region, and all points in a region reach the same set of colours (any path between them stays inside the region). So deciding ownership per region, not per point, is exact. Neutral regions (*dame*) bordering both colours, and an empty board, score for nobody.

## Complexity
Each point is pushed at most once and checks 4 neighbours: $O(N^2)$.

## Common bugs
- Scoring a region as soon as it touches one colour, before the fill has seen the whole border.
- Forgetting to mark points as visited when they are **pushed**, so a point gets counted twice.
- Returning `black - white` without komi, or the sign the wrong way round.

## Interview follow-ups
- AlphaGo Zero's value target is $z = \pm 1$ from this margin. Why not regress the margin itself, and what did later engines (KataGo) gain by predicting score and ownership as auxiliary targets?
- How would you vectorise region labelling (connected-component labelling of the empty mask, then a dilation per label to read the border colours)?
"""
