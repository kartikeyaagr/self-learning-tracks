import numpy as np

META = dict(title='Expand an MCTS node: legal edge statistics', tags=['Reinforcement Learning'], diff='Easy',
            runtime='numpy', lessons=[],
            related=['build-legal-go-action-mask', 'select-action-with-puct', 'back-up-mcts-simulation-path'])

STATEMENT = r"""
When MCTS reaches a new position, it evaluates it with the network and **expands** the node: every action gets an edge carrying four statistics.

| Name | Meaning | Initial value |
|---|---|---|
| `P` | prior probability from the policy network | renormalised over legal moves |
| `N` | visit count | `0` |
| `W` | total backed-up value | `0.0` |
| `Q` | mean value `W / N` | `0.0` |

Write `expand_node(priors, legal)` returning a dict with keys `"P"`, `"N"`, `"W"`, `"Q"`, each an array of shape `(A,)`.

## Rules
- `priors`: float array `(A,)` of non-negative network probabilities over **all** actions (it may put mass on illegal moves).
- `legal`: bool array `(A,)` with at least one `True`.
- `P[a] = 0` for illegal `a`. On legal actions, `P` is `priors` rescaled to sum to 1. If the legal actions have total prior mass exactly `0`, use a uniform distribution over legal actions instead.
- `N` is an integer array of zeros; `W` and `Q` are float arrays of zeros.
"""

STARTER = '''import numpy as np

def expand_node(priors, legal):
    """Return {"P": float (A,), "N": int (A,), "W": float (A,), "Q": float (A,)}."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def expand_node(priors, legal):
    legal = np.asarray(legal, dtype=bool)
    p = np.where(legal, np.asarray(priors, dtype=float), 0.0)
    total = p.sum()
    p = p / total if total > 0 else legal / legal.sum()   # fall back to uniform over legal moves
    A = len(p)
    return {'P': p, 'N': np.zeros(A, dtype=int), 'W': np.zeros(A), 'Q': np.zeros(A)}
'''

HINTS = [
    'Zero out the illegal priors first, then normalise what is left.',
    '`np.where(legal, priors, 0.0)` masks; divide by its sum. If that sum is 0, the network gave the legal moves no mass at all.',
    'For the zero-mass case, `legal / legal.sum()` is the uniform distribution over legal moves. Return the four arrays in a dict, with `N` as `np.zeros(A, dtype=int)`.',
]

_rng = np.random.default_rng(3)
_pr = _rng.dirichlet(np.ones(26))
_lg = _rng.integers(0, 2, 26).astype(bool); _lg[-1] = True

TESTS = [
    (np.array([0.5, 0.25, 0.25]), np.array([True, False, True])),
    (np.array([0.1, 0.2, 0.3, 0.4]), np.array([True, True, True, True])),
    (np.array([0.0, 1.0, 0.0]), np.array([True, False, True])),
    (np.array([0.0, 0.0, 0.0, 0.0, 1.0]), np.array([False, False, False, False, True])),
    (np.array([0.7, 0.2, 0.1]), np.array([False, True, False])),
    (_pr, _lg),
    (_pr, np.ones(26, dtype=bool)),
]

EXPLANATION = r"""
## Why renormalise
The policy head may leak probability onto occupied points or ko-forbidden moves, especially early in training. PUCT spends exploration in proportion to $P$, so leaving mass on illegal moves would waste the exploration budget, and simply zeroing them without rescaling would make the legal priors sum to less than 1, shrinking every exploration bonus. Renormalising keeps the relative preferences among legal moves intact.

## The zero-mass fallback
If every legal move received prior 0 (possible after a hard mask or numerical underflow), division by zero would produce `nan` and poison the search. A uniform prior is the neutral choice.

## Why store W and Q separately
Backup only ever adds to `N` and `W`; `Q = W / N` is derived. Keeping `W` avoids accumulating rounding error from repeatedly updating a running mean.

## Complexity
$O(A)$ time and memory per expanded node. With 362 actions on 19x19 and 1600 simulations per move, the tree holds about 580k edges per move, which is why real engines store edge stats in flat arrays instead of per-edge objects.

## Interview follow-ups
- Should unvisited edges start at $Q = 0$, at the parent's value, or at a pessimistic "first play urgency" value? How does each choice change how wide the search is?
- Why might you store priors as float16 in a large tree, and what precision problems can that cause?
"""
