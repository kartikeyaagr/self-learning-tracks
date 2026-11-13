import numpy as np

META = dict(title='Pick the next edge with PUCT', tags=['Reinforcement Learning'], diff='Easy', runtime='numpy',
            lessons=[],
            related=['initialize-legal-mcts-edge-statistics', 'back-up-mcts-simulation-path',
                     'run-deterministic-neural-guided-mcts'])

STATEMENT = r"""
At every internal node, AlphaGo-style search walks down the edge that maximises value plus an exploration bonus scaled by the network prior:

$$a^* = \arg\max_{a \text{ legal}} \; Q(a) + c_\text{puct}\, P(a)\, \frac{\sqrt{\max(1, \sum_b N(b))}}{1 + N(a)}, \qquad Q(a) = \begin{cases} W(a)/N(a) & N(a) > 0 \\ 0 & N(a) = 0 \end{cases}$$

The $\max(1, \cdot)$ makes the very first selection at a fresh node follow the prior (with $\sqrt{0}$ every bonus would vanish).

Write `puct_select(P, N, W, legal, c_puct)` returning the chosen action as a Python `int`.

## Inputs
- `P`: float `(A,)` priors; `N`: int `(A,)` visit counts; `W`: float `(A,)` total values (from the point of view of the player choosing at this node); `legal`: bool `(A,)` with at least one `True`; `c_puct`: positive float.
- $\sum_b N(b)$ runs over **all** actions.

## Ties
If several legal actions share the maximum score exactly, return the **smallest** index. Illegal actions are never returned, whatever their score.
"""

STARTER = '''import numpy as np

def puct_select(P, N, W, legal, c_puct):
    """Return the int index of the legal action with the highest Q + U (ties: smallest index)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def puct_select(P, N, W, legal, c_puct):
    P, W = np.asarray(P, dtype=float), np.asarray(W, dtype=float)
    N = np.asarray(N)
    Q = np.divide(W, N, out=np.zeros_like(W), where=N > 0)       # unvisited edges have Q = 0
    U = c_puct * P * np.sqrt(max(1, int(N.sum()))) / (1 + N)
    score = np.where(legal, Q + U, -np.inf)
    return int(np.argmax(score))                                  # argmax returns the first maximum
'''

HINTS = [
    'Compute `Q` with a guard for `N == 0`: `np.divide(W, N, out=np.zeros_like(W), where=N > 0)` avoids the 0/0.',
    'The bonus is `c_puct * P * sqrt(max(1, N.sum())) / (1 + N)`. Add it to `Q` to get a score per action.',
    'Set illegal scores to `-np.inf` and return `int(np.argmax(score))`; `np.argmax` already returns the first index among exact ties.',
]

_rng = np.random.default_rng(19)
_P = _rng.dirichlet(np.ones(10))
_N = _rng.integers(0, 30, 10)
_W = np.where(_N > 0, _rng.uniform(-1, 1, 10) * _N, 0.0)

TESTS = [
    (np.array([0.2, 0.5, 0.3]), np.array([0, 0, 0]), np.zeros(3), np.array([True, True, True]), 1.0),
    (np.array([0.5, 0.5]), np.array([3, 1]), np.array([1.5, -0.5]), np.array([True, True]), 1.0),
    (np.array([0.25, 0.25, 0.5]), np.array([0, 0, 0]), np.zeros(3), np.array([True, True, False]), 2.0),
    (np.array([0.5, 0.5]), np.array([1, 1]), np.array([0.25, 0.25]), np.array([True, True]), 1.0),
    (np.array([0.1, 0.9]), np.array([10, 0]), np.array([8.0, 0.0]), np.array([True, True]), 0.1),
    (np.array([0.1, 0.9]), np.array([10, 0]), np.array([8.0, 0.0]), np.array([True, True]), 4.0),
    (_P, _N, _W, np.ones(10, dtype=bool), 1.5),
    (_P, _N, _W, np.array([True, False] * 5), 1.5),
]

EXPLANATION = r"""
## Why it works
$Q$ is exploitation: the average outcome seen through that edge. $U$ is exploration: large for edges the network likes ($P$) and for edges visited rarely relative to their siblings ($\sqrt{\sum N} / (1 + N)$). As a node's total visits grow, $U$ shrinks like $1/\sqrt{\sum N}$ for heavily visited edges, so visits concentrate on moves whose $Q$ holds up. The prior lets the search ignore most of the 362 moves without ever trying them.

## The two exact-tie tests
Test 3 starts a fresh node with two equal priors: both scores are exactly $2.0 \cdot 0.25$, and the smallest index wins. Test 4 has two identical edges. Equal inputs give bit-identical scores, so the tie rule is well defined.

## Complexity
$O(A)$ per selection; a simulation of depth $d$ costs $O(dA)$.

## Common bugs
- `W / N` without a guard, producing `nan` that `argmax` treats as the maximum.
- Summing visits over legal actions only, or using the parent's visit count plus one (another valid variant, but not this statement).
- Using $\sqrt{\sum N}$ without the $\max(1, \cdot)$ and then always picking index 0 at fresh nodes.
- Taking $Q$ from the wrong player's perspective: $W$ must be stored from the viewpoint of the player choosing at this node.

## Interview follow-ups
- How does PUCT differ from UCT's $\sqrt{\ln N / N(a)}$ term, and why does a learned prior allow the gentler exploration schedule?
- What is "virtual loss" and how does it let several threads descend the same tree without all choosing the same path?
"""
