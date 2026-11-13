import numpy as np

META = dict(title='Back up a leaf value along the search path', tags=['Reinforcement Learning'], diff='Medium',
            runtime='numpy', lessons=[],
            related=['select-action-with-puct', 'initialize-legal-mcts-edge-statistics',
                     'run-deterministic-neural-guided-mcts', 'value-iteration-step'])

STATEMENT = r"""
After a simulation reaches a leaf and the network evaluates it, the value is pushed back up the path so every edge on it learns from the result. In a two-player zero-sum game the sign flips at every ply: a position that is good for the player to move is bad for the player who moved into it.

The tree's statistics are stored as 2-D arrays indexed by `[node, action]`.

Write `backup(N, W, path, leaf_value)` returning `(N_new, W_new, Q_new)`.

## Inputs
- `N`: int array `(num_nodes, A)` of visit counts; `W`: float array `(num_nodes, A)` of total values. Do not modify them in place; return updated copies.
- `path`: list of `(node, action)` pairs from the root down to the leaf, in order. Edge `path[d]` leaves the node at depth `d`. Players alternate at every edge (a pass is also an edge). Each node appears at most once.
- `leaf_value`: float in $[-1, 1]$, the evaluation of the position reached after the last edge, **from the point of view of the player to move there**.

## Update
For every edge `(node, action)` on the path, increment `N[node, action]` by 1 and add to `W[node, action]` the leaf value seen from the perspective of **the player who chose that edge**. With $L$ = `len(path)`, the edge at depth $d$ receives

$$(-1)^{L - d}\; v_\text{leaf}$$

so the last edge gets $-v_\text{leaf}$, the one before it $+v_\text{leaf}$, and so on.

## Output
`Q_new = W_new / N_new` element-wise, with `Q_new = 0.0` wherever `N_new == 0`. An empty path leaves everything unchanged.
"""

STARTER = '''import numpy as np

def backup(N, W, path, leaf_value):
    """N: int (nodes, A); W: float (nodes, A); path: [(node, action), ...] root to leaf.
    Return (N_new, W_new, Q_new)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def backup(N, W, path, leaf_value):
    N = np.array(N, dtype=int)                      # np.array copies, so the inputs stay untouched
    W = np.array(W, dtype=float)
    v = float(leaf_value)
    for node, action in reversed(path):
        v = -v                                      # switch to the viewpoint of the player who chose this edge
        N[node, action] += 1
        W[node, action] += v
    Q = np.divide(W, N, out=np.zeros_like(W), where=N > 0)
    return N, W, Q
'''

HINTS = [
    'Walk the path from the leaf back to the root, carrying the value and negating it once per edge.',
    'The leaf value is from the view of the player to move *at* the leaf, but the last edge was chosen by the *other* player, so negate before the first update.',
    'Copy with `np.array(N, dtype=int)` and `np.array(W, dtype=float)`, loop over `reversed(path)` doing `v = -v; N[n, a] += 1; W[n, a] += v`, then `Q = np.divide(W, N, out=np.zeros_like(W), where=N > 0)`.',
]

_rng = np.random.default_rng(37)
_N = _rng.integers(0, 6, (6, 4))
_W = np.where(_N > 0, _rng.uniform(-1, 1, (6, 4)) * _N, 0.0)

TESTS = [
    (np.zeros((2, 2), dtype=int), np.zeros((2, 2)), [(0, 1)], 0.5),
    (np.zeros((3, 2), dtype=int), np.zeros((3, 2)), [(0, 0), (1, 1)], 0.5),
    (np.array([[2, 1], [1, 0]]), np.array([[1.0, -0.5], [0.25, 0.0]]), [], 0.9),
    (np.array([[4, 0, 2], [0, 3, 0], [1, 0, 0]]), np.array([[2.0, 0.0, -1.0], [0.0, 1.5, 0.0], [-0.5, 0.0, 0.0]]),
     [(0, 2), (2, 0), (1, 1)], -1.0),
    (np.zeros((4, 3), dtype=int), np.zeros((4, 3)), [(0, 2), (1, 0), (2, 1), (3, 2)], 0.75),
    (_N, _W, [(0, 3), (2, 1), (4, 0), (5, 2), (1, 1)], 0.3),
    (_N, _W, [(5, 0), (3, 3)], -0.6),
]

EXPLANATION = r"""
## Why the sign alternates
Values are stored from the viewpoint of the player who makes the choice at each node, because that is the viewpoint PUCT maximises. The network's leaf value is from the viewpoint of whoever moves at the leaf. The parent's player is the opponent, so the leaf's $v$ is worth $-v$ to them, $+v$ to the grandparent's player, and so on. Storing values this way lets every node simply pick the maximum $Q + U$; a single-perspective implementation would need to minimise at alternate levels.

## Why W and N, not a running mean
Adding to sums is exact bookkeeping; the mean $Q = W / N$ is derived on demand. It also makes undoing a virtual loss trivial: subtract what you added.

## Complexity
$O(L)$ updates for a path of length $L$, plus $O(\text{nodes} \cdot A)$ here to recompute the full `Q` array (real engines update only the touched entries).

## Common bugs
- Not negating before the first update, so the last edge receives the leaf's own-perspective value.
- Negating per node instead of per edge after a pass (a pass still switches the player).
- Updating in place and corrupting the caller's arrays.
- `W / N` without a guard producing `nan` for unvisited edges.

## Interview follow-ups
- With virtual loss, what is added to `N` and `W` on the way **down**, and how is it reverted in backup?
- How would the update change for a single-player puzzle, or for a game where a turn can contain several moves by the same player?
"""
