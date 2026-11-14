import numpy as np

META = dict(title='One GCN layer', tags=['Neural Networks', 'Linear Algebra'], diff='Easy', runtime='numpy',
            related=['normalized-adjacency'])

STATEMENT = r"""
Write `gcn_layer(A_hat, H, W, b, relu)` for a normalised adjacency `A_hat (n, n)`, node features `H (n, f)`, weights `W (f, h)` and bias `b (h,)`:

$$H' = \sigma(\hat A\,H\,W + b),$$

with $\sigma$ = ReLU if `relu` is true and the identity otherwise. Return `H'` of shape `(n, h)`.
"""

STARTER = '''import numpy as np

def gcn_layer(A_hat, H, W, b, relu):
    """Aggregate neighbour features with A_hat, then transform."""
    Z = np.asarray(H) @ np.asarray(W) + np.asarray(b)
    return np.maximum(Z, 0) if relu else Z
'''

SOLUTION = '''import numpy as np

def gcn_layer(A_hat, H, W, b, relu):
    Z = np.asarray(A_hat, float) @ (np.asarray(H, float) @ np.asarray(W, float)) + np.asarray(b, float)
    return np.maximum(Z, 0.0) if relu else Z
'''

HINTS = [
    'A_hat (H W) and (A_hat H) W are equal; compute H W first when h is smaller than f, because that matrix is cheaper to multiply by A_hat.',
    'The bias is added after aggregation, once per node.',
]

def _ahat(edges, n):
    A = np.eye(n)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    d = A.sum(1) ** -0.5
    return d[:, None] * A * d[None, :]

_r = np.random.default_rng(3)

TESTS = [
    (np.eye(2), [[1.0, 0.0], [0.0, 1.0]], [[1.0], [-1.0]], [0.0], True),
    (_ahat([(0, 1)], 2), [[1.0], [3.0]], [[1.0]], [0.0], False),
    (_ahat([(0, 1), (1, 2)], 3), [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[1.0, -1.0], [0.5, 0.5]], [0.1, -0.1], True),
    (_ahat([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 5), _r.normal(size=(5, 4)), _r.normal(size=(4, 3)), _r.normal(size=3), True),
    (_ahat([(0, 1), (2, 3)], 4), _r.normal(size=(4, 2)), _r.normal(size=(2, 2)), np.zeros(2), False),
]

EXPLANATION = r"""
## Message passing in one line
$\hat A H$ replaces each node's features with a degree-weighted average of itself and its neighbours; $W$ then mixes the features. After one layer a node knows about its 1-hop neighbourhood, after two layers its 2-hop neighbourhood, and so on.

## Why it works on citation graphs
Papers mostly cite papers on the same topic (homophily), so averaging over neighbours removes noise from a paper's own bag-of-words features. On graphs where linked nodes tend to differ (heterophily), this averaging can hurt, and models that keep a node's own features separate do better; the next step builds one.

## Cost
$\hat A$ is sparse with one entry per edge, so a layer costs $O(|E|\,h + n f h)$, not $O(n^2)$. Real implementations never form $\hat A$ as a dense matrix.

## Interview follow-ups
- A two-layer GCN only sees 2-hop neighbourhoods. Why not stack 20 layers to see further?
"""
