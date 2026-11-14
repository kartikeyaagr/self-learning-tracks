import numpy as np

META = dict(title='GraphSAGE mean aggregation', tags=['Neural Networks'], diff='Medium', runtime='numpy',
            related=['gcn-layer'])

STATEMENT = r"""
GraphSAGE keeps a node's own features separate from the summary of its neighbours.

Write `sage_layer(edges, H, W_self, W_neigh, b, normalize)` for an undirected graph given as an edge list (duplicates and self-edges ignored, as in the adjacency step):

1. $m_v$ = mean of $H_u$ over the neighbours $u$ of $v$, or the zero vector if $v$ has none.
2. $z_v = H_v W_{self} + m_v W_{neigh} + b$, then ReLU.
3. If `normalize`, divide each row by its L2 norm (leave all-zero rows as zero).

Return the `(n, h)` output, where $n$ is the number of rows of `H`.
"""

STARTER = '''import numpy as np

def sage_layer(edges, H, W_self, W_neigh, b, normalize):
    """Self features and the mean of neighbour features, transformed separately."""
    H = np.asarray(H, float)
    n = len(H)
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] += 1
    M = A @ H
    return np.maximum(H @ W_self + M @ W_neigh + b, 0)
'''

SOLUTION = '''import numpy as np

def sage_layer(edges, H, W_self, W_neigh, b, normalize):
    H = np.asarray(H, float)
    n = len(H)
    A = np.zeros((n, n))
    for i, j in edges:
        if i != j:
            A[i, j] = A[j, i] = 1.0
    deg = A.sum(1, keepdims=True)
    M = np.where(deg > 0, (A @ H) / np.where(deg > 0, deg, 1.0), 0.0)
    Z = np.maximum(H @ np.asarray(W_self, float) + M @ np.asarray(W_neigh, float) + np.asarray(b, float), 0.0)
    if normalize:
        nr = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = np.where(nr > 0, Z / np.where(nr > 0, nr, 1.0), 0.0)
    return Z
'''

HINTS = [
    'Mean, not sum: divide the neighbour sum by the degree, and guard nodes with degree 0.',
    'Two weight matrices let the model learn how much to trust a node versus its neighbourhood.',
]

_r = np.random.default_rng(5)

TESTS = [
    ([(0, 1)], [[1.0], [3.0]], [[1.0]], [[1.0]], [0.0], False),
    ([(0, 1), (0, 2)], [[0.0], [2.0], [4.0]], [[1.0]], [[0.5]], [0.0], False),
    ([(0, 1)], [[1.0], [3.0], [5.0]], [[1.0]], [[1.0]], [0.0], False),
    ([(0, 1), (1, 2), (2, 3), (1, 0)], _r.normal(size=(4, 3)), _r.normal(size=(3, 2)), _r.normal(size=(3, 2)), _r.normal(size=2), True),
    ([(0, 1), (1, 2), (2, 0), (3, 4)], _r.normal(size=(5, 4)), _r.normal(size=(4, 3)), _r.normal(size=(4, 3)), np.zeros(3), True),
]

EXPLANATION = r"""
## Keeping yourself apart
A GCN mixes a node's features into the same average as its neighbours'. GraphSAGE (Hamilton et al., 2017) gives the node and its neighbourhood separate weights, so the model can learn to rely on a node's own features when its neighbours are uninformative or misleading. That matters on heterophilous graphs and for nodes with very few links.

## Inductive by design
The layer is a function of a node's features and its neighbours' features, not of a fixed node index, so it applies unchanged to new nodes and even new graphs. GraphSAGE also samples a fixed number of neighbours per node, which bounds the cost on graphs where some nodes have millions of links.

## Interview follow-ups
- Mean, max-pooling and LSTM aggregators were all proposed. Which can tell apart a node with neighbours {a, a} from one with {a}?
"""
