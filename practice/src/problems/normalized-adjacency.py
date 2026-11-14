import numpy as np

META = dict(title='The normalised adjacency matrix', tags=['Linear Algebra', 'Neural Networks'], diff='Easy', runtime='numpy')

STATEMENT = r"""
Write `normalized_adjacency(edges, n, self_loops)` for an undirected graph with `n` nodes.

1. Build the symmetric 0/1 adjacency $A$ from the edge list. Duplicate edges and both directions count once. An edge `(i, i)` in the list is ignored.
2. If `self_loops`, use $\tilde A = A + I$; otherwise $\tilde A = A$.
3. Return $D^{-1/2}\tilde A D^{-1/2}$, where $D$ holds the row sums of $\tilde A$. A node with degree 0 gets a zero row and column.
"""

STARTER = '''import numpy as np

def normalized_adjacency(edges, n, self_loops):
    """Symmetric degree-normalised adjacency, optionally with self-loops."""
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] += 1
    if self_loops:
        A += np.eye(n)
    return A / A.sum(1, keepdims=True)
'''

SOLUTION = '''import numpy as np

def normalized_adjacency(edges, n, self_loops):
    A = np.zeros((n, n))
    for i, j in edges:
        if i != j:
            A[i, j] = A[j, i] = 1.0
    if self_loops:
        A = A + np.eye(n)
    d = A.sum(1)
    inv = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    return inv[:, None] * A * inv[None, :]
'''

HINTS = [
    'Set entries to 1 rather than adding, so duplicates and reverse edges do not double-count.',
    'Scale rows and columns by d^{-1/2} with broadcasting instead of building D as a matrix.',
]

TESTS = [
    ([(0, 1)], 2, True),
    ([(0, 1), (1, 0), (0, 1)], 2, False),
    ([(0, 1), (1, 2)], 4, False),
    ([(0, 1), (1, 2), (2, 0), (2, 3)], 4, True),
    ([(0, 0), (1, 2)], 3, True),
]

EXPLANATION = r"""
## Why normalise
Multiplying features by $A$ sums the neighbours, so high-degree nodes get huge values and low-degree nodes small ones. Normalising turns the sum into a weighted average. The symmetric form $D^{-1/2}\tilde AD^{-1/2}$ (Kipf and Welling, 2017) weights an edge by $1/\sqrt{d_i d_j}$, so a link from a node with thousands of citations counts for less than a link from a node with three.

## Why self-loops
Without them, a node's new representation is built only from its neighbours and its own features are thrown away. Adding $I$ keeps them in the average.

## Interview follow-ups
- The eigenvalues of this matrix lie in $(-1, 1]$. Why does that matter when layers are stacked?
"""
