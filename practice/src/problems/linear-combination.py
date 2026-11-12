import numpy as np

META = dict(title='Linear combinations of vectors', tags=['Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:04.01'], related=['matrix-vector-multiply', 'dot-product', 'gram-schmidt-orthogonalization'])

STATEMENT = r"""
Given vectors $v_1, \dots, v_k \in \mathbb{R}^d$ and scalars $c_1, \dots, c_k$, their linear combination is

$$c_1 v_1 + c_2 v_2 + \dots + c_k v_k$$

The span of the $v_i$ is the set of all such combinations. Mixing value vectors with attention weights, interpolating embeddings and averaging model checkpoints are all linear combinations.

Write `combine(coeffs, vectors)`.

## Inputs
- `vectors`: float array `(k, d)` with $k, d \ge 1$; row $i$ is $v_i$.
- `coeffs`: either a 1-D array `(k,)` holding one set of coefficients, or a 2-D array `(m, k)` holding $m$ sets, one per row.

## Output
- For 1-D `coeffs`: the combined vector, shape `(d,)`.
- For 2-D `coeffs`: shape `(m, d)`, where row $r$ is $\sum_i$ `coeffs[r, i] * vectors[i]`.

## Requirements
No Python loops over $k$ or $m$.
"""

STARTER = '''import numpy as np

def combine(coeffs, vectors):
    """coeffs: (k,) or (m, k). vectors: (k, d). Return (d,) or (m, d)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def combine(coeffs, vectors):
    c = np.asarray(coeffs, dtype=float)
    V = np.asarray(vectors, dtype=float)
    # Row-vector times matrix: each output row is the coefficient-weighted sum of the rows of V.
    # The same expression handles (k,) @ (k, d) -> (d,) and (m, k) @ (k, d) -> (m, d).
    return c @ V
'''

HINTS = [
    'Write out the 1-D case on paper for $k = 2$, $d = 3$. Which matrix product has exactly those entries?',
    'A row vector of coefficients times a matrix whose **rows** are the vectors gives their weighted sum. Stacking several coefficient rows stacks several results.',
    '`np.asarray(coeffs, float) @ np.asarray(vectors, float)` handles both shapes.',
]

_r = np.random.default_rng(13)
TESTS = [
    (np.array([2.0, -1.0]), np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])),
    (np.array([[1.0, 1.0], [0.5, 0.5], [1.0, -1.0]]), np.array([[1.0, 2.0], [3.0, 4.0]])),
    (np.array([0.0, 0.0, 0.0]), np.array([[1.0], [2.0], [3.0]])),
    (np.array([3.0]), np.array([[1.0, -2.0, 0.5, 4.0]])),
    (np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), np.eye(3)),
    (np.array([1e8, -1e8]), np.array([[1.0, 2.0], [1.0, 2.0 + 1e-8]])),
    (_r.normal(size=5), _r.normal(size=(5, 7))),
    (_r.dirichlet(np.ones(4), size=3), _r.normal(size=(4, 6))),
]

EXPLANATION = r"""
## Why a matrix product
Entry $j$ of $\sum_i c_i v_i$ is $\sum_i c_i V_{ij}$, which is exactly entry $j$ of the row-vector-times-matrix product $c^\top V$. With several coefficient rows stacked into $C$, row $r$ of $CV$ is the combination for row $r$. Read the other way, $CV$ says: **every row of a matrix product is a linear combination of the rows of the right-hand factor**, and every column is a combination of the columns of the left factor.

Test 8 uses rows of a Dirichlet sample, which are non-negative and sum to 1, so each output is a convex combination (a weighted average). That is what one row of attention computes.

## Complexity
$O(mkd)$ time, $O(md)$ for the output. BLAS-backed `@` is far faster than a Python loop.

## Common bugs
- `vectors @ coeffs`: wrong orientation; it fails unless $d = k$, and then gives a meaningless answer.
- `(coeffs * vectors).sum(0)`: correct for 1-D `coeffs` only if you add `[:, None]`, and wrong for the 2-D case.

## Interview follow-ups
- How do you test whether a target vector $t$ lies in the span of the $v_i$? (Solve least squares $\min_c \lVert c^\top V - t\rVert$ and check the residual is ~0, or compare matrix ranks with and without $t$.)
- When are the coefficients of a combination unique? (When the $v_i$ are linearly independent.)
"""
