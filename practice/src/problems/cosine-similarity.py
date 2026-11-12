import numpy as np

META = dict(title='Cosine similarity of two vectors', tags=['Linear Algebra', 'Embeddings'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:02.01', 'aieng:02.02'], related=['dot-product', 'vector-norms', 'cosine-similarity-matrix'])

STATEMENT = r"""
Cosine similarity compares the **direction** of two vectors and ignores their length:

$$\cos(a, b) = \frac{a \cdot b}{\lVert a\rVert_2\,\lVert b\rVert_2}$$

It is the standard score for comparing text or image embeddings. Write `cosine_similarity(a, b)`.

## Inputs
- `a`, `b`: 1-D float arrays of the same length $d \ge 1$.

## Output
- A Python float in $[-1, 1]$.

## Conventions
- If either vector is all zeros, the angle is undefined; return `0.0`.
- Rounding can push the ratio slightly outside $[-1, 1]$ (for example `1.0000000000000002`). Clip the result into that range.
- Must work for very large or very small magnitudes, e.g. entries around `1e200` or `1e-200`, without overflow or underflow turning the answer into `nan` or `0`.
"""

STARTER = '''import numpy as np

def cosine_similarity(a, b):
    """a, b: 1-D arrays of equal length. Return cos(angle) as a float in [-1, 1]; 0.0 if either is zero."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ma, mb = np.abs(a).max(), np.abs(b).max()
    if ma == 0 or mb == 0:
        return 0.0
    # Cosine ignores scale, so divide by the largest entry first: squares of 1e200 or 1e-200
    # would overflow / underflow, but entries in [-1, 1] are safe.
    a, b = a / ma, b / mb
    c = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(c, -1.0, 1.0))
'''

HINTS = [
    'Handle the zero-vector case before you divide by anything.',
    'Cosine does not change if you rescale either vector. `np.linalg.norm` squares the entries, so `1e200` overflows to `inf` and `1e-200` underflows to 0. Rescale each vector so its largest entry is 1 before doing anything else.',
    '`ma, mb = abs(a).max(), abs(b).max()`; return 0.0 if either is 0; then `a, b = a/ma, b/mb` and return `float(np.clip(a @ b / (norm(a) * norm(b)), -1, 1))`.',
]

_r = np.random.default_rng(5)
TESTS = [
    (np.array([1.0, 0.0]), np.array([1.0, 1.0])),
    (np.array([1.0, 2.0, 3.0]), np.array([-2.0, -4.0, -6.0])),
    (np.array([3.0, -1.0]), np.array([1.0, 3.0])),
    (np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0])),
    (np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])),
    (np.array([1e200, 2e200]), np.array([3e200, -1e200])),
    (np.array([1e-200, 1e-200]), np.array([2e-200, 0.0])),
    (np.array([7.0]), np.array([-0.5])),
    (_r.normal(size=16), _r.normal(size=16)),
]

EXPLANATION = r"""
## Why it works
From $a\cdot b = \lVert a\rVert\lVert b\rVert\cos\theta$, dividing by both norms leaves $\cos\theta$. Scaling either vector by a positive constant cancels out, which is why cosine is the natural score for embeddings whose length carries little meaning. The same invariance is what makes the computation safe: dividing each vector by its largest absolute entry changes nothing mathematically, but puts every entry in $[-1, 1]$, so the squares inside the norms can neither overflow nor underflow.

## Complexity
$O(d)$ time, $O(d)$ extra memory for the two rescaled copies.

## Common bugs
- Dividing by zero for a zero vector and returning `nan`.
- Assuming `np.linalg.norm` is overflow-safe. It is not: it squares the entries, so `[1e200, 2e200]` has norm `inf` and `[1e-200, 1e-200]` has norm `0.0`, turning the answer into `nan`.
- Returning `1.0000000000000002` for identical vectors, which breaks a later `arccos`.

## Interview follow-ups
- If all embeddings are L2-normalised, how do cosine similarity and Euclidean distance relate? ($\lVert a-b\rVert^2 = 2 - 2\cos(a,b)$, so they rank neighbours identically.)
- How would you score one query against a million stored embeddings quickly? (Normalise everything once at index time, then it's a single matrix-vector product.)
"""
