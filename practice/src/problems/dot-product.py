import numpy as np

META = dict(title='Dot products along the last axis', tags=['Linear Algebra'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:06.01', 'air-fanout-v1:02.01'], related=['cosine-similarity', 'matrix-vector-multiply', 'vector-projection'])

STATEMENT = r"""
The dot product of two vectors $a, b \in \mathbb{R}^d$ is

$$a \cdot b = \sum_{i=1}^{d} a_i b_i$$

In ML code you rarely have just two vectors: you have a batch of embeddings and one query, or two batches paired row by row. Write `dot(a, b)` that computes the dot product **along the last axis**, broadcasting the leading axes the way NumPy does.

## Inputs
- `a`: array of shape `(..., d)`, `b`: array of shape `(..., d)`. The last dimensions are equal; the leading shapes are broadcast-compatible (for example `(n, d)` with `(d,)`, or `(n, d)` with `(n, d)`).

## Output
- A float array with the broadcast leading shape, e.g. `(n,)` for `(n, d)` and `(d,)`. Two 1-D inputs give a scalar (a 0-d result or a Python float are both fine).
- If `d == 0`, every dot product is `0.0`.

## Requirements
- Do it with elementwise multiplication and a sum (or `np.einsum`), not a Python loop over rows.
- Note that `a @ b` does **not** do this for two `(n, d)` batches: it tries a matrix product.
"""

STARTER = '''import numpy as np

def dot(a, b):
    """a: (..., d), b: (..., d), broadcastable. Return the dot product over the last axis."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def dot(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # Multiply elementwise (broadcasting the leading axes), then collapse the shared last axis.
    return (a * b).sum(axis=-1)
'''

HINTS = [
    'The dot product is "multiply matching entries, then add them up". Which NumPy operations do those two steps for a whole batch at once?',
    '`a * b` broadcasts the leading axes for you; then sum over `axis=-1`.',
    '`return (np.asarray(a, float) * np.asarray(b, float)).sum(axis=-1)`. Equivalently `np.einsum("...i,...i->...", a, b)`.',
]

_r = np.random.default_rng(11)
TESTS = [
    (np.array([1.0, 2.0, 3.0]), np.array([4.0, -5.0, 6.0])),
    (np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]), np.array([2.0, 1.0])),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])),
    (np.array([1.0, -1.0]), np.array([1.0, 1.0])),
    (np.zeros(0), np.zeros(0)),
    (np.array([1e150, 1e150]), np.array([1e150, -1e150])),
    (np.array([3, 4]), np.array([3, 4])),
    (_r.normal(size=(5, 8)), _r.normal(size=(5, 8))),
    (_r.normal(size=(2, 3, 4)), _r.normal(size=(4,))),
]

EXPLANATION = r"""
## Why it works
`a * b` lines up the last axes entry by entry and broadcasts the leading axes, so each output position holds the products $a_i b_i$ for one pair of vectors. Summing over `axis=-1` adds those products up. `np.einsum('...i,...i->...', a, b)` says the same thing in index notation: repeat `i`, drop it from the output, so it is summed.

Geometrically, $a\cdot b = \lVert a\rVert\,\lVert b\rVert \cos\theta$: positive when the vectors point the same way, zero when they are orthogonal (test 4), negative when they oppose.

## Complexity
$O(N d)$ time for $N$ pairs. The `a * b` intermediate costs $O(Nd)$ memory; `einsum` can avoid it.

## Common bugs
- Using `a @ b` for two `(n, d)` batches: it raises a shape error, and `a @ b.T` computes all $n^2$ cross products when you only wanted the $n$ diagonal ones.
- Summing over `axis=0` or with no axis, which mixes different pairs together.
- Integer inputs: with `int8` arrays the products can overflow silently. Casting to float first avoids it.

## Interview follow-ups
- How would you compute the dot product of every row of `A` with every row of `B`? (`A @ B.T`, an `(n, m)` matrix.)
- Why do attention scores divide the dot product by $\sqrt{d}$? (For random unit-variance entries the dot product has variance $d$; scaling keeps softmax out of its saturated region.)
"""
