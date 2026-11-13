import numpy as np

META = dict(title='Pairwise distances and nearest neighbours with nested vmap', tags=['JAX', 'Embeddings'],
            diff='Medium', runtime='jax', lessons=['aieng:02.06'],
            related=['pairwise-differences', 'clip-cosine-retrieval', 'batched-attention-with-vmap'], tol=[1e-4, 1e-5])

STATEMENT = r"""
Write the distance between **two vectors**, then turn it into a full distance matrix with two nested `jax.vmap` calls: the inner one maps over the rows of `Y`, the outer one over the rows of `X`.

Write `pairwise(X, Y, metric)`:

- `X`: `(N, D)`; `Y`: `(M, D)`; float32.
- `metric`: Python string, one of
  - `'euclidean'`: $\lVert x - y \rVert_2$
  - `'manhattan'`: $\sum_i |x_i - y_i|$
  - `'cosine'`: $1 - \dfrac{x \cdot y}{\lVert x \rVert\, \lVert y \rVert}$. If either vector is all zeros, the cosine similarity is defined as 0, so the distance is 1.

Return `(dist, nearest)`:

- `dist`: `(N, M)` float32 with `dist[i, j]` the distance between `X[i]` and `Y[j]`.
- `nearest`: `(N,)` int, the index of the closest row of `Y` for each row of `X`. Ties go to the **lowest** index.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def pairwise(X, Y, metric):
    """Return (dist (N, M), nearest (N,))."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def _dist(metric):
    def euclidean(x, y):
        return jnp.sqrt(jnp.sum((x - y) ** 2))

    def manhattan(x, y):
        return jnp.sum(jnp.abs(x - y))

    def cosine(x, y):
        denom = jnp.linalg.norm(x) * jnp.linalg.norm(y)
        sim = jnp.where(denom > 0, jnp.dot(x, y) / jnp.where(denom > 0, denom, 1.0), 0.0)  # safe divide
        return 1.0 - sim

    return {'euclidean': euclidean, 'manhattan': manhattan, 'cosine': cosine}[metric]

def pairwise(X, Y, metric):
    X, Y = jnp.asarray(X, jnp.float32), jnp.asarray(Y, jnp.float32)
    d = _dist(metric)                                     # metric is a Python string: picked at trace time
    row = jax.vmap(d, in_axes=(None, 0))                  # one x against every y   -> (M,)
    dist = jax.vmap(row, in_axes=(0, None))(X, Y)         # every x                 -> (N, M)
    return dist, jnp.argmin(dist, axis=1)                 # argmin returns the first minimum
'''

HINTS = [
    'Each metric is a one-liner on two `(D,)` vectors. Pick the function with a Python dict lookup on `metric`.',
    '`jax.vmap(d, in_axes=(None, 0))` compares one `x` with every row of `Y`. Wrapping that in `jax.vmap(..., in_axes=(0, None))` covers every row of `X`.',
    'For cosine, compute the denominator, and use `jnp.where` both to avoid dividing by zero and to return similarity 0. `jnp.argmin` already breaks ties toward the lowest index.',
]

_r = np.random.default_rng(15)
_Y = _r.normal(size=(5, 3))
TESTS = [
    (np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([[3.0, 4.0], [1.0, 0.0]]), 'euclidean'),
    (np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 2.0], [-1.0, 0.0], [3.0, 0.0]]), 'cosine'),
    (np.array([[1.0, 2.0, 3.0]]), np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]]), 'manhattan'),
    (_r.normal(size=(4, 3)), np.concatenate([_Y, _Y]), 'euclidean'),
    (_r.normal(size=(6, 8)), _r.normal(size=(7, 8)), 'cosine'),
    (_r.normal(size=(3, 5)), _r.normal(size=(2, 5)), 'manhattan'),
    (np.array([[2.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]]), 'manhattan'),
    (10 * _r.normal(size=(5, 4)), 10 * _r.normal(size=(9, 4)), 'euclidean'),
]

EXPLANATION = r"""
## Nested `vmap` is an outer product of loops
`vmap(d, in_axes=(None, 0))` holds `x` fixed and maps over the rows of `Y`. Mapping that function with `in_axes=(0, None)` holds `Y` fixed and maps over `X`. The result is the $(N, M)$ grid of every pair, built from a function that only knows about two vectors. JAX fuses it into broadcasted array ops.

## Ties and the duplicate test
Test 4 stacks `Y` on top of itself, so every nearest neighbour appears twice with identical distances. `argmin` returns the first occurrence, which is the lowest index, as required.

## Complexity and the matmul trick
Nested `vmap` of the Euclidean formula materialises the $(N, M, D)$ differences: $O(NMD)$ memory. For large retrieval, use $\lVert x - y\rVert^2 = \lVert x\rVert^2 + \lVert y\rVert^2 - 2 x \cdot y$ so the heavy part is one matmul with $O(NM)$ memory, and clamp at 0 before the square root because cancellation can make it slightly negative.

## Common bugs
- `x @ y / (norm(x) * norm(y))` with a zero vector gives `nan`. A single `jnp.where` on the result does not help under `grad` either: the `nan` leaks into the gradient. Make the divisor safe first.
- Swapping the `in_axes` order and getting `(M, N)`.

## Interview follow-ups
- Why is the gradient of `sqrt(sum((x - y)**2))` `nan` at `x == y`, and how would you fix it if you trained through this distance?
- How would you compute the top-5 neighbours for a million queries without building the full matrix? (Chunk the queries with `lax.map` or a scan and keep a running top-k.)
"""
