import numpy as np

META = dict(title='Functional array updates with .at[]', tags=['JAX'], diff='Easy', runtime='jax',
            lessons=['air-fanout-v1:03.05'], related=['prng-key-splitting', 'masked-mean-without-boolean-indexing'],
            tol=[1e-5, 1e-6])

STATEMENT = r"""
JAX arrays are immutable: `x[i] = v` raises an error. Updates go through the `.at[...]` property instead, which returns a **new** array and leaves the old one untouched. Under `jit`, XLA usually turns these copies back into in-place writes, so the functional style costs nothing.

Write `scatter_three_ways(grid, idx, vals)`:

- `grid`: `(H, W)` float array.
- `idx`: `(N, 2)` int array of `(row, col)` pairs, all in range. **The same cell can appear more than once.**
- `vals`: `(N,)` float array; `vals[n]` is aimed at cell `idx[n]`.

Return a tuple of three float32 `(H, W)` arrays:

1. `added`: `grid` with every `vals[n]` added to its cell. A cell listed twice receives both updates.
2. `peaked`: `grid` where each listed cell becomes the maximum of its old value and every value aimed at it. Unlisted cells are unchanged.
3. `framed`: `grid` with its first row and last column set to `0.0`, everything else unchanged (this one ignores `idx`).

Use `jax.numpy` and `.at[]`; no Python loops over `N`.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime. JAX computes in float32 here (64-bit mode is off), so NumPy float64 inputs become float32 inside `jnp`. The notebook includes the same tests.
"""

STARTER = '''import jax.numpy as jnp

def scatter_three_ways(grid, idx, vals):
    """grid (H, W), idx (N, 2) int, vals (N,). Return (added, peaked, framed), each (H, W) float32."""
    raise NotImplementedError
'''

SOLUTION = '''import jax.numpy as jnp

def scatter_three_ways(grid, idx, vals):
    g = jnp.asarray(grid, jnp.float32)
    v = jnp.asarray(vals, jnp.float32)
    r, c = idx[:, 0], idx[:, 1]                       # advanced indexing: one (row, col) pair per update
    added = g.at[r, c].add(v)                         # scatter-add: duplicates accumulate
    peaked = g.at[r, c].max(v)                        # scatter-max: duplicates combine with max
    framed = g.at[0, :].set(0.0).at[:, -1].set(0.0)   # each .at returns a new array, so chain them
    return added, peaked, framed
'''

HINTS = [
    '`x.at[rows, cols]` with two index arrays selects one cell per pair, exactly like NumPy advanced indexing.',
    'Besides `.set`, the `.at` property has `.add`, `.mul`, `.min` and `.max`. Each one combines repeated indices correctly instead of keeping only the last write.',
    'For the frame, chain two updates: `g.at[0, :].set(0.0).at[:, -1].set(0.0)`. The original `g` is never modified.',
]

_r = np.random.default_rng(3)
TESTS = [
    (np.zeros((2, 3)), np.array([[0, 1], [1, 2], [0, 1]]), np.array([1.0, 2.0, 5.0])),
    (np.arange(6.0).reshape(2, 3), np.array([[1, 0], [1, 0]]), np.array([-4.0, 10.0])),
    (np.ones((1, 1)), np.array([[0, 0], [0, 0], [0, 0]]), np.array([0.5, -2.0, 0.25])),
    (-np.ones((3, 2)), np.array([[2, 1], [0, 0]]), np.array([-5.0, -0.5])),
    (_r.normal(size=(4, 5)), _r.integers(0, 4, size=(12, 1)).repeat(2, axis=1) % np.array([4, 5]), _r.normal(size=12)),
    (_r.normal(size=(6, 3)), np.stack([_r.integers(0, 6, 20), _r.integers(0, 3, 20)], axis=1), _r.normal(size=20)),
]

EXPLANATION = r"""
## Why `.at[]` exists
JAX traces Python functions into pure dataflow graphs, so a value can't change after it is created. `x.at[i].add(v)` is a description of a new array: "`x`, but with `v` added at `i`". It lowers to an XLA scatter. When the old array is dead afterwards (very common inside `jit`), XLA reuses its buffer and the update happens in place.

## Duplicate indices
The combining modes are the point of this exercise. In NumPy, `a[r, c] += v` with a repeated cell silently keeps one update, because it reads all cells, adds and writes back. `.at[r, c].add(v)` accumulates every update (like `np.add.at`). `.max` and `.min` combine duplicates the same way. `.set` with duplicates is the one ambiguous case: JAX makes no promise about which write wins, so avoid it.

## Cost
Each scatter is $O(N)$ work plus, outside `jit`, an $O(HW)$ copy. Inside `jit` the copy usually disappears. Chaining `.at` calls is cheap for the same reason.

## Common bugs
- Trying `x[0] = 0` or `x += ...` on a slice: raises `TypeError` on JAX arrays.
- Converting to NumPy and using `a[r, c] += v`: loses repeated updates.
- Out-of-range indices: JAX doesn't raise. Scatter updates out of bounds are dropped and gathers are clamped, so bad indices fail silently.

## Interview follow-ups
- How does XLA decide whether a functional update can be done in place? (Buffer donation and liveness: the input must not be used again.)
- How would you build a histogram of token ids in JAX under `jit`? (`jnp.zeros(V).at[ids].add(1)`, or `jnp.bincount` with a static `length`.)
"""
