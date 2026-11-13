import numpy as np

META = dict(title='Masked statistics under jit, no boolean indexing', tags=['JAX', 'Data Processing'], diff='Easy',
            runtime='jax', lessons=['air-fanout-v1:02.03'],
            related=['masked-mean-pooling', 'pad-and-mask-to-bucket-length'], tol=[1e-5, 1e-6])

STATEMENT = r"""
In NumPy you might write `x[mask].mean()`. Under `jax.jit` that fails: the number of selected elements depends on the data, and compiled JAX code needs every shape known at trace time. The fix is to keep the full shape and let the mask decide what counts.

Write `masked_stats(x, mask)`, decorated with `@jax.jit` (keep the decorator from the starter):

- `x`: `(B, T)` float array.
- `mask`: `(B, T)` bool array. `True` marks a real entry.

Return a tuple of three `(B,)` arrays:

1. `mean`: float32 mean of the real entries in each row.
2. `maxv`: float32 maximum of the real entries in each row.
3. `count`: int32 number of real entries in each row.

A row with no real entries has `mean = 0.0`, `maxv = 0.0` and `count = 0`. No boolean indexing, no Python loops.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

@jax.jit
def masked_stats(x, mask):
    """x (B, T) float, mask (B, T) bool. Return (mean, maxv, count), each (B,)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

@jax.jit
def masked_stats(x, mask):
    m = mask.astype(bool)
    count = m.sum(axis=1, dtype=jnp.int32)
    total = jnp.where(m, x, 0.0).sum(axis=1)               # padded entries contribute 0 to the sum
    mean = total / jnp.maximum(count, 1)                    # empty rows: 0 / 1 = 0
    maxv = jnp.where(m, x, -jnp.inf).max(axis=1)            # padded entries can never win the max
    maxv = jnp.where(count > 0, maxv, 0.0)                  # empty rows would otherwise be -inf
    return mean.astype(jnp.float32), maxv.astype(jnp.float32), count
'''

HINTS = [
    '`x[mask]` has a data-dependent shape, so it cannot be traced. `jnp.where(mask, x, fill)` keeps the shape `(B, T)`.',
    'Pick the fill value per reduction: `0` is neutral for a sum, `-inf` is neutral for a max.',
    'Divide by `jnp.maximum(count, 1)` so empty rows give `0 / 1`, and patch the empty-row max with another `jnp.where`.',
]

_r = np.random.default_rng(11)
TESTS = [
    (np.array([[1.0, 2.0, 9.0], [4.0, -1.0, 0.0]]), np.array([[True, True, False], [True, True, True]])),
    (np.array([[5.0, 5.0], [-3.0, -7.0]]), np.array([[False, False], [False, True]])),
    (-np.abs(_r.normal(size=(3, 6))), _r.random((3, 6)) < 0.5),
    (_r.normal(size=(4, 1)), np.array([[True], [False], [True], [False]])),
    (_r.normal(size=(5, 8)), np.ones((5, 8), dtype=bool)),
    (_r.normal(size=(2, 7)), np.zeros((2, 7), dtype=bool)),
    (_r.normal(size=(6, 10)) * 100, _r.random((6, 10)) < 0.3),
]

EXPLANATION = r"""
## Static shapes are the rule
`jit` traces your function with abstract arrays that know only shape and dtype. Every intermediate must have a shape computable from those. `x[mask]` breaks this because its length depends on values. Masked reductions with `jnp.where` keep the full `(B, T)` shape and simply make padded entries neutral.

## Neutral elements
- Sum: fill with `0`.
- Max: fill with `-inf`. Min: `+inf`. Product: `1`.
- Mean: masked sum divided by the masked count. `jnp.mean(x, axis=1, where=m)` does the same, but returns `nan` for empty rows, so you still need a guard.

## Cost
Two fused elementwise passes and three reductions over `B·T` values. Computing on padding wastes a little work, which is the price of static shapes; bucketing sequence lengths (next problem) keeps the waste small.

## Common bugs
- Boolean indexing inside `jit`: `NonConcreteBooleanIndexError`.
- `jnp.where(m, x, 0).max()`: a row of negative real values reports `0` as its max.
- Dividing by the raw count: `0/0 = nan` for empty rows, and `nan` spreads into everything downstream.

## Interview follow-ups
- When is `jnp.nonzero` allowed under `jit`? (Only with a static `size=` argument and a `fill_value`, which pads the result to a fixed length.)
- Why does a `nan` in an unselected branch of `jnp.where` still break gradients? (Both branches are differentiated; the zero cotangent times `nan` is `nan`. Make the unselected input safe first.)
"""
