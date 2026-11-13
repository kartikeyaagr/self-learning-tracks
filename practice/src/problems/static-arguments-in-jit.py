import numpy as np

META = dict(title='Static arguments in jit: windowed pooling', tags=['JAX'], diff='Easy', runtime='jax',
            lessons=['ieng:15.03'], related=['traced-learning-rate-schedule', 'pad-and-mask-to-bucket-length'],
            tol=[1e-5, 1e-6])

STATEMENT = r"""
`jax.jit` traces every argument as an abstract array by default. That is a problem when an argument decides a **shape** (a window size) or the **structure** of the computation (a string choosing a branch): a tracer has no concrete value to use. Mark such arguments static with `static_argnames`, and JAX compiles one version per distinct value.

Write `pool_sequence(x, pool, mode)`, compiled with `jax.jit` (fix the decorator in the starter):

- `x`: `(T, D)` float array.
- `pool`: int $\ge 1$, the window length.
- `mode`: `'mean'` or `'max'`.

Window $j$ covers rows $[j \cdot \text{pool},\ \min((j+1)\cdot\text{pool},\ T))$, so there are $\lceil T / \text{pool} \rceil$ windows. Return a float32 array of shape `(ceil(T / pool), D)` holding each window's mean or max over its rows. When `T` is not a multiple of `pool`, the last window is shorter and uses **only its real rows** (no padding values leak into the mean or max).

No Python loop over windows.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

@jax.jit                     # pool sets a shape and mode picks a branch: this decorator needs to know that
def pool_sequence(x, pool, mode):
    """x (T, D). Return (ceil(T / pool), D) window means or maxes."""
    raise NotImplementedError
'''

SOLUTION = '''from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=('pool', 'mode'))
def pool_sequence(x, pool, mode):
    x = x.astype(jnp.float32)
    T, D = x.shape                                    # shapes are static even for traced arrays
    n = -(-T // pool)                                 # ceil(T / pool): a Python int, fine for reshape
    pad = n * pool - T
    valid = (jnp.arange(n * pool) < T).reshape(n, pool, 1)
    xw = jnp.pad(x, ((0, pad), (0, 0))).reshape(n, pool, D)
    if mode == 'mean':                                # a Python branch: legal because mode is static
        return jnp.where(valid, xw, 0.0).sum(axis=1) / valid.sum(axis=1)
    if mode == 'max':
        return jnp.where(valid, xw, -jnp.inf).max(axis=1)
    raise ValueError(f'unknown mode {mode!r}')
'''

HINTS = [
    'Use `functools.partial(jax.jit, static_argnames=(\'pool\', \'mode\'))`. Static arguments are ordinary Python values inside the function, so `if mode == ...` and `reshape(n, pool, D)` both work.',
    'Pad `x` to a multiple of `pool` rows, then reshape to `(n_windows, pool, D)` and reduce over axis 1.',
    'Build a `(n_windows, pool, 1)` validity mask from `arange(n * pool) < T`: fill padding with `0` for the mean (and divide by the valid count) or `-inf` for the max.',
]

_r = np.random.default_rng(6)
TESTS = [
    (np.arange(8.0).reshape(4, 2), 2, 'mean'),
    (np.array([[1.0], [5.0], [2.0], [-1.0], [3.0]]), 2, 'max'),
    (np.array([[1.0], [5.0], [2.0], [-1.0], [3.0]]), 2, 'mean'),
    (-np.abs(_r.normal(size=(7, 3))), 3, 'max'),
    (_r.normal(size=(7, 3)), 3, 'mean'),
    (_r.normal(size=(5, 4)), 1, 'max'),
    (_r.normal(size=(5, 4)), 8, 'mean'),
    (_r.normal(size=(16, 6)), 4, 'max'),
]

EXPLANATION = r"""
## Static vs traced
Inside `jit`, a traced argument is a placeholder with a shape and dtype but no value. Shapes must be Python integers at trace time, so `reshape(n, pool, D)` with a traced `pool` fails, and `mode == 'mean'` can't even be passed in because a string isn't an array type. `static_argnames` turns these into compile-time constants: they are hashed into the cache key, and each new value triggers a new trace and compile.

## When to make something static
- It decides a shape, a loop trip count you need in Python, or which code path runs: static.
- It is numeric data that changes often (learning rate, temperature, step): keep it traced, or you will recompile on every call.
Static arguments must be hashable, so pass tuples, not lists or arrays.

## Cost
One fused pad, reshape and reduce: $O(TD)$. Compilation happens once per `(x.shape, pool, mode)` combination.

## Common bugs
- Padding the last window with zeros and then taking the plain mean or max: the short window is biased toward 0.
- Marking `x` static: arrays aren't hashable, and if they were you would recompile for every new input.
- Using `jnp.ceil(T / pool)` for the window count: that is a JAX array, not a Python int, so it can't be used as a shape.

## Interview follow-ups
- What happens to compile time if a caller passes `pool` from a list of 100 different values? How would you notice in production? (Watch the jit cache size, or log compiles with `jax.config.update('jax_log_compiles', True)`.)
- How does `jax.lax.reduce_window` express the same pooling without the reshape trick?
"""
