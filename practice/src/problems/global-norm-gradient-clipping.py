import numpy as np

META = dict(title='Global-norm clipping with a non-finite guard', tags=['JAX', 'Training'], diff='Medium', runtime='jax',
            related=['gradient-clipping-global-norm-rescaling', 'per-example-gradient-clipping', 'sgd-with-momentum-over-pytree'],
            tol=[1e-5, 1e-6])

STATEMENT = r"""
Clip a gradient pytree by its **global** L2 norm, the norm of all leaves concatenated into one vector, and protect the step against overflow: if any gradient is `inf` or `nan`, the step should be skipped.

Write `clip_global(grads, max_norm)`:

- `grads`: a pytree (nested dicts) of float arrays of any shape.
- Global norm: $n = \sqrt{\sum_{\text{leaves}} \sum_i g_i^2}$.
- If $n$ is finite: scale every leaf by $\min(1,\ C / n)$, where $C$ = `max_norm` $> 0$. When $n = 0$ the factor is 1.
- If $n$ is not finite (`inf` or `nan`): return all-zero gradients.

Return `(clipped, norm, finite)`:

- `clipped`: same structure as `grads`, float32.
- `norm`: 0-d float32 array, the global norm **before** clipping (it may be `inf` or `nan`).
- `finite`: 0-d bool array.

The function must work under `jax.jit`, so no Python `if` on array values: use `jnp.where`.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def clip_global(grads, max_norm):
    """Return (clipped pytree, global norm, finite flag)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

@jax.jit
def clip_global(grads, max_norm):
    grads = jax.tree_util.tree_map(lambda g: jnp.asarray(g, jnp.float32), grads)
    leaves = jax.tree_util.tree_leaves(grads)
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in leaves))
    finite = jnp.isfinite(norm)                        # nan or inf anywhere makes the norm non-finite
    safe = jnp.where(norm > 0, norm, 1.0)              # avoid 0/0; factor is 1 there anyway
    factor = jnp.minimum(1.0, max_norm / safe)
    factor = jnp.where(finite, factor, 0.0)            # skip the step: zero gradients
    # where(), not multiplication: inf * 0 = nan, so scaling by 0 would not clear bad leaves.
    clipped = jax.tree_util.tree_map(lambda g: jnp.where(finite, g * factor, 0.0), grads)
    return clipped, norm, finite
'''

HINTS = [
    '`jax.tree_util.tree_leaves(grads)` gives a flat list of arrays. Sum `jnp.sum(g * g)` over it, then take the square root.',
    '`jnp.isfinite(norm)` is enough to detect a bad leaf: any `nan` or `inf` propagates into the sum of squares.',
    'Multiplying an `inf` leaf by 0 gives `nan`. Use `jnp.where(finite, g * factor, 0.0)` on each leaf instead.',
]

_r = np.random.default_rng(17)
_t = lambda s=1.0: {'enc': {'w': s * _r.normal(size=(4, 3)), 'b': s * _r.normal(size=3)}, 'head': s * _r.normal(size=5)}
_bad = _t()
_bad['enc']['b'][1] = np.inf
_nan = _t()
_nan['head'][0] = np.nan
TESTS = [
    ({'a': np.array([3.0]), 'b': np.array([4.0])}, 1.0),
    ({'a': np.array([0.3]), 'b': np.array([0.4])}, 1.0),
    (_t(), 1.0),
    (_t(10.0), 0.5),
    ({'a': np.zeros(3), 'b': {'c': np.zeros((2, 2))}}, 1.0),
    (_bad, 1.0),
    (_nan, 5.0),
    ({'s': np.array(-6.0), 'v': np.array([8.0, 0.0])}, 5.0),
]

EXPLANATION = r"""
## Global, not per-leaf
Clipping each leaf separately changes the gradient's direction: a layer with large gradients is shrunk more than the others. Scaling every leaf by one factor keeps the direction and only caps the step length. This is what `optax.clip_by_global_norm` and `torch.nn.utils.clip_grad_norm_` do.

## The non-finite guard
One overflow anywhere (common in fp16 training) makes the norm `inf` or `nan`. Applying that update would destroy the weights, so the step is skipped by zeroing the gradients; mixed-precision trainers also lower the loss scale at that point. The flag is returned so the caller can count skipped steps.

## Why `jnp.where` and not `if`
Under `jit` the norm is a tracer with no concrete value, so `if norm > max_norm:` raises a concretisation error. `jnp.where` computes both options and selects, which compiles to a single fused elementwise op.

## Common bugs
- `g * 0` to clear a bad leaf: `inf * 0 = nan`.
- `max_norm / norm` without guarding `norm = 0`: `inf` factor, then `0 * inf = nan`.
- Returning the clipped norm instead of the raw one: logging then hides how often clipping fires.

## Interview follow-ups
- Why does clipping interact with Adam less than with SGD? (Adam's update is roughly scale-invariant per coordinate; clipping mostly changes the moment estimates.)
- How would you compute the global norm when the gradient is sharded across devices? (Local sum of squares, then `psum`, then the square root.)
"""
