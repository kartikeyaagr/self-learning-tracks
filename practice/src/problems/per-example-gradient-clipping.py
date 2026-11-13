import numpy as np

META = dict(title='Per-example gradient clipping with vmap', tags=['JAX', 'Training'], diff='Medium', runtime='jax',
            related=['global-norm-gradient-clipping', 'gradient-clipping', 'value-and-grad-of-loss'], tol=[1e-4, 1e-5])

STATEMENT = r"""
Differentially private SGD clips **each example's** gradient before averaging, so no single example can move the model by more than a bound $C$. Frameworks that only give the batch gradient make this awkward. In JAX it is `vmap` over `grad`.

Model and loss for one example $(x, y)$, with `params = {'w': (D,), 'b': ()}`:

$$\ell(w, b; x, y) = \tfrac12\,(x^\top w + b - y)^2$$

Write `clipped_mean_grad(params, x, y, max_norm)`:

1. Compute the per-example gradients $g_i = \nabla_{(w,b)}\, \ell(\cdot\,; x_i, y_i)$ for every row of `x` `(B, D)` and `y` `(B,)`.
2. Each example's norm is the global L2 norm over both leaves: $n_i = \sqrt{\lVert g_{i,w}\rVert^2 + g_{i,b}^2}$.
3. Scale each gradient by $\min(1, C / n_i)$, using factor `1` when $n_i = 0$.
4. Average the clipped gradients over the batch.

Return `(grads, norms)`: `grads` is a dict `{'w': (D,), 'b': ()}` and `norms` is `(B,)` (the norms before clipping). float32. No Python loop over examples.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def clipped_mean_grad(params, x, y, max_norm):
    """params {'w': (D,), 'b': ()}. Return (mean clipped grads dict, per-example norms (B,))."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def clipped_mean_grad(params, x, y, max_norm):
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), params)
    x, y = jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32)

    def loss_one(p, xi, yi):
        return 0.5 * (xi @ p['w'] + p['b'] - yi) ** 2

    # in_axes: params shared (None), one row of x and one label per call (0). Result leaves gain a leading B axis.
    per_ex = jax.vmap(jax.grad(loss_one), in_axes=(None, 0, 0))(params, x, y)

    sq = sum(jnp.sum(g.reshape(g.shape[0], -1) ** 2, axis=1) for g in jax.tree_util.tree_leaves(per_ex))
    norms = jnp.sqrt(sq)                                                     # (B,)
    factor = jnp.minimum(1.0, max_norm / jnp.maximum(norms, 1e-12))         # n = 0 gives a huge ratio, so factor 1

    def clip_mean(g):
        f = factor.reshape((-1,) + (1,) * (g.ndim - 1))                      # broadcast over the leaf's own dims
        return jnp.mean(g * f, axis=0)

    return jax.tree_util.tree_map(clip_mean, per_ex), norms
'''

HINTS = [
    '`jax.vmap(jax.grad(loss_one), in_axes=(None, 0, 0))` gives a dict whose leaves have an extra leading batch axis: `w` becomes `(B, D)`, `b` becomes `(B,)`.',
    'For the norm, flatten each leaf to `(B, -1)`, sum the squares per row, add the leaves together and take the square root.',
    'Reshape the `(B,)` factor so it broadcasts against each leaf, multiply, then take the mean over axis 0.',
]

_r = np.random.default_rng(21)
TESTS = [
    ({'w': np.zeros(2), 'b': np.array(0.0)}, np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 2.0]), 10.0),
    ({'w': np.zeros(2), 'b': np.array(0.0)}, np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 2.0]), 1.0),
    ({'w': np.array([1.0, -1.0]), 'b': np.array(0.5)}, np.array([[1.0, 1.0], [2.0, 1.0]]), np.array([0.5, 1.5]), 0.1),
    ({'w': _r.normal(size=4), 'b': np.array(0.0)}, _r.normal(size=(8, 4)), _r.normal(size=8), 1.0),
    ({'w': _r.normal(size=4), 'b': np.array(1.0)}, 5 * _r.normal(size=(8, 4)), _r.normal(size=8), 0.5),
    ({'w': _r.normal(size=3), 'b': np.array(-0.2)}, _r.normal(size=(16, 3)), _r.normal(size=16), 1e6),
    ({'w': _r.normal(size=5), 'b': np.array(0.3)}, _r.normal(size=(32, 5)), 10 * _r.normal(size=32), 2.0),
]

EXPLANATION = r"""
## Why `vmap(grad(...))` works
`grad(loss_one)` is the gradient for a single example. `vmap` vectorises it over the batch axis without writing any batching code: JAX pushes the batch dimension through every primitive. The result has the same pytree structure as `params`, with an extra leading axis on each leaf. In PyTorch this needs `torch.func` or tricks with hooks; in JAX it is the natural composition.

## Clipping is per example, the norm is over all leaves
DP-SGD bounds each example's total influence, so the norm is computed across every parameter leaf together, not leaf by leaf. Clipping happens before averaging; clipping the average would only bound the batch, not any one example. With $C$ huge (test 6), the result equals the ordinary mean gradient.

## Cost
Per-example gradients materialise $B$ copies of the gradient: $O(B \cdot P)$ memory for $P$ parameters, versus $O(P)$ for the batch gradient. For large models, DP libraries compute per-example norms without materialising the gradients (the "ghost clipping" trick).

## Common bugs
- Computing the norm per leaf and clipping each leaf separately: a different and weaker guarantee.
- `in_axes=0` for `params`: vmap would try to slice the weights along their first axis.
- Dividing by the norm with no guard: `0 / 0 = nan` for an example with zero gradient.

## Interview follow-ups
- Where does the Gaussian noise go in DP-SGD, and why is its scale proportional to $C$?
- How does per-example clipping interact with batch normalisation? (It breaks the per-example independence that the privacy proof assumes; use group or layer norm.)
"""
