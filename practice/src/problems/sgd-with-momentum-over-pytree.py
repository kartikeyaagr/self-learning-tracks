import numpy as np

META = dict(title='SGD with momentum over a nested pytree', tags=['JAX', 'Optimization'], diff='Easy', runtime='jax',
            lessons=['air-fanout-v1:05.05'], related=['classical-momentum', 'adamw', 'global-norm-gradient-clipping'],
            tol=[1e-5, 1e-6])

STATEMENT = r"""
JAX parameters are **pytrees**: nested dicts, lists and tuples with arrays at the leaves. An optimizer is a function that maps `(params, state, grads)` to new values with the same structure, usually written with `jax.tree_util.tree_map` (or `jax.tree.map`).

Write `momentum_step(params, velocity, grads, lr, beta, weight_decay)` using the PyTorch convention for every leaf $p$ with gradient $g$ and velocity $v$:

$$g' = g + \lambda\, p, \qquad v \leftarrow \beta\, v + g', \qquad p \leftarrow p - \eta\, v$$

- `params`, `grads`: pytrees with the same structure (nested dicts of arrays of any shape, including 0-d).
- `velocity`: a pytree of the same structure, or `None` on the very first step, meaning all zeros.
- `lr` $=\eta$, `beta` $=\beta$, `weight_decay` $=\lambda$: floats.

Return `(new_params, new_velocity)`, both with the same structure as `params`. float32 leaves. Do not hard-code the key names: the tests use different trees.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def momentum_step(params, velocity, grads, lr, beta, weight_decay):
    """Return (new_params, new_velocity), same pytree structure as params."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def momentum_step(params, velocity, grads, lr, beta, weight_decay):
    f32 = lambda t: jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), t)
    params, grads = f32(params), f32(grads)
    velocity = jax.tree_util.tree_map(jnp.zeros_like, params) if velocity is None else f32(velocity)

    # tree_map over several trees walks them in lockstep; they must share one structure.
    new_v = jax.tree_util.tree_map(lambda v, g, p: beta * v + g + weight_decay * p, velocity, grads, params)
    new_p = jax.tree_util.tree_map(lambda p, v: p - lr * v, params, new_v)
    return new_p, new_v
'''

HINTS = [
    '`jax.tree_util.tree_map(fn, tree_a, tree_b, ...)` calls `fn(a_leaf, b_leaf, ...)` on matching leaves and rebuilds the same structure.',
    'Handle `velocity is None` with a Python `if` before any math: `tree_map(jnp.zeros_like, params)` builds the zero tree.',
    'Update the velocity first (it includes the weight-decay term), then step the parameters with the **new** velocity.',
]

_r = np.random.default_rng(16)
_tree = lambda: {'dense': {'w': _r.normal(size=(3, 2)), 'b': _r.normal(size=2)}, 'scale': np.array(_r.normal())}
TESTS = [
    ({'w': np.array([1.0, 2.0])}, None, {'w': np.array([0.5, -1.0])}, 0.1, 0.9, 0.0),
    ({'w': np.array([1.0, 2.0])}, {'w': np.array([1.0, 1.0])}, {'w': np.array([0.0, 0.0])}, 0.1, 0.9, 0.0),
    (_tree(), None, _tree(), 0.01, 0.9, 1e-2),
    (_tree(), _tree(), _tree(), 0.05, 0.5, 0.0),
    ({'a': {'b': {'c': np.array(3.0)}}, 'd': np.ones(4)}, {'a': {'b': {'c': np.array(1.0)}}, 'd': np.zeros(4)},
     {'a': {'b': {'c': np.array(-2.0)}}, 'd': np.full(4, 0.5)}, 0.1, 0.0, 0.1),
    (_tree(), _tree(), _tree(), 0.1, 0.99, 0.5),
    ({'emb': _r.normal(size=(5, 3)), 'head': {'w': _r.normal(size=(3, 4))}}, None,
     {'emb': _r.normal(size=(5, 3)), 'head': {'w': _r.normal(size=(3, 4))}}, 1.0, 0.9, 0.0),
]

EXPLANATION = r"""
## Pytrees make optimizers structure-agnostic
`tree_map` flattens each tree to a list of leaves, applies the function leaf by leaf, and unflattens with the first tree's structure. The optimizer never mentions `'dense'` or `'w'`, so it works for any model. This is how Optax is built: every transformation is a pair of pure functions over pytrees.

## The update is pure
Nothing is modified in place: the function returns new trees, and the caller rebinds them. That is what lets `jax.jit` compile the whole training step, and why optimizer state must be passed in and returned rather than stored on an object.

## Conventions differ
This is PyTorch's form ($v \leftarrow \beta v + g$, step $\eta v$). Another common form folds the learning rate in, $v \leftarrow \beta v - \eta g$, $p \leftarrow p + v$. They agree for a constant learning rate but differ when $\eta$ changes over time. Weight decay here is coupled (added to the gradient), unlike AdamW.

## Common bugs
- Stepping with the old velocity: the update lags one step.
- Building the zero velocity with `jnp.zeros(p.shape)` and a default dtype that differs from the parameters.
- `tree_map` over trees whose structures differ (for example `velocity` built from a different model): JAX raises a structure mismatch. That error is a feature.

## Interview follow-ups
- How would you exclude biases and norm scales from weight decay in a structure-agnostic way? (A mask pytree, or a rule on leaf rank.)
- Why does `tree_map` treat `None` as an empty subtree, and how can that surprise you?
"""
