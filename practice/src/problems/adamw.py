import numpy as np

META = dict(title='AdamW over a pytree with a weight-decay mask', tags=['JAX', 'Optimization'], diff='Medium', runtime='jax',
            lessons=['air-fanout-v1:05.06'], related=['adamw-decoupled-weight-decay', 'adam-step', 'sgd-with-momentum-over-pytree'],
            tol=[1e-5, 1e-6])

STATEMENT = r"""
Write one AdamW update for an arbitrary parameter pytree, in the functional style JAX uses: the optimizer state goes in and comes back out.

Write `adamw_update(params, grads, state, lr, b1, b2, eps, wd)`:

- `params`, `grads`: pytrees (nested dicts) of float arrays.
- `state`: dict `{'m': pytree, 'v': pytree, 'count': int}`. `m` and `v` match `params`; `count` is the number of steps taken so far (0 before the first step).

With $t$ = `count + 1`, for each leaf:

$$m \leftarrow \beta_1 m + (1-\beta_1) g, \qquad v \leftarrow \beta_2 v + (1-\beta_2) g^2$$

$$\hat m = \frac{m}{1-\beta_1^t}, \qquad \hat v = \frac{v}{1-\beta_2^t}, \qquad p \leftarrow p - \eta\left(\frac{\hat m}{\sqrt{\hat v} + \epsilon} + \lambda_p\, p\right)$$

Weight decay is **decoupled** (it is not added to $g$, so it skips the moment estimates) and **masked**: $\lambda_p = $ `wd` for leaves with `ndim >= 2` (weight matrices, embeddings) and $\lambda_p = 0$ for leaves with `ndim < 2` (biases, norm scales).

Return `(new_params, new_state)` with `new_state = {'m': ..., 'v': ..., 'count': count + 1}`. float32 leaves; `count` may be a Python int or a 0-d int array.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def adamw_update(params, grads, state, lr, b1, b2, eps, wd):
    """state = {'m', 'v', 'count'}. Return (new_params, new_state)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def adamw_update(params, grads, state, lr, b1, b2, eps, wd):
    tm = jax.tree_util.tree_map
    f32 = lambda t: tm(lambda a: jnp.asarray(a, jnp.float32), t)
    params, grads, m, v = f32(params), f32(grads), f32(state['m']), f32(state['v'])
    t = jnp.asarray(state['count'], jnp.int32) + 1

    m = tm(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
    v = tm(lambda v, g: b2 * v + (1 - b2) * g * g, v, grads)
    c1, c2 = 1 - b1 ** t, 1 - b2 ** t                     # bias corrections, shared by every leaf

    def step(p, m, v):
        decay = wd if p.ndim >= 2 else 0.0                 # mask by rank: shapes are static, so a Python if is fine
        return p - lr * ((m / c1) / (jnp.sqrt(v / c2) + eps) + decay * p)

    new_params = tm(step, params, m, v)
    return new_params, {'m': m, 'v': v, 'count': t}
'''

HINTS = [
    'Update `m` and `v` with `tree_map` over `(m, grads)` and `(v, grads)` first. The bias corrections depend only on `t = count + 1`.',
    'Shapes are known at trace time, so `wd if p.ndim >= 2 else 0.0` inside the per-leaf function is fine, even under `jit`.',
    'Decoupled means the decay term `wd * p` sits next to the Adam direction in the update, scaled by `lr`, and is never added to `g`.',
]

_r = np.random.default_rng(18)
_P = lambda: {'w': _r.normal(size=(3, 2)), 'b': _r.normal(size=2), 'ln': {'scale': np.ones(2)}}
_Z = lambda: {'w': np.zeros((3, 2)), 'b': np.zeros(2), 'ln': {'scale': np.zeros(2)}}
_S = lambda: {'w': np.abs(_r.normal(size=(3, 2))), 'b': np.abs(_r.normal(size=2)), 'ln': {'scale': np.abs(_r.normal(size=2))}}
TESTS = [
    ({'w': np.array([[1.0]]), 'b': np.array([1.0])}, {'w': np.array([[0.5]]), 'b': np.array([0.5])},
     {'m': {'w': np.zeros((1, 1)), 'b': np.zeros(1)}, 'v': {'w': np.zeros((1, 1)), 'b': np.zeros(1)}, 'count': 0},
     0.1, 0.9, 0.999, 1e-8, 0.1),
    ({'w': np.array([[2.0, -2.0]])}, {'w': np.array([[0.0, 0.0]])},
     {'m': {'w': np.zeros((1, 2))}, 'v': {'w': np.zeros((1, 2))}, 'count': 0}, 0.5, 0.9, 0.999, 1e-8, 0.1),
    (_P(), _P(), {'m': _Z(), 'v': _Z(), 'count': 0}, 1e-3, 0.9, 0.999, 1e-8, 0.01),
    (_P(), _P(), {'m': _P(), 'v': _S(), 'count': 7}, 1e-2, 0.9, 0.95, 1e-6, 0.1),
    (_P(), _P(), {'m': _P(), 'v': _S(), 'count': 1000}, 3e-4, 0.9, 0.999, 1e-8, 0.0),
    ({'emb': _r.normal(size=(4, 3)), 'bias': np.array(0.5)}, {'emb': _r.normal(size=(4, 3)), 'bias': np.array(-1.0)},
     {'m': {'emb': np.zeros((4, 3)), 'bias': np.array(0.0)}, 'v': {'emb': np.zeros((4, 3)), 'bias': np.array(0.0)}, 'count': 0},
     0.01, 0.8, 0.9, 1e-8, 0.5),
    (_P(), _Z(), {'m': _P(), 'v': _S(), 'count': 3}, 0.1, 0.9, 0.999, 1e-8, 0.2),
]

EXPLANATION = r"""
## Decoupled decay, and why it matters
With Adam plus L2 (adding $\lambda p$ to $g$), the decay term is divided by $\sqrt{\hat v}$: parameters with large gradient history are barely decayed. AdamW applies $\eta \lambda p$ directly, so every decayed parameter shrinks at the same relative rate regardless of its gradient scale. Test 2 has zero gradients: Adam's direction is 0, and only the decay moves the weights.

## The first step
At $t = 1$, $\hat m = g$ and $\hat v = g^2$, so the Adam direction is $g / (|g| + \epsilon) \approx \mathrm{sign}(g)$. Each coordinate moves by about $\eta$ regardless of gradient scale. Bias correction is what makes the first steps this size instead of $(1 - \beta_1)$ times smaller.

## The decay mask
Decaying biases and LayerNorm scales pulls them toward 0, which hurts and has no regularising benefit. A rank-based rule (`ndim >= 2`) is the common default; Optax exposes it as the `mask` argument of `optax.adamw`.

## Complexity
Two extra pytrees of state, the same size as the parameters: AdamW triples the optimizer memory compared with plain SGD, which is why 8-bit optimizer states and sharded optimizers exist.

## Common bugs
- Bias correction with `count` instead of `count + 1`: division by zero on the first step.
- Applying decay with the bias-corrected learning rate or before the moments are updated: small numeric drift that is hard to spot.
- Putting `eps` inside the square root: a different (also used, but not this) variant.

## Interview follow-ups
- How does the effective decay change with the learning-rate schedule, and why do some codebases scale `wd` by `lr / peak_lr`?
- How would you implement this as an Optax `GradientTransformation` (`init` and `update`)?
"""
