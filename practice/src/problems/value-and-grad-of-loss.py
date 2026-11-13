import numpy as np

META = dict(title='value_and_grad with an auxiliary metric', tags=['JAX', 'Loss Functions'], diff='Easy', runtime='jax',
            lessons=['air-fanout-v1:01.04'], related=['hessian-vector-product', 'per-example-gradient-clipping'],
            tol=[1e-4, 1e-5])

STATEMENT = r"""
`jax.grad` turns a scalar function into its gradient function. In training you want the loss value too, and often a metric computed from the same forward pass. `jax.value_and_grad(f, has_aux=True)` returns all of them from one pass.

Write `loss_acc_grads(params, x, y, l2)` for a softmax classifier:

- `params`: dict `{'w': (D, K), 'b': (K,)}`.
- `x`: `(N, D)` float; `y`: `(N,)` int labels in `[0, K)`; `l2`: float.
- Logits $z = x W + b$. Loss:

$$\mathcal{L} = \frac{1}{N}\sum_{n} -\log \mathrm{softmax}(z_n)_{y_n} \;+\; \frac{l_2}{2}\lVert W \rVert_F^2$$

  The penalty applies to `w` only, not `b`.
- Accuracy: the fraction of rows whose `argmax` logit equals the label (the lowest index wins a tie).

Return `(loss, acc, grads)`: two float32 scalars and a dict `{'w': (D, K), 'b': (K,)}` holding $\partial \mathcal{L}/\partial W$ and $\partial \mathcal{L}/\partial b$. The accuracy carries no gradient. Use one call to `jax.value_and_grad`.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def loss_acc_grads(params, x, y, l2):
    """params {'w': (D, K), 'b': (K,)}. Return (loss, acc, grads dict)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def loss_acc_grads(params, x, y, l2):
    x = jnp.asarray(x, jnp.float32)
    y = jnp.asarray(y)

    def loss_fn(p):
        logits = x @ p['w'] + p['b']
        logp = jax.nn.log_softmax(logits, axis=-1)                       # stable: no exp overflow
        nll = -jnp.take_along_axis(logp, y[:, None], axis=1).mean()
        loss = nll + 0.5 * l2 * jnp.sum(p['w'] ** 2)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)                 # aux: returned, not differentiated
        return loss, acc

    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), params)
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return loss, acc, grads
'''

HINTS = [
    'Write an inner `loss_fn(params)` that returns `(loss, acc)`. With `has_aux=True`, JAX differentiates only the first element.',
    'Use `jax.nn.log_softmax` and pick each row\'s label with `jnp.take_along_axis(logp, y[:, None], axis=1)`.',
    '`jax.value_and_grad(loss_fn, has_aux=True)(params)` returns `((loss, acc), grads)`, and `grads` has the same dict structure as `params`.',
]

_r = np.random.default_rng(4)
TESTS = [
    ({'w': np.zeros((2, 3)), 'b': np.zeros(3)}, np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 2]), 0.0),
    ({'w': np.array([[1.0, -1.0], [0.5, 0.5]]), 'b': np.array([0.0, 0.1])}, np.array([[1.0, 2.0]]), np.array([1]), 0.1),
    ({'w': _r.normal(size=(4, 3)), 'b': _r.normal(size=3)}, _r.normal(size=(8, 4)), _r.integers(0, 3, 8), 0.0),
    ({'w': _r.normal(size=(4, 3)), 'b': _r.normal(size=3)}, _r.normal(size=(8, 4)), _r.integers(0, 3, 8), 0.5),
    ({'w': 10 * _r.normal(size=(3, 5)), 'b': np.zeros(5)}, _r.normal(size=(6, 3)), _r.integers(0, 5, 6), 0.01),
    ({'w': _r.normal(size=(6, 2)), 'b': np.array([3.0, -3.0])}, _r.normal(size=(20, 6)), np.zeros(20, dtype=np.int64), 0.0),
    ({'w': _r.normal(size=(5, 4)), 'b': _r.normal(size=4)}, _r.normal(size=(32, 5)), _r.integers(0, 4, 32), 1e-3),
]

EXPLANATION = r"""
## How it works
`jax.value_and_grad(f)` runs `f` once forward while recording a linearisation, then runs the reverse pass to get $\nabla f$ with respect to the first argument. The argument here is a dict, so the gradient is a dict of the same shape: JAX treats nested containers as **pytrees** and differentiates every array leaf.

## `has_aux`
Without it, `f` must return a scalar. With `has_aux=True`, `f` returns `(scalar, aux)` and you get `((scalar, aux), grads)`. The aux value is computed in the same forward pass, so the accuracy costs nothing extra and no second forward is needed.

## The gradient you should expect
For softmax cross-entropy, $\partial \mathcal{L}/\partial z_n = (\mathrm{softmax}(z_n) - \mathrm{onehot}(y_n))/N$. So $\nabla_b$ is the column mean of that, and $\nabla_W = x^\top (\cdot) + l_2 W$. You can check the autodiff result against this formula.

## Cost
Reverse mode costs a small constant multiple of the forward pass, whatever the number of parameters. That is why `grad` scales to large models.

## Common bugs
- Returning `(loss, acc)` without `has_aux=True`: JAX raises because the output isn't a scalar.
- Computing `log(softmax(z))` directly: underflows to `-inf` for confident wrong predictions.
- Including `b` in the penalty, or forgetting the `1/2`.

## Interview follow-ups
- How do you get the gradient with respect to the second argument, or several arguments at once? (`argnums=1` or `argnums=(0, 1)`.)
- Why is `argmax` safe inside the loss function even though it isn't differentiable? (It only feeds the aux output; its integer result has no gradient path to the loss.)
"""
