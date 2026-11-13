import numpy as np

META = dict(title='Scan over layers with jax.checkpoint', tags=['JAX', 'Training', 'Transformers'], diff='Medium',
            runtime='jax', lessons=['ieng:03.05'],
            related=['gradient-accumulation-with-scan', 'exponential-moving-average-with-scan', 'value-and-grad-of-loss'],
            tol=[1e-4, 1e-5])

STATEMENT = r"""
Deep models in JAX usually stack their layers' weights along a leading axis and run the layers with `jax.lax.scan`, which compiles one layer body instead of $L$. To cut activation memory, the body is wrapped in `jax.checkpoint` (also called `jax.remat`): the backward pass recomputes each layer's internals instead of storing them.

Each of the $L$ residual MLP blocks maps `(B, D)` to `(B, D)`:

$$h \leftarrow h + \mathrm{gelu}(h W_1 + b_1)\, W_2 + b_2$$

with the **tanh approximation** of GELU, `jax.nn.gelu(..., approximate=True)`:

$$\mathrm{gelu}(z) = \tfrac12 z \left(1 + \tanh\!\left(\sqrt{2/\pi}\,(z + 0.044715 z^3)\right)\right)$$

Write `deep_mlp_loss_and_grads(params, x, y, remat)`:

- `params`: `{'w1': (L, D, F), 'b1': (L, F), 'w2': (L, F, D), 'b2': (L, D)}`; `x`, `y`: `(B, D)`.
- Loss: $\mathrm{mean}\bigl((h_L - y)^2\bigr)$ over all elements, where $h_0 = x$.
- `remat`: Python string.
  - `'none'`: plain scan;
  - `'full'`: wrap the block in `jax.checkpoint` (save only the block input);
  - `'dots'`: wrap it in `jax.checkpoint` with `policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable` (keep matmul outputs, recompute the cheap elementwise ops).

Return `(loss, grads)`: a 0-d array and a dict shaped like `params`. float32. All three settings must give the same numbers, since rematerialisation changes memory, not math. The tests check values only; use `jax.ad_checkpoint.print_saved_residuals` in Colab to see what each policy keeps.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def deep_mlp_loss_and_grads(params, x, y, remat):
    """Stacked params with a leading layer axis. Return (loss, grads dict)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def _block(h, layer):
    z = jax.nn.gelu(h @ layer['w1'] + layer['b1'], approximate=True)
    return h + z @ layer['w2'] + layer['b2'], None          # scan body: (carry, per-step output)

def deep_mlp_loss_and_grads(params, x, y, remat):
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), params)
    x, y = jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32)
    body = {
        'none': _block,
        'full': jax.checkpoint(_block),
        'dots': jax.checkpoint(_block, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable),
    }[remat]

    def loss_fn(p):
        h, _ = jax.lax.scan(body, x, p)                     # scan slices every leaf along the layer axis
        return jnp.mean((h - y) ** 2)

    return jax.value_and_grad(loss_fn)(params)
'''

HINTS = [
    'The scan body is `block(h, layer) -> (new_h, None)`. Passing the whole `params` dict as `xs` gives the body one layer\'s weights per step.',
    '`jax.checkpoint(fn)` and `jax.checkpoint(fn, policy=...)` return a function with the same signature, so you can swap the body without touching the scan.',
    'Wrap the scan in a loss function of `params` and call `jax.value_and_grad` on it.',
]

_r = np.random.default_rng(22)

def _p(L, D, F, s=0.3):
    return {'w1': s * _r.normal(size=(L, D, F)), 'b1': 0.1 * _r.normal(size=(L, F)),
            'w2': s * _r.normal(size=(L, F, D)), 'b2': 0.1 * _r.normal(size=(L, D))}

_one = {'w1': np.ones((1, 1, 1)), 'b1': np.zeros((1, 1)), 'w2': np.ones((1, 1, 1)), 'b2': np.zeros((1, 1))}
_P4 = _p(4, 3, 5)
_X4, _Y4 = _r.normal(size=(6, 3)), _r.normal(size=(6, 3))
TESTS = [
    (_one, np.array([[1.0]]), np.array([[0.0]]), 'none'),
    (_one, np.array([[1.0]]), np.array([[0.0]]), 'full'),
    (_P4, _X4, _Y4, 'none'),
    (_P4, _X4, _Y4, 'full'),
    (_P4, _X4, _Y4, 'dots'),
    (_p(1, 2, 2), _r.normal(size=(3, 2)), _r.normal(size=(3, 2)), 'dots'),
    (_p(8, 4, 8, 0.2), _r.normal(size=(2, 4)), _r.normal(size=(2, 4)), 'full'),
]

EXPLANATION = r"""
## What checkpointing trades
Without remat, the backward pass needs every layer's intermediates: the block input, the pre-activation $hW_1 + b_1$ and the GELU output, so memory grows with $L \cdot B \cdot (D + 2F)$. With `jax.checkpoint` on the block, scan saves only each layer's input $h$ ($L \cdot B \cdot D$) and recomputes the rest during backprop: roughly one extra forward pass (about 33% more FLOPs) for a large cut in activation memory. Tests 3 to 5 give identical numbers, which is the point: remat is a scheduling choice, not a different model.

## Policies
`dots_with_no_batch_dims_saveable` keeps the matmul outputs (expensive to recompute) and recomputes the elementwise GELU (cheap, but its intermediates are large). It is a middle ground; `jax.checkpoint_policies` has others, such as `save_only_these_names` with `checkpoint_name` tags.

## Why scan over layers
A Python loop over 48 layers traces and compiles 48 copies of the block. `scan` compiles one, which cuts compile time dramatically for large models and is what most large JAX codebases do.

## Common bugs
- Applying `jax.checkpoint` to the whole loss function: it then saves only `x` and recomputes the entire forward during backward, but scan still stores its own per-step residuals unless the **body** is checkpointed.
- Closing over Python-level random numbers or side effects inside a checkpointed function: recomputation must be deterministic. Thread PRNG keys explicitly.

## Interview follow-ups
- With $L$ layers and memory for $k$ activations, what is the best placement of checkpoints? (Every $\sqrt{L}$ layers gives $O(\sqrt{L})$ memory for one extra forward.)
- How do you decide between remat, smaller micro-batches and activation offloading to host memory?
"""
