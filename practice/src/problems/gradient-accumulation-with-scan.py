import numpy as np

META = dict(title='Token-weighted gradient accumulation with lax.scan', tags=['JAX', 'Training'], diff='Hard', runtime='jax',
            related=['gradient-accumulation', 'exponential-moving-average-with-scan', 'data-parallel-step-with-psum'],
            tol=[1e-4, 1e-5])

STATEMENT = r"""
When a batch does not fit in memory, split it into $M$ micro-batches, accumulate gradients and take one optimizer step. The accumulated gradient must equal the gradient of the **full** batch, even when micro-batches contain different numbers of real (unpadded) examples.

Model: a one-hidden-layer network with `params = {'w1': (D, H), 'b1': (H,), 'w2': (H,), 'b2': ()}`:

$$f(x) = \tanh(x W_1 + b_1)\, w_2 + b_2$$

Batch: `x` `(M, b, D)`, `y` `(M, b)` and `mask` `(M, b)` of 0/1 (1 = real example). The full-batch loss is the mean squared error over **real** examples only:

$$L = \frac{\sum_{m, i} \text{mask}_{m,i}\,\bigl(f(x_{m,i}) - y_{m,i}\bigr)^2}{\max\bigl(1,\ \sum_{m,i} \text{mask}_{m,i}\bigr)}$$

Write `accumulate_and_step(params, x, y, mask, lr)`:

1. Loop over micro-batches with `jax.lax.scan`, computing each micro-batch's **summed** masked loss and its gradient with `jax.value_and_grad`, and carrying the running gradient sum, loss sum and example count.
2. Divide by $\max(1, \text{count})$ to get $L$ and $\nabla L$.
3. Take one SGD step: $p \leftarrow p - \eta \nabla L$.

Return `(new_params, loss, grads)`: two dicts with the structure of `params` and a 0-d loss. float32. If no example is real, the loss is 0, the gradients are zero and the parameters are unchanged.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def accumulate_and_step(params, x, y, mask, lr):
    """x (M, b, D), y/mask (M, b). Return (new_params, loss, grads)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def _sum_loss(p, x, y, m):
    pred = jnp.tanh(x @ p['w1'] + p['b1']) @ p['w2'] + p['b2']      # (b,)
    return jnp.sum(m * (pred - y) ** 2)                               # summed, not averaged

def accumulate_and_step(params, x, y, mask, lr):
    tm = jax.tree_util.tree_map
    params = tm(lambda a: jnp.asarray(a, jnp.float32), params)
    x, y, mask = (jnp.asarray(a, jnp.float32) for a in (x, y, mask))
    vg = jax.value_and_grad(_sum_loss)

    def body(carry, mb):
        g_sum, l_sum, n = carry
        l, g = vg(params, *mb)
        return (tm(jnp.add, g_sum, g), l_sum + l, n + jnp.sum(mb[2])), None

    init = (tm(jnp.zeros_like, params), jnp.zeros(()), jnp.zeros(()))
    (g_sum, l_sum, n), _ = jax.lax.scan(body, init, (x, y, mask))     # scans along the micro-batch axis

    denom = jnp.maximum(n, 1.0)                                       # weight by examples, not by micro-batch
    grads = tm(lambda g: g / denom, g_sum)
    new_params = tm(lambda p, g: p - lr * g, params, grads)
    return new_params, l_sum / denom, grads
'''

HINTS = [
    'Differentiate the **summed** masked loss of a micro-batch. Averages of per-micro-batch means are wrong when the micro-batches have different numbers of real examples.',
    'The scan carry is `(grad_sum_tree, loss_sum, count)`. Start the gradient tree with `tree_map(jnp.zeros_like, params)` and add with `tree_map(jnp.add, ...)`. `xs = (x, y, mask)` scans all three along axis 0.',
    'After the scan, divide everything by `max(1, count)`. With no real examples the sums are 0, so the step is a no-op automatically.',
]

_r = np.random.default_rng(19)

def _p(D, H):
    return {'w1': _r.normal(size=(D, H)), 'b1': _r.normal(size=H) * 0.1, 'w2': _r.normal(size=H), 'b2': np.array(0.1)}

_m1 = np.array([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]], float)
TESTS = [
    ({'w1': np.zeros((1, 1)), 'b1': np.zeros(1), 'w2': np.zeros(1), 'b2': np.array(0.0)},
     np.zeros((2, 1, 1)), np.array([[1.0], [3.0]]), np.ones((2, 1)), 0.5),
    ({'w1': np.zeros((1, 1)), 'b1': np.zeros(1), 'w2': np.zeros(1), 'b2': np.array(0.0)},
     np.zeros((2, 2, 1)), np.array([[1.0, 1.0], [4.0, 0.0]]), np.array([[1.0, 1.0], [1.0, 0.0]]), 0.5),
    (_p(3, 4), _r.normal(size=(3, 4, 3)), _r.normal(size=(3, 4)), _m1, 0.1),
    (_p(3, 4), _r.normal(size=(1, 8, 3)), _r.normal(size=(1, 8)), np.ones((1, 8)), 0.05),
    (_p(2, 3), _r.normal(size=(4, 2, 2)), _r.normal(size=(4, 2)), np.zeros((4, 2)), 0.1),
    (_p(2, 3), _r.normal(size=(4, 3, 2)), _r.normal(size=(4, 3)),
     np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0], [0, 0, 0]], float), 0.2),
    (_p(5, 6), _r.normal(size=(6, 5, 5)), 3 * _r.normal(size=(6, 5)), (_r.random((6, 5)) < 0.7).astype(float), 0.01),
]

EXPLANATION = r"""
## Why sums, not means
The full-batch gradient is $\frac{1}{N}\sum_{\text{real}} \nabla \ell_i$. If each micro-batch returns its own mean and you average those, examples in a micro-batch with 1 real example get the weight of 4 examples in a full one (test 2: losses 1, 1 and 16 give $18/3 = 6$, not $\frac{(1+1)/2 + 16}{2} = 8.5$). Summing within micro-batches and dividing once by the total count makes accumulation exactly equal to the big batch, which is the whole point.

## Why `scan`
A Python loop under `jit` unrolls $M$ copies of the forward and backward pass, so compile time and code size grow with $M$. `scan` compiles the body once and keeps only one micro-batch's activations alive at a time, which is the memory saving accumulation exists for.

## Complexity
Same FLOPs as the full batch; activation memory drops by a factor of about $M$; one extra gradient-sized buffer for the running sum.

## Common bugs
- Averaging per-micro-batch means (unequal weighting, see above).
- Taking an optimizer step inside the scan body: that is $M$ small steps, not accumulation.
- Initialising the carry with Python `0.0` for the loss while the body returns a float32 array: `scan` requires the carry to keep the same type and shape.

## Interview follow-ups
- How does accumulation interact with batch norm? (Statistics are per micro-batch, so it is not equivalent; LayerNorm-based models do not have this problem.)
- How would you combine accumulation with data parallelism? (Accumulate locally, `psum` the gradient sums and counts across devices once per step.)
"""
