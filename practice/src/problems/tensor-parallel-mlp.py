import numpy as np

META = dict(title='Megatron-style tensor-parallel MLP', tags=['Distributed Computing', 'JAX', 'Transformers'],
            diff='Medium', runtime='jax', lessons=['ieng:18.03'],
            related=['device-mesh-and-named-sharding', 'data-parallel-step-with-psum', 'ring-all-reduce-with-ppermute'],
            tol=[1e-4, 1e-5])

STATEMENT = r"""
Tensor parallelism splits one layer's weights across devices. For a transformer MLP $y = \mathrm{gelu}(x W_1 + b_1) W_2 + b_2$, the standard split needs only **one** all-reduce:

- $W_1$ `(D, F)` and $b_1$ `(F,)` are split by **columns**: device $n$ gets columns $[nF/N, (n+1)F/N)$. GELU is elementwise, so each device can apply it to its own slice of the hidden layer.
- $W_2$ `(F, D)` is split by **rows**, with the matching row range. Each device computes a partial product $h_n W_2^{(n)}$ of shape `(B, D)`.
- The partial products are summed with `psum`; then $b_2$ is added once.

Write `tp_mlp(x, w1, b1, w2, b2, n_shards)`:

- `x`: `(B, D)`; full weights as above; `n_shards` (a Python int) divides `F`. GELU is the exact (erf) form, `jax.nn.gelu(z, approximate=False)`.
- Split the weights yourself, then run a per-device function with `jax.vmap(..., axis_name='tp')` (one CPU device simulates $N$).

Return `(y, partials)`: `y` is `(B, D)`, the output from device 0 (all devices hold the same result after the all-reduce); `partials` is `(N, B, D)`, each device's $h_n W_2^{(n)}$ **before** the all-reduce. float32.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def tp_mlp(x, w1, b1, w2, b2, n_shards):
    """Return (y (B, D), partial products before the all-reduce (N, B, D))."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def _device(x, w1, b1, w2, b2):
    h = jax.nn.gelu(x @ w1 + b1, approximate=False)      # (B, F/N): this device's slice of the hidden layer
    partial = h @ w2                                     # (B, D), a partial sum over the hidden dimension
    y = jax.lax.psum(partial, axis_name='tp') + b2       # one all-reduce; bias added after, exactly once
    return y, partial

def tp_mlp(x, w1, b1, w2, b2, n_shards):
    x, w1, b1, w2, b2 = (jnp.asarray(a, jnp.float32) for a in (x, w1, b1, w2, b2))
    D, F = w1.shape
    f = F // n_shards
    w1s = w1.reshape(D, n_shards, f).transpose(1, 0, 2)   # (N, D, F/N): column blocks
    b1s = b1.reshape(n_shards, f)                         # (N, F/N)
    w2s = w2.reshape(n_shards, f, D)                      # (N, F/N, D): matching row blocks
    run = jax.vmap(_device, in_axes=(None, 0, 0, 0, None), axis_name='tp')
    y, partials = run(x, w1s, b1s, w2s, b2)
    return y[0], partials
'''

HINTS = [
    'Column blocks of `w1`: `w1.reshape(D, N, F // N).transpose(1, 0, 2)`. Row blocks of `w2`: `w2.reshape(N, F // N, D)`. `b1` splits like the columns of `w1`.',
    'The per-device function computes its hidden slice, applies GELU, multiplies by its rows of `w2`, and calls `jax.lax.psum(partial, "tp")`.',
    'Add `b2` **after** the `psum`. Adding it on every device before the sum adds it $N$ times.',
]

_r = np.random.default_rng(35)

def _w(B, D, F, s=0.5):
    return (_r.normal(size=(B, D)), s * _r.normal(size=(D, F)), 0.1 * _r.normal(size=F),
            s * _r.normal(size=(F, D)), _r.normal(size=D))

TESTS = [
    (np.array([[1.0, 0.0]]), np.array([[1.0, 2.0], [0.0, 0.0]]), np.zeros(2), np.array([[1.0, 0.0], [0.0, 1.0]]),
     np.array([0.5, 0.5]), 2),
    (np.array([[1.0, -1.0]]), np.eye(2), np.zeros(2), np.eye(2), np.zeros(2), 1),
    _w(3, 4, 8) + (2,),
    _w(3, 4, 8) + (4,),
    _w(2, 6, 12) + (3,),
    _w(5, 8, 16, 0.3) + (8,),
    _w(1, 2, 6) + (6,),
]

EXPLANATION = r"""
## Why this split needs one collective
Splitting $W_1$ by columns gives each device complete hidden units, so the nonlinearity is applied locally with no communication. The second matmul contracts over the hidden dimension, which is now split, so each device produces a partial sum and a single all-reduce completes it: $\sum_n h_n W_2^{(n)} = h W_2$. Splitting $W_1$ by rows instead would need an all-reduce **before** GELU, since $\mathrm{gelu}(a + b) \ne \mathrm{gelu}(a) + \mathrm{gelu}(b)$.

## Memory and communication
Each device stores $1/N$ of both weight matrices and of the hidden activations. Per layer and forward pass it sends one all-reduce of a `(B, D)` activation. Attention splits the same way, by heads, so a transformer block costs two all-reduces forward and two backward.

## Why it stays inside a node
The all-reduce is on the critical path of every layer and cannot be overlapped with that layer's compute, so tensor parallelism is usually limited to GPUs connected by NVLink (often 8), with data or pipeline parallelism across nodes.

## Common bugs
- Adding $b_2$ before the `psum` (tests 1 and 3 to 7 catch it with a nonzero bias).
- Splitting $W_2$ by columns: shapes still work out for square cases but the math is wrong.
- Non-matching ranges: device $n$'s rows of $W_2$ must be the same hidden units as its columns of $W_1$.

## Interview follow-ups
- How does sequence parallelism replace the all-reduce with a reduce-scatter and an all-gather, and what does it save?
- With 8-way tensor parallelism, how is a GQA model with 4 KV heads sharded?
"""
