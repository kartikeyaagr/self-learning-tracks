import numpy as np

META = dict(title='Data-parallel SGD step with psum', tags=['Distributed Computing', 'JAX', 'Training'], diff='Medium',
            runtime='jax', lessons=['ieng:18.07', 'ieng:18.04'],
            related=['gradient-accumulation-with-scan', 'device-mesh-and-named-sharding', 'ring-all-reduce-with-ppermute'],
            tol=[1e-4, 1e-5])

STATEMENT = r"""
In data parallelism every device holds the full model and a slice of the batch. Each computes local gradients, an all-reduce (`jax.lax.psum`) sums them across devices, and every replica applies the same update. You only have one CPU device, but `jax.vmap(..., axis_name='devices')` runs collectives over a mapped axis exactly as `shard_map` would over real devices.

Model: linear regression, `params = {'w': (D,), 'b': ()}`. Device $n$ holds `x[n]` `(b, D)`, `y[n]` `(b,)` and `mask[n]` `(b,)` of 0/1 (padding examples are 0). The global loss is the mean squared error over all **real** examples on all devices:

$$L = \frac{\sum_{n,i} \text{mask}_{n,i} (x_{n,i} \cdot w + b - y_{n,i})^2}{\max\bigl(1, \sum_{n,i} \text{mask}_{n,i}\bigr)}$$

Write `dp_step(params, x, y, mask, lr)`:

1. A per-device function computes the **local** summed loss, its gradient and the local count of real examples.
2. `psum` the gradient (a pytree), the summed loss and the count over the axis `'devices'`, divide, and apply $p \leftarrow p - \eta \nabla L$.
3. Run it with `jax.vmap(per_device, in_axes=(None, 0, 0, 0, None), axis_name='devices')`.

Return `(new_params, loss)` as the vmap produces them: `new_params` has leaves with a leading device axis (`'w'`: `(N, D)`, `'b'`: `(N,)`), and every row must be identical; `loss` is `(N,)`, also identical across devices. float32.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def dp_step(params, x, y, mask, lr):
    """x (N, b, D), y/mask (N, b). Return (per-device new params, per-device loss (N,))."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def _local_sum_loss(p, x, y, m):
    return jnp.sum(m * (x @ p['w'] + p['b'] - y) ** 2)

def _per_device(params, x, y, m, lr):
    loss_sum, g = jax.value_and_grad(_local_sum_loss)(params, x, y, m)
    # One all-reduce for everything: psum accepts a pytree. Summing counts, not averaging means,
    # keeps the result correct when devices hold different numbers of real examples.
    g, loss_sum, n = jax.lax.psum((g, loss_sum, jnp.sum(m)), axis_name='devices')
    n = jnp.maximum(n, 1.0)
    new = jax.tree_util.tree_map(lambda p, g: p - lr * g / n, params, g)
    return new, loss_sum / n

def dp_step(params, x, y, mask, lr):
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), params)
    x, y, mask = (jnp.asarray(a, jnp.float32) for a in (x, y, mask))
    step = jax.vmap(_per_device, in_axes=(None, 0, 0, 0, None), axis_name='devices')
    return step(params, x, y, mask, lr)
'''

HINTS = [
    'Inside the per-device function, `jax.value_and_grad` of the **summed** local loss gives `(loss_sum, grads)`. Count the local real examples with `jnp.sum(mask)`.',
    '`jax.lax.psum(tree, axis_name="devices")` sums every leaf of a pytree across the mapped axis. Do it for the gradients, the loss sum and the count.',
    'Divide by the global count (at least 1) after the `psum`. `jax.lax.pmean` of local means would weight devices equally regardless of how many real examples they hold.',
]

_r = np.random.default_rng(34)
_p = lambda D: {'w': _r.normal(size=D), 'b': np.array(0.1)}
TESTS = [
    ({'w': np.zeros(1), 'b': np.array(0.0)}, np.zeros((2, 1, 1)), np.array([[2.0], [4.0]]), np.ones((2, 1)), 0.25),
    ({'w': np.zeros(1), 'b': np.array(0.0)}, np.zeros((2, 2, 1)), np.array([[2.0, 2.0], [8.0, 0.0]]),
     np.array([[1.0, 1.0], [1.0, 0.0]]), 0.25),
    (_p(3), _r.normal(size=(4, 5, 3)), _r.normal(size=(4, 5)), np.ones((4, 5)), 0.1),
    (_p(3), _r.normal(size=(4, 5, 3)), _r.normal(size=(4, 5)), (_r.random((4, 5)) < 0.6).astype(float), 0.1),
    (_p(2), _r.normal(size=(3, 2, 2)), _r.normal(size=(3, 2)), np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]), 0.5),
    (_p(2), _r.normal(size=(2, 3, 2)), _r.normal(size=(2, 3)), np.zeros((2, 3)), 0.5),
    (_p(6), _r.normal(size=(8, 4, 6)), 5 * _r.normal(size=(8, 4)), np.ones((8, 4)), 0.01),
]

EXPLANATION = r"""
## Why the replicas stay identical
Every device starts with the same parameters and receives the same all-reduced gradient, so every device computes the same update; no parameter broadcast is needed. That invariant is what makes data parallelism simple, and why non-determinism in the reduction (different summation orders on different devices) is a real concern at scale.

## Sum, then divide once
With unequal real counts (test 2: device 0 has two examples, device 1 has one), averaging local means with `pmean` gives each device's examples different weights. Summing losses, gradients and counts and dividing once reproduces the single-device full-batch step exactly. It is the same reasoning as gradient accumulation, applied across devices instead of across time.

## vmap as a stand-in for devices
`vmap(..., axis_name=...)` gives collectives like `psum`, `all_gather` and `ppermute` the same semantics they have under `shard_map` with a mesh, so the code moves to real hardware by swapping the transform. With `shard_map`, each device would get its slice of `x` from a `P('devices')` input spec and the replicated parameters from `P()`.

## Communication cost
One all-reduce of the gradient per step: about $2 \cdot |\theta|$ bytes sent per device with a ring algorithm, independent of the number of devices. Overlapping it with the backward pass (reducing layer by layer as gradients become ready) hides most of that cost.

## Interview follow-ups
- What does FSDP / ZeRO change, and which collectives replace the single all-reduce? (Reduce-scatter of gradients, all-gather of parameters.)
- Why does the effective learning rate change when you scale from 8 to 64 devices with a fixed per-device batch?
"""
