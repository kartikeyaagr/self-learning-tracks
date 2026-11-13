import numpy as np

META = dict(title='fp16 training step with dynamic loss scaling', tags=['JAX', 'Training', 'Quantization'], diff='Medium',
            runtime='jax', lessons=['ieng:11.01', 'ieng:11.02'],
            related=['global-norm-gradient-clipping', 'int8-matmul-with-per-channel-scales'], tol=[2e-3, 1e-4])

STATEMENT = r"""
Mixed-precision training keeps float32 **master** weights but runs the forward and backward pass in float16. float16 tops out at 65504 and flushes values below about $6 \times 10^{-8}$ to zero, so small gradients vanish. **Loss scaling** multiplies the loss by a large factor $S$ before differentiating (shifting gradients into fp16's range) and divides the gradients by $S$ afterwards. **Dynamic** scaling lowers $S$ when gradients overflow and raises it after a run of good steps.

Model: linear regression, `params = {'w': (D,), 'b': ()}` in float32, loss $L = \mathrm{mean}\bigl((x w + b - y)^2\bigr)$.

Write `scaled_step(params, x, y, lr, scale, good_steps, growth_interval)`:

1. Cast `params`, `x` (`(N, D)`) and `y` (`(N,)`) to **float16**. Compute the scaled loss $S \cdot L$ in float16 (cast `scale` to float16 too) and its gradient with respect to the float16 parameters.
2. Cast the gradients to float32 and divide by `scale`.
3. If every gradient value is finite:
   - $p \leftarrow p - \eta\, g$ on the float32 master weights;
   - `good_steps += 1`; if it reaches `growth_interval`, set `scale *= 2` and `good_steps = 0`.
4. Otherwise (overflow): leave the parameters unchanged, `scale /= 2`, `good_steps = 0`.

Return `(new_params, new_scale, new_good_steps, finite)`: a float32 dict, a 0-d float32, a 0-d int and a 0-d bool. Use `jnp.where`, not Python `if`, on array values.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime. Results are compared with a loose tolerance because float16 rounding can differ slightly between JAX versions.
"""

STARTER = '''import jax
import jax.numpy as jnp

def scaled_step(params, x, y, lr, scale, good_steps, growth_interval):
    """Return (new_params f32, new_scale, new_good_steps, finite)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def scaled_step(params, x, y, lr, scale, good_steps, growth_interval):
    tm = jax.tree_util.tree_map
    master = tm(lambda a: jnp.asarray(a, jnp.float32), params)
    x16, y16 = jnp.asarray(x, jnp.float16), jnp.asarray(y, jnp.float16)
    scale = jnp.asarray(scale, jnp.float32)

    def scaled_loss(p16):
        err = x16 @ p16['w'] + p16['b'] - y16
        return jnp.mean(err * err) * scale.astype(jnp.float16)      # overflow here shows up as inf grads

    g16 = jax.grad(scaled_loss)(tm(lambda a: a.astype(jnp.float16), master))
    grads = tm(lambda g: g.astype(jnp.float32) / scale, g16)         # unscale in float32
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))

    stepped = tm(lambda p, g: p - lr * g, master, grads)
    new_params = tm(lambda new, old: jnp.where(finite, new, old), stepped, master)

    good = jnp.where(finite, jnp.asarray(good_steps) + 1, 0)
    grow = good >= growth_interval
    new_scale = jnp.where(finite, jnp.where(grow, scale * 2, scale), scale / 2)
    return new_params, new_scale, jnp.where(grow, 0, good), finite
'''

HINTS = [
    'Differentiate a function of the **float16** copy of the parameters: `jax.grad(scaled_loss)(params16)`. The gradients come back in float16, so overflow appears as `inf` or `nan` there.',
    'Unscale after casting to float32 (`g.astype(jnp.float32) / scale`); dividing in float16 can underflow the values you just rescued.',
    'Compute the stepped parameters unconditionally, then select with `jnp.where(finite, stepped, master)`. The scale logic is two nested `jnp.where`s.',
]

_r = np.random.default_rng(20)
_p = lambda D: {'w': _r.normal(size=D), 'b': np.array(0.2)}
TESTS = [
    ({'w': np.array([1.0, 0.0]), 'b': np.array(0.0)}, np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([0.0, 1.0]),
     0.1, 1024.0, 0, 100),
    ({'w': np.array([1.0, 0.0]), 'b': np.array(0.0)}, np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([0.0, 1.0]),
     0.1, 65536.0, 5, 100),
    (_p(4), _r.normal(size=(8, 4)), _r.normal(size=8), 0.05, 256.0, 9, 10),
    (_p(4), 20 * _r.normal(size=(8, 4)), 20 * _r.normal(size=8), 0.01, 32768.0, 3, 10),
    (_p(3), 1e-3 * _r.normal(size=(16, 3)), 1e-3 * _r.normal(size=16), 0.1, 32768.0, 0, 2000),
    (_p(5), _r.normal(size=(4, 5)), _r.normal(size=4), 0.1, 1.0, 1, 2),
    (_p(2), np.array([[300.0, 0.0], [0.0, 1.0]]), np.array([0.0, 0.0]), 0.1, 8.0, 7, 8),
]

EXPLANATION = r"""
## What scaling fixes
Gradients of small losses (test 5, with inputs around $10^{-3}$) have components around $10^{-7}$ that float16 rounds to zero or to a few bits of precision. Multiplying the loss by $S$ multiplies every gradient by $S$ (the chain rule is linear), moving them into float16's normal range. Dividing in float32 afterwards restores the true gradient. The master weights stay float32 because $\eta g$ is often smaller than float16's spacing around $p$, so fp16 weights would stop changing.

## Why dynamic
No single $S$ works for a whole run: too small and gradients underflow, too large and they overflow (test 2: $65536 \times$ a loss of 1 is above 65504, so the scaled loss itself is `inf`). Halving on overflow and doubling after `growth_interval` clean steps keeps $S$ near the largest safe value. Skipped steps are rare in practice, so they barely cost progress.

## Why bf16 rarely needs this
bfloat16 has float32's exponent range (8 bits) with less precision (7 mantissa bits). Gradients do not underflow, so bf16 training usually runs without a loss scaler.

## Common bugs
- Unscaling in float16: dividing by $S$ in fp16 underflows again.
- Checking `isfinite` on the loss only: the loss can be finite while a gradient overflows.
- Updating the parameters before checking for overflow: one `inf` step ruins the model.

## Interview follow-ups
- Why must gradient clipping happen **after** unscaling?
- In data-parallel training, how do workers agree to skip a step? (All-reduce the finite flag, or check the all-reduced gradients.)
"""
