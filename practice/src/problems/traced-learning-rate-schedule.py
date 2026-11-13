import numpy as np

META = dict(title='Warmup-cosine schedule on a traced step', tags=['JAX', 'Training', 'Optimization'], diff='Medium',
            runtime='jax', lessons=['air-fanout-v1:05.05'], related=['static-arguments-in-jit', 'adamw'],
            tol=[1e-5, 1e-8])

STATEMENT = r"""
A training step compiled with `jax.jit` receives the step counter as a traced array. Python `if step < warmup:` can't branch on a tracer, and making the step static would recompile every step. Schedules are therefore written with array ops that compute every branch and select one.

Write `lr_at(steps, base_lr, warmup, total, min_lr)` decorated with `@jax.jit`, where **all arguments are traced**:

- `steps`: int array of any shape (step numbers $t \ge 0$).
- `base_lr`, `min_lr`: floats; `warmup`: int $\ge 0$; `total`: int $>$ `warmup`.

The schedule is

$$
\eta(t) = \begin{cases}
\text{base} \cdot \dfrac{t + 1}{\text{warmup}} & t < \text{warmup} \\[2mm]
\text{min} + \dfrac{\text{base} - \text{min}}{2}\left(1 + \cos\left(\pi \dfrac{t - \text{warmup}}{\text{total} - \text{warmup}}\right)\right) & \text{warmup} \le t < \text{total} \\[2mm]
\text{min} & t \ge \text{total}
\end{cases}
$$

With `warmup = 0` the first case never applies. Return a float32 array with the same shape as `steps`.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

@jax.jit
def lr_at(steps, base_lr, warmup, total, min_lr):
    """Learning rate for each step (same shape as steps, float32). All arguments are traced."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

@jax.jit
def lr_at(steps, base_lr, warmup, total, min_lr):
    t = steps.astype(jnp.float32)
    warm = base_lr * (t + 1.0) / jnp.maximum(warmup, 1)        # guard: never divide by 0, even in an unused branch
    progress = jnp.clip((t - warmup) / (total - warmup), 0.0, 1.0)
    cosine = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + jnp.cos(jnp.pi * progress))
    lr = jnp.where(t < warmup, warm, cosine)                     # both branches are computed, one is selected
    return jnp.where(t >= total, min_lr, lr).astype(jnp.float32)
'''

HINTS = [
    'Under `jit`, `if t < warmup:` raises a concretization error. Compute each case as an array and choose with `jnp.where(condition, a, b)`.',
    'Every branch is evaluated for every step, so keep each one finite: divide by `jnp.maximum(warmup, 1)` and clip the cosine progress to `[0, 1]`.',
    'Cast `steps` to float32 before dividing, and apply the `t >= total` case last.',
]

TESTS = [
    (np.array([0, 1, 2, 3, 4]), 1.0, 2, 4, 0.0),
    (np.arange(0, 12, 2), 0.1, 0, 10, 0.01),
    (np.array([0, 9, 10, 55, 100, 101, 5000]), 3e-4, 10, 101, 3e-5),
    (np.array([[0, 1], [2, 3]]), 1.0, 4, 8, 0.5),
    (np.array([7]), 2.0, 8, 9, 0.0),
    (np.arange(0, 1000, 37), 1e-3, 100, 900, 0.0),
    (np.array(5), 0.5, 3, 6, 0.1),
]

EXPLANATION = r"""
## Why `jnp.where`
`jit` builds one program for all step values, so control flow that depends on the step must become data flow. `jnp.where` evaluates both candidates and picks per element. For a schedule the extra work is a handful of scalar ops, far cheaper than recompiling each step (which is what `static_argnames=('steps',)` would cause).

## Keep unused branches finite
Both branches are computed even where they are discarded. If the warmup branch divides by zero when `warmup = 0`, the forward value is still fine but any gradient through the schedule (for example in learned-schedule research) becomes `nan`. Guarding with `jnp.maximum(warmup, 1)` and clipping `progress` keeps every branch finite.

## `lax.cond` vs `jnp.where`
`jax.lax.cond(pred, f, g, x)` runs only one branch, but needs a scalar predicate, and under `vmap` it turns into `select` anyway. For cheap elementwise branches `jnp.where` is the idiomatic choice.

## Common bugs
- `int / int` in the warmup branch done in integer arithmetic.
- Off-by-one: the warmup ramp here ends at `base_lr` exactly on step `warmup - 1`.
- Forgetting the final `t >= total` clamp, so the cosine wraps back up.

## Interview follow-ups
- Why do large-batch Transformer runs need warmup at all? (Adam's second-moment estimates are noisy early, and large early updates can push training into an unstable region.)
- How would you implement warmup-stable-decay (WSD) and resume it correctly from a checkpoint?
"""
