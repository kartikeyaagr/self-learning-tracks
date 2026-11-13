import numpy as np

META = dict(title='Bias-corrected EMA with resets via lax.scan', tags=['JAX', 'Time Series'], diff='Medium', runtime='jax',
            lessons=['air-fanout-v1:01.14'], related=['gradient-accumulation-with-scan', 'greedy-decode-loop-with-scan'],
            tol=[1e-4, 1e-6])

STATEMENT = r"""
`jax.lax.scan` is JAX's compiled loop: it threads a **carry** through a function applied to each slice of a sequence and stacks the per-step outputs. Python `for` loops inside `jit` are unrolled, so a 10,000-step loop becomes 10,000 copies of the body; `scan` compiles the body once.

Write `ema_with_resets(x, reset, decay, debias)`:

- `x`: `(T, D)` float array, `T >= 1`.
- `reset`: `(T,)` bool. `reset[t] = True` starts a new segment at step `t` (as at an episode boundary).
- `decay`: float in `[0, 1)`; `debias`: Python bool.

Carry a state `s` (starting at zeros) and a count `c` (starting at 0). At each step `t`:

1. If `reset[t]`, set `s = 0` and `c = 0` first.
2. $s \leftarrow \text{decay}\cdot s + (1 - \text{decay})\, x_t$ and $c \leftarrow c + 1$.
3. Output $s / (1 - \text{decay}^{c})$ if `debias`, else $s$.

Return `(ema, final_s)`: the `(T, D)` stacked outputs and the raw (not bias-corrected) state after the last step, `(D,)`. float32. Use `jax.lax.scan`.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def ema_with_resets(x, reset, decay, debias):
    """Return (ema (T, D), raw final state (D,)) computed with jax.lax.scan."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def ema_with_resets(x, reset, decay, debias):
    x = jnp.asarray(x, jnp.float32)
    reset = jnp.asarray(reset, bool)

    def step(carry, inp):
        s, c = carry
        xt, rt = inp
        s = jnp.where(rt, 0.0, s)                          # reset before the update
        c = jnp.where(rt, 0, c)
        s = decay * s + (1.0 - decay) * xt
        c = c + 1
        out = s / (1.0 - decay ** c) if debias else s     # debias is a Python bool: resolved at trace time
        return (s, c), out

    init = (jnp.zeros(x.shape[1], jnp.float32), jnp.zeros((), jnp.int32))
    (s, _), ema = jax.lax.scan(step, init, (x, reset))    # scan slices x and reset along axis 0
    return ema, s
'''

HINTS = [
    'The scan body has signature `step(carry, inp) -> (new_carry, out)`. Use `carry = (s, c)` and `inp = (x_t, reset_t)`; `jax.lax.scan(step, init, (x, reset))` slices both along the first axis.',
    'Resets are data, so use `jnp.where(reset_t, 0, s)` rather than a Python `if`. The carry must keep the same shape and dtype every step.',
    'Bias correction uses the count since the last reset: `s / (1 - decay ** c)`. The returned final state is the raw `s`.',
]

_r = np.random.default_rng(17)
TESTS = [
    (np.array([[1.0], [1.0], [1.0]]), np.array([False, False, False]), 0.5, False),
    (np.array([[1.0], [1.0], [1.0]]), np.array([False, False, False]), 0.5, True),
    (np.array([[2.0, 0.0], [4.0, 1.0], [6.0, 2.0], [8.0, 3.0]]), np.array([False, False, True, False]), 0.9, True),
    (_r.normal(size=(6, 3)), np.array([True, False, False, True, True, False]), 0.0, True),
    (_r.normal(size=(10, 4)), np.zeros(10, dtype=bool), 0.99, True),
    (_r.normal(size=(10, 4)), _r.random(10) < 0.3, 0.8, False),
    (_r.normal(size=(50, 2)) + 3.0, _r.random(50) < 0.1, 0.95, True),
]

EXPLANATION = r"""
## Why scan
`scan(f, init, xs)` is a functional `for` loop: it compiles `f` once and runs it `T` times inside XLA, carrying state between iterations. A Python loop under `jit` unrolls into `T` copies of the body, so compile time grows linearly with `T`. Scan also gives reverse-mode autodiff an efficient loop to differentiate.

## Carry rules
The carry must be a pytree with the **same structure, shapes and dtypes** on every iteration. Starting `c` as Python `0` and returning an int32 array is fine only if the dtypes match. Initialise it as a JAX int32 scalar to be explicit. Changing a float carry to float64 or a shape from `()` to `(1,)` fails at trace time.

## Bias correction
With $s_0 = 0$, after $c$ updates of a constant input $x$ the state is $(1 - \text{decay}^c)\,x$. Dividing by $1 - \text{decay}^c$ recovers $x$ exactly (test 2). This is the same correction Adam applies to its moment estimates, and it must restart from the reset, not from the global step.

## Cost
$O(TD)$ work, sequential over $T$. For a pure linear recurrence like this one, `jax.lax.associative_scan` can compute it in $O(\log T)$ depth, which matters on accelerators for long sequences.

## Common bugs
- Python `if rt:` on a traced bool inside the body: concretization error.
- Resetting after the update instead of before.
- Using the global step for bias correction, so post-reset values come out too small.

## Interview follow-ups
- How would you express this EMA with `associative_scan`? (Each step is an affine map $s \mapsto a s + b$; composition of affine maps is associative, and a reset sets $a = 0$.)
- How do you keep an EMA of model weights in a JAX training loop without doubling device memory traffic each step?
"""
