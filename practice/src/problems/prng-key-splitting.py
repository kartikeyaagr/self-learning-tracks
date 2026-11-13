import numpy as np

META = dict(title='Explicit PRNG keys: split, never reuse', tags=['JAX', 'Neural Networks'], diff='Easy', runtime='jax',
            related=['functional-array-update', 'model-ensemble-with-in-axes'], tol=[1e-4, 1e-5])

STATEMENT = r"""
JAX has no global random state. Every random call takes a **key**, and the same key always gives the same numbers. To get fresh randomness you split a key into new subkeys, and each subkey is used exactly once.

Write `noisy_dropout_layer(seed, x, rate, sigma)` for `x` of shape `(B, D)`:

1. Make the key with `jax.random.PRNGKey(seed)` and split it **once** into three subkeys: `k_w, k_drop, k_noise = jax.random.split(key, 3)`.
2. Weights: `w = jax.random.normal(k_w, (D, D)) / sqrt(D)`, then `h = x @ w`.
3. Inverted dropout on `h` with `keep = jax.random.bernoulli(k_drop, 1 - rate, h.shape)`: kept entries are divided by `1 - rate`, dropped entries become `0`. `0 <= rate < 1`.
4. Per-row noise: split `k_noise` into `B` row keys with `jax.random.split(k_noise, B)`. Row `b` gets `jax.random.normal(row_keys[b], (D,))`. Add `sigma` times this noise to the dropout output.

Return `(w, out)`: float32 arrays of shapes `(D, D)` and `(B, D)`.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime and a recent JAX (0.5 or later; the expected numbers come from JAX's default threefry generator). JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def noisy_dropout_layer(seed, x, rate, sigma):
    """Return (w (D, D), out (B, D)) using keys split from jax.random.PRNGKey(seed)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def noisy_dropout_layer(seed, x, rate, sigma):
    x = jnp.asarray(x, jnp.float32)
    B, D = x.shape
    k_w, k_drop, k_noise = jax.random.split(jax.random.PRNGKey(seed), 3)   # one split, three independent streams

    w = jax.random.normal(k_w, (D, D)) / jnp.sqrt(D)
    h = x @ w

    keep = jax.random.bernoulli(k_drop, 1.0 - rate, h.shape)
    dropped = jnp.where(keep, h / (1.0 - rate), 0.0)                       # inverted dropout keeps E[out] = h

    row_keys = jax.random.split(k_noise, B)                                # (B, 2): one key per row
    noise = jax.vmap(lambda k: jax.random.normal(k, (D,)))(row_keys)       # vmap over keys = loop over rows
    return w, dropped + sigma * noise
'''

HINTS = [
    '`jax.random.split(key, 3)` returns three keys you can unpack. Using `key` itself again after splitting it is the classic bug.',
    'Dropout: `jnp.where(keep, h / (1 - rate), 0.0)` with `keep` from `jax.random.bernoulli(k_drop, 1 - rate, h.shape)`.',
    'For the noise, `row_keys = jax.random.split(k_noise, B)` is an array of keys. `jax.vmap(lambda k: jax.random.normal(k, (D,)))(row_keys)` draws one row per key.',
]

_r = np.random.default_rng(5)
TESTS = [
    (0, np.ones((2, 3)), 0.0, 0.0),
    (1, np.array([[1.0, -1.0], [0.5, 2.0]]), 0.5, 0.1),
    (7, _r.normal(size=(4, 5)), 0.25, 0.0),
    (7, _r.normal(size=(4, 5)), 0.25, 1.0),
    (123, np.zeros((3, 4)), 0.9, 0.5),
    (2024, _r.normal(size=(6, 8)), 0.1, 0.05),
    (42, _r.normal(size=(1, 16)), 0.0, 2.0),
]

EXPLANATION = r"""
## Why keys instead of a global seed
A global generator is hidden mutable state. That breaks `jit` (the trace would bake one draw in), `vmap` (every batch element would share a stream) and reproducibility across devices. A key is an ordinary value, so random functions are pure: same key in, same numbers out. Splitting is a deterministic hash that yields statistically independent keys.

## The discipline
- Split once per consumer and never use a key after you split it.
- For many independent draws (rows, ensemble members, devices), split into `n` keys and `vmap` over them. The result does not depend on batch size or device count, unlike drawing one big tensor and slicing it.

## Cost
Key splitting and threefry generation are cheap elementwise integer ops that XLA fuses. Generating per-row noise with `vmap` costs the same as one `(B, D)` draw.

## Common bugs
- Reusing `key` for both the weights and the dropout mask: the two are then correlated. Here the expected output changes completely.
- `jax.random.split(key)` gives 2 keys, not 3: unpacking fails or, worse, you split again from the parent.
- Forgetting the `1 / (1 - rate)` rescale, so activations shrink at train time compared to evaluation.

## Interview follow-ups
- How do you get a different dropout mask on every training step inside a `jit`-compiled loop? (Carry the key in the train state and split it each step, or `jax.random.fold_in(key, step)`.)
- In multi-device training, how do you give each device its own dropout mask but identical weight init? (`fold_in` the device index for dropout, share the init key.)
"""
