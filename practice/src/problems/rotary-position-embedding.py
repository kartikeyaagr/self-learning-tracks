import numpy as np

META = dict(title='RoPE at arbitrary positions in JAX', tags=['Transformers', 'JAX'], diff='Easy', runtime='jax',
            lessons=['aieng:04.01', 'ieng:13.06'], related=['rotary-position-embedding-rope', 'kv-cache-update-at-position'],
            tol=[1e-4, 1e-5])

STATEMENT = r"""
Rotary position embeddings rotate pairs of query/key features by an angle proportional to the token's position. In a serving system positions are not always `0 .. T-1`: a decode step sits at the end of a cache, and packed sequences restart at 0 in the middle of a row. So the positions come in as data.

Write `apply_rope(x, positions, base)`:

- `x`: `(B, T, H, d)` float32, `d` even.
- `positions`: `(B, T)` int, the position of each token.
- `base`: float (10000.0 in most models).

Use the **half-split** layout (as in GPT-NeoX and Llama): feature $i$ is paired with feature $i + d/2$ for $i < d/2$. With frequencies $\omega_i = \text{base}^{-2i/d}$ and angle $\theta = p\, \omega_i$ for a token at position $p$:

$$y_i = x_i \cos\theta - x_{i + d/2} \sin\theta, \qquad y_{i + d/2} = x_{i + d/2} \cos\theta + x_i \sin\theta$$

The same rotation applies to every head. Return `y` with the shape of `x`, float32.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def apply_rope(x, positions, base):
    """x (B, T, H, d), positions (B, T). Return rotated x."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def apply_rope(x, positions, base):
    x = jnp.asarray(x, jnp.float32)
    half = x.shape[-1] // 2
    freqs = base ** (-jnp.arange(half, dtype=jnp.float32) * 2 / x.shape[-1])     # (d/2,)
    theta = jnp.asarray(positions, jnp.float32)[..., None] * freqs              # (B, T, d/2)
    cos, sin = jnp.cos(theta)[:, :, None, :], jnp.sin(theta)[:, :, None, :]     # add the head axis
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
'''

HINTS = [
    'Frequencies depend only on the feature index: `base ** (-2 * arange(d/2) / d)`. Angles are `positions[..., None] * freqs`, shape `(B, T, d/2)`.',
    'Insert a head axis so the `(B, T, 1, d/2)` cos and sin broadcast over every head.',
    'Split `x` into the first and second halves of the last axis, rotate, and concatenate back.',
]

_r = np.random.default_rng(25)
TESTS = [
    (np.array([[[[1.0, 0.0]]]]), np.array([[1]]), 10000.0),
    (np.array([[[[1.0, 2.0, 3.0, 4.0]], [[1.0, 2.0, 3.0, 4.0]]]]), np.array([[0, 2]]), 100.0),
    (_r.normal(size=(2, 5, 3, 8)), np.tile(np.arange(5), (2, 1)), 10000.0),
    (_r.normal(size=(1, 4, 2, 8)), np.array([[0, 1, 0, 1]]), 10000.0),
    (_r.normal(size=(2, 1, 4, 16)), np.array([[37], [4095]]), 10000.0),
    (_r.normal(size=(1, 3, 1, 6)), np.array([[5, 5, 5]]), 500000.0),
    (_r.normal(size=(3, 2, 2, 4)), np.zeros((3, 2), int), 10000.0),
]

EXPLANATION = r"""
## Why rotation encodes relative position
Rotating $q$ by $p\omega$ and $k$ by $p'\omega$ makes their dot product depend on $(p - p')\omega$ only, because rotations compose: $R(p\omega)^\top R(p'\omega) = R((p' - p)\omega)$. Attention scores then depend on relative distance, while each vector is encoded with its absolute position, which is what makes caching work: a key is rotated once, when it is written.

## Positions as data
Test 4 restarts positions mid-row (two packed sequences of length 2); test 5 is a single decode step at position 4095. Taking positions as an array handles both without special cases. Position 0 is the identity (test 7).

## Layouts
Half-split pairs $(i, i + d/2)$; the original paper's interleaved layout pairs $(2i, 2i+1)$. Both are valid, but weights trained with one layout produce garbage with the other, a classic bug when porting checkpoints between frameworks.

## Complexity
$O(BTHd)$ elementwise work. In practice cos and sin are precomputed per position (a table of shape `(max_len, d/2)`) and gathered by `positions`, or fused into the attention kernel.

## Common bugs
- Computing angles in low precision: at position 4095 a bf16 angle is off by several radians. Keep the angle math in float32.
- Rotating the values `v` too: RoPE applies to queries and keys only.

## Interview follow-ups
- How do NTK-aware scaling and YaRN change `base` or the frequencies to extend the context length?
- Why does RoPE make the KV cache position-dependent, and what does that imply for prefix sharing?
"""
