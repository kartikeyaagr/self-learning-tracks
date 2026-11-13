import numpy as np

META = dict(title='Batched key-padded attention with two vmaps', tags=['JAX', 'Attention'], diff='Medium', runtime='jax',
            lessons=['aieng:05.02', 'ieng:00.03'], related=['scaled-dot-product-attention', 'grouped-query-decode-step',
                                                            'model-ensemble-with-in-axes'], tol=[1e-4, 1e-5])

STATEMENT = r"""
In JAX you can write a function for **one** example and let `jax.vmap` add the batch dimensions. Attention is a good test: write it for a single head of a single sequence, then map it over heads and over the batch.

Write `batched_attention(q, k, v, lengths)`:

- `q`: `(B, H, T, d)` queries; `k`, `v`: `(B, H, S, d)` keys and values. float32.
- `lengths`: `(B,)` int. Sequence `b` has `lengths[b]` real keys at positions `0 .. lengths[b] - 1`. The rest are padding. Every head of a sequence shares the same length, and `1 <= lengths[b] <= S`.

For one head with queries $Q$ `(T, d)`, keys $K$ and values $V$ `(S, d)` and length $n$:

$$O = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt d} + M\right) V, \qquad M_{ij} = \begin{cases} 0 & j < n \\ -\infty & j \ge n \end{cases}$$

Return `O` with shape `(B, H, T, d)`, float32.

Structure the code as a function `attend(q, k, v, n)` for **one** head (2-D arrays and a scalar length), and get the batched version from two nested `jax.vmap` calls. The length is per sequence, not per head, so the inner map must not map over it (`in_axes=(0, 0, 0, None)`).

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def batched_attention(q, k, v, lengths):
    """q (B, H, T, d), k/v (B, H, S, d), lengths (B,). Return (B, H, T, d)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def attend(q, k, v, n):
    """One head: q (T, d), k/v (S, d), n = number of real keys."""
    scores = q @ k.T / jnp.sqrt(q.shape[-1])                  # (T, S)
    keep = jnp.arange(k.shape[0]) < n                         # (S,), broadcasts over the T rows
    scores = jnp.where(keep, scores, -jnp.inf)                # n >= 1, so no row is all -inf
    return jax.nn.softmax(scores, axis=-1) @ v                # (T, d)

def batched_attention(q, k, v, lengths):
    q, k, v = (jnp.asarray(a, jnp.float32) for a in (q, k, v))
    over_heads = jax.vmap(attend, in_axes=(0, 0, 0, None))   # heads share the sequence length
    over_batch = jax.vmap(over_heads, in_axes=(0, 0, 0, 0))   # one length per sequence
    return over_batch(q, k, v, jnp.asarray(lengths))
'''

HINTS = [
    'Write `attend` with no batch or head axes at all: `scores = q @ k.T / sqrt(d)` is `(T, S)`.',
    'Inside `attend`, `jnp.arange(S) < n` is a `(S,)` boolean that broadcasts across the query rows. Use `jnp.where(keep, scores, -jnp.inf)` before the softmax.',
    'The inner `vmap` maps heads and must broadcast the length: `in_axes=(0, 0, 0, None)`. The outer one maps everything, including `lengths`, along axis 0.',
]

_r = np.random.default_rng(31)
_f = lambda *s: _r.normal(size=s)
TESTS = [
    (np.array([[[[1.0, 0.0]]]]), np.array([[[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]]]),
     np.array([[[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]]]), np.array([2])),
    (np.zeros((2, 1, 1, 2)), np.ones((2, 1, 3, 2)), np.arange(12.0).reshape(2, 1, 3, 2), np.array([3, 1])),
    (_f(2, 3, 4, 8), _f(2, 3, 5, 8), _f(2, 3, 5, 8), np.array([5, 2])),
    (_f(3, 2, 1, 4), _f(3, 2, 6, 4), _f(3, 2, 6, 4), np.array([1, 6, 3])),
    (_f(1, 4, 3, 16), _f(1, 4, 3, 16), _f(1, 4, 3, 16), np.array([3])),
    (10 * _f(2, 2, 3, 4), 10 * _f(2, 2, 7, 4), _f(2, 2, 7, 4), np.array([4, 7])),
    (_f(4, 1, 2, 2), _f(4, 1, 2, 2), _f(4, 1, 2, 2), np.array([1, 2, 1, 2])),
]

EXPLANATION = r"""
## Why two vmaps
`vmap` adds one batch axis per call. `attend` handles `(T, d)`; the first `vmap` lifts it to `(H, T, d)`; the second to `(B, H, T, d)`. `in_axes` says which argument has the new axis: the head map broadcasts `n` (`None`) because every head of a sequence shares one length, while the batch map slices `lengths` along axis 0. JAX turns the nested maps into batched matmuls, so this is as fast as a hand-written `einsum`.

## Masking with `-inf`
Padded keys get score $-\infty$, so their softmax weight is exactly 0. This is safe only because each row keeps at least one real key; a fully masked row would be `0/0 = nan`. Test 1 shows the padding value (`[100, 100]`) has no effect on the output.

## Complexity
$O(B H T S d)$ time and $O(B H T S)$ memory for the score matrices, the same as batched attention written by hand.

## Common bugs
- Inner `in_axes=(0, 0, 0, 0)`: JAX then expects `lengths` to have a head axis and raises a size mismatch.
- Masking queries instead of keys (`arange(T) < n`): the mask must run along the key axis, the last axis of `scores`.
- Masking after the softmax by multiplying with 0: the remaining weights no longer sum to 1.

## Interview follow-ups
- How would you add a causal mask, and how does it combine with key padding?
- When does `vmap` produce worse code than a hand-batched `einsum`? (Rarely; mostly with data-dependent control flow, where `cond` becomes `select` and both branches run.)
"""
