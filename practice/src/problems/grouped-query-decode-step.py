import numpy as np

META = dict(title='Grouped-query attention for one decode step', tags=['Attention', 'Inference', 'JAX'], diff='Medium',
            runtime='jax', lessons=['ieng:14.04', 'ieng:03.06'],
            related=['multi-head-attention', 'kv-cache-update-at-position', 'paged-kv-cache-gather'], tol=[1e-4, 1e-5])

STATEMENT = r"""
In grouped-query attention (GQA), $H_q$ query heads share $H_{kv}$ key/value heads, with $G = H_q / H_{kv}$ query heads per group. The KV cache shrinks by a factor of $G$. During decoding there is one new query per sequence, attending over that sequence's cache.

Write `gqa_decode(q, k_cache, v_cache, lengths)`:

- `q`: `(B, Hq, d)`, the new token's queries.
- `k_cache`, `v_cache`: `(B, S, Hkv, d)`; `Hq` is a multiple of `Hkv`.
- `lengths`: `(B,)` int, `1 <= lengths[b] <= S`: only cache positions `< lengths[b]` are valid.
- Query head `h` uses KV head `h // G` (consecutive query heads share a KV head).

Each query head computes $\mathrm{softmax}\bigl(q k^\top / \sqrt d\bigr) v$ over the valid positions. Return `(B, Hq, d)` float32.

Do **not** copy the cache to `Hq` heads (no `jnp.repeat` of `k_cache` or `v_cache`): reshape the queries into groups instead, so the computation reads each cached key once.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def gqa_decode(q, k_cache, v_cache, lengths):
    """q (B, Hq, d), caches (B, S, Hkv, d), lengths (B,). Return (B, Hq, d)."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def gqa_decode(q, k_cache, v_cache, lengths):
    q, k, v = (jnp.asarray(a, jnp.float32) for a in (q, k_cache, v_cache))
    B, Hq, d = q.shape
    S, Hkv = k.shape[1], k.shape[2]
    qg = q.reshape(B, Hkv, Hq // Hkv, d)                       # head h -> group h // G, member h % G
    s = jnp.einsum('bngd,bsnd->bngs', qg, k) / jnp.sqrt(d)     # (B, Hkv, G, S), no cache copy
    valid = jnp.arange(S)[None, :] < jnp.asarray(lengths)[:, None]          # (B, S)
    s = jnp.where(valid[:, None, None, :], s, -jnp.inf)
    out = jnp.einsum('bngs,bsnd->bngd', jax.nn.softmax(s, axis=-1), v)
    return out.reshape(B, Hq, d)
'''

HINTS = [
    'Because query head `h` maps to KV head `h // G`, reshaping `q` from `(B, Hq, d)` to `(B, Hkv, G, d)` lines every group up with its KV head.',
    'One `einsum` gives the scores: `"bngd,bsnd->bngs"`. Another applies the weights to the values: `"bngs,bsnd->bngd"`.',
    'Build the `(B, S)` valid mask from `lengths` and broadcast it to `(B, 1, 1, S)` before the softmax.',
]

_r = np.random.default_rng(28)
_f = lambda *s: _r.normal(size=s)
TESTS = [
    (np.array([[[1.0, 0.0], [0.0, 1.0]]]), np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]]), np.array([[[[1.0, 1.0]], [[3.0, 5.0]]]]),
     np.array([2])),
    (np.ones((1, 2, 2)), np.array([[[[0.0, 0.0]], [[5.0, 5.0]]]]), np.array([[[[1.0, 2.0]], [[9.0, 9.0]]]]), np.array([1])),
    (_f(2, 8, 4), _f(2, 6, 2, 4), _f(2, 6, 2, 4), np.array([6, 3])),
    (_f(3, 4, 8), _f(3, 5, 4, 8), _f(3, 5, 4, 8), np.array([1, 5, 2])),
    (_f(1, 6, 4), _f(1, 7, 1, 4), _f(1, 7, 1, 4), np.array([7])),
    (4 * _f(2, 4, 16), 4 * _f(2, 9, 2, 16), _f(2, 9, 2, 16), np.array([9, 1])),
    (_f(4, 2, 2), _f(4, 3, 2, 2), _f(4, 3, 2, 2), np.array([3, 3, 2, 1])),
]

EXPLANATION = r"""
## Reshape the queries, not the cache
Query heads `g * G .. g * G + G - 1` all read KV head `g`. Reshaping `q` to `(B, Hkv, G, d)` makes that explicit, and the einsum contracts each group against its single KV head. `jnp.repeat(k, G, axis=2)` gives the same numbers but materialises a $G$-times larger cache every step, giving back the memory GQA saved. Test 5 is multi-query attention ($H_{kv} = 1$) and test 4 is ordinary multi-head attention ($G = 1$).

## Why decode is memory-bound
Each step reads the whole valid cache, $2 S H_{kv} d$ numbers per sequence, to do $O(S H_q d)$ FLOPs: about $G$ FLOPs per byte loaded. That arithmetic intensity is far below what a GPU needs to be compute-bound, so decode speed is set by memory bandwidth, and shrinking the cache (GQA, MQA, KV quantisation) directly speeds it up.

## Head ordering
"Consecutive heads share a KV head" (`h // G`) is the common convention. The other one, `h % Hkv`, would need a different reshape (`(B, G, Hkv, d)`). A checkpoint converted with the wrong convention loads fine and produces nonsense.

## Common bugs
- Masking with the wrong axis, or masking query heads instead of cache positions.
- Forgetting the $1/\sqrt{d}$ scale.

## Interview follow-ups
- Compute the KV cache saving of 32 query heads with 8 KV heads, and the effect on maximum batch size.
- How does multi-head latent attention (MLA) reduce the cache further?
"""
