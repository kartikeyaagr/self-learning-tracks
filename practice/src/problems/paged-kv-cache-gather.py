import numpy as np

META = dict(title='Decode attention over a paged KV cache', tags=['Inference', 'Attention', 'JAX'], diff='Medium',
            runtime='jax', lessons=['ieng:08.02', 'ieng:08.03'],
            related=['kv-cache-update-at-position', 'grouped-query-decode-step', 'blockwise-attention-with-online-softmax'],
            tol=[1e-4, 1e-5])

STATEMENT = r"""
A paged KV cache stores keys and values in a shared pool of fixed-size **pages**. Each sequence owns a list of page ids (its **block table**), so memory is allocated one page at a time instead of reserving `max_len` slots per request. Attention must gather a sequence's pages back into order.

Write `paged_decode(q, k_pages, v_pages, block_table, lengths)`:

- `q`: `(B, H, d)`, one decode query per sequence.
- `k_pages`, `v_pages`: `(P, page_size, H, d)`, the shared pool.
- `block_table`: `(B, n_blocks)` int. Logical position $s$ of sequence $b$ lives in page `block_table[b, s // page_size]` at slot `s % page_size`. Entries for blocks a sequence does not use are `-1`.
- `lengths`: `(B,)` int, `1 <= lengths[b] <= n_blocks * page_size`.

Return `(out, k_seq)`:

- `k_seq`: `(B, n_blocks * page_size, H, d)`, each sequence's keys in logical order, with positions `>= lengths[b]` set to 0.
- `out`: `(B, H, d)`, softmax attention $\mathrm{softmax}(q k^\top / \sqrt d)\, v$ per head over positions `< lengths[b]`.

float32. Do not rely on what indexing with `-1` does: unused entries must not affect the result.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def paged_decode(q, k_pages, v_pages, block_table, lengths):
    """Return (out (B, H, d), k_seq (B, n_blocks * page_size, H, d))."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

def paged_decode(q, k_pages, v_pages, block_table, lengths):
    q, kp, vp = (jnp.asarray(a, jnp.float32) for a in (q, k_pages, v_pages))
    table = jnp.maximum(jnp.asarray(block_table), 0)        # -1 -> page 0: a valid index whose data we mask out
    B, nb = table.shape
    _, page, H, d = kp.shape
    L = nb * page

    # Fancy indexing gathers whole pages: (B, nb, page, H, d), then flatten pages into one time axis.
    k = kp[table].reshape(B, L, H, d)
    v = vp[table].reshape(B, L, H, d)
    valid = jnp.arange(L)[None, :] < jnp.asarray(lengths)[:, None]           # (B, L)
    k = jnp.where(valid[:, :, None, None], k, 0.0)
    v = jnp.where(valid[:, :, None, None], v, 0.0)

    s = jnp.einsum('bhd,blhd->bhl', q, k) / jnp.sqrt(d)
    s = jnp.where(valid[:, None, :], s, -jnp.inf)
    out = jnp.einsum('bhl,blhd->bhd', jax.nn.softmax(s, axis=-1), v)
    return out, k
'''

HINTS = [
    '`k_pages[block_table]` with a `(B, n_blocks)` integer array gathers pages into `(B, n_blocks, page_size, H, d)`. Reshape to `(B, n_blocks * page_size, H, d)` to get logical order.',
    'Replace `-1` entries with any valid page id (say 0) before gathering, then make the length mask do the work: zero the gathered keys and mask the scores at positions `>= lengths[b]`.',
    'With keys in logical order, decode attention is two einsums: `"bhd,blhd->bhl"` for scores and `"bhl,blhd->bhd"` for the output.',
]

_r = np.random.default_rng(30)
_f = lambda *s: _r.normal(size=s)
_kp1 = np.arange(8.0).reshape(4, 2, 1, 1)                  # page p holds keys 2p, 2p + 1
TESTS = [
    (np.zeros((1, 1, 1)), _kp1, _kp1, np.array([[2, 0]]), np.array([3])),
    (np.zeros((2, 1, 1)), _kp1, _kp1, np.array([[3, -1], [1, 2]]), np.array([2, 4])),
    (_f(2, 2, 4), _f(6, 3, 2, 4), _f(6, 3, 2, 4), np.array([[5, 0, 2], [1, 4, -1]]), np.array([7, 6])),
    (_f(3, 1, 8), _f(5, 4, 1, 8), _f(5, 4, 1, 8), np.array([[4, -1], [0, 3], [2, 1]]), np.array([1, 8, 5])),
    (_f(1, 4, 2), _f(3, 2, 4, 2), _f(3, 2, 4, 2), np.array([[2, 2, 0]]), np.array([6])),
    (3 * _f(4, 2, 4), _f(10, 2, 2, 4), _f(10, 2, 2, 4),
     np.array([[9, 8, -1, -1], [0, 1, 2, 3], [7, -1, -1, -1], [6, 5, 4, -1]]), np.array([3, 8, 1, 6])),
]

EXPLANATION = r"""
## What the block table buys
With contiguous allocation every request reserves `max_len` slots, most of which stay empty, so memory fragments and batch size suffers. Pages are allocated as a sequence grows and returned to the pool when it finishes; the waste is at most one partial page per sequence. Sharing becomes possible too: two sequences with the same prompt can point at the same pages (test 5 maps one page twice).

## The gather, and why `-1` is masked
`k_pages[table]` is a gather along the page axis. JAX normalises negative indices like Python and clamps out-of-range ones instead of raising, so `-1` would silently read the **last** page. Replacing it with page 0 and masking by length makes the result independent of whatever garbage those pages hold.

## Gather versus a real paged kernel
This version materialises `(B, L, H, d)`, which is fine for teaching but copies the cache every step. A PagedAttention kernel reads pages straight from the pool inside the attention loop, combining blocks with an online softmax, so no contiguous copy ever exists.

## Complexity
$O(B L H d)$ reads and FLOPs per step, the same as contiguous decode attention; the table lookup is negligible.

## Interview follow-ups
- How do you pick the page size? (Small pages waste less memory; large pages give longer contiguous reads and smaller tables.)
- How does copy-on-write work when a beam-search branch modifies a shared page?
"""
