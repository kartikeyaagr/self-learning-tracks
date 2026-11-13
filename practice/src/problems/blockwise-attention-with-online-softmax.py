import numpy as np

META = dict(title='Blockwise attention with online softmax in lax.scan', tags=['Attention', 'JAX', 'Inference'],
            diff='Hard', runtime='jax', lessons=['ieng:14.02', 'aieng:06.03'],
            related=['scaled-dot-product-attention', 'batched-attention-with-vmap', 'paged-kv-cache-gather'], tol=[1e-4, 1e-5])

STATEMENT = r"""
FlashAttention never builds the full $T \times S$ score matrix. It walks over key blocks, keeping for each query row a running maximum $m$, a running normaliser $\ell$ and an unnormalised output $a$, and rescales them whenever the maximum grows. Write that algorithm with `jax.lax.scan` over key blocks.

Write `blockwise_attention(q, k, v, block, causal)`:

- `q`: `(T, d)`; `k`, `v`: `(S, d)`; float32. Scores are $s_{ij} = q_i \cdot k_j / \sqrt d$.
- `block`: Python int, the number of keys per block. `S` need not be a multiple of `block`: pad the keys and values up to the next multiple and mask the padding.
- `causal`: Python bool. If true, query $i$ may see key $j$ only when $j \le i + (S - T)$ (queries are the **last** $T$ positions of the sequence, as in decoding with a cache). `S >= T` when causal.

For each block, with $\tilde s$ the masked scores of that block:

$$m' = \max(m, \max_j \tilde s_j), \qquad \ell \leftarrow e^{m - m'} \ell + \sum_j e^{\tilde s_j - m'}, \qquad a \leftarrow e^{m - m'} a + \sum_j e^{\tilde s_j - m'} v_j, \qquad m \leftarrow m'$$

Return `(out, lse)`: `out = a / ℓ` with shape `(T, d)` and `lse = m + log ℓ` with shape `(T,)`, the log-sum-exp of each row's visible scores. Every row sees at least one key, but **a whole block can be masked for some rows**, so the update must not produce `nan` when $m$ and every $\tilde s_j$ are $-\infty$.

This problem runs in **Colab** (JAX isn't available in the browser). Use the default CPU runtime; JAX computes in float32 here. The notebook includes the same tests.
"""

STARTER = '''import jax
import jax.numpy as jnp

def blockwise_attention(q, k, v, block, causal):
    """Return (out (T, d), lse (T,)) computed one key block at a time."""
    raise NotImplementedError
'''

SOLUTION = '''import jax
import jax.numpy as jnp

NEG = -1e30   # finite stand-in for -inf: exp(NEG - NEG) = 1 instead of nan, and exp(NEG - real) = 0

def blockwise_attention(q, k, v, block, causal):
    q, k, v = (jnp.asarray(a, jnp.float32) for a in (q, k, v))
    T, d = q.shape
    S = k.shape[0]
    nb = -(-S // block)                                     # ceil(S / block)
    pad = nb * block - S
    kb = jnp.pad(k, ((0, pad), (0, 0))).reshape(nb, block, d)
    vb = jnp.pad(v, ((0, pad), (0, 0))).reshape(nb, block, d)
    starts = jnp.arange(nb) * block
    rows = jnp.arange(T)[:, None]

    def body(carry, blk):
        m, l, a = carry
        kj, vj, j0 = blk
        cols = j0 + jnp.arange(block)[None, :]              # global key index, (1, block)
        visible = cols < S                                  # padding
        if causal:
            visible = visible & (cols <= rows + (S - T))
        s = jnp.where(visible, q @ kj.T / jnp.sqrt(d), NEG)  # (T, block)
        m_new = jnp.maximum(m, s.max(axis=1))
        alpha = jnp.exp(m - m_new)                          # rescales the old partial sums
        p = jnp.where(visible, jnp.exp(s - m_new[:, None]), 0.0)   # masked keys contribute exactly 0
        return (m_new, alpha * l + p.sum(axis=1), alpha[:, None] * a + p @ vj), None

    init = (jnp.full((T,), NEG), jnp.zeros(T), jnp.zeros((T, d)))
    (m, l, a), _ = jax.lax.scan(body, init, (kb, vb, starts))
    return a / l[:, None], m + jnp.log(l)
'''

HINTS = [
    'Pad `k` and `v` to `nb * block` rows and reshape to `(nb, block, d)`. Scan over `(k_blocks, v_blocks, block_start_indices)` so the body knows the global key index of each column.',
    'Carry `(m, l, a)` with shapes `(T,)`, `(T,)`, `(T, d)`. Rescale the old `l` and `a` by `exp(m - m_new)` before adding the block\'s contribution.',
    'With true `-inf`, a fully masked block gives `exp(-inf - (-inf)) = nan`. Use a large finite negative like `-1e30` for masked scores and the initial `m`, and zero masked probabilities explicitly with `jnp.where`.',
]

_r = np.random.default_rng(24)
_f = lambda *s: _r.normal(size=s)
TESTS = [
    (np.array([[1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]),
     2, False),
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 2.0], [3.0, 4.0]]), 1, True),
    (_f(5, 8), _f(12, 8), _f(12, 8), 4, False),
    (_f(5, 8), _f(13, 8), _f(13, 8), 4, False),
    (_f(6, 4), _f(6, 4), _f(6, 4), 2, True),
    (_f(3, 4), _f(10, 4), _f(10, 4), 3, True),
    (8 * _f(4, 4), 8 * _f(9, 4), _f(9, 4), 16, True),
    (_f(7, 16), _f(20, 16), _f(20, 16), 1, True),
]

EXPLANATION = r"""
## Why the rescaling is exact
After any prefix of blocks, $\ell = \sum_j e^{s_j - m}$ and $a = \sum_j e^{s_j - m} v_j$ over the keys seen so far, with $m$ their running maximum. When the maximum moves to $m'$, multiplying both by $e^{m - m'}$ re-expresses them relative to $m'$; then the new block's terms are added. At the end $a / \ell$ is exactly the softmax-weighted sum, and $m + \log \ell$ is the log-sum-exp. Block size does not change the answer (tests 3, 4 and 8), only the memory footprint.

## Memory and speed
The full score matrix needs $O(TS)$ memory; this needs $O(T \cdot \text{block})$ for one block plus $O(Td)$ state. On a GPU the real gain is bandwidth: scores stay in on-chip SRAM instead of round-tripping through HBM. Here, `scan` shows the algorithm; a Pallas or Triton kernel gets the speed.

## The `nan` trap
With causal masking, early query rows see none of the later blocks (test 6). If masked scores are $-\infty$ and $m$ is still $-\infty$, then $m - m' = -\infty - (-\infty) = $ `nan`. A finite sentinel like $-10^{30}$ makes $e^{m - m'} = 1$ in that case, and the explicit `where` keeps masked probabilities at exactly 0 (otherwise $e^{-10^{30} - (-10^{30})} = 1$ would count masked keys).

## Common bugs
- Forgetting to rescale `a` along with `l`.
- Masking only the causal pattern and not the padding added to the last block (test 4).
- Aligning the causal mask top-left ($j \le i$) when $S > T$: decode queries would not see the prefix.

## Interview follow-ups
- Why is `lse` saved for the backward pass of FlashAttention? (It lets backward recompute the probabilities block by block without storing them.)
- How does this extend to ring attention across devices? (Each device holds a key block; blocks rotate with `ppermute` and the same $(m, \ell, a)$ merge is applied.)
"""
