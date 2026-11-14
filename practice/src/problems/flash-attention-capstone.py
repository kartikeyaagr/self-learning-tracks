import numpy as np

META = dict(title='Capstone: tiled causal FlashAttention forward with tile skipping', tags=['Attention', 'GPU Programming'], diff='Hard', runtime='numpy',
            related=['masked-safe-softmax', 'online-softmax', 'merge-partial-attention', 'flashattention-with-online-softmax', 'causal-tile-skipping'])

STATEMENT = r"""
Write `flash_forward(Q, K, V, Bq, Bk, causal)` for one head, `Q (n, d)`, `K, V (m, d)`, with query $i$ at position $i + (m - n)$ (the queries are the last $n$ positions) and key $j$ at position $j$.

Loop over query blocks of `Bq` rows (outer) and key blocks of `Bk` columns (inner), like FlashAttention-2. For each query block keep running statistics per row: max $m_i$ (start $-\infty$), denominator $\ell_i$ (start 0) and unnormalised output $\tilde O_i$ (start 0).

- With `causal`, classify each tile from its corner positions: skip it if its first key is after its last query; apply the causal mask only if it is partially visible.
- For a processed tile: $S = QK^\top/\sqrt d$ (masked entries $-\infty$); $m' = \max(m, \text{rowmax}(S))$; $P = e^{S - m'}$ (0 where masked); $\ell \leftarrow e^{m - m'}\ell + \text{rowsum}(P)$; $\tilde O \leftarrow e^{m - m'}\tilde O + PV$; $m \leftarrow m'$. Treat $e^{-\infty - (-\infty)}$ as 0 (rows that have seen nothing yet).
- After the inner loop: $O = \tilde O / \ell$ and $\mathrm{lse} = m + \log \ell$; rows that saw no key get $O = 0$, lse $= -\infty$.

Return `{"O", "lse", "processed": tiles computed, "skipped": tiles skipped, "masked": tiles that needed a mask}`.
"""

STARTER = '''import numpy as np

def flash_forward(Q, K, V, Bq, Bk, causal):
    """Tiled attention with an online softmax and causal tile skipping."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def flash_forward(Q, K, V, Bq, Bk, causal):
    Q, K, V = (np.asarray(a, float) for a in (Q, K, V))
    n, d = Q.shape; m_k = len(K); off = m_k - n
    O, lse = np.zeros((n, V.shape[1])), np.full(n, -np.inf)
    done = skipped = masked = 0
    for q0 in range(0, n, Bq):
        q1 = min(q0 + Bq, n)
        qpos = np.arange(q0, q1) + off
        mrow, l, acc = np.full(q1 - q0, -np.inf), np.zeros(q1 - q0), np.zeros((q1 - q0, V.shape[1]))
        for k0 in range(0, m_k, Bk):
            k1 = min(k0 + Bk, m_k)
            if causal and k0 > qpos[-1]:
                skipped += 1; continue
            S = Q[q0:q1] @ K[k0:k1].T / np.sqrt(d)
            if causal and k1 - 1 > qpos[0]:
                S = np.where(np.arange(k0, k1)[None, :] <= qpos[:, None], S, -np.inf); masked += 1
            done += 1
            mnew = np.maximum(mrow, S.max(1))
            safe = np.where(np.isneginf(mnew), 0.0, mnew)
            P = np.where(np.isneginf(S), 0.0, np.exp(S - safe[:, None]))
            scale = np.where(np.isneginf(mrow), 0.0, np.exp(mrow - safe))
            l = scale * l + P.sum(1)
            acc = scale[:, None] * acc + P @ V[k0:k1]
            mrow = mnew
        seen = l > 0
        O[q0:q1][seen] = acc[seen] / l[seen][:, None]
        lse[q0:q1][seen] = mrow[seen] + np.log(l[seen])
    return {"O": O, "lse": lse, "processed": done, "skipped": skipped, "masked": masked}
'''

HINTS = [
    'Rescale the running denominator and output by exp(m_old - m_new) every time the running max grows.',
    'Use a finite stand-in for a -inf max when exponentiating, then zero the masked probabilities explicitly.',
    'Check your output against naive softmax attention with a causal mask; they should agree to round-off.',
]

def _qkv(seed, n, m, d):
    r = np.random.default_rng(seed)
    return r.normal(size=(n, d)), r.normal(size=(m, d)), r.normal(size=(m, d))

TESTS = [
    _qkv(1, 8, 8, 4) + (2, 2, True),
    _qkv(2, 8, 8, 4) + (4, 2, False),
    _qkv(3, 7, 10, 3) + (3, 4, True),
    _qkv(4, 1, 13, 4) + (1, 4, True),
    _qkv(5, 12, 12, 8) + (5, 3, True),
]

EXPLANATION = r"""
## What the kernel does differently
Naive attention builds the $n \times m$ score matrix, normalises it, and multiplies by $V$. This version never holds more than one $B_q \times B_k$ tile: the online softmax rescales the running result whenever a new tile raises the row maximum, so the final answer is exact. The log-sum-exp it returns is what the backward pass needs to recompute probabilities tile by tile.

## Where the speed comes from on a GPU
Each tile lives in shared memory or registers; HBM sees each input read a bounded number of times and the output written once. Causal skipping removes about half the tiles, and masks are applied only on the diagonal band.

## Writing it in a notebook
The same loop in Triton maps the outer query loop to program instances, the tiles to `tl.load` of blocks, and the rescaling to a few vector operations. Comparing against `torch.nn.functional.scaled_dot_product_attention` on random inputs is the standard correctness test, followed by timing against it.

## Interview follow-ups
- Why does processing the outer loop over queries (FA-2) parallelise better than over keys (FA-1)?
"""
