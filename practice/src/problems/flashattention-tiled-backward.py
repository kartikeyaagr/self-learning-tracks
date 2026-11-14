import numpy as np

META = dict(title='FlashAttention backward: recompute tiles from the logsumexp', tags=['Attention', 'GPU Programming'],
            diff='Hard', runtime='numpy', lessons=['ieng:14.03', 'aieng:06.03'],
            related=['flashattention-with-online-softmax', 'blockwise-online-softmax', 'online-softmax', 'scaled-dot-product-attention'])

STATEMENT = r"""
FlashAttention never stores the $N\times N$ attention matrix. The forward pass keeps only the output $O$ and one logsumexp value per query row; the backward pass recomputes each tile of probabilities from them. Implement both passes block by block for a single head.

Write `flash_attention(Q, K, V, dO, block_q, block_k, causal)` returning `(O, L, dQ, dK, dV)`.

## Setup
`Q`, `K`, `V`, `dO` are `(N, d)`; $S = QK^\top/\sqrt d$, with $S_{ij} = -\infty$ for $j > i$ when `causal`. $P = \mathrm{softmax}(S)$ row-wise, $O = PV$, and $L_i = \log\sum_j e^{S_{ij}}$.

## Forward
Loop over query blocks of `block_q` rows and, inside, key blocks of `block_k` rows, keeping a running max, running sum and unnormalised output per row (the online softmax). Return $O$ and $L$ `(N,)`. Skip key blocks that are entirely masked.

## Backward (given the upstream gradient `dO`)
$D_i = \sum_c dO_{ic}O_{ic}$. For each tile: $P_{ij} = \exp(S_{ij} - L_i)$, then
$dV_j \mathrel{+}= P_{ij}\,dO_i$, $dP_{ij} = dO_i\cdot V_j$, $dS_{ij} = P_{ij}(dP_{ij} - D_i)$, $dQ_i \mathrel{+}= dS_{ij}K_j/\sqrt d$, $dK_j \mathrel{+}= dS_{ij}Q_i/\sqrt d$.

$N$ need not be a multiple of the block sizes.
"""

STARTER = '''import numpy as np

def flash_attention(Q, K, V, dO, block_q, block_k, causal):
    """Return (O, L, dQ, dK, dV)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _tile(Q, K, i0, i1, j0, j1, sc, causal):
    s = Q[i0:i1] @ K[j0:j1].T * sc
    if causal:
        s = np.where(np.arange(j0, j1)[None, :] > np.arange(i0, i1)[:, None], -np.inf, s)
    return s

def flash_attention(Q, K, V, dO, block_q, block_k, causal):
    Q, K, V, dO = (np.asarray(a, dtype=float) for a in (Q, K, V, dO))
    N, d = Q.shape
    sc = 1 / np.sqrt(d)
    O, L = np.zeros_like(Q), np.zeros(N)
    blocks = lambda i0, i1: [(j0, min(j0 + block_k, N)) for j0 in range(0, N, block_k) if not (causal and j0 > i1 - 1)]
    for i0 in range(0, N, block_q):
        i1 = min(i0 + block_q, N)
        m, l, acc = np.full(i1 - i0, -np.inf), np.zeros(i1 - i0), np.zeros((i1 - i0, d))
        for j0, j1 in blocks(i0, i1):
            s = _tile(Q, K, i0, i1, j0, j1, sc, causal)
            mn = np.maximum(m, s.max(1))
            p = np.exp(s - mn[:, None])
            a = np.exp(m - mn)
            l, acc, m = a * l + p.sum(1), a[:, None] * acc + p @ V[j0:j1], mn
        O[i0:i1], L[i0:i1] = acc / l[:, None], m + np.log(l)
    Dv = (dO * O).sum(1)
    dQ, dK, dV = np.zeros_like(Q), np.zeros_like(K), np.zeros_like(V)
    for i0 in range(0, N, block_q):
        i1 = min(i0 + block_q, N)
        for j0, j1 in blocks(i0, i1):
            P = np.exp(_tile(Q, K, i0, i1, j0, j1, sc, causal) - L[i0:i1, None])
            dV[j0:j1] += P.T @ dO[i0:i1]
            dS = P * (dO[i0:i1] @ V[j0:j1].T - Dv[i0:i1, None])
            dQ[i0:i1] += dS @ K[j0:j1] * sc
            dK[j0:j1] += dS.T @ Q[i0:i1] * sc
    return O, L, dQ, dK, dV
'''

HINTS = [
    'Online softmax per query block: when a new tile raises the running max from $m$ to $m\'$, rescale the running sum and output by $e^{m - m\'}$ before adding the tile\'s contribution. At the end $L = m + \\log l$.',
    'In the backward pass, $\\exp(S - L)$ gives the exact normalised probabilities for any tile, so no second reduction is needed. The softmax gradient $dS = P\\odot(dP - D)$ uses $D_i = \\sum_j P_{ij}dP_{ij} = dO_i\\cdot O_i$, which is cheap to precompute.',
    'With a causal mask, a key block whose first index exceeds the query block\'s last index is all $-\\infty$: skip it (and never take the max of an all-masked row).',
]

_r = np.random.default_rng(2022)
def _c(N, d, bq, bk, causal, scale=1.0):
    n = lambda: scale * _r.normal(size=(N, d))
    return (n(), n(), _r.normal(size=(N, d)), _r.normal(size=(N, d)), bq, bk, causal)
TESTS = [
    _c(4, 2, 2, 2, False),
    _c(5, 3, 2, 3, True),
    _c(8, 4, 4, 2, True),
    _c(7, 4, 3, 3, False, scale=3.0),
    _c(9, 2, 9, 1, True),
    _c(6, 8, 1, 4, False),
]

EXPLANATION = r"""
## Why tile at all
Standard attention writes $S$ and $P$ to GPU memory and reads them back, $O(N^2)$ memory traffic that dominates run time for long sequences even though the FLOPs are the same. FlashAttention keeps each tile in on-chip SRAM, so the forward pass reads $Q, K, V$ and writes $O$ and $L$, and never materialises $P$.

## Why store the logsumexp
For the backward pass you need $P$ again. Storing it costs $N^2$; storing $L$ costs $N$, and $P_{ij} = e^{S_{ij} - L_i}$ recomputes any tile exactly from $Q$, $K$ and $L$. The extra recomputation FLOPs are cheaper than the memory traffic they replace, which is why the backward pass is still faster than the standard one.

## The softmax gradient trick
$dS_{ij} = P_{ij}(dP_{ij} - \sum_k P_{ik}dP_{ik})$, and the row sum equals $dO_i\cdot O_i$, so it is computed once per row up front instead of needing a full row of $dP$.

## Common bugs
- Forgetting to rescale the running output when the max changes.
- Forgetting the $1/\sqrt d$ factor on $dQ$ and $dK$ (it is inside $S$).
- Recomputing $P$ with a per-tile max instead of the global $L$, which gives unnormalised tiles.

## Interview follow-ups
- How does the work split across thread blocks differ between the forward and backward kernels, and where are atomics needed?
- What changes for multi-query or grouped-query attention?
"""
