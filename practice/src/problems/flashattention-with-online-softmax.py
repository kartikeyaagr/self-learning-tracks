import numpy as np

META = dict(title='Tiled attention with an online softmax (FlashAttention forward)', tags=['Attention', 'GPU Programming', 'Inference'],
            diff='Hard', runtime='torch', lessons=['ieng:14.02', 'ieng:14.03'],
            related=['scaled-dot-product-attention-3', 'softmax-numerical-stability-and-log-sum-exp', 'scaled-dot-product-attention'])

STATEMENT = r"""
FlashAttention never builds the full $T \times S$ score matrix. It walks over key/value **blocks**, keeping for each query row a running max $m$, a running normaliser $\ell$ and an unnormalised output accumulator. When a new block raises the max, the old state is rescaled. Implement that forward pass in PyTorch.

Write `flash_attention(q, k, v, block_q, block_k, causal=False)`.

## Shapes
- `q`: `(B, H, T, d)`, `k`: `(B, H, S, d)`, `v`: `(B, H, S, d_v)`, float64. Scale $s = 1/\sqrt{d}$.
- `block_q`, `block_k`: tile sizes, positive ints that need not divide $T$ or $S$ (the last tile is shorter).
- `causal`: only used with $T = S$; query $i$ may attend to keys $j \le i$.

## Algorithm (per query tile $Q_b$, rows $r$)
Start with $m_r = -\infty$, $\ell_r = 0$, $O_r = 0$. For each key tile $(K_c, V_c)$ in order:
1. $S = s\,Q_b K_c^\top$ (apply the causal mask inside the tile with $-\infty$).
2. $m^{\text{new}}_r = \max(m_r, \max_j S_{rj})$, $\;P = \exp(S - m^{\text{new}})$, $\;\alpha_r = \exp(m_r - m^{\text{new}}_r)$.
3. $\ell_r \leftarrow \alpha_r \ell_r + \sum_j P_{rj}$, $\;O_r \leftarrow \alpha_r O_r + P V_c$, $\;m_r \leftarrow m^{\text{new}}_r$.

At the end, $O_r / \ell_r$ is the attention output and $\mathrm{lse}_r = m_r + \log \ell_r$ is the log-sum-exp of row $r$'s scaled scores (what the backward pass stores). With `causal`, skip key tiles that lie entirely in the future of the query tile.

## Output
A tuple `(O, lse)`: `O` is `(B, H, T, d_v)`, `lse` is `(B, H, T)`. Loop over tiles in Python, but vectorise within a tile (over batch, heads and rows). Never allocate a `(T, S)` score matrix.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def flash_attention(q, k, v, block_q, block_k, causal=False):
    """Return (O (B,H,T,dv), lse (B,H,T)) without materialising the full score matrix."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import torch

def flash_attention(q, k, v, block_q, block_k, causal=False):
    B, H, T, d = q.shape
    S = k.shape[2]
    scale = 1.0 / math.sqrt(d)
    out = torch.empty(B, H, T, v.shape[-1], dtype=q.dtype)
    lse = torch.empty(B, H, T, dtype=q.dtype)

    for q0 in range(0, T, block_q):
        qb = q[:, :, q0:q0 + block_q] * scale                   # fold the scale into Q once
        nq = qb.shape[2]
        m = torch.full((B, H, nq, 1), float('-inf'), dtype=q.dtype)
        l = torch.zeros(B, H, nq, 1, dtype=q.dtype)
        acc = torch.zeros(B, H, nq, v.shape[-1], dtype=q.dtype)
        k_end = min(S, q0 + nq) if causal else S                 # later key tiles are fully in the future
        for k0 in range(0, k_end, block_k):
            s = qb @ k[:, :, k0:k0 + block_k].transpose(-2, -1)  # (B, H, nq, nk): one tile only
            if causal:
                qi = torch.arange(q0, q0 + nq)[:, None]
                kj = torch.arange(k0, k0 + s.shape[-1])[None, :]
                s = s.masked_fill(kj > qi, float('-inf'))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))   # finite: key 0 is visible to every row
            p = torch.exp(s - m_new)
            alpha = torch.exp(m - m_new)                         # 0 on the first tile (m = -inf)
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + p @ v[:, :, k0:k0 + block_k]
            m = m_new
        out[:, :, q0:q0 + nq] = acc / l
        lse[:, :, q0:q0 + nq] = (m + torch.log(l)).squeeze(-1)
    return out, lse
'''

HINTS = [
    'Keep three tensors per query tile: `m` and `l` of shape `(B, H, nq, 1)` and `acc` of shape `(B, H, nq, dv)`. Start `m` at `-inf`.',
    'For each key tile: `m_new = max(m, rowmax(S))`, `alpha = exp(m - m_new)`, then `l = alpha*l + rowsum(exp(S - m_new))` and `acc = alpha*acc + exp(S - m_new) @ V_tile`.',
    'Divide `acc / l` only once at the end, and `lse = m + log(l)`. For causal, the key loop can stop at `min(S, q0 + nq)`, and masking inside diagonal tiles uses absolute indices `q0 + r` and `k0 + c`.',
]

_r = np.random.default_rng(7)
_n = lambda *s: _r.normal(size=s)
TESTS = [
    (np.array([[[[1.0, 0.0], [0.0, 1.0]]]]), np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]]),
     np.array([[[[1.0], [2.0], [3.0]]]]), 1, 2),
    (np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]]), np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]]),
     np.array([[[[1.0], [2.0], [3.0]]]]), 2, 2, True),
    (_n(2, 2, 8, 4), _n(2, 2, 8, 4), _n(2, 2, 8, 4), 4, 4),
    (_n(1, 3, 7, 8), _n(1, 3, 11, 8), _n(1, 3, 11, 5), 3, 4),
    (_n(2, 1, 9, 4), _n(2, 1, 9, 4), _n(2, 1, 9, 4), 4, 3, True),
    (_n(1, 2, 6, 4) * 15, _n(1, 2, 6, 4) * 15, _n(1, 2, 6, 4), 2, 2, True),
    (_n(1, 1, 5, 2), _n(1, 1, 5, 2), _n(1, 1, 5, 3), 8, 8),
    (_n(1, 2, 10, 6), _n(1, 2, 10, 6), _n(1, 2, 10, 6), 1, 1, True),
]

EXPLANATION = r"""
## Why it works
Softmax is $\exp(s_j - M)/\sum_k \exp(s_k - M)$ for **any** shift $M$; the max is used only to avoid overflow. If the state so far was computed with shift $m$ and a new tile raises the max to $m'$, multiplying the old sum and the old accumulator by $\alpha = e^{m - m'}$ converts them to shift $m'$ exactly. After the last tile, $\ell$ is $\sum_j e^{s_j - m}$ and the accumulator is $\sum_j e^{s_j - m} v_j$, so their ratio is the exact softmax-weighted output: tiling changes the order of the arithmetic, not the result.

## Why it is fast on a GPU
Standard attention writes the $T \times S$ scores and probabilities to HBM and reads them back: $O(TS)$ memory traffic. FlashAttention keeps each tile in on-chip SRAM, so HBM traffic drops to reading Q, K, V and writing O, about $O(Td + Sd)$ per pass over K/V tiles. FLOPs are unchanged ($\approx 4TSd$); the speedup comes entirely from moving fewer bytes. Storing only `lse` (one number per row) lets the backward pass recompute probabilities tile by tile instead of saving them.

## Common bugs
- Normalising every tile (`acc / l` inside the loop) and then rescaling with $\alpha$: the normaliser is applied twice.
- Rescaling `l` but not `acc` (or the reverse).
- Masking with relative tile indices instead of absolute positions in causal tiles.
- In fp16/bf16 kernels, keeping `m`, `l` and `acc` in low precision; real kernels keep them in fp32.

## Interview follow-ups
- For decode ($T = 1$), a single query tile has too little parallelism. How does split-K ("flash decoding") parallelise over the key axis and merge partial results using each split's `lse`?
- What is the arithmetic intensity of attention with and without tiling, and at what sequence length does it become compute-bound?
"""
