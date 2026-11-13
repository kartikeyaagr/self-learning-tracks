import numpy as np

META = dict(title='RoPE for a ragged decode batch with a cos/sin cache', tags=['Transformers', 'Inference', 'Attention'],
            diff='Medium', runtime='torch', lessons=['aieng:04.01', 'ieng:13.06'],
            related=['rotary-position-embedding-rope', 'autoregressive-decoding-with-kv-cache', 'multi-head-latent-attention-mla'])

STATEMENT = r"""
In a serving engine, the sequences in one decode batch sit at **different** positions: one request is on token 17, another on token 950. Engines precompute a cos/sin table once and gather rows from it per token, instead of recomputing angles every step.

Write `rope_decode(q, k, positions, max_pos, base=10000.0, scale=1.0)` in PyTorch.

## Shapes
- `q`: `(B, H, T, d)` and `k`: `(B, H_kv, T, d)`, float64, $d$ even. $H$ and $H_{kv}$ may differ (grouped-query attention).
- `positions`: int64 `(B, T)`: the position of each new token in its own sequence. All values are in $[0, \text{max\_pos})$.

## Computation
1. Build the tables once: $\theta_i = \text{base}^{-2i/d}$ for $i = 0, \dots, d/2 - 1$, and for every $m \in [0, \text{max\_pos})$ the angle $(m / \text{scale})\,\theta_i$, giving `cos` and `sin` of shape `(max_pos, d/2)`. (`scale > 1` is linear position interpolation, used to stretch a model to longer contexts.)
2. Gather the rows for `positions`, giving `(B, T, d/2)`, and broadcast them over the head axis of both `q` and `k`.
3. Rotate with the **half** layout (Llama, GPT-NeoX): pair $i$ is features $(i, i + d/2)$:
$$(a, b) \mapsto (a\cos\phi - b\sin\phi,\;\; a\sin\phi + b\cos\phi).$$

## Output
A tuple `(q_rot, k_rot)` with the same shapes as `q` and `k`. No Python loops over batch, heads or tokens.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def rope_decode(q, k, positions, max_pos, base=10000.0, scale=1.0):
    """q: (B,H,T,d), k: (B,Hkv,T,d), positions: (B,T) int64. Return (q_rot, k_rot)."""
    raise NotImplementedError
'''

SOLUTION = '''import torch

def rope_decode(q, k, positions, max_pos, base=10000.0, scale=1.0):
    d = q.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)          # (d/2,)
    ang = (torch.arange(max_pos, dtype=torch.float64) / scale)[:, None] * theta  # (max_pos, d/2)
    cos_tab, sin_tab = ang.cos(), ang.sin()          # built once per model in a real engine

    cos = cos_tab[positions].unsqueeze(1).to(q.dtype)  # (B, 1, T, d/2): broadcasts over any head count
    sin = sin_tab[positions].unsqueeze(1).to(q.dtype)

    def rot(x):
        a, b = x[..., : d // 2], x[..., d // 2:]
        return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)

    return rot(q), rot(k)
'''

HINTS = [
    'Build `theta` of length `d/2`, then an outer product with `arange(max_pos) / scale` to get the angle table `(max_pos, d/2)`.',
    'Indexing a table with a `(B, T)` integer tensor, `cos_tab[positions]`, gives `(B, T, d/2)`. Insert a head axis with `unsqueeze(1)` so it broadcasts against `(B, H, T, d/2)` and `(B, Hkv, T, d/2)`.',
    'Half layout: `a = x[..., :d//2]`, `b = x[..., d//2:]`, output `cat([a*cos - b*sin, a*sin + b*cos], -1)`.',
]

_r = np.random.default_rng(5)
TESTS = [
    (np.array([[[[1.0, 0.0]]]]), np.array([[[[0.0, 1.0]]]]), np.array([[1]]), 4),
    (np.ones((2, 1, 1, 4)), np.ones((2, 1, 1, 4)), np.array([[0], [3]]), 8, 100.0),
    (_r.normal(size=(3, 4, 1, 8)), _r.normal(size=(3, 2, 1, 8)), np.array([[17], [0], [950]]), 1024),
    (_r.normal(size=(2, 2, 3, 6)), _r.normal(size=(2, 1, 3, 6)), np.array([[4, 5, 6], [0, 1, 2]]), 16),
    (_r.normal(size=(1, 8, 2, 16)), _r.normal(size=(1, 8, 2, 16)), np.array([[2047, 2048]]), 4096, 10000.0, 4.0),
    (_r.normal(size=(2, 3, 1, 4)), _r.normal(size=(2, 3, 1, 4)), np.array([[0], [0]]), 1),
    (_r.normal(size=(4, 2, 2, 8)), _r.normal(size=(4, 1, 2, 8)), _r.integers(0, 500, size=(4, 2)), 500, 500000.0),
]

EXPLANATION = r"""
## Why it works
RoPE's rotation for a token depends only on that token's own position, so a ragged batch needs nothing more than a per-token gather from a table indexed by position. The rotation is applied identically to every head, so a `(B, 1, T, d/2)` table broadcasts over the query heads and the (fewer) KV heads alike. Because $\langle R_m q, R_n k\rangle$ depends only on $m - n$, rotating keys once when they enter the cache and queries at use time gives the right relative scores.

## Performance
The table costs $\text{max\_pos} \cdot d$ numbers and is built once. Per step, the work is a gather plus a few elementwise ops: purely memory-bound, which is why engines fuse RoPE into the QKV projection or the attention kernel instead of running it as a separate pass.

## Common bugs
- Using `arange(T)` positions in decode: every sequence then thinks it is at position 0.
- Rotating the cached keys again each step (they were rotated when written).
- Mixing layouts: a checkpoint trained with the interleaved layout gives garbage with the half layout, with no error.
- Building the table in float16: angles like $950 \theta_0$ lose precision, so build it in float32 or float64.

## Interview follow-ups
- Compare linear position interpolation (`scale`) with NTK-aware scaling or YaRN, which change `base` or treat frequencies differently. Why does interpolation hurt short-context quality?
- In speculative decoding, the draft tokens get positions $P, P+1, \dots$ before they are accepted. What has to happen to their rotated keys if some are rejected?
"""
