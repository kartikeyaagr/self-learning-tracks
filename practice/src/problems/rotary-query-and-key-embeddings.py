import numpy as np

META = dict(title='Partial rotary embeddings on queries and keys', tags=['Transformers', 'Attention'], diff='Medium',
            runtime='torch', lessons=['aieng:04.01', 'ieng:13.06'],
            related=['rotary-position-embedding-rope', 'rotary-position-embeddings-for-decoding', 'causal-grouped-query-attention'])

STATEMENT = r"""
RoPE rotates pairs of query and key features by angles proportional to position, so the dot product $q_m \cdot k_n$ depends only on $m - n$. Several model families rotate only the **first** `rotary_dim` features of each head and pass the rest through unchanged ("partial rotary").

Write `rope_qk(q, k, q_pos, k_pos, rotary_dim, base=10000.0)` in PyTorch.

## Inputs
- `q`: `float64` `(B, H, T_q, D)`; `k`: `float64` `(B, H_kv, T_k, D)`. $H_{kv}$ divides $H$.
- `q_pos`: `int64` `(T_q,)`, `k_pos`: `int64` `(T_k,)`: the absolute positions of the query and key tokens (shared by the batch). They can start anywhere, e.g. when queries are new tokens after a cache.
- `rotary_dim`: even int, $2 \le$ `rotary_dim` $\le D$.

## Rotation (applied to `q` with `q_pos` and to `k` with `k_pos`)
Let $r$ = `rotary_dim`. Use the **half-split** layout on the first $r$ features: for $i = 0, \dots, r/2 - 1$, features $i$ and $i + r/2$ form a pair, rotated by angle $p\,\theta_i$ with $\theta_i = \text{base}^{-2i/r}$ (note: $r$, not $D$):

$$\begin{aligned} u' &= u\cos(p\theta_i) - v\sin(p\theta_i) \\ v' &= u\sin(p\theta_i) + v\cos(p\theta_i) \end{aligned} \qquad (u, v) = (x_i,\ x_{i + r/2})$$

Features $r, \dots, D-1$ are copied unchanged.

## Output
A tuple `(q_rot, k_rot, scores)`:
- `q_rot`, `k_rot`: same shapes as `q` and `k`.
- `scores`: `(B, H, T_q, T_k)`, $\;S = q_\text{rot}\, k_\text{rot}^\top / \sqrt{D}$, where query head $h$ uses key head $\lfloor h / (H / H_{kv}) \rfloor$.

No masking, no softmax. Do not modify the inputs.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import math
import torch

def rope_qk(q, k, q_pos, k_pos, rotary_dim, base=10000.0):
    """Return (q_rot, k_rot, scores) with partial half-split RoPE on the first rotary_dim features."""
    D = q.shape[-1]
    scores = q @ k.transpose(-1, -2) / math.sqrt(D)      # TODO: no rotation, no GQA head sharing
    return q, k, scores
'''

SOLUTION = '''import math
import torch

def _rotate(x, pos, r, base):
    half = r // 2
    theta = base ** (-torch.arange(half, dtype=torch.float64) * 2 / r)       # (r/2,)
    ang = pos.to(torch.float64)[:, None] * theta[None, :]                    # (T, r/2)
    cos, sin = torch.cos(ang), torch.sin(ang)                                # broadcast over (B, H)
    u, v, rest = x[..., :half], x[..., half:r], x[..., r:]
    return torch.cat([u * cos - v * sin, u * sin + v * cos, rest], dim=-1)

def rope_qk(q, k, q_pos, k_pos, rotary_dim, base=10000.0):
    q_rot = _rotate(q, q_pos, rotary_dim, base)
    k_rot = _rotate(k, k_pos, rotary_dim, base)
    B, H, Tq, D = q.shape
    Hkv = k.shape[1]
    # group view instead of repeating K: (B, Hkv, H/Hkv, Tq, D) against (B, Hkv, 1, Tk, D)
    qg = q_rot.reshape(B, Hkv, H // Hkv, Tq, D)
    scores = qg @ k_rot[:, :, None].transpose(-1, -2) / math.sqrt(D)
    return q_rot, k_rot, scores.reshape(B, H, Tq, -1)
'''

HINTS = [
    'Build the angle table once: `theta = base ** (-arange(r/2) * 2 / r)` and `ang = pos[:, None] * theta[None, :]`, shape `(T, r/2)`. It broadcasts against `(B, H, T, r/2)`.',
    'Slice `u = x[..., :r/2]`, `v = x[..., r/2:r]`, `rest = x[..., r:]`, rotate the first two, then `torch.cat` all three back in order.',
    'For GQA, reshape the rotated queries to `(B, H_kv, H/H_kv, T_q, D)` so consecutive query heads share a key head, then matmul against `k_rot[:, :, None]`.',
]

_r = np.random.default_rng(21)
_q = _r.normal(size=(1, 4, 3, 8))
_k = _r.normal(size=(1, 2, 5, 8))
TESTS = [
    (np.array([[[[1.0, 0.0, 0.0, 0.0]]]]), np.array([[[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]]),
     np.array([1]), np.array([0, 1]), 4, 1.0),
    (np.array([[[[1.0, 2.0, 3.0, 4.0]]]]), np.array([[[[1.0, 2.0, 3.0, 4.0]]]]), np.array([5]), np.array([5]), 2, 10000.0),
    (_q, _k, np.array([4, 5, 6]), np.array([0, 1, 2, 3, 4]), 8),
    (_q, _k, np.array([104, 105, 106]), np.array([100, 101, 102, 103, 104]), 8),     # same scores as test 3
    (_q, _k, np.array([4, 5, 6]), np.array([0, 1, 2, 3, 4]), 4),
    (_r.normal(size=(2, 2, 4, 6)), _r.normal(size=(2, 1, 4, 6)), np.arange(4), np.arange(4), 6, 500.0),
    (_r.normal(size=(1, 6, 2, 16)), _r.normal(size=(1, 3, 7, 16)), np.array([7000, 7001]), np.arange(6995, 7002), 10),
]

EXPLANATION = r"""
## Why it works
Each pair is a complex number $z = u + iv$, and the rotation is $z \mapsto z\,e^{ip\theta_i}$. The real part of $z_q \bar z_k$ is the pair's contribution to the dot product, and it becomes $\mathrm{Re}(z_q \bar z_k\, e^{i(m-n)\theta_i})$: only the offset $m - n$ appears. Tests 3 and 4 shift every position by 100 and get identical scores (up to rounding), while `q_rot` and `k_rot` themselves change.

## Partial rotary
The unrotated features carry position-free content, so some heads can match tokens regardless of distance. Note the frequencies use $r$ in the exponent: the slowest pair still has period about $2\pi\,\text{base}$, whatever $r$ is.

## Complexity
Rotation is $O(B H T D)$ elementwise work; the scores are $O(B H T_q T_k D)$. Engines precompute the cos/sin table for all positions once and fuse the rotation into the QKV projection kernel.

## Common bugs
- Using $\theta_i = \text{base}^{-2i/D}$ with partial rotary: the frequencies come out wrong for every pair but the first.
- Mixing layouts: the half-split pairs $(i, i + r/2)$, not $(2i, 2i+1)$. Weights trained with one layout give garbage with the other.
- Repeating K with `repeat` instead of `repeat_interleave` (or the group reshape): query head $h$ must use key head $h // (H/H_{kv})$, not $h \bmod H_{kv}$.

## Interview follow-ups
- Why apply RoPE after the K projection but before writing to the KV cache, and what breaks if you cache unrotated keys?
- How do position interpolation and NTK-aware scaling change $\theta_i$ to extend context length?
"""
