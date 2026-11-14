import numpy as np

META = dict(title='RoPE attention scores with PI, NTK and YaRN scaling', tags=['Transformers', 'Attention'], diff='Hard',
            runtime='numpy', lessons=['aieng:04.01', 'ieng:20.02'],
            related=['rotary-position-embedding-rope', 'rotary-position-embedding', 'rotary-query-and-key-embeddings'])

STATEMENT = r"""
Rotary embeddings rotate each pair of query/key dimensions by an angle proportional to the position. To run a model beyond the context length $L$ it was trained on, the rotation frequencies are rescaled. Implement RoPE attention scores with four frequency rules.

Write `rope_scores(q, k, pos_q, pos_k, base, method, scale=1.0, orig_ctx=None)` returning the `(Tq, Tk)` score matrix.

## RoPE (interleaved pairs)
For head width $d$ (even), pair $i \in [0, d/2)$ holds dims $(2i, 2i+1)$ and has frequency $\theta_i = \text{base}^{-2i/d}$. A vector at position $m$ has pair $i$ rotated by angle $m\,\theta'_i$: $(a, b) \mapsto (a\cos - b\sin,\ a\sin + b\cos)$. Scores are $\mu^2\,\langle R q, R k\rangle/\sqrt{d}$.

## Frequency rules (`scale` $= s$)
| `method` | $\theta'_i$ | $\mu$ |
|---|---|---|
| `'none'` | $\theta_i$ | 1 |
| `'linear'` (position interpolation) | $\theta_i / s$ | 1 |
| `'ntk'` | computed from base $\cdot s^{d/(d-2)}$ | 1 |
| `'yarn'` | $(1-\gamma_i)\,\theta_i/s + \gamma_i\,\theta_i$ | $0.1\ln s + 1$ |

For YaRN, the number of full rotations pair $i$ makes over the original context is $r_i = L\theta_i/(2\pi)$ with $L$ = `orig_ctx`, and $\gamma_i = \mathrm{clip}\big((r_i - \alpha)/(\beta - \alpha), 0, 1\big)$ with $\alpha = 1$, $\beta = 32$. High-frequency pairs ($r_i > 32$) are left alone, low-frequency pairs ($r_i < 1$) are fully interpolated, and pairs in between are blended.

- `q`: `(Tq, d)`, `k`: `(Tk, d)`, `pos_q`, `pos_k`: integer positions.
"""

STARTER = '''import numpy as np

def rope_scores(q, k, pos_q, pos_k, base, method, scale=1.0, orig_ctx=None):
    """Return (Tq, Tk) RoPE attention scores."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _freqs(d, base, method, s, L):
    i2 = np.arange(0, d, 2) / d
    theta = base ** (-i2)
    if method == 'linear':
        return theta / s, 1.0
    if method == 'ntk':
        return (base * s ** (d / (d - 2))) ** (-i2), 1.0
    if method == 'yarn':
        r = L * theta / (2 * np.pi)
        gamma = np.clip((r - 1.0) / (32.0 - 1.0), 0.0, 1.0)
        return (1 - gamma) * theta / s + gamma * theta, 0.1 * np.log(s) + 1.0
    return theta, 1.0

def _rotate(x, pos, theta):
    ang = np.asarray(pos, dtype=float)[:, None] * theta
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = a * c - b * s
    out[:, 1::2] = a * s + b * c
    return out

def rope_scores(q, k, pos_q, pos_k, base, method, scale=1.0, orig_ctx=None):
    q, k = np.asarray(q, dtype=float), np.asarray(k, dtype=float)
    d = q.shape[1]
    theta, mu = _freqs(d, base, method, scale, orig_ctx)
    return mu * mu * (_rotate(q, pos_q, theta) @ _rotate(k, pos_k, theta).T) / np.sqrt(d)
'''

HINTS = [
    'Write one function that returns the per-pair frequencies and the factor $\\mu$ for each method, and one that rotates a `(T, d)` array given positions and frequencies. Scores are then a matmul.',
    'NTK scaling does not touch positions or frequencies directly: it changes the base to $\\text{base}\\cdot s^{d/(d-2)}$ and recomputes $\\theta_i$. That leaves pair 0 unchanged and stretches the lowest frequency by exactly $s$.',
    'YaRN: $r_i = L\\theta_i/(2\\pi)$, $\\gamma_i = \\mathrm{clip}((r_i - 1)/31, 0, 1)$; the temperature factor multiplies the score by $\\mu^2$ because it is applied to both $q$ and $k$.',
]

_r = np.random.default_rng(2104)
_q, _k = _r.normal(size=(3, 8)), _r.normal(size=(4, 8))
_pq, _pk = np.array([10, 200, 4000]), np.array([0, 9, 199, 3999])
TESTS = [
    (_q, _k, _pq, _pk, 10000.0, 'none'),
    (_q, _k, _pq, _pk, 10000.0, 'linear', 4.0),
    (_q, _k, _pq, _pk, 10000.0, 'ntk', 4.0),
    (_q, _k, _pq, _pk, 10000.0, 'yarn', 4.0, 1024),
    (_q, _k, _pq + 5, _pk + 5, 10000.0, 'none'),
    (_r.normal(size=(2, 16)), _r.normal(size=(5, 16)), np.array([30000, 65000]), np.array([0, 100, 4096, 32768, 64999]), 500000.0, 'yarn', 8.0, 8192),
    (_r.normal(size=(2, 4)), _r.normal(size=(2, 4)), np.array([0, 1]), np.array([0, 1]), 100.0, 'ntk', 2.0),
]

EXPLANATION = r"""
## Why RoPE gives relative positions
A rotation by $m\theta$ followed by the transpose of a rotation by $n\theta$ is a rotation by $(m-n)\theta$, so $\langle R_m q, R_n k\rangle = \langle q, R_{m-n} k\rangle$. Test 5 shifts every position by 5 and gets the same scores as test 1. This is the property RoFormer introduced, and it is why no position table is needed.

## Extending the context
Beyond the trained length $L$, low-frequency pairs reach angles they never saw, and the model's attention degrades.
- **Position interpolation** divides every angle by $s$: all angles stay in the trained range, but nearby tokens become $s$ times harder to tell apart (high frequencies are compressed too). It needs some fine-tuning.
- **NTK-aware scaling** raises the base instead, which barely changes high frequencies and stretches the lowest ones by $s$. It works reasonably even without fine-tuning.
- **YaRN** interpolates per pair according to how many rotations it completes within $L$, and adds a temperature $\mu^2$ that sharpens the softmax back toward its original entropy at long range. gpt-oss, Qwen and DeepSeek models ship YaRN settings.

## Complexity
$O((T_q + T_k)d)$ for the rotations and $O(T_qT_kd)$ for the scores.

## Common bugs
- Using $\theta_i = \text{base}^{-i/d}$ (missing the factor 2).
- Multiplying by $\mu$ once instead of $\mu^2$ when comparing scores.
- Computing YaRN's $r_i$ from the wavelength in positions, $2\pi/\theta_i$, but forgetting to divide $L$ by it.

## Interview follow-ups
- For $d = 128$, base $10^4$, $L = 4096$: which pairs does YaRN leave untouched?
- Why does position interpolation need fine-tuning while YaRN often works with very little?
"""
