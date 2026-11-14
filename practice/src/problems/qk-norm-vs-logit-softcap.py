import numpy as np

META = dict(title='Bounding attention logits: soft-capping vs QK-norm', tags=['Attention', 'Transformers'], diff='Medium',
            runtime='numpy', lessons=['aieng:04.02'],
            related=['scaled-dot-product-attention', 'norm-gated-linear-transform', 'causal-grouped-query-attention'])

STATEMENT = r"""
Large attention logits saturate the softmax and are a common source of training instability. Two fixes appear in recent model families: **soft-capping** squashes the scores through a scaled tanh after the dot product, and **QK-norm** RMS-normalises each query and key head vector before it.

Write `attention_logits(q, k, mode, wq=None, wk=None, cap=None)` that returns the pre-mask score tensor `(H, Tq, Tk)` for:

| `mode` | scores |
|---|---|
| `'plain'` | $S = QK^\top/\sqrt{d}$ |
| `'softcap'` | $S = c\,\tanh\!\big(QK^\top/(\sqrt{d}\,c)\big)$ with $c$ = `cap` |
| `'qknorm'` | $S = \hat Q \hat K^\top/\sqrt{d}$ with $\hat q = (1 + w_q) \odot q/\sqrt{\mathrm{mean}(q^2) + 10^{-6}}$ per head vector, same for keys with $w_k$ |

- `q`: `(H, Tq, d)`, `k`: `(H, Tk, d)`; `wq`, `wk`: `(d,)` norm weights shared across heads. The $(1 + w)$ form is the zero-centred weight convention used by the Gemma family (a weight of 0 is the identity scale).
"""

STARTER = '''import numpy as np

def attention_logits(q, k, mode, wq=None, wk=None, cap=None):
    """Return (H, Tq, Tk) attention scores before masking."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _rms(x, w, eps=1e-6):
    return (1.0 + w) * x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)

def attention_logits(q, k, mode, wq=None, wk=None, cap=None):
    q, k = np.asarray(q, dtype=float), np.asarray(k, dtype=float)
    d = q.shape[-1]
    if mode == 'qknorm':
        q, k = _rms(q, np.asarray(wq, dtype=float)), _rms(k, np.asarray(wk, dtype=float))
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    if mode == 'softcap':
        s = cap * np.tanh(s / cap)
    return s
'''

HINTS = [
    'All three modes share $QK^\\top/\\sqrt{d}$. Soft-capping changes what happens after it; QK-norm changes the inputs before it.',
    'Normalise along the last axis (each head vector of length $d$ on its own), and remember the weight enters as $(1 + w)$.',
]

_r = np.random.default_rng(303)
_q, _k = _r.normal(size=(2, 3, 4)), _r.normal(size=(2, 5, 4))
_big_q, _big_k = _r.normal(size=(1, 2, 8)) * 30, _r.normal(size=(1, 3, 8)) * 30
_w = 0.2 * _r.normal(size=4)
TESTS = [
    (_q, _k, 'plain'),
    (_q, _k, 'softcap', None, None, 2.0),
    (_q, _k, 'qknorm', np.zeros(4), np.zeros(4)),
    (_q, _k, 'qknorm', _w, -_w),
    (_big_q, _big_k, 'softcap', None, None, 50.0),
    (_big_q, _big_k, 'qknorm', 0.1 * _r.normal(size=8), 0.1 * _r.normal(size=8)),
    (_big_q, _big_k, 'plain'),
]

EXPLANATION = r"""
## Two ways to keep logits bounded
- **Soft-capping** leaves small scores almost unchanged ($c\tanh(s/c) \approx s$ for $|s| \ll c$) and smoothly limits large ones to $(-c, c)$. It needs the full score matrix to be materialised and transformed, which fits poorly with fused attention kernels.
- **QK-norm** fixes the scale at the source: after RMS normalisation every head vector has norm about $\sqrt{d}$ before the weight, so $|S| \le \sqrt{d}\,\max|1+w_q|\,\max|1+w_k|$: bounded by the learned weights instead of by whatever the projections drift to. It needs nothing special inside the attention kernel.

Gemma 2 used soft-capping on attention and final logits; Gemma 3 replaced the attention soft-cap with QK-norm. Several other recent models (OLMo 2, Qwen3) adopted QK-norm as well.

## Complexity
Both add $O(HTd)$ (QK-norm) or $O(HT_qT_k)$ (soft-cap) elementwise work on top of the $O(HT_qT_kd)$ matmul.

## Common bugs
- Normalising over the head axis or the whole `(T, d)` slice instead of per vector.
- Using $w$ instead of $1 + w$ with zero-centred weights: an untrained weight of 0 then zeroes the input.
- Applying the cap to the softmax output instead of the logits.

## Interview follow-ups
- Why does QK-norm interact with RoPE ordering? (It is applied before the rotation, which preserves norms.)
- What happens to the gradient of a soft-capped logit that sits near $\pm c$?
"""
