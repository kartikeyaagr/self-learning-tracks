import numpy as np

META = dict(title='Transformer encoder block: pre-LN and post-LN', tags=['Transformers', 'Attention'], diff='Hard', runtime='numpy',
            lessons=['aieng:04.03', 'aieng:05.03'],
            related=['multi-head-attention', 'layer-normalization', 'transformer-decoder-block'])

STATEMENT = r"""
An encoder block is two sublayers, each wrapped in a residual connection and a LayerNorm: multi-head self-attention, then a position-wise feed-forward network. Where the LayerNorm sits is a real design choice:

| `norm_first` | block |
|---|---|
| `True` (pre-LN, GPT-2 and most modern models) | $h = x + \mathrm{MHA}(\mathrm{LN}_1(x))$, then $y = h + \mathrm{FFN}(\mathrm{LN}_2(h))$ |
| `False` (post-LN, the original Transformer and BERT) | $h = \mathrm{LN}_1(x + \mathrm{MHA}(x))$, then $y = \mathrm{LN}_2(h + \mathrm{FFN}(h))$ |

Write `encoder_block(x, p, n_heads, pad_mask=None, norm_first=True)`.

## Pieces
- `x`: `(B, T, D)`. `p` is a dict: `Wq`, `Wk`, `Wv`, `Wo` `(D, D)` (no attention biases); `W1` `(D, F)`, `b1` `(F,)`, `W2` `(F, D)`, `b2` `(D,)`; `ln1`, `ln2`: tuples `(gamma, beta)` of `(D,)` arrays.
- **LN**: over the last axis, population variance, $\epsilon = 10^{-5}$.
- **MHA** (self-attention, bidirectional): $H$ = `n_heads` divides $D$, $d_h = D/H$; head $h$ uses feature columns $h d_h : (h+1) d_h$ of $Q = zW_q$, $K = zW_k$, $V = zW_v$ (where $z$ is the sublayer input); scores divided by $\sqrt{d_h}$; concatenate heads and multiply by $W_o$.
- `pad_mask`: optional bool `(B, T)`, `True` = real token. Padded positions are hidden as **keys** (score $-\infty$); they are still computed as queries. If a sample has no real token at all, its attention output is zero.
- **FFN**: $\mathrm{GELU}(z W_1 + b_1) W_2 + b_2$ with the tanh approximation $\mathrm{GELU}(u) = \tfrac12 u\,(1 + \tanh(\sqrt{2/\pi}(u + 0.044715 u^3)))$.

## Output
Float array `(B, T, D)`.
"""

STARTER = '''import numpy as np

def encoder_block(x, p, n_heads, pad_mask=None, norm_first=True):
    """Return the (B, T, D) output of one transformer encoder block."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _ln(z, gb, eps=1e-5):
    g, b = gb
    mu = z.mean(axis=-1, keepdims=True)
    return g * (z - mu) / np.sqrt(z.var(axis=-1, keepdims=True) + eps) + b

def _mha(z, p, H, mask):
    B, T, D = z.shape
    dh = D // H
    split = lambda M: (z @ M).reshape(B, T, H, dh).transpose(0, 2, 1, 3)   # (B, H, T, dh)
    Q, K, V = split(p['Wq']), split(p['Wk']), split(p['Wv'])
    S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh)
    if mask is not None:
        S = np.where(np.asarray(mask)[:, None, None, :], S, -np.inf)       # hide padded keys
    m = S.max(axis=-1, keepdims=True)
    w = np.exp(S - np.where(np.isfinite(m), m, 0.0))
    tot = w.sum(axis=-1, keepdims=True)
    A = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
    return (A @ V).transpose(0, 2, 1, 3).reshape(B, T, D) @ p['Wo']

def _ffn(z, p):
    u = z @ p['W1'] + p['b1']
    gelu = 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3)))
    return gelu @ p['W2'] + p['b2']

def encoder_block(x, p, n_heads, pad_mask=None, norm_first=True):
    p = {k: (tuple(np.asarray(a, dtype=float) for a in v) if isinstance(v, tuple) else np.asarray(v, dtype=float))
         for k, v in p.items()}
    x = np.asarray(x, dtype=float)
    if norm_first:
        h = x + _mha(_ln(x, p['ln1']), p, n_heads, pad_mask)
        return h + _ffn(_ln(h, p['ln2']), p)
    h = _ln(x + _mha(x, p, n_heads, pad_mask), p['ln1'])
    return _ln(h + _ffn(h, p), p['ln2'])
'''

HINTS = [
    'Write three helpers, `ln`, `mha` and `ffn`, each taking a `(B, T, D)` array; the block is then four lines for either ordering.',
    'In `mha`, reshape each projection to `(B, H, T, d_h)`, mask keys with `pad_mask[:, None, None, :]`, and guard the softmax for rows that are entirely `-inf`.',
    'Pre-LN normalises the **input of each sublayer** and adds the raw residual; post-LN adds first and normalises the **sum**. Both use `ln1` around attention and `ln2` around the FFN.',
]

_r = np.random.default_rng(97)
def _p(D, F):
    s = 1 / np.sqrt(D)
    return {'Wq': _r.normal(size=(D, D)) * s, 'Wk': _r.normal(size=(D, D)) * s, 'Wv': _r.normal(size=(D, D)) * s,
            'Wo': _r.normal(size=(D, D)) * s, 'W1': _r.normal(size=(D, F)) * s, 'b1': _r.normal(size=F) * 0.1,
            'W2': _r.normal(size=(F, D)) / np.sqrt(F), 'b2': _r.normal(size=D) * 0.1,
            'ln1': (1 + 0.1 * _r.normal(size=D), 0.1 * _r.normal(size=D)), 'ln2': (1 + 0.1 * _r.normal(size=D), 0.1 * _r.normal(size=D))}
_z = {'Wq': np.zeros((2, 2)), 'Wk': np.zeros((2, 2)), 'Wv': np.eye(2), 'Wo': np.eye(2), 'W1': np.zeros((2, 1)), 'b1': np.zeros(1),
      'W2': np.zeros((1, 2)), 'b2': np.zeros(2), 'ln1': (np.ones(2), np.zeros(2)), 'ln2': (np.ones(2), np.zeros(2))}
_P = _p(8, 16)
_x = _r.normal(size=(2, 4, 8))
TESTS = [
    (np.array([[[1.0, 3.0], [3.0, 1.0]]]), _z, 1),
    (np.array([[[1.0, 3.0], [3.0, 1.0]]]), _z, 1, np.array([[True, False]]), False),
    (_x, _P, 2),
    (_x, _P, 2, None, False),
    (_x, _P, 4, np.array([[True, True, True, False], [True, False, False, False]])),
    (_x, _P, 4, np.array([[True, True, False, False], [False, False, False, False]]), False),
    (_r.normal(size=(1, 5, 6)) * 3, _p(6, 24), 3),
]

EXPLANATION = r"""
## Pre-LN vs post-LN
In post-LN every residual sum passes through a LayerNorm, so the gradient reaching early layers is repeatedly rescaled; deep post-LN models need learning-rate warmup and careful initialisation to train at all. Pre-LN leaves an untouched identity path from input to output ($y = x + \sum \text{sublayers}$), which trains stably without warmup. The cost: the residual stream's scale grows with depth, so pre-LN models add a final LayerNorm after the last block, and post-LN can reach slightly better quality when it does train.

## What the mask does and doesn't do
The key-padding mask stops real tokens from attending to padding. Padded positions still produce outputs (they attend to the real tokens), and those outputs are garbage that the loss must ignore. Masking queries too would only save compute.

## Shapes worth knowing
$Q, K, V$: `(B, H, T, d_h)`; scores and weights: `(B, H, T, T)`; the FFN widens to $F$ (usually $4D$) and back. Parameters per block: $4D^2$ for attention plus $2DF + F + D$ for the FFN plus $4D$ for the norms.

## Complexity
$O(BT^2D)$ for attention and $O(BTDF)$ for the FFN; at short context the FFN dominates.

## Common bugs
- Pre-LN with the residual taken from the normalised input ($h = \mathrm{LN}(x) + \ldots$) instead of $x$.
- Masking the query axis instead of the key axis (`mask[:, None, :, None]`).
- Forgetting the $1/\sqrt{d_h}$, or dividing by $\sqrt{D}$.

## Interview follow-ups
- Where would dropout go in this block during training?
- Count the parameters and FLOPs of one block of BERT-base ($D = 768$, $F = 3072$, $H = 12$) at $T = 512$.
"""
