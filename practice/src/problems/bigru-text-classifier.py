import numpy as np

META = dict(title='Bidirectional GRU text classifier with masked pooling', tags=['Neural Networks', 'NLP'], diff='Hard',
            runtime='numpy', lessons=['dlcore:05.02'],
            related=['gru-cell', 'lstm-cell-2', 'masked-mean-pooling', 'padding-and-truncation'])

STATEMENT = r"""
A common sequence classifier: embed the tokens, run a GRU forward and another backward, concatenate the two states at every position, average over the real tokens and classify.

Write `bigru_classify(ids, lengths, p)` returning `(logits, outputs)`.

## GRU cell (PyTorch convention)
With `W_ih` `(D, 3H)`, `W_hh` `(H, 3H)`, `b_ih`, `b_hh` `(3H,)` and blocks in order $r, z, n$:
$a = xW_{ih} + b_{ih}$, $u = hW_{hh} + b_{hh}$; $r = \sigma(a_r + u_r)$, $z = \sigma(a_z + u_z)$, $n = \tanh(a_n + r\odot u_n)$, $h' = (1 - z)\odot n + z\odot h$.

## Model
- `ids` `(B, L)`, `lengths` `(B,)` (at least 1); positions $\ge$ length are padding.
- Forward GRU (`p['fwd']`) runs over positions $0..\text{len}-1$; backward GRU (`p['bwd']`) runs over positions $\text{len}-1$ down to 0. Both start from $h = 0$ and **never see padding** (the backward pass starts at the last real token, not at position $L-1$).
- `outputs[b, t]` $= [h^{fwd}_t;\ h^{bwd}_t]$ for real positions and zeros for padding, shape `(B, L, 2H)`.
- Pool: mean of `outputs` over real positions. `logits = pooled @ W_c + b_c`.

`p` keys: `E` `(V, D)`, `fwd`, `bwd` (dicts with the four GRU arrays), `W_c` `(2H, K)`, `b_c` `(K,)`.
"""

STARTER = '''import numpy as np

def bigru_classify(ids, lengths, p):
    """Return (logits (B, K), outputs (B, L, 2H))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sig(z):
    return 0.5 * (1 + np.tanh(0.5 * z))

def _gru(x, h, g):
    H = h.shape[-1]
    a = x @ g['W_ih'] + g['b_ih']
    u = h @ g['W_hh'] + g['b_hh']
    r = _sig(a[:H] + u[:H])
    z = _sig(a[H:2 * H] + u[H:2 * H])
    n = np.tanh(a[2 * H:] + r * u[2 * H:])
    return (1 - z) * n + z * h

def bigru_classify(ids, lengths, p):
    ids = np.asarray(ids)
    B, L = ids.shape
    H = p['fwd']['W_hh'].shape[0]
    out = np.zeros((B, L, 2 * H))
    for b in range(B):
        n = int(lengths[b])
        x = p['E'][ids[b, :n]]
        h = np.zeros(H)
        for t in range(n):
            h = _gru(x[t], h, p['fwd'])
            out[b, t, :H] = h
        h = np.zeros(H)
        for t in range(n - 1, -1, -1):
            h = _gru(x[t], h, p['bwd'])
            out[b, t, H:] = h
    pooled = out.sum(1) / np.asarray(lengths, dtype=float)[:, None]
    return pooled @ p['W_c'] + p['b_c'], out
'''

HINTS = [
    'Write the GRU cell for one vector first. Note the reset gate multiplies $hW_{hn} + b_{hn}$ inside the tanh, not $h$ itself.',
    'Handle each row with its own length: the forward loop covers $0..n-1$, the backward loop $n-1..0$. Padding positions are never visited and stay zero.',
    'Because padding rows are zero, the masked mean is `outputs.sum(1) / lengths[:, None]`.',
]

_r = np.random.default_rng(2014)
def _g(D, H):
    return dict(W_ih=_r.normal(size=(D, 3 * H)) / np.sqrt(D), W_hh=_r.normal(size=(H, 3 * H)) / np.sqrt(H),
                b_ih=0.1 * _r.normal(size=3 * H), b_hh=0.1 * _r.normal(size=3 * H))
def _p(V, D, H, K):
    return dict(E=_r.normal(size=(V, D)), fwd=_g(D, H), bwd=_g(D, H), W_c=_r.normal(size=(2 * H, K)) / np.sqrt(2 * H),
                b_c=0.1 * _r.normal(size=K))
_p1, _p2 = _p(6, 3, 2, 2), _p(10, 4, 5, 3)
TESTS = [
    (np.array([[1, 2, 3]]), np.array([3]), _p1),
    (np.array([[1, 2, 3], [4, 5, 0]]), np.array([3, 2]), _p1),
    (np.array([[4, 0, 0, 0]]), np.array([1]), _p1),
    (_r.integers(1, 10, size=(3, 6)), np.array([6, 4, 1]), _p2),
    (_r.integers(1, 10, size=(2, 8)), np.array([5, 8]), _p2),
]

EXPLANATION = r"""
## The GRU in one line
A GRU has two gates instead of the LSTM's three and no separate cell state: the update gate $z$ interpolates between keeping the old state and writing the candidate $n$, and the reset gate $r$ decides how much of the old state feeds the candidate. It has three weight blocks instead of four, so about 25% fewer parameters, and performs comparably on many tasks.

## Why bidirectional, and why padding matters there
The forward state at position $t$ summarises the left context, the backward state the right context; together each position sees the whole sentence. If the backward GRU started at position $L-1$ it would first read padding embeddings and carry them into every real position, so results would change with the amount of padding. Packed sequences in frameworks exist to prevent exactly this.

## Complexity
$O(BL(3DH + 3H^2))$ per direction, sequential in $L$.

## Common bugs
- Using the "reset after" formula $r\odot h$ before the matrix multiply (the original Cho et al. variant) when the weights follow PyTorch's convention; the two are not interchangeable.
- Swapping $z$ and $1 - z$.
- Averaging over all $L$ positions instead of the real ones.

## Interview follow-ups
- What changes for a sequence-labelling task (one label per token)?
- Why can't the backward direction be used in a language model?
"""
