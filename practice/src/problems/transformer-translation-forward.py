import numpy as np

META = dict(title='Forward pass of a tiny encoder-decoder Transformer', tags=['Transformers', 'Attention'], diff='Hard',
            runtime='numpy', lessons=['aieng:04.03', 'dlcore:05.04'],
            related=['transformer-encoder-block', 'transformer-decoder-block', 'sinusoidal-positional-encoding',
                     'scaled-dot-product-attention'])

STATEMENT = r"""
Assemble the original Transformer: an encoder stack reads the source sentence, a decoder stack reads the shifted target and attends to the encoder output, and a tied projection turns decoder states into next-token logits.

Write `translate_logits(src, tgt_in, p, n_heads)` for **one** sentence pair (no batch axis).

## Inputs
- `src`: int array `(S,)`, `tgt_in`: int array `(T,)`. Token id `0` is padding; `src` has at least one real token.
- `p['emb']`: `(V, D)` embedding table, shared by source, target and the output layer.
- `p['enc']`: list of encoder layers, each a dict with `Wq Wk Wv Wo` `(D, D)`, `W1` `(D, F)`, `b1` `(F,)`, `W2` `(F, D)`, `b2` `(D,)`, `ln1`, `ln2` (each a tuple `(gamma, beta)`).
- `p['dec']`: list of decoder layers with self-attention weights `sWq sWk sWv sWo`, cross-attention weights `cWq cWk cWv cWo`, the same FFN weights, and `ln1`, `ln2`, `ln3`.

## The model (post-LN, as in the paper)
1. **Embed**: $x = \sqrt{D}\,E[\text{ids}] + PE$ with the sinusoidal table $PE_{t,2i} = \sin(t/10000^{2i/D})$, $PE_{t,2i+1} = \cos(t/10000^{2i/D})$, positions from 0. $D$ is even.
2. **Encoder layer**: $h = \mathrm{LN}_1(x + \mathrm{MHA}(x, x))$, $x' = \mathrm{LN}_2(h + \mathrm{FFN}(h))$. Source padding is hidden as keys.
3. **Decoder layer**: $a = \mathrm{LN}_1(y + \mathrm{MHA}_{self}(y, y))$ with a causal mask, $c = \mathrm{LN}_2(a + \mathrm{MHA}_{cross}(a, \text{memory}))$ with source padding hidden, $y' = \mathrm{LN}_3(c + \mathrm{FFN}(c))$. Target padding is not masked (the loss ignores those rows).
4. **Output**: logits $= y_{final} E^\top$, shape `(T, V)`.

MHA($q$-source $z$, $kv$-source $m$): $Q = zW_q$, $K = mW_k$, $V = mW_v$; head $h$ uses columns $h d_h:(h+1)d_h$; scores scaled by $1/\sqrt{d_h}$; heads concatenated and multiplied by $W_o$. FFN is $\max(0, zW_1 + b_1)W_2 + b_2$. LayerNorm uses population variance over the last axis and $\epsilon = 10^{-5}$.
"""

STARTER = '''import numpy as np

def translate_logits(src, tgt_in, p, n_heads):
    """Return (T, V) next-token logits for one (source, shifted target) pair."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _pe(T, D):
    pos = np.arange(T)[:, None]
    freq = 10000.0 ** (-np.arange(0, D, 2) / D)
    pe = np.zeros((T, D))
    pe[:, 0::2] = np.sin(pos * freq)
    pe[:, 1::2] = np.cos(pos * freq)
    return pe

def _ln(z, gb, eps=1e-5):
    g, b = gb
    mu = z.mean(-1, keepdims=True)
    return g * (z - mu) / np.sqrt(z.var(-1, keepdims=True) + eps) + b

def _mha(z, m, Wq, Wk, Wv, Wo, H, allowed):
    """allowed: bool (Tq, Tk), True where the query may look at the key."""
    dh = z.shape[1] // H
    heads = lambda X: X.reshape(X.shape[0], H, dh).transpose(1, 0, 2)
    Q, K, V = heads(z @ Wq), heads(m @ Wk), heads(m @ Wv)
    S = np.where(allowed[None], Q @ K.transpose(0, 2, 1) / np.sqrt(dh), -np.inf)
    A = np.exp(S - S.max(-1, keepdims=True))
    A = A / A.sum(-1, keepdims=True)
    return (A @ V).transpose(1, 0, 2).reshape(z.shape[0], -1) @ Wo

def _ffn(z, L):
    return np.maximum(0.0, z @ L['W1'] + L['b1']) @ L['W2'] + L['b2']

def translate_logits(src, tgt_in, p, n_heads):
    src, tgt_in = np.asarray(src), np.asarray(tgt_in)
    E = np.asarray(p['emb'], dtype=float)
    D = E.shape[1]
    S, T = len(src), len(tgt_in)
    key_ok = src != 0
    x = np.sqrt(D) * E[src] + _pe(S, D)
    for L in p['enc']:
        h = _ln(x + _mha(x, x, L['Wq'], L['Wk'], L['Wv'], L['Wo'], n_heads, np.broadcast_to(key_ok, (S, S))), L['ln1'])
        x = _ln(h + _ffn(h, L), L['ln2'])
    memory = x
    y = np.sqrt(D) * E[tgt_in] + _pe(T, D)
    causal = np.tril(np.ones((T, T), dtype=bool))
    cross_ok = np.broadcast_to(key_ok, (T, S))
    for L in p['dec']:
        a = _ln(y + _mha(y, y, L['sWq'], L['sWk'], L['sWv'], L['sWo'], n_heads, causal), L['ln1'])
        c = _ln(a + _mha(a, memory, L['cWq'], L['cWk'], L['cWv'], L['cWo'], n_heads, cross_ok), L['ln2'])
        y = _ln(c + _ffn(c, L), L['ln3'])
    return y @ E.T
'''

HINTS = [
    'Write one attention helper that takes a query source, a key/value source and a boolean "allowed" matrix. Encoder self-attention, decoder self-attention and cross-attention are then the same call with different arguments.',
    'The encoder mask is the source key mask broadcast to `(S, S)`; the decoder self-attention mask is `np.tril(ones((T, T)))`; the cross-attention mask is the source key mask broadcast to `(T, S)`.',
    'Embeddings are multiplied by $\\sqrt{D}$ before the positional table is added, and the same table $E$ is used (transposed) for the output logits.',
]

_r = np.random.default_rng(1706)
def _ln_p(D):
    return (1 + 0.1 * _r.normal(size=D), 0.1 * _r.normal(size=D))
def _enc(D, F):
    s = 1 / np.sqrt(D)
    L = {k: _r.normal(size=(D, D)) * s for k in ('Wq', 'Wk', 'Wv', 'Wo')}
    L.update(W1=_r.normal(size=(D, F)) * s, b1=0.1 * _r.normal(size=F), W2=_r.normal(size=(F, D)) / np.sqrt(F),
             b2=0.1 * _r.normal(size=D), ln1=_ln_p(D), ln2=_ln_p(D))
    return L
def _dec(D, F):
    s = 1 / np.sqrt(D)
    L = {k: _r.normal(size=(D, D)) * s for k in ('sWq', 'sWk', 'sWv', 'sWo', 'cWq', 'cWk', 'cWv', 'cWo')}
    L.update(W1=_r.normal(size=(D, F)) * s, b1=0.1 * _r.normal(size=F), W2=_r.normal(size=(F, D)) / np.sqrt(F),
             b2=0.1 * _r.normal(size=D), ln1=_ln_p(D), ln2=_ln_p(D), ln3=_ln_p(D))
    return L
def _model(V, D, F, ne, nd):
    return {'emb': _r.normal(size=(V, D)) / np.sqrt(D), 'enc': [_enc(D, F) for _ in range(ne)],
            'dec': [_dec(D, F) for _ in range(nd)]}

_m1 = _model(7, 4, 8, 1, 1)
_m2 = _model(11, 8, 16, 2, 2)
_m3 = _model(9, 6, 12, 1, 2)
TESTS = [
    (np.array([3, 4, 5]), np.array([1, 2]), _m1, 1),
    (np.array([3, 4, 0, 0]), np.array([1, 6, 2]), _m1, 2),
    (np.array([5, 9, 2, 7, 0]), np.array([1, 3, 8, 4]), _m2, 2),
    (np.array([5, 9, 2, 7, 0]), np.array([1, 3, 8, 4]), _m2, 4),
    (np.array([4, 0, 0]), np.array([1, 5, 0]), _m3, 3),
    (np.array([2, 8, 1, 3, 6, 6]), np.array([1, 7, 7, 2, 5]), _m3, 2),
]

EXPLANATION = r"""
## How the pieces map onto the paper
- **Shared embeddings**: the paper ties the source embedding, the target embedding and the pre-softmax projection (it used a shared BPE vocabulary). Multiplying by $\sqrt{D}$ keeps the embedding on the same scale as the positional table, whose entries are in $[-1, 1]$.
- **Three masks, one function**: padding in the source must be invisible both to the encoder and to cross-attention; the decoder additionally can't see future target tokens. Encoding all three as a boolean `(Tq, Tk)` matrix keeps the attention code identical.
- **Post-LN**: every sublayer is $\mathrm{LN}(x + \mathrm{Sublayer}(x))$. This is the arrangement in the paper; most later models moved the norm inside the residual branch (pre-LN) because post-LN needs warmup to train deep stacks.

## Training and inference
During training `tgt_in` is the target shifted right (a start token, then the target without its last token), and a cross-entropy against the unshifted target is taken at every position in parallel. At inference the decoder runs one step at a time: feed the tokens produced so far, take the last row of the logits, append the chosen token. The encoder runs once.

## Complexity
Encoder: $O(S^2 D + S D F)$ per layer; decoder: $O(T^2 D + T S D + T D F)$ per layer; output: $O(T D V)$, which dominates for real vocabularies.

## Common bugs
- Masking padded source tokens in the encoder but forgetting them in cross-attention.
- Adding the positional table before the $\sqrt{D}$ scaling, or scaling the output logits.
- Using the decoder's own states as keys in the cross-attention.

## Interview follow-ups
- Why can the decoder be trained in parallel but must decode sequentially?
- What has to be cached to make step-by-step decoding $O(T)$ per token instead of $O(T^2)$?
"""
