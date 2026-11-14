import numpy as np

META = dict(title='BERT pre-training loss: MLM and NSP heads', tags=['Transformers', 'Language Models'], diff='Hard',
            runtime='numpy', lessons=['aieng:04.03'],
            related=['masked-language-modeling', 'next-sentence-prediction', 'transformer-encoder-block', 'cross-entropy-logits'])

STATEMENT = r"""
BERT is pre-trained with two losses computed from one bidirectional encoder pass: **masked language modelling** (predict the original token at corrupted positions) and **next-sentence prediction** (is segment B the real continuation of segment A?). Wire the whole thing together.

Write `bert_losses(ids, segs, mlm_labels, nsp_label, p, n_heads)` returning a tuple `(mlm_loss, nsp_loss)` of floats.

## Inputs
- `ids` `(T,)` int token ids (already corrupted), `0` = padding. `segs` `(T,)` segment ids in {0, 1}.
- `mlm_labels` `(T,)` int: the original token at scored positions, `-100` elsewhere.
- `nsp_label`: 0 if B follows A, 1 if B is random.
- `p`: `tok` `(V, D)`, `pos` `(P, D)` with $P \ge T$, `seg` `(2, D)`, `emb_ln` (gamma, beta); `layers` (list of encoder layers, keys as below); MLM head `Wt` `(D, D)`, `bt` `(D,)`, `head_ln`, `b_out` `(V,)`; pooler `Wp` `(D, D)`, `bp` `(D,)`; NSP classifier `Wn` `(D, 2)`, `bn` `(2,)`.

## Model
1. $x = \mathrm{LN}_{emb}(\text{tok}[ids] + \text{pos}[0..T) + \text{seg}[segs])$.
2. Each layer is post-LN: $h = \mathrm{LN}_1(x + \mathrm{MHA}(x))$, $x = \mathrm{LN}_2(h + \mathrm{GELU}(hW_1 + b_1)W_2 + b_2)$. Layer keys: `Wq Wk Wv Wo` `(D, D)` (no biases), `W1 b1 W2 b2`, `ln1`, `ln2`. Attention is bidirectional; padded positions are hidden as keys. $1/\sqrt{d_h}$ scaling, heads are contiguous column blocks.
3. **MLM head** (weights tied to `tok`): $u = \mathrm{LN}_{head}(\mathrm{GELU}(xW_t + b_t))$, logits $= u\,\text{tok}^\top + b_{out}$. `mlm_loss` is the mean cross-entropy over positions with a label (0.0 if there are none).
4. **NSP head**: pooled $= \tanh(x_0 W_p + b_p)$ from the first (`[CLS]`) position, logits $=$ pooled $W_n + b_n$, `nsp_loss` is the cross-entropy against `nsp_label`.

GELU is the tanh approximation $\tfrac12 u(1+\tanh(\sqrt{2/\pi}(u + 0.044715u^3)))$. Every LayerNorm uses population variance and $\epsilon = 10^{-12}$ (BERT's value).
"""

STARTER = '''import numpy as np

def bert_losses(ids, segs, mlm_labels, nsp_label, p, n_heads):
    """Return (mlm_loss, nsp_loss)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _ln(z, gb, eps=1e-12):
    g, b = gb
    mu = z.mean(-1, keepdims=True)
    return g * (z - mu) / np.sqrt(z.var(-1, keepdims=True) + eps) + b

def _gelu(u):
    return 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3)))

def _mha(x, L, H, key_ok):
    T, D = x.shape
    dh = D // H
    heads = lambda M: (x @ M).reshape(T, H, dh).transpose(1, 0, 2)
    Q, K, V = heads(L['Wq']), heads(L['Wk']), heads(L['Wv'])
    S = np.where(key_ok[None, None, :], Q @ K.transpose(0, 2, 1) / np.sqrt(dh), -np.inf)
    A = np.exp(S - S.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    return (A @ V).transpose(1, 0, 2).reshape(T, D) @ L['Wo']

def _xent(logits, y):
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return -logp[np.arange(len(y)), y]

def bert_losses(ids, segs, mlm_labels, nsp_label, p, n_heads):
    ids, segs, lab = np.asarray(ids), np.asarray(segs), np.asarray(mlm_labels)
    T = len(ids)
    x = _ln(p['tok'][ids] + p['pos'][:T] + p['seg'][segs], p['emb_ln'])
    key_ok = ids != 0
    for L in p['layers']:
        h = _ln(x + _mha(x, L, n_heads, key_ok), L['ln1'])
        x = _ln(h + _gelu(h @ L['W1'] + L['b1']) @ L['W2'] + L['b2'], L['ln2'])
    scored = lab != -100
    if scored.any():
        u = _ln(_gelu(x[scored] @ p['Wt'] + p['bt']), p['head_ln'])
        mlm = float(_xent(u @ p['tok'].T + p['b_out'], lab[scored]).mean())
    else:
        mlm = 0.0
    pooled = np.tanh(x[0] @ p['Wp'] + p['bp'])
    nsp = float(_xent((pooled @ p['Wn'] + p['bn'])[None], np.array([nsp_label]))[0])
    return mlm, nsp
'''

HINTS = [
    'The embedding sum has three terms (token, position, segment) and is normalised before the first layer. There is no $\\sqrt{D}$ scaling in BERT.',
    'Only the scored positions go through the MLM head: select `x[mlm_labels != -100]` first, then apply dense, GELU, LayerNorm and the tied projection.',
    'NSP reads only position 0 after the last layer, passes it through a tanh "pooler", then a 2-way classifier.',
]

_r = np.random.default_rng(1810)
def _lnp(D):
    return (1 + 0.1 * _r.normal(size=D), 0.1 * _r.normal(size=D))
def _model(V, D, F, nl, P=12):
    s = 1 / np.sqrt(D)
    layers = []
    for _ in range(nl):
        L = {k: _r.normal(size=(D, D)) * s for k in ('Wq', 'Wk', 'Wv', 'Wo')}
        L.update(W1=_r.normal(size=(D, F)) * s, b1=0.1 * _r.normal(size=F), W2=_r.normal(size=(F, D)) / np.sqrt(F),
                 b2=0.1 * _r.normal(size=D), ln1=_lnp(D), ln2=_lnp(D))
        layers.append(L)
    return {'tok': _r.normal(size=(V, D)) * 0.5, 'pos': _r.normal(size=(P, D)) * 0.1, 'seg': _r.normal(size=(2, D)) * 0.1,
            'emb_ln': _lnp(D), 'layers': layers, 'Wt': _r.normal(size=(D, D)) * s, 'bt': 0.1 * _r.normal(size=D),
            'head_ln': _lnp(D), 'b_out': 0.1 * _r.normal(size=V), 'Wp': _r.normal(size=(D, D)) * s,
            'bp': 0.1 * _r.normal(size=D), 'Wn': _r.normal(size=(D, 2)) * s, 'bn': 0.1 * _r.normal(size=2)}

# ids: 1=[CLS] 2=[SEP] 3=[MASK]
_m1, _m2 = _model(10, 4, 8, 1), _model(14, 8, 16, 2)
TESTS = [
    (np.array([1, 5, 3, 2, 6, 7, 2]), np.array([0, 0, 0, 0, 1, 1, 1]), np.array([-100, -100, 8, -100, -100, -100, -100]), 0, _m1, 1),
    (np.array([1, 5, 3, 2, 6, 7, 2, 0, 0]), np.array([0, 0, 0, 0, 1, 1, 1, 0, 0]), np.array([-100, -100, 8, -100, 9, -100, -100, -100, -100]), 1, _m1, 2),
    (np.array([1, 4, 4, 2, 5, 2]), np.array([0, 0, 0, 0, 1, 1]), np.full(6, -100), 1, _m1, 2),
    (np.array([1, 3, 9, 10, 2, 3, 12, 2, 0, 0]), np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0]), np.array([-100, 7, -100, 11, -100, 13, -100, -100, -100, -100]), 0, _m2, 2),
    (np.array([1, 3, 9, 10, 2, 3, 12, 2, 0, 0]), np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0]), np.array([-100, 7, -100, 11, -100, 13, -100, -100, -100, -100]), 1, _m2, 4),
    (np.array([1, 6, 7, 8, 2, 9, 3, 11, 5, 2]), np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), np.array([-100, -100, 4, -100, -100, -100, 10, -100, -100, -100]), 0, _m2, 8),
]

EXPLANATION = r"""
## Why two losses from one pass
The encoder runs once over `[CLS] A [SEP] B [SEP]`. MLM reads the final states at the corrupted positions; NSP reads only `[CLS]`. Because attention is bidirectional, every position sees both segments, which is what makes the masked-token objective possible (a left-to-right model can't use the right context).

## Details that are easy to miss
- **Segment embeddings** are the only thing that tells the model where A ends and B begins, apart from `[SEP]`.
- **The MLM head has its own transform** (dense, GELU, LayerNorm) before the tied output projection, plus a free output bias per vocabulary entry.
- **Loss over scored positions only.** About 15% of tokens are selected, and of those 80% become `[MASK]`, 10% a random token, 10% stay unchanged; the label is the original token in all three cases.
- **The pooler** is a learned dense layer plus tanh on `[CLS]`. Later work (RoBERTa) found NSP unhelpful and dropped it, but the pooler stayed in many fine-tuning recipes.

## Complexity
One encoder pass, $O(L(T^2 D + TDF))$, plus the MLM projection $O(mDV)$ for $m$ scored positions. Computing the vocabulary projection only at scored positions saves most of the head's cost.

## Common bugs
- Taking the MLM mean over all $T$ positions instead of the labelled ones.
- Projecting with a separate output matrix instead of the tied token embeddings.
- Masking padded positions as queries (harmless) but forgetting them as keys (changes every real token's output).

## Interview follow-ups
- Why does BERT leave 10% of selected tokens unchanged?
- How would you fine-tune this encoder for extractive question answering?
"""
