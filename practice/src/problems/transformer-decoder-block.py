import numpy as np

META = dict(title='Transformer decoder block with cross-attention', tags=['Transformers', 'Attention'], diff='Hard', runtime='numpy',
            lessons=['aieng:05.01', 'aieng:04.03'],
            related=['transformer-encoder-block', 'multi-head-attention', 'scaled-dot-product-attention-2'])

STATEMENT = r"""
The decoder of an encoder-decoder transformer has three pre-LN sublayers, each with a residual connection:

$$\begin{aligned}
h_1 &= x + \mathrm{SelfAttn}(\mathrm{LN}_1(x)) && \text{causal: position } i \text{ sees } j \le i \\
h_2 &= h_1 + \mathrm{CrossAttn}(\mathrm{LN}_2(h_1),\ \text{memory}) && \text{queries from the decoder, keys/values from the encoder} \\
y &= h_2 + \mathrm{FFN}(\mathrm{LN}_3(h_2))
\end{aligned}$$

Write `decoder_block(x, memory, p, n_heads, mem_mask=None)` that returns `(y, cross_weights)`.

## Pieces
- `x`: `(B, T, D)` decoder states; `memory`: `(B, S, D)` encoder output (**not** normalised again).
- `p['self']` and `p['cross']`: dicts with `Wq`, `Wk`, `Wv`, `Wo`, all `(D, D)`, no biases. For cross-attention, $Q = \mathrm{LN}_2(h_1) W_q$ while $K = \text{memory}\,W_k$ and $V = \text{memory}\,W_v$.
- Heads: $H$ = `n_heads`, $d_h = D/H$, head $h$ uses columns $h d_h:(h+1)d_h$; scores divided by $\sqrt{d_h}$; stable softmax; concatenate heads, multiply by $W_o$.
- `mem_mask`: optional bool `(B, S)`, `True` = real encoder token; padded memory keys get $-\infty$. Every test leaves at least one real memory token per sample.
- `p['W1']` `(D, F)`, `p['b1']`, `p['W2']` `(F, D)`, `p['b2']`: FFN $\mathrm{ReLU}(zW_1 + b_1)W_2 + b_2$.
- `p['ln1']`, `p['ln2']`, `p['ln3']`: `(gamma, beta)` tuples; LayerNorm over the last axis, population variance, $\epsilon = 10^{-5}$.

## Output
- `y`: `(B, T, D)`.
- `cross_weights`: the cross-attention probabilities, `(B, H, T, S)` (rows sum to 1; padded memory columns are 0).
"""

STARTER = '''import numpy as np

def decoder_block(x, memory, p, n_heads, mem_mask=None):
    """Return (y (B, T, D), cross_weights (B, H, T, S))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _ln(z, gb, eps=1e-5):
    g, b = (np.asarray(a, dtype=float) for a in gb)
    mu = z.mean(axis=-1, keepdims=True)
    return g * (z - mu) / np.sqrt(z.var(axis=-1, keepdims=True) + eps) + b

def _attn(q_in, kv_in, W, H, allowed):
    """allowed: bool, broadcastable to (B, H, Tq, Tk). Returns (output, weights)."""
    B, Tq, D = q_in.shape
    Tk, dh = kv_in.shape[1], D // H
    heads = lambda z, M, T: (z @ np.asarray(M, dtype=float)).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    Q, K, V = heads(q_in, W['Wq'], Tq), heads(kv_in, W['Wk'], Tk), heads(kv_in, W['Wv'], Tk)
    S = np.where(allowed, Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh), -np.inf)
    A = np.exp(S - S.max(axis=-1, keepdims=True))        # every row has an allowed key here
    A /= A.sum(axis=-1, keepdims=True)
    return (A @ V).transpose(0, 2, 1, 3).reshape(B, Tq, D) @ np.asarray(W['Wo'], dtype=float), A

def decoder_block(x, memory, p, n_heads, mem_mask=None):
    x, mem = np.asarray(x, dtype=float), np.asarray(memory, dtype=float)
    B, T, _ = x.shape
    S = mem.shape[1]
    causal = np.tril(np.ones((T, T), dtype=bool))                       # j <= i
    mem_ok = np.ones((B, S), dtype=bool) if mem_mask is None else np.asarray(mem_mask, dtype=bool)
    z = _ln(x, p['ln1'])
    h1 = x + _attn(z, z, p['self'], n_heads, causal)[0]
    out, A = _attn(_ln(h1, p['ln2']), mem, p['cross'], n_heads, mem_ok[:, None, None, :])
    h2 = h1 + out
    f = np.maximum(_ln(h2, p['ln3']) @ np.asarray(p['W1'], dtype=float) + p['b1'], 0.0)
    return h2 + f @ np.asarray(p['W2'], dtype=float) + p['b2'], A
'''

HINTS = [
    'One attention helper can serve both sublayers if it takes separate query and key/value inputs plus a boolean "allowed" mask that broadcasts to `(B, H, T_q, T_k)`.',
    'Self-attention: queries, keys and values all come from `LN1(x)`, mask `np.tril(np.ones((T, T), bool))`. Cross-attention: queries from `LN2(h1)`, keys and values from the raw `memory`, mask `mem_mask[:, None, None, :]`.',
    'Keep the cross-attention weights `(B, H, T, S)` from the helper to return them. Each sublayer adds its output to the un-normalised residual stream.',
]

_r = np.random.default_rng(101)
def _att(D):
    s = 1 / np.sqrt(D)
    return {k: _r.normal(size=(D, D)) * s for k in ('Wq', 'Wk', 'Wv', 'Wo')}
def _p(D, F):
    ln = lambda: (1 + 0.1 * _r.normal(size=D), 0.1 * _r.normal(size=D))
    return {'self': _att(D), 'cross': _att(D), 'W1': _r.normal(size=(D, F)) / np.sqrt(D), 'b1': _r.normal(size=F) * 0.1,
            'W2': _r.normal(size=(F, D)) / np.sqrt(F), 'b2': _r.normal(size=D) * 0.1, 'ln1': ln(), 'ln2': ln(), 'ln3': ln()}
_Z, _E = np.zeros((2, 2)), np.eye(2)
_tiny = {'self': {'Wq': _Z, 'Wk': _Z, 'Wv': _E, 'Wo': _E}, 'cross': {'Wq': _Z, 'Wk': _Z, 'Wv': _E, 'Wo': _E},
         'W1': np.zeros((2, 1)), 'b1': np.zeros(1), 'W2': np.zeros((1, 2)), 'b2': np.zeros(2),
         'ln1': (np.ones(2), np.zeros(2)), 'ln2': (np.ones(2), np.zeros(2)), 'ln3': (np.ones(2), np.zeros(2))}
_P = _p(8, 16)
TESTS = [
    (np.array([[[1.0, -1.0], [3.0, 1.0]]]), np.array([[[2.0, 0.0], [0.0, 4.0]]]), _tiny, 1),
    (np.array([[[1.0, -1.0], [3.0, 1.0]]]), np.array([[[2.0, 0.0], [0.0, 4.0]]]), _tiny, 2, np.array([[True, False]])),
    (_r.normal(size=(2, 4, 8)), _r.normal(size=(2, 5, 8)), _P, 2),
    (_r.normal(size=(2, 4, 8)), _r.normal(size=(2, 5, 8)), _P, 4, np.array([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=bool)),
    (_r.normal(size=(1, 1, 8)), _r.normal(size=(1, 3, 8)), _P, 1),
    (_r.normal(size=(1, 6, 6)) * 2, _r.normal(size=(1, 2, 6)) * 2, _p(6, 12), 3),
]

EXPLANATION = r"""
## Three sublayers, three jobs
- **Causal self-attention** lets each target position read the target prefix only, so training on a whole sequence at once matches left-to-right generation.
- **Cross-attention** is where the source enters: decoder queries look up encoder keys and values. Its weights are the classic "alignment" between target and source tokens, which is why they're worth returning and inspecting.
- **FFN** processes each position independently.

## Masks
Self-attention never needs a fully-masked guard because the diagonal is always allowed. Cross-attention masks **memory** positions (padded source tokens); a padded target position is harmless here and is handled by the loss.

## Caching at inference
During generation the memory is fixed, so the cross-attention $K$ and $V$ are computed once per sequence and reused for every new token, while self-attention $K$/$V$ grow by one row per step. The causal mask makes the cached and uncached computations identical for the prefix.

## Complexity
$O(BT^2D)$ self-attention, $O(BTSD)$ cross-attention, $O(BTDF)$ FFN.

## Common bugs
- Normalising the memory with `ln2` (the encoder already applied its final LayerNorm).
- Building the causal mask with `np.triu` or with $j < i$ (a token must see itself).
- Taking keys from the decoder and queries from the memory in the cross sublayer.

## Interview follow-ups
- How do decoder-only models (GPT) differ? Which sublayer disappears and why?
- What shape does the cross-attention KV cache have, and when is it computed?
"""
