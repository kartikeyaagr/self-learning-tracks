import numpy as np

META = dict(title='Attention with learned sink logits', tags=['Attention'], diff='Medium', runtime='numpy',
            lessons=['aieng:04.02', 'aieng:07.02'],
            related=['scaled-dot-product-attention', 'causal-grouped-query-attention', 'softmax-numerical-stability-and-log-sum-exp'])

STATEMENT = r"""
Softmax forces every query to spend all of its attention somewhere, even when no key is relevant. Trained models often dump that excess on the first token (an "attention sink"), which breaks when the first token falls out of a sliding window. A cleaner fix is to give each head a **learned sink logit** $s_h$: it joins the softmax denominator as an extra column that has no value vector, so a head can attend to "nothing".

$$a_{t,j} = \frac{e^{S_{t,j}}}{e^{s_h} + \sum_{j' \in \mathcal{A}(t)} e^{S_{t,j'}}}, \qquad o_t = \sum_{j \in \mathcal{A}(t)} a_{t,j} v_j$$

Write `sink_attention(q, k, v, sinks, window=None)` returning `(out, sink_mass)`:
- `q`, `k`, `v`: `(H, T, d)` (KV heads already expanded); `sinks`: `(H,)`.
- $S = qk^\top/\sqrt{d}$. $\mathcal{A}(t)$ is causal ($j \le t$) and, if `window` is given, also $j > t - \text{window}$.
- `out`: `(H, T, d)`. `sink_mass`: `(H, T)`, the weight the sink column received, $e^{s_h}/(\ldots)$.

It must stay finite for large logits (e.g. $|S|$ or $|s_h|$ around 1000).
"""

STARTER = '''import numpy as np

def sink_attention(q, k, v, sinks, window=None):
    """Return (out (H, T, d), sink_mass (H, T))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def sink_attention(q, k, v, sinks, window=None):
    q, k, v = (np.asarray(a, dtype=float) for a in (q, k, v))
    H, T, d = q.shape
    S = q @ k.transpose(0, 2, 1) / np.sqrt(d)
    t, j = np.arange(T)[:, None], np.arange(T)[None, :]
    ok = j <= t
    if window is not None:
        ok &= j > t - window
    S = np.where(ok, S, -np.inf)
    sink = np.broadcast_to(np.asarray(sinks, dtype=float)[:, None, None], (H, T, 1))
    full = np.concatenate([S, sink], axis=-1)                 # sink is one extra column per row
    m = full.max(-1, keepdims=True)
    e = np.exp(full - m)
    w = e / e.sum(-1, keepdims=True)
    return w[..., :T] @ v, w[..., T]
'''

HINTS = [
    'Append the sink logit as one more column of the score matrix, take an ordinary stable softmax over the widened rows, then drop that column before multiplying by $V$.',
    'The row max must include the sink logit; otherwise a large sink overflows `exp`.',
]

_r = np.random.default_rng(2508)
_q, _k, _v = (_r.normal(size=(2, 5, 4)) for _ in range(3))
TESTS = [
    (_q, _k, _v, np.array([-30.0, -30.0])),
    (_q, _k, _v, np.array([0.0, 1.5])),
    (_q, _k, _v, np.array([0.5, -0.5]), 2),
    (_q * 400, _k * 3, _v, np.array([1000.0, -1000.0])),
    (_r.normal(size=(3, 7, 2)), _r.normal(size=(3, 7, 2)), _r.normal(size=(3, 7, 2)), np.array([2.0, 0.0, -2.0]), 3),
    (_r.normal(size=(1, 1, 3)), _r.normal(size=(1, 1, 3)), _r.normal(size=(1, 1, 3)), np.array([0.0])),
]

EXPLANATION = r"""
## What the sink buys
Without a sink the weights over real keys always sum to 1. With one they sum to $1 - \text{sink\_mass}$, so a head can effectively switch itself off for a token. That removes the incentive to park attention on the first token, which matters for sliding-window layers (the first token is not in the window) and for streaming. Very negative sinks recover ordinary attention; a sink that dominates makes the output shrink toward zero.

## Relation to other designs
- "Off-by-one" softmax adds a constant 1 to the denominator, i.e. a fixed sink logit of 0.
- StreamingLLM keeps the first few tokens in the KV cache for the same reason; a learned sink achieves it without extra cache entries.
- gpt-oss uses one learned sink per head in every attention layer.

## Implementation notes
The sink column needs no key and no value, so in a fused kernel it only adjusts the running max and normaliser of the online softmax at the end. Its cost is $O(HT)$.

## Common bugs
- Leaving the sink out of the max shift: `exp(1000)` overflows.
- Multiplying the full widened weight matrix by $V$ (shape mismatch) or dropping the sink before normalising (no effect at all).
- Applying the window mask to the sink column.

## Interview follow-ups
- What is the gradient of the loss with respect to $s_h$, in terms of `sink_mass`?
- Why can't a model simply learn zero-valued keys to get the same effect?
"""
