import numpy as np

META = dict(title='Sliding-window causal mask with attention sinks and a cache offset', tags=['Attention', 'Transformers'], diff='Easy', runtime='numpy',
            related=['causal-grouped-query-attention'])

STATEMENT = r"""
Write `swa_mask(q_len, past, window, sinks)` returning a boolean array of shape `(q_len, past + q_len)`.

Query row $i$ sits at absolute position $p = $ `past` $+ i$; key column $j$ is position $j$. Key $j$ is visible to query $p$ when it is causal ($j \le p$) and either

- inside the window: $p - j <$ `window`, or
- one of the first `sinks` positions: $j <$ `sinks`.

`window <= 0` means no window limit (plain causal attention).
"""

STARTER = '''import numpy as np

def swa_mask(q_len, past, window, sinks):
    """Boolean visibility mask for sliding-window attention."""
    T = past + q_len
    return np.tril(np.ones((q_len, T), bool))
'''

SOLUTION = '''import numpy as np

def swa_mask(q_len, past, window, sinks):
    p = past + np.arange(q_len)[:, None]
    j = np.arange(past + q_len)[None, :]
    causal = j <= p
    near = np.ones_like(causal) if window <= 0 else (p - j < window)
    return causal & (near | (j < sinks))
'''

HINTS = [
    'Work with absolute positions: the query rows start at past, not 0, during cached decoding.',
    'Build the mask by broadcasting a column of query positions against a row of key positions.',
    'Sinks are an extra OR term; they still have to be causal.',
]

TESTS = [(4, 0, 2, 0), (5, 0, 3, 1), (1, 7, 3, 0), (2, 6, 4, 2), (3, 0, 0, 0), (3, 2, 10, 0)]

EXPLANATION = r"""
## Why a window
Full causal attention lets every token see the whole prefix, so the KV cache grows with the sequence. A window of $W$ keeps only the last $W$ keys per layer: cache memory and per-token cost become constant. Information from further back can still flow through stacked layers, each extending the reach by up to $W$.

## Attention sinks
Models put a lot of attention mass on the first few tokens regardless of content. When a sliding window evicts them, quality can collapse. Keeping a handful of initial "sink" tokens visible (StreamingLLM) fixes most of that for little cost.

## The offset bug
During decoding the new token is at position `past`, not 0. Building the mask as if it were row 0 lets the token see only the first key, a silent and common bug.

## Interview follow-ups
- Why do hybrid models mix sliding-window layers with a few full-attention layers?
"""
