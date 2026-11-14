import numpy as np

META = dict(title='Merge two partial attention results with their log-sum-exps', tags=['Attention', 'GPU Programming'], diff='Medium', runtime='numpy',
            related=['masked-safe-softmax', 'online-softmax'])

STATEMENT = r"""
Attention over keys split into two disjoint sets gives two partial results: normalised outputs $O_1, O_2$ (shape `(n, d)`) and log-sum-exps $\ell_1, \ell_2$ (shape `(n,)`, `-inf` where a row saw no keys). Write `merge_attention(O1, lse1, O2, lse2)` returning the attention over the union:

$$\ell = \log(e^{\ell_1} + e^{\ell_2}), \qquad O = e^{\ell_1 - \ell} O_1 + e^{\ell_2 - \ell} O_2$$

computed stably via $m = \max(\ell_1, \ell_2)$. Rows where both are `-inf` give $O = 0$ and $\ell = -\infty$.

Return `(O, lse)`.
"""

STARTER = '''import numpy as np

def merge_attention(O1, lse1, O2, lse2):
    """Combine two partial attention results."""
    O1, O2 = np.asarray(O1), np.asarray(O2)
    return (O1 + O2) / 2, np.maximum(lse1, lse2)
'''

SOLUTION = '''import numpy as np

def merge_attention(O1, lse1, O2, lse2):
    O1, O2 = np.asarray(O1, float), np.asarray(O2, float)
    l1, l2 = np.asarray(lse1, float), np.asarray(lse2, float)
    m = np.maximum(l1, l2)
    both = np.isneginf(m)
    ms = np.where(both, 0.0, m)
    a, b = np.exp(l1 - ms), np.exp(l2 - ms)
    s = a + b
    lse = np.where(both, -np.inf, ms + np.log(np.where(both, 1.0, s)))
    w1 = np.where(both, 0.0, a / np.where(both, 1.0, s))
    w2 = np.where(both, 0.0, b / np.where(both, 1.0, s))
    return w1[:, None] * O1 + w2[:, None] * O2, lse
'''

HINTS = [
    'Each partial output is already normalised by its own softmax denominator; the lse tells you how large that denominator was.',
    'Subtract the larger lse before exponentiating, exactly as in a safe softmax.',
    'exp(-inf - m) = 0 makes an empty partial drop out automatically, as long as m itself is finite.',
]

def _split_case(seed, n, m, d, cut):
    r = np.random.default_rng(seed)
    S, V = r.normal(size=(n, m)) * 3, r.normal(size=(m, d))
    def part(Sp, Vp):
        if Sp.shape[1] == 0:
            return np.zeros((n, d)), np.full(n, -np.inf)
        mx = Sp.max(1, keepdims=True); E = np.exp(Sp - mx); l = E.sum(1, keepdims=True)
        return (E / l) @ Vp, (mx + np.log(l))[:, 0]
    o1, l1 = part(S[:, :cut], V[:cut]); o2, l2 = part(S[:, cut:], V[cut:])
    return o1, l1, o2, l2

TESTS = [
    _split_case(1, 3, 6, 2, 3),
    _split_case(2, 4, 10, 3, 1),
    _split_case(3, 2, 5, 4, 0),
    (np.zeros((2, 2)), np.array([-np.inf, 0.0]), np.ones((2, 2)), np.array([-np.inf, 0.0])),
    _split_case(4, 5, 8, 2, 8),
]

EXPLANATION = r"""
## Why this is the core of FlashAttention
Softmax needs the whole row, but the row does not fit in fast memory. Because partial results can be merged exactly using only their outputs and log-sum-exps, a kernel can process keys tile by tile and keep a running $(O, \ell)$ per query row. The online softmax is this merge applied repeatedly.

## Beyond one GPU
The same merge combines attention computed on different devices (ring attention, context parallelism) or on different cache segments (split-KV decoding, where several thread blocks each handle part of a long cache and a final kernel merges them).

## Interview follow-ups
- Why store the log-sum-exp rather than the max and the sum separately?
"""
