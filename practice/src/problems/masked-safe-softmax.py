import numpy as np

META = dict(title='Safe softmax with fully masked rows', tags=['Attention', 'GPU Programming'], diff='Easy', runtime='numpy',
            related=['online-softmax'])

STATEMENT = r"""
Write `masked_softmax(S, mask)` for scores `S (n, m)` and a boolean `mask` of the same shape (True = keep).

- For each row, subtract the maximum over **kept** entries before exponentiating, so large scores cannot overflow.
- Masked entries get probability 0.
- A row with no kept entries must come out as all zeros, not NaN.

Return `(P, lse)` where `lse[i]` $= \log\sum_{j\,kept} e^{S_{ij}}$, or `-inf` for an empty row.
"""

STARTER = '''import numpy as np

def masked_softmax(S, mask):
    """Row softmax over kept entries, safe for empty rows."""
    S = np.where(mask, S, -np.inf)
    P = np.exp(S - S.max(1, keepdims=True))
    return P / P.sum(1, keepdims=True), np.log(np.exp(S).sum(1))
'''

SOLUTION = '''import numpy as np

def masked_softmax(S, mask):
    S, mask = np.asarray(S, float), np.asarray(mask, bool)
    Z = np.where(mask, S, -np.inf)
    m = Z.max(1, keepdims=True)
    empty = ~mask.any(1, keepdims=True)
    m = np.where(empty, 0.0, m)
    E = np.where(mask, np.exp(Z - m), 0.0)
    l = E.sum(1, keepdims=True)
    P = np.where(empty, 0.0, E / np.where(empty, 1.0, l))
    lse = np.where(empty[:, 0], -np.inf, (m + np.log(np.where(empty, 1.0, l)))[:, 0])
    return P, lse
'''

HINTS = [
    'The max of an all -inf row is -inf, and -inf - (-inf) is NaN; replace the max with 0 for empty rows.',
    'Compute the log-sum-exp as max + log(sum), which stays finite for large scores.',
    'Downstream kernels use lse = -inf as the marker for "this row saw nothing" when merging tiles.',
]

_r = np.random.default_rng(1)
TESTS = [
    ([[1.0, 2.0, 3.0]], [[True, True, True]]),
    ([[1000.0, 1001.0], [-1000.0, -999.0]], [[True, True], [True, True]]),
    ([[1.0, 2.0], [3.0, 4.0]], [[True, False], [False, False]]),
    (_r.normal(size=(4, 6)) * 10, _r.random((4, 6)) < 0.5),
    ([[0.0, 0.0, 0.0]], [[False, False, False]]),
]

EXPLANATION = r"""
## Why rows can be empty
Padding, causal masks with offsets, and tiled kernels all produce rows (or row-tiles) where nothing is visible. A naive softmax turns them into NaN, and one NaN in attention spreads through the whole residual stream on the next layer.

## The convention
Return zeros for the probabilities and $-\infty$ for the log-sum-exp. Then a later merge with another tile weights this one by $e^{-\infty} = 0$ and the empty tile contributes nothing, which is exactly right.

## Interview follow-ups
- Why do kernels often use a large negative finite number instead of $-\infty$, and what can go wrong with that?
"""
