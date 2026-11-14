import numpy as np

META = dict(title='Entropy, Gini impurity and information gain for a split', tags=['Classic ML', 'Classification'], diff='Easy', runtime='numpy',
            related=['decision-tree-classifier-cart', 'random-forest', 'decision-tree-regressor'])

STATEMENT = r"""
Write `split_scores(y, left_mask)` where `y` holds integer class labels at a tree node and `left_mask` is a boolean array saying which rows go to the left child.

Return a dict with:
- `"entropy"`: entropy of the parent node in bits, $H = -\sum_c p_c \log_2 p_c$ (terms with $p_c = 0$ count as 0);
- `"gini"`: Gini impurity of the parent, $G = 1 - \sum_c p_c^2$;
- `"info_gain"`: $H(\text{parent}) - \frac{n_L}{n} H(L) - \frac{n_R}{n} H(R)$;
- `"gini_gain"`: the same with Gini instead of entropy.

An empty child contributes 0 (its weight is 0). All values are floats.
"""

STARTER = '''import numpy as np

def split_scores(y, left_mask):
    """Return {"entropy", "gini", "info_gain", "gini_gain"} for the parent node and the split."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _probs(y):
    if len(y) == 0:
        return np.array([])
    _, c = np.unique(y, return_counts=True)
    return c / c.sum()

def _H(y):
    p = _probs(y)
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0

def _G(y):
    p = _probs(y)
    return float(1 - (p ** 2).sum()) if len(p) else 0.0

def split_scores(y, left_mask):
    y, m = np.asarray(y), np.asarray(left_mask, dtype=bool)
    n, nl = len(y), m.sum()
    wl, wr = nl / n, (n - nl) / n
    L, R = y[m], y[~m]
    return {"entropy": _H(y), "gini": _G(y),
            "info_gain": _H(y) - wl * _H(L) - wr * _H(R),
            "gini_gain": _G(y) - wl * _G(L) - wr * _G(R)}
'''

HINTS = [
    'Get class proportions with `np.unique(y, return_counts=True)`; classes that are absent simply do not appear, so there is no log 0.',
    'Information gain is parent impurity minus the size-weighted average impurity of the two children.',
    'Write one helper per impurity and call it on the parent, left and right subsets.',
]

TESTS = [
    (np.array([1] * 10 + [0] * 10), np.array([True] * 8 + [False] * 2 + [True] * 2 + [False] * 8)),
    (np.array([0, 0, 1, 1]), np.array([True, False, True, False])),
    (np.array([0, 1, 2, 0, 1, 2]), np.array([True, True, False, False, False, False])),
    (np.array([1, 1, 1]), np.array([True, False, False])),
    (np.array([0, 1, 1, 1]), np.array([False, False, False, False])),
    (np.array([0, 0, 0, 1, 1, 2, 2, 2]), np.array([True, True, True, False, False, False, False, False])),
]

EXPLANATION = r"""
## Reading the numbers
For the first test, the parent is 10/10 (1 bit). Each child is 8/2 with entropy 0.722 bits, so the gain is 0.278 bits. A split that leaves both children 50/50 gains nothing.

## Entropy versus Gini
Both are zero for a pure node and largest for a uniform mix. Gini avoids logarithms and is slightly cheaper; in practice the two choose the same split most of the time. CART uses Gini by default; ID3 and C4.5 use entropy.

## Information gain is mutual information
The gain equals the mutual information between the split indicator and the label on the node's data, so it is never negative.

## Bias towards many-valued features
A feature with a unique value per row can split into pure singletons and get the maximum gain while generalising terribly. Gain ratio and minimum leaf sizes counter that.

## Interview follow-ups
- Which impurity would you use for regression trees? (Variance reduction.)
"""
