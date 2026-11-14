import numpy as np

META = dict(title='Per-label thresholds for multi-label product tagging', tags=['Classification', 'Computer Vision', 'Metrics & Evaluation'], diff='Medium', runtime='numpy',
            related=['threshold-at-precision'])

STATEMENT = r"""
An image tagger outputs one probability per product attribute (for example "has sleeves", "striped", "leather"). A tag is shown only if it is reliable enough. Write `tag_thresholds(P, Y, min_precision, min_support)`:

- `P` and `Y` are `(n, L)` arrays of scores and 0/1 labels on a validation set.
- For each label, consider each distinct score as a candidate threshold (tag shown if `score >= t`). Choose the **smallest** threshold whose precision is at least `min_precision`, which maximises recall under the constraint.
- If no threshold reaches the precision, or the label has fewer than `min_support` positives, disable it with threshold `inf`.
- Report `[threshold, precision, recall]` per label; disabled labels report `[inf, 0.0, 0.0]`.

Then apply the thresholds to `P` and compute micro-averaged precision and recall over all enabled labels. Return `{"labels": [...], "micro_precision": float, "micro_recall": float}`, with micro values 0.0 when undefined.
"""

STARTER = '''import numpy as np

def tag_thresholds(P, Y, min_precision, min_support):
    """Choose per-label thresholds under a precision floor."""
    L = np.asarray(P).shape[1]
    return {"labels": [[0.5, 0.0, 0.0]] * L, "micro_precision": 0.0, "micro_recall": 0.0}
'''

SOLUTION = '''import numpy as np

def tag_thresholds(P, Y, min_precision, min_support):
    P, Y = np.asarray(P, float), np.asarray(Y).astype(int)
    rows, th = [], []
    for j in range(P.shape[1]):
        p, y = P[:, j], Y[:, j]
        chosen = None
        if y.sum() >= min_support:
            for t in np.unique(p):
                f = p >= t
                prec = y[f].mean()
                if prec >= min_precision:
                    chosen = (float(t), float(prec), float(y[f].sum() / y.sum()))
                    break
        rows.append(list(chosen) if chosen else [float("inf"), 0.0, 0.0])
        th.append(chosen[0] if chosen else np.inf)
    th = np.asarray(th)
    enabled = np.isfinite(th)
    pred = (P >= th) & enabled
    tp = int((pred & (Y == 1))[:, enabled].sum())
    npred = int(pred[:, enabled].sum())
    npos = int(Y[:, enabled].sum())
    return {"labels": rows, "micro_precision": tp / npred if npred else 0.0, "micro_recall": tp / npos if npos else 0.0}
'''

HINTS = [
    'Scan thresholds upward from the lowest score; the first one meeting the precision floor gives the most recall.',
    'Precision is not monotonic in the threshold, so do not binary search.',
    'Micro averages pool true positives and predictions across enabled labels before dividing.',
]

def _case(seed, n, L, sep):
    r = np.random.default_rng(seed)
    base = r.uniform(0.05, 0.4, L)
    Y = (r.random((n, L)) < base).astype(int)
    P = np.clip(0.3 + sep * (Y - 0.3) + 0.15 * r.normal(size=(n, L)), 0, 1)
    return P, Y

TESTS = [
    ([[0.9, 0.2], [0.6, 0.8], [0.4, 0.7], [0.1, 0.1]], [[1, 0], [0, 1], [1, 1], [0, 0]], 0.6, 1),
    _case(1, 300, 4, 0.5) + (0.8, 5),
    _case(2, 300, 6, 0.3) + (0.9, 5),
    _case(3, 100, 3, 0.6) + (0.7, 40),
    ([[0.5], [0.5]], [[0], [1]], 0.6, 1),
]

EXPLANATION = r"""
## Why one threshold per label
Attributes differ in difficulty and frequency. A single 0.5 cut-off shows unreliable rare tags and hides confident common ones. A precision floor per label is what the product cares about: a wrong "leather" tag on a product page misleads buyers and returns cost money.

## Why disable labels
A label with too few validation positives has an unmeasurable precision, and one that never reaches the floor should not be shown at all. Disabling is safer than guessing, and the list of disabled labels tells the data team where to collect more examples.

## Micro versus macro
Micro averages are dominated by frequent labels and describe what users see on a typical page. Macro averages weight every label equally and expose weak rare labels. Report both.

## Interview follow-ups
- How would you keep thresholds valid after the model is retrained and its scores shift?
"""
