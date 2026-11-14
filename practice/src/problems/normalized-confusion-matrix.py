import numpy as np

META = dict(title='Confusion matrix with row, column and total normalisation', tags=['Metrics & Evaluation'], diff='Hard', runtime='numpy',
            related=['classification-report', 'cross-tabulation', 'mean-iou-for-segmentation'])

STATEMENT = r"""
Write `confusion(y_true, y_pred, labels, normalize=None, weights=None)`.

- Rows are true classes and columns predicted classes, both in the order of `labels`. Entry $(i, j)$ counts rows with true label `labels[i]` and prediction `labels[j]`. Rows whose true or predicted label is not in `labels` are ignored.
- `weights` (optional) gives a weight per row; counts become sums of weights.
- `normalize`:
  - `None`: raw (weighted) counts;
  - `"true"`: each row divided by its sum (recall per class on the diagonal);
  - `"pred"`: each column divided by its sum (precision on the diagonal);
  - `"all"`: everything divided by the grand total.
  A zero row, column or total divides to 0.

Return a float array of shape `(k, k)`. Build it with one `np.add.at` call rather than a Python double loop.
"""

STARTER = '''import numpy as np

def confusion(y_true, y_pred, labels, normalize=None, weights=None):
    """Return the (k, k) confusion matrix."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def confusion(y_true, y_pred, labels, normalize=None, weights=None):
    t, p = np.asarray(y_true), np.asarray(y_pred)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)
    pos = {c: i for i, c in enumerate(labels)}
    keep = np.array([a in pos and b in pos for a, b in zip(t.tolist(), p.tolist())], dtype=bool)
    k = len(labels)
    M = np.zeros((k, k))
    if keep.any():
        ti = np.array([pos[a] for a in t[keep].tolist()]); pi = np.array([pos[b] for b in p[keep].tolist()])
        np.add.at(M, (ti, pi), w[keep])
    if normalize == "true":
        s = M.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        s = M.sum(axis=0, keepdims=True)
    elif normalize == "all":
        s = M.sum()
    else:
        return M
    return np.divide(M, s, out=np.zeros_like(M), where=np.broadcast_to(s, M.shape) != 0)
'''

HINTS = [
    'Map labels to positions with a dictionary, drop rows whose labels are not listed, and convert the rest to index arrays.',
    '`np.add.at(M, (rows, cols), weights)` accumulates correctly even when the same cell appears many times, unlike `M[rows, cols] += w`.',
    'For safe normalisation, use `np.divide(..., out=zeros, where=denominator != 0)`.',
]

TESTS = [
    (np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), [0, 1]),
    (np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), [0, 1], "true"),
    (np.array([0, 1, 2, 2, 1]), np.array([0, 2, 2, 1, 1]), [0, 1, 2], "pred"),
    (np.array([0, 1, 2]), np.array([0, 1, 2]), [0, 1, 2, 3], "true"),
    (np.array([1, 1, 0, 2]), np.array([1, 0, 0, 5]), [0, 1], "all", np.array([2.0, 1.0, 1.0, 3.0])),
    (np.array(['cat', 'dog', 'cat']), np.array(['dog', 'dog', 'cat']), ['dog', 'cat'], None),
]

EXPLANATION = r"""
## Which normalisation answers which question
- Normalise by true class (rows): "of the real dogs, what share did we call cats?" The diagonal is recall per class.
- Normalise by prediction (columns): "of the things we called dogs, what share were dogs?" The diagonal is precision.
- Normalise by total: the joint distribution of truth and prediction, useful for cost-weighted analysis.

## Why np.add.at
Fancy-index assignment `M[i, j] += w` applies only the last write when an index repeats. `np.add.at` performs unbuffered accumulation, so repeated cells add up correctly.

## Weights
Sample weights let the matrix reflect business value (transaction amount, importance) or correct for sampling in the evaluation set.

## Interview follow-ups
- Which off-diagonal cells would you look at first when debugging a model?
"""
