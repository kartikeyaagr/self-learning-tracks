import numpy as np

META = dict(title='Accuracy, precision, recall and F1: per class, micro and macro', tags=['Metrics & Evaluation', 'Classification'], diff='Medium', runtime='numpy',
            related=['normalized-confusion-matrix', 'precision-recall-curve-and-average-precision', 'cohens-kappa'])

STATEMENT = r"""
Write `classification_report(y_true, y_pred, labels)` for integer labels. `labels` lists the classes to report, in order.

Return a dict:
- `"accuracy"`: share of rows where `y_true == y_pred` (over all rows);
- `"per_class"`: list of `(precision, recall, f1, support)` tuples, one per label, where support is the number of true rows of that class;
- `"micro"`: `(precision, recall, f1)` computed from TP, FP and FN summed over `labels`;
- `"macro"`: unweighted mean of the per-class precision, recall and F1;
- `"weighted"`: support-weighted mean of the per-class values.

Any 0/0 ratio is defined as 0.0. F1 is $2PR/(P+R)$ (0 if $P + R = 0$). Floats everywhere except support (int).
"""

STARTER = '''import numpy as np

def classification_report(y_true, y_pred, labels):
    """Return accuracy, per-class, micro, macro and weighted metrics."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _div(a, b):
    return float(a / b) if b else 0.0

def _f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0

def classification_report(y_true, y_pred, labels):
    t, p = np.asarray(y_true), np.asarray(y_pred)
    rows, TP, FP, FN = [], 0, 0, 0
    for c in labels:
        tp = int(((t == c) & (p == c)).sum()); fp = int(((t != c) & (p == c)).sum()); fn = int(((t == c) & (p != c)).sum())
        pr, rc = _div(tp, tp + fp), _div(tp, tp + fn)
        rows.append((pr, rc, _f1(pr, rc), tp + fn))
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    mp, mr = _div(TP, TP + FP), _div(TP, TP + FN)
    arr = np.array([r[:3] for r in rows], dtype=float)
    sup = np.array([r[3] for r in rows], dtype=float)
    macro = tuple(float(v) for v in arr.mean(axis=0))
    weighted = tuple(float(v) for v in (arr * sup[:, None]).sum(axis=0) / sup.sum()) if sup.sum() else (0.0, 0.0, 0.0)
    return {"accuracy": float((t == p).mean()), "per_class": rows, "micro": (mp, mr, _f1(mp, mr)),
            "macro": macro, "weighted": weighted}
'''

HINTS = [
    'For each class, count TP (true c, predicted c), FP (predicted c, true other) and FN (true c, predicted other).',
    'Micro averaging adds up the counts first and then divides; macro averaging divides per class and then averages.',
    'Guard every division: a class never predicted has precision 0/0, which the statement defines as 0.',
]

TESTS = [
    (np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), [0, 1]),
    (np.array([0, 1, 2, 2, 1, 0]), np.array([0, 2, 2, 2, 1, 1]), [0, 1, 2]),
    (np.array([1, 1, 1]), np.array([0, 0, 0]), [0, 1]),
    (np.array([0, 0, 1, 2]), np.array([0, 0, 1, 2]), [0, 1, 2]),
    (np.array([0, 1, 2, 3, 1, 2]), np.array([1, 1, 2, 0, 3, 2]), [1, 2]),
]

EXPLANATION = r"""
## Micro versus macro
Micro averaging pools all decisions, so frequent classes dominate. In single-label multiclass problems where every class is included, micro precision, recall and F1 all equal accuracy. Macro averaging gives each class equal weight, so poor performance on a rare class shows up clearly. Weighted averaging sits in between.

## When to use which
- Imbalanced classes where rare ones matter (fraud types, diseases): report macro and per-class numbers.
- A single number for overall correctness: micro or accuracy.

## Zero-division conventions
Libraries differ in how they treat a class that is never predicted. Defining 0/0 as 0 penalises the model for never predicting that class, which is usually what you want; state the convention when you report numbers.

## Interview follow-ups
- Why can macro F1 differ from the F1 of macro precision and macro recall?
"""
