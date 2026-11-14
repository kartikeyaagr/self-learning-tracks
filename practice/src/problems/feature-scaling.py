import numpy as np

META = dict(title='Standard, min-max and robust scaling', tags=['Feature Engineering', 'Data Processing'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:10.15'], related=['missing-value-imputation', 'knn-classifier'])

STATEMENT = r"""
Write `scale(X_train, X_test, method)` that learns a per-column shift $c_j$ and scale $s_j$ from `X_train`, then returns $(X - c)/s$ for both arrays.

| `method` | shift $c_j$ | scale $s_j$ |
| --- | --- | --- |
| `'standard'` | mean | standard deviation with `ddof=0` |
| `'minmax'` | minimum | maximum minus minimum |
| `'robust'` | median | interquartile range $Q_{75} - Q_{25}$ |

## Rules
- Quantiles use NumPy's default linear interpolation (`np.percentile(col, [25, 75])`).
- If a column's scale is 0 (for example a constant column), use $s_j = 1$ for it, so that column is just shifted.
- Statistics come from `X_train` only; apply the same $c$ and $s$ to `X_test`. Test values may land outside $[0, 1]$ for `minmax`; that's expected.

## Input / output
`X_train`: float `(n, d)` with $n \ge 1$; `X_test`: float `(m, d)`. Return `(train_scaled, test_scaled)` as float arrays of the input shapes. Vectorise over columns.
"""

STARTER = '''import numpy as np

def scale(X_train, X_test, method):
    """Fit per-column shift/scale on X_train ('standard', 'minmax' or 'robust'); return (train_scaled, test_scaled)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def scale(X_train, X_test, method):
    X, Xt = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
    if method == 'standard':
        c, s = X.mean(axis=0), X.std(axis=0)
    elif method == 'minmax':
        c, s = X.min(axis=0), X.max(axis=0) - X.min(axis=0)
    else:
        q25, q75 = np.percentile(X, [25, 75], axis=0)
        c, s = np.median(X, axis=0), q75 - q25
    s = np.where(s == 0, 1.0, s)                           # constant columns: shift only
    return (X - c) / s, (Xt - c) / s
'''

HINTS = [
    'Each method is just a choice of two length-$d$ vectors. Compute them with `axis=0` reductions on the training data.',
    '`np.std` defaults to `ddof=0`, which is what\'s asked. `np.percentile(X, [25, 75], axis=0)` returns both quartile rows at once.',
    'Guard the division with `np.where(s == 0, 1.0, s)`, then apply the same `(X - c) / s` to train and test.',
]

_r = np.random.default_rng(149)
_X = np.column_stack([_r.normal(50, 10, size=30), _r.exponential(2.0, size=30), np.full(30, 7.0), _r.integers(0, 3, size=30)])
_Xt = np.column_stack([_r.normal(50, 10, size=5), _r.exponential(2.0, size=5), np.full(5, 8.0), _r.integers(0, 3, size=5)])
TESTS = [
    (np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]), np.array([[2.0, 40.0]]), 'standard'),
    (np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]), np.array([[2.0, 40.0]]), 'minmax'),
    (np.array([[1.0], [2.0], [3.0], [4.0], [100.0]]), np.array([[0.0], [50.0]]), 'robust'),
    (np.array([[4.0, 1.0], [4.0, 2.0]]), np.array([[5.0, 0.0]]), 'standard'),
    (np.array([[9.0, -3.0]]), np.array([[10.0, -3.0]]), 'minmax'),
    (np.array([[1e9, -1e-9], [3e9, 1e-9], [2e9, 0.0]]), np.array([[0.0, 2e-9]]), 'standard'),
    (_X, _Xt, 'standard'),
    (_X, _Xt, 'robust'),
]

EXPLANATION = r"""
## Why scale at all
Distance-based models (k-NN, k-means, SVMs with RBF kernels) and penalised models (ridge, lasso) treat all features on the same footing, so a feature measured in thousands would dominate one measured in fractions. Gradient descent also converges faster on well-conditioned, similarly scaled inputs. Tree models are invariant to monotone per-feature transforms and don't need it.

## Which scaler
- **Standard** gives zero mean, unit variance. It's sensitive to outliers, which inflate the standard deviation.
- **Min-max** maps training data into $[0, 1]$. One extreme value squashes everything else into a tiny range.
- **Robust** uses the median and IQR, so the outlier at 100 in test 3 barely moves the other values' scaled positions.

## The zero-scale guard
A constant training column (test 4, and column 3 of the random tests) has zero spread. Dividing by it gives `inf`/`nan`. Using a scale of 1 keeps the column finite; dropping it is also common.

## Common bugs
- Fitting the scaler on the full dataset or on the test set (leakage).
- `ddof=1` instead of `ddof=0`.
- Scaling per row (`axis=1`) instead of per column.

## Interview follow-ups
- Should you scale one-hot encoded columns? What about the target in regression?
- With cross-validation, where exactly must the scaler be fitted? (Inside each fold, on that fold's training part only, which is what a pipeline object guarantees.)
"""
