import numpy as np

META = dict(title='Gaussian naive Bayes posteriors', tags=['Classification', 'Probability'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:10.01', 'mlmaths:12.02'], related=['linear-discriminant-analysis', 'softmax-rows'])

STATEMENT = r"""
Gaussian naive Bayes assumes that, within each class, every feature is an independent normal variable. Bayes' rule then gives

$$P(c \mid x) \propto \pi_c \prod_{j=1}^{d} \mathcal N\!\left(x_j;\ \mu_{cj},\ \sigma^2_{cj}\right).$$

Write `gaussian_nb(X_train, y_train, X_test, var_eps)` that fits the model and returns `(classes, proba)`.

## Fitting
- `classes`: the distinct labels of `y_train`, sorted ascending.
- Prior $\pi_c$: fraction of training rows with label $c$.
- $\mu_{cj}$: mean of feature $j$ over class $c$.
- $\sigma^2_{cj}$: **population** variance (divide by the class count, `ddof=0`) of feature $j$ over class $c$, **plus `var_eps`**. The epsilon keeps a class with constant features (or one sample) from dividing by zero.

## Output
- `classes`: int array `(C,)`.
- `proba`: float array `(m, C)`; `proba[i, k]` is $P(\text{classes}[k] \mid x_i)$. Each row sums to 1.

## Requirements
- Work in log space. Products of many small densities underflow, and the tests include a point far from every class that must still get a valid distribution (no `nan`).
- `X_train`: float `(n, d)`; `y_train`: int `(n,)`; `X_test`: float `(m, d)`; `var_eps > 0`.
"""

STARTER = '''import numpy as np

def gaussian_nb(X_train, y_train, X_test, var_eps):
    """Return (classes sorted ascending, posterior probabilities (m, C))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def gaussian_nb(X_train, y_train, X_test, var_eps):
    X, y, Xt = np.asarray(X_train, float), np.asarray(y_train), np.asarray(X_test, float)
    classes, counts = np.unique(y, return_counts=True)
    mu = np.array([X[y == c].mean(axis=0) for c in classes])                 # (C, d)
    var = np.array([X[y == c].var(axis=0) for c in classes]) + var_eps       # ddof=0, then smoothed
    # log N(x; mu, var) summed over features -> (m, C)
    diff2 = (Xt[:, None, :] - mu[None, :, :]) ** 2
    log_lik = -0.5 * (np.log(2 * np.pi * var)[None] + diff2 / var[None]).sum(axis=-1)
    log_joint = log_lik + np.log(counts / len(y))
    log_joint -= log_joint.max(axis=1, keepdims=True)                        # normalise with log-sum-exp
    p = np.exp(log_joint)
    return classes, p / p.sum(axis=1, keepdims=True)
'''

HINTS = [
    'Fit is just per-class bookkeeping: prior, per-feature mean and per-feature variance. Prediction is where the care goes.',
    'Compute the log joint $\\log\\pi_c + \\sum_j \\log \\mathcal N(x_j;\\mu_{cj},\\sigma^2_{cj})$ for every (test point, class) pair, with $\\log\\mathcal N = -\\tfrac12[\\log(2\\pi\\sigma^2) + (x-\\mu)^2/\\sigma^2]$.',
    'Turn log joints into posteriors with the softmax trick: subtract each row\'s max, exponentiate, divide by the row sum. Broadcasting `Xt[:, None, :] - mu[None]` gives all pairs at once.',
]

_r = np.random.default_rng(29)
_X = np.vstack([_r.normal(0, 1, size=(20, 3)), _r.normal(1.5, 0.5, size=(25, 3)), _r.normal(-1, 2, size=(15, 3))])
_y = np.repeat([2, 5, 9], [20, 25, 15])
TESTS = [
    (np.array([[0.0], [1.0], [4.0], [5.0]]), np.array([0, 0, 1, 1]), np.array([[0.5], [3.0], [4.5]]), 1e-9),
    (np.array([[0.0, 0.0], [0.0, 2.0], [4.0, 4.0], [6.0, 4.0], [5.0, 7.0]]), np.array([1, 1, 2, 2, 2]), np.array([[1.0, 1.0], [5.0, 5.0]]), 1e-9),
    (np.array([[1.0], [1.0], [3.0], [5.0]]), np.array([0, 0, 1, 1]), np.array([[1.0], [2.0], [4.0]]), 0.01),
    (np.array([[0.0], [1.0], [10.0], [11.0]]), np.array([0, 0, 1, 1]), np.array([[1000.0], [-1000.0]]), 1e-9),
    (np.array([[2.0, 3.0], [4.0, 1.0], [3.0, 2.0]]), np.array([7, 7, 7]), np.array([[0.0, 0.0]]), 1e-9),
    (np.array([[0.0], [2.0], [1.0]]), np.array([0, 0, 1]), np.array([[1.0], [0.0]]), 0.1),
    (_X, _y, _r.normal(0, 2, size=(10, 3)), 1e-9),
]

EXPLANATION = r"""
## Why it works
Bayes' rule says $P(c\mid x) = \pi_c\,p(x\mid c) / p(x)$. The "naive" assumption factorises $p(x \mid c)$ into one univariate Gaussian per feature, so fitting reduces to $C \times d$ means and variances, which are the maximum-likelihood estimates (hence `ddof=0`). The denominator $p(x)$ is the same for every class, so normalising the joint over classes gives the posterior.

## Log space
Test 4 queries $x = 1000$ when the classes sit near 0 and 10. Each density is around $e^{-10^5}$, which is exactly 0 in float64, and $0/0$ is `nan`. In log space the joints are large negative numbers whose *difference* is perfectly representable, and subtracting the row max before exponentiating recovers the posterior.

## Variance smoothing
A class whose feature is constant (test 3, class 0) has $\sigma^2 = 0$, and the density is a spike. Adding `var_eps` keeps it finite. Libraries scale the epsilon by the largest feature variance so it's unit-aware; here it's a fixed number for simplicity.

## Complexity
Fit is $O(nd)$; prediction is $O(mCd)$.

## Common bugs
- Using `ddof=1` (sample variance). It's not the MLE and breaks single-sample classes.
- Forgetting the $\log\sigma^2$ term, which turns the score into a scaled distance and ignores how wide each class is.
- Normalising densities directly instead of log densities.

## Interview follow-ups
- Naive Bayes is often badly calibrated but still ranks well. Why does the independence assumption hurt probabilities more than the argmax?
- How do you handle a mix of categorical and continuous features? (Multiply a categorical likelihood with Laplace smoothing by the Gaussian ones, i.e. add their logs.)
"""
