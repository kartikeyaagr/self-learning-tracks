import numpy as np

META = dict(title='The averaged perceptron', tags=['Classification', 'Classic ML'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:05.01'], related=['svm-with-hinge-loss-sgd', 'logistic-regression'])

STATEMENT = r"""
The perceptron learns a linear rule $\hat y = \mathrm{sign}(w^\top x + b)$ by correcting itself on every mistake. Its final weights depend heavily on the last few examples, so a far more stable answer is the **average** of the weights over the whole run.

Write `averaged_perceptron(X, y, n_epochs)` that returns `(w_avg, b_avg)`.

## Procedure (follow exactly)
1. Start with `w = 0` (shape `(d,)`) and `b = 0`.
2. For each of `n_epochs` epochs, visit the samples in their given order $i = 0, 1, \dots, n-1$ (no shuffling). For each sample:
   - If $y_i\,(w^\top x_i + b) \le 0$ (a mistake, **including** a margin of exactly 0), update $w \leftarrow w + y_i x_i$ and $b \leftarrow b + y_i$.
   - Then, whether or not you updated, add the current `w` and `b` to running totals.
3. Return the totals divided by the number of visits, $T = n \cdot \text{n\_epochs}$.

## Input / output
- `X`: float `(n, d)`; `y`: int `(n,)` with values in $\{-1, +1\}$; `n_epochs >= 1`.
- Return `w_avg` as a float array `(d,)` and `b_avg` as a Python `float`.
"""

STARTER = '''import numpy as np

def averaged_perceptron(X, y, n_epochs):
    """Perceptron (update when y*(w.x+b) <= 0), returning the average (w, b) over all T visits."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def averaged_perceptron(X, y, n_epochs):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    w_sum, b_sum = np.zeros(d), 0.0
    for _ in range(n_epochs):
        for xi, yi in zip(X, y):
            if yi * (xi @ w + b) <= 0:                  # a zero margin counts as a mistake
                w += yi * xi
                b += yi
            w_sum += w                                  # accumulate after every visit, not just on mistakes
            b_sum += b
    T = n * n_epochs
    return w_sum / T, float(b_sum / T)
'''

HINTS = [
    'Write the plain perceptron first. The averaging is two extra running sums.',
    'The totals are updated on **every** visit, mistake or not. A weight vector that survives many correct predictions should count many times.',
    'Use `<= 0` for the mistake test, so the very first sample (where $w = 0$) always triggers an update. Divide the sums by `n * n_epochs` at the end.',
]

_r = np.random.default_rng(31)
_X = _r.integers(-5, 6, size=(30, 3)).astype(float)
_y = np.where(_X @ np.array([2.0, -1.0, 1.0]) + 1.0 > 0, 1, -1)
_Xn = _r.integers(-4, 5, size=(25, 2)).astype(float)
_yn = _r.choice([-1, 1], size=25)
TESTS = [
    (np.array([[1.0, 1.0], [-1.0, -1.0]]), np.array([1, -1]), 1),
    (np.array([[2.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-2.0, -1.0]]), np.array([1, 1, -1, -1]), 3),
    (np.array([[0.0, 0.0]]), np.array([1]), 2),
    (np.array([[1.0], [2.0], [3.0]]), np.array([-1, -1, -1]), 4),
    (np.array([[1.0], [-1.0], [1.0], [-1.0]]), np.array([1, 1, -1, -1]), 5),
    (_X, _y, 5),
    (_Xn, _yn, 3),
    (_X * 100.0, _y, 2),
]

EXPLANATION = r"""
## Why the perceptron converges
If the data are separable with margin $\gamma$ by a unit vector $u$, and $\lVert x_i\rVert \le R$, each mistake increases $w^\top u$ by at least $\gamma$ while increasing $\lVert w\rVert^2$ by at most $R^2$. Together these bound the number of mistakes by $(R/\gamma)^2$, independent of $n$ and $d$ (Novikoff's theorem). The bias is handled by thinking of it as a weight on a constant feature.

## Why average
On non-separable data (test 7 has random labels) the plain perceptron never settles. Its final $w$ is whatever the last few mistakes made it. The average weights every intermediate hypothesis by how long it survived. That's an approximation of the voted perceptron, and it generalises much better in practice.

## Efficient averaging
Adding `w` to a sum on every visit is $O(d)$ per visit even when nothing changes. The standard trick keeps a second accumulator $u \mathrel{+}= t\,y_i x_i$ on mistakes only (with $t$ the visit counter) and returns $w - u/T$. That makes the cost proportional to the number of updates, which matters for sparse NLP features.

## Common bugs
- Testing `< 0` instead of `<= 0`, so the zero-initialised model never makes its first update.
- Accumulating only on mistakes, or dividing by the number of mistakes instead of $T$.
- Shuffling the data, which changes the answer.

## Interview follow-ups
- Derive the $w - u/T$ trick and check it on test 2 by hand.
- How is the perceptron related to SGD on the hinge loss $\max(0, -y\,w^\top x)$, and what changes when the hinge is moved to $\max(0, 1 - y\,w^\top x)$?
"""
