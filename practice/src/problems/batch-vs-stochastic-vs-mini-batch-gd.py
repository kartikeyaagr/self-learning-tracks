import numpy as np

META = dict(title='Batch, stochastic and mini-batch gradient descent', tags=['Optimization', 'Regression', 'Training'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:08.02', 'air-fanout-v1:12.01'],
            related=['vanilla-gradient-descent', 'gradient-accumulation'])

STATEMENT = r"""
Fit linear regression $\hat y = X w$ (no intercept) by minimising the mean squared error

$$L(w) = \frac1n \lVert Xw - y\rVert^2$$

with one routine that covers all three classic variants through `batch_size`: `n` gives full-batch GD, `1` gives SGD, anything between gives mini-batch GD.

Write `minibatch_gd(X, y, lr, epochs, batch_size, seed)` that returns the tuple `(w, losses)`.

## Exact procedure (follow it so your random draws match)
1. `rng = np.random.default_rng(seed)`; start from `w = np.zeros(d)`.
2. For each epoch: `perm = rng.permutation(n)`. Walk through `perm` in consecutive chunks of `batch_size`; the last chunk may be smaller.
3. For a chunk with index set $B$, take one step with the gradient of the **mean** loss over that chunk:
$$w \leftarrow w - \eta \cdot \frac{2}{|B|} X_B^\top (X_B w - y_B).$$
4. After the epoch's last step, append the **full-data** loss $L(w)$ to `losses`.

## Output
- `w`: float array `(d,)`. `losses`: float array `(epochs,)`.
- `X`: float `(n, d)`, `y`: float `(n,)`, `n >= 1`. `batch_size >= 1`; values above `n` behave like `n`. `epochs >= 1`.
"""

STARTER = '''import numpy as np

def minibatch_gd(X, y, lr, epochs, batch_size, seed):
    """Return (w (d,), losses (epochs,)) following the exact shuffling procedure in the statement."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def minibatch_gd(X, y, lr, epochs, batch_size, seed):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w = np.zeros(d)
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(n)                     # one fresh shuffle per epoch, even for full batch
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]              # last chunk may be short
            Xb, yb = X[idx], y[idx]
            w = w - lr * (2.0 / len(idx)) * (Xb.T @ (Xb @ w - yb))
        losses.append(np.mean((X @ w - y) ** 2))
    return w, np.array(losses)
'''

HINTS = [
    'All three variants are the same loop; only the chunk size changes. Shuffle with `rng.permutation(n)` once per epoch.',
    'Iterate `for s in range(0, n, batch_size)` and slice `perm[s:s + batch_size]`. Python slicing handles the short last chunk.',
    'Divide the chunk gradient by `len(idx)`, not by `batch_size`, so a short last chunk still uses the mean. Record the loss on **all** of `X` after the epoch.',
]

_r = np.random.default_rng(42)
_X = _r.normal(size=(30, 3))
_y = _X @ np.array([1.5, -2.0, 0.5]) + 0.1 * _r.normal(size=30)
TESTS = [
    (np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([2.0, 4.0, 6.0, 8.0]), 0.05, 3, 4, 0),
    (np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([2.0, 4.0, 6.0, 8.0]), 0.05, 3, 1, 0),
    (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([1.0, 2.0, 3.0]), 0.1, 5, 2, 7),
    (_X, _y, 0.05, 10, 30, 1),
    (_X, _y, 0.01, 10, 1, 1),
    (_X, _y, 0.05, 10, 8, 2),
    (_X, _y, 0.05, 2, 1000, 3),
    (np.array([[2.0]]), np.array([3.0]), 0.1, 4, 1, 9),
]

EXPLANATION = r"""
## One estimator, three noise levels
The chunk gradient $\frac{2}{|B|}X_B^\top(X_B w - y_B)$ is an **unbiased** estimate of the full gradient when $B$ is a uniformly random subset. Its variance scales like $1/|B|$:
- **Batch** ($|B| = n$): exact gradient, one step per epoch, smooth but slow per epoch.
- **SGD** ($|B| = 1$): $n$ noisy steps per epoch. Lots of progress early, then it jitters around the optimum unless the learning rate decays.
- **Mini-batch**: the practical middle. Big enough to use vectorised hardware, small enough to take many steps.

## Why reshuffle every epoch
Sampling without replacement within an epoch visits every example once and empirically converges faster than sampling with replacement. A fixed order can create cyclic bias: the last chunks always "win".

## Complexity
Each epoch costs $O(nd)$ regardless of batch size; what changes is the number of updates ($\lceil n/B\rceil$) and the noise in each.

## Common bugs
- Dividing by `batch_size` for the short last chunk, which shrinks its step.
- Shuffling `X` and `y` with two separate permutations, which scrambles the pairs.
- Reporting the loss of the last mini-batch instead of the full dataset, so the curve is noisy by construction.

## Interview follow-ups
- If you double the batch size, how should you scale the learning rate? (The linear scaling rule says double it too, at least until the "critical batch size", with warmup to stay stable.)
- Why does SGD with a constant learning rate not converge to the exact minimum? (The gradient noise never vanishes, so the iterate keeps bouncing in a ball whose radius scales with $\eta$.)
"""
