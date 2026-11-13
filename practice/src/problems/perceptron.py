import numpy as np

META = dict(title='Perceptron until a clean epoch', tags=['Neural Networks', 'Classification'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:05.01'], related=['averaged-perceptron', 'multi-layer-perceptron-forward-pass'])

STATEMENT = r"""
The perceptron is a single neuron with a step activation, $\hat y = \operatorname{sign}(w^\top x + b)$. It learns by nudging the weights toward every example it gets wrong, and on linearly separable data it is guaranteed to stop making mistakes after finitely many updates.

Write `perceptron(X, y, lr=1.0, max_epochs=100)` that trains it and returns `(w, b, epochs)`.

## Procedure (follow exactly)
1. Start from `w = 0` (shape `(d,)`) and `b = 0`.
2. An epoch visits the samples in their given order $i = 0, 1, \dots, n-1$ (no shuffling). For each sample, if $y_i\,(w^\top x_i + b) \le 0$ it is a **mistake** (a margin of exactly 0 counts) and you update immediately:
$$w \leftarrow w + \eta\, y_i\, x_i, \qquad b \leftarrow b + \eta\, y_i.$$
3. Stop after the first epoch in which no mistake was made, or after `max_epochs` epochs, whichever comes first.

## Input / output
- `X`: float `(n, d)`; `y`: int `(n,)` with values in $\{-1, +1\}$; `lr` $= \eta > 0$; `max_epochs >= 1`.
- Return `w` as a float array `(d,)`, `b` as a Python `float`, and `epochs`, the number of epochs actually run, as an `int` (the clean epoch counts).
"""

STARTER = '''import numpy as np

def perceptron(X, y, lr=1.0, max_epochs=100):
    """Return (w (d,), b float, epochs int) after training until a mistake-free epoch."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def perceptron(X, y, lr=1.0, max_epochs=100):
    X = np.asarray(X, dtype=float)
    w, b = np.zeros(X.shape[1]), 0.0
    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for xi, yi in zip(X, y):
            if yi * (xi @ w + b) <= 0:          # wrong side, or exactly on the boundary
                w = w + lr * yi * xi
                b += lr * yi
                mistakes += 1
        if mistakes == 0:
            return w, float(b), epoch
    return w, float(b), max_epochs
'''

HINTS = [
    'Two nested loops: epochs outside, samples in order inside. Count mistakes per epoch.',
    'A mistake is `y[i] * (X[i] @ w + b) <= 0`. With `w = 0` and `b = 0` the very first sample is always a mistake, which is what gets training started.',
    'Return as soon as an epoch finishes with zero mistakes, reporting that epoch number (1-based). If you fall out of the loop, return `max_epochs`.',
]

_r = np.random.default_rng(11)
_X = np.round(_r.uniform(-3, 3, size=(30, 2)), 1)
_X = _X[np.abs(_X[:, 0] + 2 * _X[:, 1] - 0.5) > 0.4]          # keep a margin so it converges
_y = np.where(_X[:, 0] + 2 * _X[:, 1] - 0.5 > 0, 1, -1)
_X5 = np.round(_r.normal(size=(40, 5)), 2)
_y5 = np.where(_X5 @ np.array([1.0, -2.0, 0.5, 0.0, 1.5]) > 0, 1, -1)
TESTS = [
    (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), np.array([-1, -1, -1, 1])),
    (np.array([[2.0], [-1.0]]), np.array([1, -1])),
    (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), np.array([-1, 1, 1, -1]), 1.0, 7),
    (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), np.array([-1, -1, -1, 1]), 0.5),
    (np.array([[1.0, 1.0]]), np.array([-1])),
    (_X, _y, 0.1),
    (_X5, _y5, 1.0, 50),
]

EXPLANATION = r"""
## Why the update helps
After a mistake on $(x_i, y_i)$, the new margin is $y_i(w^\top x_i + b) + \eta(\lVert x_i\rVert^2 + 1)$: strictly larger than before. One update may not fix the example, but it always moves in the right direction for it.

## Convergence
If some unit vector separates the data with margin $\gamma$ and every point has $\lVert (x, 1)\rVert \le R$, Novikoff's theorem bounds the number of updates by $(R/\gamma)^2$, independent of $n$ and of the learning rate. Starting from zero, the learning rate only rescales $w$ and $b$ together, so it never changes the predictions (compare tests 1 and 4). On XOR (test 3) no separator exists and training runs until `max_epochs`.

## Complexity
$O(n d)$ per epoch, $O(d)$ memory.

## Common bugs
- Using `< 0` instead of `<= 0`: with $w = 0$ every margin is 0, so nothing ever updates.
- Updating with the whole batch at the end of an epoch; the classic perceptron updates online, sample by sample.
- Counting epochs from 0 or not counting the final clean epoch.

## Interview follow-ups
- How does a perceptron relate to logistic regression trained by SGD? (Same linear model; the perceptron uses the step function, so its loss $\max(0, -y\,w^\top x)$ has zero gradient once a point is correct.)
- Why can't one perceptron learn XOR, and what is the smallest network that can? (Two hidden units and an output unit.)
"""
