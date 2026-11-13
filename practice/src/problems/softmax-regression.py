import numpy as np

META = dict(title='Multinomial logistic (softmax) regression', tags=['Classification', 'Optimization'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:02.02'], related=['softmax-rows', 'cross-entropy-logits', 'logistic-regression'])

STATEMENT = r"""
Softmax regression generalises logistic regression to $C$ classes. With weights $W \in \mathbb R^{d\times C}$ and biases $b \in \mathbb R^C$, the class probabilities are $P = \mathrm{softmax}(XW + b)$ row-wise, and the training objective is

$$L(W, b) = -\frac{1}{n}\sum_{i=1}^n \log P_{i, y_i} \;+\; \frac{\lambda}{2}\lVert W\rVert_F^2 .$$

The biases are not regularised.

Write `train_softmax_regression(X, y, n_classes, lr, n_steps, reg)` that runs **full-batch gradient descent** and returns `(W, b)`.

## Procedure (follow exactly)
1. Start from `W = 0` with shape `(d, n_classes)` and `b = 0` with shape `(n_classes,)`.
2. Repeat `n_steps` times: compute $\nabla_W L$ and $\nabla_b L$ at the current parameters, then update both: `W -= lr * grad_W`, `b -= lr * grad_b`.

## Input / output
- `X`: float `(n, d)`; `y`: int `(n,)` with values in `[0, n_classes)`. A class may have no samples at all.
- `lr > 0`, `n_steps >= 0`, `reg` ($\lambda$) `>= 0`.
- Return float arrays `W` `(d, n_classes)` and `b` `(n_classes,)`.
- Your softmax must be stable for large logits. Vectorise over samples and classes; the only loop is over steps.
"""

STARTER = '''import numpy as np

def train_softmax_regression(X, y, n_classes, lr, n_steps, reg):
    """Full-batch GD on mean cross-entropy + (reg/2)||W||^2 from zeros. Return (W (d, C), b (C,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def train_softmax_regression(X, y, n_classes, lr, n_steps, reg):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    n, d = X.shape
    W, b = np.zeros((d, n_classes)), np.zeros(n_classes)
    Y = np.eye(n_classes)[y]                            # one-hot targets (n, C)
    for _ in range(n_steps):
        Z = X @ W + b
        Z -= Z.max(axis=1, keepdims=True)               # stable softmax
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        G = (P - Y) / n                                 # dL/dZ
        W, b = W - lr * (X.T @ G + reg * W), b - lr * G.sum(axis=0)
    return W, b
'''

HINTS = [
    'The gradient of mean cross-entropy with respect to the logit matrix $Z = XW + b$ is "probabilities minus one-hot targets", divided by $n$.',
    'With $G = (P - Y)/n$: $\\nabla_W L = X^\\top G + \\lambda W$ and $\\nabla_b L = \\sum_i G_i$ (a column sum).',
    'Build the one-hot matrix with `np.eye(n_classes)[y]`, subtract the row max before `np.exp`, and update `W` and `b` from the same `G`.',
]

_r = np.random.default_rng(13)
_X = _r.normal(size=(60, 4))
_y = (_X @ _r.normal(size=(4, 3))).argmax(axis=1)
TESTS = [
    (np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), np.array([0, 1, 2]), 3, 0.5, 1, 0.0),
    (np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), np.array([0, 1, 2]), 3, 0.5, 100, 0.01),
    (np.array([[1.0], [2.0]]), np.array([0, 0]), 2, 1.0, 0, 0.0),
    (np.array([[1.0], [-1.0], [2.0]]), np.array([0, 2, 0]), 4, 0.3, 40, 0.0),
    (np.array([[50.0, -40.0], [-60.0, 30.0], [45.0, 45.0]]), np.array([0, 1, 2]), 3, 1.0, 25, 0.0),
    (_X, _y, 3, 0.5, 150, 0.0),
    (_X, _y, 3, 0.5, 150, 0.5),
]

EXPLANATION = r"""
## Why the gradient is P - Y
For one sample, $\ell = \mathrm{logsumexp}(z) - z_y$. Differentiating, $\partial\ell/\partial z_c = \mathrm{softmax}(z)_c - [c = y]$. Averaging over samples gives $G = (P - Y)/n$, and the chain rule through $Z = XW + \mathbf 1 b^\top$ gives $X^\top G$ for $W$ and the column sums of $G$ for $b$.

## Over-parameterisation
Adding the same vector to every column of $W$ (and the same constant to every $b_c$) doesn't change any probability. Without regularisation the optimum is a ridge of equivalent solutions, not a point. Starting from zero, plain gradient descent stays in the subspace where the columns of $W$ sum to zero, because each column of $P - Y$ sums to zero across classes. L2 regularisation also picks out a unique solution.

## Empty classes
A class with no samples (test 4) only ever gets negative gradient on its bias, so its probability is pushed toward 0. That's correct behaviour, and it falls out of the same formula with no special case.

## Complexity
$O(ndC)$ per step for the two matrix products.

## Common bugs
- Taking `exp` of raw logits. Test 5 has logits in the thousands after a few steps.
- Using the regularised `W` from this step inside the same step's gradient, or regularising `b`.
- Building `Y` with a loop, or indexing `P[y]` instead of `P[np.arange(n), y]` when computing the loss.

## Interview follow-ups
- Show that with $C = 2$ this is equivalent to logistic regression with $w = W_{:,1} - W_{:,0}$.
- Why is softmax regression preferred over $C$ independent one-vs-rest logistic models? (Probabilities are calibrated to sum to 1, and a single convex problem is solved jointly.)
"""
