import numpy as np

META = dict(title='Backward pass of a softmax classifier with smoothing and class weights', tags=['Loss Functions', 'Calculus', 'Classification'],
            diff='Hard', runtime='numpy', lessons=['air-fanout-v1:02.02', 'air-fanout-v1:01.09'],
            related=['cross-entropy-logits', 'softmax-jacobian', 'softmax-numerical-stability-and-log-sum-exp'])

STATEMENT = r"""
A linear softmax classifier computes logits $Z = XW + \mathbf 1 b^\top$ for inputs $X \in \mathbb R^{N\times d}$, weights $W \in \mathbb R^{d\times C}$ and bias $b \in \mathbb R^C$. Train it with two common tweaks:

- **Label smoothing** $\varepsilon$: the target for sample $n$ is $q_n = (1-\varepsilon)\,\mathrm{onehot}(y_n) + \varepsilon/C$.
- **Class weights** $\alpha \in \mathbb R^C$: sample $n$ gets weight $a_n = \alpha_{y_n}$, and the loss is the weighted mean

$$L = \frac{\sum_n a_n \ell_n}{\sum_n a_n}, \qquad \ell_n = -\sum_c q_{nc}\,\log \mathrm{softmax}(z_n)_c$$

Write `softmax_xent_backward(X, W, b, y, eps, class_weight)` that returns `(loss, dX, dW, db)`: the loss as a Python `float` and the gradients of $L$ with shapes `(N, d)`, `(d, C)` and `(C,)`.

## Inputs
- `X`: `(N, d)` floats, `W`: `(d, C)`, `b`: `(C,)`, `y`: int array `(N,)` with values in `[0, C)`.
- `eps` in $[0, 1)$. `class_weight`: `None` (all ones) or a positive float array `(C,)`.
- Compute the log-softmax stably: tests include logits around $\pm 1000$.
- Vectorise; no loops. Derive $\partial L / \partial Z$ by hand rather than building a softmax Jacobian.
"""

STARTER = '''import numpy as np

def softmax_xent_backward(X, W, b, y, eps, class_weight):
    """Return (loss, dL/dX, dL/dW, dL/db) for a weighted, label-smoothed softmax classifier."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def softmax_xent_backward(X, W, b, y, eps, class_weight):
    X, W, b, y = np.asarray(X, dtype=float), np.asarray(W, dtype=float), np.asarray(b, dtype=float), np.asarray(y)
    N, C = len(y), W.shape[1]
    Z = X @ W + b
    shifted = Z - Z.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))   # stable log-softmax
    q = np.full((N, C), eps / C)
    q[np.arange(N), y] += 1 - eps                                          # smoothed targets
    a = np.ones(N) if class_weight is None else np.asarray(class_weight, dtype=float)[y]
    a = a / a.sum()                                                        # normalised sample weights
    loss = -np.sum(a * np.sum(q * logp, axis=1))
    dZ = a[:, None] * (np.exp(logp) - q)                                   # softmax - target, per sample weight
    return float(loss), dZ @ W.T, X.T @ dZ, dZ.sum(axis=0)
'''

HINTS = [
    'Get the gradient with respect to the logits first. For one sample, $\\partial \\ell_n / \\partial z_n = \\mathrm{softmax}(z_n) - q_n$, because each $q_n$ sums to 1. The class weights and the normaliser just scale row $n$ by $a_n / \\sum_m a_m$.',
    'Then it\'s a linear layer: with $G = \\partial L/\\partial Z$, $\\partial L/\\partial W = X^\\top G$, $\\partial L / \\partial b = G^\\top \\mathbf 1$ and $\\partial L/\\partial X = G W^\\top$.',
    'Stable log-softmax: `s = Z - Z.max(1, keepdims=True)`, `logp = s - log(exp(s).sum(1, keepdims=True))`. Build `q` with `np.full((N, C), eps / C)` and add `1 - eps` at `[arange(N), y]`.',
]

_r = np.random.default_rng(67)
TESTS = [
    (np.array([[1.0, 0.0]]), np.array([[1.0, -1.0], [0.0, 0.0]]), np.zeros(2), np.array([0]), 0.0, None),
    (np.array([[1.0, 2.0], [0.5, -1.0]]), np.eye(2), np.array([0.0, 0.5]), np.array([1, 0]), 0.1, None),
    (np.array([[1.0, 2.0], [0.5, -1.0], [0.0, 1.0]]), np.ones((2, 3)), np.zeros(3), np.array([0, 2, 2]), 0.0,
     np.array([1.0, 1.0, 3.0])),
    (np.array([[100.0], [-100.0]]), np.array([[10.0, -10.0]]), np.zeros(2), np.array([1, 1]), 0.0, None),
    (np.zeros((3, 2)), np.zeros((2, 4)), np.zeros(4), np.array([0, 1, 3]), 0.2, np.array([0.5, 1.0, 1.0, 2.0])),
    (_r.normal(size=(8, 5)), _r.normal(size=(5, 4)), _r.normal(size=4), _r.integers(0, 4, size=8), 0.1, None),
    (_r.normal(size=(12, 3)), _r.normal(size=(3, 6)) * 2, _r.normal(size=6), _r.integers(0, 6, size=12), 0.05,
     _r.uniform(0.5, 2.0, size=6)),
]

EXPLANATION = r"""
## The key identity
$\ell = -\sum_c q_c \log s_c$ with $\log s_c = z_c - \mathrm{LSE}(z)$, so
$$\frac{\partial \ell}{\partial z_j} = -q_j + \Bigl(\sum_c q_c\Bigr) s_j = s_j - q_j$$
using $\sum_c q_c = 1$. The softmax Jacobian $\mathrm{diag}(s) - ss^\top$ never needs to be built: the cross-entropy cancels everything but a subtraction. That's why frameworks fuse softmax and cross-entropy into one op.

## Smoothing and weights
- Smoothing changes only the target: the gradient stops pushing the true-class probability all the way to 1 and settles at $1 - \varepsilon + \varepsilon/C$, which keeps logits bounded and calibrates confidence.
- Weights scale each row of $\partial L/\partial Z$ by $a_n/\sum_m a_m$. Normalising by the weight sum (not $N$) keeps the loss scale comparable when weights change. Note this is a choice; some libraries divide by $N$ instead.

## Then it's an ordinary linear layer
$Z = XW + b$ gives $\partial L/\partial W = X^\top G$, $\partial L/\partial b = \sum_n G_n$ and $\partial L/\partial X = GW^\top$. The shapes force the transposes.

## Common bugs
- Computing `np.log(softmax(Z))`: at logit gaps around 2000 (test 4) the softmax underflows to 0 and the log is $-\infty$.
- Spreading $\varepsilon$ over the $C - 1$ wrong classes only. That's a real variant, but not the one specified here, so every gradient entry comes out slightly off.
- Dividing by $N$ and by $\sum a_n$, or applying class weights to the target instead of the sample.

## Interview follow-ups
- What are the logits at the optimum for a single sample with smoothing $\varepsilon$? (The gaps are $\log\frac{1-\varepsilon+\varepsilon/C}{\varepsilon/C}$, finite.)
- How would you add an `ignore_index`, and how should it interact with the weight normaliser?
"""
