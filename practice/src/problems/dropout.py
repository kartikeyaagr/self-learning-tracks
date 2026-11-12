import numpy as np

META = dict(title='Inverted dropout in a tiny MLP', tags=['Neural Networks', 'Training'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:05.07', 'air-fanout-v1:01.08'],
            related=['early-stopping', 'gradient-clipping'])

STATEMENT = r"""
Implement the forward and backward pass of a one-hidden-layer regression MLP with **inverted dropout** on the hidden activations:

$$h = XW_1 + b_1,\quad a = \mathrm{ReLU}(h),\quad \tilde a = \frac{a \odot M}{1 - p},\quad \hat y = \tilde a\, W_2 + b_2,\quad L = \frac1n\sum_i (\hat y_i - y_i)^2.$$

$M$ is a 0/1 keep-mask: each hidden unit of each sample is kept with probability $1 - p$. Dividing by $1 - p$ keeps $\mathbb E[\tilde a] = a$, so nothing needs rescaling at test time.

Write `dropout_mlp(X, y, W1, b1, W2, b2, p, seed, train=True)` that returns the tuple `(loss, dW1, db1, dW2, db2)`, the loss as a Python `float` and the gradients of $L$ with the same shapes as the parameters.

## Exact conventions (so your mask matches)
- Shapes: `X` `(n, d)`, `y` `(n,)`, `W1` `(d, H)`, `b1` `(H,)`, `W2` `(H,)`, `b2` a float, so $\hat y$ has shape `(n,)`.
- When `train` is `True` and `p > 0`: `rng = np.random.default_rng(seed)` and `M = rng.random((n, H)) >= p`. Draw nothing else from `rng`.
- When `train` is `False` or `p == 0`: no mask and no scaling ($\tilde a = a$), and no random numbers are drawn.
- $0 \le p < 1$. Use $\mathrm{ReLU}'(h) = 1$ where $h > 0$ and $0$ elsewhere (including $h = 0$).
"""

STARTER = '''import numpy as np

def dropout_mlp(X, y, W1, b1, W2, b2, p, seed, train=True):
    """Return (loss float, dW1 (d,H), db1 (H,), dW2 (H,), db2 float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def dropout_mlp(X, y, W1, b1, W2, b2, p, seed, train=True):
    X, y, W1, b1, W2 = (np.asarray(a, dtype=float) for a in (X, y, W1, b1, W2))
    n = len(X)
    h = X @ W1 + b1
    a = np.maximum(h, 0.0)
    if train and p > 0:
        keep = np.random.default_rng(seed).random(a.shape) >= p
        scale = keep / (1.0 - p)                   # the same factor multiplies forward and backward
    else:
        scale = np.ones_like(a)
    a_t = a * scale
    pred = a_t @ W2 + b2
    r = pred - y
    loss = float(np.mean(r * r))

    d_pred = 2.0 * r / n                           # dL/dpred, shape (n,)
    dW2 = a_t.T @ d_pred
    db2 = float(d_pred.sum())
    d_at = np.outer(d_pred, W2)                    # (n, H)
    d_h = d_at * scale * (h > 0)                   # back through dropout, then ReLU
    return loss, X.T @ d_h, d_h.sum(axis=0), dW2, db2
'''

HINTS = [
    'Draw the mask exactly once, `rng.random((n, H)) >= p`, and turn it into a scale array `mask / (1 - p)`. Forward multiplies by it, and so does backward.',
    'Backward from the output: $\\partial L/\\partial \\hat y = 2(\\hat y - y)/n$. Then $dW_2 = \\tilde a^\\top d\\hat y$, $db_2 = \\sum d\\hat y$, and $d\\tilde a = d\\hat y\\, W_2^\\top$ (an outer product).',
    'Through dropout multiply by the same scale; through ReLU multiply by `(h > 0)`. Then $dW_1 = X^\\top dh$ and $db_1 = \\sum_i dh_i$.',
]

_r = np.random.default_rng(53)
_X = _r.normal(size=(6, 3))
_y = _r.normal(size=6)
_W1, _b1 = _r.normal(size=(3, 5)) * 0.8, _r.normal(size=5) * 0.1
_W2, _b2 = _r.normal(size=5), 0.3
_Xb, _yb = _r.normal(size=(40, 4)), _r.normal(size=40)
_W1b, _b1b, _W2b = _r.normal(size=(4, 16)) * 0.5, np.zeros(16), _r.normal(size=16) * 0.5
TESTS = [
    (np.array([[1.0, 2.0]]), np.array([1.0]), np.array([[1.0, -1.0], [0.5, 1.0]]), np.array([0.0, 0.0]), np.array([1.0, 2.0]), 0.0, 0.0, 0),
    (np.array([[1.0, 2.0]]), np.array([1.0]), np.array([[1.0, -1.0], [0.5, 1.0]]), np.array([0.0, 0.0]), np.array([1.0, 2.0]), 0.0, 0.5, 1),
    (_X, _y, _W1, _b1, _W2, _b2, 0.5, 7),
    (_X, _y, _W1, _b1, _W2, _b2, 0.5, 7, False),
    (_X, _y, _W1, _b1, _W2, _b2, 0.0, 7),
    (_X, _y, _W1, _b1, _W2, _b2, 0.9, 3),
    (_Xb, _yb, _W1b, _b1b, _W2b, -0.1, 0.25, 11),
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.0, 1.0]), np.array([[1.0, 0.0], [0.0, -1.0]]), np.array([-1.0, 0.0]), np.array([1.0, 1.0]), 0.5, 0.3, 2),
]

EXPLANATION = r"""
## Why it regularises
Each forward pass trains a random sub-network, so no hidden unit can rely on a specific partner being present. Units are pushed to be individually useful (less co-adaptation), and the full network at test time behaves like an average over exponentially many thinned networks. For a linear model, dropout on inputs is equivalent in expectation to an adaptive L2 penalty weighted by feature variance.

## Why "inverted"
Original dropout kept activations unscaled during training and multiplied weights by $1 - p$ at test time. Scaling by $1/(1-p)$ during training instead keeps the expected activation unchanged, so inference is a plain forward pass with no special case. That's what every framework does.

## The backward pass
Dropout is element-wise multiplication by a fixed (per pass) array $D = M/(1-p)$, so its Jacobian is diagonal and the backward pass multiplies by the same $D$. The mask must be **saved** from the forward pass, never redrawn. ReLU contributes `(h > 0)`, and a unit that was dropped gets zero gradient for its incoming weights on that sample.

## Complexity
Same order as the MLP without dropout: $O(ndH)$ time, plus $O(nH)$ memory for the mask.

## Common bugs
- Forgetting the $1/(1-p)$ in the backward pass while keeping it in forward.
- Applying dropout when `train=False`, or drawing the mask anyway (which changes RNG state for later code).
- `rng.random(...) < p` as the keep mask: that keeps with probability $p$, the opposite of the convention.

## Interview follow-ups
- Why is dropout used less in modern large transformers? (Single-epoch training on huge data rarely overfits, and dropout slows convergence; it returns for fine-tuning on small data.)
- What does "MC dropout" do at inference time? (Keep dropout on, run several passes, and use the spread of predictions as an uncertainty estimate.)
"""
