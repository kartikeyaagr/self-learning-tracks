import numpy as np

META = dict(title='Backprop by hand: tanh hidden layer, softmax output', tags=['Neural Networks', 'Loss Functions'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:01.08', 'air-fanout-v1:01.07'],
            related=['multi-layer-perceptron-forward-pass', 'softmax-cross-entropy-gradient', 'finite-difference-gradient-checking'])

STATEMENT = r"""
Derive and implement the gradients of a one-hidden-layer classifier:

$$H = \tanh(X W_1 + b_1), \qquad S = H W_2 + b_2, \qquad P = \operatorname{softmax}(S), \qquad L = -\frac1N \sum_{i=1}^N \log P_{i, y_i} + \frac{\lambda}{2}\left(\lVert W_1\rVert_F^2 + \lVert W_2\rVert_F^2\right).$$

Write `grads(X, y, W1, b1, W2, b2, lam=0.0)` that returns `(loss, dW1, db1, dW2, db2)`.

## Shapes
- `X`: float `(N, D)`; `y`: int `(N,)` with labels in `0..K-1`.
- `W1`: `(D, H)`, `b1`: `(H,)`, `W2`: `(H, K)`, `b2`: `(K,)`.
- The L2 penalty applies to the weight matrices only, never to the biases.

## Output
`loss` as a Python `float`; each gradient is a float array with the shape of its parameter. Compute the softmax and the log-probabilities stably (logits in the tests reach the hundreds).
"""

STARTER = '''import numpy as np

def grads(X, y, W1, b1, W2, b2, lam=0.0):
    """Return (loss, dW1, db1, dW2, db2) for the tanh -> softmax network."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def grads(X, y, W1, b1, W2, b2, lam=0.0):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    # forward
    H = np.tanh(X @ W1 + b1)
    S = H @ W2 + b2
    S = S - S.max(axis=1, keepdims=True)                  # stable log-softmax
    logP = S - np.log(np.exp(S).sum(axis=1, keepdims=True))
    loss = -logP[np.arange(N), y].mean() + 0.5 * lam * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    # backward
    dS = np.exp(logP)
    dS[np.arange(N), y] -= 1.0
    dS /= N                                               # dL/dS = (P - onehot) / N
    dW2 = H.T @ dS + lam * W2
    db2 = dS.sum(axis=0)
    dZ = (dS @ W2.T) * (1.0 - H ** 2)                     # tanh'(z) = 1 - tanh(z)^2
    dW1 = X.T @ dZ + lam * W1
    db1 = dZ.sum(axis=0)
    return float(loss), dW1, db1, dW2, db2
'''

HINTS = [
    'Go backwards one node at a time: logits $S$, then $W_2, b_2$, then the hidden activations $H$, then the pre-activation, then $W_1, b_1$.',
    'For softmax plus cross-entropy, $\\partial L/\\partial S = (P - Y_{\\text{onehot}})/N$. Then $dW_2 = H^\\top dS$, $db_2 = \\sum_i dS_i$, $dH = dS\\,W_2^\\top$.',
    'Through tanh: $dZ = dH \\odot (1 - H^2)$, then $dW_1 = X^\\top dZ$, $db_1 = \\sum_i dZ_i$. Add $\\lambda W$ to both weight gradients only.',
]

_r = np.random.default_rng(19)
def _params(D, H, K, s=1.0):
    return _r.normal(size=(D, H)) * s, _r.normal(size=H), _r.normal(size=(H, K)) * s, _r.normal(size=K)
TESTS = [
    (np.array([[1.0, 0.0]]), np.array([0]), np.array([[1.0], [0.0]]), np.array([0.0]), np.array([[1.0, -1.0]]), np.array([0.0, 0.0])),
    (np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([1, 0]), np.ones((2, 2)), np.zeros(2), np.eye(2), np.zeros(2), 0.1),
    (_r.normal(size=(6, 4)), _r.integers(0, 3, size=6), *_params(4, 5, 3)),
    (_r.normal(size=(6, 4)), _r.integers(0, 3, size=6), *_params(4, 5, 3), 0.05),
    (_r.normal(size=(5, 3)), np.array([0, 1, 2, 3, 0]), *_params(3, 8, 4, s=40.0)),
    (_r.normal(size=(1, 2)), np.array([1]), *_params(2, 1, 2)),
    (_r.normal(size=(10, 7)), _r.integers(0, 5, size=10), *_params(7, 6, 5), 1.0),
]

EXPLANATION = r"""
## The two facts that do all the work
1. **Softmax + cross-entropy collapse.** The Jacobian of softmax is messy, but composed with $-\log P_y$ the gradient wrt the logits is simply $P - \text{onehot}(y)$. Never backprop through softmax separately.
2. **Affine layers.** For $S = H W + b$: $dW = H^\top dS$, $db = \sum_i dS_i$, $dH = dS\,W^\top$. Check shapes: $dW$ must match $W$, which pins the transpose.

Through $\tanh$, use the cached output: $\tanh'(z) = 1 - \tanh^2(z)$, so no need to keep $z$.

## Verifying
Always confirm with central finite differences on a tiny network: perturb one weight by $\pm 10^{-5}$ and compare $(L_+ - L_-)/(2h)$ to your gradient; relative error near $10^{-7}$ means correct.

## Complexity
Backward costs about twice the forward: $O(N(DH + HK))$.

## Common bugs
- Forgetting the $1/N$ (the loss is a mean), or applying it twice.
- Regularising the biases, or forgetting the $\lambda W$ term in the gradient while adding it to the loss.
- Computing `np.log(softmax(S))`, which gives `-inf` for confident wrong predictions (test 5 has logits around $\pm 100$).

## Interview follow-ups
- What changes if the hidden layer is ReLU? Where can it kill gradients?
- Why does the bias gradient of a layer equal the column sum of the upstream gradient?
"""
