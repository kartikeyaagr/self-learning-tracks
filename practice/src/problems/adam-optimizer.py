import numpy as np

META = dict(title='Train logistic regression with Adam', tags=['Optimization', 'Classification', 'Training'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:05.06'],
            related=['adam-step', 'rmsprop', 'adamw-decoupled-weight-decay', 'cross-entropy-logits'])

STATEMENT = r"""
Write a full Adam training loop for binary logistic regression. The model is $p = \sigma(Xw + c)$ with weights $w \in \mathbb{R}^d$ and a scalar bias $c$, trained on the mean binary cross-entropy

$$L(w, c) = \frac1n\sum_{i=1}^n \left[\log\left(1 + e^{z_i}\right) - y_i z_i\right], \qquad z = Xw + c,$$

whose gradient is $\nabla_w L = \frac1n X^\top(\sigma(z) - y)$, $\partial_c L = \frac1n\sum_i (\sigma(z_i) - y_i)$.

Treat $w$ and $c$ as one parameter vector $\theta = [w, c]$ and apply Adam to it, starting from $\theta_0 = 0$, $m_0 = v_0 = 0$. At step $t = 1, 2, \dots$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad \theta_t = \theta_{t-1} - \eta\,\frac{m_t / (1-\beta_1^t)}{\sqrt{v_t / (1-\beta_2^t)} + \epsilon}.$$

Write `adam_logistic(X, y, lr, steps, b1=0.9, b2=0.999, eps=1e-8)` returning the tuple `(w, c, losses)`:
- `w`: float array `(d,)`; `c`: Python `float`.
- `losses`: float array `(steps + 1,)`, the loss at $\theta_0, \theta_1, \dots, \theta_{\text{steps}}$ (so `losses[0]` is $\log 2$).

## Requirements
- `X`: float `(n, d)`; `y`: float `(n,)` of 0s and 1s.
- Compute the loss stably: $\log(1 + e^{z})$ must not overflow for $z = 1000$ (use `np.logaddexp(0, z)`).
"""

STARTER = '''import numpy as np

def adam_logistic(X, y, lr, steps, b1=0.9, b2=0.999, eps=1e-8):
    """Return (w (d,), c float, losses (steps+1,)) after Adam on mean binary cross-entropy."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _loss_grad(theta, Xb, y):
    z = Xb @ theta
    loss = np.mean(np.logaddexp(0.0, z) - y * z)      # log(1 + e^z) without overflow
    p = 0.5 * (1.0 + np.tanh(0.5 * z))                 # sigmoid that never overflows
    return loss, Xb.T @ (p - y) / len(y)

def adam_logistic(X, y, lr, steps, b1=0.9, b2=0.999, eps=1e-8):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    Xb = np.hstack([X, np.ones((len(X), 1))])          # bias as an extra always-1 feature
    theta = np.zeros(Xb.shape[1])
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    loss, g = _loss_grad(theta, Xb, y)
    losses = [loss]
    for t in range(1, steps + 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)   # undo the zero-initialisation bias
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        loss, g = _loss_grad(theta, Xb, y)
        losses.append(loss)
    return theta[:-1], float(theta[-1]), np.array(losses)
'''

HINTS = [
    'Append a column of ones to `X` so the bias is just one more weight. Then Adam runs on a single vector.',
    'Write one helper that returns `(loss, grad)` at a given $\\theta$. The loop calls it once per step: record the loss, use the gradient for the update.',
    'Bias correction uses the 1-based step: `m / (1 - b1**t)`. For the sigmoid, `0.5 * (1 + np.tanh(z / 2))` is exact and overflow-free.',
]

_r = np.random.default_rng(29)
_X = _r.normal(size=(60, 3))
_y = (_X @ np.array([2.0, -1.0, 0.5]) + 0.3 + 0.5 * _r.normal(size=60) > 0).astype(float)
TESTS = [
    (np.array([[1.0], [-1.0], [2.0]]), np.array([1.0, 0.0, 1.0]), 0.1, 3),
    (np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), np.array([0.0, 1.0, 1.0]), 0.5, 2),
    (np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 1.0, 1.0]), 0.1, 0),
    (np.array([[500.0], [-300.0], [200.0]]), np.array([1.0, 0.0, 1.0]), 1.0, 3),
    (np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -1.0], [-2.0, 0.5]]), np.array([1.0, 1.0, 0.0, 0.0]), 0.05, 50, 0.8, 0.99),
    (_X, _y, 0.05, 100),
    (_X, _y, 0.01, 200, 0.9, 0.999, 1e-6),
]

EXPLANATION = r"""
## Why Adam works
Adam combines two ideas. The first moment $m$ is momentum: an exponential average of gradients that smooths noise. The second moment $v$ is RMSProp: an average of squared gradients that normalises each coordinate's step to about $\eta$. Because both averages start at 0, early estimates are biased toward 0 by a factor $1 - \beta^t$; dividing by it makes $\hat m, \hat v$ unbiased from the very first step, which matters most for $\beta_2 = 0.999$ (without correction $v_1$ is 1000x too small, giving a huge first step).

## Scale invariance
Multiplying the loss by a constant leaves the Adam step unchanged (up to $\epsilon$), because $\hat m/\sqrt{\hat v}$ is unitless. Test 4 has features of size 500: plain GD would need a tiny learning rate, while Adam's first step is still about $\eta$ per parameter.

## Complexity
$O(nd)$ per step for the forward and backward pass, $O(d)$ for the update; memory $3d$ for $\theta, m, v$.

## Common bugs
- Starting `t` at 0, which divides by $1 - \beta^0 = 0$.
- Forgetting the bias in the parameter vector, or giving it its own (incorrect) moment updates.
- Computing $\log\sigma(z)$ as `np.log(1 / (1 + np.exp(-z)))`: overflow and `log(0)` on large-magnitude logits.

## Interview follow-ups
- Why does Adam converge to a separating direction with growing norm on separable data (test 4)? What fixes it? (The loss has no finite minimiser; add L2 regularisation or early stopping.)
- What do `amsgrad=True` and Adam's $\epsilon$ actually change in practice? (AMSGrad keeps $\max v_t$ so steps never grow; a large $\epsilon$ blends Adam toward momentum SGD.)
"""
