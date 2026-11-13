import numpy as np

META = dict(title='One SGD step on the policy and value heads', tags=['Training', 'Reinforcement Learning'], diff='Hard',
            runtime='numpy', lessons=['air-fanout-v1:01.08', 'air-fanout-v1:01.07'],
            related=['alphago-policy-value-loss', 'policy-and-value-heads', 'softmax-cross-entropy-gradient'])

STATEMENT = r"""
Here the trunk is frozen and its output for each position is a feature vector $f_i \in \mathbb{R}^D$. Train the two heads on a batch of self-play targets with one step of plain gradient descent, deriving the gradients by hand.

## Model
- Policy: $\ell_i = W_p f_i + b_p \in \mathbb{R}^A$, $\;p_i = \mathrm{softmax}(\ell_i)$.
- Value: $a_i = W_1 f_i + b_1 \in \mathbb{R}^H$, $\;h_i = \mathrm{relu}(a_i)$, $\;v_i = \tanh(w_2 \cdot h_i + b_2)$.

## Loss
$$L = \frac{1}{B}\sum_{i=1}^{B}\Big[(z_i - v_i)^2 - \sum_a \pi_{i,a}\log p_{i,a}\Big] + c\big(\lVert W_p\rVert^2 + \lVert W_1\rVert^2 + \lVert w_2\rVert^2\big)$$

Biases are not regularised. Each row of $\pi$ is non-negative and sums to 1. Use $\mathrm{relu}'(0) = 0$.

## Task
Write `head_train_step(features, target_pi, z, params, lr, c)` that computes $L$ and all gradients at the **current** parameters, then updates every parameter as $\theta \leftarrow \theta - \text{lr}\cdot\partial L/\partial\theta$.

## Inputs
- `features`: float `(B, D)`; `target_pi`: float `(B, A)`; `z`: float `(B,)`.
- `params`: dict with `"Wp"` `(A, D)`, `"bp"` `(A,)`, `"W1"` `(H, D)`, `"b1"` `(H,)`, `"w2"` `(H,)`, `"b2"` (a float). Do not modify it.

## Output
`(new_params, loss)`: a new dict with the same keys and shapes (`"b2"` a Python float) holding the updated values, and `loss`, the Python float $L$ **before** the update.
"""

STARTER = '''import numpy as np

def head_train_step(features, target_pi, z, params, lr, c):
    """Return (new_params dict, loss before the step)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def head_train_step(features, target_pi, z, params, lr, c):
    f = np.asarray(features, dtype=float)
    pi = np.asarray(target_pi, dtype=float)
    z = np.asarray(z, dtype=float)
    Wp, bp, W1, b1, w2 = (np.asarray(params[k], dtype=float) for k in ('Wp', 'bp', 'W1', 'b1', 'w2'))
    b2 = float(params['b2'])
    B = f.shape[0]

    # forward
    logits = f @ Wp.T + bp
    logits = logits - logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    p = np.exp(logp)
    a = f @ W1.T + b1
    h = np.maximum(a, 0.0)
    v = np.tanh(h @ w2 + b2)
    reg = np.sum(Wp ** 2) + np.sum(W1 ** 2) + np.sum(w2 ** 2)
    loss = np.mean((z - v) ** 2) - np.sum(pi * logp) / B + c * reg

    # policy head: d(cross-entropy)/d(logits) = p - pi when each pi row sums to 1
    dlog = (p - pi) / B
    gWp = dlog.T @ f + 2 * c * Wp
    gbp = dlog.sum(axis=0)

    # value head: chain rule through (z - v)^2, tanh, the dot product and the relu
    ds = -2.0 * (z - v) / B * (1.0 - v ** 2)
    gw2 = h.T @ ds + 2 * c * w2
    gb2 = ds.sum()
    da = np.outer(ds, w2) * (a > 0)
    gW1 = da.T @ f + 2 * c * W1
    gb1 = da.sum(axis=0)

    new = {'Wp': Wp - lr * gWp, 'bp': bp - lr * gbp, 'W1': W1 - lr * gW1,
           'b1': b1 - lr * gb1, 'w2': w2 - lr * gw2, 'b2': float(b2 - lr * gb2)}
    return new, float(loss)
'''

HINTS = [
    'Do the forward pass once and keep the intermediates you will need: `p`, `a`, `h`, `v`. The two heads share no parameters, so their gradients are independent.',
    'Policy: the gradient of the mean cross-entropy w.r.t. the logits is `(p - pi) / B`; then `dWp = dlog.T @ f`, `dbp = dlog.sum(0)`. Value: with $s = w_2 \\cdot h + b_2$, `ds = -2 (z - v) (1 - v**2) / B`, then `dw2 = h.T @ ds`, `db2 = ds.sum()`.',
    'Back through the relu: `da = np.outer(ds, w2) * (a > 0)`, `dW1 = da.T @ f`, `db1 = da.sum(0)`. Add `2 * c * W` to the three weight gradients only, then subtract `lr * grad` from every parameter.',
]

def _params(rng, D, A, H):
    return {'Wp': rng.normal(0, 0.3, (A, D)), 'bp': rng.normal(0, 0.1, A),
            'W1': rng.normal(0, 0.3, (H, D)), 'b1': rng.normal(0, 0.1, H),
            'w2': rng.normal(0, 0.3, H), 'b2': float(rng.normal(0, 0.1))}

_p1 = {'Wp': np.zeros((2, 1)), 'bp': np.zeros(2), 'W1': np.array([[1.0]]), 'b1': np.zeros(1),
       'w2': np.array([0.5]), 'b2': 0.0}
_rng = np.random.default_rng(61)
_B, _D, _A, _H = 6, 5, 4, 3
_f = _rng.normal(size=(_B, _D))
_pi = _rng.random((_B, _A)); _pi /= _pi.sum(axis=1, keepdims=True)
_z = _rng.choice([-1.0, 1.0], _B)
_pr = _params(_rng, _D, _A, _H)
_f2 = _rng.normal(size=(8, 10))
_pi2 = np.zeros((8, 7)); _pi2[np.arange(8), _rng.integers(0, 7, 8)] = 1.0
_pr2 = _params(_rng, 10, 7, 5)

TESTS = [
    (np.array([[1.0]]), np.array([[1.0, 0.0]]), np.array([1.0]), _p1, 0.5, 0.0),
    (np.array([[-1.0], [2.0]]), np.array([[0.5, 0.5], [0.0, 1.0]]), np.array([1.0, -1.0]), _p1, 0.1, 0.0),
    (_f, _pi, _z, _pr, 0.1, 0.0),
    (_f, _pi, _z, _pr, 0.1, 1e-2),
    (_f, _pi, _z, _pr, 0.0, 1e-4),
    (_f2, _pi2, _rng.choice([-1.0, 0.0, 1.0], 8), _pr2, 0.05, 1e-3),
]

EXPLANATION = r"""
## The two gradients worth remembering
- **Softmax + cross-entropy**: $\partial(-\sum_a \pi_a \log p_a)/\partial \ell = p - \pi$ when $\sum_a \pi_a = 1$. No Jacobian of the softmax is needed; the terms cancel.
- **tanh + squared error**: with $s = w_2 \cdot h + b_2$, $\partial (z - v)^2 / \partial s = -2(z - v)(1 - v^2)$, because $\tanh' = 1 - \tanh^2$.

Everything else is the linear-layer rule ($\partial L / \partial W = \delta^\top x$, $\partial L / \partial b = \sum \delta$) and the relu mask. The $1/B$ from the batch mean is folded into the upstream deltas once, and the regulariser contributes $2cW$ to each weight gradient.

## Checking your gradients
Perturb one parameter by $\pm 10^{-6}$ and compare the central finite difference of $L$ with your analytic gradient; they should agree to about 6 digits. Test 5 uses `lr = 0`, so the parameters must come back unchanged while the loss is still checked.

## Complexity
$O(BD(A + H))$ for both passes, the same order as the forward pass alone.

## Common bugs
- Forgetting the $1 - v^2$ factor of `tanh`.
- Reporting the loss after the update instead of before it.
- Regularising biases, or using $c$ instead of $2c$.
- Summing over the batch instead of averaging, which scales the effective learning rate with $B$.

## Interview follow-ups
- AlphaGo Zero trained with SGD + momentum 0.9 and a stepped learning-rate schedule. How would you add momentum to this update?
- If the trunk were trainable, how would the two heads' gradients combine at the shared features, and why might you want to weight the value term down?
"""
