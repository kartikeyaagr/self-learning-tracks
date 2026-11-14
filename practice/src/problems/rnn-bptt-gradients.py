import numpy as np

META = dict(title='Backpropagation through time for an Elman RNN', tags=['Neural Networks', 'Training'], diff='Hard',
            runtime='numpy', lessons=['dlcore:05.02'],
            related=['vanilla-rnn-cell', 'rnn-cell', 'gradient-clipping', 'backpropagation-single-hidden-layer'])

STATEMENT = r"""
An Elman network carries a hidden state through time and makes a prediction at every step. Training it means unrolling the recurrence and backpropagating through every step, and watching the gradient that reaches early steps shows why plain RNNs struggle with long-range dependencies.

Write `rnn_bptt(x, y, h0, p)` returning `(loss, grads, h_grad_norms)`.

## Model (row vectors)
- $h_t = \tanh(x_tW_{xh} + h_{t-1}W_{hh} + b_h)$ for $t = 0..T-1$, starting from `h0`.
- $z_t = h_tW_{hy} + b_y$; loss $= \frac1T\sum_t -\log\mathrm{softmax}(z_t)[y_t]$.
- `x` `(T, D_in)`, `y` `(T,)` ints, `h0` `(H,)`; `p` has `Wxh` `(D_in, H)`, `Whh` `(H, H)`, `bh` `(H,)`, `Why` `(H, K)`, `by` `(K,)`.

## Output
- `loss`: float.
- `grads`: dict with the same keys as `p` (and the same shapes) holding $\partial\,\text{loss}/\partial\theta$.
- `h_grad_norms`: `(T,)`, the L2 norm of the **total** gradient $\partial\,\text{loss}/\partial h_t$ (from the output at $t$ and from all later steps).
"""

STARTER = '''import numpy as np

def rnn_bptt(x, y, h0, p):
    """Return (loss, grads dict, h_grad_norms (T,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def rnn_bptt(x, y, h0, p):
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    T = len(y)
    hs, prev = [], np.asarray(h0, dtype=float)
    for t in range(T):
        prev = np.tanh(x[t] @ p['Wxh'] + prev @ p['Whh'] + p['bh'])
        hs.append(prev)
    Hm = np.array(hs)                                        # (T, H)
    z = Hm @ p['Why'] + p['by']
    z = z - z.max(1, keepdims=True)
    P = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    loss = float(-np.log(P[np.arange(T), y]).mean())
    dz = P.copy()
    dz[np.arange(T), y] -= 1
    dz /= T
    g = {k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in p.items()}
    g['Why'] = Hm.T @ dz
    g['by'] = dz.sum(0)
    norms = np.zeros(T)
    carry = np.zeros(Hm.shape[1])                            # gradient arriving from step t+1
    for t in range(T - 1, -1, -1):
        dh = dz[t] @ p['Why'].T + carry
        norms[t] = np.linalg.norm(dh)
        da = dh * (1 - Hm[t] ** 2)
        h_prev = Hm[t - 1] if t > 0 else np.asarray(h0, dtype=float)
        g['Wxh'] += np.outer(x[t], da)
        g['Whh'] += np.outer(h_prev, da)
        g['bh'] += da
        carry = da @ p['Whh'].T
    return loss, g, norms
'''

HINTS = [
    'Run the forward pass and keep every $h_t$. The output layer\'s gradients are the usual softmax ones, $\\partial/\\partial z_t = (\\mathrm{softmax}(z_t) - \\text{onehot}(y_t))/T$.',
    'Go backwards in time with a "carry" vector: $\\partial L/\\partial h_t = (\\partial L/\\partial z_t)W_{hy}^\\top + \\text{carry}$, then through tanh, and the new carry is that times $W_{hh}^\\top$.',
    'Parameter gradients are sums over time of outer products: $x_t^\\top \\delta_t$, $h_{t-1}^\\top \\delta_t$ (with $h_{-1} = h_0$), where $\\delta_t$ is the gradient at the tanh input.',
]

_r = np.random.default_rng(1990)
def _p(D, H, K, s=1.0):
    return dict(Wxh=_r.normal(size=(D, H)) / np.sqrt(D), Whh=_r.normal(size=(H, H)) / np.sqrt(H) * s, bh=0.1 * _r.normal(size=H),
                Why=_r.normal(size=(H, K)) / np.sqrt(H), by=0.1 * _r.normal(size=K))
TESTS = [
    (_r.normal(size=(1, 2)), np.array([1]), np.zeros(3), _p(2, 3, 2)),
    (_r.normal(size=(4, 2)), np.array([0, 1, 1, 0]), np.zeros(3), _p(2, 3, 2)),
    (_r.normal(size=(6, 3)), _r.integers(0, 4, size=6), _r.normal(size=5) * 0.5, _p(3, 5, 4)),
    (_r.normal(size=(12, 3)), _r.integers(0, 4, size=12), np.zeros(5), _p(3, 5, 4, 0.3)),
    (_r.normal(size=(12, 3)), _r.integers(0, 4, size=12), np.zeros(5), _p(3, 5, 4, 3.0)),
]

EXPLANATION = r"""
## Why gradients vanish or explode
The gradient from step $t$ to step $s < t$ is a product of $t - s$ Jacobians $\mathrm{diag}(1 - h^2)\,W_{hh}^\top$. If their norms are below 1 the product shrinks geometrically (test 4 scales $W_{hh}$ by 0.3: the norms at early steps carry almost nothing from the end of the sequence); if above 1 it can grow (test 5, though tanh saturation limits it). Plain RNNs therefore learn short-range structure easily and long-range structure hardly at all, which is what LSTMs and GRUs were designed to fix.

## Truncated BPTT
In practice the sequence is cut into chunks: the hidden state is carried forward across chunks, but gradients stop at chunk boundaries. That caps memory and compute but also caps the dependency length the model can learn through gradients.

## Complexity
Forward and backward are both $O(T(D_{in}H + H^2 + HK))$; memory is $O(TH)$ for the stored states.

## Common bugs
- Forgetting the carry term, which gives the gradient of the per-step outputs only.
- Using $h_t$ instead of $h_{t-1}$ in the $W_{hh}$ gradient.
- Dividing by $T$ in the loss but not in $\partial L/\partial z$.

## Interview follow-ups
- Gradient clipping fixes exploding gradients. Why doesn't it fix vanishing ones?
- How would you check these gradients numerically?
"""
