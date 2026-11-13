import numpy as np

META = dict(title='AdaGrad on least squares', tags=['Optimization', 'Training'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:05.05'],
            related=['rmsprop', 'adam-optimizer', 'minimum-of-axis-aligned-paraboloid'])

STATEMENT = r"""
AdaGrad gives every parameter its own learning rate. It keeps a running **sum** of squared gradients $G$ and divides each step by its square root, so coordinates that have seen large gradients slow down and rarely-updated ones keep moving fast:

$$G_t = G_{t-1} + g_t^2, \qquad w_t = w_{t-1} - \eta\, \frac{g_t}{\sqrt{G_t} + \epsilon}$$

(all operations element-wise). Use it to fit least squares,

$$L(w) = \frac{1}{2n}\lVert Xw - y\rVert^2, \qquad g = \nabla L(w) = \frac1n X^\top (Xw - y).$$

Write `adagrad(X, y, lr, steps, eps=1e-8, init_acc=0.0)` that starts from $w_0 = 0$ and $G_0 = $ `init_acc` (the same value in every coordinate), runs `steps` full-batch updates, and returns the tuple `(w, eff_lr)`:

- `w`: float array `(d,)` after the last step.
- `eff_lr`: float array `(d,)`, the per-coordinate step size the **next** update would use, $\eta / (\sqrt{G} + \epsilon)$.

## Inputs
`X`: float `(n, d)`; `y`: float `(n,)`; `lr > 0`; `steps >= 0`; `init_acc >= 0`.
"""

STARTER = '''import numpy as np

def adagrad(X, y, lr, steps, eps=1e-8, init_acc=0.0):
    """Return (w (d,), eff_lr (d,)) after `steps` AdaGrad updates on least squares."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def adagrad(X, y, lr, steps, eps=1e-8, init_acc=0.0):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    G = np.full(d, float(init_acc))
    for _ in range(steps):
        g = X.T @ (X @ w - y) / n
        G = G + g * g                          # the sum only grows, so steps only shrink
        w = w - lr * g / (np.sqrt(G) + eps)    # eps outside the root, as in the formula
    return w, lr / (np.sqrt(G) + eps)
'''

HINTS = [
    'Two vectors of state: the weights and the accumulator `G`, both shape `(d,)`. Fill `G` with `init_acc` at the start.',
    'Each step: gradient, then add its element-wise square to `G`, then divide the step element-wise by `np.sqrt(G) + eps`.',
    'Update `G` **before** using it in the same step. `eff_lr` is just `lr / (np.sqrt(G) + eps)` with the final `G`.',
]

_r = np.random.default_rng(23)
_Xs = (_r.random(size=(40, 5)) < np.array([0.9, 0.5, 0.2, 0.05, 0.05])) * _r.normal(size=(40, 5))
_ys = _Xs @ np.array([1.0, -1.0, 2.0, 3.0, -2.0])
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 2.0]), 0.5, 2),
    (np.array([[2.0], [1.0]]), np.array([4.0, 2.0]), 1.0, 3),
    (np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([1.0, 0.0]), 0.1, 5),
    (np.array([[3.0, 1.0], [1.0, 2.0]]), np.array([1.0, 1.0]), 0.1, 0, 1e-8, 0.1),
    (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.array([1.0, 0.0, -1.0]), 0.2, 50, 1e-8, 0.1),
    (_Xs, _ys, 0.5, 200),
    (np.array([[1e3, 0.0], [0.0, 1e-3]]), np.array([1e3, 1e-3]), 0.1, 20),
]

EXPLANATION = r"""
## Why per-coordinate scaling helps
On a badly scaled problem (test 7 has curvatures $10^6/2$ and $10^{-6}/2$), one global learning rate must be tiny to stay stable on the steep coordinate, and then the flat coordinate barely moves. AdaGrad divides by the gradient's own magnitude history, so early steps are about $\eta$ in **every** coordinate regardless of scale. For sparse features (test 6), a rare feature's accumulator stays small, so the few times it does get a gradient it takes a big step.

## Why it eventually stalls
$G$ is a sum, so it never shrinks and the effective rate $\eta/\sqrt{G_t}$ decays roughly like $1/\sqrt t$. That's ideal for convex online learning (it gives $O(\sqrt T)$ regret), but in deep learning the rate can die before training finishes. RMSProp and Adam replace the sum with an exponential moving average to fix exactly this.

## Complexity
Each step is $O(nd)$ for the gradient plus $O(d)$ for the update; memory $O(d)$ for the accumulator.

## Common bugs
- Putting $\epsilon$ inside the square root, or adding it to $G$ permanently. Both change the numbers (PyTorch puts it outside).
- Using the accumulator from **before** adding $g_t^2$, which takes a huge first step ($g/\epsilon$).
- Dividing by `G` instead of `sqrt(G)`.

## Interview follow-ups
- What is AdaGrad's first step, in each coordinate, when `init_acc = 0`? ($-\eta\,\mathrm{sign}(g_1)$, ignoring $\epsilon$: sign descent.)
- How much memory do adaptive optimizers add for a model with $P$ parameters? (AdaGrad and RMSProp: $P$ floats; Adam: $2P$. This is why 8-bit and factored optimizer states exist.)
"""
