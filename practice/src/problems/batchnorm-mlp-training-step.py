import numpy as np

META = dict(title='Train an MLP with BatchNorm: backward pass, running statistics, eval mode', tags=['Neural Networks', 'Training'],
            diff='Hard', runtime='numpy', lessons=['dlcore:02.01'],
            related=['batch-normalization', 'per-channel-mean-and-std', 'layer-normalization', 'backpropagation-single-hidden-layer'])

STATEMENT = r"""
Batch normalisation normalises each hidden unit over the mini-batch during training and switches to stored running statistics at test time. Train a two-layer classifier with a BatchNorm layer for one SGD step, including the backward pass through the batch statistics.

Write `bn_mlp_step(x, y, p, running, lr, momentum, eps)` returning `(loss, new_p, new_running, eval_logits)`.

## Forward (training mode)
- $a = xW_1 + b_1$ `(B, H)`.
- Batch statistics per unit: $\mu = \mathrm{mean}_B(a)$, $\sigma^2 = \mathrm{var}_B(a)$ (biased, divide by $B$).
- $\hat a = (a - \mu)/\sqrt{\sigma^2 + \epsilon}$, $\;u = \gamma\odot\hat a + \beta$, $\;h = \mathrm{ReLU}(u)$.
- logits $= hW_2 + b_2$; loss $=$ mean softmax cross-entropy against integer labels `y`.

## Update
- `p` has `W1 b1 gamma beta W2 b2`. Every parameter takes one step $\theta - \text{lr}\cdot\partial L/\partial\theta$, with the gradient flowing through $\mu$ and $\sigma^2$.
- `running = {'mean': (H,), 'var': (H,)}`: $r \leftarrow (1 - m)\,r + m\,s$ with $m$ = `momentum`, using $\mu$ for the mean and the **unbiased** batch variance ($\times B/(B-1)$) for the variance.

## Eval
`eval_logits`: the same network on the same `x` with the **updated** parameters, normalising with the **updated** running statistics instead of batch statistics.

Return `loss` (before the update, float), new dicts, and `eval_logits` `(B, K)`. Don't modify the inputs.
"""

STARTER = '''import numpy as np

def bn_mlp_step(x, y, p, running, lr, momentum, eps):
    """Return (loss, new_p, new_running, eval_logits)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def bn_mlp_step(x, y, p, running, lr, momentum, eps):
    x, y = np.asarray(x, dtype=float), np.asarray(y)
    q = {k: np.asarray(v, dtype=float) for k, v in p.items()}
    B = x.shape[0]
    a = x @ q['W1'] + q['b1']
    mu, var = a.mean(0), a.var(0)
    inv = 1 / np.sqrt(var + eps)
    ah = (a - mu) * inv
    u = q['gamma'] * ah + q['beta']
    h = np.maximum(u, 0)
    lg = h @ q['W2'] + q['b2']
    z = lg - lg.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    loss = float(-lp[np.arange(B), y].mean())
    dl = np.exp(lp)
    dl[np.arange(B), y] -= 1
    dl /= B
    G = {'W2': h.T @ dl, 'b2': dl.sum(0)}
    du = (dl @ q['W2'].T) * (u > 0)
    G['gamma'], G['beta'] = (du * ah).sum(0), du.sum(0)
    dah = du * q['gamma']
    da = inv * (dah - dah.mean(0) - ah * (dah * ah).mean(0))
    G['W1'], G['b1'] = x.T @ da, da.sum(0)
    new_p = {k: q[k] - lr * G[k] for k in q}
    rm = (1 - momentum) * np.asarray(running['mean'], dtype=float) + momentum * mu
    rv = (1 - momentum) * np.asarray(running['var'], dtype=float) + momentum * var * B / (B - 1)
    ae = x @ new_p['W1'] + new_p['b1']
    he = np.maximum(new_p['gamma'] * (ae - rm) / np.sqrt(rv + eps) + new_p['beta'], 0)
    return loss, new_p, {'mean': rm, 'var': rv}, he @ new_p['W2'] + new_p['b2']
'''

HINTS = [
    'Keep $\\hat a$ and $1/\\sqrt{\\sigma^2 + \\epsilon}$ from the forward pass; $\\partial L/\\partial\\gamma = \\sum_B g_u\\odot\\hat a$ and $\\partial L/\\partial\\beta = \\sum_B g_u$.',
    'With $g = \\gamma\\odot g_u$, the gradient through the batch statistics collapses to $\\frac{1}{\\sqrt{\\sigma^2+\\epsilon}}\\big(g - \\mathrm{mean}_B(g) - \\hat a\\,\\mathrm{mean}_B(g\\odot\\hat a)\\big)$.',
    'The running variance uses the unbiased estimate; the normalisation in training mode uses the biased one. Eval mode uses the updated parameters and the updated running statistics.',
]

_r = np.random.default_rng(2015)
def _p(D, H, K):
    n = lambda *s: _r.normal(size=s)
    return dict(W1=n(D, H) / np.sqrt(D), b1=0.1 * n(H), gamma=1 + 0.1 * n(H), beta=0.1 * n(H), W2=n(H, K) / np.sqrt(H), b2=0.1 * n(K))
def _run(H):
    return dict(mean=np.zeros(H), var=np.ones(H))
TESTS = [
    (_r.normal(size=(4, 3)), np.array([0, 1, 1, 0]), _p(3, 4, 2), _run(4), 0.1, 0.1, 1e-5),
    (_r.normal(size=(8, 5)) * 3 + 2, _r.integers(0, 3, 8), _p(5, 6, 3), _run(6), 0.5, 0.1, 1e-5),
    (_r.normal(size=(16, 4)), _r.integers(0, 4, 16), _p(4, 8, 4), dict(mean=0.3 * _r.normal(size=8), var=1 + _r.random(8)), 0.2, 0.01, 1e-3),
    (_r.normal(size=(2, 3)), np.array([1, 0]), _p(3, 3, 2), _run(3), 1.0, 1.0, 1e-5),
    (_r.normal(size=(10, 6)), _r.integers(0, 2, 10), _p(6, 5, 2), _run(5), 0.0, 0.5, 1e-5),
]

EXPLANATION = r"""
## Why BatchNorm trained faster
Normalising each pre-activation over the batch keeps its scale under control as earlier layers change, which allowed much higher learning rates and made networks less sensitive to initialisation. The paper's explanation was reduced "internal covariate shift"; later work argues the main effect is a smoother loss landscape. The learned $\gamma, \beta$ let the network undo the normalisation where that helps.

## The backward pass
Every example's normalised value depends on every other example through $\mu$ and $\sigma^2$, so the gradient is not elementwise. The compact form subtracts the batch mean of the upstream gradient and its projection onto $\hat a$; the gradient through $a$ always sums to zero over the batch, which is also why the bias $b_1$ gets exactly zero gradient.

## Train vs eval
At test time there is no batch to normalise over, so the layer uses exponential moving averages collected during training. A model that looks fine in training and fails in eval often has stale or never-updated running statistics, or ran eval in training mode.

## Common bugs
- Treating $\mu$ and $\sigma^2$ as constants in the backward pass.
- Using the biased variance for the running estimate (frameworks use the unbiased one) or the unbiased one for normalisation.
- Mixing up the momentum convention: here $m$ is the weight on the new batch statistic.

## Interview follow-ups
- Why does BatchNorm behave badly with very small batches, and what do LayerNorm and GroupNorm change?
- What happens to $b_1$'s gradient, and why do networks with BatchNorm often drop the preceding bias?
"""
