import numpy as np

META = dict(title='Dropout end to end: train with masks, predict deterministically and with MC dropout', tags=['Neural Networks', 'Training'],
            diff='Hard', runtime='numpy', lessons=['dlcore:02.03'],
            related=['dropout', 'dropout-2', 'multi-layer-perceptron-forward-pass', 'backpropagation-single-hidden-layer'])

STATEMENT = r"""
Dropout randomly zeroes hidden units during training so that no unit can rely on specific others. With inverted dropout, the kept units are scaled up at training time and the network is used unchanged at test time. Keeping dropout on at test time and averaging many passes (MC dropout) also gives a rough uncertainty estimate.

Write `dropout_regression(X, Y, Xq, p, train_masks, mc_masks, rate, lr)` returning `(losses, p_trained, pred, mc_mean, mc_std)`.

## Model
$h = \mathrm{ReLU}(xW_1 + b_1)$; with a keep mask $m \in \{0, 1\}^H$, $\tilde h = h\odot m/(1 - \text{rate})$; $\hat y = \tilde h\,w_2 + b_2$ (a scalar per row). `p = {'W1': (D, H), 'b1': (H,), 'w2': (H,), 'b2': float}`.

## Training
`X` `(I, B, D)`, `Y` `(I, B)`, `train_masks` `(I, B, H)`. Iteration $i$ uses mask row `train_masks[i, b]` for example $b$; loss $= \mathrm{mean}_b(\hat y - y)^2$, recorded before one SGD step on all four parameters.

## Prediction on `Xq` `(Q, D)` with the trained parameters
- `pred`: no dropout and no scaling.
- MC dropout: `mc_masks` `(M, Q, H)`; pass $j$ applies `mc_masks[j]` (with the same inverted scaling). `mc_mean` and `mc_std` (population, `ddof=0`) are taken over the $M$ passes, each `(Q,)`.

Return `losses` `(I,)`, a new parameter dict (`b2` as a float), and the three `(Q,)` arrays.
"""

STARTER = '''import numpy as np

def dropout_regression(X, Y, Xq, p, train_masks, mc_masks, rate, lr):
    """Return (losses, p_trained, pred, mc_mean, mc_std)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def dropout_regression(X, Y, Xq, p, train_masks, mc_masks, rate, lr):
    W1, b1, w2 = (np.array(p[k], dtype=float) for k in ('W1', 'b1', 'w2'))
    b2 = float(p['b2'])
    keep = 1.0 - rate
    losses = []
    for x, y, m in zip(*(np.asarray(a, dtype=float) for a in (X, Y, train_masks))):
        pre = x @ W1 + b1
        h = np.maximum(pre, 0)
        ht = h * m / keep
        r = ht @ w2 + b2 - y
        losses.append(float((r ** 2).mean()))
        g = 2 * r / len(r)
        gw2, gb2 = ht.T @ g, g.sum()
        dpre = np.outer(g, w2) * m / keep * (pre > 0)
        W1 -= lr * x.T @ dpre
        b1 -= lr * dpre.sum(0)
        w2 -= lr * gw2
        b2 -= lr * gb2
    hq = np.maximum(np.asarray(Xq, dtype=float) @ W1 + b1, 0)
    pred = hq @ w2 + b2
    mc = np.array([(hq * m / keep) @ w2 + b2 for m in np.asarray(mc_masks, dtype=float)])
    return np.array(losses), {'W1': W1, 'b1': b1, 'w2': w2, 'b2': float(b2)}, pred, mc.mean(0), mc.std(0)
'''

HINTS = [
    'The mask multiplies after the ReLU, so in the backward pass it multiplies the gradient flowing into $h$ (with the same $1/(1 - \\text{rate})$ factor), before the ReLU gate.',
    'With $g = 2(\\hat y - y)/B$: $\\partial w_2 = \\tilde h^\\top g$, $\\partial b_2 = \\sum g$, and the hidden gradient is $g\\,w_2^\\top \\odot m/(1-\\text{rate}) \\odot [a > 0]$.',
    'The deterministic prediction uses no mask and no scaling, which is exactly why the training-time scaling is there: it keeps the expected activation the same in both modes.',
]

_r = np.random.default_rng(2016)
def _case(I, B, D, H, Q, M, rate):
    w = _r.normal(size=D)
    X = _r.normal(size=(I, B, D))
    Y = np.sin(X @ w) + 0.1 * _r.normal(size=(I, B))
    p = dict(W1=_r.normal(size=(D, H)) / np.sqrt(D), b1=0.1 * _r.normal(size=H), w2=_r.normal(size=H) / np.sqrt(H), b2=0.0)
    return (X, Y, _r.normal(size=(Q, D)), p, (_r.random((I, B, H)) > rate).astype(float),
            (_r.random((M, Q, H)) > rate).astype(float), rate)
TESTS = [
    (*_case(1, 4, 2, 3, 2, 3, 0.5), 0.1),
    (*_case(20, 8, 3, 8, 4, 10, 0.2), 0.05),
    (*_case(40, 16, 2, 16, 5, 20, 0.5), 0.05),
    (*_case(5, 4, 3, 6, 3, 5, 0.0), 0.1),
    (*_case(30, 8, 4, 10, 4, 50, 0.3), 0.02),
]

EXPLANATION = r"""
## What dropout does
Each training step samples a different thinned network, so units can't co-adapt and the full network at test time behaves like an average over an exponential number of sub-networks. The paper scaled the weights by the keep probability at test time; inverted dropout moves that scaling into training so inference code doesn't change.

## Why the scaling matches
A unit kept with probability $1 - \text{rate}$ and scaled by $1/(1 - \text{rate})$ has the same expected value as the unscaled unit, so `pred` matches the average of the training-mode network's pre-output in expectation. The network is linear after the mask here, so `mc_mean` approaches `pred` as $M$ grows; with nonlinear layers after the dropout they differ.

## MC dropout
Leaving dropout on at test time and looking at the spread of predictions is a cheap approximation to Bayesian model averaging (Gal and Ghahramani). It gives larger spread far from the training data, though it is not calibrated without tuning. Test 4 uses rate 0, so the spread is exactly 0.

## Common bugs
- Applying the mask in the backward pass to the wrong tensor, or forgetting the scale there.
- Scaling at test time as well as training time (double compensation).
- Reusing one mask for the whole batch instead of one per example.

## Interview follow-ups
- Why is dropout used less in large transformers trained for one epoch?
- How does dropout interact with BatchNorm's running statistics?
"""
