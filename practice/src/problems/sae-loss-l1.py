import numpy as np

META = dict(title='SAE loss: reconstruction plus sparsity', tags=['Neural Networks', 'Loss Functions'], diff='Easy', runtime='numpy',
            related=['sae-forward'])

STATEMENT = r"""
Write `sae_loss(x, x_hat, f, W_dec, l1_coeff)` for a batch of `B` activations.

- Reconstruction: $\text{mse} = \frac{1}{B}\sum_b \lVert x_b - \hat{x}_b \rVert_2^2$ (sum over dimensions, mean over the batch).
- Sparsity: $\text{l1} = \frac{1}{B}\sum_b \sum_i f_{b,i}\,\lVert W_{dec,i} \rVert_2$, so a feature cannot dodge the penalty by shrinking its activation and growing its decoder row.
- $\text{total} = \text{mse} + \lambda\,\text{l1}$ with $\lambda$ = `l1_coeff`.
- $\text{l0}$: the mean number of features with $f > 0$ per example. It is reported, not optimised.

Return a dict with keys `mse`, `l1`, `total`, `l0`, all floats.
"""

STARTER = '''import numpy as np

def sae_loss(x, x_hat, f, W_dec, l1_coeff):
    """Reconstruction error plus a decoder-norm-weighted L1 penalty."""
    mse = float(np.mean((np.asarray(x) - np.asarray(x_hat)) ** 2))
    l1 = float(np.abs(f).sum())
    return {"mse": mse, "l1": l1, "total": mse + l1_coeff * l1, "l0": 0.0}
'''

SOLUTION = '''import numpy as np

def sae_loss(x, x_hat, f, W_dec, l1_coeff):
    x, x_hat, f, W_dec = (np.asarray(a, float) for a in (x, x_hat, f, W_dec))
    mse = float(((x - x_hat) ** 2).sum(1).mean())
    norms = np.linalg.norm(W_dec, axis=1)
    l1 = float((f * norms).sum(1).mean())
    l0 = float((f > 0).sum(1).mean())
    return {"mse": mse, "l1": l1, "total": mse + l1_coeff * l1, "l0": l0}
'''

HINTS = [
    'Sum the squared error over dimensions first, then average over the batch.',
    'np.linalg.norm(W_dec, axis=1) gives one norm per feature; multiply it into f column-wise.',
    'L0 counts strictly positive activations; it is the number people quote when they say how sparse an SAE is.',
]

def _case(seed, B, d, m, lam):
    g = np.random.default_rng(seed)
    x = g.normal(size=(B, d))
    f = np.maximum(g.normal(size=(B, m)) - 0.5, 0)
    W = g.normal(size=(m, d))
    return (x, f @ W, f, W, lam)

TESTS = [
    ([[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 2.0]], [[1.0, 0.0], [0.0, 3.0]], 0.1),
    ([[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.6, 0.8]], 1.0),
    _case(1, 4, 3, 6, 0.01),
    _case(2, 8, 5, 20, 0.5),
    _case(3, 2, 2, 4, 0.0),
]

EXPLANATION = r"""
## The trade-off
The two terms pull in opposite directions. With $\lambda = 0$ the SAE reconstructs well using many features at once, which tells you nothing. With a large $\lambda$ it uses very few features and reconstruction suffers. Training sweeps $\lambda$ and reports a frontier of L0 against reconstruction quality.

## Why weight by the decoder norm
The model's output only depends on $f_i W_{dec,i}$. Halving $f_i$ and doubling $W_{dec,i}$ changes nothing except a plain L1 penalty, so without a fix the optimiser shrinks activations forever. Two common fixes: renormalise decoder rows to unit length after every step, or multiply the penalty by the row norm as here.

## Judging reconstruction properly
A small MSE is not enough. The real test is to splice $\hat{x}$ back into the model and measure how much the next-token loss rises. An SAE can have low MSE and still destroy the directions the model needs.

## Interview follow-ups
- Why does L1 shrink active features, and how would you measure that shrinkage?
"""
