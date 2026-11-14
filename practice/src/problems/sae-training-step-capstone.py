import numpy as np

META = dict(title='Capstone: one sparse autoencoder training step', tags=['Neural Networks', 'Training'], diff='Hard', runtime='numpy',
            related=['sae-forward', 'sae-loss-l1', 'feature-activation-frequency'])

STATEMENT = r"""
Put the SAE pieces together into one full training step, with gradients written by hand.

Write `sae_train_step(x, W_enc, b_enc, W_dec, b_dec, l1_coeff, lr)`. The decoder rows start with unit norm.

1. Forward: $p = (x - b_{dec}) W_{enc} + b_{enc}$, $f = \mathrm{ReLU}(p)$, $\hat{x} = f W_{dec} + b_{dec}$.
2. Loss: $L = \frac{1}{B}\sum_b \lVert x_b - \hat{x}_b\rVert^2 + \lambda \frac{1}{B}\sum_b \sum_i f_{b,i}$ (plain L1, because the rows are kept at unit norm).
3. Backward through every parameter, including both places $b_{dec}$ appears. Use the ReLU derivative $\mathbb{1}[p > 0]$.
4. One SGD step on all four parameters with learning rate `lr`, then rescale each decoder row back to unit norm.
5. Recompute the loss with the new parameters.

Return `(loss_before, loss_after, l0_before, W_enc, b_enc, W_dec, b_dec)`, where `l0_before` is the mean number of active features per example in step 1.
"""

STARTER = '''import numpy as np

def sae_train_step(x, W_enc, b_enc, W_dec, b_dec, l1_coeff, lr):
    """One SGD step for a sparse autoencoder, with decoder rows renormalised."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _forward(x, W_enc, b_enc, W_dec, b_dec, lam):
    xc = x - b_dec
    p = xc @ W_enc + b_enc
    f = np.maximum(p, 0.0)
    xh = f @ W_dec + b_dec
    loss = float(((x - xh) ** 2).sum(1).mean() + lam * f.sum(1).mean())
    return xc, p, f, xh, loss

def sae_train_step(x, W_enc, b_enc, W_dec, b_dec, l1_coeff, lr):
    x, W_enc, b_enc, W_dec, b_dec = (np.asarray(a, float) for a in (x, W_enc, b_enc, W_dec, b_dec))
    B = x.shape[0]
    xc, p, f, xh, loss0 = _forward(x, W_enc, b_enc, W_dec, b_dec, l1_coeff)
    l0 = float((f > 0).sum(1).mean())
    dxh = 2.0 * (xh - x) / B
    dW_dec = f.T @ dxh
    df = dxh @ W_dec.T + l1_coeff / B
    dp = df * (p > 0)
    dW_enc = xc.T @ dp
    db_enc = dp.sum(0)
    db_dec = dxh.sum(0) - (dp @ W_enc.T).sum(0)
    W_enc = W_enc - lr * dW_enc
    b_enc = b_enc - lr * db_enc
    W_dec = W_dec - lr * dW_dec
    b_dec = b_dec - lr * db_dec
    W_dec = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)
    loss1 = _forward(x, W_enc, b_enc, W_dec, b_dec, l1_coeff)[-1]
    return loss0, loss1, l0, W_enc, b_enc, W_dec, b_dec
'''

HINTS = [
    'b_dec appears twice: added at the output, and subtracted at the input. Its gradient is the sum of both paths, and the input path has a minus sign.',
    'The L1 term adds the constant lambda / B to the gradient of every f entry; the ReLU mask then zeroes it for inactive features.',
    'Renormalise after the SGD step, not before, and compute loss_after with the renormalised decoder.',
]

def _case(seed, B, d, m, lam, lr):
    g = np.random.default_rng(seed)
    true = g.normal(size=(m, d)); true /= np.linalg.norm(true, axis=1, keepdims=True)
    codes = np.maximum(g.normal(size=(B, m)) - 1.0, 0)
    x = codes @ true + 0.05 * g.normal(size=(B, d))
    W_dec = g.normal(size=(m, d)); W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
    return (x, W_dec.T.copy(), np.zeros(m), W_dec, x.mean(0), lam, lr)

TESTS = [
    _case(1, 16, 4, 8, 0.1, 0.05),
    _case(2, 32, 6, 12, 0.3, 0.02),
    _case(3, 8, 3, 6, 0.0, 0.1),
    _case(4, 64, 8, 32, 1.0, 0.01),
    ([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], 0.5, 0.1),
]

EXPLANATION = r"""
## The full loop
A real SAE run repeats this step over billions of cached activations. Everything here carries over: the tied start ($W_{enc} = W_{dec}^\top$), $b_{dec}$ set to the data mean, unit-norm decoder rows, and an L1 penalty that trades reconstruction against sparsity.

## The gradient of the decoder bias
$b_{dec}$ is added at the output and subtracted at the input, so
$$\frac{\partial L}{\partial b_{dec}} = \sum_b \frac{\partial L}{\partial \hat{x}_b} - \sum_b W_{enc} \frac{\partial L}{\partial p_b}.$$
Forgetting the second path is the most common bug.

## Why renormalise
The penalty on $f$ can be dodged by growing $W_{dec}$ and shrinking $f$. Projecting rows back to unit norm after each step closes that loophole. A stricter version also removes the gradient component parallel to each row before the step.

## What the last test shows
With an identity SAE and inputs that are already features, the only thing to learn is that the L1 penalty wants smaller activations, so the loss after the step is lower even though reconstruction got slightly worse.

## Interview follow-ups
- Training uses Adam in practice. Why does Adam interact badly with decoder renormalisation, and what do people do about it?
"""
