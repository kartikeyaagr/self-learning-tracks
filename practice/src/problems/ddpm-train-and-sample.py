import numpy as np

META = dict(title='DDPM end to end: train a toy denoiser, then sample', tags=['Diffusion Models', 'Sampling'], diff='Hard',
            runtime='numpy', lessons=['dlcore:07.03', 'dlcore:07.04', 'dlcore:07.05'],
            related=['linear-beta-schedule', 'forward-diffusion-sample', 'toy-diffusion-training-step', 'ddpm-ancestral-step'])

STATEMENT = r"""
Run the whole DDPM recipe on a model small enough to write by hand: build the noise schedule, train an $\epsilon$-prediction model with the simple loss, and generate samples with ancestral sampling.

Write `ddpm_toy(x0s, ts, eps, zs, T, beta1, betaT, W, emb, lr)` returning `(losses, W, emb, samples)`.

## Schedule
Linear: $\beta_t$ for $t = 1..T$ evenly spaced from `beta1` to `betaT` (`np.linspace`). $\alpha_t = 1 - \beta_t$, $\bar\alpha_t = \prod_{s\le t}\alpha_s$. Index arrays with $t - 1$.

## Model
$\hat\epsilon(x_t, t) = x_tW + \text{emb}[t-1]$, with `W` `(D, D)` and a learned per-timestep vector `emb` `(T, D)`.

## Training (Algorithm 1 of the paper)
`x0s` `(I, B, D)`, `ts` `(I, B)` ints in $1..T$, `eps` `(I, B, D)`. For iteration $i$:
$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$ (per row, with that row's $t$); loss $= \mathrm{mean}\,(\hat\epsilon - \epsilon)^2$ over all $B \cdot D$ entries; record it, then one SGD step on `W` and `emb` with rate `lr` (rows of `emb` used several times accumulate).

## Sampling (Algorithm 2), with the trained parameters
`zs` `(T + 1, S, D)`: `zs[0]` is $x_T$; for $t = T, \dots, 1$:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\hat\epsilon(x_t, t)\Big) + \sigma_t z,\quad \sigma_t = \sqrt{\beta_t},\ z = \texttt{zs}[T - t + 1]$$
except that no noise is added at $t = 1$.

Return `losses` `(I,)`, the trained `W` and `emb` (new arrays), and `samples` $= x_0$ `(S, D)`.
"""

STARTER = '''import numpy as np

def ddpm_toy(x0s, ts, eps, zs, T, beta1, betaT, W, emb, lr):
    """Return (losses, W, emb, samples)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def ddpm_toy(x0s, ts, eps, zs, T, beta1, betaT, W, emb, lr):
    beta = np.linspace(beta1, betaT, T)
    alpha = 1 - beta
    abar = np.cumprod(alpha)
    W, emb = np.array(W, dtype=float), np.array(emb, dtype=float)
    losses = []
    for x0, t, e in zip(np.asarray(x0s, dtype=float), np.asarray(ts), np.asarray(eps, dtype=float)):
        ab = abar[t - 1][:, None]
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * e
        diff = xt @ W + emb[t - 1] - e
        losses.append(float((diff ** 2).mean()))
        g = 2 * diff / diff.size
        gW = xt.T @ g
        gE = np.zeros_like(emb)
        np.add.at(gE, t - 1, g)
        W -= lr * gW
        emb -= lr * gE
    zs = np.asarray(zs, dtype=float)
    x = zs[0].copy()
    for t in range(T, 0, -1):
        e_hat = x @ W + emb[t - 1]
        x = (x - beta[t - 1] / np.sqrt(1 - abar[t - 1]) * e_hat) / np.sqrt(alpha[t - 1])
        if t > 1:
            x = x + np.sqrt(beta[t - 1]) * zs[T - t + 1]
    return np.array(losses), W, emb, x
'''

HINTS = [
    'Build $\\beta$, $\\alpha$ and $\\bar\\alpha$ once with `linspace` and `cumprod`; a timestep $t$ reads index $t - 1$.',
    'The loss is a mean over $B \\cdot D$ entries, so its gradient with respect to $\\hat\\epsilon$ is $2(\\hat\\epsilon - \\epsilon)/(BD)$. Then $\\partial/\\partial W = x_t^\\top g$, and each row of $g$ goes to `emb[t-1]` with `np.add.at`.',
    'Sampling walks $t$ from $T$ down to 1 and uses `zs[T - t + 1]` as the fresh noise, skipping it at the last step.',
]

_r = np.random.default_rng(2006)
def _case(I, B, D, T, S):
    data = np.array([1.5, -0.5, 0.8, 0.0][:D])
    x0s = data + 0.3 * _r.normal(size=(I, B, D))
    return (x0s, _r.integers(1, T + 1, size=(I, B)), _r.normal(size=(I, B, D)), _r.normal(size=(T + 1, S, D)), T)
TESTS = [
    (*_case(1, 4, 2, 5, 3), 1e-4, 0.2, np.zeros((2, 2)), np.zeros((5, 2)), 0.1),
    (*_case(5, 8, 2, 10, 4), 1e-4, 0.1, np.zeros((2, 2)), np.zeros((10, 2)), 0.5),
    (*_case(40, 16, 3, 20, 5), 1e-4, 0.2, 0.1 * _r.normal(size=(3, 3)), np.zeros((20, 3)), 0.5),
    (*_case(30, 8, 4, 8, 2), 1e-3, 0.3, np.eye(4) * 0.5, 0.1 * _r.normal(size=(8, 4)), 0.3),
    (*_case(1, 2, 2, 1, 2), 0.5, 0.5, np.zeros((2, 2)), np.zeros((1, 2)), 0.0),
]

EXPLANATION = r"""
## The two algorithms
Training never runs the chain: thanks to the closed form $q(x_t|x_0)$, each example jumps straight to a random noise level and the model learns to predict the noise that was added. Sampling does run the chain, $T$ model calls from pure noise to data, each removing the predicted noise and adding a little fresh noise.

## Why predict noise
With $\epsilon$-prediction and the weights dropped (the "simple" loss), the objective is a denoising score-matching loss summed over noise levels; the paper found it gave better samples than the full variational bound.

## What the toy model can learn
A linear $\hat\epsilon$ with a per-step offset can represent the exact optimal denoiser for Gaussian data, so with enough training the samples approach the data's mean and spread. Real models replace $x_tW + \text{emb}$ with a U-Net or transformer conditioned on a timestep embedding.

## Choices this spec fixes
$\sigma_t^2 = \beta_t$ (the paper also tried the posterior variance $\tilde\beta_t$; both work), no noise at the final step, and the linear schedule from $10^{-4}$ to $0.02$ for $T = 1000$ in the paper.

## Common bugs
- Off-by-one between timesteps $1..T$ and array indices.
- Using $\sqrt{\beta_t}$ instead of $\beta_t/\sqrt{1 - \bar\alpha_t}$ as the noise coefficient in the mean.
- Adding noise at $t = 1$, which leaves the samples visibly noisy.

## Interview follow-ups
- How does DDIM sample in 50 steps from a model trained for 1000?
- What changes if the model predicts $x_0$ or $v$ instead of $\epsilon$?
"""
