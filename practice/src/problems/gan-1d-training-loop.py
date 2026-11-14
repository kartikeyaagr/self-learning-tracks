import numpy as np

META = dict(title='Train a 1-D GAN: alternating updates and the non-saturating loss', tags=['Generative Models', 'Training'],
            diff='Hard', runtime='numpy',
            related=['gan-training-step', 'log-loss-binary-cross-entropy', 'logistic-regression'])

STATEMENT = r"""
A GAN trains two models against each other. Run the full alternating loop on a problem small enough to follow by hand: the data are 1-D Gaussian samples, the generator is an affine map of noise, and the discriminator is logistic regression on $(x, x^2)$, which is enough to notice a wrong mean **or** a wrong spread.

Write `train_gan(real, zd, zg, g0, d0, lr_d, lr_g)` returning `(g_hist, d_losses, g_losses)`.

## Models
- Generator $G(z) = wz + b$, parameters `g = (w, b)`.
- Discriminator logit $\ell(x) = a x + c x^2 + e$, parameters `d = (a, c, e)`, $D(x) = \sigma(\ell(x))$.

## Loop
`real` has shape `(N, K, M)`, `zd` `(N, K, M)`, `zg` `(N, M)`. For step $n = 0..N-1$:
1. Repeat for $k = 0..K-1$: a discriminator step on real batch `real[n, k]` and fake batch $G(\text{zd}[n, k])$ with the current generator.
   $L_D = -\mathrm{mean}\log D(x_{real}) - \mathrm{mean}\log(1 - D(x_{fake}))$; record it (before the update) and take a gradient step on $(a, c, e)$ with `lr_d`.
2. One generator step on $x = G(\text{zg}[n])$ with the **non-saturating** loss $L_G = -\mathrm{mean}\log D(x)$; record it (before the update) and step $(w, b)$ with `lr_g`.

Compute $\log\sigma$ stably. Gradients flow into the generator through $\partial\ell/\partial x = a + 2cx$; the discriminator is not changed during the generator step.

## Output
`g_hist` `(N+1, 2)`: $(w, b)$ before the first step and after every generator step. `d_losses` `(N, K)`, `g_losses` `(N,)`.
"""

STARTER = '''import numpy as np

def train_gan(real, zd, zg, g0, d0, lr_d, lr_g):
    """Return (g_hist (N+1, 2), d_losses (N, K), g_losses (N,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sig(u):
    return 0.5 * (1 + np.tanh(0.5 * u))

def _logsig(u):
    return -np.logaddexp(0.0, -u)

def train_gan(real, zd, zg, g0, d0, lr_d, lr_g):
    real, zd, zg = (np.asarray(a, dtype=float) for a in (real, zd, zg))
    w, b = map(float, g0)
    a, c, e = map(float, d0)
    N, K = real.shape[:2]
    hist, dl, gl = [(w, b)], np.zeros((N, K)), np.zeros(N)
    for n in range(N):
        for k in range(K):
            xr, xf = real[n, k], w * zd[n, k] + b
            lr_, lf = a * xr + c * xr ** 2 + e, a * xf + c * xf ** 2 + e
            dl[n, k] = -_logsig(lr_).mean() - _logsig(-lf).mean()
            gr, gf = _sig(lr_) - 1, _sig(lf)                   # dL/dlogit, per sample (before the mean)
            a -= lr_d * ((gr * xr).mean() + (gf * xf).mean())
            c -= lr_d * ((gr * xr ** 2).mean() + (gf * xf ** 2).mean())
            e -= lr_d * (gr.mean() + gf.mean())
        z = zg[n]
        x = w * z + b
        l = a * x + c * x ** 2 + e
        gl[n] = -_logsig(l).mean()
        dx = (_sig(l) - 1) * (a + 2 * c * x)
        w -= lr_g * (dx * z).mean()
        b -= lr_g * dx.mean()
        hist.append((w, b))
    return np.array(hist), dl, gl
'''

HINTS = [
    'For a logistic discriminator, $\\partial L_D/\\partial\\ell$ is $\\sigma(\\ell) - 1$ on real samples and $\\sigma(\\ell)$ on fake ones. Multiply by $x$, $x^2$ or 1 and average to get the three parameter gradients.',
    'The non-saturating generator loss has $\\partial L_G/\\partial\\ell = \\sigma(\\ell) - 1$; chain through $\\partial\\ell/\\partial x = a + 2cx$ and $\\partial x/\\partial w = z$, $\\partial x/\\partial b = 1$.',
    'Record each loss before its update, and make sure the $K$ discriminator steps all use the generator from before this step\'s generator update.',
]

_r = np.random.default_rng(2014)
def _case(N, K, M, mu=2.0, s=0.5):
    return mu + s * _r.normal(size=(N, K, M)), _r.normal(size=(N, K, M)), _r.normal(size=(N, M))
TESTS = [
    (*_case(1, 1, 4), (1.0, 0.0), (0.0, 0.0, 0.0), 0.1, 0.1),
    (*_case(3, 1, 8), (1.0, 0.0), (0.1, -0.1, 0.0), 0.2, 0.2),
    (*_case(20, 2, 16), (1.0, 0.0), (0.0, 0.0, 0.0), 0.05, 0.05),
    (*_case(30, 1, 32, -1.0, 2.0), (0.2, 0.5), (0.0, 0.0, 0.0), 0.02, 0.05),
    (*_case(10, 5, 8), (0.01, 2.0), (0.0, 0.0, 0.0), 0.1, 0.1),
]

EXPLANATION = r"""
## Minimax, in practice
The original objective is $\min_G\max_D\ \mathbb E\log D(x) + \mathbb E\log(1 - D(G(z)))$. Early in training $D$ easily rejects fakes, $D(G(z)) \approx 0$, and $\log(1 - D)$ is flat there, so the generator gets almost no gradient. The paper's fix, used here, is to have $G$ maximise $\log D(G(z))$ instead: same fixed point, much stronger gradient when the generator is losing.

## What to watch
- With a linear discriminator ($c = 0$) only the mean can be matched; the quadratic feature lets $D$ also detect a too-narrow or too-wide spread. Test 5 starts with $w \approx 0$, a generator that outputs nearly one value: a collapsed distribution that the $x^2$ feature lets the discriminator punish.
- $K > 1$ discriminator steps per generator step keeps $D$ near its optimum, as in Algorithm 1 of the paper.
- Training need not converge: the two players can oscillate around the equilibrium, which is visible in `g_hist` for larger learning rates.

## Common bugs
- Letting the generator step also change the discriminator, or updating $D$ with the generator's loss.
- Using the saturating $\log(1 - D)$ loss for the generator.
- Updating the generator before recording $L_G$.

## Interview follow-ups
- What is mode collapse, and why can't this affine generator show the multi-modal kind?
- How do WGAN and gradient penalties change the discriminator's job?
"""
