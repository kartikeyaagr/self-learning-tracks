import numpy as np

META = dict(title='One VAE training step with hand-derived gradients', tags=['Generative Models', 'Training'], diff='Hard',
            runtime='numpy', lessons=['dlcore:07.01'],
            related=['variational-autoencoder-forward-pass', 'reparameterized-gaussian-sample', 'diagonal-gaussian-kl-divergence',
                     'log-loss-binary-cross-entropy'])

STATEMENT = r"""
Train a small variational autoencoder for one SGD step, with every gradient derived by hand. The reparameterisation trick is what makes the sampled latent differentiable.

Write `vae_step(x, eps, p, lr)` returning `(loss, new_p)`.

## Model (batch of $B$ binary-valued vectors)
- Encoder: $h = \tanh(xW_1 + b_1)$; $\mu = hW_\mu + b_\mu$; $\log\sigma^2 = hW_v + b_v$.
- Sample: $z = \mu + \sigma\odot\epsilon$ with $\sigma = \exp(\tfrac12\log\sigma^2)$ and the given noise `eps` `(B, Z)`.
- Decoder: $\text{logits} = \tanh(zW_2 + b_2)W_3 + b_3$; $\hat x = \sigma(\text{logits})$.
- Loss (mean over the batch of the negative ELBO):
  $$\mathcal L = \frac1B\sum_b\Big[\sum_j \mathrm{BCE}(\text{logits}_{bj}, x_{bj}) + \tfrac12\sum_k\big(\mu_{bk}^2 + \sigma_{bk}^2 - \log\sigma_{bk}^2 - 1\big)\Big]$$
  with $\mathrm{BCE}(\ell, x) = \max(\ell, 0) - \ell x + \log(1 + e^{-|\ell|})$.

## Update
`p` has `W1 b1 Wmu bmu Wv bv W2 b2 W3 b3`. Return `loss` (before the update, a float) and a new dict with every parameter $\theta - \text{lr}\cdot\partial\mathcal L/\partial\theta$. Don't modify `p`.
"""

STARTER = '''import numpy as np

def vae_step(x, eps, p, lr):
    """Return (loss, new_p)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def vae_step(x, eps, p, lr):
    x, eps = np.asarray(x, dtype=float), np.asarray(eps, dtype=float)
    B = x.shape[0]
    h = np.tanh(x @ p['W1'] + p['b1'])
    mu, lv = h @ p['Wmu'] + p['bmu'], h @ p['Wv'] + p['bv']
    sd = np.exp(0.5 * lv)
    z = mu + sd * eps
    g = np.tanh(z @ p['W2'] + p['b2'])
    lg = g @ p['W3'] + p['b3']
    rec = (np.maximum(lg, 0) - lg * x + np.log1p(np.exp(-np.abs(lg)))).sum()
    kl = 0.5 * (mu ** 2 + np.exp(lv) - lv - 1).sum()
    loss = float((rec + kl) / B)
    # backward
    dlg = (1 / (1 + np.exp(-lg)) - x) / B
    G = {'W3': g.T @ dlg, 'b3': dlg.sum(0)}
    dpre2 = (dlg @ p['W3'].T) * (1 - g ** 2)
    G['W2'], G['b2'] = z.T @ dpre2, dpre2.sum(0)
    dz = dpre2 @ p['W2'].T
    dmu = dz + mu / B                                   # reconstruction path + KL
    dlv = dz * eps * 0.5 * sd + 0.5 * (np.exp(lv) - 1) / B
    G['Wmu'], G['bmu'] = h.T @ dmu, dmu.sum(0)
    G['Wv'], G['bv'] = h.T @ dlv, dlv.sum(0)
    dpre1 = (dmu @ p['Wmu'].T + dlv @ p['Wv'].T) * (1 - h ** 2)
    G['W1'], G['b1'] = x.T @ dpre1, dpre1.sum(0)
    return loss, {k: np.asarray(p[k], dtype=float) - lr * G[k] for k in p}
'''

HINTS = [
    'Do the forward pass keeping $h, \\mu, \\log\\sigma^2, \\sigma, z, g$ and the logits. BCE with logits has gradient $\\sigma(\\ell) - x$ (divide by $B$).',
    'Through the sample: $\\partial z/\\partial\\mu = 1$ and $\\partial z/\\partial\\log\\sigma^2 = \\tfrac12\\sigma\\epsilon$. The KL term adds $\\mu/B$ to the $\\mu$ gradient and $\\tfrac12(\\sigma^2 - 1)/B$ to the $\\log\\sigma^2$ gradient.',
    'Both heads feed back into $h$: sum `dmu @ Wmu.T + dlv @ Wv.T` before multiplying by $1 - h^2$.',
]

_r = np.random.default_rng(2013)
def _p(D, H, Z):
    n = lambda *s: _r.normal(size=s)
    return dict(W1=n(D, H) / np.sqrt(D), b1=0.1 * n(H), Wmu=n(H, Z) / np.sqrt(H), bmu=0.1 * n(Z), Wv=n(H, Z) / np.sqrt(H) * 0.5,
                bv=0.1 * n(Z), W2=n(Z, H) / np.sqrt(Z), b2=0.1 * n(H), W3=n(H, D) / np.sqrt(H), b3=0.1 * n(D))
_p1, _p2 = _p(4, 3, 2), _p(8, 6, 3)
TESTS = [
    ((_r.random((2, 4)) > 0.5).astype(float), np.zeros((2, 2)), _p1, 0.1),
    ((_r.random((2, 4)) > 0.5).astype(float), _r.normal(size=(2, 2)), _p1, 0.1),
    ((_r.random((5, 8)) > 0.5).astype(float), _r.normal(size=(5, 3)), _p2, 0.05),
    (_r.random((5, 8)), _r.normal(size=(5, 3)), _p2, 0.5),
    ((_r.random((1, 8)) > 0.3).astype(float), _r.normal(size=(1, 3)) * 2, _p2, 1.0),
]

EXPLANATION = r"""
## The reparameterisation trick
Sampling $z \sim \mathcal N(\mu, \sigma^2)$ directly has no gradient with respect to $\mu$ and $\sigma$. Writing $z = \mu + \sigma\epsilon$ with $\epsilon \sim \mathcal N(0, I)$ drawn outside the model moves the randomness into an input, so $z$ is a deterministic, differentiable function of the encoder outputs. This low-variance gradient estimator is what made VAEs practical.

## The two terms
The reconstruction term rewards decoding $x$ well from a sampled $z$; the KL term pulls each posterior $q(z|x)$ toward the prior $\mathcal N(0, I)$, which keeps the latent space smooth enough to sample from. Their sum is the negative evidence lower bound (ELBO).

## Why predict $\log\sigma^2$
It is unconstrained, so the encoder needs no positivity activation, and the KL formula uses it directly.

## Checking the gradients
Every gradient here can be checked with central differences on the loss; that is the standard way to debug a hand-written backward pass.

## Common bugs
- Forgetting the $\tfrac12\sigma\epsilon$ factor (gradient through $\sigma = e^{\log\sigma^2/2}$).
- Dropping the KL contribution to the $\mu$ and $\log\sigma^2$ gradients.
- Computing BCE from $\hat x$ with `log`, which gives `-inf` when $\hat x$ rounds to 0 or 1.

## Interview follow-ups
- What is posterior collapse, and how does KL annealing address it?
- How would the loss change for real-valued data with a Gaussian decoder?
"""
