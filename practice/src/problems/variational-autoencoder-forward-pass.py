import numpy as np

META = dict(title='VAE forward pass and the ELBO', tags=['Generative Models', 'Probability', 'Loss Functions'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:01.12', 'air-fanout-v1:01.11'],
            related=['autoencoder-forward-pass', 'kl-divergence-discrete-and-continuous', 'gan-training-step'])

STATEMENT = r"""
A variational autoencoder's encoder outputs a Gaussian $q(z \mid x) = \mathcal N(\mu, \operatorname{diag}(e^{\ell}))$, where $\ell$ is the log-variance. A sample is drawn with the **reparameterisation trick**, $z = \mu + e^{\ell/2} \odot \varepsilon$ with $\varepsilon \sim \mathcal N(0, I)$, so gradients can flow through $\mu$ and $\ell$. The training loss is the negative ELBO:

$$\mathcal L = \underbrace{-\log p(x \mid z)}_{\text{reconstruction}} + \beta\,\underbrace{\mathrm{KL}\big(q(z\mid x)\,\Vert\,\mathcal N(0, I)\big)}_{\text{regulariser}}.$$

Write `vae_forward(X, eps, p, beta=1.0)` that returns a dict.

## Network (`p` is a dict of arrays, weights `(in, out)`)
1. $h = \mathrm{ReLU}(X W_1 + b_1)$.
2. $\mu = h W_\mu + b_\mu$, $\ell = h W_\ell + b_\ell$ (keys `Wmu`, `bmu`, `Wlv`, `blv`).
3. $z = \mu + e^{\ell/2} \odot \varepsilon$, where `eps` is given, shape `(N, Z)`.
4. $g = \mathrm{ReLU}(z W_2 + b_2)$, logits $s = g W_3 + b_3$; the decoder is Bernoulli with mean $\sigma(s)$.

## Losses (per sample, then averaged over the batch)
- Reconstruction: binary cross-entropy summed over the $D$ features, computed stably from the logits.
- KL in closed form, summed over the $Z$ latent dimensions: $\mathrm{KL} = -\tfrac12 \sum_j \left(1 + \ell_j - \mu_j^2 - e^{\ell_j}\right)$.

## Output
A dict with keys `'mu'`, `'logvar'`, `'z'` (each `(N, Z)`), `'x_mean'` ($\sigma(s)$, `(N, D)`), and Python floats `'recon'`, `'kl'` (batch means) and `'loss'` $= $ `recon + beta * kl`.
"""

STARTER = '''import numpy as np

def vae_forward(X, eps, p, beta=1.0):
    """Return {'mu', 'logvar', 'z', 'x_mean', 'recon', 'kl', 'loss'}."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def vae_forward(X, eps, p, beta=1.0):
    P = {k: np.asarray(v, dtype=float) for k, v in p.items()}
    X = np.asarray(X, dtype=float)
    h = np.maximum(X @ P['W1'] + P['b1'], 0.0)
    mu, lv = h @ P['Wmu'] + P['bmu'], h @ P['Wlv'] + P['blv']
    z = mu + np.exp(0.5 * lv) * eps                        # reparameterisation: noise enters from outside
    s = np.maximum(z @ P['W2'] + P['b2'], 0.0) @ P['W3'] + P['b3']
    e = np.exp(-np.abs(s))
    x_mean = np.where(s >= 0, 1 / (1 + e), e / (1 + e))
    recon = (np.maximum(s, 0.0) + np.log1p(e) - X * s).sum(axis=1)     # BCE from logits, per sample
    kl = -0.5 * (1 + lv - mu ** 2 - np.exp(lv)).sum(axis=1)
    return {'mu': mu, 'logvar': lv, 'z': z, 'x_mean': x_mean,
            'recon': float(recon.mean()), 'kl': float(kl.mean()), 'loss': float((recon + beta * kl).mean())}
'''

HINTS = [
    'The encoder has one shared hidden layer and two linear heads, one for $\\mu$ and one for the log-variance $\\ell$. The standard deviation is $e^{\\ell/2}$, not $e^{\\ell}$.',
    'Reconstruction per sample: $\\sum_j \\big[\\max(s_j, 0) + \\log(1 + e^{-|s_j|}) - x_j s_j\\big]$. KL per sample: $-\\tfrac12\\sum_j (1 + \\ell_j - \\mu_j^2 - e^{\\ell_j})$.',
    'Average both over the batch, and form the loss as the mean of `recon + beta * kl` (equal to `recon_mean + beta * kl_mean`).',
]

_r = np.random.default_rng(109)
def _p(D, Hd, Z, s=0.5):
    g = lambda *sh: _r.normal(size=sh) * s
    return {'W1': g(D, Hd), 'b1': g(Hd) * 0.2, 'Wmu': g(Hd, Z), 'bmu': g(Z) * 0.2, 'Wlv': g(Hd, Z), 'blv': g(Z) * 0.2,
            'W2': g(Z, Hd), 'b2': g(Hd) * 0.2, 'W3': g(Hd, D), 'b3': g(D) * 0.2}
_zero = {'W1': np.zeros((2, 1)), 'b1': np.zeros(1), 'Wmu': np.zeros((1, 1)), 'bmu': np.zeros(1), 'Wlv': np.zeros((1, 1)),
         'blv': np.zeros(1), 'W2': np.zeros((1, 1)), 'b2': np.zeros(1), 'W3': np.zeros((1, 2)), 'b3': np.zeros(2)}
_pp = _p(6, 5, 2)
_X = _r.random((4, 6))
TESTS = [
    (np.array([[1.0, 0.0]]), np.array([[0.5]]), _zero),
    (np.array([[1.0, 0.0]]), np.array([[0.5]]), dict(_zero, bmu=np.array([2.0]), blv=np.array([np.log(4.0)]))),
    (_X, _r.normal(size=(4, 2)), _pp),
    (_X, _r.normal(size=(4, 2)), _pp, 0.25),
    (_X, np.zeros((4, 2)), _pp, 4.0),
    ((_r.random((3, 8)) > 0.5).astype(float), _r.normal(size=(3, 3)), _p(8, 6, 3, s=1.5)),
    (_r.random((2, 5)), _r.normal(size=(2, 4)), _p(5, 7, 4)),
]

EXPLANATION = r"""
## The reparameterisation trick
Sampling $z \sim \mathcal N(\mu, \sigma^2)$ directly is not differentiable in $\mu$ and $\sigma$. Writing $z = \mu + \sigma\varepsilon$ moves the randomness into an input $\varepsilon$ that doesn't depend on the parameters, so $\partial z/\partial \mu = 1$ and $\partial z/\partial \sigma = \varepsilon$. Passing `eps` as an argument also makes the forward pass deterministic and testable.

## The closed-form KL
For $q = \mathcal N(\mu, \sigma^2)$ and $p = \mathcal N(0, 1)$, $\mathrm{KL}(q\Vert p) = \tfrac12(\sigma^2 + \mu^2 - 1 - \log\sigma^2)$ per dimension. With $\ell = \log\sigma^2$ that is the formula above. Test 1 has $\mu = 0$, $\ell = 0$: the posterior equals the prior and the KL is exactly 0. Test 2 has $\mu = 2$, $\sigma^2 = 4$: $\tfrac12(4 + 4 - 1 - \log 4) \approx 2.807$.

## Why predict the log-variance
$\ell$ ranges over all reals, so the encoder head needs no positivity constraint, and $e^{\ell/2}$ is always a valid standard deviation.

## $\beta$
$\beta > 1$ ($\beta$-VAE) pushes the posterior toward the prior, trading reconstruction for a smoother, more disentangled latent space; $\beta < 1$ does the reverse. Annealing $\beta$ from 0 to 1 early in training is a standard fix for posterior collapse, where the KL drops to 0 and the decoder ignores $z$.

## Complexity
One encoder and one decoder pass, $O(N(DH + HZ))$.

## Common bugs
- Using $e^{\ell}$ as the standard deviation.
- Averaging the reconstruction over the $D$ pixels instead of summing it: relative to the KL that is an accidental $\beta = D$, and samples come out blurry.
- Sampling inside the function instead of using `eps`.

## Interview follow-ups
- Derive the Gaussian KL formula.
- How do you generate new samples from a trained VAE, and why are they often blurry?
"""
