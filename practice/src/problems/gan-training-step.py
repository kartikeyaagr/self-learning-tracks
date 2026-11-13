import numpy as np

META = dict(title='One GAN training step, gradients by hand', tags=['Generative Models', 'Training', 'Loss Functions'], diff='Hard',
            runtime='numpy', lessons=['air-fanout-v1:01.08', 'air-fanout-v1:04.13'],
            related=['variational-autoencoder-forward-pass', 'backpropagation-single-hidden-layer', 'log-loss-binary-cross-entropy'])

STATEMENT = r"""
A GAN alternates two updates: the discriminator learns to tell real samples from generated ones, then the generator learns to fool the **updated** discriminator. Implement one full step with a linear generator and a one-hidden-layer discriminator.

## Models
- Generator: $G(z) = z\,W_g + b_g$, with `Wg` `(k, d)` and `bg` `(d,)`.
- Discriminator: $h = \tanh(x W_1 + b_1)$, logit $D_s(x) = h\,w_2 + b_2$, probability $\sigma(D_s(x))$; `W1` `(d, m)`, `b1` `(m,)`, `w2` `(m,)`, `b2` a float.

## The step
1. **Discriminator.** With fake samples $\tilde x = G(z)$ held fixed,
$$L_D = \frac1N \sum_i \operatorname{softplus}(-D_s(x_i)) + \frac1M \sum_j \operatorname{softplus}(D_s(\tilde x_j)),$$
which is the BCE with real labelled 1 and fake labelled 0. Take one SGD step on `W1, b1, w2, b2` with learning rate `lr_d`.
2. **Generator.** Using the **updated** discriminator, the non-saturating loss
$$L_G = \frac1M \sum_j \operatorname{softplus}(-D_s(G(z_j))).$$
Backpropagate through the discriminator (whose weights are now frozen) into `Wg, bg` and take one SGD step with learning rate `lr_g`.

Write `gan_step(x_real, z, p, lr_d, lr_g)`, where `x_real` is `(N, d)`, `z` is `(M, k)`, and `p` is a dict with keys `Wg, bg, W1, b1, w2, b2`.

## Output
A dict with `'d_loss'` ($L_D$ before the D update), `'g_loss'` ($L_G$ with the updated D, before the G update) as Python floats, and the six **updated** parameters under their original keys (`b2` as a float). Compute softplus stably: $\operatorname{softplus}(u) = \max(u, 0) + \log(1 + e^{-|u|})$.
"""

STARTER = '''import numpy as np

def gan_step(x_real, z, p, lr_d, lr_g):
    """Return {'d_loss', 'g_loss', 'Wg', 'bg', 'W1', 'b1', 'w2', 'b2'} after one D step and one G step."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _softplus(u):
    return np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u)))

def _sigmoid(u):
    e = np.exp(-np.abs(u))
    return np.where(u >= 0, 1 / (1 + e), e / (1 + e))

def _disc(x, W1, b1, w2, b2):
    h = np.tanh(x @ W1 + b1)
    return h, h @ w2 + b2

def _disc_grads(x, dlogit, h, W1, w2):
    """Backprop a gradient on the logits (one per row of x) through D. Returns dW1, db1, dw2, db2, dx."""
    dh = np.outer(dlogit, w2) * (1 - h ** 2)              # through w2 and tanh
    return x.T @ dh, dh.sum(axis=0), h.T @ dlogit, float(dlogit.sum()), dh @ W1.T

def gan_step(x_real, z, p, lr_d, lr_g):
    x_real, z = np.asarray(x_real, dtype=float), np.asarray(z, dtype=float)
    Wg, bg, W1, b1, w2 = (np.asarray(p[k], dtype=float) for k in ('Wg', 'bg', 'W1', 'b1', 'w2'))
    b2 = float(p['b2'])
    N, M = len(x_real), len(z)

    # 1) discriminator step, fakes treated as constants
    fake = z @ Wg + bg
    hr, sr = _disc(x_real, W1, b1, w2, b2)
    hf, sf = _disc(fake, W1, b1, w2, b2)
    d_loss = float(_softplus(-sr).mean() + _softplus(sf).mean())
    gr = _disc_grads(x_real, -_sigmoid(-sr) / N, hr, W1, w2)   # d softplus(-s)/ds = -sigmoid(-s)
    gf = _disc_grads(fake, _sigmoid(sf) / M, hf, W1, w2)        # d softplus(s)/ds  =  sigmoid(s)
    W1, b1 = W1 - lr_d * (gr[0] + gf[0]), b1 - lr_d * (gr[1] + gf[1])
    w2, b2 = w2 - lr_d * (gr[2] + gf[2]), b2 - lr_d * (gr[3] + gf[3])

    # 2) generator step through the updated, frozen discriminator
    hf, sf = _disc(fake, W1, b1, w2, b2)
    g_loss = float(_softplus(-sf).mean())
    dx = _disc_grads(fake, -_sigmoid(-sf) / M, hf, W1, w2)[4]   # (M, d): gradient on the fake samples
    Wg, bg = Wg - lr_g * z.T @ dx, bg - lr_g * dx.sum(axis=0)
    return {'d_loss': d_loss, 'g_loss': g_loss, 'Wg': Wg, 'bg': bg, 'W1': W1, 'b1': b1, 'w2': w2, 'b2': float(b2)}
'''

HINTS = [
    'Everything flows from the logit gradients: $\\frac{d}{ds}\\operatorname{softplus}(s) = \\sigma(s)$ and $\\frac{d}{ds}\\operatorname{softplus}(-s) = -\\sigma(-s)$, each divided by its batch size.',
    'One helper backprops a logit gradient $g$ through D: $dh = g\\,w_2^\\top \\odot (1 - h^2)$, then $dW_1 = x^\\top dh$, $db_1 = \\sum dh$, $dw_2 = h^\\top g$, $db_2 = \\sum g$, and the input gradient $dx = dh\\,W_1^\\top$.',
    'D step: sum the real and fake parameter gradients, update. G step: recompute D on the same fakes with the **new** weights, take $dx$ from the helper, then $dW_g = z^\\top dx$ and $db_g = \\sum dx$.',
]

_r = np.random.default_rng(113)
def _p(k, d, m, s=0.5):
    return {'Wg': _r.normal(size=(k, d)) * s, 'bg': _r.normal(size=d) * 0.1, 'W1': _r.normal(size=(d, m)) * s,
            'b1': _r.normal(size=m) * 0.1, 'w2': _r.normal(size=m) * s, 'b2': float(_r.normal() * 0.1)}
_zero = {'Wg': np.zeros((1, 1)), 'bg': np.zeros(1), 'W1': np.zeros((1, 1)), 'b1': np.zeros(1), 'w2': np.zeros(1), 'b2': 0.0}
TESTS = [
    (np.array([[1.0]]), np.array([[1.0]]), _zero, 0.1, 0.1),
    (np.array([[1.0]]), np.array([[1.0]]), dict(_zero, W1=np.array([[1.0]]), w2=np.array([1.0])), 0.5, 0.5),
    (_r.normal(2, 1, size=(6, 2)), _r.normal(size=(4, 3)), _p(3, 2, 4), 0.1, 0.05),
    (_r.normal(2, 1, size=(6, 2)), _r.normal(size=(4, 3)), _p(3, 2, 4), 1.0, 1.0),
    (_r.normal(size=(5, 3)) * 3, _r.normal(size=(7, 2)), _p(2, 3, 5, s=2.0), 0.2, 0.3),
    (_r.normal(size=(3, 1)), _r.normal(size=(3, 1)), _p(1, 1, 2), 0.0, 0.1),
]

EXPLANATION = r"""
## The two losses
The discriminator minimises the usual BCE. The generator could minimise $-L_D$'s fake term, $\frac1M\sum \log(1 - \sigma(D_s(\tilde x)))$, but when D confidently rejects early, bad fakes ($\sigma \approx 0$) that function is flat and the generator learns nothing. The **non-saturating** loss $-\log\sigma(D_s(\tilde x))$ has the same fixed point but a large gradient exactly when the generator is losing.

## Order matters
D is updated first, and G's step uses the new D. During the G step D's weights are frozen: gradients flow **through** D into the generated samples, then through $G$'s linear map. In a framework you'd `detach()` the fakes for the D step and exclude D's parameters from the G optimiser; here that's explicit.

## Backprop through D
For a logit gradient $g$ (one entry per sample), $dh = g\,w_2^\top \odot (1 - h^2)$; then $dW_1 = x^\top dh$ and $dx = dh\,W_1^\top$. The same helper serves real samples, fake samples and the generator's input gradient.

## Complexity
Three discriminator forwards (real, fake, fake again) and backwards: $O((N + M)dm)$, plus $O(Mkd)$ for G.

## Common bugs
- Computing the G gradient with the **old** D weights.
- Letting the D step update G (or the G step update D).
- Mixing up the signs of $\sigma(s)$ and $-\sigma(-s)$, or averaging the real and fake terms together over $N + M$.

## Interview follow-ups
- What is mode collapse, and what do Wasserstein losses with gradient penalties change?
- Show that for a fixed G the optimal discriminator is $D^*(x) = p_\text{data}(x)/(p_\text{data}(x) + p_g(x))$.
"""
