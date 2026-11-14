import numpy as np

META = dict(title='Latent diffusion: cross-attention conditioning, guidance and DDIM', tags=['Diffusion Models', 'Conditional Generation'],
            diff='Hard', runtime='numpy', lessons=['dlcore:07.05', 'ieng:26.08'],
            related=['ddim-step', 'classifier-free-guidance', 'autoencoder-forward-pass', 'spatial-self-attention-block'])

STATEMENT = r"""
Latent diffusion runs the diffusion process in the compact latent space of a pretrained autoencoder and conditions the denoiser on text through **cross-attention**. Sampling uses classifier-free guidance and a deterministic DDIM schedule, then decodes the latent. Implement the sampling pipeline with tiny fixed weights.

Write `ldm_sample(zT, cond, null, p, ts, abar, w)` returning `(x, z0)`.

## Denoiser
Latents are sets of $N$ tokens, $z$ `(N, Dz)`; conditioning is a sequence $c$ `(L, Dc)`.
$$\hat\epsilon(z, t, c) = zW + \text{temb}[t] + \mathrm{softmax}\!\Big(\frac{(zW_q)(cW_k)^\top}{\sqrt{d}}\Big)(cW_v)\,W_o$$
with `W` `(Dz, Dz)`, `temb` `(T, Dz)`, `Wq` `(Dz, d)`, `Wk`, `Wv` `(Dc, d)`, `Wo` `(d, Dz)`; softmax over the $L$ conditioning tokens.

## Guidance
$\hat\epsilon = \hat\epsilon(z, t, \text{null}) + w\,\big(\hat\epsilon(z, t, \text{cond}) - \hat\epsilon(z, t, \text{null})\big)$, where `null` is the "empty prompt" sequence.

## DDIM ($\eta = 0$)
`ts` is a decreasing list of timesteps (indices into `abar`, the cumulative $\bar\alpha$ table). Starting from $z$ = `zT`, for each $t$ in `ts` with $t'$ the next entry (or $\bar\alpha_{t'} = 1$ after the last):
$\hat z_0 = (z - \sqrt{1 - \bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$, then $z \leftarrow \sqrt{\bar\alpha_{t'}}\,\hat z_0 + \sqrt{1 - \bar\alpha_{t'}}\,\hat\epsilon$.

## Decode
`z0` is the final latent. The autoencoder's latents were scaled by `p['scale']` for diffusion, so decode $x = (z_0/\text{scale})\,W_{dec}$ with `W_dec` `(Dz, Dx)`.
"""

STARTER = '''import numpy as np

def ldm_sample(zT, cond, null, p, ts, abar, w):
    """Return (x (N, Dx), z0 (N, Dz))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _eps(z, t, c, p):
    q, k, v = z @ p['Wq'], c @ p['Wk'], c @ p['Wv']
    s = q @ k.T / np.sqrt(q.shape[1])
    a = np.exp(s - s.max(1, keepdims=True))
    a /= a.sum(1, keepdims=True)
    return z @ p['W'] + p['temb'][t] + (a @ v) @ p['Wo']

def ldm_sample(zT, cond, null, p, ts, abar, w):
    z = np.asarray(zT, dtype=float)
    abar = np.asarray(abar, dtype=float)
    ts = [int(t) for t in ts]
    for i, t in enumerate(ts):
        eu = _eps(z, t, null, p)
        e = eu + w * (_eps(z, t, cond, p) - eu)
        x0 = (z - np.sqrt(1 - abar[t]) * e) / np.sqrt(abar[t])
        ap = abar[ts[i + 1]] if i + 1 < len(ts) else 1.0
        z = np.sqrt(ap) * x0 + np.sqrt(1 - ap) * e
    return (z / p['scale']) @ p['W_dec'], z
'''

HINTS = [
    'Write the denoiser as one function of `(z, t, c)`. The latent tokens are the queries; the conditioning tokens provide keys and values.',
    'Each DDIM step calls the denoiser twice (null and real prompt), mixes them with the guidance weight, predicts $\\hat z_0$, and re-noises it to the next (lower) noise level.',
    'After the last listed timestep the target is the clean latent: use $\\bar\\alpha = 1$. Divide by the scale factor before decoding.',
]

_r = np.random.default_rng(2112)
def _p(Dz, Dc, d, Dx, T):
    n = lambda *s: _r.normal(size=s)
    return dict(W=0.3 * n(Dz, Dz), temb=0.1 * n(T, Dz), Wq=n(Dz, d) / np.sqrt(Dz), Wk=n(Dc, d) / np.sqrt(Dc),
                Wv=n(Dc, d) / np.sqrt(Dc), Wo=n(d, Dz) / np.sqrt(d), W_dec=n(Dz, Dx), scale=0.18215)
_T = 50
_abar = np.cumprod(1 - np.linspace(1e-4, 0.05, _T))
_p1, _p2 = _p(2, 3, 2, 4, _T), _p(4, 5, 3, 6, _T)
TESTS = [
    (_r.normal(size=(2, 2)), _r.normal(size=(3, 3)), np.zeros((1, 3)), _p1, [49], _abar, 0.0),
    (_r.normal(size=(2, 2)), _r.normal(size=(3, 3)), np.zeros((1, 3)), _p1, [49, 24, 0], _abar, 1.0),
    (_r.normal(size=(3, 4)), _r.normal(size=(4, 5)), 0.1 * _r.normal(size=(2, 5)), _p2, list(range(49, -1, -10)), _abar, 3.0),
    (_r.normal(size=(3, 4)), _r.normal(size=(4, 5)), 0.1 * _r.normal(size=(2, 5)), _p2, list(range(49, -1, -5)), _abar, 7.5),
    (_r.normal(size=(1, 4)), _r.normal(size=(1, 5)), np.zeros((1, 5)), _p2, [30, 10], _abar, 2.0),
]

EXPLANATION = r"""
## Why diffuse in latent space
Pixel-space diffusion spends most of its compute modelling imperceptible high-frequency detail. A perceptual autoencoder compresses a 512×512×3 image to a 64×64×4 latent (8× downsampling per side), and the diffusion model only has to model that. The paper showed similar quality at a fraction of the training and sampling cost.

## The scale factor
Autoencoder latents don't have unit variance. Stable Diffusion multiplies them by about 0.18215 so the diffusion process starts from data with roughly unit variance, matching the noise; decoding must undo it.

## Cross-attention conditioning
Each latent position queries the text tokens and pulls in what is relevant to it. The same mechanism works for any token-shaped condition (text encoder outputs, layouts, class embeddings).

## Guidance and DDIM
Classifier-free guidance pushes the prediction away from the unconditional one, trading diversity for prompt adherence ($w \approx 7.5$ is a common default). DDIM with $\eta = 0$ makes sampling deterministic given $z_T$ and lets you skip most timesteps: tests 3 and 4 use 5 and 10 steps.

## Common bugs
- Forgetting the scale factor when decoding.
- Swapping $\hat\epsilon_{cond}$ and $\hat\epsilon_{null}$ in the guidance formula.
- Re-noising with fresh random noise (that is DDPM, $\eta = 1$), or using $\bar\alpha_{t'}$ from a dense schedule instead of the next listed step.

## Interview follow-ups
- Where does the cost go in one sampling step at high resolution: the U-Net, the guidance doubling, or the decoder?
- How would you implement guidance with one batched forward pass instead of two?
"""
