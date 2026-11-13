import numpy as np

META = dict(title='One training step of a toy diffusion model', tags=['Diffusion Models', 'Training'], diff='Hard', runtime='numpy',
            lessons=[], related=['noise-prediction-loss', 'toy-ddpm-sampler', 'forward-diffusion-sample'])

STATEMENT = r"""
Put the pieces together and train. The toy denoiser predicts noise with a shared linear map plus a learned bias **per timestep**:

$$\hat\varepsilon = x_t\,W + E_{t}, \qquad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \qquad \mathcal L = \frac{1}{BD}\sum_{b,d}\big(\hat\varepsilon_{bd} - \varepsilon_{bd}\big)^2$$

One step of training:
1. Build $x_t$ for every example (each has its own step `t[b]`).
2. Compute the prediction and the loss $\mathcal L$.
3. Compute the exact gradients $\partial\mathcal L/\partial W$ and $\partial\mathcal L/\partial E$ by hand (no autograd). Only the rows of $E$ that appear in `t` get a gradient, and a step that appears several times in the batch **accumulates** the contributions of all its examples.
4. If `clip_norm` is not `None`, compute the global norm $g = \sqrt{\lVert\nabla_W\rVert_F^2 + \lVert\nabla_E\rVert_F^2}$ and, if $g >$ `clip_norm`, multiply **both** gradients by `clip_norm / g`.
5. Plain SGD: $W \leftarrow W - \eta\nabla_W$, $E \leftarrow E - \eta\nabla_E$.

Write `train_step(x0, t, noise, alpha_bars, W, E, lr, clip_norm=None)` and return `(loss, W_new, E_new)`.

## Inputs
- `x0`, `noise`: `(B, D)`; `t`: `(B,)` integer steps (0-indexed, repeats allowed); `alpha_bars`: `(T,)`.
- `W`: `(D, D)` applied as `x_t @ W`; `E`: `(T, D)`; `lr`: float; `clip_norm`: positive float or `None`.

## Output
- `loss`: Python `float`, measured **before** the update.
- `W_new`, `E_new`: new arrays (do not modify the inputs in place).
"""

STARTER = '''import numpy as np

def train_step(x0, t, noise, alpha_bars, W, E, lr, clip_norm=None):
    """Forward-diffuse, predict eps = x_t @ W + E[t], backprop by hand, clip, SGD. Return (loss, W_new, E_new)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def train_step(x0, t, noise, alpha_bars, W, E, lr, clip_norm=None):
    x0, eps, W, E = (np.asarray(a, dtype=float) for a in (x0, noise, W, E))
    t = np.asarray(t)
    ab = np.asarray(alpha_bars, dtype=float)[t][:, None]
    x_t = np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps
    r = x_t @ W + E[t] - eps                          # residual (B, D)
    loss = float(np.mean(r ** 2))
    g = 2.0 * r / r.size                              # dL/dprediction
    dW = x_t.T @ g
    dE = np.zeros_like(E)
    np.add.at(dE, t, g)                               # unbuffered: repeated t values accumulate
    if clip_norm is not None:
        norm = np.sqrt((dW ** 2).sum() + (dE ** 2).sum())
        if norm > clip_norm:
            dW, dE = dW * (clip_norm / norm), dE * (clip_norm / norm)
    return loss, W - lr * dW, E - lr * dE
'''

HINTS = [
    'Write the residual $r = \\hat\\varepsilon - \\varepsilon$ once. The derivative of the mean squared error with respect to the prediction is $2r/(BD)$.',
    'The prediction is linear in $W$ with input $x_t$, so $\\nabla_W = x_t^\\top \\cdot \\frac{2r}{BD}$. Note $x_t$ does not depend on $W$: the noising happens before the model.',
    '`dE[t] += g` does **not** accumulate when `t` has repeats (fancy-index assignment keeps only one). Use `np.add.at(dE, t, g)`. Clip both gradients by the same factor.',
]

_r = np.random.default_rng(83)
_ab = np.cumprod(1 - np.linspace(1e-4, 0.02, 50))
_x0, _z = _r.normal(size=(6, 3)), _r.normal(size=(6, 3))
_W, _E = _r.normal(size=(3, 3)) * 0.2, _r.normal(size=(50, 3)) * 0.1
TESTS = [
    (np.array([[1.0]]), np.array([0]), np.array([[1.0]]), np.array([0.0]), np.zeros((1, 1)), np.zeros((1, 1)), 0.5),
    (np.array([[1.0], [1.0]]), np.array([0, 0]), np.array([[0.0], [0.0]]), np.array([1.0]), np.zeros((1, 1)), np.array([[1.0]]), 0.25),
    (_x0, np.array([0, 10, 10, 49, 25, 10]), _z, _ab, _W, _E, 0.1),
    (_x0, np.array([0, 10, 10, 49, 25, 10]), _z, _ab, _W, _E, 0.1, 0.05),
    (_x0, np.array([1, 2, 3, 4, 5, 6]), _z, _ab, _W, _E, 1.0, 100.0),
    (_r.normal(size=(10, 2)), _r.integers(0, 50, size=10), _r.normal(size=(10, 2)), _ab, np.eye(2), np.zeros((50, 2)), 0.3, 0.5),
]

EXPLANATION = r"""
## Why the gradients look like that
$\mathcal L = \frac1{BD}\sum r^2$, so $\partial\mathcal L/\partial\hat\varepsilon = 2r/(BD)$. The prediction is $x_tW + E_t$: the gradient flows to $W$ through $x_t^\top$ (sum of outer products over examples) and to each bias row $E_k$ as the sum of the upstream gradients of the examples with $t_b = k$. That sum is a scatter-add, which is why repeated timesteps (test 3 has step 10 three times) need `np.add.at`.

## Why clip
Rare, large residuals (an unlucky $x_0$ at a nearly clean step) can produce a huge gradient. Clipping the **global** norm keeps the update direction and caps its length, so one batch cannot wreck the parameters. Test 5 sets a loose limit that never triggers; tests 4 and 6 are tight.

## Complexity
$O(BD^2 + TD)$ per step: the matrix products plus zeroing the bias-gradient table. A sparse update would touch only the rows in `t`.

## Common bugs
- `dE[t] += g` with repeated indices silently drops contributions.
- Normalising by $B$ instead of $BD$ (the loss is a mean over all entries).
- Clipping each gradient separately, which changes the update direction.
- Mutating `W` or `E` in place: the caller still holds them, and in the harness they would be shared across tests.

## Interview follow-ups
- Run this step repeatedly with fresh $x_0$, $t$, $\varepsilon$. What does $E_t$ converge to when the data is $\mathcal N(\mu, I)$? (For a fixed $W$ it absorbs the mean of the residual at that step.)
- How would importance sampling of $t$ (by loss magnitude) change the estimator, and what reweighting keeps it unbiased?
"""
