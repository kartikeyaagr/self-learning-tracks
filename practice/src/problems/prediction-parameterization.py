import numpy as np

META = dict(title='Convert between epsilon, x0 and v predictions', tags=['Diffusion Models'], diff='Medium', runtime='numpy',
            lessons=[], related=['forward-diffusion-sample', 'min-snr-loss-weighting', 'rescale-to-zero-terminal-snr'])

STATEMENT = r"""
A denoiser can be trained to output any of three quantities. Writing $a = \sqrt{\bar\alpha_t}$ and $\sigma = \sqrt{1-\bar\alpha_t}$ (so $a^2 + \sigma^2 = 1$) and $x_t = a\,x_0 + \sigma\,\varepsilon$:

- `"eps"`: the noise $\varepsilon$
- `"x0"`: the clean sample $x_0$
- `"v"`: the velocity $v = a\,\varepsilon - \sigma\,x_0$

Given $x_t$ and the noise level, any one of them determines the other two. Samplers need this: a DDIM step wants both $\hat x_0$ and $\hat\varepsilon$, whatever the network predicts.

Write `convert_prediction(pred, x_t, alpha_bar, src, dst)`.

## Inputs
- `pred`, `x_t`: `(B, D)` float arrays. `pred` is a prediction of kind `src`.
- `alpha_bar`: `(B,)`, one $\bar\alpha_t$ per example, each in the **closed** interval $[0, 1]$.
- `src`, `dst`: each one of `"eps"`, `"x0"`, `"v"`.

## Output
`(B, D)` float array: the prediction of kind `dst`. If `src == dst`, return `pred` unchanged (as a float array).

## Edge cases
A conversion that would divide by zero is undefined; tests only use well-defined ones. Converting **from `"v"`** works for any $\bar\alpha \in [0, 1]$ without dividing, so use the division-free formulas there.
"""

STARTER = '''import numpy as np

def convert_prediction(pred, x_t, alpha_bar, src, dst):
    """Convert a src-type prediction ('eps', 'x0' or 'v') at noise level alpha_bar into a dst-type one."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def convert_prediction(pred, x_t, alpha_bar, src, dst):
    pred, x_t = np.asarray(pred, dtype=float), np.asarray(x_t, dtype=float)
    ab = np.asarray(alpha_bar, dtype=float)[:, None]
    a, s = np.sqrt(ab), np.sqrt(1.0 - ab)
    if src == dst:
        return pred
    # 1) recover x0 and eps from whatever was predicted
    if src == 'eps':
        eps = pred
        x0 = (x_t - s * eps) / a
    elif src == 'x0':
        x0 = pred
        eps = (x_t - a * x0) / s
    else:                                # v: rotate (x_t, v) back to (x0, eps), no division
        x0 = a * x_t - s * pred
        eps = s * x_t + a * pred
    # 2) build the requested target
    return {'eps': eps, 'x0': x0, 'v': a * eps - s * x0}[dst]
'''

HINTS = [
    'Go through a common middle: from any `src`, first recover both $\\hat x_0$ and $\\hat\\varepsilon$, then build `dst` from them.',
    'From $\\varepsilon$: $x_0 = (x_t - \\sigma\\varepsilon)/a$. From $x_0$: $\\varepsilon = (x_t - a x_0)/\\sigma$.',
    'Because $a^2 + \\sigma^2 = 1$, the pair $(x_t, v)$ is a rotation of $(x_0, \\varepsilon)$: $x_0 = a x_t - \\sigma v$ and $\\varepsilon = \\sigma x_t + a v$. Reshape `alpha_bar` to `(B, 1)` first.',
]

_r = np.random.default_rng(29)
_x, _p = _r.normal(size=(4, 3)), _r.normal(size=(4, 3))
_ab = np.array([0.9, 0.5, 0.1, 0.01])
TESTS = [
    (np.array([[1.0]]), np.array([[1.0]]), np.array([0.5]), 'eps', 'x0'),
    (np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), np.array([0.25]), 'v', 'eps'),
    (_p, _x, _ab, 'eps', 'v'),
    (_p, _x, _ab, 'x0', 'eps'),
    (_p, _x, _ab, 'x0', 'v'),
    (_p, _x, np.array([1.0, 0.0, 0.5, 0.75]), 'v', 'x0'),
    (_p, _x, _ab, 'v', 'v'),
    (_p, _x, _ab, 'eps', 'x0'),
]

EXPLANATION = r"""
## Why it works
$x_t = a x_0 + \sigma\varepsilon$ is one linear equation in two unknowns; a prediction of either fixes the other. For $v$, stack the two definitions:

$$\begin{pmatrix} x_t \\ v \end{pmatrix} = \begin{pmatrix} a & \sigma \\ -\sigma & a \end{pmatrix}\begin{pmatrix} x_0 \\ \varepsilon \end{pmatrix}$$

That matrix is a rotation (since $a^2 + \sigma^2 = 1$), so its inverse is its transpose. No division appears, which is why $v$ stays well-defined at both ends: at $\bar\alpha = 0$ it gives $x_0 = -v$, and at $\bar\alpha = 1$ it gives $x_0 = x_t$ (test 6).

## Why it matters
- $\varepsilon$-prediction turns into $\hat x_0 = (x_t - \sigma\hat\varepsilon)/a$, which amplifies errors by $1/a$ when the image is almost pure noise.
- $x_0$-prediction does the same to $\hat\varepsilon$ when $\sigma \to 0$.
- $v$-prediction avoids both, so it is the usual choice with zero-terminal-SNR schedules and distillation.

## Complexity
$O(BD)$.

## Common bugs
- Forgetting to reshape `alpha_bar` to `(B, 1)`, which broadcasts it across features instead of examples.
- Sign errors in $v$: it is $a\varepsilon - \sigma x_0$, not $a x_0 - \sigma\varepsilon$.
- Converting from $v$ by first computing $\varepsilon$ through a division that blows up at the edges.

## Interview follow-ups
- Show that $v = \frac{d x_t}{d\phi}$ when $a = \cos\phi$, $\sigma = \sin\phi$. Why is it called a velocity?
- How does an $\varepsilon$-MSE loss compare with an $x_0$-MSE loss in terms of per-step weighting? (They differ by a factor of SNR.)
"""
