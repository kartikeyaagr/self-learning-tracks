import numpy as np

META = dict(title='Conditional flow matching loss', tags=['Flow Matching', 'Generative Models', 'Loss Functions'], diff='Medium',
            runtime='numpy', lessons=[], related=['noise-prediction-loss', 'denoising-score-matching-loss', 'karras-euler-sampler'])

STATEMENT = r"""
Flow matching trains a velocity field $v_\theta(x, t)$ whose ODE carries noise ($t = 0$) to data ($t = 1$). Like score matching, the marginal target is intractable, but a **conditional** target for one (noise, data) pair is easy. Use the straight-line path with a small minimum width $\sigma_{\min}$:

$$x_t = \big(1 - (1-\sigma_{\min})\,t\big)\,x_0 + t\,x_1, \qquad u_t = \frac{d x_t}{dt} = x_1 - (1-\sigma_{\min})\,x_0$$

and regress $v_\theta(x_t, t)$ onto $u_t$ with mean squared error. Here $x_0$ is the noise sample and $x_1$ is the data sample.

The model is a toy linear field $v_\theta(x, t) = x\,W + t\,c + b$.

Write `cfm_loss(x0, x1, t, W, c, b, sigma_min=0.0)` and return `(x_t, u_t, loss)`.

## Inputs
- `x0`: `(B, D)` noise; `x1`: `(B, D)` data; `t`: `(B,)` times in $[0, 1]$, one per pair.
- `W`: `(D, D)` applied as `x @ W`; `c`, `b`: `(D,)`.

## Output
- `x_t`, `u_t`: `(B, D)` float arrays.
- `loss`: Python `float`, the mean of $(v_\theta(x_t, t) - u_t)^2$ over all $B\times D$ entries.
"""

STARTER = '''import numpy as np

def cfm_loss(x0, x1, t, W, c, b, sigma_min=0.0):
    """Straight-path conditional flow matching. Return (x_t, u_t, loss)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def cfm_loss(x0, x1, t, W, c, b, sigma_min=0.0):
    x0, x1, W, c, b = (np.asarray(a, dtype=float) for a in (x0, x1, W, c, b))
    tt = np.asarray(t, dtype=float)[:, None]               # (B, 1)
    k = 1.0 - sigma_min
    x_t = (1.0 - k * tt) * x0 + tt * x1
    u_t = x1 - k * x0                                      # constant along the path, so no t
    v = x_t @ W + tt * c + b
    return x_t, u_t, float(np.mean((v - u_t) ** 2))
'''

HINTS = [
    'Reshape `t` to `(B, 1)` so each pair gets its own time, then build $x_t$ by interpolating between noise and data.',
    'Differentiate the path with respect to $t$: the target velocity $x_1 - (1-\\sigma_{\\min})x_0$ does not depend on $t$ at all.',
    'The model output is `x_t @ W + t[:, None] * c + b`; the loss is `np.mean((v - u_t) ** 2)` over everything.',
]

_r = np.random.default_rng(71)
_x0, _x1 = _r.normal(size=(6, 3)), _r.normal(size=(6, 3)) * 2 + 1
_t = np.array([0.0, 0.1, 0.4, 0.5, 0.9, 1.0])
TESTS = [
    (np.array([[1.0, 0.0]]), np.array([[3.0, 2.0]]), np.array([0.5]), np.zeros((2, 2)), np.zeros(2), np.zeros(2)),
    (np.array([[1.0]]), np.array([[2.0]]), np.array([0.0]), np.zeros((1, 1)), np.zeros(1), np.array([1.0])),
    (np.array([[2.0], [2.0]]), np.array([[0.0], [0.0]]), np.array([0.0, 1.0]), np.zeros((1, 1)), np.zeros(1), np.zeros(1), 0.1),
    (_x0, _x1, _t, _r.normal(size=(3, 3)) * 0.3, _r.normal(size=3), _r.normal(size=3)),
    (_x0, _x1, _t, _r.normal(size=(3, 3)) * 0.3, _r.normal(size=3), _r.normal(size=3), 1e-3),
    (_r.normal(size=(10, 2)), _r.normal(size=(10, 2)), _r.uniform(size=10), np.eye(2), np.zeros(2), np.zeros(2), 0.05),
]

EXPLANATION = r"""
## Why it works
Each pair defines a path whose marginal over pairs, $p_t(x)$, interpolates from the noise distribution to the data distribution. The marginal velocity that generates $p_t$ is the conditional velocity averaged over all pairs passing through $x$. Because MSE regression recovers a conditional expectation, minimising the conditional loss trains $v_\theta$ to the marginal field, with gradients identical to the intractable marginal loss.

With $\sigma_{\min} = 0$ this is **rectified flow**: straight lines from noise to data, and target $x_1 - x_0$. A positive $\sigma_{\min}$ leaves the path a little noise at $t = 1$ (test 3: at $t = 1$, $x_t = \sigma_{\min}x_0 + x_1$).

## Relation to diffusion
A diffusion model with $\varepsilon$-prediction is also a velocity field along a (curved) Gaussian path; flow matching just picks the path directly. Straighter paths need fewer ODE steps to sample.

## Complexity
$O(BD^2)$ for the model; the targets are $O(BD)$.

## Common bugs
- Swapping noise and data (here $t = 0$ is noise; some papers use the opposite convention).
- Writing the target as $x_1 - x_t$ or $(x_1 - x_t)/(1-t)$: equal to $u_t$ only when $\sigma_{\min} = 0$, and the second divides by zero at $t = 1$.
- Using one $t$ for the whole batch.

## Interview follow-ups
- How would you sample from the trained model? (Integrate $dx/dt = v_\theta(x, t)$ from $t = 0$ to $1$ with Euler or Heun.)
- What does minibatch optimal-transport pairing of $x_0$ and $x_1$ change, and why does it help?
"""
