import numpy as np

META = dict(title='Flow matching end to end: train a velocity field, integrate the ODE', tags=['Flow Matching', 'ODE Solvers'],
            diff='Hard', runtime='numpy', lessons=['dlcore:07.06'],
            related=['conditional-flow-matching-loss', 'karras-euler-sampler', 'prediction-parameterization', 'ddpm-train-and-sample'])

STATEMENT = r"""
Flow matching trains a velocity field $v_\theta(x, t)$ by regressing onto the velocity of a simple path from noise to data, and generates by integrating the ODE $dx/dt = v_\theta(x, t)$ from $t = 0$ to $1$. Do both with a linear model.

Write `flow_matching(x1s, x0s, ts, z, sigma_min, p, lr, n_steps, method)` returning `(losses, p_trained, samples)`.

## Optimal-transport conditional path (Lipman et al.)
For data $x_1$ and noise $x_0$: $x_t = (1 - (1 - \sigma_{min})t)\,x_0 + t\,x_1$, with target velocity $u_t = x_1 - (1 - \sigma_{min})\,x_0$.

## Model
$v(x, t) = xW + t\,a + b$ with `p = {'W': (D, D), 'a': (D,), 'b': (D,)}`.

## Training
`x1s`, `x0s` `(I, B, D)`, `ts` `(I, B)` in $[0, 1]$. Iteration $i$: loss $= \mathrm{mean}\,(v(x_t, t) - u_t)^2$ over all $B\cdot D$ entries; record it, then one SGD step on `W`, `a`, `b` with rate `lr`.

## Sampling
Start from `z` `(S, D)` at $t = 0$ and take `n_steps` equal steps of size $h = 1/n$ with the trained model:
- `'euler'`: $x \leftarrow x + h\,v(x, t)$.
- `'midpoint'`: $x_{mid} = x + \tfrac h2 v(x, t)$, then $x \leftarrow x + h\,v(x_{mid}, t + \tfrac h2)$.

Return `losses` `(I,)`, the trained parameters as a dict (new arrays) and `samples` `(S, D)`.
"""

STARTER = '''import numpy as np

def flow_matching(x1s, x0s, ts, z, sigma_min, p, lr, n_steps, method):
    """Return (losses, p_trained, samples)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def flow_matching(x1s, x0s, ts, z, sigma_min, p, lr, n_steps, method):
    W, a, b = (np.array(p[k], dtype=float) for k in ('W', 'a', 'b'))
    v = lambda x, t: x @ W + np.multiply.outer(t, a) + b if np.ndim(t) else x @ W + t * a + b
    losses = []
    for x1, x0, t in zip(*(np.asarray(u, dtype=float) for u in (x1s, x0s, ts))):
        tc = t[:, None]
        xt = (1 - (1 - sigma_min) * tc) * x0 + tc * x1
        ut = x1 - (1 - sigma_min) * x0
        diff = v(xt, t) - ut
        losses.append(float((diff ** 2).mean()))
        g = 2 * diff / diff.size
        W -= lr * xt.T @ g
        a -= lr * (t @ g)
        b -= lr * g.sum(0)
    x, h = np.asarray(z, dtype=float).copy(), 1.0 / n_steps
    for k in range(n_steps):
        t = k * h
        if method == 'euler':
            x = x + h * v(x, t)
        else:
            x = x + h * v(x + 0.5 * h * v(x, t), t + 0.5 * h)
    return np.array(losses), {'W': W, 'a': a, 'b': b}, x
'''

HINTS = [
    'Build $x_t$ and $u_t$ row by row with `t[:, None]`; the residual is $v(x_t, t) - u_t$ and the loss gradient with respect to it is $2(\\cdot)/(BD)$.',
    'Gradients: $\\partial/\\partial W = x_t^\\top g$, $\\partial/\\partial a = t^\\top g$ (each row weighted by its own $t$), $\\partial/\\partial b = \\sum_{rows} g$.',
    'During sampling $t$ is a scalar shared by every sample; the midpoint method evaluates the field twice per step.',
]

_r = np.random.default_rng(2210)
def _case(I, B, D, S):
    data = np.array([2.0, -1.0, 0.5][:D])
    return (data + 0.2 * _r.normal(size=(I, B, D)), _r.normal(size=(I, B, D)), _r.random((I, B)), _r.normal(size=(S, D)))
_z2 = dict(W=np.zeros((2, 2)), a=np.zeros(2), b=np.zeros(2))
_z3 = dict(W=np.zeros((3, 3)), a=np.zeros(3), b=np.zeros(3))
TESTS = [
    (*_case(1, 4, 2, 3), 0.0, _z2, 0.1, 1, 'euler'),
    (*_case(1, 4, 2, 3), 1e-4, _z2, 0.1, 4, 'midpoint'),
    (*_case(50, 16, 2, 5), 1e-4, _z2, 0.3, 10, 'euler'),
    (*_case(50, 16, 2, 5), 1e-4, _z2, 0.3, 10, 'midpoint'),
    (*_case(80, 8, 3, 4), 1e-3, dict(W=0.1 * _r.normal(size=(3, 3)), a=np.zeros(3), b=np.zeros(3)), 0.2, 20, 'midpoint'),
    (*_case(3, 2, 3, 2), 0.0, _z3, 0.0, 2, 'euler'),
]

EXPLANATION = r"""
## Why this works
The marginal velocity field that moves the whole noise distribution onto the data distribution is intractable, but regressing onto per-sample conditional velocities $u_t(x|x_1)$ has the same minimiser in expectation. So training is simulation-free (no ODE solve, like diffusion's closed-form noising) and the loss is a plain regression.

## The OT path
Straight lines from noise to data with constant speed give a velocity that doesn't depend on $t$ for a fixed pair, so the learned field tends to have straighter trajectories than diffusion's curved ones. Straighter paths need fewer ODE steps; rectified flow pushes the same idea further. $\sigma_{min}$ leaves a little noise at $t = 1$ and is often set to 0 in practice.

## Euler vs midpoint
Euler uses one field evaluation per step and has $O(h)$ global error; the midpoint method uses two and has $O(h^2)$. For a fixed budget of evaluations, a higher-order solver with fewer steps is often better, which is why samplers like Heun's method are popular.

## Relation to diffusion
Diffusion with the probability-flow ODE is a flow with a particular (variance-preserving) path, and $\epsilon$-, $x_0$- and $v$-prediction can be converted into a velocity. Recent image models (e.g. Stable Diffusion 3) train with the straight path instead.

## Common bugs
- Target $x_1 - x_0$ when $\sigma_{min} > 0$ (it must be $x_1 - (1 - \sigma_{min})x_0$).
- Integrating from $t = 1$ to 0 (that is the diffusion convention; here noise is at $t = 0$).
- Evaluating the midpoint field at $t$ instead of $t + h/2$.

## Interview follow-ups
- Why can flow matching use any interpolation path, and what does the choice change?
- How would you add classifier-free guidance to a velocity model?
"""
