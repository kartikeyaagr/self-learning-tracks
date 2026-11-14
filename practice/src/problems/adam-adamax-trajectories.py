import numpy as np

META = dict(title='Adam, AdaMax and the bias correction: optimiser trajectories on a noisy quadratic', tags=['Optimization', 'Training'],
            diff='Hard', runtime='numpy', lessons=['mlmaths:14.03', 'air-fanout-v1:05.06'],
            related=['adam-step', 'adam-optimizer', 'rmsprop', 'classical-momentum', 'optimizer-showdown'])

STATEMENT = r"""
Run Adam and its infinity-norm variant AdaMax for several steps on an ill-conditioned quadratic with noisy gradients, and see what happens without the bias correction.

Write `run_optimizer(A, c, theta0, noise, lr, beta1, beta2, eps, method)` returning the trajectory `(T + 1, D)`: $\theta_0$ followed by the parameters after each step.

## Objective
$f(\theta) = \tfrac12(\theta - c)^\top A(\theta - c)$ with symmetric `A` `(D, D)`. At step $t = 1..T$ the gradient is $g_t = A(\theta_{t-1} - c) + \text{noise}[t - 1]$, with `noise` `(T, D)`.

## Methods ($m_0 = v_0 = u_0 = 0$)
- `'adam'`: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$, $\hat m = m_t/(1-\beta_1^t)$, $\hat v = v_t/(1-\beta_2^t)$, $\theta_t = \theta_{t-1} - \text{lr}\,\hat m/(\sqrt{\hat v} + \epsilon)$.
- `'adam_nobias'`: the same with $\hat m = m_t$, $\hat v = v_t$.
- `'adamax'`: the same $m_t$; $u_t = \max(\beta_2 u_{t-1}, |g_t|)$; $\theta_t = \theta_{t-1} - \frac{\text{lr}}{1-\beta_1^t}\,m_t/u_t$ (no $\epsilon$).
"""

STARTER = '''import numpy as np

def run_optimizer(A, c, theta0, noise, lr, beta1, beta2, eps, method):
    """Return the trajectory (T + 1, D)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def run_optimizer(A, c, theta0, noise, lr, beta1, beta2, eps, method):
    A, c, noise = (np.asarray(a, dtype=float) for a in (A, c, noise))
    th = np.array(theta0, dtype=float)
    m, v = np.zeros_like(th), np.zeros_like(th)
    traj = [th.copy()]
    for t in range(1, len(noise) + 1):
        g = A @ (th - c) + noise[t - 1]
        m = beta1 * m + (1 - beta1) * g
        if method == 'adamax':
            v = np.maximum(beta2 * v, np.abs(g))
            th = th - lr / (1 - beta1 ** t) * m / v
        else:
            v = beta2 * v + (1 - beta2) * g * g
            if method == 'adam':
                mh, vh = m / (1 - beta1 ** t), v / (1 - beta2 ** t)
            else:
                mh, vh = m, v
            th = th - lr * mh / (np.sqrt(vh) + eps)
        traj.append(th.copy())
    return np.array(traj)
'''

HINTS = [
    'Recompute the gradient from the current $\\theta$ every step; the noise row is indexed by $t - 1$.',
    'Bias correction divides by $1 - \\beta^t$ with $t$ starting at 1. AdaMax reuses $m_t$ but replaces $v_t$ with an exponentially weighted infinity norm that needs no correction.',
    'For `adam_nobias` the first step is $\\text{lr}\\,(1-\\beta_1)g/(\\sqrt{1-\\beta_2}\\,|g| + \\epsilon)$, about $3\\times$ larger than Adam\'s for the default betas.',
]

_r = np.random.default_rng(2017)
def _quad(D, cond):
    Q, _ = np.linalg.qr(_r.normal(size=(D, D)))
    return Q @ np.diag(np.geomspace(1, cond, D)) @ Q.T
_A2, _A3 = _quad(2, 50.0), _quad(3, 100.0)
TESTS = [
    (_A2, np.array([1.0, -1.0]), np.zeros(2), np.zeros((3, 2)), 0.1, 0.9, 0.999, 1e-8, 'adam'),
    (_A2, np.array([1.0, -1.0]), np.zeros(2), np.zeros((3, 2)), 0.1, 0.9, 0.999, 1e-8, 'adam_nobias'),
    (_A2, np.array([1.0, -1.0]), np.zeros(2), 0.5 * _r.normal(size=(50, 2)), 0.05, 0.9, 0.999, 1e-8, 'adamax'),
    (_A3, _r.normal(size=3), _r.normal(size=3), 0.3 * _r.normal(size=(60, 3)), 0.05, 0.9, 0.999, 1e-8, 'adam'),
    (_A3, _r.normal(size=3), _r.normal(size=3), 0.3 * _r.normal(size=(60, 3)), 0.05, 0.9, 0.99, 1e-8, 'adam_nobias'),
    (_A3, _r.normal(size=3), _r.normal(size=3), 0.3 * _r.normal(size=(60, 3)), 0.02, 0.8, 0.99, 1e-8, 'adamax'),
]

EXPLANATION = r"""
## What Adam does
Adam keeps an exponential average of the gradient (momentum) and of its square, and divides one by the square root of the other. Each coordinate's step is roughly $\text{lr}$ times a signal-to-noise ratio, so steep and shallow directions of an ill-conditioned problem move at similar speeds, and the scale of the gradient doesn't matter.

## Why the bias correction
Both averages start at zero, so early on they are biased towards zero by factors $1 - \beta_1^t$ and $1 - \beta_2^t$. Because $\beta_2$ is close to 1, the second moment is far more biased: without correction the first steps are inflated by about $(1-\beta_1)/\sqrt{1-\beta_2} \approx 3.2$ for the defaults. Tests 1 and 2 show this on the same problem.

## AdaMax
Replacing the $L^2$ average with an $L^\infty$ one gives $u_t = \max(\beta_2u_{t-1}, |g_t|)$, which is not biased towards zero, so only $m_t$ needs correcting. It can be more stable when gradients are occasionally huge.

## Common bugs
- Starting $t$ at 0 in the bias correction (division by zero) or not incrementing it.
- Putting $\epsilon$ inside the square root (some libraries do; the paper doesn't).
- Adding weight decay to the gradient and calling it AdamW: decoupled decay is applied to $\theta$ directly.

## Interview follow-ups
- Why can Adam converge to worse solutions than SGD with momentum on some vision tasks?
- What does $\epsilon$ control when gradients are tiny, and why do some LLM recipes raise it?
"""
