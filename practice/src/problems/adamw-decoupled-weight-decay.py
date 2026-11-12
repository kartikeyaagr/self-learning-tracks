import numpy as np

META = dict(title='AdamW vs Adam with L2', tags=['Optimization', 'Training'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:05.06', 'air-fanout-v1:02.04'],
            related=['adam-optimizer', 'adam-step', 'lasso-regression-using-ista'])

STATEMENT = r"""
There are two ways to add weight decay to Adam, and they are **not** the same algorithm.

- **`mode='l2'`** (Adam + L2 penalty): fold the decay into the gradient, $g_t \leftarrow g_t + \lambda\,\theta_{t-1}$, then run the normal Adam update with that gradient.
- **`mode='adamw'`** (decoupled): run Adam on the raw gradient, and shrink the weights separately:
$$\theta_t = \theta_{t-1} - \eta\lambda\,\theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$
Both terms use the parameters from **before** the step.

In both modes, the Adam part is
$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$, step $\eta\,\hat m_t/(\sqrt{\hat v_t} + \epsilon)$, with $m_0 = v_0 = 0$ and $t$ starting at 1.

Decay applies only where `decay_mask` is `True` (in real models, weights decay but biases and norm gains don't). Where it is `False`, that coordinate is plain Adam in both modes.

Optimise $f(\theta) = \tfrac12\theta^\top Q\theta - b^\top\theta$ (symmetric $Q$, $\nabla f = Q\theta - b$).

Write `adam_weight_decay(Q, b, x0, lr, steps, weight_decay, decay_mask, mode, b1=0.9, b2=0.999, eps=1e-8)` that returns the trajectory, a float array `(steps + 1, d)` whose row 0 is `x0`.

## Inputs
- `Q`: float `(d, d)` symmetric; `b`, `x0`: float `(d,)`; `decay_mask`: bool `(d,)`.
- `lr > 0`, `weight_decay >= 0`, `steps >= 0`; `mode` is `'l2'` or `'adamw'`.
"""

STARTER = '''import numpy as np

def adam_weight_decay(Q, b, x0, lr, steps, weight_decay, decay_mask, mode, b1=0.9, b2=0.999, eps=1e-8):
    """mode: 'l2' (penalty in the gradient) or 'adamw' (decoupled). Return the path (steps+1, d)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def adam_weight_decay(Q, b, x0, lr, steps, weight_decay, decay_mask, mode, b1=0.9, b2=0.999, eps=1e-8):
    Q, b = np.asarray(Q, dtype=float), np.asarray(b, dtype=float)
    x = np.array(x0, dtype=float)
    wd = weight_decay * np.asarray(decay_mask, dtype=float)    # per-coordinate decay, 0 where masked out
    m, v = np.zeros_like(x), np.zeros_like(x)
    path = [x.copy()]
    for t in range(1, steps + 1):
        g = Q @ x - b
        if mode == 'l2':
            g = g + wd * x                     # the penalty's gradient goes through Adam's scaling
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        step = lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        if mode == 'adamw':
            x = x - lr * wd * x - step         # decay acts on the weights directly, unscaled by v
        else:
            x = x - step
        path.append(x.copy())
    return np.array(path)
'''

HINTS = [
    'Turn the mask into a per-coordinate decay vector, `weight_decay * decay_mask`. Then both modes are one line different from plain Adam.',
    'In `l2` mode the decay term $\\lambda\\theta$ is added to the gradient **before** the moment updates, so it gets averaged into $m$ and normalised by $\\sqrt{v}$. In `adamw` mode it never touches $m$ or $v$.',
    'For `adamw`, compute the Adam step from the old `x`, then set `x = x - lr * wd * x - step` in one expression so both terms use the old parameters.',
]

_r = np.random.default_rng(31)
_M = _r.normal(size=(5, 5))
_Q = _M @ _M.T / 5 + 0.1 * np.eye(5)
_b, _x0 = _r.normal(size=5), _r.normal(size=5) * 2
_mask = np.array([True, True, False, True, False])
_ill = np.diag([100.0, 0.01])
TESTS = [
    (np.array([[1.0]]), np.array([0.0]), np.array([1.0]), 0.1, 2, 0.5, np.array([True]), 'adamw'),
    (np.array([[1.0]]), np.array([0.0]), np.array([1.0]), 0.1, 2, 0.5, np.array([True]), 'l2'),
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.1, 3, 1.0, np.array([True, False]), 'adamw'),
    (_ill, np.array([1.0, 1.0]), np.array([1.0, 1.0]), 0.05, 60, 0.1, np.array([True, True]), 'adamw'),
    (_ill, np.array([1.0, 1.0]), np.array([1.0, 1.0]), 0.05, 60, 0.1, np.array([True, True]), 'l2'),
    (_Q, _b, _x0, 0.02, 80, 0.05, _mask, 'adamw'),
    (_Q, _b, _x0, 0.02, 80, 0.05, _mask, 'l2'),
    (_Q, _b, _x0, 0.02, 0, 0.05, _mask, 'l2'),
    (_Q, _b, _x0, 0.02, 30, 0.0, _mask, 'adamw'),
]
META['tol'] = [1e-6, 1e-7]

EXPLANATION = r"""
## Why the two modes differ
With SGD, adding $\lambda\theta$ to the gradient and shrinking $\theta$ by $\eta\lambda\theta$ are the same thing. With Adam they aren't: in `l2` mode the decay gradient is divided by $\sqrt{\hat v}$ like everything else. Coordinates with large gradient history (big $v$) get **less** effective decay, and coordinates with tiny gradients get decay amplified up to about $\eta$ per step. The regulariser's strength ends up depending on the loss's gradient scale, which is rarely what you want.

AdamW decouples them: the shrink $\theta \leftarrow (1 - \eta\lambda)\theta$ is the same relative amount for every decayed weight, and Adam's normalisation only sees the loss gradient. Loshchilov and Hutter showed this makes $\lambda$ and $\eta$ much less entangled to tune, and it's the default in transformer training.

## Reading the tests
Tests 4 and 5 use $Q = \mathrm{diag}(100, 0.01)$. In `l2` mode the Adam step can only vanish where the penalised gradient does, so any resting point solves $(Q + \lambda I)\theta = b$: the ridge solution. In `adamw` mode the iterate rests where the normalised Adam step exactly cancels the shrink $\eta\lambda\theta$, a balance measured in Adam's unitless step size rather than in raw gradient units. The flat coordinate (curvature 0.01) is where the two trajectories separate most.

## Complexity
$O(d^2)$ per step for the gradient, $O(d)$ for the update, $2d$ floats of optimizer state.

## Common bugs
- Decaying with the **new** parameters (after the Adam step) instead of the old ones.
- Scaling decoupled decay by $\lambda$ only, not $\eta\lambda$. PyTorch's `AdamW` multiplies by the learning rate, so a schedule also schedules the decay.
- Applying decay to masked-out coordinates, or skipping their Adam update entirely.

## Interview follow-ups
- Why exclude biases and LayerNorm gains from weight decay? (They don't contribute to the function's complexity the way weight matrices do; shrinking a norm gain toward 0 only hurts.)
- With a cosine schedule driving $\eta \to 0$, what happens to the effective decay in AdamW? (It anneals too, since the shrink is $\eta\lambda$ per step.)
"""
