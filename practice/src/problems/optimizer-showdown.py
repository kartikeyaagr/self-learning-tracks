import numpy as np

META = dict(title='Optimizer showdown on an ill-conditioned bowl', tags=['Optimization', 'Training'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:05.06', 'mlmaths:07.01'],
            related=['vanilla-gradient-descent', 'classical-momentum', 'nesterov-momentum', 'rmsprop', 'adam-optimizer'])

STATEMENT = r"""
Build a small benchmark harness that races optimizers on the same problem and ranks them. The objective is

$$f(x) = \tfrac12 x^\top Q x - b^\top x, \qquad Q \text{ symmetric positive definite}, \qquad f^\star = f(Q^{-1} b),$$

and the score of an iterate is its **gap** $f(x_t) - f^\star$.

Each entry of `configs` is a dict with keys `'name'` (a label), `'kind'`, `'lr'`, and optionally the hyperparameters below. Every run starts at `x0` with all state at zero, uses $g_t = \nabla f(x_{t-1}) = Qx_{t-1} - b$, and steps $t = 1, 2, \dots$:

| kind | update | defaults |
|---|---|---|
| `'gd'` | $x \leftarrow x - \eta g$ | |
| `'momentum'` | $v \leftarrow \mu v - \eta g$; $x \leftarrow x + v$ | `mu=0.9` |
| `'nesterov'` | $v \leftarrow \mu v - \eta \nabla f(x + \mu v)$; $x \leftarrow x + v$ | `mu=0.9` |
| `'rmsprop'` | $s \leftarrow \rho s + (1-\rho) g^2$; $x \leftarrow x - \eta g / (\sqrt{s} + \epsilon)$ | `rho=0.9`, `eps=1e-8` |
| `'adam'` | $m, v$ EMAs with `b1`, `b2`; bias-corrected; $x \leftarrow x - \eta \hat m / (\sqrt{\hat v} + \epsilon)$ | `b1=0.9`, `b2=0.999`, `eps=1e-8` |

Write `optimizer_showdown(Q, b, x0, configs, target, max_steps)` that runs each config for `max_steps` steps and returns a **ranked** list of tuples `(name, steps_to_target, final_gap)`:

- `steps_to_target`: the smallest $t \in \{0, \dots, \text{max\_steps}\}$ with gap $\le$ `target` ($t = 0$ means `x0` already qualifies), or `None` if never reached.
- `final_gap`: the gap after `max_steps` steps as a Python `float`; any non-finite value is reported as `inf`.
- Order: runs that reached the target first, by `steps_to_target` ascending; then runs that didn't, by `final_gap` ascending. Ties keep the order of `configs`.

Divergence is allowed; suppress overflow warnings with `np.errstate`.
"""

STARTER = '''import numpy as np

def optimizer_showdown(Q, b, x0, configs, target, max_steps):
    """Return a ranked list of (name, steps_to_target or None, final_gap float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _run(Q, b, x0, cfg, max_steps, f, f_star, target):
    kind, lr = cfg['kind'], cfg['lr']
    grad = lambda z: Q @ z - b
    x = x0.copy()
    m, v = np.zeros_like(x), np.zeros_like(x)          # generic state slots (velocity / moments)
    hit = 0 if f(x) - f_star <= target else None
    for t in range(1, max_steps + 1):
        if kind == 'gd':
            x = x - lr * grad(x)
        elif kind in ('momentum', 'nesterov'):
            mu = cfg.get('mu', 0.9)
            at = x + mu * v if kind == 'nesterov' else x
            v = mu * v - lr * grad(at)
            x = x + v
        elif kind == 'rmsprop':
            g = grad(x)
            v = cfg.get('rho', 0.9) * v + (1 - cfg.get('rho', 0.9)) * g * g
            x = x - lr * g / (np.sqrt(v) + cfg.get('eps', 1e-8))
        else:  # adam
            b1, b2, eps = cfg.get('b1', 0.9), cfg.get('b2', 0.999), cfg.get('eps', 1e-8)
            g = grad(x)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            x = x - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        if hit is None and f(x) - f_star <= target:
            hit = t
    gap = f(x) - f_star
    return cfg['name'], hit, float(gap) if np.isfinite(gap) else float('inf')

def optimizer_showdown(Q, b, x0, configs, target, max_steps):
    Q, b, x0 = (np.asarray(a, dtype=float) for a in (Q, b, x0))
    f = lambda z: 0.5 * z @ Q @ z - b @ z
    f_star = f(np.linalg.solve(Q, b))
    with np.errstate(over='ignore', invalid='ignore'):
        rows = [_run(Q, b, x0, c, max_steps, f, f_star, target) for c in configs]
    # sorted() is stable, so equal keys keep config order.
    return sorted(rows, key=lambda r: (r[1] is None, r[1] if r[1] is not None else 0, r[2] if r[1] is None else 0.0))
'''

HINTS = [
    'Write one `run(config)` function that returns `(name, steps_to_target, final_gap)`. Check the gap at $t = 0$ before the first step, then after every step, and remember only the first hit.',
    'Give every optimizer the same state slots (two zero vectors) and branch on `kind` inside the loop. Use `cfg.get(key, default)` for the optional hyperparameters.',
    'Rank with a stable sort on the key `(not reached, steps if reached, final_gap if not reached)`. Python\'s `sorted` keeps config order on ties. A `nan` gap must become `inf` first or the sort misbehaves.',
]

_Q = np.diag([1.0, 50.0])
_b = np.array([1.0, 1.0])
_r = np.random.default_rng(67)
_M = _r.normal(size=(6, 6))
_U = np.linalg.qr(_M)[0]
_Q6 = _U @ np.diag([0.1, 0.3, 1.0, 3.0, 10.0, 30.0]) @ _U.T
_Q6 = (_Q6 + _Q6.T) / 2
_all = [dict(name='gd', kind='gd', lr=0.035), dict(name='heavy-ball', kind='momentum', lr=0.01, mu=0.8),
        dict(name='nag', kind='nesterov', lr=0.01, mu=0.8), dict(name='rms', kind='rmsprop', lr=0.01),
        dict(name='adam', kind='adam', lr=0.05)]
TESTS = [
    (np.array([[1.0]]), np.array([1.0]), np.array([0.0]), [dict(name='slow', kind='gd', lr=0.1), dict(name='fast', kind='gd', lr=0.9)], 1e-3, 20),
    (_Q, _b, np.array([3.0, 3.0]), [dict(name='gd', kind='gd', lr=0.02), dict(name='mom', kind='momentum', lr=0.02)], 1e-4, 300),
    (_Q, _b, np.array([3.0, 3.0]), _all, 1e-6, 400),
    (_Q, _b, np.array([1.05, 0.03]), [dict(name='a', kind='gd', lr=0.01), dict(name='b', kind='adam', lr=0.1)], 1e-2, 5),
    (_Q, _b, np.array([3.0, 3.0]), [dict(name='boom', kind='gd', lr=0.05), dict(name='ok', kind='gd', lr=0.01), dict(name='boom2', kind='momentum', lr=0.5)], 1e-8, 1000),
    (_Q6, _r.normal(size=6), _r.normal(size=6) * 3, [c for c in _all if c['kind'] != 'rmsprop'] + [dict(name='nag-0.95', kind='nesterov', lr=0.02, mu=0.95)], 1e-5, 600),
]
META['tol'] = [1e-6, 1e-7]

EXPLANATION = r"""
## What the race shows
On $Q = \mathrm{diag}(1, 50)$ ($\kappa = 50$), plain GD is limited to $\eta < 2/50$ and then converges along the flat direction at about $1 - \eta$ per step. Momentum and Nesterov reach the same gap in roughly $\sqrt\kappa$-proportional time. RMSProp and Adam rescale each coordinate, which is perfect for a **diagonal** $Q$ but only partial help for the rotated $Q_6$ in test 6, where the ill-conditioning isn't axis-aligned. With a constant learning rate they also stall at a noise floor set by $\eta$ (the normalised step doesn't shrink as $g \to 0$), so they can win early and lose the race to a tight target. That hovering is chaotic, tiny round-off differences change the exact final gap, which is why test 6 leaves RMSProp out.

## Benchmark hygiene
A fair comparison tunes each optimizer's learning rate separately (one shared $\eta$ favours whichever method it happens to suit), runs from identical starting points, and reports time-to-target rather than loss at an arbitrary step. Ranking by "reached first, then by final gap" is a simple way to fold failures into one table.

## Complexity
$O(K \cdot T \cdot d^2)$ for $K$ configs, $T$ steps, and dimension $d$ (Nesterov evaluates one gradient per step, at the look-ahead point; the gap check costs one more matrix-vector product).

## Common bugs
- Measuring $f(x_t)$ instead of the gap, so the target depends on $b$.
- Sharing state between runs (e.g. reusing the velocity array).
- Starting Adam's step counter at 0 (division by zero in the bias correction).
- Letting a `nan` gap into the sort: comparisons with `nan` are all `False`, so the order becomes arbitrary.

## Interview follow-ups
- Why can Adam beat tuned SGD early on a transformer but lose on a convex quadratic like this one? (Its advantage comes from per-parameter scale differences and noisy, heavy-tailed gradients, not from conditioning that's spread across rotated directions.)
- How would you make the comparison robust to the random seed in a stochastic setting? (Several seeds per config; report the median and spread of time-to-target.)
"""
