import numpy as np

META = dict(title='Where an exponential LR schedule settles', tags=['Calculus', 'Training'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:05.05'], related=['learning-rate-stability-and-convergence-bounds', 'adam-step'])

STATEMENT = r"""
A common learning-rate schedule decays geometrically towards a floor:

$$\eta_t = \eta_{\min} + (\eta_0 - \eta_{\min})\,\gamma^t, \qquad t = 0, 1, 2, \dots, \qquad 0 \le \gamma \le 1$$

Write `lr_limit(lr0, lr_min, gamma, eps)` that returns a tuple `(limit, t_eps)`:

- `limit`: the value $L = \lim_{t\to\infty} \eta_t$, as a float.
- `t_eps`: the smallest integer $t \ge 0$ with $|\eta_t - L| < \varepsilon$ (strict). This is the $N$ in the $\varepsilon$-$N$ definition of a limit: the first step after which the schedule is (and stays) within $\varepsilon$ of where it is heading.

## Rules
- `lr0` and `lr_min` are floats, and either may be the larger one (a warm-up that rises towards a ceiling uses the same formula).
- Measure the gap at step $t$ as $|\eta_0 - \eta_{\min}|\,\gamma^t$ when $\gamma < 1$.
- `gamma == 1`: the schedule never moves, so the limit is `lr0` and `t_eps` is `0`.
- `gamma == 0`: use $0^0 = 1$, so $\eta_0 = $ `lr0` and every later step equals `lr_min`.
- `eps > 0`. Return `t_eps` as a Python `int`.
"""

STARTER = '''def lr_limit(lr0, lr_min, gamma, eps):
    """Return (limit of the schedule, first step t with |eta_t - limit| < eps)."""
    raise NotImplementedError
'''

SOLUTION = '''import math

def lr_limit(lr0, lr_min, gamma, eps):
    if gamma == 1:
        return float(lr0), 0
    gap = abs(lr0 - lr_min)                    # |eta_0 - L|; the gap at step t is gap * gamma**t
    if gap < eps:
        return float(lr_min), 0
    if gamma == 0:
        return float(lr_min), 1
    # Solve gap * gamma**t < eps for t. log(gamma) < 0 flips the inequality.
    t = max(0, math.ceil(math.log(eps / gap) / math.log(gamma)))
    while gap * gamma ** t >= eps:             # floating-point log can land one step short...
        t += 1
    while t > 0 and gap * gamma ** (t - 1) < eps:   # ...or one step long
        t -= 1
    return float(lr_min), t
'''

HINTS = [
    'For $0 \\le \\gamma < 1$, $\\gamma^t \\to 0$. So what does $\\eta_t$ approach? Handle $\\gamma = 1$ separately.',
    'The gap $|\\eta_0 - \\eta_{\\min}|\\,\\gamma^t$ shrinks monotonically. Take logs of $\\text{gap}\\cdot\\gamma^t < \\varepsilon$ and solve for $t$; remember $\\log\\gamma < 0$ flips the inequality.',
    '`t = ceil(log(eps / gap) / log(gamma))`, clipped at 0, then nudge `t` up or down by one while the strict inequality fails or holds at `t - 1`. A plain loop over `t` also works but can take hundreds of thousands of steps for $\\gamma$ near 1.',
]

_r = np.random.default_rng(3)
TESTS = [
    (0.1, 0.001, 0.5, 1e-3),
    (0.01, 0.0, 0.9, 1e-4),
    (0.001, 0.1, 0.8, 1e-6),
    (0.05, 0.05, 0.7, 1e-9),
    (0.3, 0.01, 1.0, 1e-3),
    (0.3, 0.01, 0.0, 1e-3),
    (0.1, 0.02, 0.95, 0.5),
    (1.0, 0.0, 0.9999, 1e-12),
    (float(_r.uniform(0.01, 1)), float(_r.uniform(0, 0.01)), float(_r.uniform(0.9, 0.99)), 1e-7),
]

EXPLANATION = r"""
## Why the limit is the floor
For $0 \le \gamma < 1$, $\gamma^t \to 0$, so $\eta_t \to \eta_{\min}$ whatever $\eta_0$ is. The gap to the limit is exactly $|\eta_0 - \eta_{\min}|\gamma^t$, which shrinks monotonically, so once it drops below $\varepsilon$ it stays there. That's what makes "first step inside the band" well defined.

## Closed form
$\text{gap}\cdot\gamma^t < \varepsilon \iff t > \log(\varepsilon/\text{gap}) / \log\gamma$, so $t_\varepsilon = \lceil \log(\varepsilon/\text{gap})/\log\gamma \rceil$ (bumped by one when the ratio is an exact integer). It's $O(1)$; a loop is $O(t_\varepsilon)$, which is about 276,000 steps for $\gamma = 0.9999$ and $\varepsilon = 10^{-12}$.

## Common bugs
- Returning `lr_min` as the limit when $\gamma = 1$. The formula only converges to the floor when $\gamma < 1$.
- Using `<=` instead of `<`, or `floor` instead of `ceil`.
- Trusting the log formula blindly. `log` is rounded, so a value that should be an exact integer can come out as `43.99999` or `44.00001`. Check the inequality at the answer and its neighbour.

## Interview follow-ups
- Cosine decay over $T$ steps has no limit as $t \to \infty$ (it's defined on $[0, T]$). How would you pick $T$ given a target final LR?
- The schedule $\eta_t = \eta_0/(1 + kt)$ also tends to 0, but $\sum_t \eta_t$ diverges while $\sum_t \eta_t^2$ converges. Why do the classic SGD convergence proofs ask for exactly that?
"""
