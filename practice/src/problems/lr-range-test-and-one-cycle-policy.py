import numpy as np

META = dict(title='LR range test to a one-cycle schedule', tags=['Training', 'Optimization'], diff='Hard', runtime='numpy',
            lessons=['air-fanout-v1:05.05', 'air-fanout-v1:01.14'],
            related=['learning-rate-sweep', 'cosine-annealing-with-warm-restarts', 'linear-warmup'])

STATEMENT = r"""
A learning-rate range test trains for a few hundred steps while raising the learning rate exponentially, and logs the loss at each step. You then read a good peak rate off that curve and plug it into a one-cycle schedule. Implement both halves.

## Part 1: pick the peak rate
You get the logged rates `lrs` (strictly increasing, positive) and `losses` (positive), both length $K$.

1. **Smooth** with a bias-corrected EMA: $a_{-1} = 0$, $a_k = \beta a_{k-1} + (1-\beta)\,\ell_k$, $s_k = a_k / (1 - \beta^{k+1})$.
2. **Cut at divergence**: find the first $k$ with $s_k > 4 \cdot \min(s_0, \dots, s_k)$. Keep only indices before it. If there is none, keep everything. Tests guarantee at least 2 points survive.
3. **Steepest descent**: over the kept points, for $j = 1, \dots, K'-1$ compute the slope against log-rate,
$$d_j = \frac{s_j - s_{j-1}}{\ln \text{lr}_j - \ln \text{lr}_{j-1}}.$$
The peak rate is `max_lr = lrs[j*]` with $j^\star = \arg\min_j d_j$ (the most negative slope; ties go to the smallest $j$).

## Part 2: one-cycle schedule
With $N$ = `total_steps` and $U$ = `up_steps` ($1 \le U \le N - 2$), let $\eta_0 = \text{max\_lr}/\text{div}$ and $\eta_{\text{end}} = \eta_0/\text{final\_div}$. Using cosine interpolation $\mathrm{cos\_interp}(p, q, f) = q + (p - q)\,\tfrac12(1 + \cos \pi f)$, which runs from $p$ at $f = 0$ to $q$ at $f = 1$:

- for $0 \le t < U$: $\eta_t = \mathrm{cos\_interp}(\eta_0, \text{max\_lr}, t/U)$
- for $U \le t < N$: $\eta_t = \mathrm{cos\_interp}(\text{max\_lr}, \eta_{\text{end}}, (t - U)/(N - 1 - U))$

Write `range_test_one_cycle(lrs, losses, total_steps, up_steps, beta=0.98, div=25.0, final_div=1e4)` returning the tuple `(max_lr, schedule)`: a Python `float` and a float array `(total_steps,)`.
"""

STARTER = '''import numpy as np

def range_test_one_cycle(lrs, losses, total_steps, up_steps, beta=0.98, div=25.0, final_div=1e4):
    """Return (max_lr float, schedule (total_steps,)) following the two-part recipe."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _cos_interp(p, q, f):
    return q + (p - q) * 0.5 * (1 + np.cos(np.pi * f))

def range_test_one_cycle(lrs, losses, total_steps, up_steps, beta=0.98, div=25.0, final_div=1e4):
    lrs, losses = np.asarray(lrs, dtype=float), np.asarray(losses, dtype=float)
    # Part 1a: bias-corrected EMA (the correction matters most for the first few points).
    s, a = np.empty_like(losses), 0.0
    for k, loss in enumerate(losses):
        a = beta * a + (1 - beta) * loss
        s[k] = a / (1 - beta ** (k + 1))
    # Part 1b: cut at the first point that blows past 4x the best smoothed loss so far.
    blown = np.flatnonzero(s > 4 * np.minimum.accumulate(s))
    keep = blown[0] if blown.size else len(s)
    s, x = s[:keep], np.log(lrs[:keep])
    # Part 1c: most negative slope of loss vs log-lr; argmin picks the first on ties.
    j = 1 + int(np.argmin(np.diff(s) / np.diff(x)))
    max_lr = float(lrs[j])

    # Part 2: cosine up from max/div, cosine down to max/(div*final_div).
    lo, end = max_lr / div, max_lr / div / final_div
    t = np.arange(total_steps, dtype=float)
    up = _cos_interp(lo, max_lr, t / up_steps)
    down = _cos_interp(max_lr, end, (t - up_steps) / (total_steps - 1 - up_steps))
    return max_lr, np.where(t < up_steps, up, down)
'''

HINTS = [
    'Do it in stages and print each: the smoothed curve, the cut index, the slopes. `np.minimum.accumulate` gives the running minimum in one call.',
    'The slopes are `np.diff(s) / np.diff(np.log(lrs))` over the kept points. `np.diff` index `i` corresponds to $j = i + 1$, so add 1 to the argmin.',
    'For the schedule, compute the warm-up and anneal formulas over all of `np.arange(total_steps)` and select with `np.where(t < up_steps, ...)`. At $t = U$ the anneal formula gives exactly `max_lr`.',
]

_r = np.random.default_rng(41)
_lrs = np.geomspace(1e-6, 1.0, 120)
_x = np.log10(_lrs)
_curve = 2.3 - 1.8 / (1 + np.exp(-(_x + 3.0) * 3)) + np.exp(np.clip((_x + 1.2) * 6, None, 30))
_noisy = _curve * np.exp(0.03 * _r.normal(size=120))
TESTS = [
    (np.array([0.001, 0.01, 0.1, 1.0, 10.0]), np.array([2.0, 1.9, 1.0, 0.9, 50.0]), 10, 3, 0.0),
    (np.array([0.01, 0.02, 0.05, 0.1]), np.array([1.0, 0.8, 0.7, 0.69]), 6, 2, 0.5, 10.0, 100.0),
    (_lrs, _curve, 100, 30),
    (_lrs, _noisy, 100, 30),
    (_lrs, _noisy, 50, 10, 0.9),
    (np.array([1e-4, 1e-3]), np.array([3.0, 2.0]), 4, 1),
    (np.array([1.0, 2.0, 4.0, 8.0]), np.array([1.0, 1.0, 1.0, 1.0]), 5, 2, 0.0),
]

EXPLANATION = r"""
## Why the steepest slope
In a range test, small rates barely move the loss, large rates blow it up, and in between the loss falls fastest. The point of steepest descent per unit of $\log \eta$ is where the rate is "most useful": large enough to make progress, not yet unstable. Taking the slope against $\log\eta$ (not $\eta$) matches the exponential sweep, so every logged step counts equally.

The EMA smooths batch noise. Its bias correction matters: with $\beta = 0.98$ and no correction, the first smoothed values are about 2% of the real loss, which would fake a huge early drop.

## Why one-cycle
The schedule spends its first part warming up from a small rate (stability, like warmup), peaks at the largest stable rate (fast progress, and the noise acts as a regulariser), then anneals far below the start (fine convergence). Smith reported "super-convergence": comparable accuracy in a fraction of the epochs. `div` and `final_div` mirror the defaults in common implementations.

## Complexity
$O(K)$ for the range test analysis and $O(N)$ for the schedule.

## Common bugs
- Measuring slope against raw $\eta$: the steep region gets crushed into the right end of the curve.
- Forgetting the divergence cut, so the steepest "descent" search runs into the exploding tail and the argmin is meaningless.
- Off-by-one when mapping `np.diff` indices back to `lrs` (pick `lrs[j]`, the right end of the steepest pair).
- An anneal phase that ends one step early or late: at $t = N - 1$ the rate must equal $\eta_{\text{end}}$ exactly.

## Interview follow-ups
- Many people pick a rate "an order of magnitude below the loss minimum" instead of the steepest point. Why might those disagree, and which is safer? (The minimum sits close to instability; dividing by 10 lands in the steep region with margin.)
- One-cycle also cycles momentum inversely (high momentum when the rate is low). Why would that help? (At peak rate, lower momentum keeps the effective step size $\eta/(1-\mu)$ from becoming unstable.)
"""
