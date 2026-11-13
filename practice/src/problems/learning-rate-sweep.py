import numpy as np

META = dict(title='Sweep the learning rate', tags=['Training', 'Optimization'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:05.05', 'air-fanout-v1:04.23'],
            related=['vanilla-gradient-descent', 'lr-range-test-and-one-cycle-policy'])

STATEMENT = r"""
The learning rate is the first hyperparameter to tune. The simplest tuning method is a sweep: train once per candidate value and keep the best.

Run plain gradient descent $x_{t+1} = x_t - \eta(Qx_t - b)$ for `steps` steps from `x0` on

$$f(x) = \tfrac12 x^\top Q x - b^\top x$$

once for every learning rate in `lrs`, and score each run by $f$ at its **final** iterate.

Write `lr_sweep(Q, b, x0, lrs, steps)` that returns the tuple `(final_losses, best_lr)`:

- `final_losses`: float array `(len(lrs),)`, in the same order as `lrs`. A run that blows up gives a non-finite value; report it as `inf` (never `nan`).
- `best_lr`: the learning rate with the smallest final loss, as a Python `float` (ties go to the one listed first). If every run diverged, return `None`.

## Inputs
- `Q`: float `(d, d)` symmetric; `b`, `x0`: float `(d,)`.
- `lrs`: list of positive floats, at least one. `steps >= 1`.
- Overflow is expected for large rates; suppress the warnings with `np.errstate`.
"""

STARTER = '''import numpy as np

def lr_sweep(Q, b, x0, lrs, steps):
    """Return (final_losses (len(lrs),), best_lr float or None)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def lr_sweep(Q, b, x0, lrs, steps):
    Q, b, x0 = (np.asarray(v, dtype=float) for v in (Q, b, x0))
    finals = []
    with np.errstate(over='ignore', invalid='ignore'):   # divergence is a legitimate outcome here
        for lr in lrs:
            x = x0.copy()
            for _ in range(steps):
                x = x - lr * (Q @ x - b)
            f = 0.5 * x @ Q @ x - b @ x
            finals.append(f if np.isfinite(f) else np.inf)
    finals = np.array(finals, dtype=float)
    if not np.isfinite(finals).any():
        return finals, None
    return finals, float(lrs[int(np.argmin(finals))])     # argmin returns the first minimum
'''

HINTS = [
    'Wrap a plain gradient-descent loop in an outer loop over `lrs`, restarting from a fresh copy of `x0` each time.',
    'After the inner loop, evaluate $f$ once. Replace anything non-finite (`inf` or `nan`) with `np.inf` using `np.isfinite`.',
    '`np.argmin` already breaks ties by taking the first index. Check for the all-diverged case before calling it.',
]

_r = np.random.default_rng(37)
_M = _r.normal(size=(3, 3))
_spd = _M @ _M.T + np.eye(3)
TESTS = [
    (np.array([[2.0]]), np.array([2.0]), np.array([0.0]), [0.1, 0.5, 1.2], 3),
    (np.array([[1.0, 0.0], [0.0, 10.0]]), np.array([1.0, 1.0]), np.array([0.0, 0.0]), [0.01, 0.1, 0.19, 0.25], 10),
    (np.array([[1.0]]), np.array([1.0]), np.array([5.0]), [3.0, 4.0], 2000),
    (np.array([[4.0, 1.0], [1.0, 3.0]]), np.array([1.0, -2.0]), np.array([1.0, 1.0]), [0.3], 4),
    (np.array([[1.0]]), np.array([0.0]), np.array([1.0]), [0.5, 1.5], 3),
    (_spd, _r.normal(size=3), _r.normal(size=3), [0.001, 0.01, 0.05, 0.1, 0.3, 1.0], 15),
]

EXPLANATION = r"""
## What the sweep shows
For a quadratic, GD multiplies the error along eigen-direction $k$ by $1 - \eta\lambda_k$ per step. The sweep traces out the classic U-shape: too small and progress is slow, too large ($\eta > 2/\lambda_{\max}$) and the steepest direction explodes. In test 2 ($\lambda_{\max} = 10$), $\eta = 0.19$ is just stable while $0.25$ diverges. The best rate in a fixed budget is typically just below the edge of stability.

Test 5 shows a tie: $\eta = 0.5$ and $\eta = 1.5$ both scale the error by $|1 - \eta| = 0.5$ per step (one from the same side, one alternating), so they reach the same loss and the first wins.

## Practical sweeps
Real sweeps use a **log-spaced** grid (e.g. $10^{-5}, 3\cdot10^{-5}, 10^{-4}, \dots$) because the useful range spans orders of magnitude, and they score on validation loss after a short budget. Coarse grid first, then refine around the winner.

## Complexity
$O(K \cdot T \cdot d^2)$ for $K$ rates and $T$ steps. The runs are independent, so they parallelise perfectly.

## Common bugs
- Reusing `x` from the previous learning rate instead of restarting from `x0`.
- Letting `nan` through: `np.argmin` on an array containing `nan` returns the `nan`'s index.
- Picking the rate with the lowest loss at **any** step instead of at the end of the budget.

## Interview follow-ups
- Why is random search over log-learning-rate often better than a grid when tuning several hyperparameters at once? (Each trial tests a new value of every important parameter.)
- How would you pick the rate without a full sweep? (A learning-rate range test: one run with an exponentially increasing rate.)
"""
