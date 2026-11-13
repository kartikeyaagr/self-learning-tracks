import numpy as np

META = dict(title='Fit learning-rate and batch-size scaling laws from sweeps', tags=['Scaling Laws', 'Training'],
            diff='Hard', runtime='numpy', lessons=['air-fanout-v1:04.23', 'air-fanout-v1:12.01'],
            related=['isoflop-sweep-exponent', 'power-law-log-fit', 'estimate-critical-batch-size'])

STATEMENT = r"""
The best learning rate and batch size drift as compute grows, and at the final scale you get one shot. The usual remedy: sweep both hyperparameters at several small budgets, find each budget's optimum, and fit power laws $\eta^\star \propto C^{a}$ and $B^\star \propto C^{b}$ to extrapolate.

Write `fit_hparam_scaling(compute, lr, bs, loss, c_target)` and return a dict.

## Inputs
Float64 arrays of equal length, one entry per training run: `compute` (FLOPs $C$), `lr`, `bs` (batch size) and final `loss`, in any order. Runs of the same budget have exactly equal `compute`. `c_target` is a float.

## Per budget
With $u = \log_{10}\eta$ and $v = \log_{10} B$, fit the quadratic surface

$$\text{loss} \approx c_0 + c_1 u + c_2 v + c_3 u^2 + c_4 uv + c_5 v^2$$

by least squares over that budget's runs (use `np.linalg.lstsq` on the design matrix with columns in this order).

- **Drop** the budget if it has fewer than 6 runs, or if the fitted surface has no minimum: keep it only if $c_3 > 0$ and $4 c_3 c_5 - c_4^2 > 0$.
- Otherwise its optimum $(u^\star, v^\star)$ solves the stationarity condition $\begin{pmatrix} 2c_3 & c_4 \\ c_4 & 2c_5 \end{pmatrix}\begin{pmatrix} u \\ v\end{pmatrix} = -\begin{pmatrix} c_1 \\ c_2 \end{pmatrix}$, and `lr_opt` $= 10^{u^\star}$, `bs_opt` $= 10^{v^\star}$, `loss_opt` is the surface's value there.

## Across budgets
Fit $\log_{10}\eta^\star$ on $\log_{10} C$ by OLS with an intercept over the kept budgets, and the same for $\log_{10} B^\star$. The slopes are the exponents; evaluate both lines at `c_target`.

## Output
| key | value |
|---|---|
| `"budgets"`, `"lr_opt"`, `"bs_opt"`, `"loss_opt"` | float64 arrays over kept budgets, sorted by budget ascending |
| `"lr_exponent"`, `"bs_exponent"` | floats |
| `"lr_pred"`, `"bs_pred"` | floats, predicted optimum at `c_target` |

If fewer than 2 budgets are kept, the last four values are `float('nan')`. The tests always keep at least one budget.
"""

STARTER = '''import numpy as np

def fit_hparam_scaling(compute, lr, bs, loss, c_target):
    """Return a dict with budgets, lr_opt, bs_opt, loss_opt, lr_exponent, bs_exponent, lr_pred, bs_pred."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _line(x, y):
    """OLS slope and intercept of y on x."""
    xc = x - x.mean()
    slope = (xc @ (y - y.mean())) / (xc @ xc)
    return slope, y.mean() - slope * x.mean()

def fit_hparam_scaling(compute, lr, bs, loss, c_target):
    compute, loss = np.asarray(compute, dtype=float), np.asarray(loss, dtype=float)
    u_all, v_all = np.log10(lr), np.log10(bs)
    kept = []                                              # (budget, u*, v*, loss*)
    for c in np.unique(compute):                           # np.unique is sorted ascending
        sel = compute == c
        if sel.sum() < 6:
            continue
        u, v = u_all[sel], v_all[sel]
        X = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
        c0, c1, c2, c3, c4, c5 = np.linalg.lstsq(X, loss[sel], rcond=None)[0]
        if not (c3 > 0 and 4 * c3 * c5 - c4 * c4 > 0):    # Hessian must be positive definite
            continue
        us, vs = np.linalg.solve(np.array([[2 * c3, c4], [c4, 2 * c5]]), -np.array([c1, c2]))
        best = c0 + c1 * us + c2 * vs + c3 * us * us + c4 * us * vs + c5 * vs * vs
        kept.append((c, us, vs, best))

    budgets, us, vs, best = (np.array(col) for col in zip(*kept))
    out = {"budgets": budgets, "lr_opt": 10.0 ** us, "bs_opt": 10.0 ** vs, "loss_opt": best}
    nan = float("nan")
    if len(budgets) < 2:
        return {**out, "lr_exponent": nan, "bs_exponent": nan, "lr_pred": nan, "bs_pred": nan}
    x, xt = np.log10(budgets), np.log10(c_target)
    a_lr, b_lr = _line(x, us)
    a_bs, b_bs = _line(x, vs)
    return {**out, "lr_exponent": float(a_lr), "bs_exponent": float(a_bs),
            "lr_pred": float(10.0 ** (a_lr * xt + b_lr)), "bs_pred": float(10.0 ** (a_bs * xt + b_bs))}
'''

HINTS = [
    'Work in log space throughout. Group the runs with `np.unique(compute)` (already sorted) and a boolean mask per budget.',
    'Build the six-column design matrix `[1, u, v, u², uv, v²]`, solve with `lstsq`, check the positive-definiteness conditions, then solve the 2×2 linear system for the stationary point.',
    'The exponents are OLS slopes of $u^\\star$ and $v^\\star$ (already base-10 logs) against $\\log_{10} C$. Predict by evaluating each line at $\\log_{10}$ `c_target` and exponentiating with base 10.',
]


def _sweep(seed, budgets, noise, saddle_at=None, few_at=None):
    r = np.random.default_rng(seed)
    rows = []
    for c in budgets:
        x = np.log10(c) - 18.0
        u0, v0 = -2.5 - 0.12 * x, 2.4 + 0.35 * x               # true optimum drifts with compute
        base = 3.0 - 0.15 * x
        us = u0 + np.array([-0.6, -0.3, 0.0, 0.3, 0.6]) + r.uniform(-0.05, 0.05)
        vs = v0 + np.array([-0.6, -0.2, 0.2, 0.6]) + r.uniform(-0.05, 0.05)
        n = 3 if c == few_at else None
        for u in us:
            for v in vs:
                du, dv = u - u0, v - v0
                curv = -0.3 if c == saddle_at else 0.25
                rows.append((c, 10.0 ** u, 10.0 ** v,
                             base + 0.8 * du * du + 0.2 * du * dv + curv * dv * dv + r.normal(0.0, noise)))
        if n is not None:
            rows = rows[:len(rows) - 20 + n]
    a = np.array(rows)
    perm = r.permutation(len(a))
    return a[perm, 0], a[perm, 1], a[perm, 2], a[perm, 3]


_u = np.array([-3.0, -2.0, -1.0, -3.0, -2.0, -1.0, -2.0])
_v = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
TESTS = [
    (np.full(7, 1e18), 10.0 ** _u, 10.0 ** _v, 1.0 + (_u + 2.0) ** 2 + 0.5 * (_v - 2.0) ** 2, 1e20),
    (np.array([1e18] * 7 + [1e20] * 7), np.tile(10.0 ** _u, 2), np.tile(10.0 ** _v, 2),
     np.concatenate([1.0 + (_u + 2.0) ** 2 + (_v - 2.0) ** 2, 0.5 + (_u + 2.5) ** 2 + (_v - 3.0) ** 2]), 1e22),
    (*_sweep(0, [1e17, 1e18, 1e19, 1e20], 0.0), 1e23),
    (*_sweep(1, [1e17, 1e18, 1e19, 1e20], 0.003), 1e23),
    (*_sweep(2, [1e17, 1e18, 1e19, 1e20, 1e21], 0.003, saddle_at=1e19), 1e24),
    (*_sweep(3, [1e17, 1e18, 1e19], 0.002, few_at=1e17), 1e22),
    (*_sweep(4, [3e18, 1e20], 0.004, saddle_at=1e20), 1e22),
]

EXPLANATION = r"""
## Why a quadratic in log space
Near the optimum, loss as a function of $\log\eta$ and $\log B$ is a smooth bowl, and a second-order Taylor expansion is the simplest model that has a minimum. The cross term $c_4$ matters: learning rate and batch size interact (a larger batch tolerates a larger rate), so optimising each separately along grid lines gives the wrong point. The positive-definiteness check rejects sweeps that missed the bowl (test 5's saddle budget) instead of reporting a maximum or saddle point as the optimum.

## Why fit the optimum, not pick the best run
The best grid point snaps to the grid, so its drift with compute is a staircase. The fitted optimum interpolates between grid points and averages out noise across all runs of a budget, which gives far more stable exponents.

## Complexity
Per budget, one $n \times 6$ least-squares solve and one $2 \times 2$ solve. Everything else is linear in the number of runs.

## Common bugs
- Natural logs in one place and base-10 in another: the exponents are base-independent, but the predictions are not.
- Dropping the $uv$ term, or using the diagonal $2c_3, 2c_5$ alone to find the optimum.
- Keeping budgets whose surface is not convex.
- Sorting kept budgets by insertion order rather than by compute.

## Interview follow-ups
- Published sweeps often find the optimal learning rate nearly flat in compute but the optimal batch size growing. How does that relate to the critical batch size?
- How would you put error bars on `lr_pred`? (Bootstrap runs within each budget and refit.)
"""
