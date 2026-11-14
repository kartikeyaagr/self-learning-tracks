import numpy as np

META = dict(title='Fit a power law with an irreducible loss floor', tags=['Scaling Laws', 'Regression'], diff='Hard', runtime='numpy',
            lessons=['mlmaths:08.03'], related=['power-law-log-fit', 'chinchilla-compute-optimal'])

STATEMENT = r"""
Loss can't fall below the entropy of the data, so scaling curves flatten. The standard fix is a power law plus a floor:

$$L(C) = E + A\, C^{-\alpha}, \qquad A, \alpha > 0$$

Fitting all three parameters is non-linear, but for a **fixed** $E$ it isn't: $\ln(L - E) = \ln A - \alpha \ln C$ is a straight line. So scan candidate floors and keep the best one. (Statisticians call this profiling out $E$.)

Write `fit_saturating(c, loss, e_grid, c_new)` that returns the tuple `(E, A, alpha, pred)`.

## Inputs
- `c`, `loss`: 1-D float arrays of the same length $n \ge 3$, all strictly positive; `c` has at least two distinct values.
- `e_grid`: 1-D float array of candidate floors, in any order.
- `c_new`: 1-D float array of compute values to extrapolate to.

## Procedure
For each candidate $E$ in `e_grid`:

1. Skip it unless $E < \min_i L_i$ (strictly), so every $\ln(L_i - E)$ is defined.
2. Fit $\ln(L_i - E)$ on $\ln C_i$ by ordinary least squares with an intercept. Slope $= -\alpha$, intercept $= \ln A$.
3. Skip it unless $\alpha > 0$.
4. Score it by the squared error **in the original loss space**: $\mathrm{SSE}(E) = \sum_i \big(E + A C_i^{-\alpha} - L_i\big)^2$.

Pick the candidate with the smallest SSE; on an exact tie, the smallest $E$. The grid is guaranteed to contain at least one candidate that survives steps 1 and 3.

## Output
`E`, `A`, `alpha` as floats and `pred = E + A * c_new ** -alpha` as a float array shaped like `c_new`.
"""

STARTER = '''import numpy as np

def fit_saturating(c, loss, e_grid, c_new):
    """Return (E, A, alpha, pred)."""
    v, u = np.log(c), np.log(loss)              # pretends the floor is 0
    slope, intercept = np.polyfit(v, u, 1)
    A, alpha = np.exp(intercept), -slope
    return 0.0, float(A), float(alpha), A * np.asarray(c_new) ** -alpha
'''

SOLUTION = '''import numpy as np

def _loglog_ols(v, u):
    """OLS of u on v with an intercept; returns (slope, intercept)."""
    vc = v - v.mean()
    slope = (vc @ (u - u.mean())) / (vc @ vc)
    return slope, u.mean() - slope * v.mean()

def fit_saturating(c, loss, e_grid, c_new):
    c, loss = np.asarray(c, dtype=float), np.asarray(loss, dtype=float)
    v = np.log(c)
    best = None                                  # (sse, E, A, alpha)
    for E in np.sort(np.asarray(e_grid, dtype=float)):   # ascending: a strict < keeps the smallest E on ties
        if not E < loss.min():
            continue
        slope, intercept = _loglog_ols(v, np.log(loss - E))
        alpha = -slope
        if not alpha > 0:
            continue
        A = np.exp(intercept)
        r = E + A * c ** -alpha - loss            # judged in loss space, not log space
        sse = r @ r
        if best is None or sse < best[0]:
            best = (sse, E, A, alpha)
    _, E, A, alpha = best
    pred = E + A * np.asarray(c_new, dtype=float) ** -alpha
    return float(E), float(A), float(alpha), pred
'''

HINTS = [
    'For one fixed $E$ the problem is the plain log-log fit of $L - E$ against $C$. Write that as a helper, then loop over the grid.',
    'Score candidates by squared error of $E + A C^{-\\alpha}$ against $L$, not by the log-space residual: the log-space error of different $E$ values is measured on different curves, so it is not comparable.',
    'Iterate the grid in ascending order and replace the best only when the SSE is strictly smaller. That gives the smallest-$E$ tie rule for free. Skip $E \\ge \\min L$ and fits with $\\alpha \\le 0$.',
]

_r = np.random.default_rng(21)
_c = np.logspace(18, 22, 9)
_c2 = np.exp(_r.uniform(np.log(1e17), np.log(1e21), size=10))
TESTS = [
    (np.array([1.0, 10.0, 100.0, 1000.0]), np.array([3.0, 2.1, 2.01, 2.001]), np.array([0.0, 1.0, 2.0, 2.5]),
     np.array([1e4])),
    (np.array([1.0, 4.0, 16.0]), np.array([2.5, 2.25, 2.125]), np.array([1.0, 1.5, 2.0]), np.array([64.0, 256.0])),
    (_c, 1.7 + 40.0 * _c ** -0.08, np.linspace(0.0, 2.0, 41), np.array([1e24, 1e26])),
    (_c, (1.7 + 40.0 * _c ** -0.08) * np.exp(_r.normal(0, 0.002, size=9)), np.linspace(0.0, 2.0, 81),
     np.array([1e23, 1e25])),
    (np.array([1.0, 2.0, 4.0, 8.0]), np.array([3.0, 2.0, 1.6, 3.2]), np.array([-50.0, 1.5, 1.7]), np.array([16.0])),
    (np.array([1.0, 10.0, 100.0]), np.array([5.0, 3.0, 2.5]), np.array([2.0, 2.0, 1.0, 3.0, 2.5]), np.array([1e3])),
    (_c2, 2.1 + 9.0 * (_c2 / 1e17) ** -0.3 * np.exp(_r.normal(0, 0.01, size=10)), np.linspace(1.0, 3.0, 21),
     np.array([1e22])),
]

EXPLANATION = r"""
## Why it works
For a given $E$, $\ln(L - E) = \ln A - \alpha \ln C$ is linear, so the best $(A, \alpha)$ has a closed form. That reduces a 3-parameter non-linear fit to a 1-D search over $E$. A grid makes the answer deterministic; in practice you would follow it with a golden-section or Brent search between the best grid point's neighbours, or hand the result to a non-linear optimiser as its starting point.

The pure log-log fit is the special case $E = 0$. With a real floor present it is badly wrong: it reads the flattening as a smaller exponent and under-predicts future gains at small scale while over-predicting them at large scale.

## Why score in loss space
Each candidate $E$ transforms the data differently. Log-space residuals of $L - E$ blow up as $E \to \min L$, so comparing them across candidates rewards whichever $E$ makes the transform kindest, not the best fit. Comparing $\sum (E + A C^{-\alpha} - L)^2$ puts every candidate on the same scale. (The within-candidate fit is still done in log space; it is an approximation that is excellent when the noise is small and multiplicative.)

## Complexity
$O(G \cdot n)$ for $G$ grid points and $n$ data points.

## Common bugs
- Allowing $E = \min L$: $\ln 0 = -\infty$ poisons the fit. The condition is strict.
- Accepting fits with $\alpha \le 0$ (test 5: loss rising with compute), which are not scaling laws.
- Not sorting the grid, so the tie rule depends on input order (test 6 has duplicates and an unsorted grid).
- Trusting the extrapolation far past the data. $E$ is the least-determined parameter: many $(E, A, \alpha)$ fit a short range almost equally well and diverge at 100x the compute.

## Interview follow-ups
- Replace the grid with a 1-D bounded optimiser. What bounds do you give it and what can go wrong near $\min L$?
- Fit this curve to only the first five points of test 3 and extrapolate. How would you report uncertainty in the prediction at $10^{24}$ FLOPs? (Bootstrap over points, refit, report the spread.)
"""
