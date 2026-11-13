import numpy as np

META = dict(title='Fit a power law in log-log space', tags=['Scaling Laws', 'Regression'], diff='Medium', runtime='numpy',
            lessons=['mlmaths:08.02'], related=['linear-regression', 'saturating-power-law-fit', 'isoflop-sweep-exponent'])

STATEMENT = r"""
Scaling curves are usually straight lines on a log-log plot. A model sweep gives pairs $(x_i, y_i)$ (say, parameter count and validation loss) and you want the power law

$$y = a\, x^{-b}$$

Taking logs turns it into a line, $\ln y = \ln a - b \ln x$, which ordinary least squares can fit.

Write `fit_power_law(x, y, target)` that fits $a$ and $b$ and answers "how large must $x$ be to reach `target`?".

## Inputs
- `x`, `y`: 1-D float arrays of the same length $n \ge 2$, all entries strictly positive. `x` has at least two distinct values and `y` is not constant.
- `target`: a positive float.

## Fit
Ordinary least squares of $u_i = \ln y_i$ on $v_i = \ln x_i$ with an intercept, every point weighted equally. The slope is $-b$ and the intercept is $\ln a$.

## Output
A dict with these float values:

| key | value |
|---|---|
| `'a'` | $a$ |
| `'b'` | $b$ (positive when $y$ falls with $x$) |
| `'r2'` | coefficient of determination **in log space**: $1 - \sum (u_i - \hat u_i)^2 / \sum (u_i - \bar u)^2$ |
| `'x_target'` | the $x$ where the fitted curve equals `target`: $(a / \text{target})^{1/b}$ if $b > 0$, else `float('inf')` |
"""

STARTER = '''import numpy as np

def fit_power_law(x, y, target):
    """Return {'a', 'b', 'r2', 'x_target'} as floats."""
    slope, intercept = np.polyfit(x, y, 1)      # fits the wrong space
    a, b = intercept, -slope
    return {'a': float(a), 'b': float(b), 'r2': 1.0, 'x_target': float((a / target) ** (1 / b))}
'''

SOLUTION = '''import numpy as np

def fit_power_law(x, y, target):
    v = np.log(np.asarray(x, dtype=float))
    u = np.log(np.asarray(y, dtype=float))
    vc, uc = v - v.mean(), u - u.mean()          # centring gives the closed-form OLS slope
    slope = (vc @ uc) / (vc @ vc)
    intercept = u.mean() - slope * v.mean()
    a, b = float(np.exp(intercept)), float(-slope)

    resid = u - (intercept + slope * v)
    r2 = 1.0 - (resid @ resid) / (uc @ uc)

    # y = a x^-b = target  =>  x = (a / target)^(1/b); unreachable if the curve does not fall
    x_target = (a / target) ** (1.0 / b) if b > 0 else float('inf')
    return {'a': a, 'b': b, 'r2': float(r2), 'x_target': float(x_target)}
'''

HINTS = [
    'Fit a straight line to `(log x, log y)`, not to `(x, y)`. Any log base works for $b$ as long as you undo it consistently for $a$.',
    'With centred values $v_c = v - \\bar v$, $u_c = u - \\bar u$: slope $= (v_c \\cdot u_c)/(v_c \\cdot v_c)$, intercept $= \\bar u - \\text{slope}\\,\\bar v$. Then $b = -\\text{slope}$, $a = e^{\\text{intercept}}$.',
    'Solve $a x^{-b} = \\text{target}$ for $x$: $x = (a/\\text{target})^{1/b}$. Guard $b \\le 0$, where no $x$ makes the curve fall to the target.',
]

_r = np.random.default_rng(11)
_xs = np.logspace(7, 10, 9)
TESTS = [
    (np.array([1.0, 4.0, 16.0]), np.array([6.0, 3.0, 1.5]), 1.0),
    (np.array([1e6, 1e7, 1e8, 1e9]), np.array([5.0, 4.1, 3.5, 2.9]), 2.5),
    (np.array([2.0, 3.0]), np.array([2.0, 3.0]), 1.0),
    (np.array([10.0, 10.0, 100.0, 100.0]), np.array([2.0, 2.2, 1.0, 1.1]), 0.5),
    (np.array([1e3, 1e4, 1e5]), np.array([0.01, 0.001, 0.0001]), 0.5),
    (_xs, 12.0 * _xs ** -0.076 * np.exp(_r.normal(0, 0.01, size=9)), 2.0),
    (np.exp(_r.uniform(0, 8, size=12)), np.exp(_r.uniform(-1, 1, size=12)), 1.0),
]

EXPLANATION = r"""
## Why it works
$\ln y = \ln a - b \ln x$ is linear in the unknowns $(\ln a, b)$, so least squares has the usual closed form. The fit minimises **relative** (log) error, which is what you want when $y$ spans orders of magnitude: a 1% miss at the largest model counts as much as a 1% miss at the smallest.

## Complexity
$O(n)$ time, $O(n)$ memory.

## Common bugs
- Fitting in linear space (or fitting $\ln y$ against $x$), which gives an exponential, not a power law.
- Sign of $b$: the slope of the log-log line is $-b$.
- Back-transforming $a$ with the wrong base, e.g. `10 ** intercept` after `np.log`.
- Dividing by $b$ when $b \le 0$ (test 3 rises with $x$, so the target is never reached).
- Pure power laws keep improving forever. Real losses flatten toward an irreducible floor, so log-log fits bend at large scale; see the saturating fit.

## Interview follow-ups
- Your fitted $b = 0.076$ for loss vs parameters. How many times larger must the model be to cut the loss by 10%? ($0.9^{-1/0.076} \approx 4.0$, so about four times larger.)
- How would you get a confidence interval for $b$, and why do extrapolations two orders of magnitude past the data need one?
"""
