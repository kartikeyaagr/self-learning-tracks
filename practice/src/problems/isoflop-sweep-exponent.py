import numpy as np
import pandas as pd

META = dict(title='Scaling exponent from an IsoFLOP sweep', tags=['Scaling Laws', 'Data Processing'], diff='Hard', runtime='pandas',
            lessons=[], related=['chinchilla-compute-optimal', 'power-law-log-fit', 'top-n-per-group'])

STATEMENT = r"""
An **IsoFLOP sweep** trains several model sizes at each of a few fixed compute budgets. Within one budget, loss against $\log N$ is roughly a parabola (too small a model underfits, too large a model sees too few tokens), and its bottom is that budget's compute-optimal size. Fitting $N_\text{opt} \propto C^{a}$ across budgets gives the scaling exponent.

Write `isoflop_fit(df)` that returns the tuple `(optima, a)`.

## Input
`df` has one row per training run, with float columns `budget` (FLOPs $C$), `params` ($N$) and `loss`. Rows are in any order; runs at the same budget have exactly equal `budget` values. Don't modify `df`.

## Per budget
1. With $u = \log_{10} N$, fit $\text{loss} \approx c_2 u^2 + c_1 u + c_0$ by least squares over that budget's runs.
2. **Drop** the budget if it has fewer than 3 distinct `params` values, or if $c_2 \le 0$ (no interior minimum).
3. Otherwise $u^\star = -c_1 / (2 c_2)$, $N_\text{opt} = 10^{u^\star}$, $D_\text{opt} = C / (6 N_\text{opt})$, and the predicted loss there is $c_2 u^{\star 2} + c_1 u^\star + c_0$.

## Output
- `optima`: a DataFrame with float columns `budget, n_opt, d_opt, loss_opt` in that order, one row per kept budget, sorted by `budget` ascending. The index is ignored.
- `a`: the OLS slope of $\log_{10} N_\text{opt}$ on $\log_{10} C$ over the kept budgets, as a float. If fewer than 2 budgets are kept, `a` is `float('nan')`.
"""

STARTER = '''import numpy as np
import pandas as pd

def isoflop_fit(df):
    """Return (optima DataFrame [budget, n_opt, d_opt, loss_opt], a)."""
    best = df.loc[df.groupby('budget')['loss'].idxmin()]      # the best run is not the optimum
    out = pd.DataFrame({'budget': best['budget'], 'n_opt': best['params'],
                        'd_opt': best['budget'] / (6 * best['params']), 'loss_opt': best['loss']})
    a = np.polyfit(np.log10(out['budget']), np.log10(out['n_opt']), 1)[0]
    return out.reset_index(drop=True), float(a)
'''

SOLUTION = '''import numpy as np
import pandas as pd

def isoflop_fit(df):
    rows = []
    for C, g in df.groupby('budget', sort=True):              # ascending budgets
        if g['params'].nunique() < 3:
            continue
        u = np.log10(g['params'].to_numpy(dtype=float))
        c2, c1, c0 = np.polyfit(u, g['loss'].to_numpy(dtype=float), 2)
        if not c2 > 0:                                         # concave or flat: no interior minimum
            continue
        us = -c1 / (2 * c2)                                    # vertex of the parabola
        n = 10.0 ** us
        rows.append((float(C), n, C / (6 * n), c2 * us * us + c1 * us + c0))

    cols = ['budget', 'n_opt', 'd_opt', 'loss_opt']
    optima = pd.DataFrame(rows, columns=cols).astype(float)
    if len(optima) < 2:
        return optima, float('nan')
    a = np.polyfit(np.log10(optima['budget']), np.log10(optima['n_opt']), 1)[0]
    return optima, float(a)
'''

HINTS = [
    'Loop over `df.groupby("budget")` (it sorts the keys for you). Inside, `np.polyfit(u, loss, 2)` returns `(c2, c1, c0)`, highest power first.',
    'The vertex of $c_2 u^2 + c_1 u + c_0$ is at $u^\\star = -c_1/(2c_2)$. It is a minimum only if $c_2 > 0$. Check `nunique() >= 3` before fitting: a parabola through two sizes is not determined.',
    'Collect kept budgets into a list of tuples, build the DataFrame once, then fit a line to `(log10 budget, log10 n_opt)`. Guard the fewer-than-2 case with `nan`.',
]

_sig = lambda x: float(f'{x:.10g}')        # exact through the 15-digit JSON round trip

def _sweep(budgets, sizes_per_budget, E=1.8, A=400.0, B=420.0, alpha=0.33, beta=0.29, noise=0.0, seed=0):
    """Synthetic sweep from L = E + A/N^alpha + B/D^beta along D = C/(6N)."""
    r = np.random.default_rng(seed)
    out = []
    for C, sizes in zip(budgets, sizes_per_budget):
        for N in sizes:
            D = C / (6 * N)
            L = E + A / N ** alpha + B / D ** beta
            out.append((float(C), _sig(N), _sig(L * (1 + noise * r.normal()))))
    return pd.DataFrame(out, columns=['budget', 'params', 'loss'])

_g = lambda lo, hi, k: list(np.logspace(lo, hi, k))
TESTS = [
    (pd.DataFrame({'budget': [1e20] * 3 + [1e22] * 3, 'params': [1e8, 1e9, 1e10] * 2,
                   'loss': [3.0, 2.0, 3.0, 2.8, 1.8, 2.2]}),),
    (pd.DataFrame({'budget': [1e21, 1e21, 1e21, 1e21], 'params': [1e7, 1e8, 1e9, 1e10],
                   'loss': [4.0, 3.0, 2.8, 3.5]}),),
    (_sweep([1e18, 1e19, 1e20, 1e21], [_g(7, 9, 7), _g(7.3, 9.3, 7), _g(7.7, 9.7, 7), _g(8, 10, 7)]),),
    (pd.DataFrame({'budget': [1e20, 1e20, 1e20, 1e21, 1e21, 1e19, 1e19, 1e19, 1e22, 1e22, 1e22],
                   'params': [1e8, 1e9, 1e10, 1e8, 1e9, 1e7, 1e8, 1e9, 1e9, 1e10, 1e11],
                   'loss': [2.0, 3.0, 2.0, 2.5, 2.4, 3.6, 3.0, 3.1, 2.4, 2.0, 2.3]}),),
    (pd.DataFrame({'budget': [5e19] * 4, 'params': [1e8, 1e8, 1e9, 1e9], 'loss': [3.1, 3.0, 2.9, 2.95]}),),
    (_sweep([1e19, 1e21, 1e23], [_g(7, 9.5, 9), _g(8, 10.5, 9), _g(9, 11.5, 9)], noise=0.003, seed=4).iloc[np.random.default_rng(9).permutation(27)],),
    (_sweep([3e18, 3e20, 3e22], [_g(7, 9, 6), _g(8, 10, 6), _g(9, 11, 6)], E=1.6, A=300.0, B=300.0, alpha=0.3, beta=0.3),),
]

EXPLANATION = r"""
## Why it works
At fixed $C$, a loss of the form $E + AN^{-\alpha} + BD^{-\beta}$ with $D = C/(6N)$ is a smooth convex bowl in $\log N$, and a parabola is its second-order approximation near the bottom. The vertex interpolates between sizes you actually trained, which is why it beats taking the best single run (the starter's approach): with a grid of half-decade sizes, the best run can be off by $\sqrt{10}\times$ from the true optimum. Fitting a line to $(\log C, \log N_\text{opt})$ then reads off the exponent directly. Because $N_\text{opt} D_\text{opt} = C/6$ for every row, the data exponent is $1 - a$ automatically; you never fit it separately.

Test 7 uses $\alpha = \beta$, where the true exponent is exactly 0.5. There the bowl is a $\cosh$ in $\log N$, symmetric about its minimum, and the size grids are centred on it, so the parabola recovers 0.5 almost exactly. With $\alpha \ne \beta$ the bowl is lopsided and a grid that is off-centre biases the vertex toward its denser side; that bias is part of why IsoFLOP exponents from different sweeps disagree in the second decimal.

## Complexity
$O(n)$ to group plus $O(k)$ per budget for $k$ runs; the fits are tiny.

## Common bugs
- `np.polyfit` returns coefficients highest power first. Reading them as `(c0, c1, c2)` puts the vertex somewhere absurd.
- Fitting against `params` instead of `log10(params)`. The bowl is symmetric-ish in log space, not linear space.
- Counting rows instead of **distinct** sizes (test 5 has four rows but two sizes).
- Keeping a budget whose parabola opens downward (test 4's $10^{20}$ budget): its "vertex" is a maximum.
- Taking the natural log for $u$ and then using `10 ** u`.

## Interview follow-ups
- The fitted optimum for your largest budget sits at the edge of the sizes you trained. What does that tell you, and what do you run next?
- Compare this approach with fitting the full $L(N, D)$ to every run at once. Which is more robust to a learning-rate schedule that was tuned only for the mid-sized models?
"""
