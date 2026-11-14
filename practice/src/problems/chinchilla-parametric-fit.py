import numpy as np

META = dict(title='Chinchilla approach 3: fit the parametric loss, then allocate compute', tags=['Scaling Laws', 'Optimization'],
            diff='Hard', runtime='numpy', lessons=['air-fanout-v1:13.05'],
            related=['chinchilla-compute-optimal', 'saturating-power-law-fit', 'isoflop-sweep-exponent', 'fixed-ratio-budget-split'])

STATEMENT = r"""
The Chinchilla paper's third approach models final loss as $L(N, D) = E + A/N^\alpha + B/D^\beta$ for $N$ parameters and $D$ training tokens, fits it to many training runs, and solves for the compute-optimal split. Do this with a grid search over the exponents.

Write `chinchilla_fit(N, D, L, alphas, betas, budgets)` returning `(params, N_opt, D_opt)`.

## Fit
For each $(\alpha, \beta)$ in `alphas` $\times$ `betas` (alpha in the outer loop), the model is linear in $(E, A, B)$: solve the ordinary least-squares problem with columns $[1,\ N^{-\alpha},\ D^{-\beta}]$ against `L` (use `np.linalg.lstsq`). Keep the pair with the smallest sum of squared residuals (the first one on ties). `params` is `(E, A, B, alpha, beta)` as floats.

## Allocate
With $C \approx 6ND$, minimising $L$ for a fixed $C$ gives
$$N_{opt}(C) = G\left(\frac C6\right)^{\beta/(\alpha+\beta)},\quad G = \left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)},\quad D_{opt} = \frac{C}{6N_{opt}}.$$
Return `N_opt` and `D_opt` for every budget in `budgets`.
"""

STARTER = '''import numpy as np

def chinchilla_fit(N, D, L, alphas, betas, budgets):
    """Return ((E, A, B, alpha, beta), N_opt, D_opt)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def chinchilla_fit(N, D, L, alphas, betas, budgets):
    N, D, L, C = (np.asarray(a, dtype=float) for a in (N, D, L, budgets))
    best = None
    for a in alphas:
        for b in betas:
            X = np.column_stack([np.ones_like(N), N ** -a, D ** -b])
            coef = np.linalg.lstsq(X, L, rcond=None)[0]
            sse = float(((X @ coef - L) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, coef, float(a), float(b))
    _, (E, A, B), a, b = best
    G = (a * A / (b * B)) ** (1 / (a + b))
    Nopt = G * (C / 6) ** (b / (a + b))
    return (float(E), float(A), float(B), a, b), Nopt, C / (6 * Nopt)
'''

HINTS = [
    'Fixing the exponents turns a nonlinear fit into linear regression on three features. Loop over the grid and keep the best residual.',
    'Substitute $D = C/(6N)$ and set $dL/dN = 0$: $\\alpha A N^{-\\alpha - 1} = \\beta B\\,(C/6)^{-\\beta}N^{\\beta - 1}$, which solves to the formula for $N_{opt}$.',
    'The exponents of $N_{opt}$ and $D_{opt}$ in $C$ are $\\beta/(\\alpha+\\beta)$ and $\\alpha/(\\alpha+\\beta)$; they add up to 1.',
]

_r = np.random.default_rng(2303)
def _runs(n, E, A, B, a, b, noise):
    N = np.exp(_r.uniform(np.log(1e7), np.log(1e10), n))
    D = np.exp(_r.uniform(np.log(1e9), np.log(1e12), n))
    return N, D, E + A / N ** a + B / D ** b + noise * _r.normal(size=n)
_g1 = np.round(np.arange(0.25, 0.46, 0.01), 2)
TESTS = [
    (*_runs(12, 1.7, 400.0, 400.0, 0.34, 0.28, 0.0), np.array([0.3, 0.34, 0.4]), np.array([0.24, 0.28, 0.32]), np.array([1e21])),
    (*_runs(40, 1.69, 406.4, 410.7, 0.34, 0.28, 0.005), _g1, _g1, np.array([1e20, 1e22, 1e24])),
    (*_runs(60, 2.0, 300.0, 800.0, 0.3, 0.35, 0.01), _g1, _g1, np.array([6e18, 6e21, 6e23])),
    (*_runs(30, 1.5, 1000.0, 200.0, 0.4, 0.25, 0.002), _g1, _g1, np.array([1e23])),
    (*_runs(25, 1.8, 500.0, 500.0, 0.3, 0.3, 0.0), np.array([0.3]), np.array([0.3]), np.array([1e19, 1e25])),
]

EXPLANATION = r"""
## The result
Hoffmann et al. trained over 400 models and found, with all three of their approaches, that parameters and tokens should grow roughly equally with compute (exponents near 0.5 each), about 20 tokens per parameter. Earlier scaling laws (Kaplan et al.) had recommended growing the model much faster than the data, so models like Gopher were undertrained; the 70B Chinchilla, trained on 1.4T tokens with Gopher's compute, beat it.

## About this fit
The paper minimised a Huber loss on log-loss residuals with L-BFGS from a grid of initialisations, which is more robust to outliers. Grid search over the exponents with linear least squares for the rest is a simple way to get the same kind of fit and shows why the problem is easy once the exponents are fixed. The fitted values reported in the paper were roughly $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$.

## Why the allocation exponent is $\beta/(\alpha+\beta)$
Whichever term decays more slowly limits the loss, so more of the budget goes to the variable whose term is slower: a smaller $\beta$ (data helps more slowly) means $D$ must grow faster, and the exponents always sum to 1 because $N\cdot D \propto C$.

## Common bugs
- Swapping the exponents in the allocation formula.
- Forgetting the factor 6 in $C = 6ND$.
- Fitting in log space for $L$ but then using the linear-space parameters (or vice versa).

## Interview follow-ups
- Why do many current models train far beyond 20 tokens per parameter?
- How does the irreducible term $E$ affect what you can extrapolate from small runs?
"""
