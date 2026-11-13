import numpy as np

META = dict(title='Compute-optimal model and data size from a fitted loss', tags=['Scaling Laws', 'Optimization'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:09.08'], related=['fixed-ratio-budget-split', 'training-compute-budget', 'isoflop-sweep-exponent'])

STATEMENT = r"""
A widely used scaling-law form models the final pretraining loss of a model with $N$ parameters trained on $D$ tokens as

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

where $E$ is the irreducible loss and $A, B, \alpha, \beta > 0$ come from a fit to many training runs. With training cost $C = 6ND$, each budget $C$ has one $(N, D)$ that minimises $L$.

Write `compute_optimal(budgets, E, A, B, alpha, beta)` that returns the tuple `(n_opt, d_opt, loss_opt)`.

## Inputs and output
- `budgets`: 1-D float array of compute budgets in FLOPs, all positive.
- `E, A, B, alpha, beta`: positive floats.
- Return three float arrays shaped like `budgets`: the optimal parameter count, the optimal token count (so that $6 \cdot$ `n_opt` $\cdot$ `d_opt` $= C$), and $L$ at that point.
- $N$ and $D$ are continuous: no rounding. No loops over budgets.

Derive the closed form; a numerical search over $N$ will not be accurate enough for the tests' tolerance of $10^{-6}$ relative.
"""

STARTER = '''import numpy as np

def compute_optimal(budgets, E, A, B, alpha, beta):
    """Return (n_opt, d_opt, loss_opt) as float arrays shaped like budgets."""
    C = np.asarray(budgets, dtype=float)
    n = np.sqrt(C / 120.0)                      # the 20-tokens-per-parameter rule, not the optimum
    d = C / (6.0 * n)
    return n, d, E + A / n ** alpha + B / d ** beta
'''

SOLUTION = '''import numpy as np

def compute_optimal(budgets, E, A, B, alpha, beta):
    C = np.asarray(budgets, dtype=float)
    s = alpha + beta
    G = (alpha * A / (beta * B)) ** (1.0 / s)
    # dL/dN = 0 along D = C/(6N) gives N^(alpha+beta) = (alpha A / (beta B)) * (C/6)^beta
    n = G * (C / 6.0) ** (beta / s)
    d = (C / 6.0) / n                            # = G^-1 (C/6)^(alpha/s); spends the whole budget
    loss = E + A / n ** alpha + B / d ** beta
    return n, d, loss
'''

HINTS = [
    'Eliminate $D$ with the constraint: $D = C/(6N)$. Now $L$ is a function of $N$ alone.',
    'Set $dL/dN = 0$: $\\alpha A N^{-\\alpha - 1} = \\beta B\\, (6/C)^{\\beta} N^{\\beta - 1}$. Collect powers of $N$ on one side.',
    'You should get $N_\\text{opt} = G\\,(C/6)^{\\beta/(\\alpha+\\beta)}$ with $G = \\left(\\frac{\\alpha A}{\\beta B}\\right)^{1/(\\alpha+\\beta)}$, and $D_\\text{opt} = (C/6)/N_\\text{opt}$.',
]

TESTS = [
    (np.array([6.0, 600.0]), 0.0, 1.0, 1.0, 0.5, 0.5),
    (np.array([1e21, 1e23]), 1.8, 400.0, 420.0, 0.33, 0.29),
    (np.array([5.76e23]), 1.8, 400.0, 420.0, 0.33, 0.29),
    (np.array([1e18, 1e20, 1e22, 1e24, 1e26]), 1.7, 480.0, 380.0, 0.36, 0.27),
    (np.array([1e22]), 2.0, 50.0, 50.0, 0.3, 0.3),
    (np.array([1e20, 1e25]), 1.5, 1000.0, 100.0, 0.25, 0.4),
    (np.exp(np.random.default_rng(5).uniform(np.log(1e19), np.log(1e25), size=5)), 1.9, 350.0, 450.0, 0.31, 0.31),
]

EXPLANATION = r"""
## Why it works
On the constraint $D = C/(6N)$ the loss is $E + A N^{-\alpha} + B (6/C)^\beta N^{\beta}$: one term falls with $N$, the other rises, and the sum is convex in $\ln N$, so the stationary point is the unique minimum. Solving it gives

$$N_\text{opt} = G \left(\tfrac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}, \qquad D_\text{opt} = G^{-1} \left(\tfrac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}}$$

The exponents add to 1, as they must since $ND = C/6$. When $\alpha \approx \beta$, both are near $0.5$: grow model and data together. The ratio $D_\text{opt}/N_\text{opt} = G^{-2}(C/6)^{(\alpha-\beta)/(\alpha+\beta)}$ is constant only when $\alpha = \beta$; otherwise the "right" tokens per parameter drifts with scale (test 4 shows it).

At the optimum the two reducible terms stand in a fixed ratio: $\alpha \cdot A N^{-\alpha} = \beta \cdot B D^{-\beta}$. That's a handy check.

## Complexity
$O(B)$ for $B$ budgets.

## Common bugs
- Forgetting the 6, i.e. using $C$ instead of $C/6$ inside the power.
- Swapping the exponents. $N$ grows as $C^{\beta/(\alpha+\beta)}$, with the *data* exponent on top: when $\beta$ is small the data term falls slowly, so the optimum keeps pushing compute into data. Sanity-check with $\alpha = \beta$, where both exponents are 0.5.
- Computing $D$ from its own closed form and $N$ from another, then losing a few ulps so $6ND \ne C$. Deriving $D = (C/6)/N$ avoids it.

## Interview follow-ups
- Fits of this form from different labs disagree on $\alpha, \beta$ by enough to change the 1e25-FLOP recommendation by 2x. What in the experimental setup (learning-rate schedule length, embedding parameters in $N$, data mix) causes that?
- If inference cost matters, add $2N \cdot T_\text{inf}$ FLOPs for $T_\text{inf}$ lifetime inference tokens. How does the optimum move?
"""
