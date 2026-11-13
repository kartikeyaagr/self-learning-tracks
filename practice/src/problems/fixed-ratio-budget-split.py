import numpy as np

META = dict(title='Split a compute budget at a tokens-per-parameter ratio', tags=['Scaling Laws'], diff='Easy', runtime='numpy',
            lessons=[], related=['training-compute-budget', 'chinchilla-compute-optimal'])

STATEMENT = r"""
A common rule of thumb for compute-optimal pretraining is to train on a fixed number of tokens per parameter, $D = rN$ (about $r = 20$ is a popular value). Combined with the cost model $C = 6ND$, a budget $C$ pins down both $N$ and $D$.

Real projects often run out of data: if the rule asks for more tokens than you have unique tokens, you train on all the data once and spend the rest of the budget on a bigger model.

Write `split_budget(budgets, ratio=20.0, max_tokens=None)` that returns the tuple `(n_params, n_tokens)`.

## Rules
- `budgets`: 1-D float array of compute budgets $C$ in FLOPs (all positive). Return two float arrays of the same shape.
- Without a cap, solve $C = 6ND$ and $D = rN$ for $N$ and $D$.
- `max_tokens`: `None` (no cap) or a positive float $D_\text{max}$. For a budget whose uncapped $D$ is **strictly greater** than $D_\text{max}$, use $D = D_\text{max}$ and $N = C / (6 D_\text{max})$. Other budgets are unchanged.
- No loops over budgets.
"""

STARTER = '''import numpy as np

def split_budget(budgets, ratio=20.0, max_tokens=None):
    """Return (n_params, n_tokens), float arrays shaped like budgets."""
    C = np.asarray(budgets, dtype=float)
    n = C / (6 * ratio)          # not quite...
    return n, ratio * n
'''

SOLUTION = '''import numpy as np

def split_budget(budgets, ratio=20.0, max_tokens=None):
    C = np.asarray(budgets, dtype=float)
    # C = 6 N D and D = r N  =>  C = 6 r N^2
    n = np.sqrt(C / (6.0 * ratio))
    d = ratio * n
    if max_tokens is not None:
        capped = d > max_tokens                      # data-limited budgets
        d = np.where(capped, float(max_tokens), d)
        n = np.where(capped, C / (6.0 * max_tokens), n)
    return n, d
'''

HINTS = [
    'Substitute $D = rN$ into $C = 6ND$. You get a quadratic in $N$ with one positive root.',
    '$N = \\sqrt{C / (6r)}$ and $D = rN$.',
    'For the cap, build a boolean mask `d > max_tokens` and use `np.where` for both arrays. Recompute $N$ from the capped $D$ so $6ND$ still equals $C$.',
]

TESTS = [
    (np.array([1.2e21, 4.8e21]),),
    (np.array([6.0, 600.0]), 1.0),
    (np.array([5.76e23]),),
    (np.array([1e18, 1e20, 1e22, 1e24]), 20.0, 1e11),
    (np.array([1e21, 1e23, 1e25]), 200.0),
    (np.array([3e22, 3e24]), 20.0, 1e15),
    (np.array([1e20, 1e26]), 5.0, 2e12),
    (np.exp(np.random.default_rng(3).uniform(np.log(1e19), np.log(1e25), size=6)), 20.0, 5e11),
]

EXPLANATION = r"""
## Why a square root
With $D = rN$ the cost is $C = 6rN^2$, so $N = \sqrt{C/(6r)}$ and $D = \sqrt{rC/6}$. Both grow as $C^{0.5}$: doubling compute should grow model and data each by $\sqrt 2$. That equal split is what compute-optimal fits found; earlier work that put most of the extra compute into $N$ trained models that were too big for their data.

## The data cap
When data runs out, $D$ is fixed and every extra FLOP has to go into $N = C/(6D_\text{max})$, which now grows linearly with $C$. The model ends up "over-parameterised" for its data. The alternatives are repeating epochs (worth roughly as much as fresh data for the first few passes, then rapidly less) or finding more data.

## Complexity
$O(B)$ for $B$ budgets, fully vectorised.

## Common bugs
- Writing $N = C/(6r)$, forgetting that $N$ appears twice in $6ND$.
- Capping $D$ but leaving $N$ alone, which silently spends less than the budget.
- Using `>=` instead of `>` at the cap. Here it gives the same numbers, but it's worth matching the spec.

## Interview follow-ups
- Inference is paid per token for the life of the model. Why do teams deliberately train past the compute-optimal ratio (e.g. hundreds of tokens per parameter)?
- With data capped, would you rather repeat data for 4 epochs or train a bigger model for one? What do you need to know to decide?
"""
