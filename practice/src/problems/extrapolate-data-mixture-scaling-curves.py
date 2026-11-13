import numpy as np

META = dict(title='Extrapolate per-domain scaling curves to pick a data mixture', tags=['Scaling Laws', 'Regression'],
            diff='Hard', runtime='numpy', lessons=[],
            related=['saturating-power-law-fit', 'power-law-log-fit', 'temperature-based-data-mixtures'])

STATEMENT = r"""
Choosing a pretraining mixture by training a full-size model per candidate is unaffordable. A cheaper recipe: fit a scaling curve per domain from small runs, extrapolate each domain's loss to the tokens it would get at full scale under each candidate mixture, and pick the mixture with the best weighted loss. Domains with little unique data must be **repeated** at full scale, and repeated tokens are worth less than fresh ones, which is what stops the recipe from piling all the weight onto one small, valuable domain.

Write `pick_mixture(domain_tokens, domain_loss, floors, unique_tokens, r_star, mixtures, target_tokens, eval_weights)`.

## Step 1: fit each domain
For domain $k$ you have $R \ge 2$ small runs: `domain_tokens[k, j]` tokens of that domain were seen (all fresh, no repetition) and `domain_loss[k, j]` is the validation loss on that domain. With the known floor $E_k$ = `floors[k]` (strictly below every loss of that domain), fit

$$L_k(D) = E_k + A_k D^{-\alpha_k}$$

by ordinary least squares of $\ln(L_{k,j} - E_k)$ on $\ln D_{k,j}$ with an intercept: slope $= -\alpha_k$, intercept $= \ln A_k$. Each row of `domain_tokens` has at least two distinct values.

## Step 2: effective tokens at full scale
Mixture $m$ (row of `mixtures`, strictly positive weights summing to 1) gives domain $k$ the raw token count $D_{mk} = w_{mk} \cdot$ `target_tokens`. With $U_k$ = `unique_tokens[k]` and $R^\star$ = `r_star`, the **effective** count is

$$D'_{mk} = \begin{cases} D_{mk} & D_{mk} \le U_k \\ U_k + U_k R^\star \left(1 - e^{-(D_{mk}/U_k - 1)/R^\star}\right) & D_{mk} > U_k \end{cases}$$

so the first epoch counts in full and each further repetition counts for less, saturating at $U_k(1 + R^\star)$.

## Step 3: score mixtures
Predicted loss $\hat L_{mk} = E_k + A_k (D'_{mk})^{-\alpha_k}$ and objective $J_m = \sum_k v_k \hat L_{mk}$ with $v$ = `eval_weights`.

## Output
A dict:

| key | value |
|---|---|
| `"A"`, `"alpha"` | float64 `(K,)` |
| `"effective_tokens"` | float64 `(M, K)`, $D'$ |
| `"pred"` | float64 `(M, K)`, $\hat L$ |
| `"objective"` | float64 `(M,)`, $J$ |
| `"best"` | int, index of the smallest $J$ (lowest index on an exact tie) |

All inputs are float64 arrays except `r_star` and `target_tokens` (floats). Shapes: `domain_tokens`, `domain_loss` `(K, R)`; `floors`, `unique_tokens`, `eval_weights` `(K,)`; `mixtures` `(M, K)`. Vectorise over mixtures.
"""

STARTER = '''import numpy as np

def pick_mixture(domain_tokens, domain_loss, floors, unique_tokens, r_star, mixtures, target_tokens, eval_weights):
    """Return a dict with A, alpha, effective_tokens, pred, objective, best."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pick_mixture(domain_tokens, domain_loss, floors, unique_tokens, r_star, mixtures, target_tokens, eval_weights):
    D = np.asarray(domain_tokens, dtype=float)
    L = np.asarray(domain_loss, dtype=float)
    E = np.asarray(floors, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)
    W = np.asarray(mixtures, dtype=float)

    # Step 1: one log-log OLS per domain, vectorised over rows.
    x = np.log(D)
    y = np.log(L - E[:, None])
    xc = x - x.mean(axis=1, keepdims=True)
    slope = (xc * (y - y.mean(axis=1, keepdims=True))).sum(axis=1) / (xc * xc).sum(axis=1)
    intercept = y.mean(axis=1) - slope * x.mean(axis=1)
    A, alpha = np.exp(intercept), -slope

    # Step 2: raw tokens per domain, then discount repeated epochs.
    raw = W * float(target_tokens)                              # (M, K)
    extra_epochs = np.maximum(raw / U - 1.0, 0.0)                # 0 while the domain still has fresh data
    repeated = U + U * r_star * (1.0 - np.exp(-extra_epochs / r_star))
    eff = np.where(raw <= U, raw, repeated)

    # Step 3: extrapolate each domain, weight, pick the best.
    pred = E + A * eff ** -alpha
    objective = pred @ np.asarray(eval_weights, dtype=float)
    return {"A": A, "alpha": alpha, "effective_tokens": eff, "pred": pred,
            "objective": objective, "best": int(np.argmin(objective))}
'''

HINTS = [
    'Step 1 is $K$ independent straight-line fits in log-log space. Centre `log(domain_tokens)` per row and the slope is a ratio of two row sums.',
    'Compute the repeated-data branch for every entry with `max(D/U - 1, 0)` as the number of extra epochs, then choose with `np.where(raw <= U, raw, repeated)`. Both branches agree at $D = U$, so the curve is continuous.',
    'Broadcast `(K,)` parameters against the `(M, K)` effective-token matrix, then `pred @ eval_weights` gives all objectives at once. `np.argmin` already returns the first minimum.',
]

_E = np.array([1.2, 0.8, 1.5])
_A = np.array([30.0, 12.0, 50.0])
_al = np.array([0.18, 0.12, 0.25])
_Dsmall = np.array([[1e8, 3e8, 1e9, 3e9]] * 3)
_r = np.random.default_rng(3)
_Lsmall = (_E[:, None] + _A[:, None] * _Dsmall ** -_al[:, None]) * np.exp(_r.normal(0.0, 0.003, size=(3, 4)))
_mix = np.array([[0.6, 0.2, 0.2], [0.34, 0.33, 0.33], [0.2, 0.2, 0.6], [0.5, 0.45, 0.05], [0.1, 0.8, 0.1]])
TESTS = [
    (np.array([[1.0, 100.0]]), np.array([3.0, 1.2]), np.array([1.0]), np.array([1e6]), 5.0,
     np.array([[1.0]]), 10000.0, np.array([1.0])),
    (np.array([[1.0, 100.0], [1.0, 100.0]]), np.array([[3.0, 1.2], [2.0, 1.1]]), np.array([1.0, 1.0]),
     np.array([1e6, 50.0]), 2.0, np.array([[0.5, 0.5], [0.9, 0.1]]), 1000.0, np.array([0.5, 0.5])),
    (_Dsmall, _Lsmall, _E, np.array([1e12, 1e12, 1e12]), 15.0, _mix, 1e11, np.array([0.5, 0.3, 0.2])),
    (_Dsmall, _Lsmall, _E, np.array([1e9, 1e12, 1e12]), 15.0, _mix, 1e11, np.array([0.5, 0.3, 0.2])),
    (_Dsmall, _Lsmall, _E, np.array([2e10, 1e12, 5e9]), 1.0, _mix, 1e12, np.array([0.2, 0.2, 0.6])),
    (np.array([[10.0, 20.0, 40.0], [10.0, 20.0, 40.0]]), np.array([[2.0, 1.5, 1.25], [2.0, 1.5, 1.25]]),
     np.array([1.0, 1.0]), np.array([1e9, 1e9]), 3.0, np.array([[0.5, 0.5], [0.25, 0.75], [0.75, 0.25]]),
     160.0, np.array([0.5, 0.5])),
    (_Dsmall[:2], _Lsmall[:2], _E[:2], np.array([1e9, 1e9]), 4.0,
     np.array([[0.5, 0.5], [0.99, 0.01], [0.01, 0.99]]), 4e9, np.array([0.7, 0.3])),
]

EXPLANATION = r"""
## Why fit per domain
Validation loss on code depends mostly on how much code the model saw, not on how much prose it saw. Modelling each domain's loss as a function of its own token count turns one expensive $K$-dimensional search into $K$ cheap one-dimensional fits, and a candidate mixture is scored by plain arithmetic. The price is ignoring transfer between domains (prose does help code a little), which more elaborate mixing laws model with cross-domain terms.

## Why the repetition discount matters
Without it, the extrapolated loss of a domain keeps falling as $D^{-\alpha}$ forever, so the optimum puts weight on whichever domain has the best curve, even if that means 40 epochs of a small corpus. With the discount, extra epochs saturate at $U(1+R^\star)$ effective tokens: test 4 has the same curves as test 3 but only $10^9$ unique tokens in domain 0, and the best mixture moves from index 3 (heavy on domain 0) to index 1 (balanced). Test 5 sets $R^\star = 1$, which makes repetition almost worthless.

## Complexity
Fitting is $O(KR)$; scoring is $O(MK)$ with everything vectorised.

## Common bugs
- Fitting $\ln L$ instead of $\ln(L - E)$: the curve then has no floor and extrapolates to absurdly low losses.
- Applying the discount to the raw count of extra epochs only, forgetting the first $U$ tokens, or using $D/U$ instead of $D/U - 1$ (that discounts the first epoch too).
- Using `np.minimum`/`np.maximum` without guarding the exponent: for $D \le U$ the repeated branch must not be used, and $-(D/U - 1)/R^\star$ would be positive there.

## Interview follow-ups
- Small runs pick mixtures that are often wrong at scale. Which assumptions of this procedure fail first as the model grows?
- How would you add a cross-domain transfer term, and how many extra small runs would you need to fit it?
"""
