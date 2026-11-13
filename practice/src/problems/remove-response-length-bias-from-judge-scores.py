import numpy as np

META = dict(title='Remove response-length bias from LLM-judge scores', tags=['Evaluation Metrics', 'Regression'],
            diff='Hard', runtime='numpy', lessons=['air-fanout-v1:10.05'],
            related=['fit-elo-ratings-from-pairwise-votes', 'linear-regression', 'score-multiple-choice-continuations'])

STATEMENT = r"""
LLM judges reward longer answers, so a model can climb a judge-scored leaderboard just by being verbose. One correction fits a regression of score on (log) length **with a separate intercept per model** and removes the length term. The per-model intercepts matter: better models may legitimately write longer answers, and a single pooled regression would mistake that for length bias.

Write `length_debias(scores, lengths, model_ids, n_models)` and return a dict.

## Inputs
- `scores`: float64 `(N,)` judge scores; `lengths`: int `(N,)` response lengths in tokens (positive); `model_ids`: int `(N,)` in `0 .. n_models-1`. Every model has at least one response.

## Model
Let $x_i = \ln(\text{length}_i) - \overline{\ln(\text{length})}$ (centred over **all** $N$ responses). Fit by ordinary least squares

$$s_i = \alpha_{m_i} + \gamma\, x_i$$

with one intercept $\alpha_m$ per model and one shared slope $\gamma$. The slope has a closed form: with $\bar x_m$ and $\bar s_m$ the per-model means,

$$\gamma = \frac{\sum_i (x_i - \bar x_{m_i})(s_i - \bar s_{m_i})}{\sum_i (x_i - \bar x_{m_i})^2}, \qquad \alpha_m = \bar s_m - \gamma\, \bar x_m$$

If the denominator is 0 (no model has any length variation), $\gamma = 0$.

Also report the **naive** pooled slope $\gamma_\text{naive}$: OLS of $s$ on $x$ with a single intercept.

## Output
| key | value |
|---|---|
| `"gamma"`, `"gamma_naive"` | floats |
| `"raw_means"` | float64 `(M,)`, $\bar s_m$ |
| `"adjusted_means"` | float64 `(M,)`, $\alpha_m$: each model's expected score at the average length |
| `"adjusted"` | float64 `(N,)`, $s_i - \gamma x_i$ |
| `"rank_raw"`, `"rank_adjusted"` | lists of ints, models sorted from best to worst by `raw_means` / `adjusted_means` (ties: lower index first) |

No Python loops over responses; use `np.bincount` or similar for the group means.
"""

STARTER = '''import numpy as np

def length_debias(scores, lengths, model_ids, n_models):
    """Return a dict: gamma, gamma_naive, raw_means, adjusted_means, adjusted, rank_raw, rank_adjusted."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def length_debias(scores, lengths, model_ids, n_models):
    s = np.asarray(scores, dtype=float)
    g = np.asarray(model_ids)
    x = np.log(np.asarray(lengths, dtype=float))
    x -= x.mean()
    counts = np.bincount(g, minlength=n_models)
    group_mean = lambda v: np.bincount(g, weights=v, minlength=n_models) / counts

    s_bar, x_bar = group_mean(s), group_mean(x)
    xw, sw = x - x_bar[g], s - s_bar[g]                  # within-model deviations (Frisch-Waugh)
    den = xw @ xw
    gamma = float(xw @ sw / den) if den > 0 else 0.0
    alpha = s_bar - gamma * x_bar

    gamma_naive = float(x @ (s - s.mean()) / (x @ x)) if x @ x > 0 else 0.0   # x is already centred
    rank = lambda v: [int(i) for i in np.argsort(-v, kind="stable")]
    return {"gamma": gamma, "gamma_naive": gamma_naive, "raw_means": s_bar, "adjusted_means": alpha,
            "adjusted": s - gamma * x, "rank_raw": rank(s_bar), "rank_adjusted": rank(alpha)}
'''

HINTS = [
    'Group means are `np.bincount(model_ids, weights=v, minlength=M) / counts`. Index them with `model_ids` to broadcast back to responses.',
    'With model intercepts in the regression, the slope only uses variation **within** each model: subtract each model\'s mean from both $x$ and $s$, then take the ordinary slope through the origin.',
    'Rank with `np.argsort(-means, kind="stable")` so equal means keep the lower index first. The naive slope is the same formula with the global means instead of per-model means.',
]


def _judge(seed, quality, base_len, n_per, bias):
    r = np.random.default_rng(seed)
    ids = np.repeat(np.arange(len(quality)), n_per)
    lengths = np.round(np.array(base_len)[ids] * np.exp(r.normal(0.0, 0.4, size=len(ids)))).astype(np.int64) + 1
    scores = np.array(quality)[ids] + bias * np.log(lengths) + r.normal(0.0, 0.3, size=len(ids))
    return scores, lengths, ids, len(quality)


TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0]), np.array([10, 20, 10, 20]), np.array([0, 0, 1, 1]), 2),
    (np.array([5.0, 6.0, 7.0]), np.array([100, 100, 100]), np.array([0, 1, 1]), 2),
    _judge(0, [6.0, 6.5, 7.0], [200, 400, 800], 30, 0.8),
    _judge(1, [7.5, 6.0, 6.2, 5.0], [150, 900, 1200, 100], 40, 1.2),
    _judge(2, [5.0, 5.0, 5.5, 4.0, 6.0], [300, 300, 300, 300, 300], 25, 0.0),
    (np.array([3.0, 4.0, 8.0, 9.0, 2.0]), np.array([50, 100, 400, 800, 30]), np.array([0, 0, 1, 1, 2]), 3),
    _judge(3, [6.0, 6.3], [100, 1000], 200, 1.0),
]

EXPLANATION = r"""
## Why the fixed effects
Suppose stronger models also tend to write longer answers. A pooled regression of score on length (`gamma_naive`) attributes part of the quality gap to length and over-corrects, punishing the strong models. With an intercept per model, the slope is identified only from **within-model** variation: does the *same* model score higher when it happens to answer at length? That is the Frisch-Waugh-Lovell theorem: regressing on the dummies is the same as demeaning by group. In test 3, where the better models are also the wordier ones, the naive slope overstates the bias by about 45%. In test 4 the two verbose models' raw lead disappears after adjustment.

## Why log length
Judges' preference for length saturates: going from 100 to 200 tokens matters more than 2,000 to 2,100. A log transform makes the effect roughly linear and keeps very long answers from dominating the fit. Centring makes $\alpha_m$ interpretable as the score at the corpus-average log length.

## Complexity
$O(N + M)$ with `bincount`; no design matrix with $M$ dummy columns is needed.

## Common bugs
- Fitting one pooled regression and calling it debiased.
- Centring $x$ per model before fitting but then reporting $\alpha_m = \bar s_m$ (the raw mean): the adjustment must use $\bar x_m$.
- Dividing by zero when no model varies its length (test 2); the spec sets $\gamma = 0$.
- Using `np.argsort(-v)` with the default unstable sort, which may reorder ties.

## Interview follow-ups
- Length is only one bias. How would you extend the model to position bias in pairwise judging (answer A vs B order)?
- This correction assumes length doesn't *cause* quality. When is that assumption wrong, and what experiment would separate the two?
"""
