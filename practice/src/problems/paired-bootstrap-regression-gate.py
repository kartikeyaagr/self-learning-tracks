import numpy as np

META = dict(title='Paired bootstrap regression gate for a prompt change', tags=['Metrics & Evaluation', 'Probability and Statistics'], diff='Medium', runtime='numpy',
            lessons=['aieng:10.06', 'mlmaths:13.05'], related=['bootstrap-confidence-interval', 'standard-error-comparison', 'pass-at-k-and-pass-hat-k'])

STATEMENT = r"""
CI runs a baseline and a candidate prompt on the same eval cases, each case `r` times. Decide whether the candidate may merge. Write `regression_gate(base, cand, margin, B, alpha, seed)`.

- `base`, `cand`: float arrays of shape `(n, r)`, scores in $[0, 1]$; row $i$ is case $i$ in both.
- `margin` $\delta > 0$: the largest drop in mean score the team accepts.

Steps:
1. Average each case's repeats: $b_i$ and $c_i$ are the row means. Let $d_i = c_i - b_i$ and `diff` $= \bar d$.
2. Paired bootstrap over **cases**, with exactly these calls:
   ```
   rng = np.random.default_rng(seed)
   idx = rng.integers(0, n, size=(B, n))
   ```
   Row $j$ of `idx` gives resample $j$; its statistic is the mean of `d[idx[j]]`.
3. `ci` $= (lo, hi)$: the $\alpha/2$ and $1-\alpha/2$ quantiles of the $B$ statistics (`np.quantile`, default method).
4. `verdict`, checked in this order: `"improved"` if $lo > 0$; `"pass"` if $lo > -\delta$; `"regressed"` if $hi < 0$; otherwise `"inconclusive"`. `block` is True for `"regressed"` and `"inconclusive"`.
5. `wins` / `losses`: number of cases with $d_i > 0$ / $d_i < 0$.
6. `flaky`: number of cases whose repeats are not all equal in at least one of the two arms.

Return `{"diff": float, "ci": [lo, hi], "verdict": str, "block": bool, "wins": int, "losses": int, "flaky": int}` with Python floats and ints.
"""

STARTER = '''import numpy as np

def regression_gate(base, cand, margin, B, alpha, seed):
    """Gate a candidate against a baseline evaluated on the same cases."""
    b, c = base.mean(), cand.mean()
    diff = float(c - b)
    se = float(np.sqrt(b * (1 - b) / base.size + c * (1 - c) / cand.size))   # unpaired, and per trial
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    verdict = "pass" if lo > -margin else "regressed"
    return {"diff": diff, "ci": [lo, hi], "verdict": verdict, "block": verdict != "pass",
            "wins": 0, "losses": 0, "flaky": 0}
'''

SOLUTION = '''import numpy as np

def regression_gate(base, cand, margin, B, alpha, seed):
    base, cand = np.asarray(base, dtype=float), np.asarray(cand, dtype=float)
    d = cand.mean(axis=1) - base.mean(axis=1)          # one paired difference per case
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))               # resample cases, never individual repeats
    boot = d[idx].mean(axis=1)
    lo, hi = (float(v) for v in np.quantile(boot, [alpha / 2, 1 - alpha / 2]))
    if lo > 0:
        verdict = "improved"
    elif lo > -margin:
        verdict = "pass"                                # non-inferior: any drop is smaller than margin
    elif hi < 0:
        verdict = "regressed"
    else:
        verdict = "inconclusive"
    flaky = (base.max(axis=1) != base.min(axis=1)) | (cand.max(axis=1) != cand.min(axis=1))
    return {"diff": float(d.mean()), "ci": [lo, hi], "verdict": verdict,
            "block": verdict in ("regressed", "inconclusive"),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "flaky": int(flaky.sum())}
'''

HINTS = [
    'Average repeats within each case first. The unit you resample is the case: repeats of one case are correlated, and treating them as independent makes the interval too narrow.',
    'Pairing means computing d = cand_case_mean - base_case_mean and bootstrapping the mean of d. Easy and hard cases cancel out, so the interval is much narrower than comparing two overall means.',
    'Check the verdicts in the stated order. A small but certain drop (hi < 0 but lo > -margin) is still a "pass": the gate asks whether a drop bigger than the margin is ruled out.',
]

def _lesson_case():
    b = np.array([1.0] * 160 + [0.0] * 40)
    c = b.copy()
    c[:9] = 0.0
    c[160:165] = 1.0
    return b[:, None], c[:, None]

def _repeats(seed, n, r, p_base, shift):
    g = np.random.default_rng(seed)
    difficulty = g.uniform(0, 1, size=n)                        # shared per-case difficulty -> correlated arms
    pb = np.clip(p_base + 0.5 - difficulty, 0, 1)
    pc = np.clip(pb + shift, 0, 1)
    base = (g.uniform(size=(n, r)) < pb[:, None]).astype(float)
    cand = (g.uniform(size=(n, r)) < pc[:, None]).astype(float)
    return base, cand

_L = _lesson_case()
_A = _repeats(1, 300, 3, 0.75, 0.0)
_Bq = _repeats(2, 150, 3, 0.7, -0.12)
_C = _repeats(3, 400, 5, 0.6, 0.08)
_J = np.random.default_rng(4).integers(1, 6, size=(60, 2)) / 5.0   # judge scores 1-5 rescaled

TESTS = [
    (_L[0], _L[1], 0.03, 10000, 0.05, 0),
    (_L[0], _L[1], 0.06, 10000, 0.05, 0),
    (_A[0], _A[1], 0.05, 4000, 0.05, 7),
    (_Bq[0], _Bq[1], 0.03, 4000, 0.05, 11),
    (_C[0], _C[1], 0.02, 4000, 0.05, 13),
    (_J, np.clip(_J - 0.2 * (np.arange(60) % 7 == 0)[:, None], 0, 1), 0.01, 3000, 0.1, 5),
]

EXPLANATION = r"""
## Why paired
Both prompts see the same cases, and case difficulty dominates the variance: easy cases pass under both, hard ones fail under both. Per-case differences cancel that shared part, $\operatorname{Var}(\bar d) = (\sigma_b^2 + \sigma_c^2 - 2\operatorname{Cov})/n$. In the first test (200 cases, 9 losses, 5 wins) the paired interval is about $(-5.5, +1.5)$ points; an unpaired interval on the two pass rates would be about $(-10, +6)$, too wide to decide anything.

## Why resample cases
Repeats of one case share its difficulty, so they are a cluster. Averaging them first and resampling cases keeps the interval honest; resampling all $n \cdot r$ trials as if independent shrinks it by roughly $\sqrt{r}$ when repeats agree, and the gate starts blocking good changes and passing bad ones by chance. The starter makes both mistakes (unpaired, per trial).

## Why a non-inferiority margin
"Block if the score dropped" blocks half of all harmless changes, because noise is symmetric. "Block if the drop is significant" lets real regressions through on small sets. Asking whether a drop larger than $\delta$ is ruled out puts the burden on the candidate and makes the required set size explicit: tests 1 and 2 differ only in the margin, and the verdict flips from inconclusive to pass.

## Complexity
$O(Bn)$ time and memory for the index matrix; chunk over $B$ for very large sets.

## Interview follow-ups
- For binary scores, what closed-form test matches this? (McNemar's test on the discordant cases.)
- How do you check the gate is calibrated? (An A/A run: baseline against itself should be "regressed" about $\alpha/2$ of the time.)
- The suite has 12 metrics. What changes? (One primary metric with this gate; the rest as looser guardrails, or a multiple-comparison correction.)
"""
