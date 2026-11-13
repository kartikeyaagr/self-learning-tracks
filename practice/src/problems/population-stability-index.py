import numpy as np

META = dict(title='Population Stability Index for feature drift', tags=['MLOps', 'Feature Engineering'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:11.12', 'air-fanout-v1:11.01'], related=['two-sample-ks-drift', 'percentiles-quantiles'])

STATEMENT = r"""
A model was trained on one population and now scores another. The **Population Stability Index** (PSI) is the usual first check that a feature's distribution hasn't moved. Bin both samples with the same edges, turn counts into proportions $r_i$ (reference) and $c_i$ (current), and sum

$$\mathrm{PSI} = \sum_i (c_i - r_i)\,\ln\frac{c_i}{r_i}$$

Every term is $\ge 0$, so PSI is $0$ only when the two binned distributions match.

Write `psi(reference, current, edges, eps=1e-4)` that returns the tuple `(value, label)`.

## Binning
- `edges` is a strictly increasing 1-D array of $k \ge 1$ inner cut points. They define $k+1$ bins: $(-\infty, e_0)$, $[e_0, e_1)$, …, $[e_{k-1}, +\infty)$.
- A value equal to an edge goes to the bin on its **right**.
- `reference` and `current` are non-empty 1-D float arrays (they may have different lengths).

## Empty bins
A bin that is empty in one sample would give $\ln 0$. Floor every proportion at `eps`: use $\max(p_i, \varepsilon)$ for both $r_i$ and $c_i$, and don't renormalise afterwards.

## Output
- `value`: the PSI as a Python float.
- `label`: `'stable'` if `value < 0.1`, `'moderate'` if `0.1 <= value < 0.25`, otherwise `'major'`.
"""

STARTER = '''import numpy as np

def psi(reference, current, edges, eps=1e-4):
    """Return (psi_value, label) for two samples binned on the same inner edges."""
    counts_r, _ = np.histogram(reference, bins=edges)    # careful: which bins does this cover?
    counts_c, _ = np.histogram(current, bins=edges)
    r, c = counts_r / counts_r.sum(), counts_c / counts_c.sum()
    value = float(np.sum((c - r) * np.log(c / r)))
    return value, 'stable' if value < 0.1 else 'major'
'''

SOLUTION = '''import numpy as np

def psi(reference, current, edges, eps=1e-4):
    edges = np.asarray(edges, dtype=float)

    def proportions(x):
        x = np.asarray(x, dtype=float)
        # side='right': a value equal to edge e_j lands in the bin that starts at e_j.
        idx = np.searchsorted(edges, x, side='right')            # 0..k, one bin per index
        counts = np.bincount(idx, minlength=len(edges) + 1)
        return np.maximum(counts / len(x), eps)                  # floor, no renormalising

    r, c = proportions(reference), proportions(current)
    value = float(np.sum((c - r) * np.log(c / r)))
    label = 'stable' if value < 0.1 else ('moderate' if value < 0.25 else 'major')
    return value, label
'''

HINTS = [
    '`np.histogram(x, bins=edges)` only covers the range between the first and last edge and drops everything outside. You need the two open-ended tail bins as well.',
    '`np.searchsorted(edges, x, side="right")` gives each value its bin index from 0 to k, with values equal to an edge going right. `np.bincount(idx, minlength=k + 1)` counts them.',
    'Divide counts by the sample length, apply `np.maximum(p, eps)` to both samples, then sum `(c - r) * log(c / r)` and map the value to a label with the 0.1 and 0.25 cut-offs.',
]

_rng = np.random.default_rng(7)
_ref = _rng.normal(0.0, 1.0, 2000)
_cur = _rng.normal(0.4, 1.2, 1500)
_edges = np.quantile(_ref, np.linspace(0.1, 0.9, 9))   # deciles of the reference, the usual production choice

TESTS = [
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), np.array([3.0, 6.0])),
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]), np.array([3.0, 6.0, 9.0])),
    (np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 1.0, 3.0, 3.0]), np.array([1.0, 3.0])),
    (np.array([-5.0, 0.0, 5.0, 10.0, 15.0]), np.array([0.5, 0.6, 0.7]), np.array([0.0, 7.5])),
    (np.array([0.1, 0.2, 0.3, 0.9]), np.array([0.15, 0.25, 0.35, 0.95, 0.85, 0.05]), np.array([0.5])),
    (np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0, 4.0]), np.array([2.5]), 0.01),
    (_ref, _ref[:1000], _edges),
    (_ref, _cur, _edges),
]

EXPLANATION = r"""
## Why it works
PSI is a symmetrised KL divergence between the two binned distributions: $\mathrm{KL}(c\,\|\,r) + \mathrm{KL}(r\,\|\,c) = \sum_i (c_i - r_i)\ln(c_i/r_i)$. Each term has $c_i - r_i$ and $\ln(c_i/r_i)$ with the same sign, so every term is non-negative and only bins that actually moved contribute. The 0.1 / 0.25 cut-offs are an industry convention from credit scoring, not a statistical test: they ignore sample size.

Binning with `searchsorted(..., side='right')` makes the two tail bins explicit, so a value far outside the training range still counts. That's exactly the drift you most want to see.

## Complexity
$O((n + m)\log k)$ for the binary searches plus $O(k)$ for the sum.

## Common bugs
- Using `np.histogram(x, bins=edges)` with the inner edges: values below the first edge or above the last are silently dropped, so tail drift is invisible (test 4).
- Choosing edges from the **current** sample, or re-fitting them per batch. Edges come from the reference and stay fixed so PSI values are comparable over time.
- Forgetting the floor: one empty bin gives `inf` or `nan`. The value of `eps` matters (test 6): it caps how much one empty bin can contribute.
- Renormalising after flooring, which changes every other term slightly.

## Interview follow-ups
- PSI on a categorical feature: what replaces the edges, and how do you handle a category never seen in training? (One bin per category plus an "other/unseen" bin.)
- With 10 million rows per day, PSI almost never fires but the model degrades anyway. What drift does PSI miss? (Drift in the joint distribution or in $P(y \mid x)$, i.e. concept drift, which needs labels.)
"""
