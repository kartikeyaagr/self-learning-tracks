import numpy as np

META = dict(title='Isotonic regression calibration with pool adjacent violators', tags=['Metrics & Evaluation', 'Classification'], diff='Hard', runtime='numpy',
            related=['expected-calibration-error', 'recalibrate-after-negative-downsampling', 'threshold-at-precision'])

STATEMENT = r"""
Isotonic calibration fits a non-decreasing step function from scores to probabilities. Write `isotonic_calibrate(scores, y, query)`.

## Fit (pool adjacent violators)
1. Sort training pairs by score (stable). Merge equal scores into one block whose value is the mean of their labels and whose weight is their count.
2. Scan blocks left to right, keeping a stack. Push each block; while the top two blocks violate order (the lower one's value is greater than the upper one's), merge them into one block with the weighted mean value and summed weight.
3. Each final block covers a range of scores `[lo, hi]` and has a value.

## Predict
For each query score $q$: if $q \le$ the first block's `lo`, return its value; if $q \ge$ the last block's `hi`, return the last value; otherwise linearly interpolate between the points $(\text{lo}_b, v_b)$ and $(\text{hi}_b, v_b)$ of all blocks with `np.interp`.

Return `(values, query_predictions)` where `values` are the fitted values per **training row in the original order**.
"""

STARTER = '''import numpy as np

def isotonic_calibrate(scores, y, query):
    """Return (fitted_values_for_training_rows, predictions_for_query)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def isotonic_calibrate(scores, y, query):
    s, y, q = (np.asarray(a, dtype=float) for a in (scores, y, query))
    order = np.argsort(s, kind="stable")
    ss, yy = s[order], y[order]
    blocks = []                                    # [lo, hi, value, weight]
    for u in np.unique(ss):
        m = ss == u
        blocks.append([u, u, yy[m].mean(), m.sum()])
    stack = []
    for b in blocks:
        stack.append(b)
        while len(stack) > 1 and stack[-2][2] > stack[-1][2]:
            hi = stack.pop(); lo = stack.pop()
            w = lo[3] + hi[3]
            stack.append([lo[0], hi[1], (lo[2] * lo[3] + hi[2] * hi[3]) / w, w])
    xs = np.array([v for b in stack for v in (b[0], b[1])])
    vs = np.array([b[2] for b in stack for _ in (0, 1)])
    fitted = np.interp(s, xs, vs)
    return fitted, np.interp(q, xs, vs)
'''

HINTS = [
    'Start with one block per distinct score; an isotonic fit can never separate equal inputs.',
    'Pool adjacent violators: whenever a block\'s mean is above the next block\'s, they must share one value, the weighted mean of both. Merging can cascade backwards, hence the while loop.',
    'Represent each block by its two end points with the same value; `np.interp` over those points gives flat steps inside blocks and linear joins between them.',
]

_r = np.random.default_rng(17)
_s = np.round(_r.random(40), 2)
_y = (_r.random(40) < _s ** 2).astype(float)
TESTS = [
    (np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.0, 0.25, 0.5])),
    (np.array([0.1, 0.2, 0.3]), np.array([1.0, 1.0, 0.0]), np.array([0.15, 0.3])),
    (np.array([0.5, 0.5, 0.1, 0.9]), np.array([1.0, 0.0, 0.0, 1.0]), np.array([0.3, 0.7, 1.0])),
    (np.array([0.2, 0.4, 0.6, 0.8]), np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.5])),
    (_s, _y, np.linspace(0, 1, 11)),
]

EXPLANATION = r"""
## Why isotonic
Platt scaling fits a sigmoid to scores, which assumes the miscalibration has that shape. Isotonic regression assumes only that higher scores should not mean lower probability, so it can fix any monotonic distortion. The price is more variance: with few calibration examples it overfits, producing many small steps.

## Why pooling is optimal
Among all non-decreasing functions, the least-squares fit must be constant on any run where the data violates order, and the best constant is the mean. Pool adjacent violators merges such runs until none remain, in $O(n)$ after sorting.

## Practical use
Fit on held-out data, never on the training set of the model being calibrated. Ranking is preserved except that ties are created inside blocks, which can slightly change AUC.

## Interview follow-ups
- When would you prefer Platt scaling or temperature scaling instead?
"""
