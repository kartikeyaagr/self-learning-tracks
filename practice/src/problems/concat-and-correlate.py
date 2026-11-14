import numpy as np

META = dict(title='Best lag by cross-correlation', tags=['Time Series', 'Probability and Statistics'], diff='Hard',
            runtime='numpy', lessons=['mlmaths:11.21', 'air-fanout-v1:03.06'],
            related=['tile-and-diff', 'sort-and-argsort', 'normalize-columns'])

STATEMENT = r"""
Two signals often move together with a delay: ad spend and sign-ups, upstream and downstream sensors. Find the delay by correlating one signal with shifted copies of the other.

Write `best_lag(x, y, max_lag)` that returns `(lags, r, best)`.

## Input
- `x`, `y`: 1-D float arrays of the same length `n`.
- `max_lag`: Python int with `0 <= max_lag <= n - 2`, so every lag has at least 2 overlapping pairs.

## Definition
For each lag $k \in \{-L, \dots, L\}$ (with $L$ = `max_lag`), pair `x[t]` with `y[t + k]` for every `t` where both indices lie in `[0, n)`. A positive `k` means `y` follows `x` by `k` steps. `r[k]` is the Pearson correlation of those pairs:

$$r_k = \frac{\sum (x_t - \bar x)(y_{t+k} - \bar y)}{\sqrt{\sum (x_t - \bar x)^2 \sum (y_{t+k} - \bar y)^2}}$$

with the means taken over the overlapping pairs only. If either side of the overlap is **constant** (all its values equal), define $r_k = 0.0$.

## Output
- `lags`: `int64` array `[-L, ..., L]`, shape `(2L+1,)`.
- `r`: `float64` array of the matching correlations.
- `best`: Python `int`, the lag with the largest `r`. Ties go to the smallest `|k|`, then to the negative lag.

Build all shifted copies of `y` at once (for example by concatenating NaN padding); a Python loop over the lags is acceptable but not needed.
"""

STARTER = '''import numpy as np

def best_lag(x, y, max_lag):
    """Return (lags (2L+1,), r (2L+1,), best int): Pearson r of x[t] with y[t+k] for k in -L..L."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def best_lag(x, y, max_lag):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n, L = len(x), max_lag
    lags = np.arange(-L, L + 1)
    pad = np.full(L, np.nan)
    # Row j of Y is y shifted by k = j - L: Y[j, t] = y[t + k], NaN where t + k is out of range.
    Y = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, y, pad]), n)
    valid = ~np.isnan(Y)
    X = np.broadcast_to(x, Y.shape)
    cnt = valid.sum(axis=1)

    def centred(A):                       # subtract each row's masked mean, zero outside the overlap
        mean = np.where(valid, A, 0.0).sum(axis=1) / cnt
        return np.where(valid, A - mean[:, None], 0.0)

    def constant(A):                      # exact test: max == min over the overlap
        return np.where(valid, A, -np.inf).max(axis=1) == np.where(valid, A, np.inf).min(axis=1)

    dx, dy = centred(X), centred(Y)
    den = np.sqrt((dx * dx).sum(axis=1) * (dy * dy).sum(axis=1))
    flat = constant(X) | constant(Y) | (den == 0)
    r = np.where(flat, 0.0, (dx * dy).sum(axis=1) / np.where(flat, 1.0, den))
    # Candidates ordered by (|k|, k); argmax returns the first maximum in that order.
    order = np.lexsort((lags, np.abs(lags)))
    best = int(lags[order[np.argmax(r[order])]])
    return lags, r, best
'''

HINTS = [
    'Pad `y` with `L` NaNs on each side using `np.concatenate`. Then every shift is a length-`n` window of the padded array, and NaN marks the positions that fall outside the overlap.',
    '`np.lib.stride_tricks.sliding_window_view(ypad, n)` returns all `2L+1` windows as rows (row `j` is lag `j - L`). With `valid = ~np.isnan(Y)`, compute masked means, centred values (0 outside the overlap) and the three sums for every lag at once.',
    'Detect constant overlaps exactly with masked max == masked min (a floating-point variance of a constant column may not be exactly 0). For the tie rule, order the candidates with `np.lexsort((lags, np.abs(lags)))` and take the first maximum.',
]

_r = np.random.default_rng(97)
_base = _r.normal(size=60)
_lagged = np.concatenate([_r.normal(size=4), _base[:-4]]) + 0.2 * _r.normal(size=60)
_lead = np.concatenate([_base[3:], _r.normal(size=3)])
TESTS = [
    (np.array([1.0, 3.0, 2.0, 5.0, 4.0]), np.array([0.0, 1.0, 3.0, 2.0, 5.0]), 2),
    (np.array([1.0, 4.0, 2.0, 0.0, 3.0]), np.array([2.0, 5.0, 1.0, 1.5, 0.0]), 1),
    (np.array([2.0, 2.0, 2.0, 2.0, 2.0]), np.array([1.0, 5.0, 2.0, 8.0, 3.0]), 2),
    (np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0]), np.array([0.1, 0.1, 0.1, 0.9, 0.2, 0.7]), 0),
    (np.array([1e6, -1e6, 3e6, 0.0, 2e6, -5e5]), np.array([5e-3, 1e-3, -2e-3, 4e-3, 3e-3, -1e-3]), 2),
    (_base, _lagged, 8),
    (_base, _lead, 6),
]

EXPLANATION = r"""
## Why it works
Padding `y` with `L` NaNs on both sides and sliding a length-`n` window across it produces every shifted copy at once: row `j` of `Y` is `y` delayed by `k = j - L`, with NaN where the shift runs off the end. The NaN mask is then the overlap mask for each lag, and the same mask applied to a broadcast copy of `x` selects the matching `x` values. Pearson's $r$ is computed with masked sums, all lags in parallel: centre each side by its overlap mean, then divide the cross-sum by the root of the two sums of squares.

Constant overlaps make $r$ undefined (0/0). Testing "max == min" is exact, whereas a computed variance of a constant segment can be a tiny non-zero number due to rounding in the mean, which would produce a meaningless $r$.

`np.lexsort` sorts by its **last** key first, so `(lags, |lags|)` orders by `|k|` and then by `k`. `argmax` returns the first maximum in that order, which implements the tie rule.

## Complexity
$O(nL)$ time and memory for the `(2L+1, n)` masked matrices. For long signals and many lags, compute the cross-correlation with an FFT in $O(n \log n)$ and normalise afterwards.

## Common bugs
- Using `np.roll` to shift. It wraps the end of `y` around to the start and correlates unrelated samples.
- Normalising with the full-series means and standard deviations instead of the overlap's, which biases large lags.
- Getting the sign convention backwards: here `best > 0` means `y` **follows** `x`.
- `np.correlate(x, y, 'full')` gives raw (unnormalised, uncentred) sums. It is useful, but not Pearson's $r$.

## Interview follow-ups
- A high cross-correlation at lag 3 between ad spend and sign-ups: does spend cause sign-ups? (Not necessarily. Both may follow a common driver such as seasonality, and autocorrelated series produce spurious cross-correlations; pre-whiten or difference first.)
- How does this relate to the convolution used in CNNs? (Deep-learning "convolution" layers actually compute cross-correlation: the kernel slides without flipping.)
"""
