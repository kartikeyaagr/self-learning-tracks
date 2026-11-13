META = dict(title='Page-Hinkley test for streaming drift', tags=['MLOps', 'Time Series'], diff='Medium', runtime='python',
            lessons=['air-fanout-v1:11.01'], related=['consecutive-breach-alerts', 'two-sample-ks-drift'])

STATEMENT = r"""
Batch drift tests compare two fixed windows. A streaming detector instead watches one value at a time (say each request's loss) and raises an alarm soon after the mean shifts. The **Page-Hinkley** test accumulates how far values sit above their running mean, and alarms when that sum climbs far above its lowest point so far.

Write `page_hinkley(stream, delta=0.005, lam=5.0, min_samples=30, direction='up')` that returns the list of alarm indices.

## The detector
State: $n = 0$, $\bar x = 0$, $S = 0$, $S_{\min} = 0$. For each value $x_t$ at index $t$, in this order:

1. $n \leftarrow n + 1$ and $\bar x \leftarrow \bar x + (x_t - \bar x)/n$ (incremental mean, **including** $x_t$).
2. $S \leftarrow S + s\,(x_t - \bar x) - \delta$, where $s = +1$ for `direction='up'` (detect an increase) and $s = -1$ for `'down'`.
3. $S_{\min} \leftarrow \min(S_{\min}, S)$.
4. If $n \ge$ `min_samples` and $S - S_{\min} >$ `lam`: record $t$ as an alarm and **reset** the state to $n = 0$, $\bar x = 0$, $S = 0$, $S_{\min} = 0$ before the next value.

## Notes
- `stream` is a list of floats (possibly empty). Return a list of ints in increasing order.
- `delta` is the magnitude of change you're willing to ignore; `lam` is the alarm threshold. Use the incremental mean exactly as written so results match to the last bit.
"""

STARTER = '''def page_hinkley(stream, delta=0.005, lam=5.0, min_samples=30, direction='up'):
    """Return the indices at which the detector raises an alarm."""
    alarms, n, mean, S = [], 0, 0.0, 0.0
    for t, x in enumerate(stream):
        n += 1
        mean += (x - mean) / n
        S += x - mean - delta
        if n >= min_samples and S > lam:          # compares S with 0, not with its running minimum
            alarms.append(t)
    return alarms
'''

SOLUTION = '''def page_hinkley(stream, delta=0.005, lam=5.0, min_samples=30, direction='up'):
    sign = 1.0 if direction == 'up' else -1.0
    alarms = []
    n, mean, S, S_min = 0, 0.0, 0.0, 0.0
    for t, x in enumerate(stream):
        n += 1
        mean += (x - mean) / n                    # running mean includes the current value
        S += sign * (x - mean) - delta            # drifts down while the stream is stable
        S_min = min(S_min, S)
        if n >= min_samples and S - S_min > lam:  # rise above the lowest point so far
            alarms.append(t)
            n, mean, S, S_min = 0, 0.0, 0.0, 0.0  # start learning the new regime from scratch
    return alarms
'''

HINTS = [
    'Keep four numbers: the count, the running mean, the cumulative sum $S$ and its minimum $S_{\\min}$. Update the mean **before** adding $x_t - \\bar x$ to $S$.',
    'The alarm statistic is $S - S_{\\min}$, the rise above the lowest point, not $S$ itself. Subtracting $\\delta$ every step makes $S$ drift downwards while nothing changes.',
    'For `direction="down"` multiply the deviation by -1 (the $\\delta$ term is still subtracted). After an alarm, reset all four state variables, including the sample count, so `min_samples` applies again.',
]

_rng = __import__('random').Random(17)
_shift = [round(_rng.gauss(0.2, 0.05), 4) for _ in range(150)] + [round(_rng.gauss(0.5, 0.05), 4) for _ in range(150)]
_noisy = [round(_rng.gauss(1.0, 0.3), 4) for _ in range(400)]

TESTS = [
    ([0.0] * 5 + [3.0] * 5, 0.0, 4.0, 3),
    ([1.0] * 5 + [0.0] * 5, 0.0, 2.0, 3, 'down'),
    ([1.0] * 40, 0.005, 1.0, 30),
    ([0.0] * 3 + [10.0] * 3 + [0.0] * 3 + [10.0] * 3, 0.0, 4.0, 2),
    ([], 0.005, 5.0, 30),
    ([0.0] * 10 + [2.0] * 10, 0.1, 3.0, 15),
    (_shift, 0.01, 2.0, 30),
    (_shift, 0.01, 2.0, 30, 'down'),
    (_noisy, 0.05, 8.0, 30),
]

EXPLANATION = r"""
## Why it works
While the stream is stable, $x_t - \bar x$ averages to zero and the $-\delta$ term pulls $S$ steadily down, so $S$ keeps setting new minima and $S - S_{\min}$ stays near zero. After an upward shift, $x_t$ sits above the (slowly adapting) mean, each step adds roughly (shift $- \delta$), and $S$ climbs away from its minimum until it crosses $\lambda$. It's a CUSUM test where the pre-change mean is estimated on the fly.

$\delta$ sets the smallest shift you care about, and $\lambda$ trades detection delay for false alarms: a larger $\lambda$ means fewer false alarms and a slower reaction. Resetting after an alarm lets the detector learn the new regime, so a second shift can be caught too (test 4). `min_samples` stops it firing on the first noisy values after a reset.

## Complexity
$O(1)$ time and memory per value: it can sit inside a streaming job next to the model.

## Common bugs
- Thresholding $S$ instead of $S - S_{\min}$ (the starter). With a long stable prefix $S$ is very negative, so a real shift takes far too long to show.
- Updating the mean after the $S$ update, which changes every number that follows.
- Flipping the sign of $\delta$ for `'down'`: the drift term always pulls the statistic away from an alarm (test 2).
- Not resetting $n$, so `min_samples` no longer protects the start of the new regime.

## Interview follow-ups
- Your loss stream needs labels, which arrive a day late. What could you run Page-Hinkley on instead to get an earlier signal? (Prediction confidence, the score distribution, or input feature statistics.)
- How would you pick $\lambda$ for a target of one false alarm per month on a stream of 1M values per day? (Simulate or replay stable historical data and choose the smallest $\lambda$ that meets the false-alarm rate.)
"""
