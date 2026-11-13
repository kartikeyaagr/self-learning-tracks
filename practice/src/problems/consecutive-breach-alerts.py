META = dict(title='Alert incidents from consecutive threshold breaches', tags=['MLOps', 'Time Series'], diff='Easy', runtime='python',
            lessons=['ieng:24.01', 'sdsd:17.04'], related=['page-hinkley-drift', 'delayed-label-monitoring'])

STATEMENT = r"""
A monitor scrapes a model metric (say the error rate) once per minute. Paging on every single point above the threshold is too noisy, so the alert needs `k` **consecutive** breaches to fire and `k` consecutive healthy points to resolve.

Write `alert_incidents(values, threshold, k)` that returns the list of incidents as tuples `(fired_at, resolved_at)`.

## Rules
- `values`: a list whose items are floats or `None` (a missed scrape). A point is a **breach** if `value > threshold` (strictly) and **healthy** if `value <= threshold`.
- While no incident is open: the incident fires at the index of the `k`-th consecutive breach. `fired_at` is that index.
- While an incident is open: it resolves at the index of the `k`-th consecutive healthy point. `resolved_at` is that index. Breaches while open reset the healthy streak.
- A `None` breaks every streak: after it, counting starts again from zero. It doesn't open or close anything by itself.
- After an incident resolves, the breach count starts from zero; points before the resolution don't count towards the next incident.
- An incident still open at the end has `resolved_at = None`.
- `k >= 1`. Return incidents in the order they fired, as a list of 2-tuples.
"""

STARTER = '''def alert_incidents(values, threshold, k):
    """Return [(fired_at, resolved_at_or_None), ...]."""
    incidents, open_at = [], None
    for i, v in enumerate(values):
        if v is not None and v > threshold and open_at is None:
            open_at = i                                   # fires on the first breach
        elif v is not None and v <= threshold and open_at is not None:
            incidents.append((open_at, i))
            open_at = None
    if open_at is not None:
        incidents.append((open_at, None))
    return incidents
'''

SOLUTION = '''def alert_incidents(values, threshold, k):
    incidents = []
    open_at = None            # index the open incident fired at, or None
    streak = 0                # consecutive breaches (closed) or consecutive healthy points (open)
    for i, v in enumerate(values):
        if v is None:         # a missed scrape breaks any streak
            streak = 0
            continue
        breach = v > threshold
        if open_at is None:
            streak = streak + 1 if breach else 0
            if streak == k:
                open_at, streak = i, 0
        else:
            streak = 0 if breach else streak + 1
            if streak == k:
                incidents.append((open_at, i))
                open_at, streak = None, 0
    if open_at is not None:
        incidents.append((open_at, None))
    return incidents
'''

HINTS = [
    'The monitor is a two-state machine: *closed* (counting consecutive breaches) and *open* (counting consecutive healthy points). One counter is enough because only one streak matters in each state.',
    'Reset the counter on every state change, and whenever the opposite kind of point arrives. `None` resets it too.',
    'When the counter reaches `k` in the closed state, remember `i` as `fired_at`. When it reaches `k` in the open state, append `(fired_at, i)`. After the loop, an incident that is still open gets `None`.',
]

TESTS = [
    ([0.1, 0.5, 0.6, 0.7, 0.2, 0.1, 0.1], 0.4, 2),
    ([0.5, 0.1, 0.5, 0.1, 0.5], 0.4, 2),
    ([0.5, 0.5, None, 0.5, 0.5, 0.1, None, 0.1, 0.1], 0.4, 2),
    ([0.9, 0.9, 0.9], 0.4, 1),
    ([0.4, 0.4, 0.4], 0.4, 1),
    ([], 0.4, 3),
    ([0.5, 0.5, 0.5, 0.1, 0.5, 0.1, 0.1, 0.1, 0.6, 0.6, 0.6], 0.4, 3),
    ([1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0], 0.5, 2),
]

EXPLANATION = r"""
## Why it works
Requiring `k` consecutive breaches before firing and `k` consecutive healthy points before resolving gives **hysteresis**: a metric that hovers around the threshold (test 2) produces no incident at all instead of a page every other minute. Each point is handled in $O(1)$ by a two-state machine with one counter, which is exactly how alert rules with a `for: 5m` clause behave in Prometheus-style systems.

Treating `None` as a streak breaker is the conservative choice: a gap in the data shouldn't make an incident look continuous. Some systems instead treat missing data as a breach (fail closed). Which you pick is a product decision; the code only has to state it.

## Complexity
$O(n)$ time, $O(1)$ extra memory besides the output, so it can run on a live stream.

## Common bugs
- Reporting the index of the *first* breach of the streak as `fired_at`: the alert can't know it's an incident until the `k`-th point.
- Not resetting the healthy counter when a breach arrives while open (test 7).
- Using `>=` for a breach: a value exactly at the threshold is healthy (test 5).
- Letting breaches that happened before a resolution count towards the next incident.

## Interview follow-ups
- Your error-rate alert is flapping during low-traffic hours. What would you change besides `k`? (Require a minimum request count per point, or alert on a rate computed over a longer window.)
- How would you turn this into multi-window burn-rate alerting for an SLO? (Fire when both a short and a long window exceed their burn-rate thresholds.)
"""
