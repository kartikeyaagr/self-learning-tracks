META = dict(title='Notification scheduler with a daily budget and quiet hours', tags=['Recommender Systems', 'Optimization'], diff='Medium', runtime='python')

STATEMENT = r"""
Write `schedule(candidates, budget, quiet, min_gap, min_value)` for one user's push notifications over a day.

- Each candidate is `(id, hour, value)`: it may be sent at its `hour` (0 to 23) and has a predicted value (engagement gain minus unsubscribe risk, from a model).
- `quiet` is a list of hours when nothing is sent.
- Greedy selection: consider candidates by value descending (ties to the earlier hour, then smaller id). Accept one if its value is at least `min_value`, its hour is not quiet, fewer than `budget` have been accepted, and every accepted notification is at least `min_gap` hours away (`abs(h1 - h2) >= min_gap`).

Return the accepted ids sorted by hour, then id.
"""

STARTER = '''def schedule(candidates, budget, quiet, min_gap, min_value):
    """Return ids to send, in time order."""
    best = sorted(candidates, key=lambda c: -c[2])[:budget]
    return [c[0] for c in sorted(best, key=lambda c: c[1])]
'''

SOLUTION = '''def schedule(candidates, budget, quiet, min_gap, min_value):
    quiet, chosen = set(quiet), []
    for cid, h, v in sorted(candidates, key=lambda c: (-c[2], c[1], c[0])):
        if len(chosen) >= budget:
            break
        if v < min_value or h in quiet:
            continue
        if all(abs(h - h2) >= min_gap for _, h2 in chosen):
            chosen.append((cid, h))
    return [cid for cid, _ in sorted(chosen, key=lambda x: (x[1], x[0]))]
'''

HINTS = [
    'Filter by value and quiet hours before checking spacing, so ineligible items never block good ones.',
    'Spacing is checked against everything already accepted, not only the previous one in time.',
    'A min_value threshold lets the scheduler send fewer than budget when nothing is worth it.',
]

_C = [("a", 8, 0.9), ("b", 9, 0.8), ("c", 12, 0.5), ("d", 23, 0.95), ("e", 18, 0.4), ("f", 19, 0.45), ("g", 2, 0.99)]
TESTS = [
    (_C, 3, [22, 23, 0, 1, 2, 3, 4, 5, 6], 2, 0.1),
    (_C, 3, [], 2, 0.1),
    (_C, 10, [0, 1, 2, 3, 4, 5, 6, 22, 23], 4, 0.1),
    (_C, 2, [], 0, 0.85),
    ([], 3, [], 1, 0.0),
    ([("x", 10, 0.5), ("y", 10, 0.5), ("z", 11, 0.5)], 3, [], 1, 0.0),
]

EXPLANATION = r"""
## Why a budget at all
Each notification has a positive expected effect on engagement and a small chance of making the user disable notifications forever. That second cost compounds, so systems cap volume per user per day and require a minimum value, instead of sending everything with a positive score.

## Why spacing and quiet hours
Two pushes ten minutes apart are worth much less than the same two pushes hours apart, and a push at 3 a.m. is mostly an annoyance. Encoding these as hard constraints is simpler and safer than hoping the value model learns them.

## Greedy is a choice
Greedy by value can block two good notifications with one great one in between. Exact selection is a small interval-scheduling problem, but greedy is predictable, fast, and usually close.

## Interview follow-ups
- How would you learn each user's budget instead of fixing it globally?
"""
