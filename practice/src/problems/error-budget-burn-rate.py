META = dict(title='Multiwindow burn-rate alerts on an error budget', tags=['MLOps', 'Time Series', 'Metrics & Evaluation'],
            diff='Medium', runtime='python', lessons=['sdsd:14.02'],
            related=['consecutive-breach-alerts', 'delayed-label-monitoring', 'p50-p95-and-p99-inference-latency'])

STATEMENT = r"""
A service has an availability SLO such as 99.9% of requests succeed over a 30-day period. Its monitoring
system records, once per minute, `(good, total)`: the number of successful requests and of all requests in
that minute. Write `burn_rate_alerts(slo, minutes, period_minutes, rules)` that returns how much of the error
budget the trace used and which minutes each alert severity was firing.

## Definitions
- `slo` is a decimal fraction (`0.999`). The **error budget** fraction is $b = 1 - \text{slo}$. Treat `slo` and
  every threshold as the exact decimal it is written as (`fractions.Fraction(str(x))` does this), so that
  $1 - 0.999$ is exactly $0.001$.
- Minute `m` is `minutes[m]`, 0-based, and has `total - good` bad requests.
- The **burn rate** over a span of minutes is its error rate divided by the budget:
  $\text{burn} = \dfrac{\sum \text{bad} / \sum \text{total}}{b}$, and 0 when $\sum \text{total} = 0$.
  A burn rate of 1 spends exactly the whole budget by the end of the SLO period.
- The window of length `L` at minute `m` is minutes `m-L+1 .. m` (inclusive, ending now).

## Alerts
Each rule is `(severity, long, short, threshold)` with window lengths in minutes. The rule **fires** at minute
`m` iff `m >= long - 1` (the long window is full; earlier minutes are not evaluated) **and** the burn rate over
the long window is `> threshold` **and** the burn rate over the short window is `> threshold` (strictly).
Several rules can share a severity; that severity is firing at `m` if any of its rules fires.

The lesson's starting rules for a 30-day SLO are: page at burn 14.4 over 1 hour and 5 minutes, page at burn 6
over 6 hours and 30 minutes, ticket at burn 1 over 3 days and 6 hours. As tuples:
`("page", 60, 5, 14.4)`, `("page", 360, 30, 6)`, `("ticket", 4320, 360, 1)`.

## Return
`{"budget_consumed": c, "alerts": {severity: [minutes firing, ascending], ...}}` with one key per severity that
appears in `rules` (an empty list if it never fired). `c` is the share of the period's budget the trace used,
from the lesson's formula *budget consumed = burn rate x window / SLO period*, taking the whole trace as the
window: $c = \text{burn}(\text{all minutes}) \cdot n / \text{period\_minutes}$ for $n$ minutes (0 for an empty
trace), returned as a float (`0.05` means 5%).
"""

STARTER = '''def burn_rate_alerts(slo, minutes, period_minutes, rules):
    """Return {"budget_consumed": float, "alerts": {severity: [minute, ...]}}."""
    budget = 1 - slo
    rates = [(t - g) / t if t else 0.0 for g, t in minutes]
    consumed = sum(rates) / budget / period_minutes
    alerts = {sev: [] for sev, *_ in rules}
    for sev, long, short, thr in rules:
        for m in range(len(minutes)):
            window = rates[max(0, m - long + 1): m + 1]
            if sum(window) / len(window) / budget >= thr:
                alerts[sev].append(m)
    return {"budget_consumed": consumed, "alerts": alerts}
'''

SOLUTION = '''from fractions import Fraction
from itertools import accumulate

def burn_rate_alerts(slo, minutes, period_minutes, rules):
    budget = 1 - Fraction(str(slo))                       # exact: 1 - 0.999 == 1/1000
    bad = [0, *accumulate(t - g for g, t in minutes)]     # prefix sums: any window in O(1)
    tot = [0, *accumulate(t for _, t in minutes)]

    def burn(lo, hi):                                     # minutes lo .. hi-1
        n = tot[hi] - tot[lo]
        return Fraction(bad[hi] - bad[lo], n) / budget if n else Fraction(0)

    n = len(minutes)
    consumed = float(burn(0, n) * n / period_minutes) if n else 0.0
    firing = {sev: set() for sev, *_ in rules}
    for sev, long, short, thr in rules:
        thr = Fraction(str(thr))
        for m in range(long - 1, n):                      # only once the long window is full
            # long window: significant burn; short window: still burning now
            if burn(m - long + 1, m + 1) > thr and burn(m - short + 1, m + 1) > thr:
                firing[sev].add(m)
    return {"budget_consumed": consumed, "alerts": {sev: sorted(s) for sev, s in firing.items()}}
'''

HINTS = [
    'An error rate over a window is total bad over total requests in that window, not the average of per-minute rates: a minute with 1 request must not weigh as much as a minute with 1,000.',
    'Build prefix sums of bad and total requests once; then every window is two subtractions, and checking all rules at every minute stays linear in the trace length.',
    'Compare burn rates exactly with `Fraction`: `1 - 0.9` in floats is `0.09999999999999998`, which pushes a burn rate of exactly 6 to `6.000000000000001` and fires an alert that should not.',
]

DAY30 = 43200
RULES = [("page", 60, 5, 14.4), ("page", 360, 30, 6), ("ticket", 4320, 360, 1)]
FAST = [("page", 60, 5, 14.4)]
_sparse = [(1, 1) if m % 6 == 0 else (0, 0) for m in range(70)]
_sparse[65] = (0, 1)
TESTS = [
    (0.999, [(1000, 1000)] * 60 + [(800, 1000)] * 10 + [(1000, 1000)] * 70, DAY30, RULES),
    (0.999, [(992, 1000)] * 400, DAY30, RULES),
    (0.999, [(986, 1000)] * 55 + [(981, 1000)] * 4 + [(982, 1000), (985, 1000)], DAY30, FAST),
    (0.999, _sparse, DAY30, FAST),
    (0.99, [(100, 100)] * 8 + [(97, 100)] * 14 + [(70, 100)] * 2 + [(100, 100)] * 6, 1440,
     [("page", 10, 2, 10), ("ticket", 20, 5, 2)]),
    (0.9, [(40, 100)] * 399 + [(39, 100)], DAY30, RULES),
    (0.999, [], DAY30, FAST),
    (0.999, [(0, 0)] * 100, DAY30, FAST),
    (0.999, [(0, 1000)] * 90, DAY30, FAST),
]

EXPLANATION = r"""
## Why burn rate, and why two windows
Alerting on "error rate above 0.1%" pages for blips that barely touch the budget. Burn rate asks how fast the
budget is going: at 14.4 for an hour you have spent $14.4 \cdot 1/720 = 2\%$ of a 30-day budget, and at that pace
it is gone in about two days, which is worth waking someone for. The long window makes the alert
**significant**; the short window (a twelfth of it) makes it **current**, so the alert clears minutes after
the errors stop instead of an hour later.

Test 1 is the lesson's outage: 10 minutes at 20% errors (burn 200) after an hour of clean traffic. The 5-minute
window crosses 14.4 at once, but the 1-hour window only does after the fifth bad minute
($200 \cdot 5/60 \approx 16.7$), so the page fires at minute 64. It clears at minute 74, the first minute
whose 5-minute window holds no bad minute, even though the 1-hour window is still far above 14.4. The trace
used $2000/(0.001 \cdot 1000 \cdot 43200) \approx 4.6\%$ of the month's budget.

## What the other tests pin down
- **Slow burns** (test 2): 0.8% errors is burn 8, never enough for the fast page, but the 6-hour page fires
  from minute 359, the first minute its long window is full.
- **Strict thresholds and exact maths** (tests 3 and 6): a long window at exactly 14.4, or exactly 6, does not
  fire; one more bad request does. With floats, $1 - 0.9$ is slightly under $0.1$ and test 6 would page on a
  burn rate of exactly 6.
- **Low traffic** (test 4): one failure among 11 requests in an hour is a burn rate of about 91. The maths is
  right, the page is arguably wrong; this is the lesson's low-traffic problem (add synthetic traffic, merge
  small services, or lower the SLO).
- **Request weighting** (test 4 again): averaging per-minute error rates would treat empty minutes as 0% and
  a 1-request minute like a 1,000-request one.
- **Budget over 100%** (test 9): a total outage for 90 minutes uses about 208% of a 99.9% budget. It is a share,
  not a capped percentage.
- **Low SLOs cap the burn rate**: at 90% the largest possible burn is $1/0.1 = 10$, so the 14.4 page can never
  fire (test 6). Thresholds have to be chosen per SLO.

## Complexity
$O(n + n \cdot r)$ for $n$ minutes and $r$ rules, with $O(n)$ memory for the prefix sums. A streaming version
keeps a running sum per window instead.

## Interview follow-ups
- Why not just alert on the 30-day burn rate? (It reacts far too slowly: a full outage needs hours to move a
  30-day average.)
- How would you implement this in Prometheus? (Recording rules for `rate(errors[5m]) / rate(requests[5m])` per
  window, and one alert expression per row: `long > 14.4 * 0.001 and short > 14.4 * 0.001`.)
- When does the 3-day ticket matter? (A burn just above 1 never pages but would quietly use the whole budget;
  the ticket catches it during working hours.)
"""
