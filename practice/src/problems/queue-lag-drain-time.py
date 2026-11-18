META = dict(title='Consumer lag, drain time and consumers needed to catch up', tags=['Distributed Computing', 'MLOps'],
            diff='Medium', runtime='python', lessons=['sdsd:08.04'],
            related=['moderation-review-queue', 'retry-backoff-schedule', 'idempotency-key-store'])

STATEMENT = r"""
Write `queue_lag(backlog, produced, per_consumer, partitions, consumers, deadline)` for a Kafka-style topic and
report how its consumer lag evolves. All quantities are integers.

- `backlog`: messages already waiting at the start.
- `produced[i]`: messages produced during interval `i`.
- `per_consumer`: messages one consumer processes per interval.
- `partitions`: the topic's partition count. Each partition is read by one consumer of the group, so at most
  `partitions` consumers do any work; extra ones sit idle.
- `consumers`: scaling events `[[start_interval, count], ...]`, sorted by start, the first starting at 0. From
  interval `start_interval` on, the group has `count` consumers.

In interval `i`, capacity is $\min(\text{count}, \text{partitions}) \cdot \text{per\_consumer}$ and the lag
becomes $\max(0,\ \text{lag} + \text{produced}[i] - \text{capacity})$: consumers cannot process messages that do
not exist yet.

Return a dict with:
- `"lag"`: the lag after each interval,
- `"peak"`: the largest lag, counting the starting `backlog`,
- `"drained"`: the smallest `k` (from 0 to `len(produced)`) such that the lag is 0 after interval `k` and after
  every later interval (`k = 0` means the starting backlog, too), or `None` if the lag after the last interval is
  above 0,
- `"min_consumers"`: the smallest constant consumer count `c` (from 0 to `partitions`) such that, with `c`
  consumers in every interval, the lag after the first `deadline` intervals is 0; `None` if even `partitions`
  consumers cannot do it. `1 <= deadline <= len(produced)`.
"""

STARTER = '''def queue_lag(backlog, produced, per_consumer, partitions, consumers, deadline):
    """Lag per interval, peak, drain time and consumers needed."""
    capacity = consumers[0][1] * per_consumer
    lag, lags = backlog, []
    for p in produced:
        lag += p - capacity
        lags.append(lag)
    drained = next((i + 1 for i, x in enumerate(lags) if x <= 0), None)
    rate = sum(produced) // len(produced)
    need = -(-(backlog + rate * deadline) // (per_consumer * deadline))
    return {"lag": lags, "peak": max([backlog] + lags), "drained": drained, "min_consumers": need}
'''

SOLUTION = '''def queue_lag(backlog, produced, per_consumer, partitions, consumers, deadline):
    def run(schedule):
        events, count, lag, out = dict(schedule), 0, backlog, []
        for i, p in enumerate(produced):
            count = events.get(i, count)
            capacity = min(count, partitions) * per_consumer      # idle consumers beyond the partition count
            lag = max(0, lag + p - capacity)
            out.append(lag)
        return out

    lags = run(consumers)
    series = [backlog] + lags                  # series[k] = lag after k intervals
    k = len(series)
    while k > 0 and series[k - 1] == 0:        # walk back over the trailing zeros
        k -= 1
    drained = None if k == len(series) else k
    # lag is non-increasing in c, so the first c that works is the minimum
    need = next((c for c in range(partitions + 1) if run([[0, c]])[deadline - 1] == 0), None)
    return {"lag": lags, "peak": max(series), "drained": drained, "min_consumers": need}
'''

HINTS = [
    'Simulate interval by interval and clamp at zero. A running sum of `produced - capacity` goes negative and then hides the next backlog.',
    'Effective consumers are `min(count, partitions)`: with 10 partitions an 11th consumer adds nothing.',
    '"Drained" means it stays drained: find the last interval with lag above 0 and answer the one after it.',
    'For `min_consumers`, reuse the same simulation with a single scaling event `[[0, c]]` and try `c = 0, 1, 2, ...`.',
]

TESTS = [
    (3600, [1000] * 20, 100, 20, [[0, 12]], 4),
    (3600, [1000] * 20, 100, 20, [[0, 10]], 4),
    (3600, [1000] * 6, 100, 20, [[0, 10], [2, 20]], 6),
    (0, [60] * 10, 5, 30, [[0, 10], [4, 18]], 10),
    (500, [100] * 5, 20, 10, [[0, 15]], 5),
    (200, [0, 300, 0, 0, 250, 0], 50, 4, [[0, 4]], 6),
    (0, [0, 0, 0], 10, 3, [[0, 1]], 3),
    (50, [10, 10, 10], 100, 8, [[0, 0], [1, 2]], 1),
    (7, [3, 3, 3, 3], 2, 3, [[0, 3]], 2),
]

EXPLANATION = r"""
## The arithmetic
With arrival rate $\lambda$ and processing rate $\mu$, lag grows by $\lambda - \mu$ per interval while
$\lambda > \mu$, and a backlog $B$ drains in $B / (\mu - \lambda)$ once $\mu > \lambda$. Test 1 is the lesson's
outage: 3,600 waiting, 1,000 arriving and 1,200 processed per interval, so it drains at 200 per interval and takes
$3600 / 200 = 18$ intervals. Test 2 has exactly enough capacity: $\mu - \lambda = 0$, the backlog never shrinks and
`drained` is `None`. Small headroom means very long recoveries.

## Why clamp at zero
Spare capacity cannot be banked: an idle consumer does not pre-process messages that have not arrived. The starter
lets lag go negative, so a quiet interval hides the spike after it (test 6).

## The partition cap
In Kafka each partition is read by one consumer in a group, so consumers beyond the partition count are idle.
Test 5 runs 15 consumers on 10 partitions and gets the throughput of 10. When `min_consumers` is `None`, the fix
is more partitions (or cheaper processing per message), not more consumers.

## Consumers needed by a deadline
With constant rates the closed form is $c = \lceil (B + \lambda D) / (\text{per\_consumer} \cdot D) \rceil$: the
backlog plus everything that arrives during the $D$ intervals, spread over them. Test 1 needs
$\lceil (3600 + 4000) / 400 \rceil = 19$ of its 20 partitions. The simulation gives the same answer and also handles
bursty production, where an average (as in the starter) is wrong.

## What to alert on
Lag in messages needs translating: 3,600 behind is a few seconds at 1,000 a second and an hour at 1 a second.
Alert on the age of the oldest message, roughly lag divided by consumption rate, which is what users feel.

## Interview follow-ups
- Lag is growing at 1,000 a second. What do you check first? (Which rate changed: produce rate up, or consume
  rate down from a slow dependency or a hot partition.)
- The drain will take five hours. How do you protect fresh traffic meanwhile? (Process new messages first and move
  the backlog to a separate queue, expire stale work, apply backpressure to producers, and make sure the database
  the consumers write to can take the extra load.)
"""
