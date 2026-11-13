META = dict(title='Dynamic batching with a size cap and a wait timeout', tags=['Inference'], diff='Medium', runtime='python',
            lessons=['ieng:07.01', 'ieng:04.02'],
            related=['simulate-continuous-batching', 'p50-p95-and-p99-inference-latency', 'optimize-throughput-and-cost-under-latency-slo'])

STATEMENT = r"""
Request-level (dynamic) batching is how most non-LLM model servers trade latency for throughput: hold incoming requests for a short time, then run them as one batch. Two knobs control it: the **maximum batch size** and the **maximum wait**, measured from the oldest queued request.

Write `dynamic_batches(arrivals, max_batch, max_wait, base_ms, per_item_ms)` that simulates one worker.

## Inputs
- `arrivals`: non-empty, non-decreasing list of arrival times in ms (floats or ints). Request $i$ arrives at `arrivals[i]`.
- `max_batch` $\ge 1$; `max_wait` $\ge 0$ (ms).
- Running a batch of $n$ requests takes `base_ms + per_item_ms * n` ms.

## Rules
Let $i$ be the oldest request not yet served and `free` the time the worker becomes idle (0 at the start).
1. The batch becomes **ready** at $r = \min(\text{arrivals}[i] + \text{max\_wait},\ \text{arrivals}[i + \text{max\_batch} - 1])$, where the second term exists only if that request exists (the moment the batch would be full).
2. It is **dispatched** at $d = \max(\text{free}, r)$.
3. It contains requests $i, i+1, \dots$ that arrived at or before $d$, at most `max_batch` of them.
4. All of them finish at $d + \text{base\_ms} + \text{per\_item\_ms} \cdot n$, which becomes the new `free`.

## Output
A tuple `(batches, latencies)`:
- `batches`: list of `[dispatch_time, size]` in order.
- `latencies`: list with each request's finish time minus its arrival time, in request order.
"""

STARTER = '''def dynamic_batches(arrivals, max_batch, max_wait, base_ms, per_item_ms):
    """Return (batches: [[dispatch_time, size], ...], latencies: [per request])."""
    batches, lat, free = [], [], 0
    for a in arrivals:                     # no batching: one request at a time
        d = max(free, a)
        free = d + base_ms + per_item_ms
        batches.append([d, 1])
        lat.append(free - a)
    return batches, lat
'''

SOLUTION = '''def dynamic_batches(arrivals, max_batch, max_wait, base_ms, per_item_ms):
    n = len(arrivals)
    batches, lat = [], [0] * n
    i, free = 0, 0
    while i < n:
        ready = arrivals[i] + max_wait                       # the oldest request's deadline
        if i + max_batch - 1 < n:
            ready = min(ready, arrivals[i + max_batch - 1])  # or the moment the batch is full
        d = max(free, ready)                                 # a busy worker delays dispatch (more may join)

        j = i
        while j < n and j - i < max_batch and arrivals[j] <= d:
            j += 1
        size = j - i
        free = d + base_ms + per_item_ms * size
        batches.append([d, size])
        for k in range(i, j):
            lat[k] = free - arrivals[k]
        i = j
    return batches, lat
'''

HINTS = [
    'Simulate batch by batch, not time step by time step: from the oldest waiting request you can compute exactly when its batch is ready.',
    'Ready time = min(oldest arrival + max_wait, arrival of the request that would fill the batch). Dispatch = max(ready, worker free time).',
    'Collect requests while they arrived by the dispatch time and the batch is not full. Request `i` always qualifies, because it arrived before its own deadline.',
]

TESTS = [
    ([0, 1, 2, 3], 2, 10, 5, 1),
    ([0, 3, 20], 4, 5, 2, 1),
    ([0, 0, 0, 0, 0], 2, 100, 10, 0),
    ([0, 1, 2], 1, 50, 3, 2),
    ([0, 10, 20, 30], 8, 0, 1, 1),
    ([0.0, 0.5, 1.0, 50.0, 50.2, 50.4, 50.6, 50.8, 51.0], 4, 2.0, 3.0, 0.5),
    ([(i * 37) % 11 + i * 4 for i in range(40)], 6, 8, 6, 1.5),
    ([0, 1, 2, 3, 4, 5, 6, 7], 3, 1, 20, 0),
]

EXPLANATION = r"""
## Why it works
Under these rules the next batch is fully determined by the oldest waiting request and the worker's state, so the simulation can jump from batch to batch. A batch leaves as soon as it is full or its oldest member has waited `max_wait`; if the worker is still busy at that moment, the batch waits and can only grow (up to the cap), which is how dynamic batching adapts to load: light traffic gives small, fast batches, heavy traffic gives full ones.

## Trade-off
Throughput per batch is $n / (\text{base} + \text{per\_item} \cdot n)$, which rises with $n$ when `base` (weight loading, kernel launches) dominates. Latency pays up to `max_wait` of queueing at low load. Test 8 shows the overload case: requests arrive faster than the worker serves them, batches fill to the cap, and latency grows without bound as the queue builds.

## Common bugs
- Measuring the wait from the most recent arrival instead of the oldest: a steady trickle of requests then postpones dispatch forever.
- Ignoring `free`: dispatching a batch while the worker is still busy, which double-books it.
- Forgetting the size cap when the worker was busy and many requests piled up.

## Interview follow-ups
- Why is request-level batching a bad fit for LLM generation? (Requests in a batch produce different numbers of tokens; short ones wait for the longest, and new requests wait for the whole batch. Continuous batching schedules per decode step instead.)
- How would you pick `max_wait`? (From the latency budget: P99 target minus P99 service time, then tune `max_batch` to the largest size that keeps service time within it.)
"""
