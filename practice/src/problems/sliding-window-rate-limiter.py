META = dict(title='Sliding window counter vs sliding log rate limiter', tags=['Distributed Computing', 'Time Series'],
            diff='Medium', runtime='python', lessons=['sdsd:14.12', 'sdsd:14.11'],
            related=['retry-backoff-schedule', 'consecutive-breach-alerts'])

STATEMENT = r"""
An API gateway limits each client to `limit` requests per `window` seconds. Write
`rate_limit(requests, limit, window)` that replays a trace through two limiters side by side and returns
`{"counter": [...], "log": [...]}`: one list of booleans per limiter, `True` for *allowed* and `False` for
*rejected* (HTTP 429), in request order.

`requests` is a list of `(t, key)` pairs: `t` is an integer timestamp in seconds (non-decreasing; several
requests can share a `t` and are handled in list order) and `key` is the client (user ID, API key or IP).
Every key is limited on its own. Only **allowed** requests are counted; a rejected request leaves the state
unchanged. Both limiters reject when the count would reach the limit, so the comparison is strict (`< limit`
allows).

## 1. Sliding window counter (the approximation)
Windows are aligned to multiples of `window`: request `t` falls in window $w = \lfloor t / \text{window} \rfloor$,
and $\text{elapsed} = t - w \cdot \text{window}$ seconds into it. Per key, keep the allowed count of the current
window `curr` and of the window just before it `prev` (0 if the key had no allowed requests in window $w-1$,
including when its last activity was two or more windows ago). Allow the request iff

$$\text{prev} \cdot \frac{\text{window} - \text{elapsed}}{\text{window}} + \text{curr} < \text{limit}$$

and on allow, increment `curr`. Evaluate this **exactly**: no rounding, so an estimate of exactly `limit` is
rejected (multiply through by `window` to stay in integers).

## 2. Sliding window log (exact)
Per key, keep the timestamps of allowed requests. A request at `t` is allowed iff fewer than `limit` stored
timestamps `s` satisfy $t - \text{window} < s \le t$; on allow, store `t`. A request exactly `window` seconds
old has left the window.

Example: limit 50, window 60. A key had 42 allowed requests in window 0 and 18 so far in window 1. At $t = 75$
the counter estimates $42 \cdot 45/60 + 18 = 49.5 < 50$, so it allows the request.
"""

STARTER = '''def rate_limit(requests, limit, window):
    """Return {"counter": [bool, ...], "log": [bool, ...]}."""
    counts, out = {}, []
    for t, key in requests:
        slot = (key, t // window)              # one counter per calendar window
        if counts.get(slot, 0) < limit:
            counts[slot] = counts.get(slot, 0) + 1
            out.append(True)
        else:
            out.append(False)
    return {"counter": out, "log": list(out)}
'''

SOLUTION = '''from collections import deque

def rate_limit(requests, limit, window):
    state = {}      # key -> [window index, curr, prev]  (sliding window counter)
    logs = {}       # key -> deque of allowed timestamps  (sliding log)
    counter, log = [], []
    for t, key in requests:
        w, elapsed = divmod(t, window)
        s = state.setdefault(key, [w, 0, 0])
        if s[0] != w:                                   # rolled into a new window
            s[2] = s[1] if s[0] == w - 1 else 0         # a gap of 2+ windows leaves nothing to carry
            s[0], s[1] = w, 0
        # prev * (window - elapsed) / window + curr < limit, multiplied by window: exact integers
        ok = s[2] * (window - elapsed) + s[1] * window < limit * window
        if ok:
            s[1] += 1
        counter.append(ok)

        q = logs.setdefault(key, deque())
        while q and q[0] <= t - window:                 # evict timestamps that left (t - window, t]
            q.popleft()
        ok = len(q) < limit
        if ok:
            q.append(t)
        log.append(ok)
    return {"counter": counter, "log": log}
'''

HINTS = [
    'Use `divmod(t, window)` to get the window index and the seconds elapsed in it. When a key moves to a new window, `curr` becomes `prev` only if the new window is the very next one; after a longer gap both are 0.',
    'Compare `prev * (window - elapsed) + curr * window < limit * window`. It is the same test as the lesson formula, multiplied by `window`, so it never rounds.',
    'For the log, keep a `deque` of allowed timestamps per key and pop from the left while the oldest is `<= t - window`, then compare its length with `limit`.',
]

_steady = [(t, "k") for t in range(0, 42, 2)]
TESTS = [
    ([(59, "u")] * 5 + [(60, "u")] * 5 + [(90, "u")] * 3, 5, 60),
    ([(t, "a") for t in range(42)] + [(74, "a")] * 18 + [(75, "a")] + [(80, "a")] * 5, 50, 60),
    ([(0, "a"), (0, "b"), (1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "a")], 2, 10),
    ([(5, "x"), (5, "x"), (25, "x"), (25, "x"), (25, "x")], 2, 10),
    ([(0, "x")] * 3 + [(10, "x"), (15, "x"), (15, "x"), (19, "x"), (20, "x")], 3, 10),
    ([], 10, 60),
    (_steady, 4, 10),
    ([(3, "z"), (6, "z"), (7, "z"), (9, "z"), (13, "z"), (14, "z")], 3, 7),
    ([(0, "solo"), (1, "solo"), (100, "solo"), (100, "solo"), (199, "solo"), (200, "solo")], 1, 100),
]

EXPLANATION = r"""
## Why the counter works
The rolling window ending at `t` covers the last `window - elapsed` seconds of the previous window plus the
`elapsed` seconds of this one. The counter knows how many requests the previous window allowed but not when,
so it assumes they were spread evenly and counts the overlapping fraction. Two integers per key, one update
per request: that is why large edge networks use it.

Test 2 replays the lesson's worked example with $t = 0$ at 12:00:00: 18 requests at 12:01:14 fit, the 19th at 12:01:15 sees an
estimate of 49.5 and is allowed, and at 12:01:20 the estimate has fallen to 47, freeing three more requests
before the fourth hits exactly 50 and is rejected.

## Where the two disagree
The even-spread assumption is the only source of error, and it cuts both ways:
- Test 1 is the boundary attack: 5 requests at 0:59 and 5 at 1:00. The counter rejects the second burst
  (estimate $5 \cdot 60/60 = 5$), fixing the fixed window's 2x leak. At 1:30 it estimates 2.5 and lets 3 more
  through, but the log rejects them because the 5 requests from 0:59 are still inside the last 60 s. Old
  requests bunched at the end of the previous window make the counter too lenient.
- Test 5 is the opposite: 3 requests at second 0 have truly left the window by $t = 10$, so the log allows
  the next request, while the counter still sees $3 \cdot 10/10 = 3$ and rejects it. Bunched at the start, the
  counter is too strict.

Both directions are bounded by `prev` times the overlap fraction, and on real traffic the errors are rare.

## Edge cases the tests pin down
- An estimate of exactly `limit` is rejected (tests 2 and 5). Doing the comparison in integers means this does
  not depend on float rounding; `(window - elapsed) / window * prev` in floats can land just under an integer.
- After a gap of two or more windows, `prev` is 0, not the last window the key was seen in (test 4).
- The log's boundary is open on the old side: a request exactly `window` seconds old no longer counts (tests 5
  and 9).
- Rejected requests are not counted, so a client hammering the limiter does not lock itself out forever. Some
  production limiters count them on purpose to punish abusers; say which you chose.

## Complexity
Counter: $O(1)$ time and memory per key. Log: amortised $O(1)$ time per request but up to `limit` timestamps
per key, which is what makes it expensive at high limits.

## Interview follow-ups
- Many gateway servers share the limit. Where does the state live? (A shared store such as Redis, with the
  read-check-increment done atomically in a Lua script; or per-PoP counters, accepting a limit per PoP.)
- The store is down. Fail open or fail closed? (Usually open for ordinary API traffic, closed for login and
  other abuse-sensitive endpoints.)
- What do you send back with the 429? (`Retry-After`, ideally with jitter, so rejected clients do not all
  retry at the same second.)
"""
