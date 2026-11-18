META = dict(title='Retry an LLM API call with backoff, jitter, Retry-After and a deadline', tags=['Inference', 'MLOps'],
            diff='Medium', runtime='python', lessons=['aieng:12.06', 'sdsd:02.09'],
            related=['latency-budget-allocation', 'trace-critical-path', 'dynamic-request-batching'])

STATEMENT = r"""
Write `retry(responses, base, cap, max_attempts, deadline, jitter, seed)` that simulates a client calling an LLM
API and returns what happened.

`responses[k]` is what attempt `k` (0-based) gets back: a dict with `status` (an HTTP code), `latency` (seconds
the attempt took) and optionally `retry_after` (seconds, from the `Retry-After` header). If there are more
attempts than entries, the last entry repeats.

Simulate with a clock `t` starting at 0:
1. Attempt `k` starts at `t` and ends at `t + latency`; move `t` there.
2. Status 200: stop with outcome `"ok"`.
3. Retryable statuses are 408, 429, 500, 502, 503, 504 and 529. Any other status: stop with `"fatal"`.
4. If `k + 1 == max_attempts`: stop with `"exhausted"`.
5. Otherwise compute the delay. The exponential ceiling is $e = \min(\text{cap}, \text{base}\cdot 2^k)$. With
   `rng = random.Random(seed)` created once per call and `u = rng.random()` drawn once per delay:
   - `"none"`: $d = e$ (draw nothing),
   - `"full"`: $d = e \cdot u$,
   - `"equal"`: $d = e/2 + (e/2)\, u$.

   If the response has `retry_after`, use $d = \max(d, \text{retry\_after})$.
6. If `t + d >= deadline`, stop with `"deadline"` (no point starting an attempt that cannot finish in time).
   Otherwise sleep: append `d` to the delays, `t += d`, and go to the next attempt.

Return `{"outcome": ..., "attempts": number of attempts made, "delays": [...], "elapsed": t}`.
"""

STARTER = '''import random

def retry(responses, base, cap, max_attempts, deadline, jitter, seed):
    """Simulate retries with exponential backoff."""
    t, delays = 0.0, []
    for k in range(max_attempts):
        r = responses[min(k, len(responses) - 1)]
        t += r["latency"]
        if r["status"] == 200:
            return {"outcome": "ok", "attempts": k + 1, "delays": delays, "elapsed": t}
        delays.append(base * 2 ** k)
        t += base * 2 ** k
    return {"outcome": "exhausted", "attempts": max_attempts, "delays": delays, "elapsed": t}
'''

SOLUTION = '''import random

RETRYABLE = {408, 429, 500, 502, 503, 504, 529}

def retry(responses, base, cap, max_attempts, deadline, jitter, seed):
    rng = random.Random(seed)
    t, delays, k = 0.0, [], 0
    while True:
        r = responses[min(k, len(responses) - 1)]
        t += r["latency"]
        done = lambda outcome: {"outcome": outcome, "attempts": k + 1, "delays": delays, "elapsed": t}
        if r["status"] == 200:
            return done("ok")
        if r["status"] not in RETRYABLE:
            return done("fatal")                 # a bad request stays bad: do not retry
        if k + 1 == max_attempts:
            return done("exhausted")
        e = min(cap, base * 2 ** k)
        if jitter == "full":
            d = e * rng.random()
        elif jitter == "equal":
            d = e / 2 + e / 2 * rng.random()
        else:
            d = e
        d = max(d, r.get("retry_after", 0))     # the server knows when capacity frees up
        if t + d >= deadline:
            return done("deadline")
        delays.append(d)
        t += d
        k += 1
'''

HINTS = [
    'Keep one `random.Random(seed)` for the whole call and draw exactly once per delay, only for the jittered modes. The test outputs depend on the order of draws.',
    'Check in this order after each attempt: success, non-retryable, out of attempts, then compute the delay, apply `retry_after`, and check the deadline.',
    'The exponent uses the 0-based attempt that just failed: the first delay ceiling is `base`, then `2 * base`, capped at `cap`.',
]

_429 = {"status": 429, "latency": 0.2}
_529 = {"status": 529, "latency": 0.1}
_OK = {"status": 200, "latency": 2.5}
TESTS = [
    ([_429, _429, _429, _OK], 0.5, 8.0, 6, 60.0, "full", 7),
    ([_429, _429, _429, _OK], 0.5, 8.0, 6, 60.0, "none", 7),
    ([_529], 1.0, 4.0, 8, 120.0, "equal", 3),
    ([_529], 1.0, 4.0, 8, 10.0, "full", 3),
    ([{"status": 429, "latency": 0.1, "retry_after": 12.0}, _OK], 0.5, 8.0, 5, 60.0, "full", 1),
    ([{"status": 429, "latency": 0.1, "retry_after": 12.0}, _OK], 0.5, 8.0, 5, 10.0, "full", 1),
    ([{"status": 400, "latency": 0.05}, _OK], 0.5, 8.0, 5, 60.0, "full", 1),
    ([{"status": 500, "latency": 30.0}, {"status": 504, "latency": 30.0}, _OK], 1.0, 16.0, 4, 61.0, "full", 11),
    ([_OK], 0.5, 8.0, 3, 60.0, "full", 0),
]

EXPLANATION = r"""
## Why jitter
Without jitter, every client that failed at the same moment retries at the same moment, $\text{base}\cdot 2^k$
later, so the overload comes back in synchronised waves. Full jitter spreads each wave uniformly over
$[0, e]$; the AWS analysis of backoff strategies found it does the least total work for contended
resources. Equal jitter keeps a guaranteed minimum wait of $e/2$. Either way, cap the ceiling: without `cap`,
attempt 10 at a 0.5 s base would wait over 4 minutes.

## Retry-After beats your formula
A 429 with `Retry-After` is the provider telling you when capacity frees up (token buckets refill
continuously). Retrying earlier only burns another request against the limit. Test 5 waits 12 s because the
header says so; test 6 gives up at once because 12 s does not fit in a 10 s deadline.

## What not to retry
A 400 (bad schema, context too long) fails identically every time (test 7). So does a 429 that means a spend
cap or quota is exhausted rather than a rate limit, which some APIs signal with an error code and no
`Retry-After`; retrying that just adds load. Retry 408, 429, 5xx and 529 (Anthropic's "overloaded").

## Deadlines, not attempt counts
The user waits for the deadline, not for attempts. Test 8 shows two 30 s timeouts eating the budget; a third
attempt cannot finish in time, so the client stops and falls back (another provider, a cached answer, an error
the UI can show) instead of hanging.

## Common bugs
- Creating a new `Random(seed)` per attempt, which repeats the same `u` every time.
- Retrying inside the SDK *and* in your own loop: three SDK retries times three of yours is nine calls.
- Retrying non-idempotent tool calls (send email, charge card) on a timeout without an idempotency key.

## Interview follow-ups
- How do you stop retries from making an outage worse? (Retry budget, e.g. at most 10% extra requests per
  client; a circuit breaker that fails fast; shed batch traffic first.)
- Retry on the same provider or fail over? (Fail over on 5xx/529 and timeouts if a second provider is
  configured and evaluated; for 429, respect the header or route to a different key or region.)
"""
