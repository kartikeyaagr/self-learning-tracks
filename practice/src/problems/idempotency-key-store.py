META = dict(title='Server-side idempotency keys: replay, conflict, in-flight and expiry', tags=['Distributed Computing', 'MLOps'],
            diff='Medium', runtime='python', lessons=['sdsd:02.09'],
            related=['retry-backoff-schedule', 'lru-cache-with-ttl'])

STATEMENT = r"""
Write `handle(requests, ttl)` that simulates a payment API's `Idempotency-Key` handling and returns what each
request got back.

`requests` arrive in order of non-decreasing `t` (integer seconds). Each is a dict with:
- `key`: the client's idempotency key,
- `body`: the request parameters as a string (compare them exactly),
- `t`: arrival time,
- `status`, `response`, `duration`: what **executing this request** would produce (an HTTP status, a response
  string) and how long the work takes. Execution finishes at `t + duration`.

The server keeps one record per key: the body, the finish time `done`, and the stored status and response.
For each request, look up its key:
1. **Forget stale records first.** If a record exists, `t >= done`, and either its status is 500 or above (the
   work failed and was rolled back, so the key is released) or `t >= done + ttl` (the retention window is
   over), delete it.
2. **No record:** execute. Count one execution, store the record, and log `["executed", status, response]`.
3. **Record with a different body:** log `["conflict", 422, None]`. The stored record is unchanged.
4. **Same body, `t < done`** (the first attempt is still running): log `["in_progress", 409, None]`.
5. **Same body, finished:** log `["replayed", stored status, stored response]` without executing.

4xx results are stored and replayed like 2xx ones: the same bad request gets the same answer.

Return `{"log": [one entry per request], "executions": number of executions}`.
"""

STARTER = '''def handle(requests, ttl):
    """Return the stored response when a key repeats."""
    store, log, executions = {}, [], 0
    for r in requests:
        if r["key"] in store:
            log.append(["replayed", *store[r["key"]]])
        else:
            executions += 1
            store[r["key"]] = (r["status"], r["response"])
            log.append(["executed", r["status"], r["response"]])
    return {"log": log, "executions": executions}
'''

SOLUTION = '''def handle(requests, ttl):
    store, log, executions = {}, [], 0     # key -> {"body", "done", "status", "response"}
    for r in requests:
        k, t = r["key"], r["t"]
        rec = store.get(k)
        if rec and t >= rec["done"] and (rec["status"] >= 500 or t >= rec["done"] + ttl):
            del store[k]                       # rolled back, or past retention: the key is free again
            rec = None
        if rec is None:
            executions += 1
            store[k] = {"body": r["body"], "done": t + r["duration"],
                        "status": r["status"], "response": r["response"]}
            log.append(["executed", r["status"], r["response"]])
        elif rec["body"] != r["body"]:
            log.append(["conflict", 422, None])          # same key, new parameters: a client bug
        elif t < rec["done"]:
            log.append(["in_progress", 409, None])       # never run the same key twice concurrently
        else:
            log.append(["replayed", rec["status"], rec["response"]])
    return {"log": log, "executions": executions}
'''

HINTS = [
    'Store the body with the result. A key alone cannot tell a retry from a different request that reused the key.',
    'A record is "in progress" from `t` until `t + duration`; a replay that arrives in that window must not execute again.',
    'Expiry and rolled-back failures are checked lazily, when the key is next seen: only then can the key be executed again.',
]

def _r(key, t, body="amount=2000&currency=USD", status=201, response="ch_91", duration=1):
    return {"key": key, "body": body, "t": t, "status": status, "response": response, "duration": duration}

TESTS = [
    ([_r("k1", 0, duration=2), _r("k1", 3, response="ch_92")], 86400),
    ([_r("k1", 0, duration=5), _r("k1", 2), _r("k1", 5), _r("k1", 9)], 86400),
    ([_r("k1", 0), _r("k1", 4, body="amount=3000&currency=USD"), _r("k1", 6)], 86400),
    ([_r("k1", 0, status=503, response="upstream down", duration=2), _r("k1", 1), _r("k1", 2, response="ch_93"),
      _r("k1", 7)], 86400),
    ([_r("k1", 0, duration=1), _r("k1", 50), _r("k1", 100), _r("k1", 101, response="ch_94"), _r("k1", 102)], 100),
    ([_r("a", 0), _r("b", 0, response="ch_95"), _r("a", 2), _r("b", 2)], 60),
    ([_r("k1", 0, status=400, response="currency missing", body="amount=2000"), _r("k1", 3, body="amount=2000")], 60),
    ([_r("k1", 0, duration=10), _r("k1", 4, body="amount=1"), _r("k1", 6)], 60),
    ([_r("k1", 0, duration=1), _r("k1", 70, body="amount=1", response="ch_96")], 60),
    ([], 60),
]

EXPLANATION = r"""
## The problem it solves
A timeout tells the client nothing: the charge may never have started, or it may have succeeded and only the
reply was lost. The client therefore retries with the **same** key, and the server makes the retry safe by
answering from what it stored instead of charging again (test 1: one execution, the retry gets `ch_91`, not a
second charge).

## Why store the body
If the same key arrives with different parameters (test 3, `amount=3000`), the caller has a bug: it reused a key
for a different operation. Returning the old result would hide it, and executing would break the guarantee, so
the answer is a validation error. This is also why the server must not derive the key from the body: two
identical requests may both be intended (test 6, two keys with the same body both execute).

## In-flight retries
A client with a short timeout can retry while the first attempt is still running (test 2 at `t = 2`). Executing
it would race the first attempt into a double charge, so the server answers 409 and the client backs off. In a
real system the record is inserted in the same transaction that starts the work, so two concurrent requests
cannot both see "no record".

## Failures and expiry
A 5xx means the work was rolled back, so the key is released and the next retry executes (test 4); a 4xx is a
final answer and is replayed (test 7). Keys cannot be kept forever: once the retention window passes, the key is
forgotten and the next request with it is treated as new (tests 5 and 9). Pick the window to outlast every
client's retry schedule (Stripe keeps keys for 24 hours).

## Common bugs
- Keying only on the key and ignoring the body, which hides client bugs.
- Storing the key outside the transaction that does the work: a crash in between either loses the key (double
  charge on retry) or keeps a key for work that never happened.
- Caching 5xx responses, which turns a transient failure into a permanent one for that key.

## Interview follow-ups
- Where do the records live? (In the same database as the charge, so key and effect commit together; a TTL or a
  reaper job cleans them up.)
- The first attempt crashed halfway after calling the card network. What does the retry do? (Resume from the
  recorded step, passing the card network its own idempotency reference so it cannot charge twice.)
"""
