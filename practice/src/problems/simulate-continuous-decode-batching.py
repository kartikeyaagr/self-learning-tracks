META = dict(title='Simulate continuous batching under a KV-memory budget', tags=['Inference'], diff='Hard',
            runtime='python', lessons=['ieng:07.03', 'ieng:07.04'],
            related=['plan-kv-cache-capacity', 'find-reusable-prefixes-in-kv-trie', 'route-cache-aware-prefill-requests'])

STATEMENT = r"""
Static batching waits for a whole batch to finish before starting the next one, so short requests sit idle behind long ones. **Continuous batching** schedules at the granularity of one decode iteration: a finished request leaves the batch immediately and a waiting one takes its slot. Two resources limit the batch: a maximum number of concurrent sequences, and the KV cache, which must hold every running request's tokens.

Write `simulate(requests, max_batch, kv_capacity)` that runs the scheduler and returns a dict.

## Requests
`requests` is a list of `(arrival, prompt_len, output_len)` int triples (arrival $\ge 0$, output $\ge 1$). Each request **reserves** `prompt_len + output_len` KV slots when admitted (no preemption) and never needs more than `kv_capacity`.

## One iteration at step $s = 0, 1, 2, \dots$
1. **Admit.** The waiting queue holds arrived ($\text{arrival} \le s$), not-yet-admitted requests ordered by (arrival, index). While fewer than `max_batch` requests are running and the queue's **head** fits in the free KV slots, admit it. Stop at the first head that doesn't fit (no skipping ahead).
2. **Run.** Every running request, including the ones admitted this step, produces one token (a newly admitted request's prefill produces its first token in the same step). Record the number of running requests as this step's batch size.
3. **Retire.** A request that has now produced `output_len` tokens finishes at step $s$; its KV slots are free from step $s + 1$.

The simulation ends after the step in which the last request finishes. Steps with no running requests (waiting for arrivals) still count, with batch size 0.

## Output
| key | value |
|---|---|
| `"start"` | list, admission step per request (input order) |
| `"finish"` | list, finishing step per request |
| `"latency"` | list, `finish - arrival + 1` |
| `"batch_sizes"` | list, one int per step from 0 to the last finish |
| `"utilization"` | float, `sum(batch_sizes) / (len(batch_sizes) * max_batch)` |

`requests` is non-empty.
"""

STARTER = '''def simulate(requests, max_batch, kv_capacity):
    """Return a dict: start, finish, latency, batch_sizes, utilization."""
    raise NotImplementedError
'''

SOLUTION = '''def simulate(requests, max_batch, kv_capacity):
    n = len(requests)
    order = sorted(range(n), key=lambda i: (requests[i][0], i))   # FCFS by arrival, then input order
    start, finish, produced = [None] * n, [None] * n, [0] * n
    nxt, running, used, done = 0, [], 0, 0                         # nxt: first not-yet-admitted in `order`
    batch_sizes, s = [], 0
    while done < n:
        # 1. Admit from the head of the queue while a slot and enough KV are free.
        while nxt < n and len(running) < max_batch:
            i = order[nxt]
            arrival, p, o = requests[i]
            if arrival > s or used + p + o > kv_capacity:
                break                                              # not arrived yet, or head-of-line blocked
            start[i], used, nxt = s, used + p + o, nxt + 1
            running.append(i)
        # 2. One decode iteration for the whole batch.
        batch_sizes.append(len(running))
        still = []
        for i in running:
            produced[i] += 1
            if produced[i] == requests[i][2]:
                finish[i] = s                                      # 3. retire; slots free next step
                used -= requests[i][1] + requests[i][2]
                done += 1
            else:
                still.append(i)
        running, s = still, s + 1
    latency = [finish[i] - requests[i][0] + 1 for i in range(n)]
    util = sum(batch_sizes) / (len(batch_sizes) * max_batch)
    return {"start": start, "finish": finish, "latency": latency, "batch_sizes": batch_sizes, "utilization": util}
'''

HINTS = [
    'Sort request indices once by `(arrival, index)`; the waiting queue is then just a pointer into that order. A request at the pointer that has not arrived yet also blocks admission.',
    'Do admission, then record the batch size, then increment every running request\'s token count and retire the finished ones. Freeing their KV after the step is what makes the slots available "from step $s+1$".',
    'Loop until every request has finished. Empty steps (nothing running, next arrival in the future) still append a batch size of 0.',
]

TESTS = [
    ([(0, 4, 2), (0, 4, 3)], 1, 100),
    ([(0, 4, 2), (0, 4, 3)], 2, 100),
    ([(0, 10, 3), (0, 10, 3), (0, 2, 1)], 4, 25),
    ([(3, 1, 2), (0, 1, 1)], 2, 10),
    ([(0, 50, 50), (1, 5, 5), (1, 5, 5), (2, 5, 5)], 3, 120),
    ([(i // 2, 8 + 3 * (i % 4), 1 + (7 * i) % 9) for i in range(16)], 4, 60),
    ([(0, 30, 10), (0, 30, 10), (0, 30, 10)], 8, 80),
    ([(5 * i, 3, 4) for i in range(4)], 2, 50),
]

EXPLANATION = r"""
## Why iteration-level scheduling wins
Decode steps are memory-bound: running 8 sequences costs barely more than running 1, so throughput is roughly proportional to the batch size. Static batching keeps the batch fixed until its *longest* request finishes, so the batch drains toward size 1 and new requests wait. Admitting at every step keeps the batch full whenever there is work. Tests 1 and 2 are the same two requests at batch limits 1 and 2: the second finishes two steps earlier with the higher limit.

## The KV constraint and head-of-line blocking
Here each request reserves its **maximum** footprint at admission, so it can never run out of memory, but memory reserved for tokens that haven't been generated yet is idle. In test 3, the second 13-slot request doesn't fit next to the first in a 25-slot cache, and the small third request (3 slots, which would fit) has to wait behind it even though batch slots are free: head-of-line blocking. Test 7 is limited purely by memory: 80 slots hold two of the 40-slot requests, never three, despite `max_batch = 8`. Production engines (vLLM) allocate KV blocks on demand and preempt a request when memory runs out, trading occasional recomputation for much higher concurrency.

## Complexity
$O(\text{steps} \times \text{max\_batch})$ plus the initial sort.

## Common bugs
- Freeing KV slots in the same step a request finishes, letting a new request start one step early.
- Skipping over a blocked head to admit a later, smaller request (a different policy; it can starve big requests).
- Not counting the prefill step as producing the first token, which shifts every finish by one.
- Stopping the simulation while requests are still waiting for their arrival time.

## Interview follow-ups
- Add chunked prefill: long prompts are split across several steps under a per-step token budget. What changes in the admission loop?
- Which metric would you optimise, time-to-first-token or throughput, and how does `max_batch` trade them off?
"""
