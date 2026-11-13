META = dict(title='Disaggregated prefill and decode pools with KV transfer', tags=['Inference', 'Distributed Computing'], diff='Hard',
            runtime='python', lessons=['ieng:21.02', 'ieng:21.05'],
            related=['simulate-chunked-prefill-scheduling', 'ttft-tpot-itl-and-token-throughput', 'optimize-throughput-and-cost-under-latency-slo'])

STATEMENT = r"""
Disaggregated serving runs prefill and decode on **separate** GPU pools: prefill workers process prompts at full compute, then ship the KV cache over the network to decode workers, which batch many sequences. Prefill no longer interrupts decoding, at the cost of a KV transfer. Simulate one such deployment.

Write `disaggregated(requests, n_prefill, n_decode, slots, prefill_tps, tpot, kv_bytes_per_token, link_bw, link_latency)`.

## Inputs
- `requests`: list of `(arrival_s, prompt_len, output_len)`, sorted by arrival; ids are list indices. `output_len >= 1`.
- Prefill pool: `n_prefill` workers, each running one prompt at a time at `prefill_tps` tokens/s.
- Decode pool: `n_decode` workers with `slots` concurrent sequences each. Each sequence produces one token every `tpot` seconds, independent of how full the worker is.
- KV transfer of a request: `link_latency + prompt_len * kv_bytes_per_token / link_bw` seconds.

## Rules
1. **Prefill**, requests in id order: pick the prefill worker that becomes free earliest (lowest worker index on ties). Start at `max(arrival, worker free)`, end after `prompt_len / prefill_tps`. The **first token** is produced at the end of prefill.
2. If `output_len == 1` the request is done at the end of prefill: no transfer, no decode.
3. Otherwise the KV cache **arrives** at the decode pool at prefill end + transfer time. Requests take decode slots in order of arrival time (ties by id). Slot $j$ of decode worker $w$ is slot number $w \cdot \text{slots} + j$; take the slot that becomes free earliest (lowest slot number on ties). Decoding starts at `max(kv arrival, slot free)` and the remaining `output_len - 1` tokens take `(output_len - 1) * tpot`; the slot is free again at the finish time.

## Output
A tuple `(ttft, e2e)`: lists of floats per request id, `ttft = first token time - arrival`, `e2e = finish time - arrival`.
"""

STARTER = '''def disaggregated(requests, n_prefill, n_decode, slots, prefill_tps, tpot, kv_bytes_per_token, link_bw, link_latency):
    """Return (ttft list, e2e list)."""
    ttft = [p / prefill_tps for a, p, o in requests]           # no queueing, no transfer, no slot limits
    e2e = [t + (o - 1) * tpot for t, (a, p, o) in zip(ttft, requests)]
    return ttft, e2e
'''

SOLUTION = '''def disaggregated(requests, n_prefill, n_decode, slots, prefill_tps, tpot, kv_bytes_per_token, link_bw, link_latency):
    n = len(requests)
    pfree = [0.0] * n_prefill
    first, ready = [0.0] * n, [None] * n
    for i, (arr, p, o) in enumerate(requests):
        w = min(range(n_prefill), key=lambda j: (pfree[j], j))      # earliest free, lowest index on ties
        end = max(arr, pfree[w]) + p / prefill_tps
        pfree[w] = end
        first[i] = end
        if o > 1:
            ready[i] = end + link_latency + p * kv_bytes_per_token / link_bw

    finish = list(first)                                           # output_len == 1: done at prefill end
    sfree = [0.0] * (n_decode * slots)
    for i in sorted((i for i in range(n) if ready[i] is not None), key=lambda i: (ready[i], i)):
        s = min(range(len(sfree)), key=lambda j: (sfree[j], j))
        finish[i] = max(ready[i], sfree[s]) + (requests[i][2] - 1) * tpot
        sfree[s] = finish[i]
    return [first[i] - requests[i][0] for i in range(n)], [finish[i] - requests[i][0] for i in range(n)]
'''

HINTS = [
    'The two pools only interact through the KV arrival time, so simulate all prefills first, then all decodes.',
    'Keep a list of "free at" times for prefill workers and another for decode slots; each assignment is `min` over `(free_time, index)`.',
    'Decode admission is ordered by KV arrival time, not by request id: a short prompt that arrives later can reach the decode pool first.',
]

TESTS = [
    ([(0.0, 100, 3)], 1, 1, 1, 100.0, 0.1, 10, 1000.0, 0.5),
    ([(0.0, 100, 2), (0.0, 100, 2)], 1, 1, 1, 100.0, 0.5, 0, 1.0, 0.0),
    ([(0.0, 400, 3), (0.1, 10, 3)], 2, 1, 1, 100.0, 1.0, 100, 1000.0, 0.0),
    ([(0.0, 50, 1), (0.0, 50, 4)], 1, 1, 1, 100.0, 0.25, 8, 400.0, 0.1),
    ([(0.1 * i, 200 + 300 * (i % 4), 20 + 10 * (i % 3)) for i in range(16)], 2, 2, 3, 4000.0, 0.02, 131072, 25e9, 0.002),
    ([(0.1 * i, 200 + 300 * (i % 4), 20 + 10 * (i % 3)) for i in range(16)], 4, 1, 2, 4000.0, 0.02, 131072, 25e9, 0.002),
    ([(0.0, 8000, 50), (0.0, 100, 50), (0.01, 100, 50)], 1, 1, 8, 10000.0, 0.03, 131072, 5e9, 0.001),
    ([(0.05 * i, 1000, 2 + (i * 7) % 30) for i in range(30)], 3, 2, 4, 20000.0, 0.025, 327680, 50e9, 0.0005),
]

EXPLANATION = r"""
## Why it works
The prefill pool is a classic multi-server FCFS queue: each prompt goes to the worker that frees up first. The decode pool is a second queue whose arrivals are the KV-ready times. Because the decode rate here does not depend on load, the pools only interact through those times, so two passes are exact. Ordering decode admission by readiness matters: in test 3 the short second prompt finishes prefill on the other worker and claims the only decode slot first.

## What it shows
TTFT depends only on the prefill pool (queueing plus prompt length) and decode capacity only on the decode pool, so each SLO is met by sizing one pool. Tests 5 and 6 send the same traffic: test 6 has more prefill workers but a third of the decode slots, so TTFT doesn't improve (prefill was not the bottleneck) while requests queue for decode and E2E latency grows. The KV transfer adds to time-to-second-token; for the 8000-token prompt over a 5 GB/s link in test 7 it adds about a quarter of the prefill time.

## Common bugs
- Starting decode at prefill end, forgetting the transfer.
- Assigning decode slots in request-id order instead of KV-arrival order.
- Sending single-token requests through the decode pool: their only token comes from prefill.

## Interview follow-ups
- When does disaggregation lose to colocated serving with chunked prefill? (Short prompts or slow interconnect: the transfer cost exceeds the interference it removes, and two pools are harder to keep both busy.)
- How would you choose the prefill:decode GPU ratio? (From the workload: prefill demand ≈ arrival rate × mean prompt / prefill throughput, decode demand ≈ arrival rate × mean output × TPOT / slots; provision each to its SLO.)
"""
