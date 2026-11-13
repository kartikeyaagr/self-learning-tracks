META = dict(title='TTFT, TPOT, ITL and token throughput from streaming timestamps', tags=['Inference'], diff='Medium',
            runtime='python', lessons=['ieng:04.01', 'ieng:04.02'],
            related=['p50-p95-and-p99-inference-latency', 'simulate-chunked-prefill-scheduling', 'optimize-throughput-and-cost-under-latency-slo'])

STATEMENT = r"""
A streaming LLM endpoint is judged on several clocks at once: how long until the first token appears, how fast tokens flow afterwards, and how much total output the server produces. Compute them from the timestamps a benchmark client records.

Write `serving_metrics(requests)`.

## Input
`requests`: a non-empty list of `(arrival_s, token_times)`: when the request was sent, and the non-empty, non-decreasing list of times at which each output token was received (all floats, `token_times[0] >= arrival_s`).

## Per request (with $n$ tokens at times $t_1 \le \dots \le t_n$, arrival $a$)
- **TTFT** = $t_1 - a$ (queueing + prefill + first decode).
- **E2E** = $t_n - a$.
- **TPOT** (time per output token after the first) = $(t_n - t_1)/(n - 1)$, or `None` if $n = 1$.
- **ITLs** (inter-token latencies) = the $n - 1$ gaps $t_{i+1} - t_i$.

## Aggregates
- `'mean_ttft'`: mean TTFT over all requests.
- `'mean_tpot'`: mean of the per-request TPOTs that exist (each request counts once), `None` if none exist.
- `'mean_itl'`: mean of **all** ITL gaps pooled across requests (each gap counts once, so long responses weigh more), `None` if there are no gaps.
- `'max_itl'`: the longest single gap (a stall the user saw), `None` if there are no gaps.
- `'throughput'`: total output tokens / (latest token time − earliest arrival), in tokens/s.
- `'per_request'`: list of `[ttft, tpot, e2e]` per request, in input order.

Return a dict with exactly these keys.
"""

STARTER = '''def serving_metrics(requests):
    """Return {'mean_ttft', 'mean_tpot', 'mean_itl', 'max_itl', 'throughput', 'per_request'}."""
    per = [[t[0] - a, (t[-1] - a) / len(t), t[-1] - a] for a, t in requests]
    tpots = [p[1] for p in per]
    return {'mean_ttft': sum(p[0] for p in per) / len(per), 'mean_tpot': sum(tpots) / len(tpots),
            'mean_itl': sum(tpots) / len(tpots), 'max_itl': max(tpots), 'throughput': 0.0, 'per_request': per}
'''

SOLUTION = '''def serving_metrics(requests):
    per, tpots, gaps = [], [], []
    for a, t in requests:
        n = len(t)
        tpot = (t[-1] - t[0]) / (n - 1) if n > 1 else None      # the first token is excluded: it is TTFT's job
        per.append([t[0] - a, tpot, t[-1] - a])
        if tpot is not None:
            tpots.append(tpot)
        gaps += [t[i + 1] - t[i] for i in range(n - 1)]

    mean = lambda xs: sum(xs) / len(xs) if xs else None
    window = max(t[-1] for _, t in requests) - min(a for a, _ in requests)
    total = sum(len(t) for _, t in requests)
    return {'mean_ttft': mean([p[0] for p in per]), 'mean_tpot': mean(tpots), 'mean_itl': mean(gaps),
            'max_itl': max(gaps) if gaps else None, 'throughput': total / window, 'per_request': per}
'''

HINTS = [
    'Do one pass over the requests, building the per-request row and appending every gap `t[i+1] - t[i]` to a single pooled list.',
    'TPOT divides by $n - 1$, not $n$: the first token belongs to TTFT. Single-token responses have no TPOT and no gaps.',
    'Mean TPOT averages per request; mean ITL averages per gap. They differ whenever response lengths differ. Throughput uses the whole benchmark window, from the first arrival to the last token.',
]

TESTS = [
    ([(0.0, [0.5, 0.6, 0.7, 0.8])],),
    ([(0.0, [0.2, 0.3]), (0.1, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])],),
    ([(0.0, [1.0]), (0.5, [1.2])],),
    ([(0.0, [0.3, 0.35, 0.4, 2.4, 2.45]), (1.0, [1.1, 1.2])],),
    ([(2.0, [2.5, 2.5, 2.6])],),
    ([(i * 0.25, [i * 0.25 + 0.1 + 0.02 * (i % 3) + 0.03 * j + (0.4 if j == 5 and i % 4 == 0 else 0.0) for j in range(1 + (i * 5) % 9)])
      for i in range(24)],),
    ([(0.0, [0.1]), (0.0, [0.2, 0.25, 0.3])],),
]

EXPLANATION = r"""
## Why several metrics
- **TTFT** is what a chat user feels first; it grows with queueing and with prompt length (prefill).
- **TPOT/ITL** is reading speed after the first token; it grows with batch size, because every running request shares each decode step.
- **Throughput** is what the operator pays for. Raising concurrency raises throughput and TPOT together, so a single number never describes a server.

## Mean TPOT vs mean ITL
Mean TPOT gives every request one vote; mean ITL gives every gap one vote, so long responses dominate it. In test 4 the first request has a 2 s stall among four gaps (TPOT 0.5375 s) and the second request has a single 0.1 s gap: mean TPOT is about 0.32 s, while mean ITL over all five gaps is 0.45 s. Report P99 or max ITL as well, because averages hide stalls and users notice them (for example a long prefill from another request blocking decodes, the problem chunked prefill solves).

## Common bugs
- Dividing $t_n - t_1$ by $n$ instead of $n - 1$.
- Measuring throughput over the sum of per-request durations instead of the wall-clock window: overlapping requests then count time twice and throughput looks far lower than it is.
- Timing on the server only: client-side TTFT includes network and HTTP overhead, which the user also waits for.

## Interview follow-ups
- The client measures a 300 ms TTFT but the engine's logs say prefill took 40 ms. Where did the rest go? (Queueing before admission, tokenization, the HTTP and network path, and the first decode step.)
- How would you compute goodput? (Count only requests that met both the TTFT and TPOT targets, and divide their tokens by the window.)
"""
