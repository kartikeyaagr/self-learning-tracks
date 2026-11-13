META = dict(title='Nearest-rank P50, P95 and P99 from a request log', tags=['Inference', 'Probability and Statistics'], diff='Easy',
            runtime='python', lessons=['ieng:04.05', 'ieng:04.06'],
            related=['percentiles-quantiles', 'ttft-tpot-itl-and-token-throughput', 'optimize-throughput-and-cost-under-latency-slo'])

STATEMENT = r"""
Benchmark tools report tail latency with the **nearest-rank** percentile: the smallest observed value that at least $p\%$ of the samples do not exceed. It always returns a latency that really happened, unlike interpolation.

Write `latency_report(requests)`.

## Input
`requests`: a non-empty list of `(start_s, end_s, status)` tuples: start and end times in seconds (floats, `end_s > start_s`) and an HTTP status code (int). A request **succeeded** if `status == 200`; only successful requests count toward latency.

## Definitions
- Latency of a request in **milliseconds**: `(end_s - start_s) * 1000`.
- Nearest-rank percentile of $n$ sorted latencies $x_1 \le \dots \le x_n$: $P_p = x_k$ with $k = \lceil p \cdot n / 100 \rceil$. Compute $k$ with integer arithmetic (`p` is an int), so $0.95 \cdot 20$ can't round to 19.000000000000004.

## Output
A dict with keys:
- `'p50'`, `'p95'`, `'p99'`: floats (ms).
- `'mean'`: mean latency of the successful requests (ms).
- `'tail_ratio'`: `p99 / p50`, how much worse the tail is than the median.
- `'error_rate'`: failed requests / all requests, a float.

If no request succeeded, every latency value (including `'tail_ratio'`) is `None`.
"""

STARTER = '''def latency_report(requests):
    """Return {'p50', 'p95', 'p99', 'mean', 'tail_ratio', 'error_rate'}."""
    lat = sorted((e - s) * 1000 for s, e, _ in requests)
    pick = lambda q: lat[int(q * (len(lat) - 1))]
    return {'p50': pick(0.5), 'p95': pick(0.95), 'p99': pick(0.99), 'mean': sum(lat) / len(lat),
            'tail_ratio': pick(0.99) / pick(0.5), 'error_rate': 0.0}
'''

SOLUTION = '''def latency_report(requests):
    lat = sorted((end - start) * 1000 for start, end, status in requests if status == 200)
    error_rate = 1 - len(lat) / len(requests)
    if not lat:
        return {'p50': None, 'p95': None, 'p99': None, 'mean': None, 'tail_ratio': None, 'error_rate': error_rate}

    n = len(lat)
    def pct(p):
        k = -(-p * n // 100)                 # ceil(p*n/100) in integers; k >= 1 because p, n >= 1
        return lat[k - 1]

    p50, p95, p99 = pct(50), pct(95), pct(99)
    return {'p50': p50, 'p95': p95, 'p99': p99, 'mean': sum(lat) / n,
            'tail_ratio': p99 / p50, 'error_rate': error_rate}
'''

HINTS = [
    'Filter to `status == 200` first, then convert to milliseconds and sort. The error rate uses all requests.',
    'The rank is $k = \\lceil p n / 100 \\rceil$ and the answer is the $k$-th smallest, i.e. index `k - 1`. `-(-p * n // 100)` is an exact integer ceiling.',
    'With few samples, P95 and P99 are both the maximum (for $n < 100$, $P_{99} = x_n$). Handle the case with zero successes before indexing.',
]


def _log(seed, n):
    # deterministic pseudo-random log: mostly fast, a slow tail, a few failures
    out, x = [], seed
    for i in range(n):
        x = (x * 1103515245 + 12345) % 2**31
        base = 0.05 + (x % 1000) / 10000
        slow = 1.5 if x % 37 == 0 else 0.0
        status = 503 if x % 23 == 0 else 200
        out.append((i * 0.1, i * 0.1 + base + slow, status))
    return out


TESTS = [
    ([(0.0, 0.1, 200), (0.0, 0.2, 200), (0.0, 0.3, 200), (0.0, 1.0, 500)],),
    ([(1.0, 1.25, 200)],),
    ([(0.0, (i + 1) / 1000, 200) for i in range(20)],),
    ([(0.0, (i + 1) / 1000, 200) for i in range(100)],),
    ([(0.0, 1.0, 429), (2.0, 2.5, 503)],),
    ([(0.0, 0.2, 200), (0.0, 0.2, 200), (0.0, 0.05, 200), (0.0, 3.0, 200)],),
    (_log(7, 250),),
    (_log(99, 1000),),
]

EXPLANATION = r"""
## Why nearest rank
The definition guarantees that at least $p\%$ of requests were at or below the reported value and that the value was really observed, which is what an SLO statement ("99% of requests under 2 s") needs. Interpolated quantiles (NumPy's default) can report a latency between two samples that no request had; on small samples the two methods differ noticeably, on large ones they converge.

## Cost
Sorting is $O(n \log n)$; each percentile is then $O(1)$. Production systems can't keep every sample, so they use histograms (Prometheus buckets) or sketches (t-digest, DDSketch) that answer percentiles approximately in bounded memory.

## Common bugs
- Including failed requests: fast errors (a 429 rejected in 2 ms) pull the percentiles down and make an overloaded server look healthy.
- Averaging per-minute P99s to get an hourly P99: percentiles don't average; recompute from the merged samples or merge the histograms.
- Float rank arithmetic: `math.ceil(0.95 * n)` is off by one for some $n$ because of representation error.

## Interview follow-ups
- Why is P99 a poor signal with 50 requests? (It equals the maximum: one outlier decides it. You need a few thousand samples for a stable P99.)
- What is coordinated omission? (A load generator that waits for slow responses before sending the next request under-samples exactly the slow periods, hiding the tail.)
"""
