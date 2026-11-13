META = dict(title='Algorithm vs bus bandwidth for collectives', tags=['Distributed Computing'], diff='Easy',
            runtime='python', lessons=['ieng:18.04', 'ieng:18.02'],
            related=['simulate-distributed-collectives', 'tensor-parallel-communication-cost', 'synchronize-data-parallel-gradients'])

STATEMENT = r"""
A benchmark says "all-reduce of 1 GB on 8 GPUs took 6 ms". Is that good? Dividing size by time gives the **algorithm bandwidth**, but different collectives move different amounts of data per byte of payload. The **bus bandwidth** corrects for that, so it can be compared directly with the link speed and across GPU counts.

Write `collective_bandwidth(op, n_bytes, n_ranks, seconds, link_gbps=None)`.

## Definitions
- `algbw` $= \dfrac{\text{n\_bytes}}{\text{seconds}} / 10^9$ in GB/s. For `all_gather` and `reduce_scatter`, `n_bytes` is the size of the **full** buffer (the gathered output, or the input being scattered), not one rank's piece.
- `busbw` $=$ `algbw` $\times$ `factor`, with $n$ = `n_ranks`:

| `op` | `factor` |
|---|---|
| `'all_reduce'` | $2(n-1)/n$ |
| `'all_gather'`, `'reduce_scatter'`, `'all_to_all'` | $(n-1)/n$ |
| `'broadcast'`, `'reduce'`, `'send_recv'` | $1$ |

- `efficiency` $=$ `busbw / link_gbps` if `link_gbps` (the per-GPU link bandwidth in GB/s) is given, else `None`.
- `ideal_seconds`: the time at which `busbw` would equal `link_gbps`, i.e. $\dfrac{\text{n\_bytes} \cdot \text{factor}}{\text{link\_gbps} \cdot 10^9}$, or `None` without a link speed.

## Output
A dict `{'algbw', 'busbw', 'factor', 'efficiency', 'ideal_seconds'}` (floats, or `None` where stated). Inputs: `n_bytes` $> 0$, `n_ranks` $\ge 1$, `seconds` $> 0$.
"""

STARTER = '''def collective_bandwidth(op, n_bytes, n_ranks, seconds, link_gbps=None):
    """Return {'algbw', 'busbw', 'factor', 'efficiency', 'ideal_seconds'}."""
    algbw = n_bytes / seconds / 1e9
    return {'algbw': algbw, 'busbw': algbw, 'factor': 1.0,    # TODO: factor depends on the collective
            'efficiency': None, 'ideal_seconds': None}
'''

SOLUTION = '''FACTOR = {
    'all_reduce': lambda n: 2 * (n - 1) / n,       # reduce-scatter + all-gather
    'all_gather': lambda n: (n - 1) / n,
    'reduce_scatter': lambda n: (n - 1) / n,
    'all_to_all': lambda n: (n - 1) / n,           # a rank keeps 1/n of its data
    'broadcast': lambda n: 1.0,
    'reduce': lambda n: 1.0,
    'send_recv': lambda n: 1.0,
}

def collective_bandwidth(op, n_bytes, n_ranks, seconds, link_gbps=None):
    factor = float(FACTOR[op](n_ranks))
    algbw = n_bytes / seconds / 1e9
    busbw = algbw * factor
    return {
        'algbw': algbw,
        'busbw': busbw,
        'factor': factor,
        'efficiency': busbw / link_gbps if link_gbps else None,
        'ideal_seconds': n_bytes * factor / (link_gbps * 1e9) if link_gbps else None,
    }
'''

HINTS = [
    'Algorithm bandwidth is just bytes over time, in units of $10^9$ bytes per second.',
    'The factor counts how many bytes each rank must send per byte of payload in a bandwidth-optimal algorithm: a ring all-reduce sends $2(n-1)/n$ of the buffer, an all-gather $(n-1)/n$.',
    'A table of lambdas keyed by `op` keeps it short. For one rank, all-reduce has factor 0: no data needs to move, so its bus bandwidth is 0.',
]

TESTS = [
    ('all_reduce', 1_000_000_000, 8, 0.006),
    ('all_gather', 1_000_000_000, 8, 0.004, 400.0),
    ('reduce_scatter', 2 ** 30, 4, 0.01, 100.0),
    ('all_to_all', 512 * 2 ** 20, 16, 0.05, 50.0),
    ('broadcast', 10 ** 8, 64, 0.002, 50.0),
    ('all_reduce', 4096, 1, 1e-5),
    ('all_reduce', 8 * 2 ** 30, 2, 0.05, 450.0),
    ('send_recv', 2 ** 20, 2, 3e-5, 25.0),
]

EXPLANATION = r"""
## Why the correction
In a ring all-reduce every rank sends and receives $2(n-1)/n \cdot S$ bytes for a buffer of $S$ bytes. If each link carries $B$ bytes/s, the best possible time is $2(n-1)S/(nB)$, so $\text{algbw} = S/t \le B \cdot \frac{n}{2(n-1)}$: it drifts with $n$ and is not comparable across cluster sizes. Multiplying by the factor gives $\text{busbw} \le B$, a number that tops out at the hardware speed for any $n$, so 80 to 90% of the link speed is a good result and 30% means something is wrong (topology, small messages, contention).

## Reading the tests
- Test 1: 1 GB in 6 ms on 8 GPUs is 166.7 GB/s algorithm bandwidth but 291.7 GB/s bus bandwidth.
- Test 7: 8 GiB on 2 GPUs over a 450 GB/s link: about 172 GB/s, 38% of the link; probably latency- or protocol-limited.
- Test 6: one rank, factor 0. A benchmark reporting non-zero bus bandwidth there is measuring a copy.

## Common bugs
- Using one rank's shard size as `n_bytes` for all-gather, which understates bandwidth by a factor $n$.
- Mixing GiB and GB: $2^{30}$ bytes is 1.074 GB.

## Interview follow-ups
- Why does bus bandwidth drop for small messages, and how would you estimate the latency term $\alpha$ from a sweep of sizes? (Fit $t = \alpha + S \cdot \text{factor}/B$.)
- On a node with NVLink and a slower network between nodes, which link does the bus bandwidth of an inter-node all-reduce approach, and how does hierarchical all-reduce help?
"""
