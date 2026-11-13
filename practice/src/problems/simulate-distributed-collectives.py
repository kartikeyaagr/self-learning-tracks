import numpy as np

META = dict(title='Simulate ring collectives on lists of shards', tags=['Distributed Computing'], diff='Medium',
            runtime='numpy', lessons=['ieng:18.04', 'ieng:18.02'],
            related=['collective-effective-bandwidth', 'synchronize-data-parallel-gradients', 'ring-all-reduce-with-ppermute'])

STATEMENT = r"""
NCCL implements all-reduce on a ring: a **reduce-scatter** pass in which partial sums travel around the ring, then an **all-gather** pass that circulates the finished chunks. Simulate it with plain arrays: rank $r$'s data is `shards[r]`, and rank $r$ only ever sends to rank $(r + 1) \bmod P$.

Write `ring_collective(op, shards)`.

## Inputs
- `op`: one of `'all_reduce'`, `'reduce_scatter'`, `'all_gather'`, `'all_to_all'`.
- `shards`: a list of $P \ge 1$ one-dimensional `float64` arrays.
  - For `all_reduce`, `reduce_scatter` and `all_to_all`, all have the same length $n$, and each is split into $P$ chunks with `np.array_split` (earlier chunks one longer when $P \nmid n$). Chunk $c$ of every rank covers the same positions.
  - For `all_gather`, `shards[r]` **is** chunk $r$; lengths may differ.

## Ring schedule (all ranks act simultaneously in each step; indices mod $P$)
- **Reduce-scatter**, steps $s = 0, \dots, P-2$: rank $r$ sends its current copy of chunk $r - 1 - s$ to rank $r + 1$, which **adds** it to its own copy (`own = own + received`). Afterwards rank $r$ holds the fully reduced chunk $r$.
- **All-gather**, steps $s = 0, \dots, P-2$: rank $r$ sends its copy of chunk $r - s$ to rank $r + 1$, which **overwrites** its copy.

## Operations
| `op` | output of rank $r$ |
|---|---|
| `all_reduce` | reduce-scatter then all-gather: the full summed array (length $n$) |
| `reduce_scatter` | reduce-scatter only: the reduced chunk $r$ |
| `all_gather` | all-gather only: the concatenation of all chunks in rank order |
| `all_to_all` | not a ring: rank $r$ sends its chunk $j$ directly to rank $j$ for every $j \ne r$. Output: chunk $r$ of rank 0, chunk $r$ of rank 1, ..., concatenated |

## Output
A tuple `(outputs, sent)`: `outputs` is a list of $P$ `float64` arrays, `sent` a list of $P$ ints, the number of **elements** each rank sent in total. Do not modify the inputs.
"""

STARTER = '''import numpy as np

def ring_collective(op, shards):
    """Return (outputs: list of P arrays, sent: list of P element counts)."""
    P = len(shards)
    total = sum(shards)                                  # TODO: only all_reduce, and no traffic accounting
    return [total.copy() for _ in range(P)], [0] * P
'''

SOLUTION = '''import numpy as np

def _reduce_scatter(chunks, sent):
    """chunks[r][c]: rank r's copy of chunk c, updated in place. Afterwards rank r owns the full chunk r."""
    P = len(chunks)
    for s in range(P - 1):
        # all sends of a step read the state from before the step
        msgs = [(r, (r - 1 - s) % P, chunks[r][(r - 1 - s) % P].copy()) for r in range(P)]
        for r, c, data in msgs:
            chunks[(r + 1) % P][c] = chunks[(r + 1) % P][c] + data
            sent[r] += data.size

def _all_gather(chunks, sent):
    P = len(chunks)
    for s in range(P - 1):
        msgs = [(r, (r - s) % P, chunks[r][(r - s) % P].copy()) for r in range(P)]
        for r, c, data in msgs:
            chunks[(r + 1) % P][c] = data
            sent[r] += data.size

def ring_collective(op, shards):
    P = len(shards)
    sent = [0] * P
    if op == 'all_gather':
        # rank r starts knowing only its own chunk; the placeholders are overwritten by the ring
        chunks = [[shards[c].copy() if c == r else np.zeros(0) for c in range(P)] for r in range(P)]
        _all_gather(chunks, sent)
        return [np.concatenate(ch) for ch in chunks], sent

    chunks = [[c.copy() for c in np.array_split(x, P)] for x in shards]
    if op == 'all_to_all':
        out = [np.concatenate([chunks[src][r] for src in range(P)]) for r in range(P)]
        sent = [sum(chunks[r][j].size for j in range(P) if j != r) for r in range(P)]
        return out, sent
    _reduce_scatter(chunks, sent)
    if op == 'reduce_scatter':
        return [chunks[r][r] for r in range(P)], sent
    _all_gather(chunks, sent)                                  # op == 'all_reduce'
    return [np.concatenate(ch) for ch in chunks], sent
'''

HINTS = [
    'Represent each rank as a list of $P$ chunk arrays. In each ring step, first collect every rank\'s outgoing message (a copy), then deliver them all; delivering as you go would let a chunk travel several hops in one step.',
    'Reduce-scatter: at step $s$ rank $r$ sends chunk $(r - 1 - s) \\bmod P$ and the receiver adds. Trace $P = 3$ by hand: chunk 0 starts at rank 1, is added into rank 2, then into rank 0, which ends with the full sum.',
    'All-reduce is reduce-scatter followed by all-gather on the same chunk lists. Count `data.size` for every message a rank sends; with equal chunks a rank sends $2(P-1)\\,n/P$ elements in total.',
]

_r = np.random.default_rng(9)
TESTS = [
    ('all_reduce', [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]), np.array([100.0, 200.0, 300.0])]),
    ('reduce_scatter', [np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0, 8.0])]),
    ('all_gather', [np.array([1.0]), np.array([2.0, 3.0]), np.array([]), np.array([4.0, 5.0, 6.0])]),
    ('all_to_all', [np.arange(6.0), np.arange(6.0) + 10, np.arange(6.0) + 20]),
    ('all_reduce', [_r.normal(size=10) for _ in range(4)]),
    ('reduce_scatter', [_r.normal(size=7) for _ in range(3)]),
    ('all_reduce', [np.array([1.0, 2.0])]),
    ('all_reduce', [_r.normal(size=3) for _ in range(5)]),
    ('all_to_all', [_r.normal(size=5) for _ in range(2)]),
]

EXPLANATION = r"""
## Why it works
In reduce-scatter, chunk $c$ starts at rank $c + 1$ and moves one hop per step, picking up each rank's contribution, so after $P - 1$ steps it reaches rank $c$ carrying all $P$ terms. Every rank is busy every step, each handling a different chunk, so all links are used in parallel. The all-gather pass then circulates the finished chunks the same way, overwriting instead of adding.

## Bandwidth optimality
Each rank sends $(P-1)\,n/P$ elements per pass, $2(P-1)\,n/P$ for all-reduce, which is within a factor $(P-1)/P$ of the lower bound for any algorithm: every rank must receive at least the information it lacks. The time is roughly $2(P-1)\bigl(\alpha + \frac{n/P}{\beta}\bigr)$: bandwidth-optimal, but the $2(P-1)\alpha$ latency term grows with $P$, which is why NCCL switches to trees for small messages and large clusters.

## Edge cases
- $P = 1$ (test 7): no steps, nothing sent, output equals input.
- $n$ not divisible by $P$ (tests 5, 6, 8): chunks differ in size, so ranks send different amounts.
- An empty shard in all-gather (test 3) still takes its turn in the ring.

## Common bugs
- Delivering messages within a step before all sends are read, letting data jump several ranks at once (the results then depend on loop order).
- Off-by-one in the chunk index so ranks end up owning chunk $r + 1$; the sum is still correct for all-reduce, which hides the bug until reduce-scatter is used.

## Interview follow-ups
- ZeRO and FSDP replace one all-reduce by a reduce-scatter (gradients) plus an all-gather (parameters). Why is the total traffic the same?
- How does a 2-D (hierarchical) all-reduce use NVLink inside a node and the network across nodes, and what does it save?
"""
