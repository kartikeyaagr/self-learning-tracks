META = dict(title='Bytes on the wire: ring all-reduce versus a parameter server', tags=['Distributed Computing'], diff='Easy', runtime='python',
            related=['simulate-distributed-collectives'])

STATEMENT = r"""
$N$ workers each hold a gradient of $M$ bytes and need the average. Links are full duplex with bandwidth $B$ bytes/s and a per-message latency $\alpha$ seconds. Write `allreduce_cost(N, M, B, alpha)` comparing three schemes by the **busiest node**:

- `"ring"`: reduce-scatter then all-gather over a ring, $2(N-1)$ sequential steps; each node sends (and receives) $2(N-1)M/N$ bytes. Time $= 2(N-1)\alpha + 2(N-1)M/(NB)$.
- `"ps"`: one dedicated server; every worker sends $M$ and receives $M$, so the server receives $NM$ in the first phase and sends $NM$ in the second. Time $= 2\alpha + 2NM/B$.
- `"sharded_ps"`: every worker also hosts a $1/N$ shard of the server. Each node sends $(N-1)M/N$ gradient bytes and later $(N-1)M/N$ parameter bytes, as $N-1$ separate messages per phase, sent one after another. Time $= 2(N-1)\alpha + 2(N-1)M/(NB)$.

Return a dict from scheme to `[bytes_sent_by_busiest_node, time]`, plus `"best"`: the scheme with the lowest time (ties in the order ring, ps, sharded_ps).
"""

STARTER = '''def allreduce_cost(N, M, B, alpha):
    """Busiest-node bytes and time for three gradient-averaging schemes."""
    ring = 2 * (N - 1) * M / N
    return {"ring": [ring, ring / B], "best": "ring"}
'''

SOLUTION = '''def allreduce_cost(N, M, B, alpha):
    ring_b = 2 * (N - 1) * M / N
    out = {"ring": [ring_b, 2 * (N - 1) * alpha + ring_b / B],
           "ps": [N * M, 2 * alpha + 2 * N * M / B],
           "sharded_ps": [ring_b, 2 * (N - 1) * alpha + ring_b / B]}
    out["best"] = min(["ring", "ps", "sharded_ps"], key=lambda k: out[k][1])
    return out
'''

HINTS = [
    'In a ring every node does the same work, so the busiest node is any node; for a parameter server it is the server.',
    'The ring sends almost 2M bytes per node regardless of N, which is why it scales; the server link carries N times M.',
    'Latency terms matter for small messages: the ring needs 2(N - 1) sequential hops.',
]

TESTS = [
    (8, 1e9, 25e9, 5e-6),
    (64, 1e9, 25e9, 5e-6),
    (64, 1e4, 25e9, 5e-6),
    (2, 4e8, 10e9, 1e-5),
    (1024, 2e9, 50e9, 2e-6),
]

EXPLANATION = r"""
## Bandwidth versus latency
The ring's bandwidth cost, about $2M/B$, does not grow with the number of workers, which is why it became the default for data-parallel training. Its latency cost does grow, $2(N-1)\alpha$, so for small tensors or very many nodes tree-based or hierarchical algorithms win. Real libraries (NCCL) switch algorithms by message size.

## Why a single server fails
All traffic crosses one link, so time grows linearly with $N$. Sharding the server across the workers removes that bottleneck and matches the ring exactly in this model. In practice the all-to-all pattern sends to every peer at once, which congests switches, while the ring only ever uses neighbour links. The sharded view is still how ZeRO and FSDP reason about the reduce-scatter and all-gather of sharded states.

## Interview follow-ups
- How does bucketing gradients and overlapping communication with the backward pass hide this cost?
"""
