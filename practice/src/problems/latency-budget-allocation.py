META = dict(title='Split a p95 latency budget across pipeline stages', tags=['Inference', 'MLOps'], diff='Medium',
            runtime='python', lessons=['aieng:12.05'],
            related=['p50-p95-and-p99-inference-latency', 'ttft-tpot-itl-and-token-throughput', 'trace-critical-path'])

STATEMENT = r"""
An LLM feature runs a pipeline of stages per request: guardrail, retrieval, rerank, model call. Some stages run
in parallel. You have a p95 end-to-end budget and per-stage latency samples from the same `n` requests.

Write `budget_report(samples, groups, budget)`.

- `samples`: dict from stage name to a list of `n` latencies in ms; index `i` is request `i` in every list.
- `groups`: a list of lists of stage names. Groups run one after another; stages inside a group run in parallel,
  so a group takes the **max** of its stages for that request.
- `budget`: the p95 target for the whole request, in ms.

Use the nearest-rank percentile: sort the values and take the element at index $\lceil q\,n \rceil - 1$
($q = 0.5$ for p50, $q = 0.95$ for p95).

Compute:
1. `e2e`: for each request, the sum over groups of the group's latency. `"p95"` is its p95 and `"meets"` is
   `p95 <= budget`.
2. `"p95_sum"`: the sum over groups of each group's own p95 (what you get if you add up per-stage dashboards).
3. `"alloc"`: split the budget across groups in proportion to each group's p50; every stage gets its group's
   share. Return a dict stage → ms.
4. `"cut"`: `None` if the budget is met. Otherwise the stage whose p95 exceeds its allocation by the most
   (largest `p95 - alloc`), ties to the stage that comes first in `groups`.
"""

STARTER = '''def budget_report(samples, groups, budget):
    """p95 of the pipeline, per-stage allocations and the stage to cut."""
    n = len(next(iter(samples.values())))
    e2e = sorted(sum(samples[s][i] for g in groups for s in g) for i in range(n))
    p95 = e2e[int(0.95 * n)]
    return {"p95": p95, "meets": p95 <= budget, "p95_sum": p95, "alloc": {}, "cut": None}
'''

SOLUTION = '''import math

def pct(xs, q):
    """Nearest-rank percentile."""
    xs = sorted(xs)
    return xs[math.ceil(q * len(xs)) - 1]

def budget_report(samples, groups, budget):
    n = len(next(iter(samples.values())))
    glat = [[max(samples[s][i] for s in g) for i in range(n)] for g in groups]   # per group, per request
    e2e = [sum(col) for col in zip(*glat)]
    p95 = pct(e2e, 0.95)
    p50s = [pct(x, 0.5) for x in glat]
    total = sum(p50s)
    alloc = {s: budget * p / total for g, p in zip(groups, p50s) for s in g}
    cut = None
    if p95 > budget:
        stages = [s for g in groups for s in g]
        cut = max(stages, key=lambda s: (pct(samples[s], 0.95) - alloc[s], -stages.index(s)))
    return {"p95": p95, "meets": p95 <= budget, "p95_sum": sum(pct(x, 0.95) for x in glat),
            "alloc": alloc, "cut": cut}
'''

HINTS = [
    'Build each group\'s latency per request first (`max` over its stages), then the end-to-end latency is the per-request sum over groups.',
    'Nearest rank: `sorted(xs)[math.ceil(q * n) - 1]`. With 20 samples the p95 is the 19th smallest value, not an interpolation.',
    'The allocation uses each group\'s p50; the cut compares each stage\'s own p95 with its group\'s allocation. Break ties with the stage\'s position in the flattened `groups`.',
]

import random as _random
_r = _random.Random(12)


def _lat(n, base, spread, spikes=(), spike=0):
    """Integer ms latencies: base + uniform noise, with a few slow requests."""
    return [base + _r.randrange(spread) + (spike if i in spikes else 0) for i in range(n)]


_N = 40
_RAG = {"guard": _lat(_N, 60, 30, {3, 17}, 150), "retrieve": _lat(_N, 90, 40, {8, 22, 31}, 400),
        "rerank": _lat(_N, 70, 20), "llm_ttft": _lat(_N, 350, 150, {5, 29}, 300)}
_G = [["guard", "retrieve"], ["rerank"], ["llm_ttft"]]
_SEQ = [["guard"], ["retrieve"], ["rerank"], ["llm_ttft"]]
TESTS = [
    (_RAG, _G, 800),
    (_RAG, _SEQ, 800),
    (_RAG, _G, 1200),
    (_RAG, [["guard", "retrieve", "rerank"], ["llm_ttft"]], 700),
    ({"a": [10, 10, 10, 10], "b": [30, 30, 30, 30]}, [["a"], ["b"]], 30),
    ({"a": [100, 100], "b": [100, 100]}, [["a"], ["b"]], 150),
    ({"x": [5, 7, 6, 50, 8, 6, 7, 5, 6, 9], "y": [20, 22, 21, 20, 80, 23, 22, 21, 20, 22]}, [["x"], ["y"]], 60),
]

EXPLANATION = r"""
## Percentiles do not add
The p95 of a sum is not the sum of the p95s. Slow requests at different stages are usually different requests,
so adding per-stage p95s (`p95_sum`) overstates the end-to-end tail; in test 1 the sum of group p95s is well
above the measured p95. The opposite also happens when stages are slow together (shared network, one hot shard).
Always measure the end-to-end distribution from per-request traces; use per-stage numbers only to explain it.

## Parallel stages
A parallel group costs the max of its members. Running the guardrail next to retrieval (test 1) instead of
before it (test 2) removes the guardrail from the path except on requests where it is slower than retrieval.
The price is wasted retrieval work on the requests the guardrail blocks.

## Allocating a budget
A budget per stage turns one SLO into targets teams can own. Splitting by p50 gives each stage room in
proportion to its normal cost; the stage whose tail blows furthest past its share is where tail work (timeouts,
hedged requests, caching, a faster index) pays most. Other rules are fine; the point is that every stage has a
number and the numbers add up to the SLO.

## Common bugs
- Interpolated percentiles (`numpy.percentile` default) instead of nearest rank.
- Summing stages inside a parallel group.
- Computing the end-to-end p95 from sorted per-stage lists, which pairs the slowest retrieval with the slowest
  model call from a different request.

## Interview follow-ups
- Your p95 budget is 800 ms and TTFT alone is 600 ms at p95. Where do you look? (Prompt length and cache hit
  rate first, then a smaller model for the first token, then parallelising everything before the call.)
- How does a 200 ms timeout on retrieval change this table? (It caps that stage's tail but turns the slow
  requests into degraded answers; count those as misses in the quality metrics.)
"""
