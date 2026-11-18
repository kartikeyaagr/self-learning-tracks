import math

META = dict(title='Choose a rerank depth under a latency budget', tags=['NLP', 'Metrics & Evaluation'], diff='Medium', runtime='python',
            lessons=['aieng:11.03'], related=['precision-recall-at-k', 'mmr-diversity-rerank', 'hit-rate-and-mrr'])

STATEMENT = r"""
A RAG retriever returns a ranked candidate list per query. A cross-encoder reranks the top $N$ of it, and the first `final_k` results go to the prompt. Deeper reranking finds more relevant chunks but costs more latency. Pick $N$.

Write `choose_depth(queries, depths, fixed_ms, batch, batch_ms, budget_ms, final_k=5)` that returns `{"depth": best, "recall": [...], "latency_ms": [...]}`.

Each query is a dict:
- `ranked`: first-stage doc ids, best first (at least `max(depths)` of them),
- `rerank`: `{doc: reranker score}` for every doc in `ranked`,
- `relevant`: the relevant doc ids (never empty; some may be absent from `ranked`).

## For each depth $N$ in `depths`
1. **Rerank:** sort the first $N$ docs of `ranked` by reranker score, highest first; ties keep their first-stage order.
2. **Final list:** the reranked $N$ docs followed by the rest of `ranked` in first-stage order. Take the first `final_k`.
3. **Recall:** $\frac{|\text{final} \cap \text{relevant}|}{|\text{relevant}|}$, averaged over queries.
4. **Latency** (the same for every query, since the model scores pairs in fixed-size GPU batches):
$$\text{latency}(N) = \text{fixed\_ms} + \Big\lceil \tfrac{N}{\text{batch}} \Big\rceil \cdot \text{batch\_ms}$$

## Choose
Among depths with `latency <= budget_ms`, take the one with the highest mean recall; ties go to the **smaller** depth (cheaper). If no depth fits, `depth` is `None`. `recall` and `latency_ms` are floats in the order of `depths`.
"""

STARTER = '''def choose_depth(queries, depths, fixed_ms, batch, batch_ms, budget_ms, final_k=5):
    """Return {"depth": N or None, "recall": [...], "latency_ms": [...]}."""
    recall, lat = [], []
    for n in depths:
        total = 0.0
        for q in queries:
            top = sorted(q["ranked"][:n], key=lambda d: -q["rerank"][d])[:final_k]
            total += len(set(top) & set(q["relevant"])) / len(q["relevant"])
        recall.append(total / len(queries))
        lat.append(fixed_ms + n / batch * batch_ms)
    ok = [i for i in range(len(depths)) if lat[i] <= budget_ms]
    best = max(ok, key=lambda i: recall[i]) if ok else None
    return {"depth": None if best is None else depths[best], "recall": recall, "latency_ms": lat}
'''

SOLUTION = '''import math

def choose_depth(queries, depths, fixed_ms, batch, batch_ms, budget_ms, final_k=5):
    recall, lat = [], []
    for n in depths:
        total = 0.0
        for q in queries:
            head = q["ranked"][:n]
            # sorted() is stable, so equal reranker scores keep first-stage order
            reranked = sorted(head, key=lambda d: -q["rerank"][d])
            final = (reranked + q["ranked"][n:])[:final_k]
            total += len(set(final) & set(q["relevant"])) / len(q["relevant"])
        recall.append(total / len(queries))
        lat.append(float(fixed_ms + math.ceil(n / batch) * batch_ms))
    fits = [i for i in range(len(depths)) if lat[i] <= budget_ms]
    if not fits:
        return {"depth": None, "recall": recall, "latency_ms": lat}
    best = max(fits, key=lambda i: (recall[i], -depths[i]))
    return {"depth": depths[best], "recall": recall, "latency_ms": lat}
'''

HINTS = [
    "Python's `sorted` is stable: sort the head by `-score` and equal scores keep their first-stage order for free.",
    'When N is smaller than `final_k`, the prompt still gets `final_k` chunks: append the un-reranked tail before slicing.',
    'Latency counts whole batches: `math.ceil(N / batch)`. Then choose with the key `(recall, -depth)` among the depths that fit.',
]


def _queries(seed, n_q, n_cand, noise):
    """Simulated candidates: relevant docs sit lower in the first stage than the reranker thinks they should."""
    import random
    r = random.Random(seed)
    qs = []
    for _ in range(n_q):
        ranked = [f"c{j:03d}" for j in range(n_cand)]
        r.shuffle(ranked)
        n_rel = r.choice([1, 2, 4])
        rel = [ranked[int(r.betavariate(1.2, 3.0) * n_cand)] for _ in range(n_rel)]
        rel = list(dict.fromkeys(rel))
        while len(rel) < n_rel:                      # keep |relevant| in {1, 2, 4}
            d = r.choice(ranked)
            if d not in rel:
                rel.append(d)
        if r.random() < 0.3:                         # sometimes a relevant doc was never retrieved
            rel[-1] = "missing-" + rel[-1]
        rerank = {d: round(r.gauss(2.0 if d in rel else 0.0, noise), 3) for d in ranked}
        qs.append({"ranked": ranked, "rerank": rerank, "relevant": rel})
    return qs


_Q1 = {"ranked": ["a", "b", "c", "d", "e", "f", "g", "h"],
       "rerank": {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.0, "f": 0.9, "g": 0.8, "h": 0.0},
       "relevant": ["f", "g"]}
_Q2 = {"ranked": ["p", "q", "r", "s", "t", "u"],
       "rerank": {"p": 1.0, "q": 1.0, "r": 0.5, "s": 0.5, "t": 2.0, "u": 0.0},
       "relevant": ["q", "t"]}

TESTS = [
    ([_Q1], [2, 4, 6, 8], 10.0, 4, 20.0, 50.0, 2),
    ([_Q1, _Q2], [1, 3, 5, 6], 5.0, 2, 7.5, 27.5, 2),
    ([_Q1, _Q2], [2, 4], 30.0, 2, 10.0, 35.0, 1),
    ([_Q2], [1, 2, 3, 4, 5], 0.0, 8, 12.5, 12.5, 1),
    (_queries(7, 40, 200, 1.0), [10, 20, 32, 50, 64, 100, 128, 200], 10.0, 32, 25.0, 120.0, 5),
    (_queries(19, 40, 200, 2.5), [16, 32, 48, 64, 96, 128, 160, 200], 8.0, 16, 9.5, 95.0, 5),
    (_queries(23, 25, 120, 0.5), [5, 10, 20, 40, 80, 120], 12.0, 20, 30.0, 200.0, 3),
]

EXPLANATION = r"""
## The ceiling
Reranking only reorders the first $N$ candidates, so recall@final_k after reranking can never exceed first-stage recall@N. A relevant chunk the first stage put at position 150 is invisible at $N = 100$ however good the reranker is. That is why recall rises with depth, and why the "missing-" relevant docs in the random tests cap recall below 1 at every depth.

## Deeper is not always better
With a noisy reranker (test 6), recall peaks at a middle depth and then falls: each extra candidate is one more chance for an irrelevant chunk to outscore a relevant one. Measure the curve; do not assume it is monotone.

## Batch steps
GPUs score pairs in batches, so latency is a staircase, not a line: with batch 32, depths 33 to 64 cost the same. Depths that fill whole batches give free recall; one doc past a boundary costs a full batch. In test 5, 64 costs no more than 50.

## Choosing
Filter by the budget first, then maximise recall, then prefer the smaller depth on ties (less GPU time or fewer billed tokens for the same quality). Real systems budget p95 latency, which also depends on queueing, so leave headroom.

## Exactness
Relevant-set sizes are 1, 2 or 4, so per-query recalls are exact binary fractions and equal sums are bitwise equal; the smaller-depth tie rule is deterministic. Latency inputs are multiples of 0.5, so the budget comparison is exact too (test 4 sits exactly on the budget).

## Common bugs
- Dropping the tail when $N <$ `final_k`, so fewer chunks reach the prompt.
- An unstable or reversed sort that breaks first-stage order on score ties.
- Linear latency ($N / \text{batch}$) instead of whole batches.
- Choosing the best recall before applying the budget.

## Interview follow-ups
- How would you set the budget? (From the end-to-end p95 target minus the other stages; see the latency budgeting lesson.)
- What changes with a hosted reranker billed per token? (Cost grows linearly with $N$ and query length; add a cost-per-query constraint next to latency.)
"""
