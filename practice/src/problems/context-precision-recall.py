META = dict(title='RAG retrieval metrics from claim-level evidence labels', tags=['Metrics & Evaluation', 'NLP'], diff='Medium', runtime='python',
            lessons=['aieng:10.04', 'mlsd:05.02'], related=['precision-recall-at-k', 'hit-rate-and-mrr', 'ndcg-at-k'])

STATEMENT = r"""
A RAG eval set stores, for each question, the reference answer split into **claims**, and for each claim the chunk ids that support it (any one of them is enough). Write `retrieval_metrics(queries, k)` that scores a retriever on such a set.

Each query is a dict:
- `ranked`: the retrieved chunk ids, best first, no duplicates (may be shorter than `k`).
- `claims`: a list with one entry per reference claim; each entry is a list of supporting chunk ids. An empty `claims` list marks an **unanswerable** question.

Only the top `k` of `ranked` count. Let the relevant set $R$ be the union of all claims' supporting ids, and $v_i = 1$ if the chunk at rank $i$ is in $R$. For each answerable query compute:

| metric | definition |
| --- | --- |
| `precision` | relevant chunks in the top $k$, divided by $k$ |
| `chunk_recall` | relevant chunks in the top $k$, divided by $|R|$ |
| `claim_recall` | claims with at least one supporting id in the top $k$, divided by the number of claims |
| `context_precision` | $\sum_{i=1}^{k} \text{P@}i \cdot v_i \,/\, \sum_{i=1}^{k} v_i$, where $\text{P@}i$ is the share of relevant chunks in ranks $1..i$; 0.0 if no relevant chunk is in the top $k$ |
| `rr` | $1/i$ for the first relevant rank $i \le k$, else 0.0 |

Every answerable query has at least one supporting id per claim. Return

```
{"per_query": [[precision, chunk_recall, claim_recall, context_precision, rr] or None, ...],
 "mean": {"precision": ..., "chunk_recall": ..., "claim_recall": ..., "context_precision": ..., "mrr": ...},
 "unanswerable": count}
```

`per_query` has one entry per input query, `None` for unanswerable ones. `mean` averages over answerable queries only (the tests always contain at least one).
"""

STARTER = '''def retrieval_metrics(queries, k):
    """Score retrieval against claim-level evidence labels."""
    per = []
    for q in queries:
        rel = {c for claim in q["claims"] for c in claim}
        hits = sum(1 for c in q["ranked"][:k] if c in rel)
        recall = hits / len(rel) if rel else 0.0
        per.append([hits / k, recall, recall, recall, 1.0 if hits else 0.0])
    names = ["precision", "chunk_recall", "claim_recall", "context_precision", "mrr"]
    mean = {n: sum(p[i] for p in per) / len(per) for i, n in enumerate(names)}
    return {"per_query": per, "mean": mean, "unanswerable": 0}
'''

SOLUTION = '''def score_query(ranked, claims, k):
    top = ranked[:k]
    rel = {c for claim in claims for c in claim}
    hits, ap_sum, rr = 0, 0.0, 0.0
    for i, c in enumerate(top, start=1):
        if c in rel:
            hits += 1
            ap_sum += hits / i          # precision@i, added only at relevant ranks
            if rr == 0.0:
                rr = 1.0 / i
    top_set = set(top)
    covered = sum(1 for claim in claims if top_set & set(claim))
    return [hits / k, hits / len(rel), covered / len(claims), ap_sum / hits if hits else 0.0, rr]

def retrieval_metrics(queries, k):
    per = [score_query(q["ranked"], q["claims"], k) if q["claims"] else None for q in queries]
    rows = [p for p in per if p is not None]
    names = ["precision", "chunk_recall", "claim_recall", "context_precision", "mrr"]
    mean = {n: sum(r[i] for r in rows) / len(rows) for i, n in enumerate(names)}
    return {"per_query": per, "mean": mean, "unanswerable": len(per) - len(rows)}
'''

HINTS = [
    'Skip unanswerable queries (empty `claims`) before averaging: they have no relevant set, so recall is undefined, not 0 or 1.',
    'Claim recall asks a different question from chunk recall: a claim is covered if any one of its supporting ids is in the top k, so duplicate evidence for one claim does not help.',
    'Context precision is average precision cut at k: walk the top k, keep a running hit count, and add `hits / i` only at relevant ranks; divide by the number of hits.',
]

import random as _random

def _gen(seed, nq, pool=12, k=5):
    r = _random.Random(seed)
    qs = []
    for _ in range(nq):
        ids = [f"c{j}" for j in range(pool)]
        r.shuffle(ids)
        ranked = ids[:r.randint(2, 8)]
        claims = []
        if r.random() > 0.2:
            for _ in range(r.randint(1, 4)):
                claims.append(r.sample([f"c{j}" for j in range(pool)], r.randint(1, 3)))
        qs.append({"ranked": ranked, "claims": claims})
    return qs

_LESSON = [{"ranked": ["P7", "X", "P8", "Y", "Z"], "claims": [["P7", "P8"], ["E2"]]},
           {"ranked": ["X", "P7", "Y", "Z", "E2"], "claims": [["P7", "P8"], ["E2"]]}]

TESTS = [
    (_LESSON, 5),
    ([{"ranked": ["a", "b", "c"], "claims": [["c"]]}, {"ranked": ["d", "e"], "claims": []}], 3),
    ([{"ranked": ["a", "b"], "claims": [["a"], ["b"], ["z"]]}], 5),
    ([{"ranked": ["x", "y", "a"], "claims": [["a"]]}, {"ranked": ["a", "x"], "claims": [["a", "b"], ["b"]]}], 2),
    (_LESSON + [{"ranked": ["Q"], "claims": []}], 3),
    (_gen(7, 12), 5),
    (_gen(11, 20), 3),
]

EXPLANATION = r"""
## Why claim-level labels
A question whose answer needs two facts is only answerable if both reach the prompt. Chunk recall counts chunks, so retrieving the same fact from the policy page and the FAQ counts twice while the second fact is missing. Claim recall (what Ragas calls context recall, computed here from labels instead of a judge) counts what the generator actually needs. In the first test, ranking A wins on MRR and context precision but covers only half the claims; ranking B covers both.

## Context precision is truncated average precision
$\text{P@}i$ is added only at ranks holding a relevant chunk, then divided by the number of relevant chunks retrieved. Relevant chunks near the top raise it; noise above them lowers it. Dividing by hits rather than by $|R|$ means it measures ordering, not coverage, which is why it must be reported next to a recall metric.

## Unanswerable questions
They have no relevant set, so recall and precision are undefined. Averaging them in as 0 punishes every retriever equally and hides real differences; as 1 it inflates scores. Score them separately with an abstention metric on the generated answer.

## Complexity
$O(k + \sum |\text{claim}|)$ per query with sets.

## Interview follow-ups
- Your labels are chunk ids. What breaks when you change the chunk size? (Store evidence as document spans and map spans to chunks at eval time.)
- How would you get labels without reading the whole corpus? (Pool the top results of several retrievers and label the union.)
"""
