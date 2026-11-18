META = dict(title='Weighted score fusion: normalise, combine and tune alpha', tags=['NLP', 'Metrics & Evaluation'], diff='Medium', runtime='python',
            lessons=['aieng:11.02'], related=['document-ranking', 'precision-recall-at-k', 'ndcg-at-k'])

STATEMENT = r"""
A hybrid retriever returns two scored lists per query: lexical (BM25) and dense (cosine). Their scales differ, so you normalise each list, mix them with one weight $\alpha$, and choose $\alpha$ on labelled queries.

Write `tune_alpha(queries, alphas, norm="minmax", k=3)` that returns `{"alpha": best, "recall": curve}`.

Each query is a dict `{"lex": {doc: score}, "dense": {doc: score}, "relevant": [doc, ...]}` (doc ids are strings; `relevant` is never empty). The candidates of a query are the union of the keys of `lex` and `dense`.

## 1. Normalise each list on its own
- `"minmax"`: $\tilde s = \frac{s - \min}{\max - \min}$ over that list. If $\max = \min$, every document in the list gets `1.0`.
- `"zscore"`: $\tilde s = \frac{s - \mu}{\sigma}$ with the **population** standard deviation $\sigma = \sqrt{\frac{1}{n}\sum (s - \mu)^2}$. If $\sigma = 0$, every document in the list gets `0.0`.

A document missing from a list gets `0.0` from it under `"minmax"`, and that list's **lowest** z-score under `"zscore"` (it ranks with the worst returned document, not at the average). If the list is empty, every document gets `0.0` from it.

## 2. Fuse
For each $\alpha$, compute exactly `alpha * dense_norm + (1 - alpha) * lex_norm` per candidate. Rank candidates by fused score, highest first, ties by doc id (string order). Take the top `k`.

## 3. Pick alpha
recall@k of a query is $\frac{|\text{top-}k \cap \text{relevant}|}{|\text{relevant}|}$. `recall` is the list of mean recall@k over all queries, one value per entry of `alphas`, in the same order. `alpha` is the entry with the highest mean recall; ties go to the one that comes first in `alphas`.
"""

STARTER = '''def tune_alpha(queries, alphas, norm="minmax", k=3):
    """Return {"alpha": best alpha, "recall": mean recall@k per alpha}."""
    curve = []
    for a in alphas:
        total = 0.0
        for q in queries:
            docs = set(q["lex"]) | set(q["dense"])
            fused = {d: a * q["dense"].get(d, 0.0) + (1 - a) * q["lex"].get(d, 0.0) for d in docs}
            top = sorted(docs, key=lambda d: (-fused[d], d))[:k]
            total += len(set(top) & set(q["relevant"])) / len(q["relevant"])
        curve.append(total / len(queries))
    return {"alpha": alphas[curve.index(max(curve))], "recall": curve}
'''

SOLUTION = '''import math

def normalise(scores, norm):
    """Per-list normalisation; returns (normalised dict, value for missing documents)."""
    if not scores:
        return {}, 0.0
    vals = list(scores.values())
    if norm == "minmax":
        lo, hi = min(vals), max(vals)
        out = {d: 1.0 if hi == lo else (s - lo) / (hi - lo) for d, s in scores.items()}
    else:
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        out = {d: 0.0 if sd == 0 else (s - mu) / sd for d, s in scores.items()}
    return out, 0.0 if norm == "minmax" else min(out.values())   # missing = worst, never the average

def tune_alpha(queries, alphas, norm="minmax", k=3):
    prepared = []                          # normalise once; alpha only changes the mix
    for q in queries:
        lex, lex_miss = normalise(q["lex"], norm)
        den, den_miss = normalise(q["dense"], norm)
        docs = sorted(set(q["lex"]) | set(q["dense"]))
        prepared.append(([(d, lex.get(d, lex_miss), den.get(d, den_miss)) for d in docs], set(q["relevant"])))
    curve = []
    for a in alphas:
        total = 0.0
        for rows, rel in prepared:
            fused = sorted(((-(a * dn + (1 - a) * lx), d) for d, lx, dn in rows))
            top = {d for _, d in fused[:k]}
            total += len(top & rel) / len(rel)
        curve.append(total / len(prepared))
    best = max(range(len(alphas)), key=lambda i: (curve[i], -i))
    return {"alpha": alphas[best], "recall": curve}
'''

HINTS = [
    'Normalise each list once per query, before looping over alpha. Alpha only changes how the two normalised numbers are mixed.',
    'For a document a list did not return, use the minimum normalised score of that list. Under z-score that minimum is negative; using 0.0 would rank an unseen document as "average".',
    'Sort by the key `(-fused, doc_id)` to get highest first with string ties. For the best alpha, keep the first index that reaches the maximum.',
]

_A11 = [i / 10 for i in range(11)]

# Two query types: id lookups (lexical finds it, dense blurs it) and paraphrases (the reverse).
_Q_ID = {"lex": {"d1": 18.0, "d2": 6.0, "d3": 5.5, "d4": 5.0},
         "dense": {"d5": 0.83, "d3": 0.82, "d6": 0.81, "d1": 0.70}, "relevant": ["d1"]}
_Q_PARA = {"lex": {"d2": 9.0, "d7": 8.0, "d8": 2.0},
           "dense": {"d8": 0.88, "d9": 0.86, "d2": 0.70, "d7": 0.69}, "relevant": ["d8", "d9"]}
_Q_BOTH = {"lex": {"a": 7.0, "b": 6.5, "c": 1.0}, "dense": {"b": 0.9, "a": 0.6, "d": 0.5},
           "relevant": ["a", "b"]}


def _rand_queries(seed, n_q, n_docs):
    import random
    r = random.Random(seed)
    qs = []
    for _ in range(n_q):
        pool = [f"x{j:02d}" for j in range(n_docs)]
        rel = r.sample(pool, r.choice([1, 2, 4]))
        lex = {d: round(r.uniform(0, 12) + (6 if d in rel and r.random() < 0.6 else 0), 3) for d in r.sample(pool, 8)}
        den = {d: round(r.uniform(0.5, 0.8) + (0.12 if d in rel and r.random() < 0.6 else 0), 4) for d in r.sample(pool, 8)}
        qs.append({"lex": lex, "dense": den, "relevant": rel})
    return qs


TESTS = [
    ([_Q_ID, _Q_PARA], _A11, "minmax", 1),
    ([_Q_ID, _Q_PARA, _Q_BOTH], [0.0, 0.25, 0.5, 0.75, 1.0], "minmax", 2),
    ([_Q_ID, _Q_PARA, _Q_BOTH], _A11, "zscore", 2),
    ([{"lex": {}, "dense": {"p": 0.5, "q": 0.9}, "relevant": ["p"]}], [0.0, 1.0], "minmax", 1),
    ([{"lex": {"p": 3.0, "q": 3.0}, "dense": {"q": 0.4, "r": 0.2}, "relevant": ["q"]}], [0.0, 0.5, 1.0], "zscore", 1),
    ([{"lex": {"p": 3.0, "q": 3.0}, "dense": {"r": 0.4}, "relevant": ["r"]}], [0.0, 0.5, 1.0], "minmax", 2),
    (_rand_queries(11, 12, 20), _A11, "minmax", 3),
    (_rand_queries(29, 12, 20), _A11, "zscore", 5),
]

EXPLANATION = r"""
## Why normalise per list
BM25 scores are unbounded and grow with query length; cosine scores cluster in a narrow band. Mixing raw scores lets BM25 decide everything. Min-max puts each list on $[0, 1]$; z-score centres it on the list mean. Both use only the documents a retriever returned, which is why the rule for missing documents matters.

## Missing documents
A document the dense retriever did not return is, as far as we know, worse than everything it did return. Giving it the list minimum encodes that: 0.0 under min-max, the lowest z-score under z-score. Using 0.0 under z-score would score it as average and quietly promote it above every below-average document the retriever did return. Min-max has its own artefact: the last returned item gets exactly 0, the same as an unseen one, and a list whose scores are all equal (test 6) gives every returned document 1.0.

## Choosing alpha
The grid search reuses the normalised scores: alpha changes only the mix, so the work is $O(|\alpha| \cdot \sum_q n_q \log n_q)$. Test 1 pits an identifier query (needs low alpha) against a paraphrase query (needs high alpha); with k = 1 no single alpha serves both. In test 2 a third query makes the middle alpha best. In test 3 (z-score) scoring missing documents as 0.0 instead of the list minimum would move the best alpha from 0.5 to 0.3. In practice, read per-slice curves: if the best alphas for two query types are far apart, route by query type instead of using one value.

## Exactness
Relevant-set sizes in the tests are 1, 2 or 4, so every recall value is a sum of exact binary fractions and equal curves compare equal; the first-in-list tie rule is then deterministic.

## Common bugs
- Normalising over the union of documents instead of each list on its own.
- Treating a missing document as 0.0 under z-score.
- Using the sample standard deviation ($n - 1$).
- Breaking fused-score ties by insertion order instead of doc id.

## Interview follow-ups
- How many labelled queries do you need to tune one parameter? (Bruch et al. found a small sample suffices; confirm on a held-out split.)
- When would you keep RRF instead? (No labels, unstable score distributions, or a tuned alpha that does not beat RRF on held-out queries.)
"""
