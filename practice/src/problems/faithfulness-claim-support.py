META = dict(title='Faithfulness and citation scores from a claim support matrix', tags=['Metrics & Evaluation', 'Language Models'], diff='Medium', runtime='python',
            lessons=['aieng:10.04'], related=['validate-citations', 'rag-context-packing', 'context-precision-recall'])

STATEMENT = r"""
An LLM judge has split each RAG answer into claims and labelled every (claim, retrieved chunk) pair. Turn those labels into faithfulness and citation scores. Write `score_answers(answers)`.

Each answer is a dict:
- `retrieved`: the chunk ids that were in the prompt.
- `support`: one dict per claim, mapping chunk id to `"supports"` or `"contradicts"`. A retrieved chunk that is missing from the dict is neutral for that claim. Keys are always retrieved ids.
- `citations`: one list per claim of the chunk ids the answer cited for it. A list may repeat an id (count it once) and may name ids that were never retrieved.

**Claim status**, from the labels over all retrieved chunks:

| status | rule |
| --- | --- |
| `"supported"` | at least one `"supports"` and no `"contradicts"` |
| `"conflicted"` | at least one of each (for example an old and a new policy) |
| `"contradicted"` | at least one `"contradicts"` and no `"supports"` |
| `"unsupported"` | no labels |

**Per answer**, with $n$ claims and the citation pairs (claim, distinct cited id):
- `faithfulness`: supported claims / $n$.
- `citation_recall`: claims for which at least one cited id is labelled `"supports"` / $n$.
- `citation_precision`: citation pairs whose id is labelled `"supports"` for that claim / number of citation pairs.
- `unknown_citations`: citation pairs whose id is not in `retrieved`.

An answer with no claims is an **abstention**: its three ratios are `None`. `citation_precision` is also `None` when an answer makes no citations.

Return `{"per_answer": [{"statuses": [...], "faithfulness": ..., "citation_recall": ..., "citation_precision": ..., "unknown_citations": int}, ...], "mean_faithfulness": mean over non-abstaining answers or None if there are none, "abstentions": count}`.
"""

STARTER = '''def score_answers(answers):
    """Faithfulness and citation scores for each answer."""
    per = []
    for a in answers:
        st = ["supported" if "supports" in s.values() else "unsupported" for s in a["support"]]
        n = len(st)
        f = st.count("supported") / n if n else 1.0
        per.append({"statuses": st, "faithfulness": f, "citation_recall": f,
                    "citation_precision": f, "unknown_citations": 0})
    mean = sum(p["faithfulness"] for p in per) / len(per)
    return {"per_answer": per, "mean_faithfulness": mean, "abstentions": 0}
'''

SOLUTION = '''def status(labels):
    vals = set(labels.values())
    if "supports" in vals:
        return "conflicted" if "contradicts" in vals else "supported"
    return "contradicted" if "contradicts" in vals else "unsupported"

def score_one(a):
    n = len(a["support"])
    statuses = [status(s) for s in a["support"]]
    retrieved = set(a["retrieved"])
    pairs = good = unknown = recalled = 0
    for labels, cited in zip(a["support"], a["citations"]):
        ids = list(dict.fromkeys(cited))            # distinct, first occurrence kept
        ok = [labels.get(c) == "supports" for c in ids]
        pairs += len(ids)
        good += sum(ok)
        unknown += sum(c not in retrieved for c in ids)
        recalled += any(ok)
    return {"statuses": statuses,
            "faithfulness": statuses.count("supported") / n if n else None,
            "citation_recall": recalled / n if n else None,
            "citation_precision": good / pairs if n and pairs else None,
            "unknown_citations": unknown}

def score_answers(answers):
    per = [score_one(a) for a in answers]
    f = [p["faithfulness"] for p in per if p["faithfulness"] is not None]
    return {"per_answer": per, "mean_faithfulness": sum(f) / len(f) if f else None,
            "abstentions": len(per) - len(f)}
'''

HINTS = [
    'Decide each claim\'s status from the set of label values it has: "supports" and "contradicts" together is a conflict, not support.',
    'Deduplicate each claim\'s citations with `dict.fromkeys(cited)` before counting pairs, and look each id up in that claim\'s label dict: a citation that supports a different claim does not count.',
    'An answer with no claims (an abstention) has undefined ratios. Return None and leave it out of the mean instead of scoring it 0 or 1.',
]

import random as _random

def _gen(seed, n_answers, pool=6):
    r = _random.Random(seed)
    out = []
    for _ in range(n_answers):
        retrieved = [f"c{j}" for j in r.sample(range(pool), r.randint(2, 5))]
        n = 0 if r.random() < 0.15 else r.randint(1, 5)
        support, citations = [], []
        for _ in range(n):
            labels = {}
            for c in retrieved:
                x = r.random()
                if x < 0.3:
                    labels[c] = "supports"
                elif x < 0.4:
                    labels[c] = "contradicts"
            support.append(labels)
            cited = [r.choice(retrieved + ["c9"]) for _ in range(r.randint(0, 3))]
            citations.append(cited)
        out.append({"retrieved": retrieved, "support": support, "citations": citations})
    return out

_LESSON = {"retrieved": ["P7", "P8", "X", "Y", "Z"],
           "support": [{"P7": "supports"}, {"P8": "supports"}, {}, {"Y": "supports"}, {"P8": "contradicts"}],
           "citations": [["P7"], ["P7"], [], ["Y"], ["P8"]]}

TESTS = [
    ([_LESSON],),
    ([{"retrieved": ["a", "b"], "support": [{"a": "supports", "b": "contradicts"}, {"b": "supports"}],
       "citations": [["a", "a", "b"], ["b", "q"]]}],),
    ([{"retrieved": ["a"], "support": [], "citations": []}, _LESSON],),
    ([{"retrieved": ["a"], "support": [], "citations": []}],),
    ([{"retrieved": ["a", "b"], "support": [{"a": "supports"}, {}], "citations": [[], []]}],),
    (_gen(3, 8),),
    (_gen(21, 15),),
]

EXPLANATION = r"""
## The arithmetic is easy; the labels are the eval
Faithfulness in Ragas, FActScore's atomic-fact precision and ALCE's citation scores all reduce to counting over a claim-by-chunk support matrix. The expensive and error-prone part is producing that matrix: splitting the answer into claims and labelling each pair with a judge. That is why the judge has to be checked against human labels before the numbers mean anything.

## Why conflicted is its own status
When one retrieved chunk supports a claim and another contradicts it, the corpus disagrees with itself, usually an outdated document next to a current one. Counting the claim as supported hides an indexing bug that a user will eventually hit. Reporting it separately points at freshness and deduplication.

## Citation precision versus recall
Recall asks whether each claim is backed by something it cites. Precision asks whether each citation backs the claim it is attached to. In the first test, claim a2 is supported by P8 but cites P7: faithful, but the user who clicks the citation finds nothing. Citation ids that were never retrieved (test 2) are fabricated and should be zero in a healthy system.

## Abstentions
An answer that says "I could not find this" has no claims. Treating its faithfulness as 1.0 rewards refusing; as 0.0 punishes correct refusals on unanswerable questions. Leave it out of the mean and score it with an abstention metric.

## Interview follow-ups
- Faithfulness is 0.95 but answers are often wrong. How? (Faithful to stale or wrong context, incomplete answers, judge error; add reference-based correctness.)
- How would you make the claim split stable across re-runs? (Pinned judge snapshot, temperature 0, cached splits.)
"""
