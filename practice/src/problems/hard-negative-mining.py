import numpy as np

META = dict(title='Mine hard negatives and drop likely false negatives', tags=['Embeddings', 'Data Processing'], diff='Medium',
            runtime='numpy', lessons=['aieng:02.10'],
            related=['info-nce-loss', 'clip-loss-hard-negatives', 'ann-recall-at-k', 'uniform-negative-sampling'])

STATEMENT = r"""
You are building fine-tuning data for an embedding model. For every training query you have its labelled positive documents and the current model's similarity score against every document in the corpus. Random documents make useless negatives; the best negatives score high but are not relevant. The danger is that many high-scoring "negatives" are relevant documents nobody labelled, so they must be filtered out.

Write `mine_hard_negatives(scores, positives, num_negatives, range_min=0, range_max=None, relative_margin=None, max_score=None)`.

- `scores`: `(Q, N)` array, `scores[q, d]` is the similarity of query `q` and document `d`.
- `positives`: a list of `Q` lists of document indices, each non-empty.

For each query:

1. **Rank** the non-positive documents by score, highest first, ties by smaller document index. Positions are 0-based.
2. **Window:** keep the documents at positions `range_min` up to but not including `range_max` (`None` means to the end).
3. **Filter** each windowed candidate with score $s$. Let $p$ be the **lowest** score among the query's positives.
   - If `relative_margin` is not `None`, drop the candidate when $s > p - |p| \cdot m$ (it scores too close to a known positive).
   - If `max_score` is not `None`, drop it when $s > $ `max_score`.
4. Take the first `num_negatives` surviving candidates in rank order (fewer if not enough survive).

Return `{"negatives": [list of int indices per query], "dropped": total number of windowed candidates removed by the filters, over all queries}`.
"""

STARTER = '''import numpy as np

def mine_hard_negatives(scores, positives, num_negatives, range_min=0, range_max=None,
                        relative_margin=None, max_score=None):
    """Top-scoring non-positives per query, with a rank window and false-negative filters."""
    S = np.asarray(scores, dtype=float)
    out = []
    for q in range(S.shape[0]):
        order = np.argsort(-S[q])
        out.append([int(d) for d in order if d not in positives[q]][:num_negatives])
    return {"negatives": out, "dropped": 0}
'''

SOLUTION = '''import numpy as np

def mine_hard_negatives(scores, positives, num_negatives, range_min=0, range_max=None,
                        relative_margin=None, max_score=None):
    S = np.asarray(scores, dtype=float)
    negatives, dropped = [], 0
    for q in range(S.shape[0]):
        pos = set(int(d) for d in positives[q])
        order = np.argsort(-S[q], kind="stable")                  # ties keep the smaller index first
        ranked = [int(d) for d in order if int(d) not in pos]
        window = ranked[range_min:range_max]
        p = min(S[q, d] for d in pos)                             # weakest labelled positive
        keep = []
        for d in window:
            s = S[q, d]
            too_close = relative_margin is not None and s > p - abs(p) * relative_margin
            too_high = max_score is not None and s > max_score
            if too_close or too_high:
                dropped += 1
            else:
                keep.append(d)
        negatives.append(keep[:num_negatives])
    return {"negatives": negatives, "dropped": dropped}
'''

HINTS = [
    'Rank with `np.argsort(-row, kind="stable")`: the stable sort keeps the smaller index first among equal scores. Then remove the positives from that order before applying the window, so positions count only candidates.',
    'The margin rule anchors on the weakest positive: compute `p = min(scores[q, d] for d in positives[q])`, then a candidate survives only if `s <= p - abs(p) * relative_margin`.',
    'Count drops over the whole window, not just until you have enough negatives: filter every windowed candidate, then slice the survivors to `num_negatives`.',
]

# Scores are multiples of 1/64 (exact in binary) so every comparison and tie is exact by construction.
_r = np.random.default_rng(2610)
_S = _r.integers(0, 64, size=(4, 12)) / 64.0
_P = [[0], [3, 7], [11], [5]]
_S[1, 3] = 40 / 64.0
_S[1, 7] = 52 / 64.0
_S[2, 4] = _S[2, 9] = 60 / 64.0            # a tie among candidates
_S2 = np.array([[0.82, 0.86, 0.80, 0.77, 0.71, 0.64, 0.52]])
_S3 = np.array([[-0.25, -0.5, -0.125, -0.25, -0.75]])

TESTS = [
    (_S2, [[0]], 3, 0, None, 0.05, None),
    (_S2, [[0]], 3, 0, None, None, None),
    (_S, _P, 3, 0, None, 0.0625, None),
    (_S, _P, 2, 2, 8, None, 0.8125),
    (_S, _P, 4, 1, None, 0.125, 0.875),
    (_S3, [[0]], 2, 0, None, 0.5, None),
    (_S, _P, 10, 0, 3, None, None),
    (np.array([[0.5, 0.5, 0.5, 0.5]]), [[2]], 2, 0, None, None, None),
]

EXPLANATION = r"""
## Why hard negatives
Contrastive training scores each query against its positive and a set of negatives. In-batch negatives are other queries' positives, almost always on a different topic, so the loss on them is near zero and they teach little. A mined negative from the top of the current model's ranking shares the query's vocabulary and topic, which is exactly the confusion the model must learn to resolve.

## Why filter
The top of the ranking is also where the unlabelled positives are. Labelled datasets mark one or two relevant documents per query; duplicates, other language versions and partly relevant passages go unmarked. Training on them as negatives pushes the model away from right answers. Three standard defences, all here:

- **Rank window** (`range_min`): skip the very top, which is most likely to hold false negatives.
- **Positive-aware margin** (`relative_margin`): NV-Retriever's finding was that the positive's own score is the best anchor; a negative scoring above, say, 95% of it ($m = 0.05$) is probably relevant. Using the weakest positive when there are several is the conservative choice. The $|p|$ keeps the rule sensible for negative scores (test 6).
- **Absolute ceiling** (`max_score`): useful with a calibrated cross-encoder score.

## Complexity
$O(QN \log N)$ for the sorts. At corpus scale you replace the full sort with an ANN top-k search (a few hundred candidates per query), which is why the window has an upper bound.

## Common bugs
- Applying the window before removing positives, so a positive at rank 0 shifts every position.
- Unstable sorting, which reorders tied candidates between runs.
- Anchoring the margin on the best positive, which lets through candidates that outscore a known positive.

## Interview follow-ups
- How would you choose the margin? (Sweep it and measure recall@k of the fine-tuned model on a held-out set; too small keeps false negatives, too large leaves only easy negatives.)
- Why use a cross-encoder for filtering instead of the bi-encoder being trained? (It is a stronger relevance judge and is not biased by the model's own mistakes.)
"""
