import numpy as np

META = dict(title='Semantic chunking: cut where neighbouring sentences diverge', tags=['NLP', 'Embeddings'], diff='Medium',
            runtime='numpy', lessons=['aieng:11.04'],
            related=['chunk-text-with-overlap', 'sentence-segmentation', 'cosine-similarity', 'rag-pipeline-capstone'])

STATEMENT = r"""
Fixed-size chunking cuts wherever the token count runs out, often mid-topic. **Semantic chunking** embeds each sentence and cuts where consecutive sentences stop being similar, subject to size limits so chunks stay useful.

Write `semantic_chunks(emb, tokens, threshold, min_tokens, max_tokens)`.

- `emb`: float array `(n, d)`, one embedding per sentence, in document order.
- `tokens`: list of `n` positive ints, the token count of each sentence.

## 1. Neighbour similarity
For each boundary $i$ between sentence $i$ and $i+1$, $s_i = \cos(\text{emb}_i, \text{emb}_{i+1})$, or $0$ if either vector has norm 0.

## 2. Greedy walk
Start a chunk at sentence 0. For each next sentence $j = 1, \dots, n-1$, with $c$ the token total of the current chunk:
1. if $c + \text{tokens}_j > \text{max\_tokens}$, cut before $j$ (reason `"size"`);
2. otherwise, if $s_{j-1} < \text{threshold}$ and $c \ge \text{min\_tokens}$, cut before $j$ (reason `"topic"`);
3. otherwise add sentence $j$ to the current chunk.

A single sentence longer than `max_tokens` becomes a chunk on its own (sentences are never split).

## 3. Tidy the tail
If there are at least two chunks, the last one has fewer than `min_tokens` tokens, and merging it into the chunk before keeps that chunk within `max_tokens`, merge them (and drop the cut between them).

## Output
Return `{"spans": [(start, end), ...], "reasons": [...]}`: half-open sentence ranges in order, and one reason per cut (so `len(reasons) == len(spans) - 1`). An empty input gives empty lists. Raise `ValueError` unless `0 <= min_tokens <= max_tokens` and `max_tokens > 0`.
"""

STARTER = '''import numpy as np

def semantic_chunks(emb, tokens, threshold, min_tokens, max_tokens):
    """Cut on size only; topic cuts and the tail merge are missing."""
    spans, reasons, start, cur = [], [], 0, 0
    for j, t in enumerate(tokens):
        if j > start and cur + t > max_tokens:
            spans.append((start, j)); reasons.append("size"); start, cur = j, 0
        cur += t
    if tokens:
        spans.append((start, len(tokens)))
    return {"spans": spans, "reasons": reasons}
'''

SOLUTION = '''import numpy as np

def semantic_chunks(emb, tokens, threshold, min_tokens, max_tokens):
    if max_tokens <= 0 or not 0 <= min_tokens <= max_tokens:
        raise ValueError("need max_tokens > 0 and 0 <= min_tokens <= max_tokens")
    n = len(tokens)
    if n == 0:
        return {"spans": [], "reasons": []}
    e = np.asarray(emb, dtype=np.float64)
    norms = np.linalg.norm(e, axis=1)
    sims = []
    for i in range(n - 1):
        d = norms[i] * norms[i + 1]
        sims.append(float(e[i] @ e[i + 1] / d) if d > 0 else 0.0)

    spans, reasons, start, cur = [], [], 0, tokens[0]
    for j in range(1, n):
        if cur + tokens[j] > max_tokens:
            why = "size"
        elif sims[j - 1] < threshold and cur >= min_tokens:
            why = "topic"
        else:
            cur += tokens[j]
            continue
        spans.append((start, j)); reasons.append(why)
        start, cur = j, tokens[j]
    spans.append((start, n))

    if len(spans) >= 2:
        (a, b), (_, c) = spans[-2], spans[-1]
        last, prev = sum(tokens[b:c]), sum(tokens[a:b])
        if last < min_tokens and prev + last <= max_tokens:
            spans[-2:] = [(a, c)]
            reasons.pop()
    return {"spans": spans, "reasons": reasons}
'''

HINTS = [
    'Compute the n - 1 neighbour similarities first; the walk then only looks up s[j - 1] for the boundary before sentence j.',
    'Check the size rule before the topic rule: a chunk that would overflow must be cut even if the topic continues.',
    "The current total after a cut is the new sentence's own tokens, not 0, or an oversized sentence slips past the size check.",
    'The tail merge removes the last cut, so pop the last reason as well.',
]

# Three topics on orthogonal axes: neighbours in the same topic have cosine 0.94-0.97, across topics exactly 0.
_T = np.array([[4.0, 0, 0, 0], [4, 1, 0, 0], [4, 0, 1, 0],
               [0, 4, 0, 0], [1, 4, 0, 0], [0, 4, 0, 1],
               [0, 0, 4, 0], [0, 0, 4, 1]])
_r = np.random.default_rng(7)
_R = _r.normal(size=(12, 8))
TESTS = [
    (_T, [40, 30, 30, 50, 20, 30, 60, 20], 0.5, 50, 200),
    (_T, [40, 30, 30, 50, 20, 30, 60, 20], 0.5, 50, 90),
    (_T, [40, 30, 30, 50, 20, 30, 60, 20], 0.5, 120, 400),
    (_T, [10, 10, 10, 10, 10, 10, 10, 10], 0.5, 0, 1000),
    (_T[:3], [300, 20, 20], 0.5, 10, 100),
    (np.array([[1.0, 0], [0, 0], [1, 0]]), [5, 5, 5], 0.1, 0, 100),
    (np.zeros((0, 4)), [], 0.5, 10, 100),
    (_T, [40, 30, 30, 50, 20, 30, 30, 10], 0.5, 50, 200),
    (_R, [int(x) for x in _r.integers(20, 80, size=12)], 0.0, 60, 180),
]

EXPLANATION = r"""
## What the threshold means
A cosine drop between neighbours is a cheap topic-shift detector: sentences about the same thing sit close in embedding space. In test 1 the neighbour similarities are 0.94 to 0.97 within a topic and 0 across topics, so the cuts land exactly at sentences 3 and 6. Real tools usually set the threshold per document from the distribution of distances (cut at the 90th or 95th percentile of $1 - s_i$) rather than as a fixed number, because the typical similarity varies between embedding models and documents.

## Why the size limits matter
- **max_tokens** keeps a chunk within the embedding model's input limit and the context budget (test 2: the size rule fires inside a topic).
- **min_tokens** stops a run of short, dissimilar sentences (a list of bullet points, a table split into rows) from producing many tiny chunks that embed poorly (test 3: the drop after sentence 2 is ignored because the chunk has only 100 tokens).
- The tail merge catches the most common leftover: a short final piece stranded on its own (tests 3 and 8).

## Order of the rules
The size rule wins. Checking topic first would let a chunk grow past the limit whenever the topic continues. After a cut the running total is the new sentence's tokens: an oversized sentence (test 5) then forces a size cut before the next sentence.

## Complexity
$O(nd)$ for the similarities and $O(n)$ for the walk. The expensive part in practice is embedding every sentence at ingestion, roughly 10 to 30 times as many embedding calls as chunk-level embedding, which is one reason semantic chunking often loses on cost to recursive splitting.

## Common bugs
- Comparing sentence $j$ with the chunk's first sentence instead of its neighbour (a slow drift never triggers a cut, or every cut is late).
- Dividing by a zero norm (test 6 has an empty sentence embedding).
- Forgetting to drop the reason when merging the tail.

## Interview follow-ups
- Does semantic chunking beat a recursive splitter on your data? Measure recall@k on a labelled question set; published comparisons are mixed and the extra embedding cost is real.
- How would you smooth the signal? Compare windows of 2 or 3 sentences on each side instead of single sentences, which ignores one-off digressions.
"""
