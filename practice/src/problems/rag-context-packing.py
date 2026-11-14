META = dict(title='Pack retrieved chunks into a context with permissions', tags=['NLP', 'Language Models'], diff='Medium', runtime='python',
            related=['chunk-text-with-overlap', 'document-ranking'])

STATEMENT = r"""
Write `pack_context(chunks, user_groups, budget, max_per_doc)` for a retrieval-augmented assistant.

Each chunk is a dict with `id`, `doc`, `score`, `tokens` (an int), `acl` (a list of groups allowed to read it) and `text`.

1. Drop chunks the user may not read: keep a chunk only if `acl` shares at least one group with `user_groups`. An empty `acl` means nobody.
2. Drop exact duplicate texts, keeping the higher-scoring copy (ties to the earlier one in the input).
3. Go through the rest by score descending (ties by input order). Add a chunk if it fits in the remaining token `budget` and its document has fewer than `max_per_doc` chunks already; otherwise skip it and keep going.

Return `{"ids": [chosen ids in the order chosen], "tokens": total, "dropped_acl": number removed in step 1}`.
"""

STARTER = '''def pack_context(chunks, user_groups, budget, max_per_doc):
    """Choose chunks for the prompt."""
    chosen, used = [], 0
    for c in sorted(chunks, key=lambda c: -c["score"]):
        if used + c["tokens"] <= budget:
            chosen.append(c["id"]); used += c["tokens"]
    return {"ids": chosen, "tokens": used, "dropped_acl": 0}
'''

SOLUTION = '''def pack_context(chunks, user_groups, budget, max_per_doc):
    groups = set(user_groups)
    allowed = [(k, c) for k, c in enumerate(chunks) if groups & set(c["acl"])]
    dropped = len(chunks) - len(allowed)
    order = sorted(allowed, key=lambda kc: (-kc[1]["score"], kc[0]))
    seen_text, per_doc, ids, used = set(), {}, [], 0
    for _, c in order:
        if c["text"] in seen_text:
            continue
        seen_text.add(c["text"])
        if used + c["tokens"] > budget or per_doc.get(c["doc"], 0) >= max_per_doc:
            continue
        ids.append(c["id"]); used += c["tokens"]
        per_doc[c["doc"]] = per_doc.get(c["doc"], 0) + 1
    return {"ids": ids, "tokens": used, "dropped_acl": dropped}
'''

HINTS = [
    'Filter by permission before anything else: a chunk the user cannot read must not influence the answer, even as a duplicate.',
    'Scanning in score order and marking texts as seen keeps the best copy of each duplicate automatically.',
    'Skip a chunk that does not fit but keep scanning; a smaller chunk further down may still fit.',
]

def _c(i, doc, score, tok, acl, text=None):
    return {"id": i, "doc": doc, "score": score, "tokens": tok, "acl": acl, "text": text or f"text-{i}"}

_CH = [_c("a1", "A", 0.9, 300, ["eng"]), _c("a2", "A", 0.85, 200, ["eng"]), _c("a3", "A", 0.8, 100, ["eng"]),
       _c("h1", "HR", 0.95, 150, ["hr"]), _c("b1", "B", 0.7, 400, ["all"]), _c("b2", "B", 0.6, 50, ["all"], "text-a2"),
       _c("c1", "C", 0.5, 120, []), _c("d1", "D", 0.4, 80, ["eng", "all"])]
TESTS = [
    (_CH, ["eng", "all"], 700, 2),
    (_CH, ["eng", "all"], 1000, 1),
    (_CH, ["all"], 500, 3),
    (_CH, ["hr"], 100, 2),
    (_CH, [], 1000, 5),
    (_CH, ["eng", "hr", "all"], 260, 2),
]

EXPLANATION = r"""
## Permissions first
In an enterprise assistant the retriever indexes documents with different readers. Filtering after generation is too late: once an HR document is in the prompt, its content can leak into the answer. Access control has to be enforced on the retrieved set, using the same ACLs as the source system.

## Why dedupe and cap per document
Overlapping chunks and copied text are common. Duplicates waste budget, and many chunks from one document crowd out other sources, which hurts answers that need to combine information.

## Greedy packing
This is a knapsack problem, but greedy by score with skip-and-continue is close to optimal when chunks are similar in size, and it is easy to reason about when debugging a bad answer.

## Interview follow-ups
- How would you keep the ACLs in the index in sync when a document's permissions change?
"""
