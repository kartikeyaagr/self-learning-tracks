META = dict(title='Dictionary spelling corrector', tags=['NLP'], diff='Medium', runtime='python',
            lessons=['sdsd:09.15'], related=['edit-distance', 'n-gram-language-model'])

STATEMENT = r"""
A classic spelling corrector proposes known words that are a small number of edits away from a typo, and prefers the closest, most common one.

Write `correct(words, counts, max_dist=2)` that returns the corrected list.

## Inputs
- `words`: a list of lower-case strings to check.
- `counts`: a dict from known (lower-case) words to their corpus frequency (positive ints).
- `max_dist`: the largest edit distance allowed for a correction.

## Distance
Use the optimal-string-alignment distance with unit costs: insert, delete, substitute and swap two adjacent characters each cost 1, and no substring is edited more than once. So `"teh"` is 1 away from `"the"`.

## Choice, per word
1. If the word is in `counts`, keep it.
2. Otherwise, among the known words at distance $\le$ `max_dist`, choose the one with the **smallest distance**, then the **highest count**, then the smallest string.
3. If there is no such word, keep the original.
"""

STARTER = '''def correct(words, counts, max_dist=2):
    """Return the list of corrected words."""
    raise NotImplementedError
'''

SOLUTION = '''def osa(a, b, limit):
    """Optimal string alignment distance, or limit + 1 once it must exceed limit."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1                         # length gap alone needs that many inserts/deletes
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1                     # every path is already too expensive
        prev2, prev = prev, cur
    return prev[-1]

def correct(words, counts, max_dist=2):
    out = []
    for w in words:
        if w in counts:
            out.append(w)
            continue
        best = None
        for cand, c in counts.items():
            d = osa(w, cand, max_dist)
            if d <= max_dist:
                key = (d, -c, cand)              # closest, then most frequent, then alphabetical
                if best is None or key < best:
                    best = key
        out.append(best[2] if best else w)
    return out
'''

HINTS = [
    'Reuse the edit-distance DP with unit costs and the adjacent-swap case. Then score every dictionary word against the typo.',
    'Rank candidates with a single tuple key `(distance, -count, word)` and take the minimum. Known words short-circuit before any search.',
    'To make it fast, skip words whose length differs by more than `max_dist`, and stop a DP early once every cell of a row exceeds `max_dist`.',
]

_C = {"the": 500, "then": 80, "than": 70, "they": 90, "there": 60, "their": 55, "spelling": 5, "speaking": 9,
      "correct": 12, "correction": 4, "model": 30, "models": 20, "modal": 2, "token": 25, "tokens": 18,
      "language": 15, "a": 300, "an": 120, "and": 400, "of": 450, "on": 200}
TESTS = [
    (["teh", "langauge", "model"], _C),
    (["thn", "tey", "speling"], _C),
    ([], _C),
    (["xyzzy", "qq"], _C, 2),
    (["moddel", "tokn", "tokenz"], _C, 1),
    (["corection", "corrrection", "c"], _C),
    (["thier", "ther", "o", "nd"], _C),
    (["abc"], {"abd": 3, "abe": 3, "xbc": 3}, 1),
]

EXPLANATION = r"""
## Why it works
This is a simple noisy-channel model. Ranking by distance first assumes a typo with one edit is much more likely than one with two. Among equally close words, the more frequent word has the higher prior probability of being what the writer meant. The final alphabetical rule only makes the output deterministic when both are tied (test 8).

## Making the search fast
Comparing against every dictionary word costs $O(|D| \cdot L^2)$. Two cheap cuts help a lot: the length difference is a lower bound on the distance, and DP rows never decrease along a path, so once every cell exceeds the limit no completion can come back under it. For large dictionaries, the usual alternatives are generating all strings within one or two edits and looking them up (Norvig-style), a symmetric delete index (SymSpell), or a BK-tree.

## Complexity
Worst case $O(|D| \cdot L^2)$ per misspelled word with $L$ the word length; much less with the pruning.

## Common bugs
- Preferring frequency over distance, so `teh` becomes `the` (fine) but `tey` becomes `the` (distance 2) instead of `they` (distance 1).
- Correcting words that are already in the dictionary.
- Forgetting transpositions, which are among the most common real typos.

## Interview follow-ups
- How would context improve corrections (`I ate a pear` vs `a pair of shoes`)? (Score candidates with a language model over the surrounding words: real-word errors need context.)
- How would you learn a better error model than unit costs? (Count which substitutions people actually make, e.g. keys that are adjacent on the keyboard.)
"""
