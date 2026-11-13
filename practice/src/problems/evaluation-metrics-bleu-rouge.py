import math

META = dict(title='Sentence BLEU and ROUGE-L', tags=['Evaluation Metrics', 'NLP'], diff='Medium', runtime='python',
            lessons=['sdsd:09.09'], related=['edit-distance', 'beam-search-decoding', 'perplexity-computation'])

STATEMENT = r"""
BLEU (precision of n-grams, for translation) and ROUGE-L (longest common subsequence, for summarization) compare a generated token list with one or more references. Write `bleu_rouge(candidate, references, max_n=4, smooth=False)` that returns a dict with keys `"precisions"`, `"bleu"` and `"rouge_l"`.

## Inputs
- `candidate`: list of token strings (may be empty).
- `references`: non-empty list of non-empty token lists.

## Modified n-gram precision, for n = 1 .. max_n
- $m_n = \sum_g \min\big(\text{count}_{\text{cand}}(g),\ \max_{\text{ref}} \text{count}_{\text{ref}}(g)\big)$ over the distinct n-grams $g$ of the candidate (clipping stops `the the the` from matching one `the` three times).
- $c_n = \max(\text{len(candidate)} - n + 1, 0)$.
- $p_n = m_n / c_n$, or 0 if $c_n = 0$. With `smooth=True`, for $n \ge 2$ use $p_n = (m_n + 1) / (c_n + 1)$ instead.

## BLEU
- If the candidate is empty or any $p_n = 0$, BLEU is 0.
- Otherwise, with $c$ the candidate length and $r$ the reference length closest to $c$ (ties go to the shorter reference), the brevity penalty is $\mathrm{BP} = 1$ if $c > r$, else $e^{1 - r/c}$, and
$$\mathrm{BLEU} = \mathrm{BP} \cdot \exp\Big(\frac{1}{N} \sum_{n=1}^{N} \ln p_n\Big), \quad N = \text{max\_n}.$$

## ROUGE-L
For each reference, let $L$ be the length of the longest common subsequence with the candidate, $P = L / c$, $R = L / |\text{ref}|$, and $F = 2PR / (P + R)$ (or 0 when $L = 0$). `rouge_l` is the largest $F$ over the references, 0 for an empty candidate.

`precisions` is the list $[p_1, \dots, p_N]$ as floats (all 0.0 for an empty candidate).
"""

STARTER = '''import math

def bleu_rouge(candidate, references, max_n=4, smooth=False):
    """Return {"precisions": [...], "bleu": float, "rouge_l": float}."""
    raise NotImplementedError
'''

SOLUTION = '''import math
from collections import Counter

def ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))

def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]

def bleu_rouge(candidate, references, max_n=4, smooth=False):
    c = len(candidate)
    if c == 0:
        return {"precisions": [0.0] * max_n, "bleu": 0.0, "rouge_l": 0.0}
    precisions = []
    for n in range(1, max_n + 1):
        cand = ngrams(candidate, n)
        best_ref = Counter()
        for ref in references:
            best_ref |= ngrams(ref, n)            # | keeps the max count per n-gram
        m = sum(min(k, best_ref[g]) for g, k in cand.items())
        total = max(c - n + 1, 0)
        if smooth and n >= 2:
            precisions.append((m + 1) / (total + 1))
        else:
            precisions.append(m / total if total else 0.0)
    if min(precisions) == 0:
        bleu = 0.0
    else:
        r = min((abs(len(ref) - c), len(ref)) for ref in references)[1]   # closest, then shorter
        bp = 1.0 if c > r else math.exp(1 - r / c)
        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    rouge = 0.0
    for ref in references:
        L = lcs(candidate, ref)
        if L:
            P, R = L / c, L / len(ref)
            rouge = max(rouge, 2 * P * R / (P + R))
    return {"precisions": precisions, "bleu": bleu, "rouge_l": rouge}
'''

HINTS = [
    'Use `collections.Counter` of n-gram tuples. For clipping, the reference count of an n-gram is its **maximum** over references; `Counter | Counter` computes exactly that.',
    'Pick the reference length with `min((abs(len(ref) - c), len(ref)) for ref in references)`: the tuple makes ties go to the shorter reference. The penalty only applies when the candidate is not longer than that reference.',
    'LCS is the classic $O(|a||b|)$ DP: `cur[j+1] = prev[j] + 1` on a match, else `max(prev[j+1], cur[j])`. Keep one previous row.',
]

_ref1 = "the cat is on the mat".split()
_ref2 = "there is a cat on the mat".split()
TESTS = [
    ("the cat is on the mat".split(), [_ref1]),
    ("the the the the the the the".split(), [_ref1, _ref2], 2),
    ([], [_ref1]),
    ("the cat".split(), [_ref1, _ref2]),
    ("the cat".split(), [_ref1, _ref2], 4, True),
    ("a cat is on the mat today".split(), [_ref1, _ref2], 4),
    ("on the mat the cat sat".split(), [_ref1, _ref2], 3, True),
    ("x y z".split(), [["x", "y"], ["a", "b", "c", "d"]], 2),
]

EXPLANATION = r"""
## Why the pieces exist
- **Clipping** caps each n-gram's credit at the number of times a reference uses it; without it, test 2 (`the` seven times) would have unigram precision 1. Clipped, it gets $2/7$.
- **Geometric mean over n** rewards fluency: unigrams measure word choice and 3-, 4-grams measure local word order.
- **Brevity penalty**: precision alone rewards very short outputs (test 4, `the cat`, has unigram and bigram precision 1). The penalty $e^{1 - r/c}$ discounts candidates shorter than the closest reference.
- **ROUGE-L** measures in-order overlap without requiring contiguity, and its recall part rewards covering the reference, which suits summaries.

## Smoothing
A single sentence often has no matching 4-gram, so unsmoothed sentence BLEU is 0 even for decent output. Adding one to the numerator and denominator for $n \ge 2$ is a simple fix (test 5). Corpus BLEU avoids the problem by summing $m_n$ and $c_n$ over all sentences before dividing, which is what papers report.

## Complexity
$O(N \cdot (c + \sum |\text{ref}|))$ for the n-gram counts and $O(c \cdot |\text{ref}|)$ per reference for LCS.

## Common bugs
- Clipping against the sum of reference counts instead of the max.
- Averaging per-sentence BLEU and calling it corpus BLEU.
- Choosing the shortest or longest reference for the brevity penalty instead of the closest one.
- Comparing BLEU scores computed with different tokenizations.

## Interview follow-ups
- Why do n-gram metrics correlate poorly with human judgments for open-ended generation, and what do embedding-based or LLM-judge metrics fix (and break)?
- How does chrF differ, and why is it popular for morphologically rich languages? (Character n-gram F-score, so partial word matches count.)
"""
