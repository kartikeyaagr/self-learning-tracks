META = dict(title='Frequently bought together with lift and support', tags=['Recommender Systems', 'Probability and Statistics'], diff='Easy', runtime='python',
            related=['jaccard-similarity'])

STATEMENT = r"""
Write `bought_together(baskets, item, k, min_support)` for a list of baskets (each a list of item ids; duplicates inside a basket count once).

- $P(x)$ is the fraction of baskets containing $x$; $P(x, y)$ the fraction containing both.
- For each other item $j$ that co-occurs with `item` in at least `min_support` baskets (a count), compute
  - `confidence` $= P(\text{item}, j) / P(\text{item})$
  - `lift` $= P(\text{item}, j) / (P(\text{item}) P(j))$
- Keep only items with lift > 1. Return the top `k` as `[j, count, round(confidence, 6), round(lift, 6)]`, sorted by lift descending, then count descending, then `j` ascending.
"""

STARTER = '''from collections import Counter

def bought_together(baskets, item, k, min_support):
    """Top-k co-purchased items ranked by lift."""
    c = Counter(j for b in baskets if item in b for j in b if j != item)
    return [[j, n, 0.0, 0.0] for j, n in c.most_common(k)]
'''

SOLUTION = '''from collections import Counter

def bought_together(baskets, item, k, min_support):
    B = [set(b) for b in baskets]
    n = len(B)
    single = Counter(x for b in B for x in b)
    pair = Counter(j for b in B if item in b for j in b if j != item)
    if not single[item]:
        return []
    p_i = single[item] / n
    rows = []
    for j, c in pair.items():
        if c < min_support:
            continue
        p_ij, p_j = c / n, single[j] / n
        lift = p_ij / (p_i * p_j)
        if lift > 1:
            rows.append([j, c, round(p_ij / p_i, 6), round(lift, 6)])
    rows.sort(key=lambda r: (-r[3], -r[1], r[0]))
    return rows[:k]
'''

HINTS = [
    'Convert each basket to a set first so a repeated item does not inflate counts.',
    'Confidence alone favours items everyone buys (bags, milk); lift divides that popularity out.',
    'Apply the support filter before ranking, because lift is extreme and noisy for rare pairs.',
]

_B = [["phone", "case", "charger"], ["phone", "case"], ["phone", "bag"], ["bag", "milk"], ["milk", "bag"],
      ["phone", "charger", "charger"], ["case", "screen"], ["phone", "screen", "case"], ["bag"], ["milk"]]
TESTS = [
    (_B, "phone", 3, 1),
    (_B, "phone", 3, 2),
    (_B, "bag", 5, 1),
    (_B, "tv", 3, 1),
    ([["a", "b"], ["a", "b"], ["c"]], "a", 2, 1),
    ([["a", "b"], ["a"], ["b"], ["a", "b"]], "a", 2, 1),
]

EXPLANATION = r"""
## Confidence versus lift
Confidence is "of baskets with the phone, how many also had $j$". If $j$ is in almost every basket, confidence is high for any item and says nothing. Lift compares that to $j$'s base rate; lift above 1 means buying the phone makes $j$ more likely.

## Why minimum support
A pair seen once can have a huge lift by chance. A support floor (or a Bayesian shrink toward 1) keeps recommendations to pairs with enough evidence.

## Where it fits
Co-purchase statistics are a cheap, strong candidate generator for "frequently bought together" and cart add-ons, and a useful sanity baseline for learned item embeddings.

## Interview follow-ups
- How would you avoid recommending substitutes (two phone cases) when the goal is complements?
"""
