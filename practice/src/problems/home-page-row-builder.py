META = dict(title='Build home page rows without repeating titles', tags=['Recommender Systems'], diff='Medium', runtime='python')

STATEMENT = r"""
A streaming home page shows several rows, each a ranked list for a theme. Write `build_page(rows, n_rows, row_len, min_len)`.

- `rows` is a list of `(name, row_score, items)`, where `items` is that row's ranked list of title ids.
- Process rows by `row_score` descending (ties by input order). For each row, keep its items in order, skipping any title already shown in an earlier **accepted** row, until it has `row_len` items.
- Accept the row only if it ends with at least `min_len` items; otherwise discard it (its titles are not marked as shown).
- Stop after `n_rows` rows are accepted.

Return a list of `[name, [items]]` in page order.
"""

STARTER = '''def build_page(rows, n_rows, row_len, min_len):
    """Return [[name, items], ...]."""
    top = sorted(rows, key=lambda r: -r[1])[:n_rows]
    return [[n, list(items)[:row_len]] for n, _, items in top]
'''

SOLUTION = '''def build_page(rows, n_rows, row_len, min_len):
    shown, page = set(), []
    for k, (name, _, items) in sorted(enumerate(rows), key=lambda kr: (-kr[1][1], kr[0])):
        if len(page) >= n_rows:
            break
        picked = []
        for t in items:
            if len(picked) >= row_len:
                break
            if t not in shown and t not in picked:
                picked.append(t)
        if len(picked) >= min_len:
            page.append([name, picked])
            shown.update(picked)
    return page
'''

HINTS = [
    'Track shown titles across accepted rows only; a rejected row must not hide titles from later rows.',
    'Also skip a title that repeats inside the same row.',
    'Sort with the input index as a secondary key to make ties deterministic.',
]

_R = [("Continue watching", 0.99, ["t1", "t2"]),
      ("Trending", 0.9, ["t3", "t1", "t4", "t5", "t6"]),
      ("Because you watched X", 0.8, ["t3", "t4", "t7", "t8", "t9", "t10"]),
      ("New releases", 0.85, ["t5", "t6", "t3", "t11"]),
      ("Comedies", 0.5, ["t12", "t13", "t14", "t14"])]
TESTS = [
    (_R, 4, 4, 2),
    (_R, 4, 4, 3),
    (_R, 2, 3, 1),
    (_R, 10, 2, 2),
    ([], 3, 3, 1),
    ([("a", 1.0, ["x", "y"]), ("b", 1.0, ["x", "z"])], 2, 2, 2),
]

EXPLANATION = r"""
## Two-level ranking
The page is ranked twice: titles within each row, and rows on the page. Each row is usually produced by a different candidate source (continue watching, trending, similar-to). Building the page is where they meet, and without coordination the same hit appears in four rows.

## Why the minimum row length
After removing duplicates, a row can shrink to one or two titles. A near-empty row wastes vertical space that a full row would use better, so it is dropped, and its titles stay available for later rows.

## What this greedy misses
Choosing rows in score order can let a high row consume the titles that made a lower row valuable. Page-level optimisation exists, but greedy with dedupe is the common baseline because it is fast and its behaviour is easy to explain to editors.

## Interview follow-ups
- How would you learn the row scores, given that a row's clicks depend on its position on the page?
"""
