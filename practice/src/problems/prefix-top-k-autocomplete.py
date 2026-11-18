META = dict(title='Prefix autocomplete with frequency and recency', tags=['Data Processing', 'Recommender Systems'], diff='Medium', runtime='python',
            lessons=['sdsd:09.14'])

STATEMENT = r"""
Build query autocomplete from a search log. Write `autocomplete(log, prefixes, k, now, half_life)`.

- `log` is a list of `(query, timestamp)` pairs. Normalise queries by lowercasing and collapsing whitespace (`" ".join(q.lower().split())`); skip queries that are empty after normalising.
- Each occurrence contributes weight $2^{-(now - t)/h}$ where $h$ is `half_life`, so recent searches count more.
- For each prefix (normalised the same way, but keep a single trailing space if the original prefix ends with whitespace, since "new " should not match "newton"), return the top `k` stored queries that start with it, by total weight descending, ties broken alphabetically.

Return a list with one list of suggestions per prefix.
"""

STARTER = '''def autocomplete(log, prefixes, k, now, half_life):
    """Top-k suggestions per prefix."""
    qs = sorted({q for q, _ in log})
    return [[q for q in qs if q.startswith(p)][:k] for p in prefixes]
'''

SOLUTION = '''from collections import defaultdict

def _norm(s):
    return " ".join(s.lower().split())

def autocomplete(log, prefixes, k, now, half_life):
    w = defaultdict(float)
    for q, t in log:
        q = _norm(q)
        if q:
            w[q] += 2.0 ** (-(now - t) / half_life)
    ranked = sorted(w, key=lambda q: (-w[q], q))
    out = []
    for p in prefixes:
        pn = _norm(p)
        if p[-1:].isspace() and pn:
            pn += " "
        out.append([q for q in ranked if q.startswith(pn)][:k])
    return out
'''

HINTS = [
    'Aggregate weights per normalised query once, then sort once; each prefix is a filter over the sorted list.',
    'The trailing-space rule matters: after normalising, "new " would become "new" and match "newton".',
    'An exponential decay with half-life h is 2 ** (-age / h).',
]

_log = [("New York weather", 90), ("new york  weather", 95), ("newton laws", 99), ("New York Times", 10),
        ("new york times", 20), ("new york times", 30), ("netflix", 98), ("   ", 99), ("nets score", 50)]
TESTS = [
    (_log, ["new", "new ", "ne", "x"], 3, 100, 10.0),
    (_log, ["new york"], 2, 100, 1000.0),
    (_log, ["NEW YORK "], 5, 100, 10.0),
    (_log, [""], 4, 100, 10.0),
    ([("a", 0), ("ab", 0), ("abc", 0)], ["a"], 2, 0, 1.0),
]

EXPLANATION = r"""
## The core trade-off
Pure counts favour queries that were popular long ago; pure recency follows noise. An exponential decay keeps both: a query searched many times last year still appears, but a burst of searches today can overtake it.

## Serving at scale
Scanning all queries per keystroke is far too slow. Production systems precompute the top $k$ per prefix in a trie (or a sorted key-value store) offline and refresh it every few minutes, so a lookup is a single read. Personalisation and filtering of unsafe suggestions are layered on top.

## Normalisation
Users type the same query many ways. Without normalising case and whitespace the counts split and the true favourite can drop out of the top $k$.

## Interview follow-ups
- How would you keep offensive or trending-abuse suggestions out without a human in the loop for every prefix?
"""
