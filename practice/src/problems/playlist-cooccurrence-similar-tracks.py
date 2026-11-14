import math

META = dict(title='Similar tracks from playlist co-occurrence', tags=['Recommender Systems'], diff='Medium', runtime='python',
            related=['co-purchase-lift', 'item-item-cf-scores'])

STATEMENT = r"""
Write `similar_tracks(playlists, artist, track, k, min_co)`.

- `playlists` is a list of lists of track ids (a track repeated in a playlist counts once). `artist` maps every track id to its artist.
- For candidate track $j \ne$ `track`, let $c_{ij}$ be the number of playlists containing both, and $n_i$, $n_j$ the number containing each.
- Similarity is $c_{ij} / \sqrt{n_i n_j}$ (cosine on the playlist incidence vectors).
- Keep candidates with $c_{ij} \ge$ `min_co` and a **different artist** from `track`, so the list is not filled with the same album.
- Return the top `k` as `[j, round(sim, 6), c_ij]`, sorted by similarity descending, then `c_ij` descending, then `j` ascending.
"""

STARTER = '''def similar_tracks(playlists, artist, track, k, min_co):
    """Top-k similar tracks by playlist co-occurrence."""
    from collections import Counter
    c = Counter(j for p in playlists if track in p for j in p if j != track)
    return [[j, float(n), n] for j, n in c.most_common(k)]
'''

SOLUTION = '''import math
from collections import Counter

def similar_tracks(playlists, artist, track, k, min_co):
    P = [set(p) for p in playlists]
    n = Counter(t for p in P for t in p)
    co = Counter(j for p in P if track in p for j in p if j != track)
    rows = []
    for j, c in co.items():
        if c >= min_co and artist[j] != artist[track]:
            rows.append([j, round(c / math.sqrt(n[track] * n[j]), 6), c])
    rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    return rows[:k]
'''

HINTS = [
    'Raw co-occurrence counts favour hits that appear in every playlist; dividing by sqrt(n_i n_j) removes that.',
    'Deduplicate within each playlist before counting.',
    'The same-artist filter is a product rule applied after scoring, not part of the similarity.',
]

_art = {"t1": "A", "t2": "A", "t3": "B", "t4": "C", "t5": "D", "hit": "E", "t6": "B"}
_P = [["t1", "t3", "hit"], ["t1", "t3", "t4"], ["t1", "t2", "hit"], ["hit", "t5"], ["hit", "t6"], ["t1", "t4", "t4"],
      ["hit", "t3"], ["t6", "t5"], ["t1", "t6"]]
TESTS = [
    (_P, _art, "t1", 3, 1),
    (_P, _art, "t1", 5, 2),
    (_P, _art, "hit", 3, 1),
    (_P, _art, "t5", 3, 1),
    ([["a", "b"], ["a", "c"]], {"a": "X", "b": "X", "c": "Y"}, "a", 5, 1),
]

EXPLANATION = r"""
## Playlists as implicit labels
A playlist is a human saying "these tracks go together". Counting co-occurrence across millions of playlists is a strong, cheap similarity signal for "fans also like" and radio seeds, and it captures mood and context that audio features miss.

## Normalisation
Without it, the most popular tracks are similar to everything. Cosine on incidence vectors divides out popularity; a further shrink term $c/(c+\beta)$ helps tracks with few playlists.

## Product rules
Users asking for similar tracks rarely want more of the same artist, so diversity constraints like this filter, or a cap per artist, sit on top of the similarity.

## Interview follow-ups
- How does this compare with learning track embeddings with word2vec on playlists treated as sentences?
"""
