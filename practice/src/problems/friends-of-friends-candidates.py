META = dict(title='People you may know from friends of friends', tags=['Recommender Systems'], diff='Medium', runtime='python')

STATEMENT = r"""
Write `pymk(edges, user, k, blocked)` for an undirected friendship graph given as a list of `(a, b)` pairs.

- Candidates are users at distance exactly 2 from `user`: not the user, not already a friend, and not in `blocked`.
- Score each candidate by the Adamic-Adar index: $\sum_{z \in N(u) \cap N(c)} 1/\ln |N(z)|$. A mutual friend always has degree at least 2, since they know both people, so the logarithm is positive.
- Return the top `k` as `[candidate, round(score, 6), mutual_count]` lists, sorted by score descending, then mutual count descending, then candidate id ascending.

Ignore self-loops and duplicate edges.
"""

STARTER = '''def pymk(edges, user, k, blocked):
    """Top-k friend suggestions with Adamic-Adar scores."""
    raise NotImplementedError
'''

SOLUTION = '''import math
from collections import defaultdict

def pymk(edges, user, k, blocked):
    N = defaultdict(set)
    for a, b in edges:
        if a != b:
            N[a].add(b); N[b].add(a)
    friends, bl = N[user], set(blocked)
    score, mutual = defaultdict(float), defaultdict(int)
    for z in friends:
        if len(N[z]) < 2:
            continue                      # knows only the user: no candidates through z
        w = 1.0 / math.log(len(N[z]))
        for c in N[z]:
            if c != user and c not in friends and c not in bl:
                score[c] += w
                mutual[c] += 1
    ranked = sorted(score, key=lambda c: (-score[c], -mutual[c], c))
    return [[c, round(score[c], 6), mutual[c]] for c in ranked[:k]]
'''

HINTS = [
    'Walk from each friend z to its neighbours; every step from z adds the same weight 1 / ln(deg z).',
    'Store neighbours in sets so duplicate edges collapse automatically.',
    'A friend whose only connection is the user contributes nothing; skip them before computing 1 / ln(1).',
    'Adamic-Adar down-weights mutual friends who know everyone, because they say little about this pair.',
]

_E = [(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (2, 6), (6, 7), (3, 8), (8, 9), (8, 4), (5, 5), (1, 2)]
_hub = [(0, i) for i in range(1, 30)] + [(1, 50), (50, 51), (1, 60), (60, 51)]
TESTS = [
    (_E, 1, 5, []),
    (_E, 1, 2, [4]),
    (_E, 9, 3, []),
    (_hub, 1, 3, []),
    ([(1, 2)], 1, 3, []),
    (_E, 42, 3, []),
]

EXPLANATION = r"""
## Why friends of friends
Most new friendships close a triangle. Restricting candidates to two hops shrinks millions of users to a few hundred plausible ones, which a heavier model can then rank with profile, location and interaction features.

## Why Adamic-Adar over common-neighbour counts
A mutual friend with 5,000 connections is weak evidence; one with 20 is strong. Weighting by $1/\ln(\text{degree})$ captures that and is a classic strong baseline in link prediction.

## Product constraints
Blocked users, people the user dismissed, and privacy settings must be filtered before ranking. Very high-degree accounts also make the two-hop walk expensive, so production systems cap the fan-out per friend or sample it.

## Interview follow-ups
- How would you evaluate this offline using friendships formed after a cutoff date?
"""
