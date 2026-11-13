import numpy as np

META = dict(title='Maximal marginal relevance re-ranking', tags=['Recommender Systems'], diff='Medium', runtime='numpy',
            lessons=['sdsd:16.05', 'sdsd:09.13'], related=['catalog-coverage-and-novelty', 'cosine-similarity-matrix'])

STATEMENT = r"""
A ranker that sorts purely by relevance often fills the top of the page with near-duplicates. **Maximal marginal relevance** (MMR) builds the list greedily, trading relevance against similarity to what's already been picked. At each step, among the remaining candidates $R \setminus S$, it picks

$$\arg\max_{i \in R \setminus S} \Big[ \lambda \cdot \mathrm{rel}_i - (1 - \lambda) \max_{j \in S} \mathrm{sim}_{ij} \Big]$$

When $S$ is empty (the first pick), the max term is taken to be 0, so for $\lambda > 0$ the first pick is the most relevant item.

Write `mmr(rel, sim, k, lam)` that returns the selected indices, in the order they were picked, as a Python list of ints.

## Inputs
- `rel`: `(n,)` float relevance scores.
- `sim`: `(n, n)` float, symmetric item-item similarity. Its diagonal is never used.
- `k`: number of items to pick, `1 <= k <= n`.
- `lam`: $\lambda \in [0, 1]$. $\lambda = 1$ is a pure relevance sort; $\lambda = 0$ ignores relevance entirely (every first-step value is 0, so the tie rule picks index 0).

## Tie rule
If several candidates have exactly the same MMR value, pick the one with the **smallest index**.
"""

STARTER = '''import numpy as np

def mmr(rel, sim, k, lam):
    """Return the k selected indices in pick order."""
    return np.argsort(-np.asarray(rel), kind='stable')[:k].tolist()
'''

SOLUTION = '''import numpy as np

def mmr(rel, sim, k, lam):
    rel, sim = np.asarray(rel, dtype=float), np.asarray(sim, dtype=float)
    n = len(rel)
    chosen = []
    max_sim = np.zeros(n)                        # max similarity to anything chosen so far
    available = np.ones(n, dtype=bool)
    for _ in range(k):
        value = lam * rel - (1 - lam) * max_sim
        value[~available] = -np.inf
        pick = int(np.argmax(value))             # argmax returns the first (smallest) index on ties
        chosen.append(pick)
        available[pick] = False
        max_sim = np.maximum(max_sim, sim[pick])  # incremental update, O(n) per step
    return chosen
'''

HINTS = [
    'Keep a running array `max_sim[i]` = the largest similarity between candidate `i` and anything picked so far. It starts at 0.',
    'After each pick, update it with `np.maximum(max_sim, sim[pick])` instead of recomputing the max over the whole selected set.',
    'Mask picked items with `-inf` before `np.argmax`. `argmax` already returns the smallest index among exact ties.',
]

_r = np.random.default_rng(12)
_E = _r.normal(size=(15, 5))
_E /= np.linalg.norm(_E, axis=1, keepdims=True)
_S = _E @ _E.T
_rel = _r.random(15)

TESTS = [
    (np.array([0.9, 0.85, 0.5]), np.array([[1.0, 0.95, 0.1], [0.95, 1.0, 0.2], [0.1, 0.2, 1.0]]), 2, 0.5),
    (np.array([0.9, 0.85, 0.5]), np.array([[1.0, 0.95, 0.1], [0.95, 1.0, 0.2], [0.1, 0.2, 1.0]]), 2, 1.0),
    (np.array([1.0, 1.0, 1.0, 1.0]), np.zeros((4, 4)), 3, 0.7),
    (np.array([0.5, 0.75, 0.25, 0.5]), np.array([[1, 0.5, 0, 0.25], [0.5, 1, 0.75, 0.5], [0, 0.75, 1, 0], [0.25, 0.5, 0, 1]]), 4, 0.5),
    (np.array([0.5, 0.75, 0.25, 0.5]), np.array([[1, 0.5, 0, 0.25], [0.5, 1, 0.75, 0.5], [0, 0.75, 1, 0], [0.25, 0.5, 0, 1]]), 4, 0.0),
    (np.array([0.3]), np.array([[1.0]]), 1, 0.2),
    (_rel, _S, 6, 0.7),
    (_rel, _S, 15, 0.3),
]

EXPLANATION = r"""
## Why it works
Each step scores a candidate by how relevant it is **minus** how redundant it is with the closest item already on the list. A near-duplicate of the top item has a large $\max_j \mathrm{sim}_{ij}$ and drops, so a less relevant but different item can overtake it (test 1 vs test 2: the same inputs pick `[0, 2]` with $\lambda = 0.5$ and `[0, 1]` with $\lambda = 1$). Because only the newest pick can raise any candidate's max similarity, the max can be maintained incrementally.

## Complexity
$O(nk)$ time with the running max, $O(n)$ extra memory. The naive version that recomputes $\max_{j \in S}$ every step is $O(nk^2)$.

## Common bugs
- Recomputing `max(sim[i, S])` from scratch, correct but $k$ times slower.
- Forgetting to mask chosen items, so the same item is picked twice when $\lambda$ is large.
- Applying the $(1-\lambda)$ penalty at the first step with a max over an empty set (NumPy raises on that), or using the diagonal of `sim` by accident.

## Interview follow-ups
- MMR is greedy. What does it optimise approximately, and what is a principled alternative? (A determinantal point process selects sets with high quality and low pairwise similarity; greedy MAP inference for DPPs is a common re-ranker.)
- Where in a recommendation stack would you run MMR, and on how many items? (After ranking, on the top few hundred candidates, often with business rules like "at most 2 per seller".)
"""
