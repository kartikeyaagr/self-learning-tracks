import numpy as np

META = dict(title='Triplet loss with batch-hard mining', tags=['Loss Functions', 'Embeddings'], diff='Medium', runtime='numpy',
            related=['contrastive-loss-siamese', 'info-nce-loss', 'euclidean-distance'])

STATEMENT = r"""
Write `triplet_loss(emb, labels, margin, mining)` for a batch of embeddings `emb` (shape `(n, d)`) with integer `labels`. Use Euclidean distance.

- `mining="all"`: average $\max(0, d(a,p) - d(a,n) + m)$ over **all valid triplets** $(a, p, n)$ with $a \ne p$, label$(a)$ = label$(p)$, label$(n) \ne$ label$(a)$.
- `mining="hard"`: for each anchor $a$ that has at least one positive and one negative, take its farthest positive and its closest negative, compute $\max(0, d(a,p^*) - d(a,n^*) + m)$, and average over those anchors.

Return the loss as a float (0.0 if there are no valid triplets or anchors).
"""

STARTER = '''import numpy as np

def triplet_loss(emb, labels, margin, mining):
    """Return the mean triplet loss for mining in {"all", "hard"}."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def triplet_loss(emb, labels, margin, mining):
    E, y = np.asarray(emb, dtype=float), np.asarray(labels)
    D = np.sqrt(np.maximum(((E[:, None] - E[None]) ** 2).sum(-1), 0.0))
    same = y[:, None] == y[None]
    pos = same & ~np.eye(len(y), dtype=bool)
    neg = ~same
    if mining == "all":
        L = D[:, :, None] - D[:, None, :] + margin          # [a, p, n]
        valid = pos[:, :, None] & neg[:, None, :]
        return float(np.maximum(L[valid], 0).mean()) if valid.any() else 0.0
    vals = []
    for a in range(len(y)):
        if pos[a].any() and neg[a].any():
            vals.append(max(0.0, D[a][pos[a]].max() - D[a][neg[a]].min() + margin))
    return float(np.mean(vals)) if vals else 0.0
'''

HINTS = [
    'Compute the full pairwise distance matrix once; positives and negatives are boolean masks over it.',
    'For all triplets, broadcast D[a, p] - D[a, n] into an (n, n, n) array and select the valid entries with a 3-D mask.',
    'Batch-hard mining uses, per anchor, the maximum over positives and the minimum over negatives.',
]

_r = np.random.default_rng(23)
TESTS = [
    (np.array([[0.0], [0.5], [3.0]]), np.array([0, 0, 1]), 1.0, "all"),
    (np.array([[0.0], [0.5], [0.8], [3.0]]), np.array([0, 0, 1, 1]), 1.0, "hard"),
    (_r.normal(size=(8, 3)), np.array([0, 0, 1, 1, 2, 2, 0, 1]), 0.5, "all"),
    (_r.normal(size=(8, 3)), np.array([0, 0, 1, 1, 2, 2, 0, 1]), 0.5, "hard"),
    (np.array([[0.0], [1.0]]), np.array([0, 1]), 1.0, "all"),
    (np.array([[0.0, 0.0], [0.0, 0.1], [5.0, 5.0], [5.0, 5.1]]), np.array([0, 0, 1, 1]), 0.2, "hard"),
]

EXPLANATION = r"""
## What the loss asks for
For each anchor, the positive should be closer than the negative by at least the margin. Triplets that already satisfy this give zero loss, so training focuses on the ones that do not.

## Why mining matters
The number of triplets grows as $n^3$ and most are easy. Averaging over all of them dilutes the signal; batch-hard mining (from person re-identification work) uses only the hardest positive and negative per anchor, which learns faster but is sensitive to label noise. Semi-hard mining, picking negatives farther than the positive but within the margin, is a common compromise.

## Relation to other losses
Contrastive loss looks at pairs, triplet loss at relative distances, and InfoNCE at one positive against many negatives with a softmax; the latter has largely replaced triplet loss for large-scale representation learning.

## Interview follow-ups
- Why normalise embeddings to unit length before computing distances?
"""
