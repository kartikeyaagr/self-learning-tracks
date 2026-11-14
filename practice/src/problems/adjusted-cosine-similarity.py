import numpy as np

META = dict(title='Adjusted cosine item similarity', tags=['Recommender Systems'], diff='Medium', runtime='numpy',
            related=['item-item-cf-scores', 'cosine-similarity-matrix'])

STATEMENT = r"""
Write `adjusted_cosine(R)` for a user-by-item rating matrix `R` where `0` means "not rated".

1. Compute each user's mean over the items they rated (a user with no ratings has mean 0).
2. Centre: for rated cells, $\tilde r_{ui} = r_{ui} - \bar r_u$; unrated cells stay 0.
3. Item similarity $s_{ij} = \dfrac{\sum_u \tilde r_{ui}\tilde r_{uj}}{\sqrt{\sum_u \tilde r_{ui}^2}\sqrt{\sum_u \tilde r_{uj}^2}}$, summing over **all** users (unrated cells contribute 0).
4. If either norm is 0, the similarity is 0. Set the diagonal to 1 for items with a non-zero norm and 0 otherwise.

Return the `(n_items, n_items)` matrix.
"""

STARTER = '''import numpy as np

def adjusted_cosine(R):
    """Item-item adjusted cosine similarity."""
    R = np.asarray(R, float)
    n = np.linalg.norm(R, axis=0)
    return (R.T @ R) / np.outer(n, n)
'''

SOLUTION = '''import numpy as np

def adjusted_cosine(R):
    R = np.asarray(R, float)
    mask = R != 0
    cnt = mask.sum(1)
    mean = np.divide(R.sum(1), cnt, out=np.zeros(len(R)), where=cnt > 0)
    C = np.where(mask, R - mean[:, None], 0.0)
    norm = np.linalg.norm(C, axis=0)
    denom = np.outer(norm, norm)
    S = np.divide(C.T @ C, denom, out=np.zeros_like(denom), where=denom > 0)
    np.fill_diagonal(S, (norm > 0).astype(float))
    return S
'''

HINTS = [
    'Centre by the user mean (row), not the item mean. That is what makes it "adjusted" rather than Pearson.',
    'Use np.divide with where= to avoid dividing by zero norms.',
    'A user who rates everything the same contributes nothing after centring.',
]

_r = np.random.default_rng(12)
_R = _r.integers(1, 6, (8, 6)) * (_r.random((8, 6)) < 0.6)
TESTS = [
    ([[5, 3, 0], [4, 0, 4], [1, 1, 5]],),
    (_R,),
    ([[3, 3, 3], [3, 3, 3]],),
    ([[0, 0], [0, 0]],),
    ([[5, 1], [1, 5], [5, 1]],),
    (_r.integers(0, 6, (20, 10)),),
]

EXPLANATION = r"""
## Why adjust
Plain cosine on raw ratings treats a 4 from a harsh rater and a 4 from a generous one as the same signal. Subtracting each user's mean turns ratings into "above or below what this person usually gives", which is what item-item similarity should compare.

## Adjusted cosine versus Pearson
Pearson correlation between two item columns centres by the item mean, over users who rated both. Adjusted cosine centres by the user mean. For item-item filtering Sarwar et al. found adjusted cosine works better, because item means hide the user scale differences.

## Practical notes
Sparse overlap makes similarities noisy; shrink them toward zero by $n_{ij}/(n_{ij}+\beta)$, where $n_{ij}$ is the number of co-raters.

## Interview follow-ups
- Why can adjusted cosine be negative, and should negative neighbours be used in prediction?
"""
