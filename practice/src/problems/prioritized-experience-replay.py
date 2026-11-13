import numpy as np

META = dict(title='Prioritized replay: stratified sampling and IS weights', tags=['Reinforcement Learning', 'Probability'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:07.03'],
            related=['experience-replay-buffer', 'dqn-loss-with-target-network', 'softmax-boltzmann-exploration'])

STATEMENT = r"""
Prioritized experience replay samples transitions with large TD error more often. With proportional prioritization, transition $i$ gets priority and sampling probability

$$p_i = (|\delta_i| + \epsilon)^\alpha, \qquad P(i) = \frac{p_i}{\sum_j p_j}$$

Non-uniform sampling biases the gradient, so each sample is reweighted by an importance-sampling weight, normalised so the largest possible weight is 1:

$$w_i = \frac{(N \cdot P(i))^{-\beta}}{\max_j (N \cdot P(j))^{-\beta}}$$

Write `per_sample(td_errors, u, alpha=0.6, beta=0.4, eps=1e-6)` and return `(indices, weights, probs)`.

## Sampling rule (stratified, with the draws passed in)
Let `cdf = np.cumsum(p)` and `total = cdf[-1]`. Split $[0, \text{total})$ into $B$ equal segments. For segment `j` (with `B = len(u)`), the target mass is `(j + u[j]) * total / B`, and the sampled index is the first `i` with `cdf[i] > target`, clipped to `N - 1`.

## Inputs
- `td_errors`: float `(N,)`, the latest TD error of each stored transition.
- `u`: float `(B,)` of uniform draws in `[0, 1)`.
- `eps > 0`, so every probability is positive.

## Output
- `indices`: int `(B,)`, in segment order (so non-decreasing).
- `weights`: float `(B,)`, the $w$ of each sampled index. The max in the normaliser is over **all** N transitions, not just the batch.
- `probs`: float `(N,)`, the $P(i)$.
"""

STARTER = '''import numpy as np

def per_sample(td_errors, u, alpha=0.6, beta=0.4, eps=1e-6):
    """Return (sampled indices (B,), IS weights (B,), sampling probabilities (N,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def per_sample(td_errors, u, alpha=0.6, beta=0.4, eps=1e-6):
    p = (np.abs(np.asarray(td_errors, dtype=float)) + eps) ** alpha
    cdf = np.cumsum(p)
    total = cdf[-1]
    probs = p / total
    N, B = len(p), len(u)
    targets = (np.arange(B) + np.asarray(u)) * total / B      # one point inside each equal-mass segment
    idx = np.minimum(np.searchsorted(cdf, targets, side='right'), N - 1)   # first cdf entry > target
    w = (N * probs[idx]) ** -beta
    w_max = (N * probs.min()) ** -beta                          # rarest transition has the largest weight
    return idx, w / w_max, probs
'''

HINTS = [
    'Priorities first: `(abs(td) + eps) ** alpha`. The cumulative sum turns sampling into a search over a sorted array.',
    'Segment `j` has target `(j + u[j]) * total / B`. `np.searchsorted(cdf, targets, side="right")` returns the first index whose cdf exceeds each target.',
    'The largest weight belongs to the transition with the smallest probability, so normalise by `(N * probs.min()) ** -beta`.',
]

_r = np.random.default_rng(171)
TD = _r.normal(size=20) * 2
TESTS = [
    (np.array([1.0, 3.0]), np.array([0.5, 0.5]), 1.0, 1.0, 1e-9),
    (np.array([0.0, 1.0, 0.0, 3.0]), np.array([0.1, 0.9]), 1.0, 0.5, 1e-3),
    (np.array([0.5, 0.5, 0.5]), np.array([0.2, 0.7, 0.4]), 0.6, 0.4),
    (np.array([2.0, -2.0, 1.0]), np.array([0.3]), 0.0, 1.0),
    (np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 0.0, 0.0]), 1.0, 1.0, 1e-3),
    (TD, _r.uniform(size=8), 0.6, 0.4),
    (TD, _r.uniform(size=5), 0.9, 1.0, 1e-2),
]

EXPLANATION = r"""
## Why prioritise, and why correct
A transition with near-zero TD error teaches almost nothing; one with a large error is where the Q function is most wrong. Sampling proportional to $|\delta|^\alpha$ spends updates where they matter. $\alpha = 0$ recovers uniform replay (test 4). But the expected gradient under $P$ is no longer the expected gradient under the buffer's distribution. Multiplying each sample's loss by $(N P(i))^{-\beta}$ undoes that, exactly at $\beta = 1$. In practice $\beta$ is annealed from about 0.4 to 1 over training, since the bias matters most near convergence.

## Why stratify
Splitting the mass into $B$ equal segments and drawing one sample per segment guarantees the batch covers the whole distribution and lowers variance compared with $B$ independent draws. It also makes each batch's indices sorted.

## The $\epsilon$
Without it a transition with $\delta = 0$ has probability 0 and is never replayed again, so its stale error is never refreshed (and its weight would be infinite). In test 2 two transitions have zero error: with a small $\epsilon$ they are almost never drawn, and they set the normaliser because their weight would be the largest.

## Complexity
Here $O(N)$ per batch for the cumsum. Real buffers use a **sum tree** (a binary tree of partial sums) so that sampling and priority updates are $O(\log N)$.

## Common bugs
- Normalising by the max weight in the batch instead of over the buffer (both appear in the wild; know which one you're asked for).
- `side='left'` in `searchsorted`, which can select a zero-probability slot at a boundary.
- Forgetting the absolute value on the TD error.

## Interview follow-ups
- Implement a sum tree with `update(i, p)` and `find(mass)`.
- Rank-based prioritisation uses $p_i = 1/\text{rank}(i)$. When is it more robust than proportional?
"""
