import numpy as np

META = dict(title='UCB1 arm selection', tags=['Reinforcement Learning', 'Probability'], diff='Easy', runtime='numpy',
            lessons=[], related=['epsilon-greedy-action-selection', 'thompson-sampling-beta-bernoulli', 'linucb-contextual-bandit'])

STATEMENT = r"""
UCB1 handles the explore/exploit trade-off with optimism: it plays the arm with the highest **upper confidence bound** on its mean reward,

$$\text{score}_k = \bar x_k + \sqrt{\frac{c \ln t}{n_k}}$$

where $n_k$ is how often arm $k$ has been pulled, $\bar x_k$ is its average reward so far, and $t = \sum_k n_k$ is the total number of pulls. The classic choice is $c = 2$.

Write `ucb1(counts, sums, c=2.0)` and return `(arm, scores)`.

## Inputs
- `counts`: int array `(K,)` of pulls per arm, all `>= 0`.
- `sums`: float array `(K,)` of total reward per arm.

## Rules
- An arm with `counts == 0` has score `inf` (every arm is tried once before any bonus is compared).
- Pick the arm with the highest score; ties go to the **lowest** index. With no pulls at all, every score is `inf` and the answer is arm 0.

## Output
`arm`: int. `scores`: float array `(K,)`.
"""

STARTER = '''import numpy as np

def ucb1(counts, sums, c=2.0):
    """Return (index of the arm to pull, UCB score of every arm (K,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def ucb1(counts, sums, c=2.0):
    n = np.asarray(counts, dtype=float)
    s = np.asarray(sums, dtype=float)
    t = n.sum()
    scores = np.full(len(n), np.inf)                 # untried arms win until each is tried
    tried = n > 0
    scores[tried] = s[tried] / n[tried] + np.sqrt(c * np.log(max(t, 1.0)) / n[tried])   # t >= 1 whenever any arm is tried
    return int(scores.argmax()), scores
'''

HINTS = [
    'Handle untried arms separately: give them `inf` so the division by zero never happens.',
    'For the tried arms, mean is `sums / counts` and the bonus is `sqrt(c * log(t) / counts)` with `t = counts.sum()`.',
    '`argmax` returns the first maximum, which is exactly the tie rule, and also picks the first untried arm.',
]

_r = np.random.default_rng(111)
NR = _r.integers(1, 40, 6)
TESTS = [
    (np.array([10, 10]), np.array([6.0, 4.0])),
    (np.array([20, 2]), np.array([12.0, 1.0])),
    (np.array([0, 0, 0]), np.zeros(3)),
    (np.array([5, 0, 3, 0]), np.array([2.0, 0.0, 3.0, 0.0])),
    (np.array([1, 1, 1]), np.array([1.0, 1.0, 0.0])),
    (np.array([100, 4]), np.array([60.0, 1.0]), 0.1),
    (np.array([100, 4]), np.array([60.0, 1.0])),
    (NR, np.round(_r.uniform(size=6) * NR, 0)),
    (NR, np.round(_r.uniform(size=6) * NR, 0), 1.0),
]

EXPLANATION = r"""
## Why optimism works
By Hoeffding's inequality, an arm's true mean is unlikely to exceed $\bar x_k + \sqrt{2 \ln t / n_k}$ (for rewards in $[0,1]$). Playing the most optimistic arm means either the arm really is good (exploit) or its bound is loose because $n_k$ is small, and pulling it tightens the bound (explore). The bonus grows only like $\sqrt{\ln t}$, so a clearly bad arm is pulled $O(\ln T)$ times, giving logarithmic regret.

## What the tests show
- Test 2: arm 1 has the lower mean (0.5 vs 0.6) but only 2 pulls, so its bonus wins.
- Test 4: an untried arm is chosen even though tried arms have good means. The first untried one (index 1) wins.
- Test 5: arms 0 and 1 have identical statistics, so they tie exactly and arm 0 is chosen.
- Tests 6 and 7 use the same statistics. With $c = 0.1$ the bonus is small and the better-looking arm 0 wins; with $c = 2$ the rarely pulled arm 1 does.

## Complexity
$O(K)$ per decision.

## Common bugs
- Using $\ln n_k$ instead of $\ln t$, or $t$ as the step count of a *different* clock.
- Dividing by zero for untried arms, producing `nan` that `argmax` handles unpredictably.
- Forgetting the square root.

## Interview follow-ups
- How does UCB1 compare with Thompson sampling in practice and in regret guarantees?
- What changes for rewards that aren't bounded in $[0, 1]$ (e.g. UCB-V or KL-UCB)?
"""
