import numpy as np

META = dict(title='Thompson sampling for Bernoulli bandits', tags=['Reinforcement Learning', 'Probability'], diff='Medium',
            runtime='numpy', lessons=[], related=['ucb1-for-multi-armed-bandits', 'epsilon-greedy-action-selection'])

STATEMENT = r"""
Thompson sampling keeps a posterior over each arm's success probability and plays the arm whose **sampled** probability is highest. With Bernoulli rewards and a $\text{Beta}(\alpha, \beta)$ prior, the posterior stays Beta: a success adds 1 to $\alpha$, a failure adds 1 to $\beta$.

Write `thompson_bernoulli(outcomes, seed, prior_a=1.0, prior_b=1.0)` that runs the bandit for `T` steps and returns `(arms, alpha, beta)`.

## Inputs
- `outcomes`: int array `(T, K)` of 0/1. `outcomes[t, k]` is the reward arm `k` would pay if pulled at step `t` (the environment is pre-drawn so runs are reproducible).
- `seed`: int for the sampler.

## Algorithm (follow it exactly so the random stream matches)
1. `rng = np.random.default_rng(seed)`. Set `alpha = np.full(K, prior_a)` and `beta = np.full(K, prior_b)` (floats).
2. For each step `t = 0 .. T-1`:
   1. Draw `theta = rng.beta(alpha, beta)`, one call per step giving a `(K,)` array.
   2. Pull `k = argmax(theta)` (first index on ties).
   3. Observe `r = outcomes[t, k]`, then `alpha[k] += r` and `beta[k] += 1 - r`.

## Output
- `arms`: int array `(T,)` of pulled arms.
- `alpha`, `beta`: float arrays `(K,)`, the final posterior parameters.
"""

STARTER = '''import numpy as np

def thompson_bernoulli(outcomes, seed, prior_a=1.0, prior_b=1.0):
    """outcomes: (T, K) of 0/1. Return (arms (T,), alpha (K,), beta (K,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def thompson_bernoulli(outcomes, seed, prior_a=1.0, prior_b=1.0):
    outcomes = np.asarray(outcomes)
    T, K = outcomes.shape
    rng = np.random.default_rng(seed)
    alpha, beta = np.full(K, float(prior_a)), np.full(K, float(prior_b))
    arms = np.zeros(T, dtype=int)
    for t in range(T):
        theta = rng.beta(alpha, beta)          # one posterior sample per arm
        k = int(theta.argmax())
        r = outcomes[t, k]
        alpha[k] += r                          # conjugate update: successes
        beta[k] += 1 - r                       # ... and failures
        arms[t] = k
    return arms, alpha, beta
'''

HINTS = [
    'The posterior of each arm is fully described by two numbers. Keep them in two float arrays initialised to the prior.',
    'Each step: one call `rng.beta(alpha, beta)` (vectorised over arms), take the argmax, and update only the pulled arm.',
    'Only the pulled arm\'s outcome is observed. The other columns of `outcomes[t]` are counterfactual and must not touch the posteriors.',
]

_r = np.random.default_rng(121)
O3 = (_r.uniform(size=(40, 3)) < np.array([0.2, 0.5, 0.8])).astype(int)
O5 = (_r.uniform(size=(60, 5)) < np.array([0.1, 0.3, 0.3, 0.6, 0.4])).astype(int)
TESTS = [
    (np.array([[1, 0], [1, 0], [0, 1]]), 0),
    (np.array([[1, 1], [1, 1], [1, 1], [1, 1]]), 7, 1.0, 1.0),
    (np.zeros((5, 1), dtype=int), 3),
    (O3, 1),
    (O3, 2, 2.0, 5.0),
    (O5, 4),
    (O5, 5, 0.5, 0.5),
]

EXPLANATION = r"""
## Why sampling explores the right amount
An arm is chosen with exactly the posterior probability that it is the best arm (probability matching). An arm with few pulls has a wide posterior, so it sometimes produces a high sample and gets tried. An arm that is confidently bad almost never does. Exploration fades on its own as posteriors sharpen, with no schedule to tune, and the regret is logarithmic like UCB1's. In practice it often does better.

## Why conjugacy keeps it cheap
Beta prior times Bernoulli likelihood is again Beta, so the posterior update is two additions. With $\text{Beta}(1, 1)$ (uniform) as the prior, $\alpha - 1$ counts successes and $\beta - 1$ failures. Test 5 uses a pessimistic prior $\text{Beta}(2, 5)$, which changes the early choices.

## Complexity
$O(TK)$ time for the samples, $O(K)$ memory.

## Common bugs
- Drawing one sample per arm with separate calls, or in a different order, so the random stream differs.
- Updating every arm with its counterfactual outcome (that's full-information feedback, not a bandit).
- Picking the arm with the highest posterior *mean*. That's greedy and can lock onto a lucky arm forever.

## Interview follow-ups
- How would you do Thompson sampling with Gaussian rewards of unknown mean? (Normal prior, Normal posterior.)
- How do you handle delayed feedback or batched decisions, where many pulls happen before any reward arrives?
"""
