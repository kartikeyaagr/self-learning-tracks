import numpy as np

META = dict(title='Beta-binomial updating, batch by batch', tags=['Probability'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:10.01', 'mlmaths:12.10'], related=['bayes-theorem', 'maximum-likelihood-estimation'])

STATEMENT = r"""
You're tracking a click-through rate $\theta$. Start with a $\mathrm{Beta}(a, b)$ prior. After seeing $s$ clicks in $t$ impressions, Bayes' rule gives another Beta:

$$\theta \mid \text{data} \sim \mathrm{Beta}(a + s,\; b + t - s), \qquad E[\theta \mid \text{data}] = \frac{a + s}{a + b + t}$$

Data arrives in batches (one per day). Each posterior becomes the prior for the next batch.

Write `beta_updates(a, b, successes, trials)` that returns the tuple `(post_a, post_b, post_mean)`, where entry $i$ of each array describes the posterior **after** batch $i$.

## Requirements
- `a`, `b`: positive floats. `successes`, `trials`: 1-D int arrays of equal length (possibly empty) with $0 \le s_i \le t_i$.
- All three outputs are float arrays with the same length as the inputs.
- Vectorised: no Python loop over batches.
"""

STARTER = '''import numpy as np

def beta_updates(a, b, successes, trials):
    """Return (post_a, post_b, post_mean), each of shape (T,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def beta_updates(a, b, successes, trials):
    s = np.cumsum(np.asarray(successes, dtype=float))     # running totals: updating is just adding counts
    f = np.cumsum(np.asarray(trials, dtype=float)) - s
    post_a, post_b = a + s, b + f
    return post_a, post_b, post_a / (post_a + post_b)
'''

HINTS = [
    'Updating twice, with $(s_1, t_1)$ then $(s_2, t_2)$, gives the same result as updating once with the totals. The order and the batching don\'t matter.',
    'So the posterior after batch $i$ only needs the running totals of successes and failures up to $i$: `np.cumsum`.',
    'Failures are trials minus successes. The posterior mean is $a\' / (a\' + b\')$.',
]

TESTS = [
    (1.0, 1.0, np.array([3, 0, 5]), np.array([10, 4, 6])),
    (2.0, 8.0, np.array([1]), np.array([1])),
    (0.5, 0.5, np.array([], dtype=int), np.array([], dtype=int)),
    (1.0, 1.0, np.array([0, 0, 0]), np.array([0, 0, 0])),
    (10.0, 990.0, np.array([12, 15, 9, 20]), np.array([1000, 1200, 800, 1500])),
    (1.0, 1.0, np.array([100, 0]), np.array([100, 100])),
    (3.0, 2.0, np.random.default_rng(4).binomial(50, 0.3, size=10), np.full(10, 50)),
]

EXPLANATION = r"""
## Why the Beta is conjugate
The binomial likelihood is proportional to $\theta^s(1-\theta)^{t-s}$, and the Beta prior to $\theta^{a-1}(1-\theta)^{b-1}$. Their product has the same form with the exponents added. So $a$ and $b$ act as "pseudo-counts" of prior successes and failures, and the data just adds real counts to them.

## Shrinkage
The posterior mean $\frac{a + s}{a + b + t}$ is a weighted average of the prior mean $\frac{a}{a+b}$ and the observed rate $\frac{s}{t}$, with weights $a + b$ and $t$. With little data the prior dominates. With a lot, the data does. A $\mathrm{Beta}(10, 990)$ prior says "about 1%, worth 1,000 impressions of evidence", which stops one lucky day from swinging the estimate.

## Complexity
$O(T)$ with cumulative sums.

## Common bugs
- Adding trials to $b$ instead of failures.
- Applying each batch to the **original** prior instead of the running posterior.
- Treating batches with zero trials as an error. They simply leave the posterior unchanged.

## Interview follow-ups
- How would you do Thompson sampling for three ad variants with this? (Keep a Beta per arm, sample one $\theta$ from each, show the arm with the largest sample, update it with the result.)
- What's the probability that variant B's rate beats A's under the posteriors? (Monte Carlo: sample both Betas many times and count how often B wins.)
"""
