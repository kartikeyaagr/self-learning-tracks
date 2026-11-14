import numpy as np

META = dict(title='Measure how often SAE features fire', tags=['Neural Networks', 'Metrics & Evaluation'], diff='Easy', runtime='numpy',
            related=['sae-forward'])

STATEMENT = r"""
Write `feature_frequency(F, dense_threshold)` for an activation matrix `F (n_tokens, m)` collected by running an SAE over many tokens.

- `freq[i]` is the fraction of tokens where feature `i` is strictly positive.
- `dead` lists, in increasing order, the features that never fire.
- `dense` lists, in increasing order, the features with `freq > dense_threshold`.
- `log_freq[i]` is $\log_{10}$ `freq[i]`, or `None` for dead features.

Return a dict with keys `freq`, `dead`, `dense`, `log_freq` (lists).
"""

STARTER = '''import numpy as np

def feature_frequency(F, dense_threshold):
    """Firing rate per feature, with dead and dense features flagged."""
    freq = np.asarray(F).mean(0)
    return {"freq": freq.tolist(), "dead": [], "dense": [], "log_freq": np.log10(freq).tolist()}
'''

SOLUTION = '''import numpy as np

def feature_frequency(F, dense_threshold):
    F = np.asarray(F, float)
    freq = (F > 0).mean(0)
    dead = [int(i) for i in np.flatnonzero(freq == 0)]
    dense = [int(i) for i in np.flatnonzero(freq > dense_threshold)]
    log_freq = [None if p == 0 else float(np.log10(p)) for p in freq]
    return {"freq": freq.tolist(), "dead": dead, "dense": dense, "log_freq": log_freq}
'''

HINTS = [
    'Frequency is about whether a feature fires, not how strongly: average the boolean F > 0.',
    'log10(0) is -inf; report dead features separately rather than letting them break a histogram.',
]

_r = np.random.default_rng(4)
_F = np.maximum(_r.normal(size=(200, 8)) + np.array([-3, -2, -1, 0, 1, -10, -2.5, 0.5]), 0)

TESTS = [
    ([[0.0, 1.0, 0.0], [0.0, 2.0, 0.5]], 0.6),
    ([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], 0.5),
    (_F, 0.3),
    (_F, 0.9),
    ([[0.0, 0.0, 0.0]], 0.1),
]

EXPLANATION = r"""
## What the histogram tells you
Plotting $\log_{10}$ of the firing rate for every feature is the first health check on a trained SAE.

- **Dead features** never fire. They waste dictionary capacity. A large dead fraction usually means the learning rate or the sparsity penalty was too high early on. Fixes include resampling dead features towards inputs the SAE reconstructs badly, or an auxiliary loss that gives dead features gradient.
- **Dense features** fire on a large share of tokens. Some are real (position, "this is English"), but a cluster of them often means the SAE is using them to cheat on the sparsity penalty, and they are rarely interpretable.
- The healthy bulk sits between roughly $10^{-5}$ and $10^{-2}$.

## Interview follow-ups
- Why do you need many millions of tokens to estimate the frequency of rare features, and how does that affect how you call a feature dead?
"""
