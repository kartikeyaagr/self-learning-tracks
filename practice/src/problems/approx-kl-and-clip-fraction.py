import numpy as np

META = dict(title='PPO diagnostics: approximate KL and clip fraction', tags=['Reinforcement Learning'], diff='Easy', runtime='numpy',
            related=['ppo-clipped-surrogate-objective'])

STATEMENT = r"""
During a PPO update you only have log-probabilities of the sampled actions under the old and new policy. Write `ppo_diagnostics(logp_new, logp_old, clip_eps)` returning:

- `kl_k1`: mean of $\log\pi_{old} - \log\pi_{new}$ (unbiased but noisy, can be negative);
- `kl_k3`: mean of $(r - 1) - \log r$ with $r = \pi_{new}/\pi_{old}$ (unbiased, never negative);
- `clip_frac`: fraction of samples with $|r - 1| > \epsilon$.

Return a dict of floats.
"""

STARTER = '''import numpy as np

def ppo_diagnostics(logp_new, logp_old, clip_eps):
    """Approximate KL(old || new) two ways, and the fraction of clipped ratios."""
    d = np.asarray(logp_new, float) - np.asarray(logp_old, float)
    return {"kl_k1": float(d.mean()), "kl_k3": float((d ** 2).mean() / 2), "clip_frac": float((np.abs(d) > clip_eps).mean())}
'''

SOLUTION = '''import numpy as np

def ppo_diagnostics(logp_new, logp_old, clip_eps):
    lr = np.asarray(logp_new, float) - np.asarray(logp_old, float)
    r = np.exp(lr)
    return {"kl_k1": float((-lr).mean()), "kl_k3": float(((r - 1) - lr).mean()),
            "clip_frac": float((np.abs(r - 1) > clip_eps).mean())}
'''

HINTS = [
    'The samples come from the old policy, so these estimate KL(old || new).',
    'k3 is k1 plus (r - 1), whose expectation under the old policy is zero; it removes variance without adding bias.',
    'Clipping is defined on the ratio, not on the log-ratio.',
]

_r = np.random.default_rng(0)
_old = np.log(_r.uniform(0.05, 0.9, 64))

TESTS = [
    ([0.0, 0.0], [0.0, 0.0], 0.2),
    ([np.log(1.1), np.log(0.7)], [0.0, 0.0], 0.2),
    (_old + 0.05 * _r.normal(size=64), _old, 0.2),
    (_old + 0.3 * _r.normal(size=64), _old, 0.2),
    ([np.log(0.5), np.log(0.25)], [np.log(0.25), np.log(0.5)], 0.1),
]

EXPLANATION = r"""
## Why log these
PPO's clipping limits how far each sample's ratio can push the gradient, but the policy can still drift far over several epochs. The approximate KL and the clip fraction show how far it actually moved.

- A clip fraction near 0 means the update barely changed anything; the learning rate may be too low.
- A clip fraction above roughly 0.2 to 0.3, or a KL that keeps rising across epochs, means updates are too aggressive. Many implementations stop the epoch loop early when KL passes a target such as 0.01 to 0.02.

## The k1 and k3 estimators
Both are unbiased for KL(old‖new) when actions come from the old policy. k1 can be negative on a minibatch, which is confusing on a dashboard. k3 adds $r - 1$, which has mean zero, and the result is always non-negative with lower variance near zero (Schulman's note on approximating KL).

## Interview follow-ups
- Why can PPO's clipped objective still allow a large policy change, and what does early stopping on KL add?
"""
