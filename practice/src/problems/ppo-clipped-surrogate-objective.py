import numpy as np

META = dict(title='PPO clipped surrogate: loss, gradient and diagnostics', tags=['Reinforcement Learning', 'Loss Functions'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:07.04'],
            related=['gae-advantages', 'actor-critic-a2c-loss-with-entropy', 'ppo-with-reference-policy-kl'])

STATEMENT = r"""
PPO reuses a batch collected by an old policy for several gradient steps. To stop the new policy from drifting too far, it clips the probability ratio $\rho_i = \exp(\log\pi_{\text{new}}(a_i|s_i) - \log\pi_{\text{old}}(a_i|s_i))$:

$$L = -\frac1B \sum_i \min\big(\rho_i A_i,\ \mathrm{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\, A_i\big)$$

Write `ppo_clip(logp_new, logp_old, advantages, clip_eps=0.2)` and return a dict with:

| key | value |
| --- | --- |
| `"loss"` | float, $L$ above |
| `"grad"` | float `(B,)`, $\partial L / \partial\,$`logp_new` |
| `"clip_frac"` | float, fraction of samples with $\lvert\rho_i - 1\rvert > \epsilon$ |
| `"approx_kl"` | float, $\frac1B\sum_i \big[(\rho_i - 1) - \log\rho_i\big]$, a non-negative estimate of $\mathrm{KL}(\pi_{\text{old}} \Vert \pi_{\text{new}})$ |

## Gradient convention
For sample $i$, if $\rho_i A_i \le \mathrm{clip}(\rho_i)\, A_i$ the unclipped term is the one selected and the gradient is $-\rho_i A_i / B$ (since $\partial\rho/\partial\log\pi_{\text{new}} = \rho$). Otherwise the clipped term is selected; it is constant in $\log\pi_{\text{new}}$, so the gradient is 0.

All inputs are float arrays of shape `(B,)`. `advantages` are used as given (no normalisation).
"""

STARTER = '''import numpy as np

def ppo_clip(logp_new, logp_old, advantages, clip_eps=0.2):
    """Return dict with loss, grad (B,), clip_frac, approx_kl."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def ppo_clip(logp_new, logp_old, advantages, clip_eps=0.2):
    log_ratio = np.asarray(logp_new, dtype=float) - np.asarray(logp_old, dtype=float)
    ratio = np.exp(log_ratio)
    A = np.asarray(advantages, dtype=float)
    B = len(A)
    unclipped = ratio * A
    clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A
    use_unclipped = unclipped <= clipped                 # the pessimistic (min) branch
    loss = -np.minimum(unclipped, clipped).mean()
    grad = np.where(use_unclipped, -unclipped / B, 0.0)  # clipped branch is flat in logp_new
    return {"loss": float(loss), "grad": grad,
            "clip_frac": float((np.abs(ratio - 1) > clip_eps).mean()),
            "approx_kl": float(((ratio - 1) - log_ratio).mean())}
'''

HINTS = [
    'Form the ratio from the log-probabilities: `exp(logp_new - logp_old)`. Then build both candidate terms elementwise.',
    'The loss is minus the mean of the elementwise minimum. Record which branch the minimum picked: `unclipped <= clipped`.',
    'Where the unclipped branch is selected the gradient is `-ratio * A / B`; elsewhere it is 0. Note that for $A < 0$ the gradient still flows when the ratio is *above* $1+\\epsilon$.',
]

_r = np.random.default_rng(231)
LO = -np.abs(_r.normal(size=10)) - 0.5
TESTS = [
    (np.log(np.array([1.0, 1.5])), np.zeros(2), np.array([1.0, 1.0])),
    (np.log(np.array([1.5, 0.5])), np.zeros(2), np.array([-1.0, 1.0])),
    (np.log(np.array([0.5, 1.5])), np.zeros(2), np.array([-1.0, 1.0])),
    (np.array([-1.0, -2.0, -0.5]), np.array([-1.0, -2.0, -0.5]), np.array([0.3, -2.0, 1.0])),
    (np.log(np.array([1.1, 0.9])), np.zeros(2), np.array([2.0, -3.0]), 0.05),
    (LO + 0.3 * _r.normal(size=10), LO, _r.normal(size=10)),
    (LO + 0.3 * _r.normal(size=10), LO, _r.normal(size=10), 0.1),
]

EXPLANATION = r"""
## What the min does
Look at each sign of $A$:
- $A > 0$ (good action): the objective $\rho A$ grows with $\rho$ but stops at $1 + \epsilon$. Beyond that the gradient is zero, so there is no incentive to push the probability up further on this batch (test 1, second sample).
- $A < 0$ (bad action): the objective stops improving once $\rho$ drops below $1 - \epsilon$.
- The min makes it pessimistic: if the ratio has moved the *wrong* way (raised a bad action or lowered a good one, the two samples of test 2) the unclipped term is smaller and the gradient keeps flowing to undo the damage. Test 3 is the mirror image: both samples moved the right way past the clip, so both gradients are 0.

So clipping only removes the incentive to move further in the direction that already looks good. It doesn't hard-limit the ratio.

## Diagnostics
- `clip_frac` near 0 means the policy barely moved; near 0.3 or more, the step size (or epochs) is probably too large.
- `approx_kl` uses $(\rho - 1) - \log\rho \ge 0$, an unbiased and always non-negative estimator. Many implementations stop the epoch early when it exceeds a target such as 0.01-0.02.
- At the first epoch $\pi_{\text{new}} = \pi_{\text{old}}$, so $\rho = 1$, the loss is $-\bar A$, nothing is clipped and the KL is 0 (test 4).

## Complexity
$O(B)$.

## Common bugs
- Computing the ratio as `p_new / p_old` from probabilities, which underflows for long sequences of tokens. Work in log space.
- Clipping the advantage instead of the ratio.
- Using `max` instead of `min` (or forgetting the leading minus).

## Interview follow-ups
- Why does PPO also clip the value function in some implementations, and is it a good idea?
- Compare the clipped objective to the KL-penalty variant of PPO and to TRPO's hard KL constraint.
"""
