import numpy as np

META = dict(title='Bradley-Terry reward model loss with a margin', tags=['Reinforcement Learning', 'Loss Functions', 'Language Models'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:07.04'],
            related=['dpo-closed-form-loss', 'log-loss-binary-cross-entropy', 'ppo-with-reference-policy-kl'])

STATEMENT = r"""
An RLHF reward model is trained on human comparisons: for each prompt, a **chosen** and a **rejected** response. The Bradley-Terry model says the chosen one wins with probability $\sigma(r_c - r_r)$, where $r$ is the scalar reward the model assigns. Training minimises the negative log-likelihood, optionally requiring a margin $m_i$ (larger for pairs the labellers felt strongly about):

$$z_i = r_{c,i} - r_{r,i} - m_i, \qquad L = -\frac1B \sum_i \log \sigma(z_i)$$

Write `bt_loss(r_chosen, r_rejected, margin=0.0)` and return a dict with:

| key | value |
| --- | --- |
| `"loss"` | float, $L$ |
| `"accuracy"` | float, fraction of pairs with $r_c > r_r$ (strictly; the margin is **not** used here) |
| `"grad_chosen"` | float `(B,)`, $\partial L / \partial r_c$ |
| `"grad_rejected"` | float `(B,)`, $\partial L / \partial r_r$ |

## Inputs
- `r_chosen`, `r_rejected`: float arrays `(B,)`.
- `margin`: a float or a float array `(B,)`.

Rewards can be far apart (e.g. a gap of 1000). The loss must stay finite and accurate: use $-\log\sigma(z) = \log(1 + e^{-z})$ computed stably.
"""

STARTER = '''import numpy as np

def bt_loss(r_chosen, r_rejected, margin=0.0):
    """Return dict: loss, accuracy, grad_chosen (B,), grad_rejected (B,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def bt_loss(r_chosen, r_rejected, margin=0.0):
    rc, rr = np.asarray(r_chosen, dtype=float), np.asarray(r_rejected, dtype=float)
    B = len(rc)
    z = rc - rr - margin
    loss = np.logaddexp(0.0, -z).mean()            # -log sigmoid(z) = log(1 + exp(-z)), overflow-free
    s_neg = np.exp(-np.logaddexp(0.0, z))          # sigmoid(-z), stable for any z
    g = -s_neg / B                                  # dL/dz
    return {"loss": float(loss), "accuracy": float((rc > rr).mean()),
            "grad_chosen": g, "grad_rejected": -g}
'''

HINTS = [
    'Everything depends on one number per pair, $z = r_c - r_r - m$. Broadcasting handles a scalar or per-pair margin.',
    '$-\\log\\sigma(z) = \\log(1 + e^{-z})$, which is `np.logaddexp(0, -z)`: no overflow for very negative $z$, no `log(0)` for very positive.',
    '$\\frac{d}{dz}[-\\log\\sigma(z)] = -\\sigma(-z)$. Divide by $B$; the rejected reward gets the opposite sign. A stable $\\sigma(-z)$ is `exp(-logaddexp(0, z))`.',
]

_r = np.random.default_rng(241)
RC, RR = _r.normal(size=8), _r.normal(size=8)
TESTS = [
    (np.array([1.0, 0.0]), np.array([0.0, 0.0])),
    (np.array([2.0, -1.0, 0.5]), np.array([1.0, 1.0, 0.5]), 0.5),
    (np.array([1000.0]), np.array([0.0])),
    (np.array([0.0]), np.array([1000.0])),
    (np.array([3.0, 3.0]), np.array([1.0, 2.5]), np.array([1.0, 1.0])),
    (RC, RR),
    (RC * 4, RR * 4, np.abs(_r.normal(size=8))),
]

EXPLANATION = r"""
## Why Bradley-Terry
It assumes each response has a latent quality $r$ and that preferences are noisy comparisons of those qualities through a logistic link. Then fitting the reward model is exactly binary logistic regression on the difference $r_c - r_r$. Only differences are identified: adding a constant to every reward changes nothing, which is why RLHF pipelines often normalise rewards before PPO.

## The margin
With $m > 0$ the loss keeps pushing until the gap exceeds $m$, so "much better" pairs get separated further than "slightly better" ones. Accuracy ignores the margin because it measures the ranking, not the confidence (test 5: both pairs are ranked correctly, but the second is inside the margin and still has a large gradient).

## Stability
Test 3: $z = 1000$, the loss is $\approx e^{-1000} \approx 0$ and the gradient vanishes. Test 4: $z = -1000$. `-np.log(sigmoid(z))` would compute `log(0) = -inf`; `logaddexp` returns 1000 exactly, and the gradient saturates at $-1/B$, the maximum push.

## Complexity
$O(B)$.

## Common bugs
- `np.log(1 + np.exp(-z))` overflows for very negative $z$.
- Treating ties ($r_c = r_r$) as correct in the accuracy.
- Sign errors in `grad_rejected`: it must be the negative of `grad_chosen`.

## Interview follow-ups
- With $K$ ranked responses per prompt, how do you build the loss? (All $\binom{K}{2}$ pairs, or a Plackett-Luce likelihood.)
- Why does reward-model overoptimisation happen during PPO, and how does the KL penalty to the reference policy help?
"""
