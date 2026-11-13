import numpy as np

META = dict(title='RLOO: REINFORCE with a leave-one-out baseline', tags=['Reinforcement Learning', 'Language Models'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:07.05', 'air-fanout-v1:07.02'],
            related=['grpo-group-relative-advantage', 'reinforce-with-baseline', 'reinforce-gradient'])

STATEMENT = r"""
RLOO samples $K$ responses per prompt and uses, as the baseline for response $k$, the mean reward of the **other** $K - 1$ responses to the same prompt. The baseline then doesn't depend on the response it is applied to, so the policy gradient stays unbiased, and no critic is needed.

Each response is treated as a single action, and its reward includes a sequence-level KL penalty to the reference model:

$$\text{KL}_{p,k} = \sum_{t\ \text{valid}} \big(\log\pi_t - \log\pi_{\text{ref},t}\big), \qquad R_{p,k} = s_{p,k} - \beta\, \text{KL}_{p,k}$$
$$b_{p,k} = \frac{1}{K-1}\sum_{j \ne k} R_{p,j}, \qquad A_{p,k} = R_{p,k} - b_{p,k}$$
$$L = -\frac{1}{PK} \sum_{p,k} A_{p,k} \sum_{t\ \text{valid}} \log\pi_t$$

Write `rloo_loss(scores, logp, logp_ref, mask, beta=0.05)` and return a dict with `"rewards"` `(P, K)`, `"advantages"` `(P, K)`, `"loss"` (float) and `"grad"` `(P, K, T)`, the gradient of $L$ w.r.t. `logp`.

## Inputs
- `scores`: float `(P, K)`, the reward-model scores, with $K \ge 2$.
- `logp`, `logp_ref`: float `(P, K, T)`, per-token log-probs of each sampled response; `mask`: 0/1 `(P, K, T)`. Padding entries are arbitrary.

## Convention
$R$ and $A$ are constants for the gradient (the KL inside $R$ is detached, as in practice), so only the $\sum_t \log\pi_t$ factor is differentiated. The gradient is 0 on padding.
"""

STARTER = '''import numpy as np

def rloo_loss(scores, logp, logp_ref, mask, beta=0.05):
    """scores: (P, K); token arrays (P, K, T). Return dict: rewards, advantages, loss, grad."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def rloo_loss(scores, logp, logp_ref, mask, beta=0.05):
    m = np.asarray(mask, dtype=float)
    lp, ref = np.asarray(logp, dtype=float) * m, np.asarray(logp_ref, dtype=float) * m
    s = np.asarray(scores, dtype=float)
    P, K = s.shape
    R = s - beta * (lp - ref).sum(axis=2)                     # score minus sequence KL estimate
    baseline = (R.sum(axis=1, keepdims=True) - R) / (K - 1)   # mean of the other K-1 rewards
    A = R - baseline
    seq_logp = lp.sum(axis=2)
    loss = -(A * seq_logp).mean()
    grad = -A[..., None] * m / (P * K)                        # A is a constant; d seq_logp / d logp_t = mask_t
    return {"rewards": R, "advantages": A, "loss": float(loss), "grad": grad}
'''

HINTS = [
    'Mask the token arrays first, then the sequence KL and the sequence log-prob are both sums over the last axis.',
    'Leave-one-out without a loop: the sum of the others is the group total minus yourself, so `b = (R.sum(1, keepdims=True) - R) / (K - 1)`.',
    'With $A$ held constant, $\\partial L/\\partial\\log\\pi_t = -A \\cdot \\text{mask}_t / (PK)$ for every token of the response.',
]

_r = np.random.default_rng(281)
def _batch(P, K, T):
    lens = _r.integers(1, T + 1, size=(P, K))
    m = (np.arange(T) < lens[..., None]).astype(int)
    lp = -np.abs(_r.normal(size=(P, K, T))) - 0.1
    return lp, lp + 0.2 * _r.normal(size=(P, K, T)), m
L1, F1, M1 = _batch(2, 4, 5)
L2, F2, M2 = _batch(3, 2, 4)
TESTS = [
    (np.array([[1.0, 0.0]]), np.full((1, 2, 1), -1.0), np.full((1, 2, 1), -1.0), np.ones((1, 2, 1), dtype=int)),
    (np.array([[3.0, 1.0, 2.0]]), np.zeros((1, 3, 2)), np.zeros((1, 3, 2)), np.ones((1, 3, 2), dtype=int)),
    (np.array([[1.0, 1.0]]), np.array([[[-1.0, -2.0], [-1.0, -1.0]]]), np.array([[[-1.0, -1.0], [-1.0, -1.0]]]),
     np.array([[[1, 1], [1, 0]]]), 0.5),
    (np.array([[0.5, 0.5, 0.5, 0.5]]), np.full((1, 4, 3), -0.3), np.full((1, 4, 3), -0.3), np.ones((1, 4, 3), dtype=int)),
    (_r.normal(size=(2, 4)), L1, F1, M1),
    ((_r.uniform(size=(3, 2)) < 0.5).astype(float), L2, F2, M2, 0.1),
]

EXPLANATION = r"""
## Why leave one out
Using the plain group mean as a baseline (as GRPO does) includes the response's own reward, so the baseline is correlated with the action and the estimator is slightly biased. Leaving it out removes the correlation: $b_{p,k}$ is built only from independent samples, so $\mathbb{E}[b\,\nabla\log\pi(y_k)] = 0$ exactly. There's a neat identity too: $A_{p,k} = \frac{K}{K-1}(R_{p,k} - \bar R_p)$, so RLOO is the group-mean baseline rescaled by $K/(K-1)$. Test 2 shows it: rewards 3, 1, 2 give advantages 1.5, -1.5, 0.

## Why a sequence-level action
RLOO treats the whole response as one action. In RLHF only the final reward is informative, so modelling each token as an action with its own value adds a critic and variance without buying much. That is the argument for REINFORCE-style methods over PPO for LLMs.

## Edge cases
- Test 3: equal scores, but response 0 drifted from the reference (KL 1), so its reward drops by $\beta$ and it gets a negative advantage. The padded token of response 1 is ignored.
- Test 4: identical rewards, so every advantage, the loss and the gradient are 0.

## Complexity
$O(PKT)$.

## Common bugs
- Dividing by $K$ instead of $K - 1$ in the baseline.
- Letting gradient flow through the KL inside the reward.
- Summing padding tokens into the KL or the log-prob.

## Interview follow-ups
- Show that RLOO's advantage is $\frac{K}{K-1}$ times the mean-baseline advantage.
- How would you add PPO-style clipping and multiple epochs to RLOO, and what would you lose?
"""
