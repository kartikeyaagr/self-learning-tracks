import numpy as np

META = dict(title='KL-regularized rewards with three KL estimators', tags=['Reinforcement Learning', 'Language Models'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:07.04'],
            related=['ppo-with-reference-policy-kl', 'kl-divergence-discrete-and-continuous', 'normalize-grpo-group-advantages'])

STATEMENT = r"""
RL fine-tuning of a language model maximises reward while staying close to a frozen reference model:

$$\max_\pi\ \mathbb{E}_{y \sim \pi}\big[r(y)\big] - \beta\, \mathrm{KL}\big(\pi \,\|\, \pi_\text{ref}\big)$$

The KL can't be computed exactly over all possible responses, so it is estimated from the sampled tokens. With the per-token log-ratio $\delta_t = \log\pi(y_t) - \log\pi_\text{ref}(y_t)$ there are three common estimators:

| name | per-token term | notes |
|---|---|---|
| $k_1$ | $\delta_t$ | unbiased, can be negative |
| $k_2$ | $\tfrac12 \delta_t^2$ | biased, always $\ge 0$ |
| $k_3$ | $e^{-\delta_t} - 1 + \delta_t$ | unbiased, always $\ge 0$ |

Write `kl_scored_batch(rewards, logp, logp_ref, mask, beta)` and return a dict.

## Inputs
- `rewards`: float64 `(B,)`, one scalar reward per sampled response.
- `logp`, `logp_ref`: float64 `(B, T)`, log-probabilities of the sampled tokens under the policy and the reference.
- `mask`: 0/1 `(B, T)`, 1 on response tokens (at least one per row). Other entries of `logp` and `logp_ref` are arbitrary.
- `beta`: float.

## Output
| key | value |
|---|---|
| `"kl_k1"`, `"kl_k2"`, `"kl_k3"` | float64 `(B,)`, the per-token terms summed over each response's tokens |
| `"penalized"` | float64 `(B,)`, `rewards - beta * kl_k1` |
| `"objective"` | float, mean of `penalized` |
| `"kl_per_token"` | float, $\sum k_1$ over all response tokens divided by the number of response tokens |
"""

STARTER = '''import numpy as np

def kl_scored_batch(rewards, logp, logp_ref, mask, beta):
    """Return a dict: kl_k1, kl_k2, kl_k3 (B,), penalized (B,), objective, kl_per_token."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def kl_scored_batch(rewards, logp, logp_ref, mask, beta):
    m = np.asarray(mask, dtype=float)
    # Zero the log-ratio on padding first, so garbage there can't overflow exp() in k3.
    d = np.where(m > 0, np.asarray(logp, dtype=float) - np.asarray(logp_ref, dtype=float), 0.0)
    k1 = (d * m).sum(axis=1)
    k2 = (0.5 * d * d * m).sum(axis=1)
    k3 = ((np.expm1(-d) + d) * m).sum(axis=1)          # expm1 keeps precision when d is tiny
    penalized = np.asarray(rewards, dtype=float) - beta * k1
    return {"kl_k1": k1, "kl_k2": k2, "kl_k3": k3, "penalized": penalized,
            "objective": float(penalized.mean()), "kl_per_token": float(k1.sum() / m.sum())}
'''

HINTS = [
    'Everything starts from the log-ratio `d = logp - logp_ref`. Set it to 0 on padding before applying any nonlinear function, or garbage values can produce `inf`.',
    'Each estimator is an elementwise function of `d`, summed over the time axis with the mask.',
    'The penalised reward uses the sequence-level $k_1$ sum. The per-token KL pools all response tokens in the batch before dividing.',
]

_r = np.random.default_rng(23)
_lp = -np.abs(_r.normal(1.0, 0.7, size=(4, 6)))
_lr = _lp + _r.normal(0.0, 0.3, size=(4, 6))
_m = (np.arange(6)[None, :] < np.array([6, 3, 1, 5])[:, None]).astype(np.int64)
TESTS = [
    (np.array([1.0]), np.array([[-1.0, -2.0]]), np.array([[-1.5, -1.0]]), np.array([[1, 1]]), 0.1),
    (np.array([0.0, 1.0]), np.log(np.array([[0.5, 0.5], [0.9, 0.1]])), np.log(np.array([[0.5, 0.5], [0.3, 0.7]])),
     np.array([[1, 1], [1, 0]]), 0.5),
    (np.array([0.2, -0.4, 1.0, 0.0]), _lp, _lr, _m, 0.05),
    (np.array([0.2, -0.4, 1.0, 0.0]), _lp, _lp, _m, 0.05),
    (np.array([1.0, 2.0]), np.array([[-0.1, -800.0], [-0.2, 900.0]]), np.array([[-0.1, 5.0], [-0.3, -900.0]]),
     np.array([[1, 0], [1, 0]]), 1.0),
    (np.array([3.0, 3.0, 3.0]), np.array([[-1e-9, -2e-9], [-1.0, -1.0], [-3.0, -0.5]]),
     np.array([[-2e-9, -1e-9], [-4.0, -4.0], [-0.5, -3.0]]), np.ones((3, 2), dtype=np.int64), 0.2),
    (_r.normal(size=8), -np.abs(_r.normal(size=(8, 12))), -np.abs(_r.normal(size=(8, 12))),
     (_r.random((8, 12)) < 0.7).astype(np.int64) | np.eye(8, 12, dtype=np.int64), 0.02),
]

EXPLANATION = r"""
## Where the estimators come from
With samples $y \sim \pi$, $\mathrm{KL}(\pi\|\pi_\text{ref}) = \mathbb{E}_\pi[\delta]$, so $k_1 = \delta$ is unbiased, but individual samples can be negative and its variance is high when $\pi \approx \pi_\text{ref}$. Writing $\rho = \pi_\text{ref}/\pi = e^{-\delta}$, we have $\mathbb{E}_\pi[\rho - 1] = 0$, so adding $\rho - 1$ to $k_1$ keeps it unbiased while making each term non-negative (because $e^{x} \ge 1 + x$): that's $k_3$. $k_2$ is a second-order approximation, biased but low-variance. GRPO uses $k_3$ inside the loss; PPO-style RLHF typically folds $k_1$ into the reward, which is what `penalized` does.

## Numerics
Test 5 puts log-probs of $\pm 900$ in the padding: `exp(900)` overflows to `inf`, and `inf * 0` is `nan`. Zeroing the log-ratio **before** the nonlinearity avoids it; multiplying by the mask afterwards doesn't. Test 6 has log-ratios of $10^{-9}$, where `exp(-d) - 1` loses most of its digits to cancellation and `np.expm1` doesn't.

## Complexity
$O(BT)$, three masked reductions.

## Common bugs
- Swapping the direction: $\log\pi_\text{ref} - \log\pi$ gives the negative of $k_1$.
- Averaging the KL per token for the reward but summing it elsewhere: be consistent, since $\beta$ is tuned for one convention.
- Masking by multiplication after `exp`, as above.

## Interview follow-ups
- Why can a negative $k_1$ sum on a single sample be a problem when used as a reward penalty, and why is $k_3$ preferred inside the loss?
- What happens to the KL term as $\beta \to 0$ in practice? (Reward hacking and drift into degenerate text.)
"""
