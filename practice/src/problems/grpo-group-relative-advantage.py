import numpy as np

META = dict(title='GRPO loss with group-relative advantages', tags=['Reinforcement Learning', 'Language Models'],
            diff='Hard', runtime='numpy', lessons=['air-fanout-v1:07.05', 'air-fanout-v1:07.04'],
            related=['ppo-clipped-surrogate-objective', 'rloo-reinforce-leave-one-out', 'ppo-with-reference-policy-kl'])

STATEMENT = r"""
Group Relative Policy Optimization drops the critic. For each prompt it samples a **group** of $G$ responses, scores them, and uses each response's standardised score within its group as the advantage of every one of its tokens:

$$A_{p,g} = \frac{r_{p,g} - \operatorname{mean}_g(r_{p,\cdot})}{\operatorname{std}_g(r_{p,\cdot}) + 10^{-4}}$$

with the **sample** standard deviation (`ddof=1`). The per-token objective is the PPO clipped term minus a KL penalty to the reference model, using the non-negative estimator $k_3$:

$$\rho_t = e^{\log\pi_t - \log\pi_{\text{old},t}}, \qquad k_t = e^{\log\pi_{\text{ref},t} - \log\pi_t} - (\log\pi_{\text{ref},t} - \log\pi_t) - 1$$
$$o_t = \min\big(\rho_t A,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A\big) - \beta\, k_t$$

Each response's objective is the mean of $o_t$ over its valid tokens, and the loss is minus the mean over all $P \cdot G$ responses.

Write `grpo_loss(rewards, logp, logp_old, logp_ref, mask, clip_eps=0.2, beta=0.04)` and return a dict with:

| key | value |
| --- | --- |
| `"advantages"` | float `(P, G)` |
| `"loss"` | float |
| `"grad"` | float `(P, G, T)`, $\partial\,\text{loss} / \partial\,$`logp` (0 on padding) |
| `"mean_kl"` | float, mean of $k_t$ over all valid tokens |
| `"clip_frac"` | float, fraction of valid tokens with $\lvert\rho_t - 1\rvert > \epsilon$ |

## Inputs
- `rewards`: float `(P, G)`, with $G \ge 2$.
- `logp`, `logp_old`, `logp_ref`: float `(P, G, T)`; `mask`: 0/1 `(P, G, T)` with at least one valid token per response. Padded entries are arbitrary.

## Gradient convention
The clipped term contributes $\rho_t A$ to $\partial o_t/\partial\log\pi_t$ when $\rho_t A \le \mathrm{clip}(\rho_t) A$, else 0. The KL term contributes $-\beta(1 - e^{\log\pi_{\text{ref},t} - \log\pi_t})$.
"""

STARTER = '''import numpy as np

def grpo_loss(rewards, logp, logp_old, logp_ref, mask, clip_eps=0.2, beta=0.04):
    """rewards: (P, G); token arrays (P, G, T). Return dict: advantages, loss, grad, mean_kl, clip_frac."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def grpo_loss(rewards, logp, logp_old, logp_ref, mask, clip_eps=0.2, beta=0.04):
    r = np.asarray(rewards, dtype=float)
    m = np.asarray(mask, dtype=float)
    lp, old, ref = (np.asarray(x, dtype=float) * m for x in (logp, logp_old, logp_ref))   # padding -> 0
    P, G = r.shape

    # group-relative advantage: standardise within each prompt's group
    adv = (r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, ddof=1, keepdims=True) + 1e-4)
    A = adv[..., None]                                       # every token of a response shares it

    ratio = np.exp(lp - old)
    unclipped, clipped = ratio * A, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A
    surr = np.minimum(unclipped, clipped)
    d = ref - lp
    kl = np.exp(d) - d - 1.0                                 # k3: unbiased, always >= 0
    obj = surr - beta * kl

    n_tok = m.sum(axis=2)                                    # valid tokens per response
    loss = -((obj * m).sum(axis=2) / n_tok).mean()

    d_obj = np.where(unclipped <= clipped, unclipped, 0.0) - beta * (1.0 - np.exp(d))
    grad = -d_obj * m / n_tok[..., None] / (P * G)
    valid = m.astype(bool)
    return {"advantages": adv, "loss": float(loss), "grad": grad,
            "mean_kl": float(kl[valid].mean()),
            "clip_frac": float((np.abs(ratio - 1) > clip_eps)[valid].mean())}
'''

HINTS = [
    'Advantages need only the `(P, G)` rewards: subtract the row mean and divide by `std(axis=1, ddof=1) + 1e-4`. Then broadcast to tokens with `adv[..., None]`.',
    'Per token: ratio, the PPO `min` of the two terms, and $k_3$ with $d = \\log\\pi_{\\text{ref}} - \\log\\pi$. Mask, sum over T, divide by each response\'s token count, then average over all responses and negate.',
    'For the gradient, $\\partial k_3/\\partial\\log\\pi = 1 - e^{d}$ and the clipped term passes $\\rho A$ only where the unclipped branch is the min. Scale by `-mask / n_tokens / (P * G)`.',
]

_r = np.random.default_rng(271)
def _batch(P, G, T):
    lens = _r.integers(1, T + 1, size=(P, G))
    m = (np.arange(T) < lens[..., None]).astype(int)
    old = -np.abs(_r.normal(size=(P, G, T))) - 0.1
    return (old + 0.2 * _r.normal(size=(P, G, T)), old, old + 0.3 * _r.normal(size=(P, G, T)), m)
B1 = _batch(2, 4, 5)
B2 = _batch(3, 3, 6)
O = np.zeros((1, 2, 2))
TESTS = [
    (np.array([[1.0, 0.0]]), O, O, O, np.ones((1, 2, 2), dtype=int)),
    (np.array([[1.0, 0.0, 0.0, 1.0]]), np.zeros((1, 4, 1)), np.zeros((1, 4, 1)), np.zeros((1, 4, 1)),
     np.ones((1, 4, 1), dtype=int)),
    (np.array([[0.7, 0.7, 0.7]]), np.zeros((1, 3, 2)), np.zeros((1, 3, 2)), np.full((1, 3, 2), -0.5),
     np.array([[[1, 1], [1, 0], [1, 1]]]), 0.2, 0.1),
    (np.array([[2.0, 0.0]]), np.log(np.array([[[1.5, 1.0], [0.5, 1.0]]])), O, O, np.ones((1, 2, 2), dtype=int)),
    (np.round(_r.uniform(size=(2, 4)), 1), *B1),
    ((_r.uniform(size=(3, 3)) < 0.5).astype(float), *B2, 0.1, 0.0),
    (_r.normal(size=(3, 3)), *B2, 0.3, 0.2),
]

EXPLANATION = r"""
## Why a group baseline replaces the critic
For LLM reasoning tasks the reward often arrives only at the end (right or wrong), and training a token-level value model is expensive and noisy. Sampling several answers to the same prompt gives a free, prompt-specific baseline: the group mean. Dividing by the group std also normalises across prompts of different difficulty. Test 2 is a typical binary-reward group: the two correct answers get $+0.866$ and the wrong ones $-0.866$.

## Groups with no signal
If every response in a group gets the same reward (test 3), all advantages are 0 and only the KL term produces a gradient. The $10^{-4}$ in the denominator is what keeps this from being $0/0$: the mean of identical floats can differ from them by an ulp, and without the floor that rounding error would be divided by zero. Prompts that are always solved or never solved carry no learning signal, which is why some pipelines filter them out.

## KL in the loss, not the reward
Unlike RLHF-PPO, GRPO keeps the reward pure and adds the KL directly to the objective with $k_3 = e^{d} - d - 1$, which is non-negative and unbiased for $\mathrm{KL}(\pi \Vert \pi_{\text{ref}})$ when tokens are sampled from $\pi$. Its gradient $1 - e^{d}$ pulls $\log\pi$ toward $\log\pi_{\text{ref}}$ token by token.

## Length weighting
Averaging within each response first gives every response equal weight regardless of length, so a token in a short answer counts more than one in a long answer. Later variants (e.g. DAPO, Dr. GRPO) change this aggregation and drop the std division to remove length and difficulty biases.

## Complexity
$O(PGT)$ time and memory.

## Common bugs
- Population std (`ddof=0`) instead of the sample std the statement specifies.
- Averaging over all valid tokens at once instead of per response then per group.
- Letting padding into the KL mean or the clip fraction.

## Interview follow-ups
- How do RLOO and GRPO differ in their baselines, and which one is unbiased?
- Why might dividing by the group std bias training toward prompts that are too easy or too hard?
"""
