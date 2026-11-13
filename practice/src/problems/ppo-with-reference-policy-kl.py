import numpy as np

META = dict(title='RLHF PPO step: KL-shaped rewards to clipped loss', tags=['Reinforcement Learning', 'Language Models'],
            diff='Hard', runtime='numpy', lessons=['air-fanout-v1:07.04', 'air-fanout-v1:01.12'],
            related=['gae-advantages', 'ppo-clipped-surrogate-objective', 'bradley-terry-reward-model-loss'])

STATEMENT = r"""
In RLHF with PPO, each sampled response is a short episode with one action per token. The reward model scores the whole response once, and a per-token KL penalty keeps the policy close to the frozen reference (SFT) model. Implement the full advantage-and-loss computation for one batch.

Write `rlhf_ppo(logp_new, logp_old, logp_ref, values, scores, mask, beta=0.1, gamma=1.0, lam=0.95, clip_eps=0.2)` and return a dict with keys `"rewards"`, `"advantages"`, `"returns"` (float `(B, T)` each) and `"loss"` (float).

## Inputs
- `logp_new`, `logp_old`, `logp_ref`: float `(B, T)`, per-token log-probs of the sampled tokens under the policy being trained, the policy that generated the batch, and the reference model.
- `values`: float `(B, T)`, the critic's value at each token position.
- `scores`: float `(B,)`, the reward model's score for each response.
- `mask`: `(B, T)` of 0/1, right-padded (1s then 0s), at least one 1 per row. Padded entries of the other arrays are arbitrary.

## Steps
1. **Rewards.** $r_t = -\beta\,(\log\pi_{\text{old}} - \log\pi_{\text{ref}})_t$ on valid tokens, plus `scores[i]` added at the **last valid token** of row $i$. Padding gets 0.
2. **GAE** per row over valid tokens: $\delta_t = r_t + \gamma V_{t+1} - V_t$ and $\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$, where $V_{t+1}$ and $\hat A_{t+1}$ are 0 past the last valid token (the response ends there).
3. **Returns** $= \hat A + V$ on valid tokens, 0 on padding (computed from the un-whitened advantages).
4. **Whiten** the advantages over all valid tokens of the batch: subtract their mean, divide by (population std + $10^{-8}$). Padding stays 0. This is the `"advantages"` output.
5. **Loss**: with $\rho = \exp(\log\pi_{\text{new}} - \log\pi_{\text{old}})$ and whitened $A$, the per-token loss is $-\min(\rho A, \mathrm{clip}(\rho, 1-\epsilon, 1+\epsilon) A)$. `"loss"` is its mean over **valid tokens** of the whole batch.
"""

STARTER = '''import numpy as np

def rlhf_ppo(logp_new, logp_old, logp_ref, values, scores, mask, beta=0.1, gamma=1.0, lam=0.95, clip_eps=0.2):
    """All (B, T) except scores (B,). Return dict: rewards, advantages, returns, loss."""
    rewards = np.zeros(np.shape(mask))
    # TODO: KL-shaped rewards, GAE, returns, whitening, clipped loss
    return {"rewards": rewards, "advantages": rewards, "returns": rewards, "loss": 0.0}
'''

SOLUTION = '''import numpy as np

def rlhf_ppo(logp_new, logp_old, logp_ref, values, scores, mask, beta=0.1, gamma=1.0, lam=0.95, clip_eps=0.2):
    m = np.asarray(mask, dtype=float)
    new, old, ref, V = (np.asarray(x, dtype=float) * m for x in (logp_new, logp_old, logp_ref, values))
    B, T = m.shape
    lengths = m.sum(axis=1).astype(int)

    # 1. per-token KL penalty, plus the sequence score on the final token
    rewards = -beta * (old - ref)
    rewards[np.arange(B), lengths - 1] += scores

    # 2. GAE backwards in time; m[:, t+1] = 0 cuts both the bootstrap and the recursion at the end
    adv = np.zeros((B, T))
    running = np.zeros(B)
    for t in reversed(range(T)):
        nxt = m[:, t + 1] if t + 1 < T else np.zeros(B)
        v_next = V[:, t + 1] if t + 1 < T else np.zeros(B)
        delta = rewards[:, t] + gamma * nxt * v_next - V[:, t]
        running = (delta + gamma * lam * nxt * running) * m[:, t]
        adv[:, t] = running

    # 3. returns for the value loss come from the raw advantages
    returns = (adv + V) * m

    # 4. whiten over valid tokens only
    valid = m.astype(bool)
    mu, sd = adv[valid].mean(), adv[valid].std()
    white = np.where(valid, (adv - mu) / (sd + 1e-8), 0.0)

    # 5. clipped surrogate, averaged over valid tokens
    ratio = np.exp(new - old)
    per_tok = -np.minimum(ratio * white, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * white)
    loss = (per_tok * m).sum() / m.sum()
    return {"rewards": rewards, "advantages": white, "returns": returns, "loss": float(loss)}
'''

HINTS = [
    'Multiply every per-token input by the mask up front so padding is all zeros. The last valid index of row $i$ is `mask[i].sum() - 1`.',
    'Run GAE backwards over columns for all rows at once. Use `mask[:, t+1]` (0 past the end) to zero both $\\gamma V_{t+1}$ and the carried $\\hat A_{t+1}$, and multiply the result by `mask[:, t]` so padding stays 0.',
    'Compute returns before whitening. Whiten with `adv[mask == 1]` statistics (`np.std` is the population std), then the loss is the clipped PPO term summed over valid tokens and divided by their count.',
]

_r = np.random.default_rng(261)
def _batch(B, T, lens):
    m = (np.arange(T) < np.array(lens)[:, None]).astype(int)
    old = -np.abs(_r.normal(size=(B, T))) - 0.2
    ref = old + 0.3 * _r.normal(size=(B, T))
    new = old + 0.25 * _r.normal(size=(B, T))
    return new, old, ref, _r.normal(size=(B, T)), _r.normal(size=B) * 2, m
N1 = _batch(3, 5, [5, 2, 4])
N2 = _batch(4, 7, [7, 7, 1, 3])
Z2 = np.zeros((2, 3))
TESTS = [
    (np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), np.array([1.0]), np.array([[1, 1, 1]])),
    (Z2, Z2, np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]), np.array([[0.2, 0.1, 9.0], [0.0, 0.3, 0.6]]), np.array([1.0, -1.0]),
     np.array([[1, 1, 0], [1, 1, 1]]), 0.2, 1.0, 1.0),
    (np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.array([[0.5, 7.0]]), np.array([3.0]), np.array([[1, 0]])),
    (np.log(np.array([[1.5, 0.5, 1.0]])), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), np.array([2.0]),
     np.array([[1, 1, 1]]), 0.1, 0.9, 0.5),
    (*N1,),
    (*N1, 0.05, 0.99, 0.9, 0.1),
    (*N2,),
]

EXPLANATION = r"""
## Why the reward looks like this
The RLHF objective is $\mathbb{E}[\text{score}] - \beta\,\mathrm{KL}(\pi \Vert \pi_{\text{ref}})$. The sequence KL decomposes into a sum over tokens of $\log\pi(y_t) - \log\pi_{\text{ref}}(y_t)$ (in expectation over sampled tokens), so it can be paid token by token as a negative reward. The score arrives only at the end. GAE then spreads that sparse terminal reward back over the tokens, with the critic reducing variance. $\gamma = 1$ is standard because responses are short and there is no reason to discount within one.

## Which log-probs go where
- The KL penalty uses $\pi_{\text{old}}$ (the policy that sampled the data) so the rewards are fixed for all PPO epochs over this batch.
- The ratio uses $\pi_{\text{new}}$ vs $\pi_{\text{old}}$, as in standard PPO.
- $\pi_{\text{ref}}$ never changes during RLHF.

## Masks
Test 2 checks that a garbage value (9.0) beyond the end of row 0 is never bootstrapped from. Test 3 has a one-token response: the score lands on token 0, $\hat A_0 = 3 - 0.5$, and after whitening a single valid token becomes exactly 0.

## Whitening
Normalising advantages over the batch makes the loss scale independent of the reward model's scale. It is done over valid tokens only, since padding zeros would bias both the mean and the std.

## Complexity
$O(BT)$ time and memory. The GAE loop is over $T$ only; rows are vectorised.

## Common bugs
- Adding the score to the last *column* instead of the last *valid* token.
- Bootstrapping from the value of a padding position.
- Whitening before computing returns (the value head then regresses toward normalised targets).
- Averaging the loss per sequence and then over sequences: a different weighting (that's GRPO's convention).

## Interview follow-ups
- How does the "adaptive KL controller" choose $\beta$ to hit a target KL?
- Why do some implementations use $\exp(\log\pi_{\text{ref}} - \log\pi) - (\log\pi_{\text{ref}} - \log\pi) - 1$ as the KL term instead?
"""
