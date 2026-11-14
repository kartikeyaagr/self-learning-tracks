import numpy as np

META = dict(title='GRPO with process supervision: step rewards to token advantages to loss',
            tags=['Reinforcement Learning', 'Language Models'], diff='Hard', runtime='numpy', lessons=['air-fanout-v1:07.04'],
            related=['grpo-group-relative-advantage', 'normalize-grpo-group-advantages', 'score-kl-regularized-policy-batch',
                     'rloo-reinforce-leave-one-out'])

STATEMENT = r"""
GRPO drops PPO's critic: it samples a group of $G$ outputs per question and uses rewards normalised within the group as advantages. With **process supervision**, a reward model scores every reasoning step, and a token's advantage is the sum of the normalised rewards of the steps from its own onward. Outcome supervision is the special case of one step ending at the last token.

Write `grpo_loss(logp, logp_old, logp_ref, mask, step_ends, step_rewards, clip, beta)` returning `(adv, loss)`.

## Inputs
`logp`, `logp_old`, `logp_ref`, `mask` `(G, T)`: per-token log-probabilities under the current, sampling and reference policies, and 1 for real tokens (each output is left-aligned; padding is 0). `step_ends[g]` lists the token index where each step of output $g$ ends (increasing, last one at its final token) and `step_rewards[g]` the matching rewards.

## Advantages
Pool every step reward in the group: $\tilde r = (r - \mathrm{mean})/(\mathrm{std} + 10^{-8})$ with the population std. Token $t$ of output $g$ gets $\hat A_{g,t} = \sum_{\text{steps } j \text{ of } g:\ \text{end}_j \ge t}\tilde r_j$; padding gets 0.

## Loss
With $\rho = e^{\text{logp} - \text{logp\_old}}$ and the unbiased KL estimate $k = e^{\text{logp\_ref} - \text{logp}} - (\text{logp\_ref} - \text{logp}) - 1$:
$$\mathcal L = \frac1G\sum_g\frac{1}{|o_g|}\sum_{t\in o_g}\Big[-\min\big(\rho\hat A,\ \mathrm{clip}(\rho, 1\pm\epsilon)\hat A\big) + \beta k\Big]$$
where $|o_g|$ is the number of real tokens of output $g$. Return `adv` `(G, T)` and `loss` as a float.
"""

STARTER = '''import numpy as np

def grpo_loss(logp, logp_old, logp_ref, mask, step_ends, step_rewards, clip, beta):
    """Return (adv (G, T), loss)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def grpo_loss(logp, logp_old, logp_ref, mask, step_ends, step_rewards, clip, beta):
    lp, lo, lr, m = (np.asarray(a, dtype=float) for a in (logp, logp_old, logp_ref, mask))
    G, T = lp.shape
    allr = np.concatenate([np.asarray(r, dtype=float) for r in step_rewards])
    mu, sd = allr.mean(), allr.std()
    adv = np.zeros((G, T))
    for g in range(G):
        for end, rw in zip(step_ends[g], step_rewards[g]):
            adv[g, :int(end) + 1] += (rw - mu) / (sd + 1e-8)
    adv *= m
    rho = np.exp(lp - lo)
    surr = np.minimum(rho * adv, np.clip(rho, 1 - clip, 1 + clip) * adv)
    d = lr - lp
    kl = np.exp(d) - d - 1
    per = ((-surr + beta * kl) * m).sum(1) / m.sum(1)
    return adv, float(per.mean())
'''

HINTS = [
    'Normalise over all step rewards in the group pooled together, not per output. A step ending at index $e$ adds its normalised reward to tokens $0..e$.',
    'With a single step per output ending at its last token, every token of output $g$ gets the same advantage: that is outcome-supervised GRPO.',
    'Average over each output\'s own real tokens first, then over the group; padding must contribute nothing to either the numerator or the count.',
]

_r = np.random.default_rng(2205)
def _c(G, T, max_steps, clip=0.2, beta=0.04, outcome=False, spread=0.1):
    lens = _r.integers(max(2, T // 2), T + 1, G)
    mask = (np.arange(T)[None, :] < lens[:, None]).astype(float)
    lo = np.log(_r.uniform(0.05, 0.9, (G, T)))
    lp = lo + spread * _r.normal(size=(G, T))
    lr = lo + spread * _r.normal(size=(G, T))
    ends, rews = [], []
    for g in range(G):
        L = int(lens[g])
        if outcome:
            e = [L - 1]
        else:
            k = int(_r.integers(1, max_steps + 1))
            e = sorted(set(int(v) for v in _r.choice(L - 1, min(k, L - 1) - 1, replace=False))) + [L - 1] if k > 1 else [L - 1]
        ends.append(e)
        rews.append([float(v) for v in (_r.random(len(e)) > 0.5)] if outcome else [float(v) for v in _r.random(len(e))])
    return lp * mask, lo * mask, lr * mask, mask, ends, rews, clip, beta
TESTS = [
    _c(2, 4, 1, outcome=True),
    _c(4, 6, 3),
    _c(8, 12, 4, spread=0.3),
    _c(6, 10, 1, outcome=True, beta=0.0),
    _c(5, 16, 5, clip=0.1, beta=0.1, spread=0.5),
]

EXPLANATION = r"""
## Why no critic
PPO for language models needs a value network as large as the policy. GRPO replaces it with a Monte Carlo baseline: sample several answers to the same question and compare each to the group's average reward. That roughly halves the memory and compute of RL training, which is what DeepSeekMath introduced it for and what later reasoning models adopted.

## Outcome vs process supervision
With outcome rewards every token of an answer gets the same credit. A process reward model scores each step, so a solution that goes wrong at step 3 can still get positive credit for steps 1 and 2: the sum over later steps assigns each token the reward of everything it leads to. DeepSeekMath reported process supervision helping in their experiments, though later work (notably DeepSeek-R1) relied on outcome rewards because process reward models are hard to build and can be exploited.

## The KL term
GRPO adds the KL to the reference policy directly to the loss (instead of folding it into the reward as RLHF-PPO does), using the estimator $e^{d} - d - 1$, which is unbiased and always non-negative.

## Common bugs
- Normalising each output's step rewards separately.
- Including padding tokens in the per-output average.
- Using the sample std; the spec fixes the population std, and degenerate groups (all rewards equal) give zero advantages thanks to the $10^{-8}$.

## Interview follow-ups
- What happens to the gradient when every answer in a group gets the same reward, and how do later variants filter such groups?
- Why might a per-output token average bias the model towards shorter or longer answers?
"""
