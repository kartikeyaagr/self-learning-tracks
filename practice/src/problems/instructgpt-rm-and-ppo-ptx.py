import numpy as np

META = dict(title='InstructGPT in miniature: train a reward model, then optimise the KL-penalised objective with pretraining mix',
            tags=['Reinforcement Learning', 'Language Models'], diff='Hard', runtime='numpy', lessons=['dlcore:08.07'],
            related=['bradley-terry-reward-model-loss', 'score-kl-regularized-policy-batch', 'ppo-with-reference-policy-kl',
                     'canonicalize-pairwise-preference-data'])

STATEMENT = r"""
InstructGPT's pipeline has three stages: supervised fine-tuning, a reward model trained on human comparisons, and RL against that reward with a KL penalty to the SFT model plus a pretraining term ("PPO-ptx"). Run stages 2 and 3 on a toy problem where each prompt has only $K$ possible responses, so expectations can be computed exactly instead of sampled.

Write `instructgpt_toy(Phi, comparisons, w0, lr_rm, n_rm, pi_sft, ptx, beta, gamma, lr_rl, n_rl)` returning `(rm_losses, w, rl_objective, pi)`.

## Stage 2: reward model
Response features `Phi` `(P, K, F)`; reward $r_{pk} = \Phi_{pk}\cdot w$. `comparisons` is a list of `(p, i, j)`: for prompt $p$, response $i$ was preferred over $j$. Run `n_rm` full-batch gradient-descent steps on $\mathcal L = -\mathrm{mean}\log\sigma(r_{pi} - r_{pj})$ from `w0`, recording $\mathcal L$ before each step.

## Stage 3: policy
Logits $Z$ `(P, K)` start at $\log\pi_{SFT}$; $\pi = \mathrm{softmax}(Z)$ per prompt. With the trained reward model fixed,
$$J(Z) = \frac1P\sum_p\sum_k \pi_{pk}\Big(r_{pk} - \beta\log\frac{\pi_{pk}}{\pi^{SFT}_{pk}}\Big) + \gamma\,\frac{1}{|\text{ptx}|}\sum_{(p, k)\in\text{ptx}}\log\pi_{pk}$$
where `ptx` is a list of `(p, k)` pairs standing in for pretraining data. Run `n_rl` gradient-**ascent** steps on $Z$, recording $J$ before each.

Return `rm_losses` `(n_rm,)`, the trained `w`, `rl_objective` `(n_rl,)` and the final policy $\pi$ `(P, K)`.
"""

STARTER = '''import numpy as np

def instructgpt_toy(Phi, comparisons, w0, lr_rm, n_rm, pi_sft, ptx, beta, gamma, lr_rl, n_rl):
    """Return (rm_losses, w, rl_objective, pi)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def instructgpt_toy(Phi, comparisons, w0, lr_rm, n_rm, pi_sft, ptx, beta, gamma, lr_rl, n_rl):
    Phi = np.asarray(Phi, dtype=float)
    w = np.array(w0, dtype=float)
    c = np.asarray(comparisons, dtype=int)
    dphi = Phi[c[:, 0], c[:, 1]] - Phi[c[:, 0], c[:, 2]]
    rm = []
    for _ in range(n_rm):
        m = dphi @ w
        rm.append(float(np.logaddexp(0, -m).mean()))
        w = w - lr_rm * (-(1 - 1 / (1 + np.exp(-m))) @ dphi / len(m))
    r = Phi @ w
    ls = np.log(np.asarray(pi_sft, dtype=float))
    Z = ls.copy()
    P = Z.shape[0]
    q = np.asarray(ptx, dtype=int)
    obj = []
    for _ in range(n_rl):
        z = Z - Z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        pi = np.exp(lp)
        g = r - beta * (lp - ls)
        J = (pi * g).sum() / P + gamma * lp[q[:, 0], q[:, 1]].mean()
        obj.append(float(J))
        grad = pi * (g - (pi * g).sum(1, keepdims=True)) / P
        gp = np.zeros_like(Z)
        np.add.at(gp, q[:, 0], -pi[q[:, 0]])
        np.add.at(gp, (q[:, 0], q[:, 1]), 1.0)
        Z = Z + lr_rl * (grad + gamma * gp / len(q))
    z = Z - Z.max(1, keepdims=True)
    pi = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return np.array(rm), w, np.array(obj), pi
'''

HINTS = [
    'Precompute the feature difference $\\Phi_{pi} - \\Phi_{pj}$ for every comparison; the reward-model gradient is $-\\mathrm{mean}\\,(1 - \\sigma(m))\\,\\Delta\\Phi$ with $m = \\Delta\\Phi\\cdot w$.',
    'For one prompt with $g_k = r_k - \\beta(\\log\\pi_k - \\log\\pi^{SFT}_k)$, the gradient of $\\sum_k\\pi_kg_k$ with respect to $Z_j$ simplifies to $\\pi_j(g_j - \\sum_k\\pi_kg_k)$: the terms from differentiating $g$ cancel.',
    'The pretraining term is a log-likelihood: $\\partial\\log\\pi_{pk}/\\partial Z_{pj} = [j = k] - \\pi_{pj}$. Accumulate it over repeated pairs.',
]

_r = np.random.default_rng(2203)
def _c(P, K, F, n_cmp, n_ptx, beta, gamma, n_rm=50, n_rl=50, lr_rm=0.5, lr_rl=0.5):
    w_true = _r.normal(size=F)
    Phi = _r.normal(size=(P, K, F))
    cmp = []
    for _ in range(n_cmp):
        p = int(_r.integers(P)); i, j = (int(v) for v in _r.choice(K, 2, replace=False))
        if Phi[p, i] @ w_true < Phi[p, j] @ w_true:
            i, j = j, i
        cmp.append((p, i, j))
    sft = _r.dirichlet(np.ones(K) * 2, size=P)
    ptx = [(int(_r.integers(P)), int(_r.integers(K))) for _ in range(n_ptx)]
    return Phi, cmp, np.zeros(F), lr_rm, n_rm, sft, ptx, beta, gamma, lr_rl, n_rl
TESTS = [
    _c(1, 2, 2, 2, 1, 0.1, 0.0, n_rm=2, n_rl=2),
    _c(3, 4, 3, 10, 3, 0.1, 0.5, n_rm=10, n_rl=10),
    _c(5, 6, 4, 40, 10, 0.05, 1.0),
    _c(5, 6, 4, 40, 10, 2.0, 0.0),
    _c(4, 3, 5, 30, 8, 0.0, 0.0, n_rl=100),
    _c(6, 5, 3, 25, 20, 0.2, 5.0, lr_rl=1.0),
]

EXPLANATION = r"""
## The three stages
1. **SFT** on human demonstrations gives a reasonable starting policy (here, `pi_sft`).
2. **Reward model**: labellers rank several responses per prompt, each ranking yields comparison pairs, and a model is trained with the Bradley-Terry loss to score preferred responses higher.
3. **RL** maximises the learned reward. The KL term keeps the policy near the SFT model, both to stay where the reward model is accurate (it is easy to exploit off-distribution) and to preserve fluency; the pretraining term ("ptx") counters regressions on public NLP benchmarks, the "alignment tax".

## What the toy shows
With $\beta = 0$ and no ptx (test 5) the policy drifts towards each prompt's single highest-reward response, however the reward model got there. A large $\beta$ (test 4) keeps it close to SFT. The KL-penalised optimum is in closed form, $\pi^* \propto \pi^{SFT}e^{r/\beta}$, which is exactly what DPO later exploited.

## Real RLHF vs this
Real responses can't be enumerated, so the expectation is estimated with sampled responses and optimised with PPO, and the KL is estimated per token from sampled log-probabilities. The objective is the same.

## Common bugs
- Applying the KL penalty in the reward-model loss instead of the RL objective.
- Forgetting that the policy gradient of an expectation needs the baseline-subtracted form (here it appears exactly as $g_j - \sum_k \pi_k g_k$).
- Letting the reward model keep training during RL.

## Interview follow-ups
- How would you detect reward hacking during stage 3?
- Why did the paper find that a 1.3B InstructGPT model was preferred to the 175B GPT-3?
"""
