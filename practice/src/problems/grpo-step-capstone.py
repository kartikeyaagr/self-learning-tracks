import numpy as np

META = dict(title='Capstone: one GRPO update and what it did to pass@k', tags=['Reinforcement Learning', 'Language Models'], diff='Hard', runtime='numpy',
            related=['grpo-group-relative-advantage', 'normalize-grpo-group-advantages', 'pass-at-k-estimator'])

STATEMENT = r"""
Run one GRPO update on a policy that, for each prompt $p$, picks one of $K$ candidate answers with $\pi_p = \text{softmax}(\theta_p)$. Some answers are correct: `correct (P, K)` is 0/1.

Write `grpo_update(theta, theta_ref, actions, correct, epochs, lr, clip_eps, kl_coef, k)`. `actions (P, G)` holds the $G$ sampled answers per prompt, drawn from $\pi_{\theta}$ (so `theta` is also the old policy).

1. Rewards $R_{pg} = $ `correct[p, actions[p, g]]`. Advantages per group: $A_{pg} = (R_{pg} - \bar R_p)/(\sigma_p + 10^{-6})$, population std, and $A = 0$ for a group whose rewards are all equal.
2. For each epoch, with current $\theta$: ratio $r = \pi_\theta(a)/\pi_{old}(a)$; policy term $\min(rA, \text{clip}(r)A)$, passing gradient $A\,r\,(e_a - \pi_p)$ only when the unclipped term is the minimum. KL term (the k3 estimator) $\rho - \log\rho - 1$ with $\rho = \pi_{ref}(a)/\pi_\theta(a)$, whose gradient is $(1 - \rho)(e_a - \pi_p)$.
3. Objective: mean over all $P \cdot G$ samples of [policy term $-$ `kl_coef` $\cdot$ KL term]. Gradient ascent with step `lr` on $\theta$.

Report, before and after: exact `pass1` (mean over prompts of $\sum_k \pi_{p,k}\,\text{correct}_{p,k}$) and `passk` (mean of $1 - (1 - \text{pass1}_p)^k$). Return a dict with `theta`, `pass1_before`, `pass1_after`, `passk_before`, `passk_after`, `zero_adv_groups` (number of groups with all-equal rewards).
"""

STARTER = '''import numpy as np

def grpo_update(theta, theta_ref, actions, correct, epochs, lr, clip_eps, kl_coef, k):
    """Group-relative advantages, clipped policy gradient and a KL penalty, on a softmax answer policy."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sm(T):
    Z = np.exp(T - T.max(1, keepdims=True))
    return Z / Z.sum(1, keepdims=True)

def _pass(T, C, k):
    p1 = (_sm(T) * C).sum(1)
    return float(p1.mean()), float((1 - (1 - p1) ** k).mean())

def grpo_update(theta, theta_ref, actions, correct, epochs, lr, clip_eps, kl_coef, k):
    th = np.asarray(theta, float).copy()
    ref = _sm(np.asarray(theta_ref, float))
    a = np.asarray(actions, int)
    C = np.asarray(correct, float)
    P, G = a.shape
    K = th.shape[1]
    rows = np.repeat(np.arange(P), G)
    af = a.ravel()
    R = C[np.arange(P)[:, None], a]
    sd = R.std(1, keepdims=True)
    A = np.where(sd > 0, (R - R.mean(1, keepdims=True)) / (sd + 1e-6), 0.0).ravel()
    zero = int((sd[:, 0] == 0).sum())
    p_old = _sm(th)[rows, af]
    p1b, pkb = _pass(th, C, k)
    E = np.eye(K)[af]
    for _ in range(epochs):
        pi = _sm(th)
        pa = pi[rows, af]
        r = pa / p_old
        live = r * A <= np.clip(r, 1 - clip_eps, 1 + clip_eps) * A
        rho = ref[rows, af] / pa
        coef = live * A * r - kl_coef * (1 - rho)
        g = coef[:, None] * (E - pi[rows])
        grad = np.zeros_like(th)
        np.add.at(grad, rows, g)
        th = th + lr * grad / (P * G)
    p1a, pka = _pass(th, C, k)
    return {"theta": th.tolist(), "pass1_before": p1b, "pass1_after": p1a, "passk_before": pkb,
            "passk_after": pka, "zero_adv_groups": zero}
'''

HINTS = [
    'Flatten (prompt, sample) pairs into one list and use np.add.at to send each sample’s gradient to its prompt’s row of theta.',
    'A group where every answer is right, or every answer is wrong, has zero advantage and teaches nothing; count them.',
    'The k3 KL term pulls theta towards the reference only through the sampled actions.',
]

def _case(seed, P, K, G, n_correct, scale):
    g = np.random.default_rng(seed)
    th = scale * g.normal(size=(P, K))
    C = np.zeros((P, K))
    for p in range(P):
        C[p, g.choice(K, size=n_correct[p], replace=False)] = 1
    pi = np.exp(th - th.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
    acts = np.array([g.choice(K, size=G, p=pi[p]) for p in range(P)])
    return th, th.copy(), acts, C

TESTS = [
    _case(1, 4, 5, 8, [1, 1, 2, 1], 0.5) + (1, 1.0, 0.2, 0.0, 8),
    _case(2, 4, 5, 8, [1, 1, 2, 1], 0.5) + (4, 1.0, 0.2, 0.0, 8),
    _case(2, 4, 5, 8, [1, 1, 2, 1], 0.5) + (4, 1.0, 0.2, 0.5, 8),
    _case(3, 6, 8, 4, [0, 8, 1, 1, 2, 1], 1.0) + (2, 2.0, 0.2, 0.05, 4),
    _case(4, 8, 10, 16, [1] * 8, 2.0) + (3, 3.0, 0.2, 0.0, 16),
]

EXPLANATION = r"""
## The algorithm in miniature
GRPO drops PPO's value network: it samples a group of answers per prompt and uses each answer's reward relative to its group as the advantage. Everything else is PPO: ratios, clipping, several epochs on the same samples, plus a KL penalty to a reference model. Here a "response" is one of K answers, so the gradient is exact and the effect on accuracy can be computed exactly instead of estimated.

## What to look at
- **Zero-advantage groups.** A group where every answer gets the same reward contributes nothing. In the fourth test that is all six groups (one prompt has no correct answer, one has only correct answers, and the other four happened to draw four wrong answers each), so the update changes nothing at all. As training goes on, more prompts become always-solved and the effective batch shrinks; some recipes filter such prompts out or sample more answers for them.
- **pass@1 versus pass@k.** In these small tests both rise. With real language models, RL on verifiable rewards often lifts pass@1 much more than pass@k at large k, and can even lower the latter, because updates concentrate probability on answers the model could already find. Reporting both, computed exactly here and with the unbiased estimator in practice, is how you would notice.
- **The KL term.** Comparing the second and third tests shows the penalty holding the policy closer to the reference and giving up some of the pass@1 gain.

## Interview follow-ups
- Dividing advantages by the group's standard deviation gives very easy and very hard prompts large weights. When is that a problem, and what does Dr. GRPO change?
"""
