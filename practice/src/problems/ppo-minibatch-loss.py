import numpy as np

META = dict(title='The complete PPO minibatch loss', tags=['Reinforcement Learning', 'Loss Functions'], diff='Medium', runtime='numpy',
            related=['ppo-clipped-surrogate-objective', 'approx-kl-and-clip-fraction'])

STATEMENT = r"""
Write `ppo_loss(logp_new, logp_old, adv, values, returns, entropy, clip_eps, vf_coef, ent_coef, norm_adv)` for one minibatch.

1. If `norm_adv`, normalise the advantages within the minibatch: $(A - \bar A)/(\sigma_A + 10^{-8})$ with the population std.
2. Policy loss: $L^{pg} = -\,\text{mean}\big(\min(rA,\ \text{clip}(r, 1-\epsilon, 1+\epsilon)A)\big)$, $r = e^{\log\pi_{new} - \log\pi_{old}}$.
3. Value loss: $L^{v} = \tfrac12\,\text{mean}\big((V - R)^2\big)$.
4. Entropy bonus: $H = \text{mean}(\text{entropy})$.
5. Total: $L = L^{pg} + c_v L^{v} - c_e H$.

Return a dict with `policy`, `value`, `entropy`, `total`.
"""

STARTER = '''import numpy as np

def ppo_loss(logp_new, logp_old, adv, values, returns, entropy, clip_eps, vf_coef, ent_coef, norm_adv):
    """Clipped policy loss + value loss - entropy bonus for one minibatch."""
    r = np.exp(np.asarray(logp_new) - np.asarray(logp_old))
    pg = -np.mean(np.clip(r, 1 - clip_eps, 1 + clip_eps) * np.asarray(adv))
    v = np.mean((np.asarray(values) - np.asarray(returns)) ** 2)
    H = np.mean(entropy)
    return {"policy": float(pg), "value": float(v), "entropy": float(H), "total": float(pg + vf_coef * v + ent_coef * H)}
'''

SOLUTION = '''import numpy as np

def ppo_loss(logp_new, logp_old, adv, values, returns, entropy, clip_eps, vf_coef, ent_coef, norm_adv):
    A = np.asarray(adv, float)
    if norm_adv:
        A = (A - A.mean()) / (A.std() + 1e-8)
    r = np.exp(np.asarray(logp_new, float) - np.asarray(logp_old, float))
    pg = -np.mean(np.minimum(r * A, np.clip(r, 1 - clip_eps, 1 + clip_eps) * A))
    v = 0.5 * np.mean((np.asarray(values, float) - np.asarray(returns, float)) ** 2)
    H = float(np.mean(entropy))
    return {"policy": float(pg), "value": float(v), "entropy": H, "total": float(pg + vf_coef * v - ent_coef * H)}
'''

HINTS = [
    'The minimum of the clipped and unclipped terms is what makes the objective pessimistic; clipping alone is not enough.',
    'The entropy term is subtracted from a loss because we want to maximise entropy.',
    'Normalising advantages per minibatch makes the policy loss scale-free, so one learning rate works across reward scales.',
]

def _case(seed, n, spread):
    g = np.random.default_rng(seed)
    old = np.log(g.uniform(0.1, 0.9, n))
    return (old + spread * g.normal(size=n), old, g.normal(size=n) * 3, g.normal(size=n), g.normal(size=n), g.uniform(0.5, 1.5, n))

TESTS = [
    ([0.0], [0.0], [1.0], [0.5], [1.0], [1.0], 0.2, 0.5, 0.01, False),
    ([np.log(1.5), np.log(0.5)], [0.0, 0.0], [1.0, -1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.2, 0.5, 0.0, False),
    ([np.log(1.5), np.log(0.5)], [0.0, 0.0], [-1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.2, 0.5, 0.0, False),
    _case(1, 32, 0.1) + (0.2, 0.5, 0.01, True),
    _case(2, 32, 0.5) + (0.1, 1.0, 0.0, True),
    _case(3, 8, 0.2) + (0.3, 0.25, 0.05, False),
]

EXPLANATION = r"""
## The pessimistic bound
For a positive advantage, the objective stops rewarding increases in $r$ above $1+\epsilon$. For a negative one, it stops rewarding decreases below $1-\epsilon$. The second test shows this: both ratios moved the helpful way and both are capped. In the third test the same ratios moved the harmful way, the unclipped term is smaller and wins the minimum, so the full penalty applies and the gradient pushes back.

## The other terms
- The value loss trains the critic that produced the advantages. With a shared network, $c_v$ balances how much the trunk serves the critic versus the policy.
- The entropy bonus slows the policy's collapse to a deterministic one. Too large and the agent never commits.

## Details that change results
Advantage normalisation, value clipping, gradient-norm clipping and learning-rate annealing are all small implementation choices. Studies of PPO implementations found that such details account for much of the gap between versions that "work" and ones that do not.

## Interview follow-ups
- Why is the value loss sometimes clipped as well, and when does that hurt?
"""
