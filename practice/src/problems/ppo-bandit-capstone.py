import numpy as np

META = dict(title='Capstone: a PPO update with early stopping on KL', tags=['Reinforcement Learning'], diff='Hard', runtime='numpy',
            related=['ppo-minibatch-loss', 'approx-kl-and-clip-fraction', 'gae-advantages'])

STATEMENT = r"""
Run one full PPO update for a softmax policy over `K` arms, $\pi = \text{softmax}(\theta)$, with the gradient written by hand.

Write `ppo_update(theta_old, actions, rewards, epochs, lr, clip_eps, ent_coef, target_kl)`:

1. Advantages: $A_i = (R_i - \bar R)/(\sigma_R + 10^{-8})$ (population std). Old log-probs come from `theta_old` and stay fixed.
2. For each epoch, with the current $\theta$ (starting at `theta_old`):
   - $r_i = \pi_\theta(a_i)/\pi_{old}(a_i)$. Sample $i$ passes gradient only when $r_i A_i \le \text{clip}(r_i)A_i$ (the unclipped term is the minimum), with $\partial r_i/\partial\theta = r_i(e_{a_i} - \pi_\theta)$.
   - Objective $J = \text{mean}_i \min(r_iA_i, \text{clip}(r_i)A_i) + c_e H(\pi_\theta)$, where $\partial H/\partial\theta_k = -\pi_k(\log\pi_k + H)$.
   - Gradient ascent: $\theta \leftarrow \theta + \text{lr}\,\nabla J$.
   - Then compute `kl_k3` and `clip_frac` (as in the diagnostics step) between the new $\theta$ and `theta_old` on the batch. Record them. If `kl_k3 > target_kl`, stop after this epoch.

Return a dict: `theta` (final), `probs` (final softmax), `kl` (list, one per epoch run), `clip_frac` (list), `epochs_run`.
"""

STARTER = '''import numpy as np

def ppo_update(theta_old, actions, rewards, epochs, lr, clip_eps, ent_coef, target_kl):
    """Several epochs of clipped policy-gradient ascent on a softmax bandit policy."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _softmax(t):
    z = np.exp(t - t.max())
    return z / z.sum()

def ppo_update(theta_old, actions, rewards, epochs, lr, clip_eps, ent_coef, target_kl):
    th0 = np.asarray(theta_old, float)
    a = np.asarray(actions, int)
    R = np.asarray(rewards, float)
    A = (R - R.mean()) / (R.std() + 1e-8)
    p_old = _softmax(th0)[a]
    th = th0.copy()
    kls, cfs = [], []
    n, K = len(a), len(th0)
    E = np.eye(K)[a]
    for _ in range(epochs):
        pi = _softmax(th)
        r = pi[a] / p_old
        live = r * A <= np.clip(r, 1 - clip_eps, 1 + clip_eps) * A
        g = ((live * A * r)[:, None] * (E - pi)).sum(0) / n
        H = -(pi * np.log(pi)).sum()
        g += ent_coef * (-pi * (np.log(pi) + H))
        th = th + lr * g
        lr_ = np.log(_softmax(th)[a]) - np.log(p_old)
        rr = np.exp(lr_)
        kls.append(float(((rr - 1) - lr_).mean()))
        cfs.append(float((np.abs(rr - 1) > clip_eps).mean()))
        if kls[-1] > target_kl:
            break
    return {"theta": th.tolist(), "probs": _softmax(th).tolist(), "kl": kls, "clip_frac": cfs, "epochs_run": len(kls)}
'''

HINTS = [
    'For a softmax policy, the gradient of log pi(a) with respect to theta is e_a - pi; the ratio’s gradient is r times that.',
    'A sample whose clipped term is the minimum contributes no policy gradient at all; build a boolean mask.',
    'Measure KL after each epoch’s step, against the fixed old policy, and break as soon as it passes the target.',
]

def _case(seed, K, n, means, theta_scale):
    g = np.random.default_rng(seed)
    th = theta_scale * g.normal(size=K)
    p = np.exp(th - th.max()); p /= p.sum()
    acts = g.choice(K, size=n, p=p)
    rew = np.asarray(means)[acts] + 0.5 * g.normal(size=n)
    return th, acts, rew

TESTS = [
    _case(1, 3, 32, [0.0, 1.0, 0.2], 0.0) + (4, 0.5, 0.2, 0.0, 1.0),
    _case(1, 3, 32, [0.0, 1.0, 0.2], 0.0) + (20, 2.0, 0.2, 0.0, 0.02),
    _case(2, 4, 64, [1.0, 0.0, 0.0, 0.5], 1.0) + (10, 1.0, 0.1, 0.01, 0.05),
    _case(3, 2, 16, [0.0, 0.0], 0.5) + (5, 0.5, 0.2, 0.1, 1.0),
    _case(4, 5, 128, [0.0, 0.2, 0.4, 0.6, 2.0], 0.0) + (8, 1.0, 0.2, 0.0, 0.03),
]

EXPLANATION = r"""
## The whole update
This is the inner loop of every PPO implementation, with a bandit so that the policy gradient is exact and easy to check: normalise advantages, reuse the same batch for several epochs, let clipping switch off samples whose ratio has moved far enough, add an entropy bonus, and watch KL.

## What the tests show
- **Clipping shuts off samples.** As epochs go by, the clip fraction rises and the gradient shrinks, so each extra epoch on the same data moves the policy less.
- **Early stopping.** With a learning rate of 2.0 and a KL target of 0.02, the second test stops after the first epoch even though 20 were allowed: one step already overshot the target, and clipping could not prevent it because clipping only acts on the next step.
- **No signal.** In the fourth test both arms pay the same on average, so the normalised advantages are pure noise. The update still moves the policy (arm 0 goes from 0.91 to 0.92), and the entropy bonus is too weak to stop it. A moving policy is not evidence of learning.

## Why clipping is not a trust region
The clip only removes the gradient of samples already outside the range; samples inside keep pushing, and the policy can move other actions' probabilities freely. That is why KL monitoring or early stopping is standard alongside it.

## Interview follow-ups
- With a neural network policy, the probability of an unclipped sample can change because of an update aimed at a different sample. How does that weaken the guarantee further?
"""
