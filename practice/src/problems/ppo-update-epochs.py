import numpy as np

META = dict(title='PPO update: GAE, normalised advantages, minibatch epochs with clipping', tags=['Reinforcement Learning', 'Training'],
            diff='Hard', runtime='numpy', lessons=['air-fanout-v1:07.04', 'dlcore:08.06'],
            related=['ppo-clipped-surrogate-objective', 'gae-advantages', 'actor-critic-a2c-loss-with-entropy', 'discounted-returns'])

STATEMENT = r"""
Implement the update phase of PPO for a linear softmax policy and a linear value function: compute advantages for a collected rollout once, then run several epochs of minibatch gradient steps on the clipped objective.

Write `ppo_update(X, x_last, actions, rewards, dones, theta, w, gamma, lam, clip, c_v, c_e, lr, perms, n_mb)` returning `(theta, w, stats)`.

## Models
Policy $\pi(\cdot|x) = \mathrm{softmax}(x\theta)$ with `theta` `(F, A)`; value $V(x) = x\cdot w$ with `w` `(F,)`.

## Before the epochs (with the initial parameters)
- Old log-probabilities $\log\pi_{old}(a_t|x_t)$ and values $V_t$ for `X` `(T, F)`; bootstrap value $V(x_{last})$.
- GAE: $\delta_t = r_t + \gamma(1 - d_t)V_{t+1} - V_t$ (with $V_T = V(x_{last})$), $\hat A_t = \delta_t + \gamma\lambda(1 - d_t)\hat A_{t+1}$, where $d_t$ = `dones[t]` marks that the episode ended after step $t$.
- Returns $R_t = \hat A_t + V_t$; then normalise $\hat A$ to zero mean and unit (population) std, adding $10^{-8}$ to the std.

## Epochs
`perms` `(K, T)`: in epoch $k$, split `perms[k]` into `n_mb` equal consecutive minibatches. For each minibatch $\mathcal B$, with $\rho = \pi(a|x)/\pi_{old}(a|x)$ under the **current** parameters:
$$L = -\mathrm{mean}_{\mathcal B}\min(\rho\hat A,\ \mathrm{clip}(\rho, 1\pm\epsilon)\hat A) + c_v\,\mathrm{mean}_{\mathcal B}(V(x) - R)^2 - c_e\,\mathrm{mean}_{\mathcal B}H(\pi(\cdot|x))$$
Record `(policy term, value term without c_v, mean entropy)` and take one SGD step on `theta` and `w`.

Return the final parameters and `stats` `(K * n_mb, 3)`.
"""

STARTER = '''import numpy as np

def ppo_update(X, x_last, actions, rewards, dones, theta, w, gamma, lam, clip, c_v, c_e, lr, perms, n_mb):
    """Return (theta, w, stats)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _pi(X, th):
    z = X @ th
    z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return np.exp(lp), lp

def ppo_update(X, x_last, actions, rewards, dones, theta, w, gamma, lam, clip, c_v, c_e, lr, perms, n_mb):
    X, x_last, r, d = (np.asarray(a, dtype=float) for a in (X, x_last, rewards, dones))
    a = np.asarray(actions)
    th, w = np.array(theta, dtype=float), np.array(w, dtype=float)
    T = len(r)
    _, lp0 = _pi(X, th)
    old = lp0[np.arange(T), a]
    V = X @ w
    Vn = np.append(V[1:], x_last @ w)
    adv, g = np.zeros(T), 0.0
    for t in range(T - 1, -1, -1):
        delta = r[t] + gamma * (1 - d[t]) * Vn[t] - V[t]
        g = delta + gamma * lam * (1 - d[t]) * g
        adv[t] = g
    ret = adv + V
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = []
    for perm in np.asarray(perms):
        for mb in np.split(perm, n_mb):
            x, ab, A, R = X[mb], a[mb], adv[mb], ret[mb]
            n = len(mb)
            p, lp = _pi(x, th)
            rho = np.exp(lp[np.arange(n), ab] - old[mb])
            surr = np.minimum(rho * A, np.clip(rho, 1 - clip, 1 + clip) * A)
            H = -(p * lp).sum(1)
            v = x @ w
            stats.append((-surr.mean(), ((v - R) ** 2).mean(), H.mean()))
            active = rho * A <= np.clip(rho, 1 - clip, 1 + clip) * A
            oh = np.eye(th.shape[1])[ab]
            dz = -(active * rho * A)[:, None] * (oh - p) / n
            dz += c_e * (p * (lp + H[:, None])) / n
            gw = c_v * 2 * (v - R) @ x / n
            th = th - lr * x.T @ dz
            w = w - lr * gw
    return th, w, np.array(stats)
'''

HINTS = [
    'Compute old log-probabilities, values, advantages and returns once, before any update; they stay fixed for every epoch.',
    'The clipped term has zero gradient exactly when it is the smaller one and $\\rho$ is outside the clip range; otherwise the gradient of $\\rho\\hat A$ with respect to the logits is $\\rho\\hat A(\\mathbf 1_a - \\pi)$.',
    'Entropy gradient with respect to the logits: $\\partial H/\\partial z = -\\pi\\odot(\\log\\pi + H)$. Chain every logit gradient through $x^\\top$ to get the gradient for $\\theta$.',
]

_r = np.random.default_rng(2202)
def _c(T, F, A, K, n_mb, clip=0.2, lr=0.1, gamma=0.99, lam=0.95, c_v=0.5, c_e=0.01, scale=1.0):
    X = _r.normal(size=(T, F))
    d = (_r.random(T) < 0.15).astype(float)
    return (X, _r.normal(size=F), _r.integers(0, A, T), _r.normal(size=T), d, scale * _r.normal(size=(F, A)),
            0.3 * _r.normal(size=F), gamma, lam, clip, c_v, c_e, lr, np.array([_r.permutation(T) for _ in range(K)]), n_mb)
TESTS = [
    _c(4, 2, 2, 1, 1),
    _c(8, 3, 3, 2, 2),
    _c(16, 4, 3, 4, 4, lr=0.5),
    _c(32, 5, 4, 10, 4, lr=1.0, clip=0.1),
    _c(12, 3, 2, 3, 3, gamma=0.9, lam=1.0, c_e=0.0),
    _c(20, 4, 5, 6, 5, lr=2.0, c_v=1.0, c_e=0.05, scale=2.0),
]

EXPLANATION = r"""
## Why several epochs need clipping
A policy-gradient estimate is only valid near the policy that collected the data. PPO squeezes more out of each rollout by taking many minibatch steps on it, and the clipped ratio removes the incentive to move $\rho$ beyond $1 \pm \epsilon$ in the direction that would increase the objective. Later epochs therefore see more clipped samples (larger learning rates in tests 4 and 6 show this in the stats).

## The pieces around the objective
- GAE trades bias for variance with $\lambda$; $\lambda = 1$ (test 5) gives Monte Carlo returns minus the baseline.
- Advantage normalisation per batch is an implementation detail rather than part of the paper's algorithm, but almost every implementation does it and it matters in practice.
- The value loss trains the critic used for the next rollout's advantages; the entropy bonus slows premature collapse to a deterministic policy.

## Common bugs
- Recomputing old log-probabilities or advantages inside the epoch loop.
- Normalising returns along with advantages (the value target must stay unnormalised here).
- Getting the clipping condition wrong for negative advantages: the gradient is zero when $\rho < 1 - \epsilon$ and $\hat A < 0$.

## Interview follow-ups
- What does a rising approximate KL or clip fraction during the epochs tell you, and how do implementations react?
- Why do RLHF implementations of PPO add a KL penalty to a reference model on top of clipping?
"""
