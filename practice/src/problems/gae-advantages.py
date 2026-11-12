import numpy as np

META = dict(title='Generalized Advantage Estimation', tags=['Reinforcement Learning'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:07.04', 'air-fanout-v1:07.02'], related=['value-iteration-step'])

STATEMENT = r"""
PPO and other actor-critic methods weight each action by an **advantage**: how much better it went than the critic expected. GAE blends n-step estimates with a decay $\lambda$:

$$\delta_t = r_t + \gamma (1 - d_t)\, V(s_{t+1}) - V(s_t), \qquad \hat A_t = \delta_t + \gamma \lambda (1 - d_t)\, \hat A_{t+1}$$

where $d_t = 1$ if the episode ended at step $t$. Returns for the value loss are $\hat R_t = \hat A_t + V(s_t)$.

Write `gae(rewards, values, dones, gamma=0.99, lam=0.95)` and return `(advantages, returns)`.

## Inputs
- `rewards`: `(T,)`; `dones`: `(T,)` of 0/1.
- `values`: `(T+1,)`. The last entry is the critic's estimate for the state **after** the final step, used to bootstrap if that step didn't end the episode.

## Output
Two float arrays of shape `(T,)`. A rollout can contain several episodes: a done at step $t$ must stop both the bootstrap and the recursion at $t$.
"""

STARTER = '''import numpy as np

def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Return (advantages (T,), returns (T,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    r, v, d = (np.asarray(x, dtype=float) for x in (rewards, values, dones))
    T = len(r)
    adv = np.zeros(T)
    running = 0.0
    for t in reversed(range(T)):                     # the recursion runs backwards in time
        live = 1.0 - d[t]
        delta = r[t] + gamma * live * v[t + 1] - v[t]
        running = delta + gamma * lam * live * running
        adv[t] = running
    return adv, adv + v[:-1]
'''

HINTS = [
    'The recursion for $\\hat A_t$ needs $\\hat A_{t+1}$, so loop `t` from `T-1` down to `0` and carry a running value.',
    'Multiply by `(1 - dones[t])` in **both** places: the bootstrap term $V(s_{t+1})$ and the carried advantage.',
    'Returns are advantages plus the value of the state they came from: `adv + values[:-1]`.',
]

TESTS = [
    (np.array([1.0]), np.array([0.0, 0.0]), np.array([1])),
    (np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0, 0.0]), np.array([0, 0, 0]), 1.0, 1.0),
    (np.array([0.0, 0.0, 1.0]), np.array([0.5, 0.5, 0.5, 10.0]), np.array([0, 0, 1])),
    (np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0]), np.array([0, 1, 0, 0]), 0.9, 0.8),
    (np.array([1.0, 0.0, -1.0, 2.0, 0.5]), np.array([0.2, -0.1, 0.3, 0.0, 0.4, 1.5]), np.array([0, 0, 0, 1, 0]), 0.99, 0.0),
    (np.array([1.0, 0.0, -1.0, 2.0, 0.5]), np.array([0.2, -0.1, 0.3, 0.0, 0.4, 1.5]), np.array([0, 0, 0, 1, 0]), 0.99, 1.0),
]

EXPLANATION = r"""
## What $\lambda$ trades off
- $\lambda = 0$: $\hat A_t = \delta_t$, the one-step TD error. Low variance, but biased whenever the critic is wrong.
- $\lambda = 1$: $\hat A_t = \sum_k \gamma^k r_{t+k} - V(s_t)$, the Monte-Carlo return minus the baseline. Unbiased, high variance.
- In between, it's an exponentially weighted mix of all n-step estimates. PPO defaults to 0.95. Tests 5 and 6 run the same rollout at both extremes.

## Episode boundaries
Rollouts are fixed-length slices that cross episode ends. At a done, the next state belongs to a **new** episode, so its value must not leak back. That's why $(1-d_t)$ multiplies both terms. Missing it is the most common GAE bug.

## Truncation vs termination
A time-limit cutoff is not a real terminal state. Strictly, you should still bootstrap from $V(s_{t+1})$ there. Gymnasium separates `terminated` from `truncated` for this reason.

## Interview follow-ups
- Why normalise advantages per batch before the PPO loss?
- In GRPO (used for LLM RL), how is the advantage computed without a critic? (A group's rewards are normalised against the group mean and standard deviation.)
"""
