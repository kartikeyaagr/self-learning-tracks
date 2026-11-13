import numpy as np

META = dict(title='Discounted returns across episode boundaries', tags=['Reinforcement Learning'], diff='Easy',
            runtime='numpy', lessons=['air-fanout-v1:07.01', 'air-fanout-v1:07.02'],
            related=['gae-advantages', 'monte-carlo-policy-evaluation', 'reinforce-gradient'])

STATEMENT = r"""
The **return** from step $t$ is the discounted sum of the rewards that follow it, up to the end of the episode:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = r_t + \gamma (1 - d_t)\, G_{t+1}$$

where $d_t = 1$ if the episode ended at step $t$.

Write `discounted_returns(rewards, dones, gamma=0.99, last_value=0.0)` that returns $G_t$ for every step of a rollout.

## Inputs
- `rewards`: float array `(T,)`, with `T >= 1`.
- `dones`: array `(T,)` of 0/1. A rollout may hold several episodes back to back.
- `last_value`: the value estimate of the state reached after the final step. If the last step is **not** a done, the unfinished episode is bootstrapped with it, i.e. $G_{T} = $ `last_value`. If the last step is a done, `last_value` is ignored.

## Output
A float array `(T,)`. A done at step $t$ means nothing after $t$ (and not `last_value`) leaks into $G_t$.
"""

STARTER = '''import numpy as np

def discounted_returns(rewards, dones, gamma=0.99, last_value=0.0):
    """rewards, dones: (T,). Return the discounted return G_t for every step, shape (T,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def discounted_returns(rewards, dones, gamma=0.99, last_value=0.0):
    r, d = np.asarray(rewards, dtype=float), np.asarray(dones, dtype=float)
    G = np.zeros_like(r)
    running = float(last_value)              # G_T: bootstrap for a trailing unfinished episode
    for t in reversed(range(len(r))):        # G_t depends on G_{t+1}, so walk backwards
        running = r[t] + gamma * (1.0 - d[t]) * running
        G[t] = running
    return G
'''

HINTS = [
    '$G_t$ is defined in terms of $G_{t+1}$. Which direction should the loop run?',
    'Start a running total at `last_value` and loop `t` from `T-1` down to 0: `running = r[t] + gamma * (1 - d[t]) * running`.',
    'The `(1 - d[t])` factor is what resets at a done: it drops both the following episode and the bootstrap value. No separate reset is needed.',
]

_r = np.random.default_rng(11)
TESTS = [
    (np.array([1.0, 1.0, 1.0]), np.array([0, 0, 1]), 0.5),
    (np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 1, 0, 0]), 0.9, 10.0),
    (np.array([5.0]), np.array([1]), 0.99, 100.0),
    (np.array([5.0]), np.array([0]), 0.99, 100.0),
    (np.array([0.0, 0.0, 0.0, 1.0]), np.array([0, 0, 0, 0]), 1.0),
    (np.array([1.0, -1.0, 2.0, 0.0, 3.0]), np.array([1, 1, 1, 1, 1]), 0.9, 7.0),
    (_r.normal(size=12), (_r.uniform(size=12) < 0.25).astype(int), 0.97, 1.5),
    (_r.normal(size=20), np.zeros(20, dtype=int), 0.0, -3.0),
]

EXPLANATION = r"""
## Why the backward recursion works
Expanding $G_t = r_t + \gamma G_{t+1}$ once per step gives the full discounted sum, so one backward pass replaces an $O(T^2)$ double loop with $O(T)$ work. Multiplying by $(1-d_t)$ zeroes everything after a terminal step, which is exactly where the sum should stop. The same factor also discards `last_value` when the rollout ends on a done (tests 3 and 4 differ only in that flag).

## Complexity
$O(T)$ time and $O(T)$ memory for the output. The loop is inherently sequential. A vectorised version via `scipy.signal.lfilter` exists but only works within one episode.

## Common bugs
- Looping forwards and summing $\gamma^k r_{t+k}$ without stopping at dones, which leaks the next episode's rewards into this one.
- Bootstrapping with `last_value` even when the final step is terminal.
- Resetting the running total *after* writing $G_t$ instead of inside the update, so it is off by one step.
- With $\gamma = 0$ the return is just the reward (test 8). Code that special-cases $\gamma$ often breaks here.

## Interview follow-ups
- Why is a time-limit truncation different from a true terminal state, and what should you bootstrap with there?
- REINFORCE uses $G_t$ directly. What does subtracting a baseline buy you, and why does it keep the gradient unbiased?
"""
