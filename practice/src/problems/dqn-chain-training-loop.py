import numpy as np

META = dict(title='DQN end to end on a chain: exploration, replay, target network', tags=['Reinforcement Learning', 'Training'],
            diff='Hard', runtime='numpy', lessons=['air-fanout-v1:07.03', 'dlcore:08.04'],
            related=['dqn-loss-with-target-network', 'experience-replay-buffer', 'epsilon-greedy-action-selection', 'double-dqn-target'])

STATEMENT = r"""
Run the full DQN loop on a small chain environment, with all randomness supplied so the run is reproducible. The Q-function is linear in a one-hot state encoding, so it is a table $W$ `(n, 2)`, but it is trained exactly as DQN trains a network.

Write `dqn_chain(n, T, W0, gamma, lr, eps_start, eps_end, decay_steps, capacity, batch, sync_every, eps_u, rand_a, samp_u, double)` returning `(losses, W, episode_lengths)`.

## Environment
States $0..n-1$, start at state 0. Action 0 moves left (staying at 0 at the edge), action 1 moves right. Entering state $n-1$ gives reward 1 and ends the episode (the next step starts again from 0); every other transition gives reward 0.

## Step $t = 0..T-1$
1. $\epsilon_t = \max(\epsilon_{end},\ \epsilon_{start} - (\epsilon_{start} - \epsilon_{end})\,t/\text{decay\_steps})$. If `eps_u[t]` $< \epsilon_t$ take action `rand_a[t]`, else the greedy action $\arg\max_a W[s, a]$ (ties to action 0).
2. Store $(s, a, r, s', \text{done})$ in a circular buffer of size `capacity` (oldest overwritten first).
3. If the buffer holds at least `batch` transitions: sample indices $\lfloor u\cdot\text{len}\rfloor$ for each $u$ in `samp_u[t]` (`samp_u` is `(T, batch)`); target $y = r + \gamma(1 - \text{done})\,Q_{tgt}(s')$ where $Q_{tgt}(s') = \max_a W_{tgt}[s', a]$, or with `double`, $W_{tgt}[s', \arg\max_a W[s', a]]$; loss $= \mathrm{mean}$ Huber($W[s, a] - y$) with $\delta = 1$. Record it and take one SGD step on $W$ (targets held fixed; repeated indices accumulate).
4. If $(t + 1)$ is a multiple of `sync_every`, copy $W$ into $W_{tgt}$ ($W_{tgt}$ starts as a copy of `W0`).

Return `losses` (one per update), the final `W`, and `episode_lengths`: the number of steps in each completed episode.
"""

STARTER = '''import numpy as np

def dqn_chain(n, T, W0, gamma, lr, eps_start, eps_end, decay_steps, capacity, batch, sync_every, eps_u, rand_a, samp_u, double):
    """Return (losses, W, episode_lengths)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def dqn_chain(n, T, W0, gamma, lr, eps_start, eps_end, decay_steps, capacity, batch, sync_every, eps_u, rand_a, samp_u, double):
    W = np.array(W0, dtype=float)
    Wt = W.copy()
    buf, pos = [], 0
    s, ep_len = 0, 0
    losses, lengths = [], []
    for t in range(T):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * t / decay_steps)
        a = int(rand_a[t]) if eps_u[t] < eps else int(np.argmax(W[s]))
        s2 = max(s - 1, 0) if a == 0 else s + 1
        done = s2 == n - 1
        tr = (s, a, 1.0 if done else 0.0, s2, float(done))
        if len(buf) < capacity:
            buf.append(tr)
        else:
            buf[pos] = tr
        pos = (pos + 1) % capacity
        ep_len += 1
        if done:
            lengths.append(ep_len)
            s, ep_len = 0, 0
        else:
            s = s2
        if len(buf) >= batch:
            idx = np.floor(np.asarray(samp_u[t]) * len(buf)).astype(int)
            S, A, R, S2, D = (np.array(c) for c in zip(*[buf[i] for i in idx]))
            S, A, S2 = S.astype(int), A.astype(int), S2.astype(int)
            q2 = Wt[S2, W[S2].argmax(1)] if double else Wt[S2].max(1)
            y = R + gamma * (1 - D) * q2
            d = W[S, A] - y
            losses.append(float(np.where(np.abs(d) <= 1, 0.5 * d * d, np.abs(d) - 0.5).mean()))
            g = np.clip(d, -1, 1) / len(d)
            np.add.at(W, (S, A), -lr * g)
        if (t + 1) % sync_every == 0:
            Wt = W.copy()
    return np.array(losses), W, np.array(lengths)
'''

HINTS = [
    'Keep the buffer as a list plus a write position that wraps at `capacity`. Sample from however many transitions it holds right now.',
    'The Huber loss gradient with respect to the prediction is the residual clipped to $[-1, 1]$, divided by the batch size. Compute all targets from $W$ and $W_{tgt}$ before changing $W$.',
    'After a terminal transition, the next step starts from state 0 and a new episode length counter. Terminal targets have no bootstrap term.',
]

_r = np.random.default_rng(2201)
def _c(n, T, capacity, batch, sync, eps=(1.0, 0.05, None), gamma=0.9, lr=0.5, double=False, W0=None):
    es, ee, ds = eps
    return (n, T, np.zeros((n, 2)) if W0 is None else W0, gamma, lr, es, ee, ds or T // 2, capacity, batch, sync,
            _r.random(T), _r.integers(0, 2, T), _r.random((T, batch)), double)
TESTS = [
    _c(3, 8, 10, 2, 4),
    _c(4, 60, 20, 4, 10),
    _c(5, 300, 50, 8, 25),
    _c(5, 300, 50, 8, 25, double=True),
    _c(6, 400, 30, 4, 50, eps=(1.0, 0.1, 200), W0=0.5 * _r.normal(size=(6, 2)), double=True),
    _c(3, 100, 8, 8, 1, eps=(1.0, 0.2, 100), gamma=0.5, lr=0.2),
]

EXPLANATION = r"""
## The three ingredients
- **Experience replay** breaks the strong correlation between consecutive transitions and reuses each one many times.
- **A target network**, refreshed every 10,000 steps in the Nature paper, keeps the regression target from moving with every update, which otherwise makes bootstrapped Q-learning with function approximation prone to divergence.
- **$\epsilon$-greedy with an annealed $\epsilon$** explores broadly early and exploits later.

## Double DQN
The max in the target picks whichever action's estimate is noisiest upwards, so plain DQN overestimates values. Double DQN selects the action with the online weights and evaluates it with the target weights, which reduces that bias at no extra cost (tests 4 and 5).

## The Huber loss
Clipping the error gradient to $[-1, 1]$ (the paper's "error clipping") keeps a few large TD errors, common early in training, from producing huge updates.

## What the chain shows
Reward appears only at the far end, so the agent needs random exploration to find it once; afterwards bootstrapped updates from replay carry that value backwards one state at a time, and the greedy policy learns to walk right. Test 6 syncs every step, which removes the target network's effect.

## Common bugs
- Bootstrapping through terminal transitions.
- Computing the target with the online network (or syncing every step by accident).
- Sampling from the buffer's capacity instead of its current size.

## Interview follow-ups
- Why does DQN need off-policy learning to use replay, and what breaks if you replay on-policy methods?
- How does prioritised replay change the sampling and why does it need importance weights?
"""
