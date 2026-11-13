import numpy as np

META = dict(title='REINFORCE with a time-dependent baseline', tags=['Reinforcement Learning'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:07.02'],
            related=['reinforce-gradient', 'discounted-returns', 'rloo-reinforce-leave-one-out'])

STATEMENT = r"""
Subtracting a baseline $b$ from the return leaves the policy gradient unbiased but can cut its variance a lot. Here the baseline is the **average return at the same time step** across a batch of episodes.

The policy is tabular softmax: logits $\theta \in \mathbb{R}^{S \times A}$, $\pi(\cdot \mid s) = \mathrm{softmax}(\theta_s)$, so $\nabla_{\theta} \log \pi(a \mid s)$ is $e_a - \pi(\cdot \mid s)$ in row $s$ and zero elsewhere.

Write `reinforce_baseline(theta, states, actions, rewards, mask, gamma=0.99)` and return `(advantages, grad)`.

## Inputs
- `theta`: float `(S, A)`.
- `states`, `actions`: int `(N, T)`; `rewards`: float `(N, T)`.
- `mask`: `(N, T)` of 0/1. Episodes are right-padded: each row is some 1s followed by 0s, with at least one 1. Each episode terminates at its last valid step. Padding entries of the other arrays are arbitrary and must be ignored.

## Steps
1. Reward-to-go per episode: $G_{i,t} = \sum_{k \ge t,\ \text{valid}} \gamma^{k-t} r_{i,k}$.
2. Baseline $b_t$ = mean of $G_{i,t}$ over the episodes still running at step $t$ (those with `mask[i, t] = 1`).
3. Advantages $A_{i,t} = G_{i,t} - b_t$ on valid steps, 0 on padding.
4. $\text{grad} = \frac1N \sum_{i,t\ \text{valid}} A_{i,t}\, \nabla_\theta \log \pi(a_{i,t} \mid s_{i,t})$, the ascent direction, shape `(S, A)`.
"""

STARTER = '''import numpy as np

def reinforce_baseline(theta, states, actions, rewards, mask, gamma=0.99):
    """Batch of N padded episodes of length T. Return (advantages (N, T), grad (S, A))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def reinforce_baseline(theta, states, actions, rewards, mask, gamma=0.99):
    th = np.asarray(theta, dtype=float)
    m = np.asarray(mask, dtype=float)
    r = np.asarray(rewards, dtype=float) * m              # padding rewards can't leak in
    N, T = r.shape
    G = np.zeros((N, T))
    running = np.zeros(N)
    for t in reversed(range(T)):                          # reward-to-go, all episodes at once
        running = r[:, t] + gamma * running * m[:, t]
        G[:, t] = running
    G *= m
    alive = m.sum(axis=0)                                 # episodes still running at each step
    b = np.divide(G.sum(axis=0), alive, out=np.zeros(T), where=alive > 0)
    adv = (G - b) * m

    z = th - th.max(axis=1, keepdims=True)
    pi = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    s, a = np.asarray(states), np.asarray(actions)
    coef = -pi[s] * adv[..., None]                        # (N, T, A): -A * pi(.|s)
    coef[np.arange(N)[:, None], np.arange(T), a] += adv   # +A on the taken action
    grad = np.zeros_like(th)
    np.add.at(grad, s, coef)                              # scatter-add rows by state
    return adv, grad / N
'''

HINTS = [
    'Mask the rewards first, then compute reward-to-go for all episodes at once by looping `t` backwards over columns.',
    'The baseline is a column statistic: sum $G$ over valid entries of column $t$ and divide by the number of valid entries. Advantages on padding must be exactly 0.',
    'Per step, the gradient contribution is $A$ times $(e_a - \\pi(\\cdot|s))$ placed in row $s$. Build an `(N, T, A)` array of those and scatter it into `(S, A)` with `np.add.at(grad, states, contrib)`, which handles repeated states.',
]

_r = np.random.default_rng(211)
_N, _T, _S, _A = 5, 7, 4, 3
_len = np.array([7, 3, 5, 1, 6])
MR = (np.arange(_T) < _len[:, None]).astype(int)
SR, AR = _r.integers(0, _S, (_N, _T)), _r.integers(0, _A, (_N, _T))
RR = np.round(_r.normal(size=(_N, _T)), 2)
TESTS = [
    (np.zeros((1, 2)), np.array([[0], [0]]), np.array([[0], [1]]), np.array([[1.0], [3.0]]), np.array([[1], [1]])),
    (np.zeros((2, 2)), np.array([[0, 1], [0, 0]]), np.array([[1, 1], [0, 0]]), np.array([[0.0, 2.0], [1.0, 0.0]]),
     np.array([[1, 1], [1, 0]]), 0.5),
    (np.zeros((1, 2)), np.array([[0, 0, 0]]), np.array([[0, 1, 0]]), np.array([[1.0, 1.0, 1.0]]), np.array([[1, 1, 1]]), 0.9),
    (np.array([[2.0, 0.0]]), np.array([[0, 0], [0, 0]]), np.array([[0, 1], [1, 0]]), np.array([[1.0, 99.0], [0.0, 99.0]]),
     np.array([[1, 0], [1, 0]]), 1.0),
    (_r.normal(size=(_S, _A)), SR, AR, RR, MR, 0.95),
    (np.zeros((_S, _A)), SR, AR, RR, MR, 1.0),
]

EXPLANATION = r"""
## Why a baseline doesn't add bias
For any $b$ that doesn't depend on the action, $\mathbb{E}_{a \sim \pi}[b\, \nabla \log \pi(a)] = b\, \nabla \sum_a \pi(a) = b\, \nabla 1 = 0$. So subtracting it changes no expectations, only variance. A good baseline is close to the expected return, so advantages are centred: actions better than usual get pushed up, worse ones down, instead of every action being pushed up because all returns are positive. (Strictly, the batch mean here includes each episode's own return, which adds a small bias of order $1/N$. RLOO removes it by leaving that episode out.)

## Why per time step
Returns early in an episode sum more rewards than returns near the end, so a single scalar baseline is a poor fit for both. Averaging column-wise gives each step a baseline from the right scale, and the mask keeps finished episodes from dragging later columns toward 0.

## Edge cases
- Test 3: one episode means each column's baseline equals its own return, so every advantage and the gradient are 0.
- Test 4: the padded rewards (99) must be ignored.

## Complexity
$O(NT A)$ time, $O(NTA)$ memory for the per-step contributions (a loop over steps would use $O(SA)$).

## Common bugs
- Including padded steps in the baseline's denominator.
- Using `grad[states] += contrib`, which silently drops repeated states (fancy-index assignment doesn't accumulate).
- Letting the reward-to-go recursion carry across the padding boundary.

## Interview follow-ups
- What is the variance-minimising constant baseline? (A return average weighted by $\lVert\nabla\log\pi\rVert^2$.)
- How does a learned state-value baseline turn this into actor-critic?
"""
