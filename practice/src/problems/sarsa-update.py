import numpy as np

META = dict(title='SARSA along one trajectory', tags=['Reinforcement Learning'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:07.01'],
            related=['q-learning-update', 'expected-sarsa', 'td-0-value-update'])

STATEMENT = r"""
SARSA is on-policy TD control. After taking $a_t$ in $s_t$, receiving $r_t$ and choosing the **next action** $a_{t+1}$ in $s_{t+1}$ with the same policy, it updates

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\big[r_t + \gamma\, Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\big]$$

The name is the tuple it needs: $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$.

Write `sarsa_trajectory(Q, states, actions, rewards, terminated, alpha=0.1, gamma=0.99)` that applies the update for every step of one trajectory, in time order, and returns the new Q table.

## Inputs
- `Q`: float array `(S, A)`. Do not modify it; return a new array.
- `states`, `actions`: int arrays of length `T + 1`. They hold $s_0, \dots, s_T$ and $a_0, \dots, a_T$, where $(s_T, a_T)$ is the pair the agent was about to take when the trajectory was cut.
- `rewards`: float array `(T,)`, with `T >= 1`.
- `terminated`: bool. If True, $s_T$ is terminal: the last update uses target $r_{T-1}$ with no bootstrap, and `states[T]`, `actions[T]` are placeholders to ignore. If False, the last update bootstraps from $Q(s_T, a_T)$ as usual.

## Output
Float array `(S, A)`. Updates are sequential: step $t$ sees the table left by step $t-1$.
"""

STARTER = '''import numpy as np

def sarsa_trajectory(Q, states, actions, rewards, terminated, alpha=0.1, gamma=0.99):
    """states, actions: (T+1,), rewards: (T,). Return the updated Q table (S, A)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def sarsa_trajectory(Q, states, actions, rewards, terminated, alpha=0.1, gamma=0.99):
    Q = np.array(Q, dtype=float)
    T = len(rewards)
    for t in range(T):
        s, a = states[t], actions[t]
        if terminated and t == T - 1:
            target = rewards[t]                                     # terminal: nothing to bootstrap
        else:
            target = rewards[t] + gamma * Q[states[t + 1], actions[t + 1]]
        Q[s, a] += alpha * (target - Q[s, a])
    return Q
'''

HINTS = [
    'Step $t$ uses the pair at $t$ and the pair at $t+1$. Loop `t` over `range(len(rewards))` on a copy of `Q`.',
    'The target is `rewards[t] + gamma * Q[states[t+1], actions[t+1]]`, read from the table as it is right now.',
    'Only the very last step of a terminated trajectory drops the bootstrap term. Every earlier step bootstraps normally.',
]

_r = np.random.default_rng(71)
_T = 15
SR, AR = _r.integers(0, 4, _T + 1), _r.integers(0, 3, _T + 1)
RR = np.round(_r.normal(size=_T), 2)
TESTS = [
    (np.zeros((3, 2)), np.array([0, 1, 2]), np.array([1, 0, 0]), np.array([0.0, 1.0]), True, 0.5, 0.9),
    (np.ones((2, 2)), np.array([0, 1, 0]), np.array([0, 1, 1]), np.array([1.0, 1.0]), False, 0.5, 0.5),
    (np.ones((2, 2)), np.array([0, 1, 0]), np.array([0, 1, 1]), np.array([1.0, 1.0]), True, 0.5, 0.5),
    (np.zeros((1, 1)), np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), np.array([1.0, 1.0, 1.0]), False, 1.0, 0.5),
    (np.array([[0.0, 5.0], [2.0, -1.0]]), np.array([1, 0]), np.array([0, 1]), np.array([0.0]), False, 0.25, 1.0),
    (np.zeros((4, 3)), SR, AR, RR, True, 0.2, 0.95),
    (_r.normal(size=(4, 3)), SR, AR, RR, False, 0.2, 0.95),
]

EXPLANATION = r"""
## On-policy
SARSA's target uses the action the behaviour policy actually picked next, exploration included. So it learns the value of the policy it is following, $\epsilon$-greedy warts and all. On a cliff-walking task that makes it prefer a safer path than Q-learning: it accounts for the chance that an exploratory step walks off the edge.

## The last step
When the trajectory ends in a terminal state there is no next action, so the target is just the reward (tests 2 and 3 differ only in this flag). If the trajectory was merely cut short (a rollout boundary), the agent has already chosen $a_T$ and SARSA bootstraps from it.

## Complexity
$O(T)$ time, $O(SA)$ memory for the table copy.

## Common bugs
- Bootstrapping from $\max_a Q(s_{t+1}, a)$. That's Q-learning, not SARSA.
- Using the action at $t$ in the bootstrap instead of the one at $t+1$.
- Treating every trajectory end as terminal, or none of them.

## Interview follow-ups
- Under what conditions does SARSA converge to the optimal Q? (GLIE: greedy in the limit with infinite exploration.)
- Write n-step SARSA. What changes in the target and when can each update be made?
"""
