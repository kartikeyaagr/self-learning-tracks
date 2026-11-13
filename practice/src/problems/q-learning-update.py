import numpy as np

META = dict(title='Tabular Q-learning from logged transitions', tags=['Reinforcement Learning'], diff='Easy',
            runtime='numpy', lessons=['air-fanout-v1:07.03'],
            related=['sarsa-update', 'expected-sarsa', 'dqn-loss-with-target-network'])

STATEMENT = r"""
Q-learning is off-policy TD control. Whatever action the agent takes next, it bootstraps from the **best** action in the next state:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\big[r + \gamma (1 - d) \max_{a'} Q(s', a') - Q(s, a)\big]$$

where $d = 1$ if $s'$ is terminal. Because the target ignores the behaviour policy, it can learn from any logged data.

Write `q_learning(Q, transitions, alpha=0.1, gamma=0.99)` and return `(Q_new, policy)`.

## Inputs
- `Q`: float array `(S, A)`. Do not modify it.
- `transitions`: float array `(N, 5)` with rows `[s, a, r, s_next, done]`. `s`, `a`, `s_next` hold integers stored as floats; `done` is 0 or 1.

## Rules
- Apply updates in row order; each one sees the table left by the previous row.

## Output
- `Q_new`: float array `(S, A)`.
- `policy`: int array `(S,)`, the greedy action in each state under `Q_new`, ties to the **lowest** index.
"""

STARTER = '''import numpy as np

def q_learning(Q, transitions, alpha=0.1, gamma=0.99):
    """transitions: (N, 5) rows of [s, a, r, s_next, done]. Return (Q_new (S, A), greedy policy (S,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def q_learning(Q, transitions, alpha=0.1, gamma=0.99):
    Q = np.array(Q, dtype=float)
    for s, a, r, s2, done in np.asarray(transitions, dtype=float):
        s, a, s2 = int(s), int(a), int(s2)
        target = r + gamma * (1.0 - done) * Q[s2].max()     # off-policy: best next action
        Q[s, a] += alpha * (target - Q[s, a])
    return Q, Q.argmax(axis=1)
'''

HINTS = [
    'Loop over the rows of `transitions` on a copy of `Q`, casting the index columns to `int`.',
    'The target is `r + gamma * (1 - done) * Q[s_next].max()`. Read it from the table as it is at that moment.',
    '`Q.argmax(axis=1)` gives the greedy policy and already returns the first index on ties.',
]

_r = np.random.default_rng(81)
_N = 30
TR = np.column_stack([_r.integers(0, 5, _N), _r.integers(0, 3, _N), np.round(_r.normal(size=_N), 2),
                      _r.integers(0, 5, _N), (_r.uniform(size=_N) < 0.15)]).astype(float)
TESTS = [
    (np.zeros((2, 2)), np.array([[0, 1, 1.0, 1, 0], [1, 0, 2.0, 1, 1]]), 0.5, 0.9),
    (np.array([[0.0, 0.0], [3.0, 1.0]]), np.array([[0, 0, 0.0, 1, 0]]), 1.0, 0.5),
    (np.array([[0.0, 0.0], [3.0, 1.0]]), np.array([[0, 0, 0.0, 1, 1]]), 1.0, 0.5),
    (np.zeros((1, 2)), np.array([[0, 1, 1.0, 0, 0]] * 4, dtype=float), 0.5, 0.9),
    (np.zeros((3, 3)), np.zeros((0, 5)), 0.1, 0.9),
    (np.zeros((5, 3)), TR, 0.3, 0.9),
    (_r.normal(size=(5, 3)), TR, 0.05, 0.99),
]

EXPLANATION = r"""
## Why the max makes it off-policy
The target estimates $r + \gamma \max_{a'} Q^*(s', a')$, the Bellman optimality backup with one sampled next state. It doesn't care which action the agent took at $s'$, so data from an exploratory or even random policy still drives Q toward $Q^*$, provided every pair keeps being visited. This is the same backup as value iteration, minus the model.

## Ties and empty input
Test 5 has no transitions, so the table stays zero and every state's greedy action is 0. Test 4 repeats a self-loop: each update raises $Q(0,1)$ and therefore its own target.

## Complexity
$O(N A)$ time (the max over A per row), $O(SA)$ memory.

## Common bugs
- Bootstrapping on terminal transitions (tests 2 and 3 differ only in `done`).
- Vectorising with fancy-index assignment `Q[s, a] += ...`. Repeated pairs then only count once, and every update reads the stale table.
- Using `Q[s_next, a]` (the current action) instead of the max.

## Interview follow-ups
- Why does the max cause overestimation bias, and how does Double Q-learning fix it?
- Compare Q-learning and SARSA on a cliff-walking gridworld: which learns the risky path and why?
"""
