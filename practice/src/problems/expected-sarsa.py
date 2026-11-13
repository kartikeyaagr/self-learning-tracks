import numpy as np

META = dict(title='Expected SARSA with an epsilon-greedy target', tags=['Reinforcement Learning'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:07.01'],
            related=['sarsa-update', 'q-learning-update', 'epsilon-greedy-action-selection'])

STATEMENT = r"""
Expected SARSA replaces SARSA's sampled next action with an **expectation** over the target policy $\pi$:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\Big[r + \gamma (1 - d) \sum_{a'} \pi(a' \mid s')\, Q(s', a') - Q(s, a)\Big]$$

Here $\pi$ is $\epsilon$-greedy with respect to the **current** table: every action gets $\epsilon / A$, and the remaining $1 - \epsilon$ is split **equally among all actions tied for the maximum** of $Q(s', \cdot)$ (exact equality).

Write `expected_sarsa(Q, transitions, epsilon=0.1, alpha=0.1, gamma=0.99)` and return the updated table.

## Inputs
- `Q`: float array `(S, A)`. Do not modify it.
- `transitions`: float array `(N, 5)` with rows `[s, a, r, s_next, done]`; integer columns are stored as floats and `done` is 0/1.
- `epsilon` in `[0, 1]`.

## Rules
- Apply updates in row order, recomputing $\pi(\cdot \mid s')$ from the table as it is at that row.
- With $d = 1$ the target is just $r$.

## Output
Float array `(S, A)`.
"""

STARTER = '''import numpy as np

def expected_sarsa(Q, transitions, epsilon=0.1, alpha=0.1, gamma=0.99):
    """transitions: (N, 5) rows of [s, a, r, s_next, done]. Return the updated Q (S, A)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def expected_sarsa(Q, transitions, epsilon=0.1, alpha=0.1, gamma=0.99):
    Q = np.array(Q, dtype=float)
    A = Q.shape[1]
    for s, a, r, s2, done in np.asarray(transitions, dtype=float):
        s, a, s2 = int(s), int(a), int(s2)
        q = Q[s2]
        best = q == q.max()                                  # every exactly-tied greedy action
        pi = np.full(A, epsilon / A) + (1.0 - epsilon) * best / best.sum()
        target = r + gamma * (1.0 - done) * (pi @ q)
        Q[s, a] += alpha * (target - Q[s, a])
    return Q
'''

HINTS = [
    'Build the target policy for `s_next` as a length-A probability vector, then the bootstrap is a dot product with `Q[s_next]`.',
    'Start from `epsilon / A` everywhere. Find the greedy set with `q == q.max()` and add `(1 - epsilon) / count` to each of those entries.',
    'Rebuild that vector inside the loop for every row: the table, and so the greedy set, changes as you update.',
]

_r = np.random.default_rng(91)
_N = 30
TR = np.column_stack([_r.integers(0, 4, _N), _r.integers(0, 3, _N), np.round(_r.normal(size=_N), 2),
                      _r.integers(0, 4, _N), (_r.uniform(size=_N) < 0.15)]).astype(float)
TESTS = [
    (np.array([[0.0, 0.0], [4.0, 0.0]]), np.array([[0, 0, 1.0, 1, 0]]), 0.5, 1.0, 1.0),
    (np.array([[0.0, 0.0], [2.0, 2.0]]), np.array([[0, 1, 0.0, 1, 0]]), 0.2, 0.5, 0.9),
    (np.array([[0.0, 0.0], [4.0, 0.0]]), np.array([[0, 0, 1.0, 1, 1]]), 0.5, 1.0, 1.0),
    (np.array([[1.0, 3.0, 3.0]]), np.array([[0, 0, 0.0, 0, 0]]), 0.0, 1.0, 1.0),
    (np.array([[1.0, 3.0, 3.0]]), np.array([[0, 0, 0.0, 0, 0]]), 1.0, 1.0, 1.0),
    (np.zeros((4, 3)), TR, 0.1, 0.3, 0.9),
    (np.round(_r.normal(size=(4, 3)), 1), TR, 0.3, 0.1, 0.99),
]

EXPLANATION = r"""
## Why take the expectation
SARSA's target is noisy for two reasons: the next state and the next action. Expected SARSA averages out the second analytically, so it has lower variance at the same cost per update ($O(A)$). It also generalises both neighbours: with $\epsilon = 0$ the expectation is the greedy value, which is Q-learning (test 4); with $\pi$ equal to the behaviour policy it is the expected form of SARSA. Because the target policy need not match the behaviour policy, it can be used off-policy too.

## Ties
Splitting the greedy mass among exact ties makes the target independent of how ties are broken. Tests start from zero (or rounded) tables, so exact ties really occur, and test 2 ties both actions in the next state.

## Complexity
$O(NA)$ time, $O(SA)$ memory.

## Common bugs
- Giving the greedy action $1 - \epsilon$ instead of $1 - \epsilon + \epsilon/A$ (forgetting that it also gets the exploration share).
- Giving the whole greedy mass to `argmax` on ties.
- Computing $\pi$ once from the initial table instead of per update.

## Interview follow-ups
- Why can Expected SARSA use a large learning rate (even $\alpha = 1$) on deterministic tasks where SARSA cannot?
- How would you write the same target for a softmax (Boltzmann) target policy?
"""
