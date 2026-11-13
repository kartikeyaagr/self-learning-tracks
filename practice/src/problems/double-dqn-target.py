import numpy as np

META = dict(title='Double DQN targets', tags=['Reinforcement Learning'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:07.03'],
            related=['dqn-loss-with-target-network', 'q-learning-update', 'dueling-dqn-decomposition'])

STATEMENT = r"""
Standard DQN uses the target network both to **choose** and to **evaluate** the next action, which overestimates values: the max of noisy estimates is biased upward. Double DQN decouples the two. The online network picks the action, the target network scores it:

$$a^*_i = \arg\max_{a} Q(s'_i, a; \theta), \qquad y^{\text{double}}_i = r_i + \gamma (1 - d_i)\, Q(s'_i, a^*_i; \theta^-)$$

For comparison, the vanilla target is $y^{\text{dqn}}_i = r_i + \gamma (1 - d_i) \max_a Q(s'_i, a; \theta^-)$.

Write `double_dqn_targets(q_online_next, q_target_next, rewards, dones, gamma=0.99)` and return `(double, vanilla)`.

## Inputs
- `q_online_next`: float `(B, A)`, the online network at the next states.
- `q_target_next`: float `(B, A)`, the target network at the next states.
- `rewards`: float `(B,)`; `dones`: `(B,)` of 0/1.

## Rules
- Ties in $\arg\max$ go to the **lowest** action index.

## Output
Two float arrays of shape `(B,)`. Note that `double <= vanilla` elementwise.
"""

STARTER = '''import numpy as np

def double_dqn_targets(q_online_next, q_target_next, rewards, dones, gamma=0.99):
    """Return (double DQN targets (B,), vanilla DQN targets (B,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def double_dqn_targets(q_online_next, q_target_next, rewards, dones, gamma=0.99):
    qo, qt = np.asarray(q_online_next, dtype=float), np.asarray(q_target_next, dtype=float)
    live = gamma * (1.0 - np.asarray(dones, dtype=float))
    a_star = qo.argmax(axis=1)                               # online net selects
    chosen = qt[np.arange(len(qt)), a_star]                  # target net evaluates
    return rewards + live * chosen, rewards + live * qt.max(axis=1)
'''

HINTS = [
    'Two steps: select with one network, evaluate with the other. Selection is an `argmax` over `q_online_next`.',
    'Evaluate with fancy indexing: `q_target_next[np.arange(B), a_star]`.',
    'Both targets share `rewards + gamma * (1 - dones) * (...)`. Only the bracket differs: the selected entry vs. the row max of the target network.',
]

_r = np.random.default_rng(181)
TESTS = [
    (np.array([[1.0, 2.0], [3.0, 0.0]]), np.array([[5.0, 1.0], [0.0, 2.0]]), np.array([0.0, 1.0]), np.array([0, 0]), 0.5),
    (np.array([[0.0, 1.0, 0.5]]), np.array([[2.0, 0.0, 4.0]]), np.array([1.0]), np.array([1]), 0.9),
    (np.array([[2.0, 2.0]]), np.array([[1.0, 3.0]]), np.array([0.0]), np.array([0]), 1.0),
    (np.array([[7.0]]), np.array([[-2.0]]), np.array([0.5]), np.array([0]), 0.99),
    (_r.normal(size=(8, 4)), _r.normal(size=(8, 4)), _r.normal(size=8), (_r.uniform(size=8) < 0.25).astype(int), 0.99),
    (_r.normal(size=(5, 6)), _r.normal(size=(5, 6)), np.zeros(5), np.zeros(5, dtype=int), 0.9),
]

EXPLANATION = r"""
## Where the overestimation comes from
Suppose all true action values are equal and each estimate has independent zero-mean noise. Then $\mathbb{E}[\max_a \hat Q(a)] > \max_a Q(a)$: the max picks whichever estimate got lucky. Bootstrapping feeds that bias back into the next targets, so it compounds. Using a second, roughly independent estimate to evaluate the chosen action removes the selection bias, because the target network's error on $a^*$ isn't the error that made $a^*$ win.

## Why `double <= vanilla`
$Q(s', a^*; \theta^-) \le \max_a Q(s', a; \theta^-)$ for any $a^*$. The two are equal exactly when both networks agree on the best action (test 1, row 1 differs; row 0 doesn't). Terminal rows (test 2) make both targets equal to the reward.

## Complexity
$O(BA)$.

## Common bugs
- Swapping roles (target network selects, online evaluates). That still decorrelates but isn't the published method, and it changes the numbers.
- Taking `argmax` over `q_target_next` and then indexing `q_online_next`: same swap, different shape of bug.
- Forgetting the terminal mask on one of the two targets.

## Interview follow-ups
- Tabular Double Q-learning keeps two tables and randomly updates one using the other. How does Double DQN get away with reusing the target network as the second estimator?
- What other DQN extensions (Rainbow) combine well with Double DQN?
"""
