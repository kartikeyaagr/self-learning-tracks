import numpy as np

META = dict(title='TD targets: terminated versus truncated', tags=['Reinforcement Learning'], diff='Easy', runtime='numpy',
            related=['reward-and-termination', 'q-learning-update'])

STATEMENT = r"""
Write `td_targets(rewards, next_values, terminated, truncated, gamma)` for a batch of transitions.

- $y_t = r_t + \gamma\,(1 - \text{terminated}_t)\,V(s_{t+1})$.
- A truncated transition **still bootstraps** from `next_values`, because the episode was cut, not finished. (The caller must pass the value of the real next observation, not of the reset state.)
- `truncated` is part of the signature so you can see it is deliberately not used in the formula.

Return a list of floats.
"""

STARTER = '''import numpy as np

def td_targets(rewards, next_values, terminated, truncated, gamma):
    """One-step TD targets that bootstrap through truncation but not termination."""
    done = np.asarray(terminated, bool) | np.asarray(truncated, bool)
    return (np.asarray(rewards, float) + gamma * (1 - done) * np.asarray(next_values, float)).tolist()
'''

SOLUTION = '''import numpy as np

def td_targets(rewards, next_values, terminated, truncated, gamma):
    r, v = np.asarray(rewards, float), np.asarray(next_values, float)
    term = np.asarray(terminated, bool)
    return (r + gamma * (~term) * v).tolist()
'''

HINTS = [
    'Only termination zeroes the bootstrap term.',
    'The starter is the bug that was common before Gymnasium split the flags.',
]

TESTS = [
    ([1.0], [5.0], [True], [False], 0.9),
    ([1.0], [5.0], [False], [True], 0.9),
    ([0.0, 0.0, 1.0], [2.0, 3.0, 0.0], [False, False, True], [False, False, False], 0.5),
    ([-0.1, -0.1], [4.0, 4.0], [False, False], [True, False], 0.99),
    ([0.5, 0.5, 0.5, 0.5], [1.0, 2.0, 3.0, 4.0], [True, False, True, False], [False, True, False, True], 1.0),
]

EXPLANATION = r"""
## Where the flags matter
The Bellman target is the only place the two flags differ. Termination says the future is worth nothing; truncation says the future exists but you did not observe it, so the value estimate stands in for it.

## The practical trap
Vectorised environments reset automatically. When an episode is truncated, the observation returned with the step is often already the first state of the next episode. Using that as $s_{t+1}$ bootstraps from the wrong state. Gymnasium provides the true final observation separately for this reason.

## Interview follow-ups
- How does the same distinction show up in GAE, where advantages are accumulated backwards across a rollout?
"""
