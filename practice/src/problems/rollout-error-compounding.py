import numpy as np

META = dict(title='Measure how rollout errors compound', tags=['Reinforcement Learning', 'Time Series'], diff='Easy', runtime='numpy',
            related=['episodes-to-transition-pairs'])

STATEMENT = r"""
A learned model can look accurate one step at a time and still drift far when it is rolled forward on its own predictions.

Both the true system and the model are linear: $s_{t+1} = A s_t + B a_t$ and $\hat s_{t+1} = \hat A \hat s_t + \hat B a_t$.

Write `rollout_errors(A, B, A_hat, B_hat, s0, actions)` for an action sequence of length $H$:

- `one_step[t]`: $\lVert \hat A s_t + \hat B a_t - s_{t+1} \rVert_2$, where $s_t$ is the **true** state (teacher forcing).
- `open_loop[t]`: $\lVert \hat s_{t+1} - s_{t+1} \rVert_2$, where $\hat s$ starts at $s_0$ and is fed its own predictions.

Return a dict with both lists of length $H$.
"""

STARTER = '''import numpy as np

def rollout_errors(A, B, A_hat, B_hat, s0, actions):
    """One-step and open-loop prediction error of a linear model over a horizon."""
    A, B, A_hat, B_hat = (np.asarray(m, float) for m in (A, B, A_hat, B_hat))
    s = np.asarray(s0, float)
    out = []
    for a in np.asarray(actions, float):
        nxt = A @ s + B @ a
        out.append(float(np.linalg.norm(A_hat @ s + B_hat @ a - nxt)))
        s = nxt
    return {"one_step": out, "open_loop": out}
'''

SOLUTION = '''import numpy as np

def rollout_errors(A, B, A_hat, B_hat, s0, actions):
    A, B, A_hat, B_hat = (np.asarray(m, float) for m in (A, B, A_hat, B_hat))
    s = np.asarray(s0, float)
    sh = s.copy()
    one, ol = [], []
    for a in np.asarray(actions, float):
        nxt = A @ s + B @ a
        one.append(float(np.linalg.norm(A_hat @ s + B_hat @ a - nxt)))
        sh = A_hat @ sh + B_hat @ a
        ol.append(float(np.linalg.norm(sh - nxt)))
        s = nxt
    return {"one_step": one, "open_loop": ol}
'''

HINTS = [
    'Keep two states: the true one, and the model’s imagined one that never sees the truth after s0.',
    'At t = 0 the two errors are identical; they separate from the second step on.',
]

def _case(seed, n, H, err, rho):
    g = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(g.normal(size=(n, n)))
    A = rho * Q
    B = g.normal(size=(n, 1))
    return (A, B, A + err * g.normal(size=(n, n)), B, g.normal(size=n), g.normal(size=(H, 1)))

TESTS = [
    ([[1.0]], [[1.0]], [[1.1]], [[1.0]], [1.0], [[0.0], [0.0], [0.0]]),
    ([[0.5]], [[1.0]], [[0.6]], [[1.0]], [1.0], [[1.0], [1.0], [1.0], [1.0]]),
    _case(1, 3, 10, 0.02, 1.0),
    _case(2, 3, 10, 0.02, 0.8),
    _case(3, 4, 20, 0.05, 1.05),
]

EXPLANATION = r"""
## Why errors compound
Each open-loop step applies the model's error to a state that is already wrong. For a linear model the gap obeys $e_{t+1} = \hat A e_t + (\hat A - A)s_t$: the old error is carried forward and amplified by $\hat A$, and a new error is added. With a stable system (spectral radius below 1, the fourth test) the old errors fade and the gap levels off. With an unstable one (the fifth test) it grows exponentially, even though the one-step error stays small.

## What this means for planning
A planner optimising against the model will happily exploit regions where the model is wrong, and the further ahead it looks, the more wrong the model is there. Practical fixes: short horizons with frequent replanning (model predictive control), ensembles that flag disagreement, and training on multi-step losses.

## Interview follow-ups
- Why is the one-step validation loss a poor model-selection metric for a world model used in planning?
"""
