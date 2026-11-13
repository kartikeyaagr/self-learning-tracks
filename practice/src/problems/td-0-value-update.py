import numpy as np

META = dict(title='TD(0) value updates from a transition stream', tags=['Reinforcement Learning'], diff='Easy',
            runtime='numpy', lessons=['air-fanout-v1:07.01'],
            related=['monte-carlo-policy-evaluation', 'sarsa-update', 'q-learning-update'])

STATEMENT = r"""
TD(0) learns state values online, one transition at a time, by moving $V(s)$ toward a bootstrapped target:

$$\delta = r + \gamma (1 - d)\, V(s') - V(s), \qquad V(s) \leftarrow V(s) + \alpha\, \delta$$

where $d = 1$ if $s'$ is terminal (then there is nothing to bootstrap from).

Write `td0(V, transitions, alpha=0.1, gamma=0.99)` and return `(V_new, deltas)`.

## Inputs
- `V`: float array `(S,)`, the initial values. Do not modify it; return a new array.
- `transitions`: float array `(N, 4)`, one row per transition `[s, r, s_next, done]`. The state columns hold integers stored as floats.

## Rules
- Apply the updates **in row order**. Each update sees the values left by the previous ones, so a later transition out of a state that was just updated uses the new value.

## Output
- `V_new`: float array `(S,)` after all updates.
- `deltas`: float array `(N,)`, the TD error $\delta$ of each transition at the moment it was applied.
"""

STARTER = '''import numpy as np

def td0(V, transitions, alpha=0.1, gamma=0.99):
    """transitions: (N, 4) rows of [s, r, s_next, done]. Return (V_new (S,), deltas (N,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def td0(V, transitions, alpha=0.1, gamma=0.99):
    V = np.array(V, dtype=float)                 # copy: never modify the caller's array
    deltas = np.zeros(len(transitions))
    for i, (s, r, s2, done) in enumerate(np.asarray(transitions, dtype=float)):
        s, s2 = int(s), int(s2)
        delta = r + gamma * (1.0 - done) * V[s2] - V[s]
        V[s] += alpha * delta
        deltas[i] = delta
    return V, deltas
'''

HINTS = [
    'This is inherently sequential: loop over the rows and update one entry of a *copy* of `V` each time.',
    'Cast the state columns to `int` before indexing. Compute $\\delta$ before you write to `V[s]`.',
    'With `done = 1` the target is just `r`: multiply the bootstrap term by `(1 - done)` rather than branching.',
]

_r = np.random.default_rng(61)
_N = 25
TR = np.column_stack([_r.integers(0, 6, _N), np.round(_r.normal(size=_N), 2), _r.integers(0, 6, _N),
                      (_r.uniform(size=_N) < 0.2)]).astype(float)
TESTS = [
    (np.zeros(3), np.array([[0, 1.0, 1, 0], [1, 2.0, 2, 1]]), 0.5, 0.9),
    (np.zeros(2), np.array([[0, 1.0, 0, 0], [0, 1.0, 0, 0], [0, 1.0, 0, 0]]), 0.5, 0.5),
    (np.array([1.0, 2.0, 3.0]), np.array([[2, 0.0, 2, 1]]), 1.0, 0.99),
    (np.array([1.0, 2.0, 3.0]), np.array([[2, 0.0, 2, 0]]), 1.0, 0.99),
    (np.array([0.0, 10.0]), np.array([[0, 0.0, 1, 0], [1, 0.0, 0, 0]]), 0.1, 1.0),
    (np.zeros(6), TR, 0.1, 0.95),
    (_r.normal(size=6), TR, 0.7, 0.0),
]

EXPLANATION = r"""
## Why the target makes sense
The Bellman expectation equation says $V^\pi(s) = \mathbb{E}[r + \gamma V^\pi(s')]$. TD(0) replaces the expectation with the one sample just observed and the unknown $V^\pi(s')$ with the current estimate. Each update is a small stochastic step toward the fixed point. With suitably decaying $\alpha$ it converges to $V^\pi$ in the tabular case.

## Order matters
Test 2 updates a self-loop three times. Each target uses the value just written, so the answer depends on processing rows in order. A vectorised `np.add.at` over all rows would apply every update against the *original* values, which is a different algorithm (batch TD).

## Complexity
$O(N)$ time, $O(S)$ memory. Each update is constant time, which is why TD scales to online settings where Monte Carlo has to wait for the episode to end.

## Common bugs
- Bootstrapping from terminal transitions: the target should be just $r$ when `done = 1` (tests 3 and 4).
- Mutating the input `V` in place.
- Writing `V[s]` before computing $\delta$ when `s == s_next`, so the bootstrap uses the updated value.

## Interview follow-ups
- TD($\lambda$) and eligibility traces: how do they interpolate between TD(0) and Monte Carlo?
- Why can TD with linear function approximation diverge off-policy (the "deadly triad")?
"""
