import numpy as np

META = dict(title='Exact policy evaluation with the Bellman expectation equation', tags=['Reinforcement Learning', 'Linear Algebra'],
            diff='Easy', runtime='numpy', lessons=['air-fanout-v1:07.01'],
            related=['value-iteration-step', 'policy-iteration', 'value-iteration'])

STATEMENT = r"""
For a fixed (possibly stochastic) policy $\pi$, the state values satisfy the **Bellman expectation equation**:

$$V^\pi(s) = \sum_a \pi(a \mid s)\Big[R(s,a) + \gamma \sum_{s'} P(s, a, s')\, V^\pi(s')\Big]$$

This is a linear system in $V^\pi$, so for a small MDP it can be solved exactly instead of iterated.

Write `evaluate_policy(P, R, pi, gamma=0.9)` and return `(V, Q)` where

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s,a,s')\, V^\pi(s')$$

## Inputs
- `P`: `(S, A, S)` transition probabilities, each `P[s, a]` sums to 1.
- `R`: `(S, A)` expected immediate rewards.
- `pi`: `(S, A)` action probabilities, each row sums to 1.
- `gamma`: in `[0, 1)`, so the system always has a unique solution.

## Output
`V` of shape `(S,)` and `Q` of shape `(S, A)`, both float. Solve the system directly (`np.linalg.solve`), don't iterate.
"""

STARTER = '''import numpy as np

def evaluate_policy(P, R, pi, gamma=0.9):
    """P: (S, A, S), R: (S, A), pi: (S, A). Return (V (S,), Q (S, A))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def evaluate_policy(P, R, pi, gamma=0.9):
    P, R, pi = (np.asarray(x, dtype=float) for x in (P, R, pi))
    S = R.shape[0]
    R_pi = (pi * R).sum(axis=1)                     # expected reward under pi, (S,)
    P_pi = np.einsum('sa,sat->st', pi, P)           # state-to-state transitions under pi, (S, S)
    V = np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)   # (I - gamma P_pi) V = R_pi
    Q = R + gamma * (P @ V)
    return V, Q
'''

HINTS = [
    'Average out the action first: under $\\pi$ the MDP becomes a Markov chain with reward vector $R_\\pi$ and transition matrix $P_\\pi$.',
    '$R_\\pi(s) = \\sum_a \\pi(a|s) R(s,a)$ and $P_\\pi(s,s\') = \\sum_a \\pi(a|s) P(s,a,s\')$. Then $V = R_\\pi + \\gamma P_\\pi V$.',
    'Rearrange to $(I - \\gamma P_\\pi) V = R_\\pi$ and call `np.linalg.solve`. Then `Q = R + gamma * (P @ V)`.',
]

_r = np.random.default_rng(21)
def _mdp(S, A):
    P = _r.uniform(size=(S, A, S)); P /= P.sum(axis=2, keepdims=True)
    pi = _r.uniform(size=(S, A)); pi /= pi.sum(axis=1, keepdims=True)
    return P, _r.normal(size=(S, A)), pi
# 2-state chain: action 0 stays, action 1 switches; reward 1 for acting in state 1
Pc = np.zeros((2, 2, 2)); Pc[0, 0, 0] = Pc[1, 0, 1] = Pc[0, 1, 1] = Pc[1, 1, 0] = 1.0
Rc = np.array([[0.0, 0.0], [1.0, 1.0]])
P5, R5, pi5 = _mdp(5, 3)
P3, R3, pi3 = _mdp(3, 2)
TESTS = [
    (Pc, Rc, np.array([[1.0, 0.0], [1.0, 0.0]]), 0.5),
    (Pc, Rc, np.array([[0.5, 0.5], [0.5, 0.5]]), 0.5),
    (Pc, Rc, np.array([[0.0, 1.0], [0.0, 1.0]]), 0.9),
    (Pc, Rc, np.array([[0.0, 1.0], [1.0, 0.0]]), 0.0),
    (np.ones((1, 1, 1)), np.array([[2.0]]), np.array([[1.0]]), 0.75),
    (P3, R3, pi3, 0.9),
    (P5, R5, pi5, 0.99),
]

EXPLANATION = r"""
## Why it's a linear system
Fixing $\pi$ removes the max that makes the *optimality* equation nonlinear. What is left is $V = R_\pi + \gamma P_\pi V$, an ordinary linear equation. $P_\pi$ is row-stochastic, so its eigenvalues have modulus at most 1 and $I - \gamma P_\pi$ is invertible for any $\gamma < 1$. That guarantees one exact answer.

## Complexity
Building $P_\pi$ is $O(S^2 A)$ and the solve is $O(S^3)$. That is fine up to a few thousand states. Beyond that, iterate $V \leftarrow R_\pi + \gamma P_\pi V$ (each sweep $O(S^2)$, error shrinking by $\gamma$) or switch to sampled TD methods.

## Common bugs
- Contracting over the wrong axis of `P` in the einsum, e.g. summing over the current state instead of the next state.
- Using `np.linalg.inv(...) @ R_pi`. It works but is slower and less accurate than `solve`.
- Forgetting the $\pi$-weighting on the reward: $R_\pi$ is a weighted average across actions, not a plain mean.

## Interview follow-ups
- How does $V^\pi$ relate to $Q^\pi$? ($V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)$. Check it on your output.)
- What breaks at $\gamma = 1$, and when does an episodic MDP still have a finite solution?
"""
