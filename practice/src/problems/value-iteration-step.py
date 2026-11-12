import numpy as np

META = dict(title='One Bellman backup (value iteration)', tags=['Reinforcement Learning'], diff='Medium', runtime='numpy',
            lessons=[], related=['gae-advantages'])

STATEMENT = r"""
For a finite MDP with transition tensor $P[s, a, s'] = \Pr(s' \mid s, a)$ and expected rewards $R[s, a]$, one sweep of value iteration applies the Bellman optimality backup to every state:

$$Q(s, a) = R(s, a) + \gamma \sum_{s'} P(s, a, s')\, V(s'), \qquad V_{\text{new}}(s) = \max_a Q(s, a)$$

Write `bellman_backup(P, R, V, gamma=0.9)` that returns `(V_new, policy)`, where `policy[s]` is the greedy action $\arg\max_a Q(s, a)$.

## Requirements
- `P`: `(S, A, S)`, `R`: `(S, A)`, `V`: `(S,)`.
- Ties in the argmax go to the **lowest** action index.
- One vectorised expression for $Q$ (hint: `einsum` or `@`), no loops over states.
"""

STARTER = '''import numpy as np

def bellman_backup(P, R, V, gamma=0.9):
    """P: (S, A, S), R: (S, A), V: (S,). Return (V_new (S,), greedy policy (S,) of ints)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def bellman_backup(P, R, V, gamma=0.9):
    P, R, V = (np.asarray(x, dtype=float) for x in (P, R, V))
    Q = R + gamma * (P @ V)          # (S, A, S) @ (S,) -> (S, A): expected next-state value
    return Q.max(axis=1), Q.argmax(axis=1)
'''

HINTS = [
    'The expected next value $\\sum_{s\'} P[s,a,s\'] V[s\']$ for every $(s, a)$ is a contraction over the last axis of P.',
    '`P @ V` does it: matmul treats the leading axes as a batch. So does `np.einsum("sat,t->sa", P, V)`.',
    '`Q.max(axis=1)` gives the new values. `Q.argmax(axis=1)` gives the policy, returning the first maximum on ties.',
]

_r = np.random.default_rng(6)
def _mdp(S, A):
    P = _r.uniform(size=(S, A, S)); P /= P.sum(axis=2, keepdims=True)
    return P, _r.normal(size=(S, A))
P1, R1 = _mdp(4, 3)
# 2-state chain: action 0 stays, action 1 switches; reward only for staying in state 1
Pc = np.zeros((2, 2, 2)); Pc[0, 0, 0] = Pc[1, 0, 1] = Pc[0, 1, 1] = Pc[1, 1, 0] = 1.0
Rc = np.array([[0.0, 0.0], [1.0, 0.0]])
TESTS = [
    (Pc, Rc, np.zeros(2)),
    (Pc, Rc, np.array([0.0, 10.0])),
    (Pc, Rc, np.array([5.0, 5.0]), 0.5),
    (P1, R1, np.zeros(4)),
    (P1, R1, _r.normal(size=4), 0.99),
    (np.full((4, 3, 4), 0.25), np.zeros((4, 3)), np.ones(4), 1.0),   # exact tie in every state: lowest action wins
]

EXPLANATION = r"""
## Why repeating this converges
The backup is a $\gamma$-contraction in the max norm: $\lVert T V - T U\rVert_\infty \le \gamma \lVert V - U\rVert_\infty$. Repeating it from any start converges to the unique fixed point $V^*$, and the greedy policy with respect to $V^*$ is optimal. The error shrinks by a factor of $\gamma$ per sweep.

## Ties
Test 6 has uniform transitions, zero rewards and a constant $V$, so every action ties exactly. Deterministic tie-breaking (lowest index) keeps policies reproducible. `argmax` already does this.

## Cost
Each sweep is $O(S^2 A)$. Fine for gridworlds, hopeless for real state spaces. That gap is why we move to sampled methods (Q-learning) and function approximation (DQN). DQN's target $r + \gamma \max_{a'} Q(s', a')$ is this same backup, estimated from one sample.

## Interview follow-ups
- Value iteration vs policy iteration: when does each converge faster?
- How would you detect convergence? (Stop when $\lVert V_{k+1} - V_k\rVert_\infty < \epsilon(1-\gamma)/2\gamma$.)
"""
