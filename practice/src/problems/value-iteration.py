import numpy as np

META = dict(title='Value iteration to convergence', tags=['Reinforcement Learning'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:07.01'],
            related=['value-iteration-step', 'policy-iteration', 'bellman-expectation-equation'])

STATEMENT = r"""
Value iteration finds the optimal values of a finite MDP by applying the Bellman optimality backup until the values stop changing:

$$Q_k(s, a) = R(s, a) + \gamma \sum_{s'} P(s, a, s')\, V_k(s'), \qquad V_{k+1}(s) = \max_a Q_k(s, a)$$

Write `value_iteration(P, R, gamma=0.9, theta=1e-8, max_iters=10000)` and return `(V, policy, n_iters)`.

## Algorithm (follow it exactly)
1. Start from $V_0 = 0$ for every state.
2. Repeat: compute $V_{k+1}$ from $V_k$ for **all** states at once (synchronous update, not in place). Count this as one iteration.
3. Stop after the first iteration where $\max_s |V_{k+1}(s) - V_k(s)| < \theta$, or after `max_iters` iterations, whichever comes first.

## Output
- `V`: float array `(S,)`, the last $V_{k+1}$ computed.
- `policy`: int array `(S,)`, the greedy action $\arg\max_a \big[R(s,a) + \gamma \sum_{s'} P(s,a,s') V(s')\big]$ for the returned `V`. Ties go to the **lowest** action index.
- `n_iters`: int, the number of iterations performed.

`P` is `(S, A, S)` with rows summing to 1, `R` is `(S, A)`, and $0 \le \gamma < 1$.
"""

STARTER = '''import numpy as np

def value_iteration(P, R, gamma=0.9, theta=1e-8, max_iters=10000):
    """Return (V (S,), greedy policy (S,) ints, number of iterations)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def value_iteration(P, R, gamma=0.9, theta=1e-8, max_iters=10000):
    P, R = np.asarray(P, dtype=float), np.asarray(R, dtype=float)
    V = np.zeros(R.shape[0])
    n = 0
    while n < max_iters:
        V_new = (R + gamma * (P @ V)).max(axis=1)   # synchronous backup of every state
        n += 1
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < theta:
            break
    policy = (R + gamma * (P @ V)).argmax(axis=1)  # greedy w.r.t. the returned V; argmax takes the first max
    return V, policy, n
'''

HINTS = [
    'One iteration is exactly one Bellman optimality backup of the whole value vector. Vectorise it: `P @ V` has shape `(S, A)`.',
    'Keep the old vector until the new one is complete, measure `abs(V_new - V).max()`, then replace. Increment the counter before the stopping test.',
    'After the loop, recompute `Q = R + gamma * (P @ V)` from the returned `V` and take `Q.argmax(axis=1)`. `argmax` already breaks ties toward the lowest index.',
]

_r = np.random.default_rng(31)
def _mdp(S, A):
    P = _r.uniform(size=(S, A, S)); P /= P.sum(axis=2, keepdims=True)
    return P, _r.normal(size=(S, A))
# Corridor 0..4, action 0 = left, 1 = right, state 4 is an absorbing goal. Entering it pays 1.
Pk = np.zeros((5, 2, 5)); Rk = np.zeros((5, 2))
for s in range(4):
    Pk[s, 0, max(s - 1, 0)] = 1.0
    Pk[s, 1, s + 1] = 1.0
Pk[4, :, 4] = 1.0
Rk[3, 1] = 1.0
# Same corridor but slippery: the intended move happens with prob 0.8, otherwise the agent stays.
Ps = 0.8 * Pk + 0.2 * np.eye(5)[:, None, :]; Ps[4] = Pk[4]
P4, R4 = _mdp(4, 3)
P6, R6 = _mdp(6, 2)
TESTS = [
    (Pk, Rk, 0.9),
    (Pk, Rk, 0.5, 1e-3),
    (Ps, Rk, 0.9),
    (Pk, Rk, 0.9, 1e-8, 2),
    (np.ones((1, 2, 1)), np.array([[1.0, 2.0]]), 0.5, 1e-6),
    (P4, R4, 0.9),
    (P6, R6, 0.95, 1e-6),
]

EXPLANATION = r"""
## Why it converges
The backup $T$ is a $\gamma$-contraction in the max norm, so $\lVert V_{k+1} - V^*\rVert_\infty \le \gamma^{k+1} \lVert V^*\rVert_\infty$. The gap between consecutive iterates also bounds the distance to the fixed point: $\lVert V_{k+1} - V^*\rVert_\infty \le \frac{\gamma}{1-\gamma}\lVert V_{k+1} - V_k\rVert_\infty$. So $\theta$ controls accuracy, and the number of iterations grows like $\log(1/\theta) / \log(1/\gamma)$. Compare tests 1 and 2: a smaller $\gamma$ and looser $\theta$ finish much sooner.

## The greedy policy is usually right long before V is
Actions only need the *ordering* of Q values to be correct, which typically settles well before the values themselves. Test 4 stops after 2 iterations and the policy is already partly right. This is the intuition behind policy iteration and modified policy iteration.

## Complexity
Each iteration is $O(S^2 A)$. Memory is $O(S^2 A)$ for `P`.

## Common bugs
- Updating `V` in place state by state. That is Gauss-Seidel value iteration: it also converges, but to different intermediate values and iteration counts.
- Counting iterations off by one (testing before incrementing, or not counting the final sweep).
- Taking the policy from the previous iteration's Q instead of from the returned `V`.

## Interview follow-ups
- How would you pick $\theta$ to guarantee the greedy policy is $\epsilon$-optimal?
- In a goal-reaching task with reward only at the goal, what does $\gamma$ do to the learned behaviour? (It's what makes shorter paths better.)
"""
