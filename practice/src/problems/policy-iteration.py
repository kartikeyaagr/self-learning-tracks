import numpy as np

META = dict(title='Policy iteration', tags=['Reinforcement Learning'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:07.01'],
            related=['bellman-expectation-equation', 'value-iteration', 'value-iteration-step'])

STATEMENT = r"""
Policy iteration alternates two steps until the policy stops changing:

1. **Evaluate** the current deterministic policy $\pi$ exactly by solving $V = R_\pi + \gamma P_\pi V$, where $R_\pi(s) = R(s, \pi(s))$ and $P_\pi(s, s') = P(s, \pi(s), s')$.
2. **Improve** it greedily using $Q(s,a) = R(s,a) + \gamma \sum_{s'} P(s,a,s')\, V(s')$.

Write `policy_iteration(P, R, gamma=0.9)` and return `(policy, V, n_evals)`.

## Exact rules
- Start from the policy that picks action 0 in every state.
- Improvement must not flip-flop between equally good actions. In each state, let $m = \max_a Q(s,a)$. If $Q(s, \pi(s)) \ge m - 10^{-10}$, **keep** the current action. Otherwise switch to the lowest action index $a$ with $Q(s,a) \ge m - 10^{-10}$.
- Stop when an improvement step changes no state.

## Output
- `policy`: int array `(S,)`, the final policy.
- `V`: float array `(S,)`, the exact value of that final policy.
- `n_evals`: int, how many policy evaluations were performed (including the last one, which confirmed the policy is stable).

`P` is `(S, A, S)` with rows summing to 1, `R` is `(S, A)`, $0 \le \gamma < 1$.
"""

STARTER = '''import numpy as np

def policy_iteration(P, R, gamma=0.9):
    """Return (policy (S,) ints, V (S,), number of policy evaluations)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def policy_iteration(P, R, gamma=0.9):
    P, R = np.asarray(P, dtype=float), np.asarray(R, dtype=float)
    S = R.shape[0]
    idx = np.arange(S)
    pi = np.zeros(S, dtype=int)
    n = 0
    while True:
        # exact evaluation of the deterministic policy
        V = np.linalg.solve(np.eye(S) - gamma * P[idx, pi], R[idx, pi])
        n += 1
        Q = R + gamma * (P @ V)
        good = Q >= Q.max(axis=1, keepdims=True) - 1e-10     # actions within tolerance of the best
        keep = good[idx, pi]
        new_pi = np.where(keep, pi, good.argmax(axis=1))      # argmax of a bool row = first True
        if np.array_equal(new_pi, pi):
            return pi, V, n
        pi = new_pi
'''

HINTS = [
    'Fancy indexing builds the policy\'s chain in one go: `P[np.arange(S), pi]` is `(S, S)` and `R[np.arange(S), pi]` is `(S,)`.',
    'Evaluate with `np.linalg.solve(I - gamma * P_pi, R_pi)`, then compute `Q` from that `V`. Count every evaluation, including the final one.',
    'Make a boolean mask `good = Q >= Q.max(1, keepdims=True) - 1e-10`. Keep `pi[s]` where `good[s, pi[s]]` is True, else use `good.argmax(1)` (the first True). Stop when nothing changed.',
]

_r = np.random.default_rng(41)
def _mdp(S, A):
    P = _r.uniform(size=(S, A, S)); P /= P.sum(axis=2, keepdims=True)
    return P, _r.normal(size=(S, A))
# Corridor 0..4, action 0 = left, 1 = right, absorbing goal 4. Entering it pays 1.
Pk = np.zeros((5, 2, 5)); Rk = np.zeros((5, 2))
for s in range(4):
    Pk[s, 0, max(s - 1, 0)] = 1.0
    Pk[s, 1, s + 1] = 1.0
Pk[4, :, 4] = 1.0
Rk[3, 1] = 1.0
# Same corridor with a step cost on every move and a big reward at the goal
Rc = np.full((5, 2), -0.1); Rc[3, 1] = 5.0; Rc[4] = 0.0
P4, R4 = _mdp(4, 3)
P7, R7 = _mdp(7, 4)
TESTS = [
    (Pk, Rk, 0.9),
    (Pk, Rc, 0.5),
    (Pk, np.zeros((5, 2)), 0.9),
    (np.ones((1, 3, 1)), np.array([[1.0, 3.0, 3.0]]), 0.5),
    (P4, R4, 0.9),
    (P7, R7, 0.95),
    (P7, R7, 0.0),
]

EXPLANATION = r"""
## Why it terminates
Each improvement step produces a policy at least as good as the last in every state (the policy improvement theorem). If it strictly improves somewhere, it can never return to an earlier policy. There are only $A^S$ deterministic policies, so the loop must stop, and when it stops the values satisfy the Bellman optimality equation, so the policy is optimal. In practice it takes a handful of evaluations even for large MDPs.

## Why the tie rule matters
Without "keep the current action if it is still (nearly) best", two actions with equal Q can swap back and forth forever. Floating-point noise from the linear solve makes exact equality unreliable, hence the $10^{-10}$ tolerance. Test 3 has all-zero rewards: every action ties, so the initial all-zero policy is kept and the loop stops after one evaluation. In test 4, action 1 wins and action 2 (equal Q) never replaces it.

## Complexity
Each evaluation costs $O(S^3)$ for the solve plus $O(S^2 A)$ for Q. Value iteration is cheaper per sweep but needs many more sweeps when $\gamma$ is close to 1.

## Common bugs
- Plain `Q.argmax(1)` as the improvement. It can oscillate on ties, or stop early because a noisy argmax keeps changing.
- Returning the value of the previous policy instead of the final one.
- Not counting the final, confirming evaluation.

## Interview follow-ups
- What is modified policy iteration, and how does it interpolate between value iteration and policy iteration?
- Why does policy iteration often need fewer iterations than value iteration? (It's Newton's method on the Bellman equation.)
"""
