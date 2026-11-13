import numpy as np

META = dict(title='LinUCB contextual bandit (disjoint arms)', tags=['Reinforcement Learning', 'Linear Algebra'],
            diff='Hard', runtime='numpy', lessons=[],
            related=['ucb1-for-multi-armed-bandits', 'thompson-sampling-beta-bernoulli', 'ridge-regression'])

STATEMENT = r"""
In a **contextual** bandit, each round shows a feature vector for every arm, and the expected reward of arm $k$ is assumed linear in its features: $\mathbb{E}[r \mid x] = \theta_k^\top x$. Disjoint LinUCB keeps a separate ridge regression per arm and adds an optimism bonus from the regression's uncertainty:

$$A_k = \lambda I + \sum x x^\top, \quad b_k = \sum r\, x, \quad \hat\theta_k = A_k^{-1} b_k, \quad p_k = \hat\theta_k^\top x + \alpha \sqrt{x^\top A_k^{-1} x}$$

where the sums run over the rounds in which arm $k$ was pulled.

Write `linucb(contexts, rewards, alpha=1.0, lam=1.0)` that runs the bandit and returns `(arms, total_reward, theta)`.

## Inputs
- `contexts`: float array `(T, K, d)`. `contexts[t, k]` is arm `k`'s feature vector at round `t`.
- `rewards`: float array `(T, K)`. `rewards[t, k]` is what arm `k` pays if pulled at round `t`. Only the pulled arm's reward is observed.
- `alpha >= 0` scales exploration, `lam > 0` is the ridge strength.

## Algorithm
1. Start every arm with $A_k = \lambda I_d$ and $b_k = 0$.
2. For each round `t`: compute $p_k$ for every arm with that arm's context, using the arm's current $A_k, b_k$. Pull the arm with the largest $p_k$ (lowest index on ties). Observe $r$ = `rewards[t, k]` and update only that arm: $A_k \mathrel{+}= x x^\top$, $b_k \mathrel{+}= r\, x$.

## Output
- `arms`: int array `(T,)`.
- `total_reward`: float, the sum of observed rewards.
- `theta`: float array `(K, d)`, every arm's $\hat\theta_k = A_k^{-1} b_k$ after the last round.
"""

STARTER = '''import numpy as np

def linucb(contexts, rewards, alpha=1.0, lam=1.0):
    """contexts: (T, K, d), rewards: (T, K). Return (arms (T,), total reward, theta (K, d))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def linucb(contexts, rewards, alpha=1.0, lam=1.0):
    X, R = np.asarray(contexts, dtype=float), np.asarray(rewards, dtype=float)
    T, K, d = X.shape
    A = np.tile(lam * np.eye(d), (K, 1, 1))     # (K, d, d) per-arm design matrices
    b = np.zeros((K, d))
    arms = np.zeros(T, dtype=int)
    total = 0.0
    for t in range(T):
        x = X[t]                                                 # (K, d)
        theta = np.linalg.solve(A, b[..., None])[..., 0]         # (K, d) ridge estimate per arm
        Ainv_x = np.linalg.solve(A, x[..., None])[..., 0]        # (K, d) A_k^{-1} x_k
        p = (theta * x).sum(1) + alpha * np.sqrt((x * Ainv_x).sum(1))
        k = int(p.argmax())
        r = R[t, k]
        A[k] += np.outer(x[k], x[k])
        b[k] += r * x[k]
        arms[t] = k
        total += r
    return arms, total, np.linalg.solve(A, b[..., None])[..., 0]
'''

HINTS = [
    'Keep the state as a stack `A` of shape `(K, d, d)` and `b` of shape `(K, d)`. `np.linalg.solve` works on stacks, so all arms can be scored at once.',
    'For each arm you need $A_k^{-1} b_k$ (the estimate) and $A_k^{-1} x_k$ (for the bonus $\\sqrt{x_k^\\top A_k^{-1} x_k}$). Solve rather than invert.',
    'After picking `k = argmax(p)`, update only that arm with `np.outer(x[k], x[k])` and `r * x[k]`. Recompute theta from the final A and b at the end.',
]

_r = np.random.default_rng(141)
def _env(T, K, d, noise):
    X = _r.normal(size=(T, K, d)) / np.sqrt(d)
    true = _r.normal(size=(K, d))
    return X, np.einsum('tkd,kd->tk', X, true) + noise * _r.normal(size=(T, K))
X1, R1 = _env(30, 3, 4, 0.1)
X2, R2 = _env(40, 4, 3, 0.3)
XS = np.repeat(_r.normal(size=(12, 1, 2)), 3, axis=1)      # all arms see the same context each round
RS = XS @ np.array([0.5, -1.0]) + np.array([0.0, 0.2, 0.4])
TESTS = [
    (np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]),
     np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]), 0.5, 1.0),
    (np.array([[[1.0], [1.0]], [[1.0], [1.0]], [[1.0], [1.0]], [[1.0], [1.0]]]),
     np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]), 1.0, 1.0),
    (np.array([[[1.0], [1.0]], [[1.0], [1.0]], [[1.0], [1.0]], [[1.0], [1.0]]]),
     np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]), 0.0, 1.0),
    (XS, RS, 1.0, 1.0),
    (X1, R1, 1.0, 1.0),
    (X1, R1, 0.1, 0.5),
    (X2, R2, 2.0, 1.0),
]

EXPLANATION = r"""
## Why the bonus is the right width
$\hat\theta_k$ is a ridge-regression estimate, and its uncertainty in the direction $x$ is proportional to $\sqrt{x^\top A_k^{-1} x}$. Directions the arm has already seen many times have a large component in $A_k$, so the bonus there is small. New directions keep a bonus near $\lVert x \rVert / \sqrt{\lambda}$. LinUCB plays the arm whose reward could plausibly be highest, the same optimism as UCB1 but in feature space, so what it learns about one context transfers to similar ones.

## What the tests show
- Test 2: arm 0 pays nothing, so after one pull its estimate drops, and arm 1 (still at the prior bonus) takes over and keeps winning.
- Test 3 is the same with $\alpha = 0$ (pure greedy). All scores start at exactly 0, arm 0 wins the tie, its estimate stays at 0, and the other arm, also at 0, never beats it. Greedy gets stuck; the bonus is what unsticks it.
- Test 4 gives all arms the same context each round, so untried arms tie exactly and the lowest index goes first.

## Complexity
Per round $O(K d^3)$ with direct solves. Production code keeps $A_k^{-1}$ and updates it in $O(d^2)$ with the Sherman-Morrison formula: $(A + xx^\top)^{-1} = A^{-1} - \frac{A^{-1}xx^\top A^{-1}}{1 + x^\top A^{-1} x}$.

## Common bugs
- Updating every arm with its counterfactual reward.
- Using one shared $A$ for all arms (that's the "hybrid" model, a different algorithm).
- Computing $x^\top A x$ instead of $x^\top A^{-1} x$ in the bonus.

## Interview follow-ups
- Derive the Sherman-Morrison update and use it to make each round $O(K d^2)$.
- How would you run Thompson sampling in the same linear model?
"""
