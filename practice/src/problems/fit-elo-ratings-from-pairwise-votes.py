import numpy as np

META = dict(title='Fit arena-style Elo ratings from pairwise votes', tags=['Evaluation Metrics', 'Optimization'],
            diff='Hard', runtime='numpy', lessons=['air-fanout-v1:10.05'],
            related=['bradley-terry-reward-model-loss', 'newton-s-method', 'remove-response-length-bias-from-judge-scores'])

STATEMENT = r"""
Chatbot leaderboards collect votes of the form "model $a$'s answer beat model $b$'s" and turn them into one rating per model. Online Elo updates depend on the order of the votes; the order-free alternative is to fit a **Bradley-Terry** model by maximum likelihood:

$$P(a \text{ beats } b) = \sigma(\theta_a - \theta_b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

Write `fit_elo(n_models, a, b, outcome, lam)` and return a dict.

## Inputs
- `a`, `b`: int `(N,)` model indices with `a[i] != b[i]`, in `0 .. n_models-1`.
- `outcome`: float64 `(N,)`: `1.0` if `a` won, `0.0` if `b` won, `0.5` for a tie (half a win each).
- `lam`: float $> 0$, an L2 penalty.

## Objective
Minimise over $\theta \in \mathbb{R}^M$

$$J(\theta) = -\sum_i \Big[y_i \log \sigma(\theta_{a_i} - \theta_{b_i}) + (1 - y_i) \log \sigma(\theta_{b_i} - \theta_{a_i})\Big] + \lambda \lVert\theta\rVert^2$$

$J$ is strictly convex, so the minimiser is unique, and the penalty keeps it finite even when a model never loses. At the optimum $\sum_m \theta_m = 0$. Solve it to high precision: Newton's method from $\theta = 0$ converges in a handful of iterations (stop when the largest step is below $10^{-12}$, at most 100 iterations). Any method that reaches the same minimiser passes.

## Output
| key | value |
|---|---|
| `"theta"` | float64 `(M,)`, the minimiser |
| `"elo"` | float64 `(M,)`, $1000 + \frac{400}{\ln 10}\,\theta$ (so a 400-point gap means 10:1 odds) |
| `"win_prob"` | float64 `(M, M)`, $\sigma(\theta_i - \theta_j)$; the diagonal is 0.5 |
| `"order"` | list of ints, models from highest to lowest rating |

The tests have no ties in the fitted ratings.
"""

STARTER = '''import numpy as np

def fit_elo(n_models, a, b, outcome, lam):
    """Return a dict: theta (M,), elo (M,), win_prob (M, M), order (list)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))      # stable for large |z|

def fit_elo(n_models, a, b, outcome, lam):
    a, b, y = np.asarray(a), np.asarray(b), np.asarray(outcome, dtype=float)
    M = n_models
    # Signed incidence matrix: theta_a - theta_b = X @ theta for every vote.
    X = np.zeros((len(a), M))
    X[np.arange(len(a)), a] = 1.0
    X[np.arange(len(a)), b] = -1.0

    theta = np.zeros(M)
    for _ in range(100):
        p = _sigmoid(X @ theta)
        grad = X.T @ (p - y) + 2 * lam * theta
        hess = (X.T * (p * (1 - p))) @ X + 2 * lam * np.eye(M)
        step = np.linalg.solve(hess, grad)
        theta -= step
        if np.max(np.abs(step)) < 1e-12:
            break

    elo = 1000.0 + 400.0 / np.log(10) * theta
    win_prob = _sigmoid(theta[:, None] - theta[None, :])
    order = [int(i) for i in np.argsort(-theta, kind="stable")]
    return {"theta": theta, "elo": elo, "win_prob": win_prob, "order": order}
'''

HINTS = [
    'Write each vote as a row of a matrix $X$ with $+1$ in column $a_i$ and $-1$ in column $b_i$. Then the logits are $X\\theta$ and this is logistic regression without an intercept (ties are just soft labels of 0.5).',
    'Gradient: $X^\\top(\\sigma(X\\theta) - y) + 2\\lambda\\theta$. Hessian: $X^\\top \\operatorname{diag}(p(1-p)) X + 2\\lambda I$, which is positive definite thanks to the penalty.',
    'Iterate $\\theta \\leftarrow \\theta - H^{-1}g$ with `np.linalg.solve` until the step is tiny, then convert with $400/\\ln 10$ and sort by $-\\theta$.',
]


def _arena(seed, strengths, n_votes, tie_rate):
    r = np.random.default_rng(seed)
    M = len(strengths)
    a = r.integers(0, M, size=n_votes)
    b = (a + r.integers(1, M, size=n_votes)) % M          # never equal to a
    p = 1.0 / (1.0 + np.exp(-(np.array(strengths)[a] - np.array(strengths)[b])))
    u = r.random(n_votes)
    y = np.where(u < tie_rate, 0.5, (r.random(n_votes) < p).astype(float))
    return a, b, y


TESTS = [
    (2, np.array([0, 0, 0, 1]), np.array([1, 1, 1, 0]), np.array([1.0, 1.0, 0.0, 0.0]), 0.1),
    (3, np.array([0, 1, 2]), np.array([1, 2, 0]), np.array([1.0, 1.0, 0.5]), 0.5),
    (2, np.array([0]), np.array([1]), np.array([1.0]), 1e-3),
    (4, *_arena(0, [1.0, 0.3, -0.2, -1.1], 200, 0.1), 0.05),
    (6, *_arena(1, [2.0, 1.2, 0.4, 0.0, -0.8, -2.5], 600, 0.15), 0.01),
    (5, np.array([0, 1, 2, 3, 0, 2, 1]), np.array([1, 2, 3, 4, 2, 4, 3]), np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.5]), 0.2),
    (8, *_arena(2, [0.5, -0.5, 1.5, 0.1, -1.0, 0.9, -0.3, 0.2], 1500, 0.05), 0.02),
]

EXPLANATION = r"""
## Why fit instead of updating online
Classic Elo updates each rating after every game with a fixed step size $K$. The final ratings then depend on the vote order and on $K$, and recent votes count more. For static models (an LLM doesn't get better between votes), every vote should count the same; the maximum-likelihood fit of the same logistic model does exactly that and is order-independent. The $400/\ln 10$ factor only converts the natural-log odds scale to the familiar chess convention (400 points = 10:1).

## Why the penalty
If a model wins every one of its votes, the unpenalised likelihood keeps increasing as its $\theta \to \infty$ (test 3 has a single vote). The $\lambda\lVert\theta\rVert^2$ term acts like a Gaussian prior centred on "everyone is equal", keeping ratings finite and shrinking poorly-sampled models toward the middle. It also pins the otherwise-free additive constant: the gradient of the likelihood part always sums to zero across models, so the optimality condition forces $\sum_m\theta_m = 0$.

## Convergence and cost
The objective is smooth and strictly convex, so Newton's method converges quadratically: usually under 10 iterations. Each iteration costs $O(NM + M^3)$, tiny for leaderboard sizes. With thousands of models you would use the MM algorithm or L-BFGS instead of dense Hessians.

## Common bugs
- Treating ties as missing data, or as a full win for one side.
- Computing $\log\sigma$ or $\sigma$ with `np.exp(-z)` without care; `tanh` or `logaddexp` forms are stable.
- Forgetting the factor 2 from $\lambda\lVert\theta\rVert^2$ in the gradient and Hessian, which converges to a different (wrong) optimum.
- Running gradient descent with a fixed step for a few hundred iterations: close, but not to $10^{-6}$.

## Interview follow-ups
- How would you put confidence intervals on the ratings? (Bootstrap the votes, or invert the Hessian at the optimum.)
- A new model has 50 votes, an old one 50,000. How does that affect their ratings and their uncertainty, and how would you display it?
"""
