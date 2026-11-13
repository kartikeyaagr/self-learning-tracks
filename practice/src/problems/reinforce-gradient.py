import numpy as np

META = dict(title='REINFORCE gradient for a linear softmax policy', tags=['Reinforcement Learning'], diff='Easy',
            runtime='numpy', lessons=['air-fanout-v1:07.02'],
            related=['reinforce-with-baseline', 'discounted-returns', 'softmax-cross-entropy-gradient'])

STATEMENT = r"""
The policy gradient theorem turns "make high-return actions more likely" into a gradient you can estimate from samples:

$$\nabla_\theta J \approx \frac1T \sum_{t} G_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

Take a linear softmax policy over features $x_t \in \mathbb{R}^d$ with parameters $\theta \in \mathbb{R}^{d \times A}$:

$$\pi_\theta(\cdot \mid s_t) = \mathrm{softmax}(x_t^\top \theta)$$

Write `reinforce_grad(theta, features, actions, returns)` and return `(surrogate, grad)`.

## Inputs
- `theta`: float `(d, A)`.
- `features`: float `(T, d)`, one row $x_t$ per step.
- `actions`: int `(T,)`; `returns`: float `(T,)`, the $G_t$ (already computed, treat as constants).

## Output
- `surrogate`: float, $\frac1T \sum_t G_t \log \pi_\theta(a_t \mid s_t)$. Its gradient is the estimate above, which is why frameworks minimise its negative.
- `grad`: float `(d, A)`, $\nabla_\theta$ of the surrogate (the **ascent** direction).

Compute log-probabilities stably: logits can be large.
"""

STARTER = '''import numpy as np

def reinforce_grad(theta, features, actions, returns):
    """theta: (d, A), features: (T, d). Return (surrogate objective, gradient (d, A))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def reinforce_grad(theta, features, actions, returns):
    X, th = np.asarray(features, dtype=float), np.asarray(theta, dtype=float)
    G = np.asarray(returns, dtype=float)
    T = len(G)
    z = X @ th
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))      # log-softmax, (T, A)
    rows = np.arange(T)
    surrogate = float((G * logp[rows, actions]).mean())
    # d log pi(a|x) / d theta = x (onehot(a) - pi)^T
    coef = -np.exp(logp)
    coef[rows, actions] += 1.0
    grad = X.T @ (coef * G[:, None]) / T
    return surrogate, grad
'''

HINTS = [
    'Logits are `features @ theta`, shape `(T, A)`. Use a stable log-softmax: subtract the row max, then subtract the log of the summed exponentials.',
    'For a softmax over linear logits, $\\nabla_\\theta \\log \\pi(a|x) = x\\,(e_a - \\pi)^\\top$, where $e_a$ is the one-hot of the action.',
    'Stack the per-step terms: build `(onehot - probs) * G[:, None]` of shape `(T, A)` and the whole gradient is `features.T @ that / T`.',
]

_r = np.random.default_rng(201)
TH, XR = _r.normal(size=(3, 4)), _r.normal(size=(10, 3))
TESTS = [
    (np.zeros((1, 2)), np.array([[1.0], [1.0]]), np.array([0, 1]), np.array([1.0, 3.0])),
    (np.array([[1.0, 0.0]]), np.array([[2.0]]), np.array([1]), np.array([2.0])),
    (np.zeros((2, 3)), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([2, 2]), np.array([0.0, 0.0])),
    (np.array([[500.0, -500.0]]), np.array([[1.0], [1.0]]), np.array([0, 1]), np.array([1.0, 1.0])),
    (TH, XR, _r.integers(0, 4, 10), _r.normal(size=10)),
    (TH * 3, XR, _r.integers(0, 4, 10), np.abs(_r.normal(size=10)) * 5),
]

EXPLANATION = r"""
## Why this is the gradient
The log-derivative trick: $\nabla_\theta \mathbb{E}_{a \sim \pi_\theta}[G] = \mathbb{E}[G\, \nabla_\theta \log \pi_\theta(a)]$. You never differentiate through the environment; you only need the gradient of the log-probability of actions you actually took, weighted by how well they turned out. For softmax over linear logits that gradient is $x(e_a - \pi)^\top$: push up the taken action's logit, push down the others in proportion to their probability.

## Reading the tests
- Test 1: a zero policy, two steps with the same feature. Action 1 had the larger return, so the gradient raises its logit more than action 0's.
- Test 3: zero returns, zero gradient. REINFORCE learns nothing from an episode with no reward signal.
- Test 4: the policy is nearly deterministic toward action 0, and the step that took action 1 has $\log\pi \approx -1000$. A naive `log(softmax(...))` returns `-inf` here.

## Complexity
$O(T d A)$, dominated by the two matmuls.

## Common bugs
- Returning the gradient of the *loss* (negated) when the ascent direction was asked for, or the other way around.
- Forgetting the $-\pi$ part of the gradient, which leaves only a push on the taken action.
- Summing instead of averaging over steps.

## Interview follow-ups
- Why does REINFORCE have high variance, and what are three ways to reduce it? (Baselines, reward-to-go, critics/GAE.)
- Why can't you just backprop through the sampled action? When can you (reparameterisation)?
"""
