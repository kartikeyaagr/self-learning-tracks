import numpy as np

META = dict(title='DQN Huber loss against a target network', tags=['Reinforcement Learning', 'Loss Functions'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:07.03'],
            related=['q-learning-update', 'double-dqn-target', 'experience-replay-buffer'])

STATEMENT = r"""
DQN regresses the online network's Q value for the action taken toward a target built from a **frozen copy** of the network (the target network $\theta^-$):

$$y_i = r_i + \gamma (1 - d_i) \max_{a'} Q(s'_i, a'; \theta^-), \qquad \delta_i = Q(s_i, a_i; \theta) - y_i$$

The loss is the mean Huber loss of the TD errors, with threshold $\kappa$:

$$\ell_\kappa(\delta) = \begin{cases} \tfrac12 \delta^2 & |\delta| \le \kappa \\ \kappa\,(|\delta| - \tfrac12 \kappa) & \text{otherwise} \end{cases}, \qquad L = \frac1B \sum_i \ell_\kappa(\delta_i)$$

Write `dqn_loss(q_online, q_target_next, actions, rewards, dones, gamma=0.99, kappa=1.0)` and return `(loss, grad, targets)`.

## Inputs
- `q_online`: float `(B, A)`, the online network's outputs $Q(s_i, \cdot; \theta)$.
- `q_target_next`: float `(B, A)`, the target network's outputs $Q(s'_i, \cdot; \theta^-)$.
- `actions`: int `(B,)`; `rewards`: float `(B,)`; `dones`: `(B,)` of 0/1.

## Output
- `loss`: float, $L$ above.
- `grad`: float `(B, A)`, $\partial L / \partial$ `q_online`. The targets are constants (no gradient flows into the target network), so only the taken action's entry of each row is nonzero.
- `targets`: float `(B,)`, the $y_i$.
"""

STARTER = '''import numpy as np

def dqn_loss(q_online, q_target_next, actions, rewards, dones, gamma=0.99, kappa=1.0):
    """Return (mean Huber loss, dL/dq_online (B, A), targets (B,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def dqn_loss(q_online, q_target_next, actions, rewards, dones, gamma=0.99, kappa=1.0):
    q, qt = np.asarray(q_online, dtype=float), np.asarray(q_target_next, dtype=float)
    a = np.asarray(actions)
    B = len(a)
    rows = np.arange(B)
    y = rewards + gamma * (1.0 - np.asarray(dones, dtype=float)) * qt.max(axis=1)
    delta = q[rows, a] - y
    absd = np.abs(delta)
    huber = np.where(absd <= kappa, 0.5 * delta**2, kappa * (absd - 0.5 * kappa))
    grad = np.zeros_like(q)
    grad[rows, a] = np.clip(delta, -kappa, kappa) / B     # Huber derivative: delta inside, +-kappa outside
    return float(huber.mean()), grad, y
'''

HINTS = [
    'Pick each row\'s taken-action value with fancy indexing: `q_online[np.arange(B), actions]`.',
    'Targets use the target network only: `rewards + gamma * (1 - dones) * q_target_next.max(1)`. Treat them as constants.',
    'The Huber derivative is $\\delta$ clipped to $[-\\kappa, \\kappa]$. Divide by $B$ for the mean and write it only at `[rows, actions]`.',
]

_r = np.random.default_rng(151)
QO, QT = _r.normal(size=(6, 4)) * 2, _r.normal(size=(6, 4)) * 2
TESTS = [
    (np.array([[1.0, 2.0], [0.5, 0.0]]), np.array([[0.0, 1.0], [3.0, 2.0]]), np.array([1, 0]),
     np.array([1.0, 0.0]), np.array([0, 1]), 0.5),
    (np.array([[5.0, 0.0]]), np.array([[0.0, 0.0]]), np.array([0]), np.array([0.0]), np.array([0]), 0.9, 1.0),
    (np.array([[5.0, 0.0]]), np.array([[0.0, 0.0]]), np.array([0]), np.array([0.0]), np.array([0]), 0.9, 10.0),
    (np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 4.0, 2.0]]), np.array([2]), np.array([-1.0]), np.array([0]), 1.0, 2.0),
    (np.array([[2.0], [2.0]]), np.array([[100.0], [100.0]]), np.array([0, 0]), np.array([2.0, 2.0]), np.array([1, 1]), 0.99),
    (QO, QT, _r.integers(0, 4, 6), _r.normal(size=6), (_r.uniform(size=6) < 0.3).astype(int), 0.99, 1.0),
    (QO, QT, _r.integers(0, 4, 6), _r.normal(size=6), np.zeros(6, dtype=int), 0.9, 0.5),
]

EXPLANATION = r"""
## Why a target network
If the target used the same weights being trained, every gradient step would also move the target, and the regression chases itself. That feedback loop is a big source of DQN instability. Freezing a copy $\theta^-$ (refreshed every few thousand steps, or slowly with Polyak averaging $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$) makes each stretch of training an ordinary supervised regression.

## Why Huber
Squared error has gradient $\delta$, so a single transition with a huge TD error (common early on, or with rare large rewards) produces a huge update. Huber is quadratic near zero but linear beyond $\kappa$, capping the per-sample gradient at $\kappa$ (compare tests 2 and 3). It's the "clip the error term" trick from the original DQN paper written as a loss.

## Gradient shape
The network outputs all A values, but only the taken action's value appears in the loss. The gradient is a sparse `(B, A)` array with one nonzero entry per row. In a framework this falls out of `gather` on the outputs.

## Complexity
$O(BA)$ for the max, $O(B)$ for the rest.

## Common bugs
- Bootstrapping through terminal transitions (test 5 has huge target values that must be ignored).
- Taking the max over the online network (that's half of Double DQN) or over `q_online` at $s$ instead of $s'$.
- Using `sum` instead of `mean`, which scales the gradient with batch size.

## Interview follow-ups
- Why does DQN need experience replay as well as a target network?
- What is the deadly triad, and which of its three parts does each DQN trick address?
"""
