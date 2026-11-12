import numpy as np

META = dict(title='One Adam optimizer step', tags=['Optimization', 'Training'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:05.06'], related=['cross-entropy-logits'])

STATEMENT = r"""
Adam keeps running averages of the gradient ($m$) and of its square ($v$), corrects their start-up bias, and scales each parameter's step by its own gradient history. At step $t$ (counting from 1):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g^2$$
$$\hat m = \frac{m_t}{1-\beta_1^t}, \qquad \hat v = \frac{v_t}{1-\beta_2^t}, \qquad \theta_t = \theta_{t-1} - \eta\,\frac{\hat m}{\sqrt{\hat v} + \epsilon}$$

Write `adam_step(theta, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8)` and return the tuple `(theta_new, m_new, v_new)`.

## Requirements
- All arrays share one shape (any shape). `t` is the 1-based step number **of this update**.
- Return new arrays and don't modify the inputs in place.
- Add $\epsilon$ **after** the square root, as written above.
"""

STARTER = '''import numpy as np

def adam_step(theta, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """Return (theta_new, m_new, v_new) after one Adam update at step t (1-based)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def adam_step(theta, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)          # m and v start at 0, so early averages are biased toward 0
    v_hat = v / (1 - b2 ** t)
    return theta - lr * m_hat / (np.sqrt(v_hat) + eps), m, v
'''

HINTS = [
    'Update `m` and `v` first. The bias correction uses the **updated** values and the step number `t`.',
    'At t = 1 with m = v = 0, the corrected values are m̂ = g and v̂ = g², so the very first step is about `lr * sign(g)`.',
    'Write `m = b1 * m + ...` rather than `m *= b1`, so the caller\'s arrays aren\'t mutated.',
]

_r = np.random.default_rng(2)
z3 = np.zeros(3)
TESTS = [
    (np.array([1.0, -2.0, 3.0]), np.array([0.1, -0.2, 0.3]), z3, z3, 1),
    (np.array([0.5]), np.array([0.0]), np.array([0.0]), np.array([0.0]), 1),
    (np.array([1.0, 1.0]), np.array([1.0, -1.0]), np.array([0.5, -0.5]), np.array([0.2, 0.2]), 10, 0.01),
    (_r.normal(size=(2, 3)), _r.normal(size=(2, 3)), _r.normal(size=(2, 3)) * 0.1, np.abs(_r.normal(size=(2, 3))) * 0.1, 100),
    (np.array([2.0]), np.array([4.0]), np.array([1.0]), np.array([9.0]), 3, 0.1, 0.8, 0.9, 1e-6),
    (np.array([0.0, 0.0]), np.array([1e-12, -1e-12]), np.zeros(2), np.zeros(2), 1),
]

EXPLANATION = r"""
## What each piece does
- **$m$ (first moment)**: momentum. It smooths noisy gradients so steps follow the consistent direction.
- **$v$ (second moment)**: per-parameter scale. Parameters with large or erratic gradients get smaller steps, and rare-gradient parameters get larger ones.
- **Bias correction**: both averages start at 0, so for small $t$ they underestimate. Dividing by $1-\beta^t$ fixes this. At $t=1$ it exactly recovers $g$ and $g^2$.

## The first step is about lr
With $\hat m = g$ and $\hat v = g^2$, the update is $\eta\, g/(|g|+\epsilon) \approx \eta\,\mathrm{sign}(g)$. Adam's step size is set by the learning rate, not by the gradient's magnitude. That's why it is robust to gradient scale.

## Common bugs
- Bias correction with the old $m, v$, or with $t$ starting at 0 (which divides by zero).
- $\epsilon$ inside the square root changes the effective scale for tiny gradients. The last test makes this visible.
- In-place updates that silently change the caller's optimizer state.

## Interview follow-ups
- How does AdamW differ? (Weight decay is applied to $\theta$ directly instead of being added to $g$, which decouples it from the adaptive scaling.)
- Memory cost: Adam stores 2 extra floats per parameter. For a 7B model in fp32 that is 56 GB of optimizer state.
"""
