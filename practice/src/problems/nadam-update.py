import numpy as np

META = dict(title='Nadam: Adam with a Nesterov look-ahead', tags=['Optimization', 'Training'], diff='Medium', runtime='numpy',
            related=['adam-step', 'nesterov-momentum', 'adamw-decoupled-weight-decay', 'adadelta-update'])

STATEMENT = r"""
Nadam replaces Adam's momentum term with a Nesterov-style look-ahead. Implement the simple form (constant $\beta_1$). Write `nadam(A, b, x0, steps, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8)`.

With $m_0 = v_0 = 0$, at step $t = 1, 2, \dots$:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad \bar m_t = \frac{\beta_1 m_t}{1-\beta_1^{t+1}} + \frac{(1-\beta_1) g_t}{1-\beta_1^t}$$
$$x_t = x_{t-1} - \text{lr}\, \frac{\bar m_t}{\sqrt{\hat v_t} + \epsilon}$$

Return the list of iterates `[x_0, ..., x_steps]` as float arrays.
"""

STARTER = '''import numpy as np

def nadam(A, b, x0, steps, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """Return the list of iterates, starting with x0."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def nadam(A, b, x0, steps, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    m, v = np.zeros_like(x), np.zeros_like(x)
    out = [x.copy()]
    for t in range(1, steps + 1):
        g = A @ x - b
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        vh = v / (1 - b2 ** t)
        mb = b1 * m / (1 - b1 ** (t + 1)) + (1 - b1) * g / (1 - b1 ** t)
        x = x - lr * mb / (np.sqrt(vh) + eps)
        out.append(x.copy())
    return out
'''

HINTS = [
    'The moments are updated exactly as in Adam; only the numerator of the step changes.',
    'The look-ahead numerator mixes the next step\'s momentum (beta1 * m_t, corrected with 1 - beta1^(t+1)) with the current gradient (corrected with 1 - beta1^t).',
    'Keep t starting at 1 so the bias corrections never divide by zero.',
]

_I2 = np.eye(2)
_ill = np.diag([1.0, 100.0])
_sp = np.array([[3.0, 1.0], [1.0, 2.0]])
TESTS = [
    (np.diag([2.0, 20.0]), np.zeros(2), np.array([1.0, 1.0]), 5, 0.1),
    (2 * np.eye(1), np.array([6.0]), np.array([0.0]), 40, 0.2),
    (_ill, np.array([1.0, 1.0]), np.array([-1.0, 1.0]), 25, 0.01),
    (_sp, np.array([1.0, -1.0]), np.array([0.5, -0.5]), 10, 1e-3, 0.95, 0.99),
    (2 * _I2, np.array([6.0, 6.0]), np.array([5.0, -5.0]), 3, 1.0, 0.5, 0.9),
]

EXPLANATION = r"""
## The idea
Nesterov momentum evaluates the step as if momentum had already been applied, which gives a correction one step earlier when the direction changes. Dozat's Nadam folds that idea into Adam by using the momentum value of the **next** step in the update, plus a share of the current raw gradient.

## Compared with Adam
Adam's numerator is $\hat m_t = m_t/(1-\beta_1^t)$. Nadam's is $\beta_1 \hat m_{t+1}$-like plus $(1-\beta_1) \hat g_t$. On smooth problems Nadam often converges slightly faster; on large models the difference is usually small compared with tuning the learning rate and schedule.

## Variants
The original paper also uses a schedule for $\beta_1$; deep-learning libraries implement versions with momentum decay. The constant-$\beta_1$ form here is the core of the method.

## Interview follow-ups
- Can you combine Nadam with decoupled weight decay? (Yes; apply the decay separately, as AdamW does.)
"""
