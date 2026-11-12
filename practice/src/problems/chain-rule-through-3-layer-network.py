import numpy as np

META = dict(title='Chain rule through a three-layer scalar network', tags=['Neural Networks', 'Calculus'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:01.07', 'air-fanout-v1:01.08'],
            related=['derivatives-of-common-activation-functions', 'backpropagation-through-computation-graph',
                     'finite-difference-gradient-checking'])

STATEMENT = r"""
A network with one neuron per layer is the smallest place to see backprop as nothing more than the chain rule:

$$h_1 = \tanh(w_1 x + b_1), \qquad h_2 = \tanh(w_2 h_1 + b_2), \qquad \hat y = w_3 h_2 + b_3$$

Over a batch of $N$ samples the loss is the halved mean squared error

$$L = \frac{1}{N}\sum_{n=1}^{N} \tfrac12\,(\hat y_n - t_n)^2$$

Write `three_layer_grads(params, x, t)` that returns `(loss, grad)`.

## Shapes
- `params`: float array `[w1, b1, w2, b2, w3, b3]`, shape `(6,)`.
- `x`, `t`: float arrays of shape `(N,)`, $N \ge 1$.
- Return `loss` as a Python `float` and `grad` as a float array of shape `(6,)` in the **same order** as `params`: $\left[\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial b_1}, \dots, \frac{\partial L}{\partial b_3}\right]$.

## Requirements
- Vectorise over the batch (no loop over samples).
- Reuse the forward activations in the backward pass.
"""

STARTER = '''import numpy as np

def three_layer_grads(params, x, t):
    """params = [w1, b1, w2, b2, w3, b3]. Return (loss, dL/dparams in the same order)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def three_layer_grads(params, x, t):
    w1, b1, w2, b2, w3, b3 = np.asarray(params, dtype=float)
    x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
    # forward, keeping every intermediate
    h1 = np.tanh(w1 * x + b1)
    h2 = np.tanh(w2 * h1 + b2)
    r = w3 * h2 + b3 - t
    loss = 0.5 * np.mean(r ** 2)
    # backward: each line multiplies by one local derivative
    dy = r / len(x)                    # dL/dy_hat per sample
    dz2 = dy * w3 * (1 - h2 ** 2)      # through y = w3 h2 + b3, then tanh' = 1 - tanh^2
    dz1 = dz2 * w2 * (1 - h1 ** 2)     # through z2 = w2 h1 + b2, then tanh
    grad = np.array([np.sum(dz1 * x), dz1.sum(),
                     np.sum(dz2 * h1), dz2.sum(),
                     np.sum(dy * h2), dy.sum()])
    return float(loss), grad
'''

HINTS = [
    'Write the pre-activations $z_1 = w_1x + b_1$ and $z_2 = w_2h_1 + b_2$ explicitly. Every gradient is a product of local derivatives along the path from that parameter to $L$.',
    'Start from $\\partial L / \\partial \\hat y_n = (\\hat y_n - t_n)/N$. Then $\\partial L/\\partial z_2 = \\partial L/\\partial \\hat y \\cdot w_3 (1 - h_2^2)$, and $\\partial L/\\partial z_1 = \\partial L/\\partial z_2 \\cdot w_2 (1 - h_1^2)$.',
    'A weight\'s gradient is (gradient at its output) times (its input), summed over the batch: $\\partial L/\\partial w_1 = \\sum_n \\delta_{1,n} x_n$. A bias\'s gradient is just $\\sum_n \\delta_n$.',
]

_r = np.random.default_rng(8)
TESTS = [
    (np.array([0.5, 0.0, -1.0, 0.0, 2.0, 0.0]), np.array([1.0]), np.array([0.0])),
    (np.array([1.0, 0.1, 1.0, -0.1, 1.0, 0.5]), np.array([-1.0, 0.0, 1.0]), np.array([0.0, 0.5, 1.0])),
    (np.zeros(6), np.array([1.0, 2.0]), np.array([1.0, -1.0])),
    (np.array([5.0, 0.0, 5.0, 0.0, 1.0, 0.0]), np.array([10.0, -10.0]), np.array([1.0, -1.0])),
    (np.array([0.3, -0.2, 0.8, 0.1, -1.5, 0.4]), np.array([2.0]), np.array([2.0])),
    (_r.normal(size=6), _r.normal(size=16), _r.normal(size=16)),
    (_r.normal(size=6) * 2, _r.normal(size=40) * 3, _r.normal(size=40)),
]

EXPLANATION = r"""
## The chain, written out
For $w_1$ the path is $w_1 \to z_1 \to h_1 \to z_2 \to h_2 \to \hat y \to L$:

$$\frac{\partial L}{\partial w_1} = \sum_n \underbrace{\frac{\hat y_n - t_n}{N}}_{\partial L/\partial \hat y}\cdot w_3 \cdot (1-h_{2,n}^2) \cdot w_2 \cdot (1 - h_{1,n}^2) \cdot x_n$$

Backprop computes this right to left and **stores** each partial product ($\delta_2$, $\delta_1$), so the shared prefix is computed once, not once per parameter.

## Saturation
With $w_1 = w_2 = 5$ and $|x| = 10$, both tanh units sit at $\pm 1$, $1 - h^2 \approx 0$, and the gradients for the first two layers vanish even though the loss isn't zero. Only $w_3$ and $b_3$ still learn.

## Complexity
One forward and one backward pass: $O(N)$ time, $O(N)$ memory for the cached activations.

## Common bugs
- Using $1 - z^2$ instead of $1 - h^2$ (the derivative of tanh is in terms of its **output**).
- Forgetting the $1/N$, which scales the whole gradient by the batch size.
- Dropping the $w_3$ or $w_2$ factor when passing the gradient back through a layer.

## Interview follow-ups
- Check your gradient with central finite differences. What step size $h$ and what relative-error threshold would you use?
- Replace tanh with ReLU. What happens to the first-layer gradient when $w_1 x + b_1 < 0$ for every sample?
"""
