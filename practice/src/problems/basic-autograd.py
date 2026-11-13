import numpy as np

META = dict(title='Logistic-regression gradients with autograd', tags=['Neural Networks', 'Loss Functions'], diff='Easy', runtime='torch',
            lessons=['air-fanout-v1:01.08', 'ieng:00.05'],
            related=['computational-graph-and-autograd', 'manual-weight-update', 'log-loss-binary-cross-entropy'])

STATEMENT = r"""
Let PyTorch's autograd do the calculus for a logistic-regression loss, and read gradients for both the parameters and an intermediate tensor.

Write `logistic_grads(x, y, w, b)`:

- `x`: float64 tensor `(N, D)`; `y`: float64 tensor `(N,)` of 0/1 labels.
- `w`: float64 tensor `(D,)`; `b`: Python float.
- Compute the logits $z = x w + b$ (shape `(N,)`) and the mean binary cross-entropy

$$L = -\frac{1}{N}\sum_{n} \Big[y_n \log \sigma(z_n) + (1-y_n)\log\big(1-\sigma(z_n)\big)\Big].$$

Return a tuple `(loss, grad_w, grad_b, grad_z)`:

- `loss`: 0-d tensor (or float).
- `grad_w` `(D,)`, `grad_b` 0-d, `grad_z` `(N,)`: the gradients $\partial L/\partial w$, $\partial L/\partial b$, $\partial L/\partial z$.

Get all of them from a single `backward()` call. Don't modify the input `w`. Large $|z|$ must not produce `inf` or `nan`, so compute the loss from the logits, not from $\sigma(z)$.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def logistic_grads(x, y, w, b):
    """Return (loss, dL/dw, dL/db, dL/dz)."""
    w = w.clone().requires_grad_()
    z = x @ w + b
    loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(z), y)
    loss.backward()
    return loss, w.grad, torch.tensor(0.0), torch.zeros_like(z)
'''

SOLUTION = '''import torch
import torch.nn.functional as F

def logistic_grads(x, y, w, b):
    w = w.detach().clone().requires_grad_()            # a fresh leaf; the caller's tensor is untouched
    b = torch.tensor(b, dtype=x.dtype, requires_grad=True)
    z = x @ w + b
    z.retain_grad()                                    # z is not a leaf: keep its .grad after backward
    loss = F.binary_cross_entropy_with_logits(z, y)    # log-sigmoid form, stable for large |z|
    loss.backward()
    return loss.detach(), w.grad, b.grad, z.grad
'''

HINTS = [
    'Make `w` and `b` leaf tensors with `requires_grad=True` (clone `w` first so you don\'t touch the caller\'s tensor). After `loss.backward()`, read `.grad`.',
    'Only leaves keep `.grad` by default. Call `z.retain_grad()` before `backward()` to keep the gradient of the intermediate logits.',
    '`F.binary_cross_entropy_with_logits(z, y)` uses the log-sum-exp form and stays finite for $|z| = 1000$. Feeding `sigmoid(z)` into `binary_cross_entropy` saturates to 0 or 1 and its log blows up.',
]

_r = np.random.default_rng(11)
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 0.0]), np.array([0.5, -0.5]), 0.0),
    (np.array([[2.0], [-1.0], [0.5]]), np.array([1.0, 1.0, 0.0]), np.array([1.0]), 0.25),
    (_r.normal(size=(8, 3)), (_r.uniform(size=8) > 0.5).astype(float), _r.normal(size=3), -0.3),
    (np.array([[1.0], [1.0]]), np.array([0.0, 1.0]), np.array([0.0]), 0.0),
    (np.array([[100.0], [-100.0]]), np.array([0.0, 1.0]), np.array([10.0]), 0.0),
    (_r.normal(size=(1, 5)), np.array([1.0]), _r.normal(size=5), 2.0),
    (_r.normal(size=(30, 4)), (_r.uniform(size=30) > 0.3).astype(float), _r.normal(size=4) * 3, 1.0),
]

EXPLANATION = r"""
## Why it works
Every operation on a tensor with `requires_grad=True` records a node in a graph. `backward()` walks that graph in reverse, applying the chain rule, and adds each leaf's gradient into its `.grad`. For this loss the chain rule gives a clean closed form you can use to check the output:

$$\frac{\partial L}{\partial z_n} = \frac{\sigma(z_n) - y_n}{N}, \qquad \frac{\partial L}{\partial w} = x^\top \frac{\partial L}{\partial z}, \qquad \frac{\partial L}{\partial b} = \sum_n \frac{\partial L}{\partial z_n}.$$

## Leaves, non-leaves and retain_grad
`w` and `b` are leaves (created by you), so they keep gradients. `z` is produced by an op, so PyTorch frees its gradient as soon as it has been propagated. `retain_grad()` asks it to keep it, which is how you inspect activations' gradients when debugging.

## Stability
$\log \sigma(z) = -\log(1 + e^{-z})$ is computed by `binary_cross_entropy_with_logits` with the max trick, so the 100-logit test stays finite. Going through `sigmoid` first rounds $\sigma(1000)$ to exactly 1.0 in floating point and then $\log(1 - 1) = -\infty$. (PyTorch clamps the log at $-100$ to hide this, which silently gives the wrong loss and gradient.)

## Complexity
Forward and backward are both $O(ND)$; backward costs about as much as forward, the usual reverse-mode ratio.

## Interview follow-ups
- Why does calling `backward()` twice on the same graph fail, and what does `retain_graph=True` change? What happens to `.grad` on the second call?
- How would you get the Hessian-vector product $H v$ with autograd? (`torch.autograd.grad` with `create_graph=True`, then differentiate $g \cdot v$ again.)
"""
