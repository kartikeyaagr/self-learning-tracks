import numpy as np

META = dict(title='Softmax as a custom autograd Function', tags=['Activation Functions', 'Neural Networks'], diff='Easy', runtime='torch',
            lessons=['ieng:00.03', 'ieng:13.05'],
            related=['softmax-rows', 'softmax-jacobian', 'basic-autograd'])

STATEMENT = r"""
Write softmax yourself, forward **and** backward, as a `torch.autograd.Function`, the way you would before fusing it into a kernel.

## Forward
Along dimension `dim`:
$$y_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \qquad m = \max_j x_j.$$
Subtracting the max keeps it finite for large logits. Don't call `torch.softmax`, `F.softmax` or `log_softmax`.

## Backward
Given the upstream gradient $g = \partial\mathcal L/\partial y$, the input gradient needs only the **output** $y$:
$$\frac{\partial\mathcal L}{\partial x_i} = y_i\Big(g_i - \sum_j g_j\, y_j\Big)$$
with the sum taken along the same `dim`. Save `y` (not `x`) for backward.

## What's graded
Write a class `Softmax(torch.autograd.Function)` with static `forward(ctx, x, dim)` and `backward(ctx, grad_y)`, and the function `softmax_and_grad(x, dim, g)`:

- `x`: float64 tensor of any shape with at least one dim; `dim`: int (may be negative); `g`: float64 tensor shaped like `x`.
- Return `(y, grad_x)`: `y = Softmax.apply(x, dim)` and the gradient of `(y * g).sum()` with respect to `x`.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        e = x.exp()
        y = e / e.sum(dim=dim, keepdim=True)
        ctx.dim = dim
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (y,) = ctx.saved_tensors
        return grad_y * y, None          # TODO: this is only the diagonal of the Jacobian

def softmax_and_grad(x, dim, g):
    """Return (y, dL/dx) for L = (Softmax.apply(x, dim) * g).sum()."""
    x = x.detach().clone().requires_grad_()
    y = Softmax.apply(x, dim)
    (y * g).sum().backward()
    return y.detach(), x.grad
'''

SOLUTION = '''import torch

class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        e = (x - x.amax(dim=dim, keepdim=True)).exp()      # largest exponent is e^0 = 1: no overflow
        y = e / e.sum(dim=dim, keepdim=True)
        ctx.dim = dim                                      # non-tensor state goes on ctx directly
        ctx.save_for_backward(y)                           # y is all backward needs
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (y,) = ctx.saved_tensors
        dot = (grad_y * y).sum(dim=ctx.dim, keepdim=True)  # g . y per slice
        return y * (grad_y - dot), None                    # one gradient per forward input; dim gets None

def softmax_and_grad(x, dim, g):
    x = x.detach().clone().requires_grad_()
    y = Softmax.apply(x, dim)
    (y * g).sum().backward()
    return y.detach(), x.grad
'''

HINTS = [
    'Use `x.amax(dim=dim, keepdim=True)` for the max so it broadcasts back over the reduced axis, and `keepdim=True` on the sum too.',
    '`backward` must return one value per `forward` input after `ctx`: a tensor for `x` and `None` for `dim`.',
    'The Jacobian is $\\mathrm{diag}(y) - y y^\\top$. Multiplying it by $g$ without building it gives $y \\odot (g - (g\\cdot y))$, with the dot product taken along `dim` and kept as a size-1 axis.',
]

_r = np.random.default_rng(13)
TESTS = [
    (np.array([0.0, 0.0]), 0, np.array([1.0, 0.0])),
    (np.array([[1.0, 2.0, 3.0]]), -1, np.array([[0.0, 0.0, 1.0]])),
    (_r.normal(size=(3, 4)), 0, _r.normal(size=(3, 4))),
    (_r.normal(size=(2, 3, 5)), 1, _r.normal(size=(2, 3, 5))),
    (np.array([[1000.0, 1001.0], [-1000.0, -1000.0]]), 1, np.array([[1.0, -1.0], [2.0, 5.0]])),
    (np.array([[3.0]]), -1, np.array([[7.0]])),
    (_r.normal(size=(4, 6)) * 5, -1, np.ones((4, 6))),
    (_r.normal(size=(2, 2, 3, 4)), -2, _r.normal(size=(2, 2, 3, 4))),
]

EXPLANATION = r"""
## Why the backward works
Differentiating $y_i = e^{x_i}/S$ gives $\partial y_i/\partial x_k = y_i(\delta_{ik} - y_k)$, so the Jacobian is $\mathrm{diag}(y) - yy^\top$. Autograd only ever needs the vector-Jacobian product $J^\top g$, which is
$$y \odot g - y\,(y^\top g) = y \odot (g - g\cdot y).$$
That's $O(n)$ per slice instead of building an $n \times n$ matrix. Test 7 is a nice check: when $g$ is all ones, $g\cdot y = 1$ and the gradient is exactly zero, because the outputs always sum to 1.

## What to save
Saving `y` rather than `x` means backward doesn't recompute the exponentials. Memory is the same either way, one tensor shaped like the input. FlashAttention takes the opposite trade: it saves only the per-row log-sum-exp and recomputes the probabilities in backward, because the attention matrix is too big to keep.

## Complexity
Forward: three passes over the data (max, exp-and-sum, divide). Backward: two (dot product, combine). Both are memory-bandwidth-bound, which is why fused softmax kernels do one read and one write.

## Common bugs
- Forgetting `keepdim=True`: the `(n, 1)` sum becomes `(n,)` and broadcasts along the wrong axis, silently, for square inputs.
- Returning only one value from `backward` (it must match the number of `forward` inputs).
- Using the input-max trick in forward but `x` instead of `y` in backward formulas.

## Interview follow-ups
- Check your Function with `torch.autograd.gradcheck`. Why does it need float64 inputs?
- Derive the backward of `log_softmax` and explain why cross-entropy is usually fused with it.
"""
