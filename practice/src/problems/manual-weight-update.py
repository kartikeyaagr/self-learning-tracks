import numpy as np

META = dict(title='SGD by hand with weight decay', tags=['Training', 'Optimization'], diff='Medium', runtime='torch',
            lessons=['air-fanout-v1:05.07', 'air-fanout-v1:01.08'],
            related=['basic-autograd', 'mini-training-loop', 'apply-one-full-sgd-training-step'])

STATEMENT = r"""
Train a softmax classifier with raw tensors: autograd computes the gradients, but you write the update yourself, no `nn.Module` and no `torch.optim`.

Write `manual_sgd(X, labels, W, b, lr, weight_decay, steps)`:

- `X`: float64 `(N, D)`; `labels`: int64 `(N,)` with values in `0..C-1`.
- `W`: float64 `(D, C)`, `b`: float64 `(C,)`: initial weights. Logits are `X @ W + b`.
- `lr`, `weight_decay`: floats; `steps`: int $\ge 1$ (full-batch steps).

## Each step
1. Compute the mean cross-entropy loss $L$ of the logits against `labels` and record it.
2. Get $\nabla_W L$ and $\nabla_b L$ with autograd.
3. Update, decaying the weights but **not** the bias:
$$W \leftarrow W - \text{lr}\,(\nabla_W L + \lambda W), \qquad b \leftarrow b - \text{lr}\,\nabla_b L.$$

## Return
`(losses, W, b)`: float64 `(steps,)` of the loss recorded before each update, and the final weights. Don't modify the input tensors.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch
import torch.nn.functional as F

def manual_sgd(X, labels, W, b, lr, weight_decay, steps):
    """Return (losses (steps,), W (D, C), b (C,))."""
    W = W.clone().requires_grad_()
    b = b.clone().requires_grad_()
    losses = []
    for _ in range(steps):
        loss = F.cross_entropy(X @ W + b, labels)
        loss.backward()
        with torch.no_grad():
            W -= lr * (W.grad + weight_decay * W)
            b -= lr * b.grad
        losses.append(loss.item())
    return torch.tensor(losses), W.detach(), b.detach()
'''

SOLUTION = '''import torch
import torch.nn.functional as F

def manual_sgd(X, labels, W, b, lr, weight_decay, steps):
    W = W.detach().clone().requires_grad_()        # fresh leaves; never mutate the caller's tensors
    b = b.detach().clone().requires_grad_()
    losses = []
    for _ in range(steps):
        loss = F.cross_entropy(X @ W + b, labels)  # mean over the batch, log-softmax inside
        loss.backward()
        with torch.no_grad():                      # the update itself must not be part of the graph
            W -= lr * (W.grad + weight_decay * W)  # in place: W stays the same leaf tensor
            b -= lr * b.grad                       # no decay on the bias
            W.grad.zero_()                         # backward() accumulates; clear for the next step
            b.grad.zero_()
        losses.append(loss.item())
    return torch.tensor(losses, dtype=torch.float64), W.detach(), b.detach()
'''

HINTS = [
    'Clone the inputs and call `.requires_grad_()` on the clones so they are leaves you own. `F.cross_entropy(logits, labels)` takes raw logits and int64 class indices.',
    'Do the update inside `torch.no_grad()` and in place (`W -= ...`). `W = W - ...` would create a new non-leaf tensor with no `.grad` next time round.',
    'After updating, zero the gradients (`W.grad.zero_()`). Otherwise the second step uses the sum of the first two gradients.',
]

_r = np.random.default_rng(29)

def _case(n, d, c, lr, wd, steps):
    return _r.normal(size=(n, d)), _r.integers(0, c, size=n), _r.normal(size=(d, c)) * 0.1, np.zeros(c), lr, wd, steps

TESTS = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]), np.zeros((2, 2)), np.zeros(2), 1.0, 0.0, 1),
    (np.array([[1.0], [-1.0]]), np.array([1, 0]), np.array([[0.5, -0.5]]), np.array([0.1, 0.0]), 0.5, 0.1, 3),
    _case(8, 3, 4, 0.1, 0.0, 5),
    _case(12, 4, 3, 0.5, 0.01, 10),
    _case(5, 2, 2, 0.2, 0.5, 4),
    (np.zeros((3, 2)), np.array([0, 0, 1]), np.ones((2, 3)), np.array([1.0, 2.0, 3.0]), 0.3, 0.2, 2),
    _case(20, 6, 5, 1.0, 1e-3, 15),
]

EXPLANATION = r"""
## Why it works
`loss.backward()` fills `W.grad` and `b.grad`. The update is ordinary arithmetic, and wrapping it in `torch.no_grad()` keeps autograd from recording it (otherwise PyTorch refuses an in-place change to a leaf that requires grad). Updating in place keeps `W` the same leaf tensor across steps, so its `.grad` slot keeps working. For cross-entropy the gradient is $\nabla_W L = X^\top(\mathrm{softmax}(XW+b) - Y)/N$, which you can use to check.

## Why not decay the bias
Weight decay is a prior that prefers small weights, which limits how sharply the model responds to inputs. The bias only shifts outputs, and shrinking it toward 0 just biases the class priors. Most training recipes exclude biases and norm parameters from decay.

## L2 vs decoupled decay
For plain SGD, adding $\lambda W$ to the gradient (L2 regularisation) and shrinking the weights directly are the same update. For Adam they are not: the L2 term gets divided by the adaptive denominator, which is why AdamW applies decay separately.

## Complexity
Each step is $O(NDC)$ for the forward and the same again for the backward.

## Common bugs
- Missing `grad.zero_()`: the starter's bug. Step 1 matches, every later step is wrong.
- `W = W - lr * W.grad` outside `no_grad()`: `W` becomes a non-leaf, `W.grad` is `None` on the next step.
- Recording `loss` (the tensor) instead of `loss.item()`: keeps every step's graph in memory.

## Interview follow-ups
- What does `torch.optim.SGD(..., weight_decay=λ)` do to this update, and how does `torch.optim.AdamW` differ?
- Why is `.data` (`W.data -= ...`) discouraged even though it works?
"""
