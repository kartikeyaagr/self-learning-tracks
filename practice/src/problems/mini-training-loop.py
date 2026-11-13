import numpy as np

META = dict(title='The canonical PyTorch training loop', tags=['Training', 'Regression'], diff='Medium', runtime='torch',
            lessons=['air-fanout-v1:05.07', 'air-fanout-v1:10.04'],
            related=['mini-batch-training-loop', 'manual-weight-update', 'simple-neural-network'])

STATEMENT = r"""
Write the loop every PyTorch project has: batches in, `zero_grad`, forward, loss, `backward`, `step`, with the bookkeeping done right.

Write `train_regressor(X, Y, W0, b0, lr, orders, batch_size)`:

- `X`: float64 `(N, D)`, `Y`: float64 `(N, K)` targets.
- `W0`: `(K, D)`, `b0`: `(K,)`: initial weights of an `nn.Linear(D, K)` (already in PyTorch's `(out, in)` layout).
- `lr`: float, `batch_size`: int $\ge 1$.
- `orders`: int64 `(E, N)`. Row `e` is the sample order for epoch `e` (a permutation of `0..N-1`), so there is no hidden randomness.

## The loop
For each epoch `e`, walk `orders[e]` in consecutive chunks of `batch_size` (the last chunk may be smaller; keep it). For each chunk:

1. forward the batch through the model,
2. loss = `nn.MSELoss()` (mean over all `batch * K` entries),
3. `zero_grad`, `backward`, then one `torch.optim.SGD(lr=lr)` step (no momentum, no weight decay).

## Return
A tuple `(epoch_loss, W, b)`:

- `epoch_loss`: float64 `(E,)`. For epoch `e`, the average over **samples** of the loss seen during that epoch: $\sum_{\text{batches}} \text{loss}_\beta \cdot |\beta| \,/\, N$, using each batch's loss before its update.
- `W`, `b`: the final weights.

Don't modify the input tensors.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch
from torch import nn

def train_regressor(X, Y, W0, b0, lr, orders, batch_size):
    """Return (epoch_loss (E,), W (K, D), b (K,))."""
    model = nn.Linear(X.shape[1], Y.shape[1]).double()
    with torch.no_grad():
        model.weight.copy_(W0)
        model.bias.copy_(b0)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    epoch_loss = []
    for order in orders:
        total = 0.0
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            loss = nn.functional.mse_loss(model(X[idx]), Y[idx])
            loss.backward()
            opt.step()
            total += loss.item()
        epoch_loss.append(total / len(order))
    return torch.tensor(epoch_loss), model.weight.detach(), model.bias.detach()
'''

SOLUTION = '''import torch
from torch import nn

def train_regressor(X, Y, W0, b0, lr, orders, batch_size):
    model = nn.Linear(X.shape[1], Y.shape[1]).double()
    with torch.no_grad():                               # copying into parameters must not be recorded
        model.weight.copy_(W0)
        model.bias.copy_(b0)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    epoch_loss = []
    model.train()
    for order in orders:
        total = 0.0
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]               # last chunk may be short
            loss = loss_fn(model(X[idx]), Y[idx])
            opt.zero_grad()                             # grads accumulate by default
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)             # weight by batch size: a mean over samples
        epoch_loss.append(total / n)
    return torch.tensor(epoch_loss, dtype=torch.float64), model.weight.detach().clone(), model.bias.detach().clone()
'''

HINTS = [
    'Copy `W0`/`b0` into the `nn.Linear` inside `torch.no_grad()` (and call `.double()` first). Index a batch with `X[order[s:s + batch_size]]`.',
    '`backward()` **adds** into `.grad`. Without `opt.zero_grad()` every step uses the sum of all previous gradients.',
    'Each batch loss is already a mean over its entries. To average over samples across uneven batches, multiply by the batch size before summing and divide by $N$ at the end.',
]

_r = np.random.default_rng(17)

def _case(n, d, k, e, lr, bs):
    X = _r.normal(size=(n, d))
    Y = X @ _r.normal(size=(d, k)) + 0.1 * _r.normal(size=(n, k))
    orders = np.stack([_r.permutation(n) for _ in range(e)])
    return X, Y, _r.normal(size=(k, d)) * 0.1, np.zeros(k), lr, orders, bs

TESTS = [
    (np.array([[1.0], [2.0]]), np.array([[2.0], [4.0]]), np.array([[0.0]]), np.array([0.0]), 0.1,
     np.array([[0, 1]]), 1),
    (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([[1.0], [2.0], [3.0]]), np.zeros((1, 2)),
     np.array([0.5]), 0.05, np.array([[2, 0, 1], [1, 2, 0]]), 2),
    _case(10, 3, 2, 3, 0.05, 4),
    _case(7, 2, 1, 2, 0.1, 7),
    _case(9, 4, 3, 4, 0.02, 1),
    _case(16, 5, 2, 5, 0.1, 5),
    (np.ones((4, 1)), np.full((4, 1), 3.0), np.array([[0.0]]), np.array([0.0]), 0.0, np.array([[3, 2, 1, 0]]), 3),
]

EXPLANATION = r"""
## Why the order of calls matters
`loss.backward()` adds into each parameter's `.grad`; it never overwrites. So the loop must clear gradients once per update: `zero_grad()` then `backward()` then `step()`. Put `zero_grad` anywhere before `backward` and it's correct; leave it out and step $t$ uses $\sum_{s \le t} g_s$, which looks like a too-large learning rate and diverges.

## Averaging the epoch loss
`MSELoss` averages within a batch, so a 1-sample tail batch counts as much as a full one if you average batch losses directly. Weighting by `len(idx)` gives the true per-sample mean. `loss.item()` also detaches the number from the graph: accumulating `total += loss` instead keeps every batch's graph alive until the epoch ends, a slow memory leak.

## Complexity
$O(E\,N\,D\,K)$ for forward plus backward. The loop overhead in Python is per batch, so tiny batches are dominated by framework overhead, not math.

## Common bugs
- Missing `zero_grad` (the starter's bug).
- Creating the optimizer before `.double()`/`.to(device)` replaces parameters: here `.double()` is in-place on the module, but rebuilding the module after making the optimizer leaves it updating orphaned tensors.
- `model.weight = nn.Parameter(W0)`: works, but then the optimizer must be created after it, and it aliases the caller's tensor, so training mutates the input.

## Interview follow-ups
- Where do `model.train()` / `model.eval()` and `torch.no_grad()` go when you add a validation pass each epoch, and what does each one control?
- How does this loop change with gradient accumulation over 4 micro-batches, or with mixed precision and a `GradScaler`?
"""
