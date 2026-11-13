import numpy as np

META = dict(title='SGD with momentum as a torch.optim.Optimizer', tags=['Optimization', 'Training'], diff='Hard',
            runtime='torch', lessons=['air-fanout-v1:05.06', 'air-fanout-v1:05.05'],
            related=['classical-momentum', 'nesterov-momentum', 'manual-weight-update'])

STATEMENT = r"""
Built-in optimizers are ordinary Python classes. Write your own subclass of `torch.optim.Optimizer` that reproduces `torch.optim.SGD` with momentum, dampening, weight decay and Nesterov, then use it to train a linear model.

## The optimizer
Write `class MomentumSGD(torch.optim.Optimizer)` with defaults `lr, momentum=0, dampening=0, weight_decay=0, nesterov=False`. Its `step()` updates every parameter `p` that has a gradient, using its **own group's** hyper-parameters:

1. $g = \nabla p + \lambda\, p$ (weight decay $\lambda$ is added to the gradient, before momentum).
2. If momentum $\mu \ne 0$: on the parameter's first step the buffer is $b = g$ (a copy, **no dampening**); afterwards $b \leftarrow \mu\, b + (1 - \tau)\, g$ with dampening $\tau$. Then the direction is $d = g + \mu\, b$ if Nesterov, else $d = b$.
3. If $\mu = 0$: $d = g$ and no buffer is kept.
4. $p \leftarrow p - \eta\, d$.

Keep the buffer in `self.state[p]`. `step()` must not be recorded by autograd.

## The training run
Write `train_with_custom_sgd(W0, b0, X, Y, steps, lr, momentum, dampening, weight_decay, nesterov)`:

- `X`: float64 `(N, D)`, `Y`: float64 `(N, K)`; `W0`: `(K, D)`, `b0`: `(K,)` initial weights. Copy them; don't modify the inputs.
- Model: $\hat Y = X W^\top + b$. Loss: mean squared error over all $N K$ entries, on the full batch every step.
- Two parameter groups: `[W]` with `weight_decay=weight_decay`, and `[b]` with `weight_decay=0` (biases are not decayed). Both share `lr`, `momentum`, `dampening`, `nesterov`.
- Run `steps` iterations of: zero the gradients, loss, backward, `step()`.

Return a tuple `(losses, W, b, buf_W, buf_b)`:

- `losses`: float64 `(steps,)`, the loss at each step **before** its update.
- `W`, `b`: final parameters.
- `buf_W`, `buf_b`: final momentum buffers, or zero tensors of the parameter's shape when `momentum == 0`.

Nesterov is only used with `momentum > 0` and `dampening == 0`.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

class MomentumSGD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p -= group["lr"] * p.grad          # TODO: weight decay, momentum buffer, nesterov


def train_with_custom_sgd(W0, b0, X, Y, steps, lr, momentum, dampening, weight_decay, nesterov):
    """Return (losses (steps,), W, b, buf_W, buf_b)."""
    raise NotImplementedError
'''

SOLUTION = '''import torch

class MomentumSGD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)          # builds self.param_groups, filling missing keys from defaults

    @torch.no_grad()                                # parameter updates are not part of the graph
    def step(self):
        for group in self.param_groups:
            lr, mu, tau = group["lr"], group["momentum"], group["dampening"]
            wd, nesterov = group["weight_decay"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)          # new tensor: never overwrite p.grad
                if mu != 0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:                 # first step: plain copy, no dampening
                        buf = state["momentum_buffer"] = g.clone()
                    else:
                        buf.mul_(mu).add_(g, alpha=1 - tau)
                    g = g.add(buf, alpha=mu) if nesterov else buf
                p.add_(g, alpha=-lr)


def train_with_custom_sgd(W0, b0, X, Y, steps, lr, momentum, dampening, weight_decay, nesterov):
    W = W0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    opt = MomentumSGD([{"params": [W], "weight_decay": weight_decay},
                       {"params": [b], "weight_decay": 0.0}],
                      lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((X @ W.T + b - Y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    def buf(p):
        s = opt.state[p].get("momentum_buffer")
        return torch.zeros_like(p) if s is None else s.clone()

    return (torch.tensor(losses, dtype=torch.float64), W.detach().clone(), b.detach().clone(),
            buf(W), buf(b))
'''

HINTS = [
    'Pass a list of dicts to the constructor: `[{"params": [W], "weight_decay": wd}, {"params": [b], "weight_decay": 0.0}]`. Keys missing from a group come from `defaults`.',
    'Decorate `step` with `@torch.no_grad()` and update in place (`p.add_(d, alpha=-lr)`). Store the buffer in `self.state[p]["momentum_buffer"]`; `self.state` is a per-parameter dict that starts empty.',
    'On the first step the buffer is `g.clone()` with no `(1 - dampening)` factor. Add weight decay into a **new** tensor (`g = p.grad.add(p, alpha=wd)`), otherwise you corrupt `.grad` and the buffer aliases it.',
]

_r = np.random.default_rng(3)

def _case(n, d, k, steps, lr, mu, tau, wd, nest):
    X = _r.normal(size=(n, d))
    Y = X @ _r.normal(size=(d, k)) + 0.5 + 0.1 * _r.normal(size=(n, k))
    return 0.1 * _r.normal(size=(k, d)), np.zeros(k), X, Y, steps, lr, mu, tau, wd, nest

TESTS = [
    (np.array([[0.0]]), np.array([0.0]), np.array([[1.0], [2.0]]), np.array([[2.0], [4.0]]),
     3, 0.1, 0.9, 0.0, 0.0, False),
    (np.array([[1.0, -1.0]]), np.array([0.5]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0], [0.0]]),
     4, 0.2, 0.5, 0.0, 0.1, True),
    _case(8, 3, 2, 6, 0.05, 0.0, 0.0, 0.0, False),
    _case(10, 3, 1, 8, 0.05, 0.9, 0.0, 0.0, False),
    _case(12, 4, 2, 7, 0.03, 0.8, 0.3, 0.01, False),
    _case(9, 2, 3, 6, 0.04, 0.9, 0.0, 0.05, True),
    _case(6, 5, 2, 1, 0.1, 0.9, 0.5, 0.1, False),
    _case(15, 4, 2, 10, 0.02, 0.0, 0.0, 0.2, False),
]

EXPLANATION = r"""
## How an Optimizer is put together
`torch.optim.Optimizer.__init__` turns the parameter list (or list of group dicts) into `self.param_groups`, filling each group's missing keys from `defaults`, and creates `self.state`, a dict keyed by parameter. `zero_grad()` comes for free. All you write is `step()`: read each group's hyper-parameters and update its parameters in place, under `torch.no_grad()` so the update itself isn't recorded.

## The update rule, in order
Weight decay is added to the gradient **first**, so the momentum buffer accumulates decayed gradients (this is L2 regularisation, not decoupled AdamW-style decay). The buffer starts as a copy of the first gradient, which avoids the slow warm-up you'd get from $b_0 = 0$ followed by a dampened update. Nesterov uses $g + \mu b$: the step looks one momentum step ahead. With $\mu = 0.9$ and no dampening, the steady-state step on a constant gradient is $\eta g / (1 - \mu) = 10\,\eta g$, which is why momentum runs usually pair with smaller learning rates.

## Complexity
One pass over every parameter: $O(P)$ time and $O(P)$ extra memory for the buffers. The training run is $O(\text{steps} \cdot N D K)$.

## Common bugs
- Adding weight decay in place (`p.grad.add_(p, alpha=wd)`): it changes `.grad`, and on the first step the buffer then aliases the gradient tensor, so `zero_grad` can wipe the momentum.
- Dampening the first step: $b_1 = (1-\tau) g$ is a different optimizer.
- Decaying the bias because one group was used for everything.
- Forgetting `@torch.no_grad()`: in-place ops on leaf tensors that require grad raise an error.

## Interview follow-ups
- Rewrite the momentum update as $p \leftarrow p - \eta v$ with $v \leftarrow \mu v + g$ versus $v \leftarrow \mu v + \eta g$. When are the two equivalent, and what breaks when the learning rate changes mid-run?
- How would you add a `foreach` implementation that updates all parameters of a group with a few fused kernels instead of one small kernel per tensor?
"""
