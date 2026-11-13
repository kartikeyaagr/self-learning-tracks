import numpy as np

META = dict(title='A LoRA linear layer as a custom module', tags=['Neural Networks', 'Language Models'], diff='Medium', runtime='torch',
            lessons=['air-fanout-v1:10.02', 'ieng:26.02'],
            related=['simple-neural-network', 'low-rank-approximation', 'basic-autograd'])

STATEMENT = r"""
Low-rank adaptation (LoRA) fine-tunes a large linear layer by freezing its weight $W$ and learning a small update $BA$ on the side:

$$y = x W^\top + b + \frac{\alpha}{r}\,(x A^\top) B^\top$$

with $W \in \mathbb{R}^{\text{out}\times\text{in}}$, $A \in \mathbb{R}^{r\times\text{in}}$, $B \in \mathbb{R}^{\text{out}\times r}$.

## The module
Write a class `LoRALinear(nn.Module)` with:
- frozen `weight` `(out, in)` and `bias` `(out,)` (stored but `requires_grad=False`),
- trainable `nn.Parameter`s `lora_A` `(r, in)` and `lora_B` `(out, r)`,
- a `forward` that accepts `x` of shape `(..., in)` (any number of leading dims) and returns `(..., out)`.

## The function that's graded
Write `lora_step(x, W, b, A, B, alpha, g)`:
- `x` `(..., in)`, `W`, `b`, `A`, `B`: float64 tensors as above; `alpha`: float; `r` is `A.shape[0]`.
- `g`: float64 tensor with the same shape as the output.

Build the module with these weights, compute `y = module(x)`, and backpropagate the scalar `(y * g).sum()`. Return a tuple

`(y, grad_A, grad_B, frozen_has_grad, n_trainable)`

- `grad_A`, `grad_B`: the gradients on the LoRA parameters.
- `frozen_has_grad`: bool, whether `weight` or `bias` ended up with a `.grad` (it should be `False`).
- `n_trainable`: int, the number of scalars in parameters with `requires_grad=True`.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, W, b, A, B, alpha):
        super().__init__()
        self.weight = nn.Parameter(W.clone())
        self.bias = nn.Parameter(b.clone())
        self.lora_A = nn.Parameter(A.clone())
        self.lora_B = nn.Parameter(B.clone())
        self.scale = alpha

    def forward(self, x):
        return x @ self.weight.T + self.bias + self.scale * x @ self.lora_A.T @ self.lora_B.T

def lora_step(x, W, b, A, B, alpha, g):
    """Return (y, grad_A, grad_B, frozen_has_grad, n_trainable)."""
    m = LoRALinear(W, b, A, B, alpha)
    y = m(x)
    (y * g).sum().backward()
    n = sum(p.numel() for p in m.parameters())
    return y, m.lora_A.grad, m.lora_B.grad, m.weight.grad is not None, n
'''

SOLUTION = '''import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, W, b, A, B, alpha):
        super().__init__()
        # Frozen base weights: Parameters with requires_grad=False still live in state_dict and move with .to().
        self.weight = nn.Parameter(W.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(b.detach().clone(), requires_grad=False)
        self.lora_A = nn.Parameter(A.detach().clone())          # (r, in)
        self.lora_B = nn.Parameter(B.detach().clone())          # (out, r)
        self.scale = alpha / A.shape[0]                         # alpha / r, not alpha

    def forward(self, x):
        base = x @ self.weight.T + self.bias                    # (..., out); matmul broadcasts leading dims
        update = (x @ self.lora_A.T) @ self.lora_B.T            # through the rank-r bottleneck, never forms B @ A
        return base + self.scale * update

def lora_step(x, W, b, A, B, alpha, g):
    m = LoRALinear(W, b, A, B, alpha)
    y = m(x)
    (y * g).sum().backward()
    frozen_has_grad = m.weight.grad is not None or m.bias.grad is not None
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return y.detach(), m.lora_A.grad, m.lora_B.grad, frozen_has_grad, n_trainable
'''

HINTS = [
    'Register the frozen tensors as `nn.Parameter(..., requires_grad=False)` (or with `register_buffer`), and the LoRA factors as ordinary `nn.Parameter`s. Autograd then never computes gradients for the frozen ones.',
    'The scale is $\\alpha / r$ where $r$ = `A.shape[0]`. Multiplying by `alpha` alone is the most common LoRA bug.',
    'Apply the update as `(x @ A.T) @ B.T` so it goes through the narrow rank-`r` dimension. `@` broadcasts over any leading dims of `x`.',
]

_r = np.random.default_rng(21)

def _case(lead, d_in, d_out, r, alpha):
    x = _r.normal(size=(*lead, d_in))
    return (x, _r.normal(size=(d_out, d_in)), _r.normal(size=d_out), _r.normal(size=(r, d_in)),
            _r.normal(size=(d_out, r)), alpha, _r.normal(size=(*lead, d_out)))

TESTS = [
    (np.array([[1.0, 2.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.0, 0.0]),
     np.array([[1.0, 1.0]]), np.array([[1.0], [0.0]]), 2.0, np.array([[1.0, 1.0]])),
    (np.array([[1.0, -1.0, 2.0]]), np.zeros((2, 3)), np.array([0.5, -0.5]),
     np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.zeros((2, 2)), 4.0, np.array([[1.0, 2.0]])),
    _case((4,), 5, 3, 2, 8.0),
    _case((2, 3), 6, 4, 1, 1.0),
    _case((1,), 4, 4, 4, 16.0),
    _case((2, 2, 2), 3, 5, 2, 0.5),
    _case((7,), 8, 6, 3, 6.0),
]

EXPLANATION = r"""
## Why it works
`nn.Module.__setattr__` registers any `nn.Parameter` assigned to `self`, so `parameters()`, `state_dict()` and optimizers see all four tensors. `requires_grad=False` on the base weights means autograd never builds the `weight.grad` path: no memory for those gradients and no optimizer state for them, which is the whole point of LoRA on a 7B-parameter model.

The gradients you get back are
$$\frac{\partial \mathcal L}{\partial B} = \frac{\alpha}{r}\, G^\top (X A^\top), \qquad \frac{\partial \mathcal L}{\partial A} = \frac{\alpha}{r}\, (G B)^\top X,$$
with $X$ and $G$ flattened over the leading dims. Test 2 has $B = 0$, the standard LoRA initialisation: the output equals the base layer and $\partial\mathcal L/\partial A = 0$, while $\partial\mathcal L/\partial B$ is not zero, so training can start.

## Complexity
The update costs $O(N r(\text{in} + \text{out}))$ instead of $O(N\,\text{in}\cdot\text{out})$. Trainable parameters drop from $\text{in}\cdot\text{out}$ to $r(\text{in}+\text{out})$; for a 4096×4096 projection at $r = 8$ that's 65k instead of 16.8M.

## Common bugs
- Scaling by $\alpha$ instead of $\alpha/r$: then changing $r$ changes the effective learning rate.
- Computing `B @ A` first: correct, but it materialises an `(out, in)` matrix every forward and defeats the efficiency.
- Freezing with `weight.detach()` inside `forward`: works for gradients, but the weight still appears trainable to the optimizer and to parameter counts.

## Interview follow-ups
- How do you merge LoRA into the base weight for serving, and when would you keep it unmerged? (Multi-adapter serving.)
- Why initialise $B = 0$ and $A$ random rather than the other way around, or both random?
"""
