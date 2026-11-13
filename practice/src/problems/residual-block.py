import numpy as np

META = dict(title='A ResNet bottleneck block as an nn.Module', tags=['Convolutional Networks', 'Neural Networks'],
            diff='Medium', runtime='torch', lessons=['air-fanout-v1:04.02', 'ieng:02.03'],
            related=['resnet-residual-block', 'apply-alphago-residual-block', 'batch-normalization'])

STATEMENT = r"""
Deep ResNets use a **bottleneck** block: a 1x1 conv shrinks the channels, a 3x3 conv works at the narrow width, and a 1x1 conv expands back. Write it as an `nn.Module`, load given weights through a `state_dict`, and run it in eval mode.

## The module
Write `class Bottleneck(nn.Module)` with `__init__(self, c_in, c_mid, c_out, stride)` and these submodules (names matter, they are the `state_dict` keys):

| name | layer |
|---|---|
| `conv1` | `Conv2d(c_in, c_mid, 1, bias=False)` |
| `bn1` | `BatchNorm2d(c_mid)` |
| `conv2` | `Conv2d(c_mid, c_mid, 3, stride=stride, padding=1, bias=False)` |
| `bn2` | `BatchNorm2d(c_mid)` |
| `conv3` | `Conv2d(c_mid, c_out, 1, bias=False)` |
| `bn3` | `BatchNorm2d(c_out)` |
| `downsample` | only if `stride != 1` or `c_in != c_out`: `Sequential(Conv2d(c_in, c_out, 1, stride=stride, bias=False), BatchNorm2d(c_out))`; otherwise `None` |

Forward: $y = \text{ReLU}\big(\text{bn3}(\text{conv3}(\text{ReLU}(\text{bn2}(\text{conv2}(\text{ReLU}(\text{bn1}(\text{conv1}(x)))))))) + s(x)\big)$, where $s$ is `downsample` or the identity. The stride sits on the 3x3 conv.

## The function
Write `bottleneck_forward(x, w1, bn1, w2, bn2, w3, bn3, wd, bnd, stride)`:

- `x`: float64 `(N, c_in, H, W)`.
- `w1`, `w2`, `w3`: conv weights in `nn.Conv2d` layout; their shapes give `c_in`, `c_mid`, `c_out`.
- `bn1`, `bn2`, `bn3`: float64 `(4, C)`, rows `[weight, bias, running_mean, running_var]` of the matching BatchNorm.
- `wd`, `bnd`: the downsample conv weight `(c_out, c_in, 1, 1)` and its BatchNorm `(4, c_out)`, or both `None` when the block has no downsample.
- `stride`: int.

Build the module in float64, load the weights with `load_state_dict` (all keys, strict), switch to **eval** mode (BatchNorm uses the running statistics, `eps = 1e-5`), and return a tuple `(y, n_params)`: the output and the number of trainable parameters (`sum of p.numel()` over `parameters()`, an int).

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch
from torch import nn

class Bottleneck(nn.Module):
    def __init__(self, c_in, c_mid, c_out, stride):
        super().__init__()
        # TODO: conv1/bn1, conv2/bn2, conv3/bn3 and an optional downsample
        self.downsample = None

    def forward(self, x):
        raise NotImplementedError


def bottleneck_forward(x, w1, bn1, w2, bn2, w3, bn3, wd, bnd, stride):
    """Return (y, n_params)."""
    raise NotImplementedError
'''

SOLUTION = '''import torch
from torch import nn

class Bottleneck(nn.Module):
    def __init__(self, c_in, c_mid, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.conv2 = nn.Conv2d(c_mid, c_mid, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.conv3 = nn.Conv2d(c_mid, c_out, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()                                   # no parameters, so it adds no state_dict keys
        self.downsample = None
        if stride != 1 or c_in != c_out:                        # shapes differ: project the shortcut
            self.downsample = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                                            nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))                         # no ReLU before the addition
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(out + identity)


def _bn_state(prefix, p):
    return {f"{prefix}.weight": p[0], f"{prefix}.bias": p[1], f"{prefix}.running_mean": p[2],
            f"{prefix}.running_var": p[3], f"{prefix}.num_batches_tracked": torch.tensor(0)}


def bottleneck_forward(x, w1, bn1, w2, bn2, w3, bn3, wd, bnd, stride):
    c_mid, c_in = w1.shape[:2]
    c_out = w3.shape[0]
    block = Bottleneck(c_in, c_mid, c_out, stride).double()
    state = {"conv1.weight": w1, "conv2.weight": w2, "conv3.weight": w3,
             **_bn_state("bn1", bn1), **_bn_state("bn2", bn2), **_bn_state("bn3", bn3)}
    if wd is not None:
        state |= {"downsample.0.weight": wd, **_bn_state("downsample.1", bnd)}
    block.load_state_dict(state)                                # strict: every key must match exactly
    block.eval()                                                # BatchNorm uses running statistics
    with torch.no_grad():
        y = block(x)
    return y, sum(p.numel() for p in block.parameters())
'''

HINTS = [
    'Assign every layer as an attribute in `__init__`; the attribute names become the `state_dict` keys (`conv1.weight`, `bn1.running_mean`, `downsample.0.weight`, ...). Print `block.state_dict().keys()` to see what `load_state_dict` expects.',
    'A BatchNorm2d has five state entries: `weight`, `bias`, `running_mean`, `running_var` and `num_batches_tracked`. Strict loading needs all of them; `torch.tensor(0)` works for the counter.',
    'Call `.double()` before loading and `.eval()` before the forward. In training mode BatchNorm normalises with the batch\'s own statistics and gives different numbers. The last ReLU comes **after** the addition.',
]

_r = np.random.default_rng(37)

def _bn(c):
    return np.stack([_r.uniform(0.5, 1.5, c), _r.normal(0, 0.1, c), _r.normal(0, 0.2, c), _r.uniform(0.5, 2.0, c)])

def _case(n, c_in, c_mid, c_out, h, w, stride):
    down = stride != 1 or c_in != c_out
    return (_r.normal(size=(n, c_in, h, w)),
            _r.normal(size=(c_mid, c_in, 1, 1)) / np.sqrt(c_in), _bn(c_mid),
            _r.normal(size=(c_mid, c_mid, 3, 3)) / np.sqrt(9 * c_mid), _bn(c_mid),
            _r.normal(size=(c_out, c_mid, 1, 1)) / np.sqrt(c_mid), _bn(c_out),
            _r.normal(size=(c_out, c_in, 1, 1)) / np.sqrt(c_in) if down else None,
            _bn(c_out) if down else None, stride)

_id_bn = lambda c: np.stack([np.ones(c), np.zeros(c), np.zeros(c), np.ones(c)])

TESTS = [
    (np.arange(4.0).reshape(1, 1, 2, 2) - 1.5, np.ones((1, 1, 1, 1)), _id_bn(1), np.zeros((1, 1, 3, 3)), _id_bn(1),
     np.ones((1, 1, 1, 1)), _id_bn(1), None, None, 1),
    (np.ones((1, 1, 2, 2)), np.ones((1, 1, 1, 1)), _id_bn(1), np.ones((1, 1, 3, 3)), _id_bn(1),
     np.ones((2, 1, 1, 1)), _id_bn(2), np.array([[[[1.0]]], [[[-1.0]]]]), _id_bn(2), 2),
    _case(2, 8, 2, 8, 5, 5, 1),
    _case(1, 4, 2, 8, 6, 6, 1),
    _case(2, 4, 4, 16, 7, 6, 2),
    _case(1, 6, 3, 6, 4, 4, 2),
    _case(3, 8, 4, 8, 3, 3, 1),
]

EXPLANATION = r"""
## Why a bottleneck
A basic block runs two 3x3 convs at full width $C$: about $18C^2$ weights. The bottleneck runs its 3x3 at $C/4$ and wraps it in 1x1 convs: $C \cdot \tfrac{C}{4} + 9(\tfrac{C}{4})^2 + \tfrac{C}{4} \cdot C \approx 1.06\,C^2$. That is how ResNet-50 and deeper get to 4x wider outputs (256, 512, ...) at a similar cost to ResNet-34.

## The shortcut
The addition needs matching shapes. When the stride or channel count changes, a strided 1x1 conv plus BatchNorm projects $x$; otherwise the identity is used, so gradients pass straight through the block. The final ReLU comes after the sum. Putting the stride on the 3x3 conv instead of the first 1x1 (the "v1.5" variant) keeps the 1x1 from throwing away three quarters of the pixels before they're seen.

## state_dict mechanics
`nn.Module` registers each attribute that is a module or parameter, and the `state_dict` keys are the dotted attribute paths: `downsample.0.weight` is the conv inside the `Sequential`. BatchNorm also stores buffers (`running_mean`, `running_var`, `num_batches_tracked`), which are in the `state_dict` but not in `parameters()`. Strict loading is a cheap check that your architecture matches the checkpoint.

## Common bugs
- Forgetting `.eval()`: BatchNorm uses batch statistics and updates the running ones as a side effect.
- A ReLU after `bn3`, before the addition (it clips the residual branch to be non-negative).
- Building the downsample whenever `c_in != c_mid`, or only when the stride changes.
- Counting buffers as parameters: `n_params` covers conv weights and BatchNorm `weight`/`bias` only.

## Interview follow-ups
- Why do many training recipes initialise `bn3.weight` to zero, and what does the block compute at the start of training then?
- How do you fold an eval-mode BatchNorm into the preceding conv's weights and bias for inference, and what changes for the downsample branch?
"""
