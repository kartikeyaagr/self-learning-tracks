import numpy as np

META = dict(title='A two-layer classifier as an nn.Module', tags=['Neural Networks', 'Classification'], diff='Easy', runtime='torch',
            lessons=['air-fanout-v1:05.03', 'ieng:00.05'],
            related=['multi-layer-perceptron-forward-pass', 'custom-linear-layer', 'mini-training-loop'])

STATEMENT = r"""
Write the standard PyTorch classifier: a module with two linear layers and a ReLU, loaded with weights that came from somewhere else.

Write `mlp_predict(x, W1, b1, W2, b2)`:

- `x`: float64 tensor `(N, D)`.
- `W1`: `(D, H)`, `b1`: `(H,)`, `W2`: `(H, C)`, `b2`: `(C,)`, all float64. The weights are given in **math layout**: the layer computes $x W_1 + b_1$.
- Build an `nn.Module` (a subclass or `nn.Sequential`) made of `nn.Linear(D, H)`, `nn.ReLU()`, `nn.Linear(H, C)`, in float64, and copy these weights into it.

Return a tuple `(logits, pred, n_params)`:

- `logits`: `(N, C)` float64, the module's output.
- `pred`: `(N,)` int64, the predicted class per row. Ties go to the smallest class index.
- `n_params`: int, the total number of trainable scalars in the module.

Compute the output in inference mode (no autograd graph).

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch
from torch import nn

def mlp_predict(x, W1, b1, W2, b2):
    """Return (logits (N, C), pred (N,), n_params)."""
    D, H = W1.shape
    C = W2.shape[1]
    model = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, C)).double()
    with torch.no_grad():
        logits = model(x)
    return logits, logits.argmax(dim=1), 0
'''

SOLUTION = '''import torch
from torch import nn

class TwoLayerNet(nn.Module):
    def __init__(self, d, h, c):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(h, c)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def mlp_predict(x, W1, b1, W2, b2):
    (d, h), c = W1.shape, W2.shape[1]
    model = TwoLayerNet(d, h, c).double()
    # nn.Linear stores weight as (out_features, in_features) and computes x @ weight.T + bias
    model.load_state_dict({'fc1.weight': W1.T, 'fc1.bias': b1, 'fc2.weight': W2.T, 'fc2.bias': b2})
    model.eval()
    with torch.inference_mode():
        logits = model(x)
    pred = logits.argmax(dim=1)                       # first maximum on ties
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return logits, pred, n_params
'''

HINTS = [
    '`nn.Linear(in, out)` stores `weight` with shape `(out, in)` and computes `x @ weight.T + bias`. Your `W1` is `(in, out)`.',
    'Copy weights with `load_state_dict` (keys like `"fc1.weight"`) or inside `torch.no_grad()` with `layer.weight.copy_(W1.T)`. Call `.double()` on the module so its parameters match the float64 inputs.',
    'Count parameters with `sum(p.numel() for p in model.parameters())`. It should be $DH + H + HC + C$.',
]

_r = np.random.default_rng(5)

def _case(n, d, h, c, scale=1.0):
    return (_r.normal(size=(n, d)), _r.normal(size=(d, h)) * scale, _r.normal(size=h) * scale,
            _r.normal(size=(h, c)) * scale, _r.normal(size=c) * scale)

TESTS = [
    (np.array([[1.0, 2.0]]), np.array([[1.0, -1.0], [0.0, 1.0]]), np.array([0.0, 0.0]),
     np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.0, 0.5])),
    (np.array([[1.0], [-1.0]]), np.array([[2.0, -3.0, 0.5]]), np.array([0.0, 1.0, 0.0]),
     np.array([[1.0], [1.0], [-2.0]]), np.array([0.1])),
    _case(5, 4, 6, 3),
    _case(1, 3, 2, 4, scale=0.1),
    (np.zeros((3, 2)), np.ones((2, 2)), np.array([-1.0, -1.0]), np.ones((2, 3)), np.array([0.0, 0.0, 0.0])),
    _case(12, 8, 16, 5),
]

EXPLANATION = r"""
## Why it works
An `nn.Module` registers every `nn.Linear` assigned as an attribute, so `parameters()` and `state_dict()` find them by name (`fc1.weight`, `fc1.bias`, ...). `load_state_dict` copies tensors in by those names and checks every shape, so a forgotten transpose fails loudly when $D \ne H$, but silently produces wrong outputs when the matrix is square. `eval()` plus `inference_mode()` turns off dropout/BatchNorm updates (none here, but it's the habit) and skips building the autograd graph.

## Layout
PyTorch stores `weight` as `(out, in)` so that each row is one output neuron's weights, which makes `x @ W.T` a contiguous GEMM. Checkpoints from other frameworks (Keras, JAX/Flax) often use `(in, out)`: transpose on load.

## Complexity
$O(N(DH + HC))$ multiply-adds. Parameters: $DH + H + HC + C$.

## Common bugs
- Forgetting `.double()`: `nn.Linear` defaults to float32 and the matmul with a float64 input raises a dtype error.
- `argmax` on probabilities instead of logits: same answer, wasted softmax.
- Counting `model.state_dict()` entries instead of parameters: buffers (like BatchNorm's running stats) are in the state dict but aren't trainable.

## Interview follow-ups
- What's the difference between `torch.no_grad()` and `torch.inference_mode()`?
- How would you load a checkpoint whose keys have a `module.` prefix from `DataParallel`?
"""
