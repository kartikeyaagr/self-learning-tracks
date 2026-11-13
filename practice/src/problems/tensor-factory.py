import numpy as np

META = dict(title='Build the helper tensors for a sequence batch', tags=['Neural Networks'], diff='Easy', runtime='torch',
            lessons=['air-fanout-v1:03.01', 'air-fanout-v1:03.07'],
            related=['arange-and-linspace', 'tensor-operations', 'tensor-shape-manipulation'])

STATEMENT = r"""
Before a sequence model runs, the training script builds a handful of small tensors: position ids, a padding mask, a causal mask, a time grid. Each one is a single PyTorch factory call if you know which one to reach for.

Write `make_helpers(lengths, T, t0, t1)`:

- `lengths`: int64 tensor `(B,)`, the number of real tokens in each sequence ($0 \le$ `lengths[b]` $\le T$).
- `T`: int, the padded sequence length ($T \ge 1$).
- `t0`, `t1`: floats, the ends of a time grid.

Return a tuple of five tensors:

| # | Name | Shape | dtype | Contents |
|---|---|---|---|---|
| 1 | `pos` | `(T,)` | int64 | `0, 1, …, T-1` |
| 2 | `pad_mask` | `(B, T)` | bool | `True` where position `t < lengths[b]` |
| 3 | `causal` | `(T, T)` | bool | `True` where key `j <= ` query `i` (lower triangle, diagonal included) |
| 4 | `grid` | `(T,)` | float64 | `T` evenly spaced points from `t0` to `t1`, both ends included (for `T = 1` it is `[t0]`) |
| 5 | `bias` | `(B, T)` | float64 | `0.0` at real positions, `-inf` at padded ones |

Use factory functions and broadcasting, no Python loops. The grader compares values, so `True`/`1` both pass, but get the dtypes right anyway: they matter the moment these tensors meet a model.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def make_helpers(lengths, T, t0, t1):
    """Return (pos, pad_mask, causal, grid, bias). See the table for shapes and dtypes."""
    pos = torch.zeros(T)
    raise NotImplementedError
'''

SOLUTION = '''import torch

def make_helpers(lengths, T, t0, t1):
    pos = torch.arange(T)                                      # int64 by default for integer arguments
    pad_mask = pos.unsqueeze(0) < lengths.unsqueeze(1)          # (1, T) vs (B, 1) broadcasts to (B, T)
    causal = torch.ones(T, T, dtype=torch.bool).tril()          # row i keeps columns 0..i
    grid = torch.linspace(t0, t1, T, dtype=torch.float64)       # linspace includes both ends
    bias = torch.zeros(pad_mask.shape, dtype=torch.float64).masked_fill(~pad_mask, float('-inf'))
    return pos, pad_mask, causal, grid, bias
'''

HINTS = [
    '`torch.arange`, `torch.ones(...).tril()` and `torch.linspace` cover three of the five. Pass `dtype=` explicitly where the default (float32) is wrong.',
    'For the padding mask compare a row of positions with a column of lengths: `pos[None, :] < lengths[:, None]` broadcasts to `(B, T)`.',
    'The additive bias is a zeros tensor with `-inf` written where the mask is `False`: `masked_fill(~pad_mask, float("-inf"))`.',
]

TESTS = [
    (np.array([2, 3]), 3, 0.0, 1.0),
    (np.array([1]), 4, -1.0, 1.0),
    (np.array([0, 5, 2]), 5, 0.0, 10.0),
    (np.array([1, 0]), 1, 3.0, 7.0),
    (np.array([2, 0]), 2, 0.0, 0.0),
    (np.array([6, 1, 4, 6]), 6, 2.5, -2.5),
    (np.array([7, 3, 0, 5, 1]), 7, 0.0, 1e-3),
]

EXPLANATION = r"""
## Why it works
Each tensor is one factory call plus, at most, one broadcast. `arange(T)` gives positions. Comparing a `(1, T)` row of positions against a `(B, 1)` column of lengths broadcasts to the full `(B, T)` mask with no loop. `tril` on a square of ones keeps entries with $j \le i$, which is exactly "a query may look at itself and the past". `linspace` includes both endpoints, so the step is $(t_1 - t_0)/(T-1)$.

## Defaults that bite
- `torch.arange(T)` is int64, but `torch.arange(0.0, T)` is float32. Position ids feed `nn.Embedding`, which needs an integer tensor.
- `torch.linspace`, `zeros`, `ones` and `full` default to float32 (`torch.get_default_dtype()`). Mixing a float32 helper with a float64 model silently promotes or errors depending on the op.
- A bool mask and an additive `-inf` bias carry the same information. Attention kernels accept either; the additive form is what you add to scores, the bool form is what `masked_fill` and indexing want.

## Complexity
Everything is $O(BT + T^2)$ memory. The causal mask is the only quadratic piece, and fused attention kernels never build it at all: they compute $j \le i$ on the fly inside each tile.

## Common bugs
- `torch.triu` instead of `tril`, which lets every token see the future.
- `pos <= lengths` (off by one), which marks one padded slot as real.
- `torch.full((B, T), -inf)` and then filling real positions with 0 is fine; `torch.empty` and partial filling is not, because `empty` holds garbage.

## Interview follow-ups
- When would you register a causal mask as a buffer on the module instead of rebuilding it every forward? (Fixed max length; it moves with `.to(device)` and isn't a parameter.)
- How do you build the same masks for left-padded batches used in batched generation?
"""
