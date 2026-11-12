import torch

META = dict(title='Masked mean pooling in PyTorch', tags=['Embeddings', 'NLP'], diff='Easy', runtime='torch',
            lessons=['ieng:00.05'], related=['cosine-similarity-matrix', 'scaled-dot-product-attention'])

STATEMENT = r"""
Sentence-embedding models turn per-token hidden states into one vector per sequence by averaging, but padding tokens must not count.

Write `masked_mean(hidden, mask)`:

- `hidden`: float tensor `(B, T, D)` of token states.
- `mask`: tensor `(B, T)` of 0/1 (or bool). 1 marks a real token.
- Return `(B, D)`: for each sequence, the mean of `hidden` over its real tokens.

A sequence with no real tokens returns a zero vector. Use tensor ops only, no Python loops over the batch.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

def masked_mean(hidden, mask):
    """hidden: (B, T, D) float, mask: (B, T) 0/1. Return (B, D) mean over real tokens."""
    raise NotImplementedError
'''

SOLUTION = '''import torch

def masked_mean(hidden, mask):
    m = mask.to(hidden.dtype).unsqueeze(-1)          # (B, T, 1) so it broadcasts over D
    total = (hidden * m).sum(dim=1)                  # (B, D) sum of real tokens
    count = m.sum(dim=1).clamp(min=1)                # (B, 1); clamp avoids 0/0 for empty rows (sum is 0 there anyway)
    return total / count
'''

HINTS = [
    'Turn the mask into the same dtype as `hidden` and add a trailing axis, `(B, T, 1)`, so multiplying zeroes out padded tokens.',
    'Sum over the time axis (`dim=1`) for both the masked states and the mask itself. Their ratio is the mean.',
    'For all-padding rows the count is 0. `clamp(min=1)` makes the division safe, and the numerator is already 0.',
]

g = torch.Generator().manual_seed(0)
TESTS = [
    (torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]]), torch.tensor([[1, 1, 0]])),
    (torch.ones(2, 3, 4), torch.tensor([[1, 0, 0], [1, 1, 1]])),
    (torch.randn(3, 5, 6, generator=g), torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]])),
    (torch.randn(1, 4, 3, generator=g), torch.tensor([[0, 1, 0, 1]])),
    (torch.randn(4, 2, 8, generator=g), torch.ones(4, 2, dtype=torch.long)),
]

EXPLANATION = r"""
## Mask by multiplication, normalise by the count
Multiplying by a `(B, T, 1)` mask zeroes padded positions, so summing over `T` adds up only real tokens. Dividing by the per-row token count turns the sum into a mean. Everything is batched: one multiply, two sums and one divide.

## Why not `hidden.mean(dim=1)`
That divides by $T$, including padding. Embeddings for short sentences shrink toward the padding vector, and similarity scores then depend on how much a batch was padded. Inference results change with batch composition, which is a classic production bug.

## dtype detail
A long or bool mask times a float tensor is fine in PyTorch thanks to type promotion, but casting with `.to(hidden.dtype)` keeps fp16/bf16 inputs in their own precision instead of promoting to fp32.

## Interview follow-ups
- Mean pooling vs CLS-token pooling vs last-token pooling (decoder models): when would you use each?
- After pooling, why L2-normalise before computing cosine similarity in a vector index?
"""
