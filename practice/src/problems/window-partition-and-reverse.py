import numpy as np

META = dict(title='Shifted window partition and reverse', tags=['Computer Vision', 'Attention'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:03.04'], related=['vit-patchify', 'vit-multi-head-self-attention', 'reshaping-arrays'])

STATEMENT = r"""
Swin Transformers compute attention inside non-overlapping `ws x ws` windows of the token grid instead of over the whole image. Every other block **cyclically shifts** the grid before partitioning, so information crosses window borders. The two building blocks are a partition (grid to windows) and its reverse (windows back to grid).

Write `window_roundtrip(x, ws, shift=0)` and return `(windows, merged)`.

## Partition
1. If `shift > 0`, roll the grid up and left: `np.roll(x, (-shift, -shift), axis=(1, 2))`.
2. Cut every image into `(H/ws) * (W/ws)` windows. `windows` has shape `(B * (H/ws) * (W/ws), ws, ws, C)`, ordered by image, then window row, then window column. Window `(r, c)` of an image holds rows `r*ws : (r+1)*ws` and columns `c*ws : (c+1)*ws` of the (rolled) grid.

## Reverse, tested through a per-window operation
3. Replace every token of each window by the **mean over that window's `ws * ws` tokens** (per channel). This stands in for the attention that would run inside each window.
4. Reverse the partition to get a `(B, H, W, C)` grid, then undo the shift with `np.roll(..., (shift, shift), axis=(1, 2))`.

`merged` is that final grid: each original token holds the mean of the window it was in, at its original position.

## Input / output
- `x`: float `(B, H, W, C)` with `H`, `W` multiples of `ws`; `0 <= shift < ws`.
- `windows`: `float64` `(nW, ws, ws, C)` from step 2. `merged`: `float64` `(B, H, W, C)`.
"""

STARTER = '''import numpy as np

def window_roundtrip(x, ws, shift=0):
    """x: (B, H, W, C). Return (windows (nW, ws, ws, C), merged (B, H, W, C) with per-window means)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.reshape(B, H // ws, ws, W // ws, ws, C).transpose(0, 1, 3, 2, 4, 5)   # (B, gh, gw, ws, ws, C)
    return x.reshape(-1, ws, ws, C)

def window_reverse(windows, ws, B, H, W):
    C = windows.shape[-1]
    x = windows.reshape(B, H // ws, W // ws, ws, ws, C).transpose(0, 1, 3, 2, 4, 5)  # undo the transpose
    return x.reshape(B, H, W, C)

def window_roundtrip(x, ws, shift=0):
    x = np.asarray(x, dtype=np.float64)
    B, H, W, _ = x.shape
    if shift:
        x = np.roll(x, (-shift, -shift), axis=(1, 2))
    windows = window_partition(x, ws)
    pooled = np.broadcast_to(windows.mean(axis=(1, 2), keepdims=True), windows.shape)
    merged = window_reverse(pooled, ws, B, H, W)
    if shift:
        merged = np.roll(merged, (shift, shift), axis=(1, 2))
    return windows, merged
'''

HINTS = [
    'Partition is the same reshape-transpose trick as cutting patches: split `H` into `(H/ws, ws)` and `W` into `(W/ws, ws)` and bring the two grid axes together.',
    'Reverse applies the inverse operations in the reverse order: reshape the windows to `(B, H/ws, W/ws, ws, ws, C)`, apply the same axis swap (it is its own inverse), and reshape to `(B, H, W, C)`.',
    'Per-window mean: `windows.mean(axis=(1, 2), keepdims=True)` broadcast back to the window shape. Roll by `-shift` before partitioning and by `+shift` after reversing.',
]

_r = np.random.default_rng(37)
TESTS = [
    (np.arange(16, dtype=float).reshape(1, 4, 4, 1), 2),
    (np.arange(16, dtype=float).reshape(1, 4, 4, 1), 2, 1),
    (np.arange(24, dtype=float).reshape(1, 2, 6, 2), 2),
    (_r.normal(size=(2, 6, 6, 3)), 3, 1),
    (_r.normal(size=(1, 8, 4, 2)), 4, 2),
    (_r.normal(size=(2, 4, 8, 1)), 4),
    (_r.normal(size=(1, 3, 3, 2)), 3, 2),
]

EXPLANATION = r"""
## Why the axis swap is self-inverse
Reshaping `(B, H, W, C)` to `(B, gh, ws, gw, ws, C)` just relabels memory. Swapping axes 2 and 3 groups each window's pixels together. Undoing it means swapping the same two axes back on the reshaped windows, which is why `window_reverse` uses the identical transpose permutation `(0, 1, 3, 2, 4, 5)`.

## Why shift
Without shifting, tokens in different windows never interact within a block. Rolling by `ws // 2` makes the next block's windows straddle the previous block's borders. The roll is cyclic, so some windows contain tokens that wrapped around from the opposite edge; real Swin adds an attention mask so those don't attend to each other. Test 2 shows the effect: the wrapped windows mix the first and last rows and columns.

## Complexity
Partition and reverse are $O(BHWC)$ copies (the transpose forces one). Windowed attention costs $O(HW \cdot ws^2 \cdot C)$ instead of $O((HW)^2 C)$ for global attention: linear in image size.

## Common bugs
- `x.reshape(-1, ws, ws, C)` with no transpose: windows become strips.
- Reversing with a different permutation than the one used for partitioning.
- Rolling back in the same direction as the forward roll, which shifts everything by `2 * shift`.

## Interview follow-ups
- Build Swin's shifted-window attention mask: label each token by which of the 9 regions it came from before the roll, partition the labels, and forbid attention between different labels.
- How does Swin downsample between stages? (Patch merging: concatenate each 2x2 group of tokens and project 4C to 2C.)
"""
