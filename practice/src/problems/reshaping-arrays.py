import numpy as np

META = dict(title='Cut an image into flat patches', tags=['Computer Vision', 'Data Processing'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:03.04', 'air-fanout-v1:03.03'], related=['basic-indexing', 'quantize-and-frame'])

STATEMENT = r"""
Vision transformers do not read pixels one by one. They cut the image into non-overlapping `p x p` patches and flatten each patch into a vector. The cut can be done with reshapes alone, without copying pixels in a loop.

Write `patchify(img, p)` that splits a 2-D image into patches.

## Input
- `img`: array of shape `(H, W)`, any numeric dtype. `H` and `W` are multiples of `p`.
- `p`: patch side length, a Python int `>= 1`.

## Output
An array of shape `(H/p * W/p, p*p)` with the same dtype as `img`.
- Patches are ordered **row-major over the patch grid**: left to right along the top strip, then the next strip down.
- Each patch is flattened row-major: its first row, then its second row, and so on.

## Example
For `img = np.arange(16).reshape(4, 4)` and `p = 2`:

```
[[ 0,  1,  4,  5],    # top-left patch
 [ 2,  3,  6,  7],    # top-right patch
 [ 8,  9, 12, 13],    # bottom-left patch
 [10, 11, 14, 15]]    # bottom-right patch
```

Note that `img.reshape(-1, p*p)` is **not** the answer: it would give `[0, 1, 2, 3]` as the first "patch", which is a row of pixels.
"""

STARTER = '''import numpy as np

def patchify(img, p):
    """img: (H, W). Return (H//p * W//p, p*p) patches, row-major over the grid and within each patch."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def patchify(img, p):
    img = np.asarray(img)
    H, W = img.shape
    # (H, W) -> (row of patch, row inside patch, column of patch, column inside patch)
    blocks = img.reshape(H // p, p, W // p, p)
    # Bring the two patch-grid axes together, then the two within-patch axes, and flatten each pair.
    return blocks.transpose(0, 2, 1, 3).reshape(-1, p * p)
'''

HINTS = [
    'A reshape never changes the order of elements in memory; it only re-labels them. You need one reshape to split the axes, a transpose to reorder them, and a second reshape to merge them.',
    'Split each axis into (which patch, position inside the patch): `img.reshape(H // p, p, W // p, p)`. The axes are now (patch row, pixel row, patch col, pixel col).',
    'Swap the middle two axes with `.transpose(0, 2, 1, 3)` so both patch-grid axes come first, then `.reshape(-1, p * p)`.',
]

_r = np.random.default_rng(5)
TESTS = [
    (np.arange(16).reshape(4, 4), 2),
    (np.arange(12).reshape(2, 6), 2),
    (np.arange(6).reshape(2, 3), 1),
    (np.arange(9).reshape(3, 3), 3),
    (np.arange(24).reshape(4, 6), 2),
    (_r.integers(0, 256, size=(6, 9)), 3),
    (_r.normal(size=(8, 4)), 4),
]

EXPLANATION = r"""
## Why it works
In a C-ordered `(H, W)` array, the pixel at `(r, c)` sits at flat position `r*W + c`. Writing `r = R*p + i` and `c = C*p + j` (patch index `R, C`, offset `i, j`) shows that `reshape(H/p, p, W/p, p)` simply re-labels the same memory with axes `(R, i, C, j)`. No data moves. We want the output order `(R, C, i, j)`: patches outermost, pixels inside. `transpose(0, 2, 1, 3)` reorders the axes (returning a view with permuted strides), and the final reshape copies the data into that order and merges `(R, C)` and `(i, j)`.

## Complexity
$O(HW)$ time for the final copy. The first reshape and the transpose are free views.

## Common bugs
- Reshaping straight to `(-1, p*p)`. This takes consecutive pixels of a row, not a 2-D patch.
- Transposing to `(0, 2, 3, 1)` or a similar order. The shape still looks right, but each patch comes out flattened column-major.
- Assuming the result is a view. After a transpose the memory is no longer contiguous in the new order, so the last `reshape` must copy.

## Interview follow-ups
- Extend it to `(B, H, W, C)` images, flattening each patch as `(i, j, channel)`. (`reshape(B, H/p, p, W/p, p, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, -1, p*p*C)`.)
- How do you invert it (unpatchify)? (Apply the inverse permutation, which happens to be the same `(0, 2, 1, 3)`, between the two reshapes.)
"""
