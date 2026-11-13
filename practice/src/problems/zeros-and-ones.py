import numpy as np

META = dict(title='Border mask of width k', tags=['Data Processing', 'Computer Vision'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:03.07'], related=['basic-indexing', 'create-arrays-from-lists'])

STATEMENT = r"""
Image pipelines often need a mask that marks the pixels near the edge, for example to ignore a convolution's unreliable border or to draw a frame.

Write `border_mask(h, w, k)` that returns an `int64` array of shape `(h, w)` where a cell is `1` if it lies within `k` cells of **any** edge and `0` otherwise. Precisely, cell $(i, j)$ is 1 exactly when

$$\min(i,\; j,\; h-1-i,\; w-1-j) < k$$

## Details
- `h, w >= 1` and `k >= 0` are Python ints.
- `k = 0` means no border: the result is all zeros.
- If the border is thick enough to cover everything (for example `k = 3` on a `4 x 4` grid), the result is all ones.

## Example
`border_mask(4, 5, 1)`:

```
[[1, 1, 1, 1, 1],
 [1, 0, 0, 0, 1],
 [1, 0, 0, 0, 1],
 [1, 1, 1, 1, 1]]
```
"""

STARTER = '''import numpy as np

def border_mask(h, w, k):
    """Return an int64 (h, w) array: 1 within k cells of an edge, 0 elsewhere."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def border_mask(h, w, k):
    mask = np.ones((h, w), dtype=np.int64)
    # Zero the interior. When 2k >= h (or w) the slice k:h-k is empty, so the mask stays all ones.
    mask[k:h - k, k:w - k] = 0
    return mask
'''

HINTS = [
    'It is easier to describe the interior than the border. Which rows and columns are at least `k` cells from every edge?',
    'Start from `np.ones((h, w), dtype=np.int64)` and set the interior rectangle to 0 with one slice assignment.',
    'The interior is `mask[k:h-k, k:w-k]`. Check what that slice is when `k = 0`, and when `k` is larger than `h / 2`.',
]

_r = np.random.default_rng(3)
TESTS = [
    (4, 5, 1),
    (5, 5, 2),
    (3, 7, 0),
    (1, 1, 1),
    (6, 4, 10),
    (2, 9, 1),
    (8, 8, 3),
    (int(_r.integers(5, 12)), int(_r.integers(5, 12)), int(_r.integers(1, 4))),
]

EXPLANATION = r"""
## Why it works
`np.ones` builds a canvas that is already "all border", and one slice assignment carves out the interior: rows `k .. h-k-1` and columns `k .. w-k-1`. Slices clamp silently, so the edge cases need no special code. With `k = 0` the slice `0:h` covers everything (all zeros). When `k` is large, the start index is at or past the stop index, the slice is empty, and nothing is zeroed (all ones).

The `min(...) < k` formula in the statement is a loop-free alternative:

```
i = np.arange(h)[:, None]; j = np.arange(w)
d = np.minimum(np.minimum(i, h - 1 - i), np.minimum(j, w - 1 - j))
mask = (d < k).astype(np.int64)
```

## Complexity
$O(hw)$ time and memory. The slice assignment writes only the interior.

## Common bugs
- Using `mask[k:-k, k:-k] = 0`. With `k = 0` this becomes `mask[0:0]`, an empty slice, so you get all ones instead of all zeros.
- Returning booleans or floats when the statement asks for `int64`. The grader accepts them here, but in a real pipeline dtype surprises break later concatenations.

## Interview follow-ups
- How would you build a mask of the cells within distance `k` of a single pixel, using broadcasting? (Compare `np.abs(i - r) + np.abs(j - c)` with `k` for the L1 diamond, or use the Euclidean distance for a disc.)
- What do `np.zeros_like`, `np.ones_like` and `np.full_like` add over `np.zeros(shape)`? (They copy the shape and dtype of an existing array.)
"""
