import numpy as np

META = dict(title='Normalised difference index', tags=['Feature Engineering', 'Computer Vision'], diff='Easy',
            runtime='numpy', lessons=['air-fanout-v1:01.10'], related=['angle-features', 'row-scaling'])

STATEMENT = r"""
Satellite imagery measures each pixel in several spectral bands. A **normalised difference** of two bands,

$$\text{ND} = \frac{a - b}{a + b}$$

is a classic per-pixel feature (vegetation reflects strongly in near-infrared and weakly in red, so ND of those two bands highlights plants). The ratio lies in $[-1, 1]$ and cancels overall brightness.

Write `normalized_difference(a, b)`.

## Input
`a`, `b`: non-negative arrays (ints or floats) of the **same shape**, any number of dimensions.

## Output
A `float64` array of that shape with $(a-b)/(a+b)$ element-wise. Where $a + b = 0$ (both bands zero), the result is `0.0`.

Compute it without Python loops, and without producing `NaN` or division warnings along the way.

## Example
`a = [[0.8, 0.0], [0.3, 0.5]]`, `b = [[0.2, 0.0], [0.3, 0.0]]` gives `[[0.6, 0.0], [0.0, 1.0]]`.
"""

STARTER = '''import numpy as np

def normalized_difference(a, b):
    """(a - b) / (a + b) element-wise, 0.0 where a + b == 0. float64, same shape."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def normalized_difference(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    total = a + b
    out = np.zeros_like(total)
    # Only divide where the denominator is non-zero; other cells keep the 0.0 already in `out`.
    np.divide(a - b, total, out=out, where=total != 0)
    return out
'''

HINTS = [
    'Arithmetic on whole arrays is element-wise, so `(a - b) / (a + b)` is already one vectorised expression. The only problem is the zero denominator.',
    '`np.where(total == 0, 0.0, (a - b) / total)` gives the right values, but it still evaluates `0/0` everywhere first and warns. Avoid computing those cells at all.',
    'Pre-fill `out = np.zeros_like(total)` and call `np.divide(a - b, total, out=out, where=total != 0)`.',
]

_r = np.random.default_rng(53)
TESTS = [
    (np.array([[0.8, 0.0], [0.3, 0.5]]), np.array([[0.2, 0.0], [0.3, 0.0]])),
    (np.array([3, 0, 1]), np.array([1, 0, 3])),
    (np.array([0.0]), np.array([0.0])),
    (np.array([0.0, 0.0]), np.array([2.0, 1e-300])),
    (np.array([1e300, 1e-300]), np.array([1e300, 1e-300])),
    (np.zeros((2, 2, 2)), np.ones((2, 2, 2))),
    (_r.uniform(0, 1, size=(4, 5)), _r.uniform(0, 1, size=(4, 5))),
    (_r.integers(0, 3, size=(6, 6)), _r.integers(0, 3, size=(6, 6))),
]

EXPLANATION = r"""
## Why it works
NumPy's arithmetic operators are **ufuncs**: they apply element by element across whole arrays in compiled code, so the formula translates directly. Every ufunc also accepts `out=` (write into an existing array) and `where=` (only compute where the mask is true). Pre-filling `out` with the fallback value and masking the division means `0/0` is never evaluated, so there is no `RuntimeWarning` and no NaN to clean up afterwards.

Converting to float first matters for integer inputs: `a - b` on unsigned image data (`uint8`, `uint16`) wraps around instead of going negative.

## Complexity
$O(N)$ time and a few temporaries of size $N$.

## Common bugs
- `np.where(total == 0, 0, (a - b) / total)`: right answer, but both branches are computed in full, so the warning still fires.
- Integer arithmetic on unsigned bands: `np.uint8(1) - np.uint8(3)` is `254`.
- Adding a tiny epsilon to the denominator. It changes the answer for very small but legitimate values (try `a = b = 1e-300`).

## Interview follow-ups
- `1e300 + 1e300` overflows to `inf`. How would you make the formula safe for huge values? (Divide `a` and `b` by `max(a, b)` first; the ratio is scale-invariant.)
- How would you compute the index for a whole `(H, W, C)` image given two band indices? (`normalized_difference(img[..., i], img[..., j])`; the ellipsis keeps all leading axes.)
"""
