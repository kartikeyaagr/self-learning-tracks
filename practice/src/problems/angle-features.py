import numpy as np

META = dict(title='Cyclical time features', tags=['Feature Engineering', 'Time Series'], diff='Easy', runtime='numpy',
            related=['normalized-difference', 'sinusoidal-positional-encoding', 'tile-and-diff'])

STATEMENT = r"""
Hour of day, day of week and compass heading wrap around: 23:00 is close to 00:00, but as raw numbers they are 23 apart. Mapping each value onto a circle fixes that:

$$\left(\sin\frac{2\pi v}{P},\; \cos\frac{2\pi v}{P}\right)$$

where $P$ is the period (24 for hours, 7 for weekdays, 360 for degrees). Values one period apart map to the same point.

Write `cyclic_features(values, period)`.

## Input
- `values`: numeric array of any shape (ints or floats, possibly negative or beyond one period).
- `period`: a positive Python float or int.

## Output
A `float64` array of shape `values.shape + (2,)`: `[..., 0]` holds the sines and `[..., 1]` the cosines.

## Example
`cyclic_features(np.array([0, 6, 12]), 24)` gives approximately `[[0, 1], [1, 0], [0, -1]]` (up to rounding near zero).
"""

STARTER = '''import numpy as np

def cyclic_features(values, period):
    """Return values.shape + (2,) array with sin and cos of 2*pi*values/period."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def cyclic_features(values, period):
    theta = 2 * np.pi * np.asarray(values, dtype=float) / period
    # Stack on a new last axis so each value gets its own (sin, cos) pair.
    return np.stack([np.sin(theta), np.cos(theta)], axis=-1)
'''

HINTS = [
    '`np.sin` and `np.cos` work element-wise on whole arrays. Convert the values to an angle in radians first.',
    'The angle is `2 * np.pi * values / period`.',
    'Combine the two results with `np.stack([s, c], axis=-1)`, which adds a new last axis of length 2. `np.concatenate` would join them along an existing axis instead.',
]

_r = np.random.default_rng(59)
TESTS = [
    (np.array([0, 6, 12, 18]), 24),
    (np.array([[0.0, 90.0], [180.0, 270.0]]), 360),
    (np.array([23, 0, 1]), 24),
    (np.array([-1, 6, 13]), 7),
    (np.array([3.5]), 1),
    (np.array([1e6]), 24),
    (_r.uniform(0, 24, size=(3, 4)), 24.0),
    (_r.integers(0, 366, size=10), 365.25),
]

EXPLANATION = r"""
## Why it works
`np.sin` and `np.cos` are ufuncs: they apply to every element of an array of any shape in one call. Dividing by the period and multiplying by $2\pi$ turns "fraction of a cycle" into an angle, and `np.stack(..., axis=-1)` puts the two coordinates side by side in a new trailing axis, so an input of shape `(B, T)` becomes `(B, T, 2)`.

Two coordinates are needed because one is ambiguous: $\sin$ alone gives 03:00 and 09:00 the same value ($\sin\frac{\pi}{4} = \sin\frac{3\pi}{4}$). The pair `(sin, cos)` identifies the angle uniquely, and the Euclidean distance between two encoded points grows smoothly with their circular distance.

## Complexity
$O(N)$ time and memory.

## Common bugs
- Forgetting $2\pi$ and passing `values / period` straight into `sin` (radians vs fractions of a cycle).
- Using `np.concatenate`, which produces shape `(2N,)` or doubles an existing axis instead of adding a new one.
- Integer division in older code (`values // period`), which throws away the fractional position.

## Interview follow-ups
- How do you recover the original value from the pair? (`np.arctan2(s, c) * period / (2 * np.pi) % period`.)
- Transformer positional encodings use the same idea at many frequencies at once. Why several frequencies instead of one? (A single period cannot tell positions a whole period apart; mixing geometric frequencies gives every position a unique code.)
"""
