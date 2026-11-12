import numpy as np

META = dict(title='Inclusive float range', tags=['Data Processing'], diff='Easy', runtime='numpy',
            lessons=['air-fanout-v1:03.07'], related=['zeros-and-ones', 'tile-and-diff'])

STATEMENT = r"""
`np.arange(start, stop, step)` excludes `stop`, and with a float step its length depends on rounding error: `np.arange(1.0, 1.3, 0.1)` returns **four** values, the last one being about `1.3`. That is why grids with float steps are usually built from an integer count instead.

Write `inclusive_range(start, stop, step)` that returns the grid `start, start + step, start + 2*step, ...` up to and **including** `stop` when `stop` lies on the grid.

## The rule
The number of points is

$$n = \left\lfloor \frac{\text{stop} - \text{start}}{\text{step}} + 10^{-9} \right\rfloor + 1$$

The small slack absorbs rounding error, so `(0.3 - 0.0) / 0.1 = 2.9999999999999996` still counts as 3 steps. Point $i$ is $\text{start} + i \cdot \text{step}$ for $i = 0, \dots, n-1$.

## Details
- `start`, `stop`, `step` are Python floats or ints with `step > 0` and `stop >= start`.
- Return a `float64` array of shape `(n,)`. When `stop == start` it is `[start]`.
- If `stop` does not lie on the grid, the last point is the largest grid point below it: `inclusive_range(0, 0.95, 0.1)` ends at `0.9`.
"""

STARTER = '''import numpy as np

def inclusive_range(start, stop, step):
    """Grid start, start+step, ... including stop when it lies on the grid. float64 (n,)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def inclusive_range(start, stop, step):
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    # Integer counter times step: no error accumulates, unlike repeatedly adding step.
    return start + step * np.arange(n, dtype=np.float64)
'''

HINTS = [
    'Work out the number of points first, using the formula in the statement. After that, a float step is no longer involved in deciding the length.',
    'Once you have `n`, either `start + step * np.arange(n)` or `np.linspace(start, start + (n - 1) * step, n)` gives the grid.',
    '`n = int(np.floor((stop - start) / step + 1e-9)) + 1`. Then build the points from an integer `arange`.',
]

TESTS = [
    (0.0, 1.0, 0.25),
    (1.0, 1.3, 0.1),
    (0.0, 0.95, 0.1),
    (5, 5, 1),
    (-2.0, 2.0, 0.5),
    (0.0, 1e6, 1e5),
    (0.1, 0.7, 0.2),
    (-3.7, -3.7 + 0.3 * 17, 0.3),
]

EXPLANATION = r"""
## Why it works
`np.arange` decides its length as `ceil((stop - start) / step)`. With float inputs that ratio carries rounding error, so a value meant to be exactly 3 may come out as `3.0000000000000004` (one extra point) or `2.9999999999999996` (one fewer). Computing the count yourself, with an explicit slack, makes the length predictable. Generating the points as `start + i * step` from an integer counter keeps each point within one rounding error of its exact value; adding `step` repeatedly would let errors accumulate.

`np.linspace(a, b, n)` is the other safe tool. It takes a **count**, includes both endpoints by default, and is the right call when you know how many points you want rather than the spacing.

## Complexity
$O(n)$ time and memory.

## Common bugs
- `np.arange(start, stop + step, step)`: this sometimes produces one point too many, landing just past `stop`.
- Using `round` instead of `floor` for the count. That rounds `0.95 / 0.1 = 9.5` up to 10 steps and overshoots `stop`.
- Accumulating `x += step` in a loop. After a million steps the drift is visible.

## Interview follow-ups
- When would you want `np.linspace(..., endpoint=False)`? (Periodic grids such as angles on $[0, 2\pi)$, where the endpoint duplicates the start.)
- How do you build log-spaced learning-rate candidates? (`np.logspace(-5, -1, 9)` or `10 ** np.linspace(-5, -1, 9)`.)
"""
