import numpy as np

META = dict(title='Mean, median and mode in one pass', tags=['Probability and Statistics'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:10.15'], related=['sample-variance-and-standard-deviation', 'percentiles-quantiles'])

STATEMENT = r"""
The three classic summaries of "where the data sits" answer different questions:

- the **mean** $\bar x = \frac1n \sum_i x_i$ is the balance point,
- the **median** is the middle value once the data is sorted,
- the **mode** is the value that occurs most often.

Write `center_summary(x)` that returns the tuple `(mean, median, mode)` as three Python floats.

## Conventions
- `x` is a non-empty 1-D array of ints or floats (negative values allowed).
- For an **even** number of values, the median is the average of the two middle values.
- If several values tie for the highest count, the mode is the **smallest** of them. So when every value is distinct, the mode is the minimum.
"""

STARTER = '''import numpy as np

def center_summary(x):
    """x: non-empty 1-D array. Return (mean, median, mode) as floats."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def center_summary(x):
    x = np.asarray(x, dtype=float)
    values, counts = np.unique(x, return_counts=True)   # values come back sorted ascending
    mode = values[np.argmax(counts)]                     # argmax returns the first max, i.e. the smallest tied value
    return float(x.mean()), float(np.median(x)), float(mode)
'''

HINTS = [
    'NumPy has `mean` and `median` built in. The mode is the only piece you have to build.',
    '`np.unique(x, return_counts=True)` gives the distinct values and how often each occurs.',
    '`np.unique` sorts the values, and `np.argmax` returns the first maximum, so `values[np.argmax(counts)]` already applies the "smallest tied value" rule.',
]

_r = np.random.default_rng(21)
TESTS = [
    (np.array([1, 2, 2, 3, 7]),),
    (np.array([4.0, 1.0, 3.0, 2.0]),),
    (np.array([5.5]),),
    (np.array([-3, -3, 2, 2, 9]),),
    (np.array([0.5, 0.5, 0.5, 100.0]),),
    (np.array([10, 20, 20, 30, 30, 40]),),
    (_r.integers(-5, 6, size=40),),
    (np.round(_r.normal(50, 10, size=25), 1),),
]

EXPLANATION = r"""
## Why the three can disagree
The mean uses every value's magnitude, so one extreme value drags it (see `[0.5, 0.5, 0.5, 100]`: mean 25.4, median 0.5). The median only uses the ordering, so it is robust to outliers. The mode only uses counts, so it is the natural summary for categorical or heavily discretised data.

For a right-skewed distribution such as incomes or latencies, you typically see mode < median < mean.

## Complexity
The mean is $O(n)$. The median needs a selection step: `np.median` uses a partition, which is $O(n)$ on average. `np.unique` sorts, so the mode is $O(n \log n)$. A hash map (`collections.Counter`) gets the counts in $O(n)$ but then needs its own tie rule.

## Common bugs
- Taking the lower middle element for an even-length median instead of averaging the two.
- Using `Counter.most_common(1)`, which breaks ties by first appearance, not by value.
- Returning NumPy scalars or an array instead of three floats.

## Interview follow-ups
- How would you keep a running median over a stream? (Two heaps: a max-heap for the lower half, a min-heap for the upper half.)
- Which of the three minimises the sum of squared errors, and which minimises the sum of absolute errors? (Mean, then median.)
"""
