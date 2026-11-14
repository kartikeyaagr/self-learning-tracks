import numpy as np

META = dict(title='Exponential moving average with bias correction', tags=['Time Series'], diff='Easy', runtime='numpy',
            related=['adam-step', 'rolling-std-features', 'exponential-moving-average-with-scan'])

STATEMENT = r"""
Write `ema(x, beta, bias_correct)` for a 1-D float array:

$$m_0 = 0, \qquad m_t = \beta\, m_{t-1} + (1-\beta)\, x_t, \quad t = 1, \dots, n.$$

If `bias_correct` is true, return $\hat m_t = m_t / (1 - \beta^t)$ instead (as Adam does). Return the array of length $n$ for $t = 1..n$. NaN entries in `x` are skipped: the state is carried unchanged, the output at that step repeats the previous output (or is NaN if no value has been seen), and the step does **not** count towards $t$ in the bias correction.
"""

STARTER = '''import numpy as np

def ema(x, beta, bias_correct):
    """Return the EMA of x (length n)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def ema(x, beta, bias_correct):
    x = np.asarray(x, dtype=float)
    out = np.empty(len(x))
    m, t, last = 0.0, 0, np.nan
    for i, v in enumerate(x):
        if not np.isnan(v):
            t += 1
            m = beta * m + (1 - beta) * v
            last = m / (1 - beta ** t) if bias_correct else m
        out[i] = last
    return out
'''

HINTS = [
    'Keep the running state m and a counter t of observed (non-NaN) values.',
    'Starting from m = 0 biases early values towards zero; dividing by 1 - beta^t removes that bias exactly when all inputs are equal.',
    'For a NaN input, do not update m or t; just repeat the previous output.',
]

TESTS = [
    (np.array([1.0, 1.0, 1.0]), 0.9, False),
    (np.array([1.0, 1.0, 1.0]), 0.9, True),
    (np.array([0.0, 10.0, 0.0, 10.0]), 0.5, False),
    (np.array([np.nan, 2.0, np.nan, 4.0]), 0.8, True),
    (np.array([np.nan, np.nan]), 0.5, False),
    (np.linspace(0, 1, 6), 0.99, True),
]

EXPLANATION = r"""
## Effective window
An EMA weights the value from $k$ steps ago by $(1-\beta)\beta^k$. The weights sum to (nearly) 1 and fall off geometrically, so the average covers roughly the last $1/(1-\beta)$ points: about 10 for $\beta = 0.9$, about 100 for $0.99$.

## Bias correction
With $m_0 = 0$ the first outputs are pulled towards zero: after one step $m_1 = (1-\beta)x_1$. The sum of the weights after $t$ steps is $1 - \beta^t$, so dividing by it gives a proper weighted average. Adam applies exactly this to its moment estimates.

## Where EMAs appear
Smoothed metrics on dashboards, momentum in optimisers, EMA copies of model weights for evaluation, and streaming features that need constant memory.

## Interview follow-ups
- How do you convert a desired half-life in steps into $\beta$? ($\beta = 0.5^{1/h}$.)
"""
