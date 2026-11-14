import numpy as np

META = dict(title='Fitting an exponential lifetime with censoring', tags=['Probability', 'Probability and Statistics'], diff='Medium',
            runtime='numpy', lessons=['mlmaths:10.35', 'mlmaths:12.10'], related=['poisson-distribution', 'maximum-likelihood-estimation'])

STATEMENT = r"""
Exponential waiting times with rate $\lambda$ have density $f(t) = \lambda e^{-\lambda t}$ and survival function $S(t) = P(T > t) = e^{-\lambda t}$, so the mean is $1/\lambda$ and the median is $\ln 2 / \lambda$.

You track $n$ GPU jobs. For each you record a time `times[i]` and a flag `observed[i]`:

- `True`: the job **crashed** at that time (an event),
- `False`: the job was still running when you stopped watching (right-**censored**). You only know $T_i > t_i$.

The likelihood multiplies $f(t_i)$ for events and $S(t_i)$ for censored jobs. Maximising it gives

$$\hat\lambda = \frac{\text{number of events}}{\sum_{i} t_i}$$

where the sum runs over **all** jobs, censored or not.

Write `fit_exponential(times, observed, t)` that returns the tuple `(rate, mean, median, survival)`:

- `rate`, `mean`, `median`: Python floats for $\hat\lambda$, $1/\hat\lambda$ and $\ln 2/\hat\lambda$,
- `survival`: float array of the same shape as `t`, the fitted $S(t) = e^{-\hat\lambda t}$.

## Requirements
- `times`: 1-D positive float array, `observed`: bool array of the same length, $n \ge 1$. `t`: 1-D float array of non-negative times.
- If no job crashed, $\hat\lambda = 0$: return `rate = 0.0`, `mean = median = inf` and a survival of all ones.
"""

STARTER = '''import numpy as np

def fit_exponential(times, observed, t):
    """Return (rate, mean, median, survival_at_t)."""
    raise NotImplementedError
'''

SOLUTION = '''import math
import numpy as np

def fit_exponential(times, observed, t):
    times = np.asarray(times, dtype=float)
    events = int(np.count_nonzero(observed))
    rate = events / times.sum()                 # censored jobs add exposure time but no events
    survival = np.exp(-rate * np.asarray(t, dtype=float))
    if events == 0:
        return 0.0, math.inf, math.inf, survival
    return float(rate), float(1 / rate), float(math.log(2) / rate), survival
'''

HINTS = [
    'Write the log-likelihood: each event contributes $\\log\\lambda - \\lambda t_i$ and each censored job contributes $-\\lambda t_i$. Differentiate and set to 0.',
    'The total exposure $\\sum t_i$ includes the censored times. Dropping censored jobs, or treating their times as crash times, both bias the rate.',
    'With zero events the rate is 0, so the mean and median are infinite. Return `math.inf` instead of dividing by zero.',
]

TESTS = [
    (np.array([2.0, 3.0, 5.0]), np.array([True, True, True]), np.array([0.0, 1.0, 10.0])),
    (np.array([1.0, 4.0, 4.0, 4.0]), np.array([True, False, False, True]), np.array([2.0])),
    (np.array([10.0, 20.0]), np.array([False, False]), np.array([0.0, 5.0])),
    (np.array([0.25]), np.array([True]), np.array([0.25, 1.0])),
    (np.array([1e4, 2e4, 3e4]), np.array([True, False, True]), np.array([1e3, 1e5])),
    (np.array([0.5, 0.1, 0.9]), np.array([True, True, False]), np.array([])),
    (np.random.default_rng(9).exponential(4.0, size=60), np.random.default_rng(10).random(60) < 0.7,
     np.linspace(0.0, 20.0, 5)),
]

EXPLANATION = r"""
## Why censored jobs still count
A job that ran 4 hours without crashing is evidence of a **low** crash rate. In the likelihood it contributes $S(4) = e^{-4\lambda}$, which pushes $\lambda$ down. Taking logs and differentiating,
$$\frac{d}{d\lambda}\Big[d \log\lambda - \lambda \sum_i t_i\Big] = \frac{d}{\lambda} - \sum_i t_i = 0 \;\Rightarrow\; \hat\lambda = \frac{d}{\sum_i t_i}$$
where $d$ is the number of events: events per unit of exposure time. Ignoring censored jobs makes the system look much flakier than it is.

## Memorylessness
$P(T > s + t \mid T > s) = e^{-\lambda t} = P(T > t)$. A job that has survived 10 hours is as likely to crash in the next hour as a fresh one. That's what makes censoring so easy to handle here, and also why the exponential is often wrong for hardware, which wears out (use a Weibull).

## Complexity
$O(n + m)$.

## Common bugs
- Using the mean of only the observed times as $1/\hat\lambda$.
- Dividing the sum of all times by $n$ instead of by the number of events.
- Returning `nan` or raising for the no-event case.

## Interview follow-ups
- Requests arrive as a Poisson process with rate $\lambda$. What's the distribution of the gap between arrivals? (Exponential($\lambda$). The two models describe the same process.)
- How would you get a confidence interval for $\lambda$? ($2\lambda \sum t_i$ is $\chi^2_{2d}$ when every job is observed. A quick approximation is $\hat\lambda \pm 1.96\,\hat\lambda/\sqrt d$.)
"""
