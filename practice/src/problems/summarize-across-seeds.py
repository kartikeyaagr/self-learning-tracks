import numpy as np

META = dict(title='Summarise RL results across seeds', tags=['Reinforcement Learning', 'Probability and Statistics'], diff='Easy', runtime='numpy')

STATEMENT = r"""
Write `summarize_seeds(returns_by_seed, last_n)`. Each entry of `returns_by_seed` is one seed's list of episode returns, in order.

1. Score each seed by the mean of its last `last_n` episodes (or all of them, if it has fewer).
2. Over the per-seed scores, report:
   - `mean`, `std` (sample standard deviation, `ddof=1`; 0.0 for one seed), `stderr` = std / $\sqrt{n}$;
   - `min`, `max`;
   - `iqm`: the interquartile mean, the mean after removing `int(0.25 * n)` scores from each end of the sorted list.
3. Also return `scores`, the per-seed scores in input order.
"""

STARTER = '''import numpy as np

def summarize_seeds(returns_by_seed, last_n):
    """Per-seed final performance and robust summaries across seeds."""
    best = [max(r) for r in returns_by_seed]
    return {"scores": best, "mean": float(np.mean(best)), "std": float(np.std(best)), "stderr": 0.0,
            "min": float(min(best)), "max": float(max(best)), "iqm": float(np.mean(best))}
'''

SOLUTION = '''import numpy as np

def summarize_seeds(returns_by_seed, last_n):
    s = np.array([float(np.mean(r[-last_n:])) for r in returns_by_seed])
    n = len(s)
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    k = int(0.25 * n)
    srt = np.sort(s)
    mid = srt[k:n - k] if n - 2 * k > 0 else srt
    return {"scores": s.tolist(), "mean": float(s.mean()), "std": std, "stderr": std / np.sqrt(n),
            "min": float(s.min()), "max": float(s.max()), "iqm": float(mid.mean())}
'''

HINTS = [
    'Score each seed by its final performance, not its best episode; the best episode is a noisy maximum.',
    'With n = 5 seeds, int(0.25 * 5) = 1, so the IQM drops the single best and single worst seed.',
]

_r = np.random.default_rng(0)

TESTS = [
    ([[0, 1, 2, 3], [0, 2, 4, 6]], 2),
    ([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [100.0, 100.0]], 1),
    ([[5.0, 6.0, 7.0]], 10),
    ([list(_r.normal(i, 1, 20)) for i in range(8)], 5),
    ([[0.0], [0.0], [10.0], [10.0]], 1),
]

EXPLANATION = r"""
## Why seeds matter so much in RL
Two runs of the same algorithm with different seeds can differ more than two different algorithms. Reporting one seed, or the best of several, is how most irreproducible RL results happen.

## What to report
- The mean with a standard error (or a bootstrap confidence interval) over at least 5 seeds, ideally 10 or more.
- The interquartile mean, recommended by Agarwal et al. (2021), which ignores the extreme seeds and is far less sensitive to one lucky or broken run than the mean, while needing fewer runs than the median. The second test shows one outlier seed moving the mean from 2.5 to 22 while the IQM stays at 3.
- Final performance (the last N episodes), not the best episode seen during training.

## Interview follow-ups
- Two methods' confidence intervals overlap slightly. What would you run before claiming one is better?
"""
