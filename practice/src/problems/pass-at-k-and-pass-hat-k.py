META = dict(title='Agent reliability: pass@k, pass^k and cost per success', tags=['Metrics & Evaluation', 'Language Models'], diff='Medium', runtime='python',
            lessons=['aieng:10.05'], related=['pass-at-k-estimator', 'evaluate-agent-trajectory', 'permutations-and-combinations'])

STATEMENT = r"""
You ran an agent several times on every task of an eval suite. Each trial is `True` (task solved), `False` (agent failed) or `None` (**harness error**: the sandbox crashed or the provider rate-limited, so the trial says nothing about the agent). Write `agent_reliability(trials, costs, k)`.

- `trials[t]` is the list of trial results for task `t`; `costs[t]` is the matching list of dollar costs (harness errors still cost money).
- For each task, drop the `None` trials. Let $n$ be the number of remaining trials and $c$ the number of `True`.
- A task with $n < k$ cannot be scored at this $k$: list its index in `rerun` and leave it out of every pass metric and bucket below.

For the scored tasks, average the unbiased per-task estimators

$$\text{pass@}1 = \frac{c}{n}, \qquad \text{pass@}k = 1 - \binom{n-c}{k} \Big/ \binom{n}{k}, \qquad \text{pass}^k = \binom{c}{k} \Big/ \binom{n}{k}$$

(with $\binom{a}{b} = 0$ when $b > a$), and put each scored task in one bucket: `solid` if $c = n$, `broken` if $c = 0$, else `flaky`.

`cost_per_success` is the total cost of **all** trials (every task, errors included) divided by the total number of `True` trials, or `None` if there are none. `errors` is the total number of `None` trials.

Return `{"pass@1": ..., "pass@k": ..., "pass^k": ..., "solid": int, "flaky": int, "broken": int, "rerun": [task indices], "errors": int, "cost_per_success": ...}`. The tests always leave at least one scored task.
"""

STARTER = '''def agent_reliability(trials, costs, k):
    """Reliability summary of a multi-trial agent eval."""
    rates = [sum(1 for x in ts if x) / len(ts) for ts in trials]
    m = len(rates)
    total = sum(sum(c) for c in costs)
    wins = sum(sum(1 for x in ts if x) for ts in trials)
    return {"pass@1": sum(rates) / m,
            "pass@k": sum(1 - (1 - p) ** k for p in rates) / m,
            "pass^k": sum(p ** k for p in rates) / m,
            "solid": sum(p == 1 for p in rates), "flaky": sum(0 < p < 1 for p in rates),
            "broken": sum(p == 0 for p in rates), "rerun": [], "errors": 0,
            "cost_per_success": total / wins if wins else None}
'''

SOLUTION = '''from math import comb

def agent_reliability(trials, costs, k):
    p1, pk, phk, rerun = [], [], [], []
    buckets = {"solid": 0, "flaky": 0, "broken": 0}
    for t, ts in enumerate(trials):
        valid = [x for x in ts if x is not None]          # harness errors say nothing about the agent
        n, c = len(valid), sum(valid)
        if n < k:
            rerun.append(t)
            continue
        p1.append(c / n)
        pk.append(1 - comb(n - c, k) / comb(n, k))         # comb(a, b) is 0 when b > a
        phk.append(comb(c, k) / comb(n, k))
        buckets["solid" if c == n else "broken" if c == 0 else "flaky"] += 1
    wins = sum(x is True for ts in trials for x in ts)
    total = sum(sum(cs) for cs in costs)
    m = len(p1)
    return {"pass@1": sum(p1) / m, "pass@k": sum(pk) / m, "pass^k": sum(phk) / m, **buckets,
            "rerun": rerun, "errors": sum(x is None for ts in trials for x in ts),
            "cost_per_success": total / wins if wins else None}
'''

HINTS = [
    'Filter out `None` before counting: a crashed sandbox is neither a success nor a failure, but its cost is real.',
    'pass^k is the chance that k trials drawn without replacement from the n are all successes: C(c, k) / C(n, k). `math.comb` returns 0 when k > c, which handles the edge cases.',
    'The plug-in forms 1 - (1 - c/n)^k and (c/n)^k treat draws as with replacement and are biased for small n; with n = 5 and c = 3, (0.6)^2 = 0.36 but the unbiased pass^2 is 0.3.',
]

import random as _random

def _gen(seed, tasks, n):
    r = _random.Random(seed)
    trials, costs = [], []
    for _ in range(tasks):
        p = r.choice([0.0, 0.3, 0.6, 0.9, 1.0])
        ts = [None if r.random() < 0.08 else r.random() < p for _ in range(n)]
        trials.append(ts)
        costs.append([round(r.uniform(0.05, 0.6), 2) for _ in range(n)])
    return trials, costs

_T = [[True] * 5, [True, False, True, False, True], [False, False, True, False, False]]
_C = [[0.1] * 5, [0.2] * 5, [0.3] * 5]
_g1, _g2 = _gen(5, 10, 5), _gen(17, 25, 8)

TESTS = [
    (_T, _C, 2),
    (_T, _C, 3),
    ([[True, None, False], [None, None, True], [False, False, False]], [[0.5, 0.1, 0.5], [0.1, 0.1, 0.4], [0.2, 0.2, 0.2]], 2),
    ([[False, False], [False, None]], [[1.0, 1.0], [1.0, 1.0]], 1),
    ([[True, True, True, True], [True, True, True, False]], [[0.25] * 4, [0.25] * 4], 4),
    (_g1[0], _g1[1], 3),
    (_g2[0], _g2[1], 4),
]

EXPLANATION = r"""
## Two questions from the same counts
pass@k asks whether at least one of k runs succeeds; it is the right number when a verifier (tests, a checker) lets you retry and keep the success. pass^k, introduced by tau-bench, asks whether all k runs succeed; it is the right number when a user gets one run and nothing checks it. In the first test the suite is 77% at pass@2 and 43% at pass^2 from the same 15 trials.

## Why the binomial forms
With $n$ trials of which $c$ succeeded, the chance that $k$ draws without replacement are all successes is $\binom{c}{k}/\binom{n}{k}$, an unbiased estimate of $p^k$. The plug-in $(c/n)^k$ is biased upward for small $n$ (by Jensen's inequality), so it flatters consistency exactly when you have few trials.

## Harness errors
Counting a crashed sandbox as a failure blames the agent for the environment; dropping it silently hides an infrastructure problem and can leave a task with too few trials to score. So: exclude it from $n$, keep its cost, count it, and rerun any task that falls below $k$ valid trials.

## Buckets and cost
Solid tasks are regression tests; broken ones are capability gaps or broken tasks; flaky ones are where reliability work pays. Cost per success divides all spend, including failed and errored trials, by successes, the honest unit for comparing a cheap unreliable agent with an expensive reliable one.

## Interview follow-ups
- How many trials per task do you need before pass^5 is meaningful? (At least 5 valid ones per task, and more tasks matter more than more trials for the error bar, because trials of one task are correlated.)
"""
