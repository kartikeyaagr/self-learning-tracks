import numpy as np

META = dict(title='Estimate pass@k from samples', tags=['Language Models', 'Metrics & Evaluation'], diff='Easy', runtime='numpy')

STATEMENT = r"""
For each problem you drew `n` samples and `c` of them were correct. The unbiased estimate of pass@k (the chance that at least one of `k` samples is correct) is
$$\text{pass@}k = 1 - \binom{n-c}{k}\Big/\binom{n}{k}.$$

Write `pass_at_k(ns, cs, k)` for lists of per-problem `n` and `c`.

- Compute it without large binomials: if $n - c < k$ the value is 1; otherwise $1 - \prod_{i=n-c+1}^{n}\left(1 - k/i\right)$.
- Every problem has $n \ge k$.

Return a dict with `per_problem` (list) and `mean`.
"""

STARTER = '''import numpy as np

def pass_at_k(ns, cs, k):
    """Unbiased pass@k for each problem and its average."""
    vals = [1 - (1 - c / n) ** k for n, c in zip(ns, cs)]
    return {"per_problem": vals, "mean": float(np.mean(vals))}
'''

SOLUTION = '''import numpy as np

def pass_at_k(ns, cs, k):
    vals = []
    for n, c in zip(ns, cs):
        if n - c < k:
            vals.append(1.0)
        else:
            vals.append(float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))))
    return {"per_problem": vals, "mean": float(np.mean(vals))}
'''

HINTS = [
    'The plug-in estimate 1 - (1 - c/n)^k is biased: it treats the k draws as with replacement.',
    'The product form avoids computing binomials like C(200, 100), which overflow floats.',
    'With c = 0 the product is over an empty range and equals 1, so pass@k is 0.',
]

TESTS = [
    ([10], [0], 1),
    ([10], [3], 1),
    ([10], [3], 5),
    ([5, 5, 5], [0, 1, 5], 2),
    ([200, 200, 100], [1, 50, 99], 10),
    ([20, 20], [17, 2], 4),
]

EXPLANATION = r"""
## Why not just sample k
Drawing exactly k samples per problem and checking whether any is correct gives a very noisy estimate. Drawing $n \gg k$ and counting the ways to pick k without any correct one uses all n samples and is unbiased (Chen et al., 2021).

## Why the plug-in estimate is wrong
$1 - (1 - c/n)^k$ treats the k picks as independent draws with replacement. Without replacement, once you pick a wrong sample there are fewer wrong ones left, so the true chance of all-wrong is smaller. The starter underestimates pass@k; the gap is largest when k is close to n.

## What RL changes
Reinforcement learning on verifiable rewards (GRPO and similar) often raises pass@1 a lot while pass@k for large k moves much less or even falls. A common reading is that training sharpens the model onto answers it could already find, rather than teaching it new ones. Reporting both is how you see that.

## Interview follow-ups
- How many samples per problem do you need for pass@1 to have a standard error below one percentage point on a 500-problem benchmark?
"""
