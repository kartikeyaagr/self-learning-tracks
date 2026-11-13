META = dict(title='Four ways to count selections', tags=['Probability'], diff='Easy', runtime='python',
            lessons=['mlmaths:10.05'], related=['binomial-distribution'])

STATEMENT = r"""
Choosing $k$ items from $n$ distinct items can be counted four ways, depending on whether **order** matters and whether an item may be picked **more than once**:

| | order matters | order ignored |
| --- | --- | --- |
| no repeats | $P(n,k) = \frac{n!}{(n-k)!}$ | $\binom{n}{k} = \frac{n!}{k!\,(n-k)!}$ |
| repeats allowed | $n^k$ | $\binom{n+k-1}{k}$ |

Write `count_selections(n, k)` that returns the tuple `(perms, combs, perms_rep, combs_rep)` of four Python ints, in that order.

## Requirements
- `n` and `k` are non-negative ints.
- If $k > n$, there are no selections without repeats, so `perms` and `combs` are `0`.
- Choosing $k = 0$ items always has exactly one way (the empty selection), even when $n = 0$.
- With $n = 0$ and $k > 0$, every count is `0`.
- Results must be exact integers, so use integer arithmetic, not floats. All test answers are below $2^{53}$.
"""

STARTER = '''def count_selections(n, k):
    """Return (perms, combs, perms_with_repeats, combs_with_repeats) as ints."""
    raise NotImplementedError
'''

SOLUTION = '''import math

def count_selections(n, k):
    # math.perm / math.comb already return 0 when k > n.
    combs_rep = 1 if k == 0 else math.comb(n + k - 1, k)   # comb(-1, 0) would raise for n = k = 0
    return math.perm(n, k), math.comb(n, k), n ** k, combs_rep
'''

HINTS = [
    'Python has exact integer helpers in `math`: `math.perm(n, k)` and `math.comb(n, k)`. Both return 0 when `k > n`.',
    'Repeats with order: each of the $k$ positions has $n$ choices independently. Repeats without order is the "stars and bars" count $\\binom{n+k-1}{k}$.',
    'Watch $n = 0, k = 0$: $\\binom{-1}{0}$ makes `math.comb` raise, but the answer is 1. Also note `0 ** 0 == 1` in Python, which is the right answer.',
]

TESTS = [
    (5, 2),
    (4, 4),
    (3, 5),
    (0, 0),
    (0, 3),
    (10, 0),
    (52, 5),
    (1, 7),
    (20, 10),
]

EXPLANATION = r"""
## Where the formulas come from
- **Ordered, no repeats:** $n$ choices for the first slot, $n-1$ for the second, down to $n-k+1$. That's $n!/(n-k)!$.
- **Unordered, no repeats:** each unordered set of $k$ shows up $k!$ times among the ordered ones, so divide by $k!$.
- **Ordered, repeats:** $k$ independent slots with $n$ choices each, $n^k$.
- **Unordered, repeats (multisets):** write the selection as $k$ stars split into $n$ bins by $n-1$ bars. Choosing where the $k$ stars go among $n+k-1$ positions gives $\binom{n+k-1}{k}$.

## Why exact integers
$52!$ has 68 digits. Computing $\binom{52}{5}$ as `factorial(52) / (factorial(5) * factorial(47))` with float division loses precision for larger inputs. `math.comb` multiplies and divides incrementally in exact integer arithmetic.

## Complexity
$O(k)$ big-integer multiplications for `perm` and `comb`.

## Common bugs
- Off-by-one in the multiset formula: $\binom{n+k-1}{k}$, not $\binom{n+k}{k}$.
- Raising on $k > n$ instead of returning 0.
- Float division with `/` turning ints into floats.

## Interview follow-ups
- Probability that a 5-card hand is a flush (including straight flushes)? ($4\binom{13}{5} / \binom{52}{5}$.)
- How many ways can you distribute 10 identical GPUs across 4 jobs, each job getting at least one? (Give each job one first, then stars and bars: $\binom{6+4-1}{6} = 84$.)
"""
