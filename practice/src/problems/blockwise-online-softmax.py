import numpy as np

META = dict(title='Blockwise online softmax with masked blocks', tags=['GPU Programming', 'Attention'], diff='Hard',
            runtime='numpy', lessons=['ieng:14.02', 'aieng:06.03'],
            related=['online-softmax', 'flashattention-with-online-softmax', 'softmax-numerical-stability-and-log-sum-exp'])

STATEMENT = r"""
FlashAttention never holds a full row of attention scores. It visits the row one block of columns at a time and keeps only two running numbers per row: the maximum so far $m$ and the sum of exponentials so far $\ell$, **relative to that maximum**. When a later block raises the maximum, the old sum is rescaled.

Causal and padding masks make whole blocks `-inf`, and the running state must survive them without producing NaN.

Write `blockwise_softmax(x, block)`.

## Inputs
- `x`: `float64` `(M, N)`, entries finite or `-inf` (masked). A row may be entirely `-inf`.
- `block`: int $\ge 1$. Block $j$ covers columns $[jb, \min((j+1)b, N))$; there are $n_b = \lceil N / b \rceil$ blocks.

## Streaming update (per row)
Start with $m = -\infty$, $\ell = 0$. For each block $j$ in order:
1. Block statistics: $m_j = \max$ of the block, and $\ell_j = \sum e^{x - m_j}$ over the block, with $\ell_j = 0$ when $m_j = -\infty$.
2. Merge: $m' = \max(m, m_j)$; if $m' = -\infty$, keep $\ell = 0$; otherwise $\ell \leftarrow \ell\, e^{m - m'} + \ell_j\, e^{m_j - m'}$ (a term whose own max is $-\infty$ contributes 0). Then $m \leftarrow m'$.
3. Record $m$ and $\ell$ after this block.

After the last block, $\text{lse} = m + \log \ell$ (or $-\infty$ if $\ell = 0$), and $p = e^{x - \text{lse}}$, with fully masked rows giving all-zero probabilities.

## Output
A tuple `(run_max, run_sum, lse, probs)`: `run_max` and `run_sum` of shape `(M, n_b)` (the state after each block), `lse` `(M,)`, `probs` `(M, N)`. Process the row block by block: do not take the max of the full row up front. No warnings or NaN for any valid input.
"""

STARTER = '''import numpy as np

def blockwise_softmax(x, block):
    """Return (run_max (M, nb), run_sum (M, nb), lse (M,), probs (M, N))."""
    M, N = x.shape
    nb = -(-N // block)
    m = x.max(axis=1, keepdims=True)                 # TODO: not streaming; NaN on fully masked rows
    p = np.exp(x - m)
    l = p.sum(axis=1)
    lse = m[:, 0] + np.log(l)
    return np.repeat(m, nb, axis=1), np.repeat(l[:, None], nb, axis=1), lse, p / l[:, None]
'''

SOLUTION = '''import numpy as np

def _scaled(s, m_from, m_to):
    """s * exp(m_from - m_to), defined as 0 when m_from is -inf (that part held no mass)."""
    finite = np.isfinite(m_from)
    return np.where(finite, s * np.exp(np.where(finite, m_from, 0.0) - np.where(finite, m_to, 0.0)), 0.0)

def blockwise_softmax(x, block):
    M, N = x.shape
    nb = -(-N // block)
    m = np.full(M, -np.inf)
    l = np.zeros(M)
    run_max, run_sum = np.empty((M, nb)), np.empty((M, nb))
    for j in range(nb):
        xb = x[:, j * block:(j + 1) * block]
        mj = xb.max(axis=1)
        safe = np.where(np.isfinite(mj), mj, 0.0)              # dummy shift for all-masked rows
        lj = np.where(np.isfinite(mj), np.exp(xb - safe[:, None]).sum(axis=1), 0.0)
        m_new = np.maximum(m, mj)
        l = _scaled(l, m, m_new) + _scaled(lj, mj, m_new)      # if m_new = -inf both parts are 0
        m = m_new
        run_max[:, j], run_sum[:, j] = m, l

    alive = l > 0
    lse = np.where(alive, m + np.log(np.where(alive, l, 1.0)), -np.inf)
    shift = np.where(alive, lse, 0.0)[:, None]
    probs = np.where(alive[:, None], np.exp(x - shift), 0.0)   # exp(-inf) = 0 for masked entries
    return run_max, run_sum, lse, probs
'''

HINTS = [
    "Merging two partial states is $(m, \\ell) \\oplus (m_j, \\ell_j) = \\bigl(m',\\; \\ell e^{m - m'} + \\ell_j e^{m_j - m'}\\bigr)$ with $m' = \\max(m, m_j)$. It is associative, which is why blocks can also be merged in a tree across thread blocks.",
    'The NaN comes from $e^{-\\infty - (-\\infty)}$. Define the scaled term as 0 whenever its own max is $-\\infty$, and replace $-\\infty$ by a dummy finite value **inside** the exponent so NumPy never evaluates `-inf - -inf`.',
    'Finish with $\\text{lse} = m + \\log \\ell$ only for rows with $\\ell > 0$, and probabilities `exp(x - lse)`; for dead rows, return zeros directly.',
]

_r = np.random.default_rng(23)
_mask = _r.normal(size=(4, 10)) * 3
_mask[0, :4] = -np.inf                     # first block (b = 4) fully masked
_mask[1, 4:] = -np.inf                     # causal-like: later blocks masked
_mask[2, :] = -np.inf                      # dead row
_big = _r.normal(size=(3, 37)) * 4
_big[:, 30:] += 500.0                      # the max arrives late: forces rescaling of the old sum
TESTS = [
    (np.array([[1.0, 2.0, 3.0, 4.0]]), 2),
    (np.array([[0.0, -np.inf, 1.0], [-np.inf, -np.inf, 0.0]]), 1),
    (_mask, 4),
    (_big, 8),
    (_r.normal(size=(5, 16)), 16),
    (_r.normal(size=(2, 7)), 3),
    (np.full((2, 5), -np.inf), 2),
    (np.array([[1000.0, -1000.0, 1000.0, 0.0, 999.0]]), 2),
]

EXPLANATION = r"""
## Why it works
For any reference value $c$, $\sum_k e^{x_k} = e^{c} \sum_k e^{x_k - c}$. The state $(m, \ell)$ means "the row so far sums to $\ell\,e^{m}$". Changing the reference from $m$ to a larger $m'$ multiplies $\ell$ by $e^{m - m'} \le 1$, so nothing ever overflows, and the final $(m, \ell)$ equals the exact $(\max, \sum e^{x - \max})$ regardless of block size. Test 4's large values arrive in the last blocks, so the earlier sums are rescaled by about $e^{-500}$, underflowing harmlessly to 0.

## Masked blocks
A fully masked block has $m_j = -\infty$ and $\ell_j = 0$: it is the identity of the merge operator. Treating "max is $-\infty$" as "contributes nothing" handles a masked first block (test 3, row 0), masked later blocks (row 1), and dead rows (row 2, test 7), where the row's softmax is defined here as all zeros. FlashAttention kernels need the same rule; a NaN in one attention row would otherwise propagate into the gradients of every parameter.

## Complexity
$O(MN)$ time, one pass over the data for the statistics, and $O(M)$ extra state: the whole point is that the state does not grow with $N$. In attention, the same rescaling factor is also applied to the running output accumulator $\sum p\,v$, so the probabilities never need to be stored at all.

## Common bugs
- Rescaling only $\ell$ but not the output accumulator (in FlashAttention), or rescaling with $e^{m' - m}$ (inverted).
- Initialising $m$ with 0 or a large negative finite number: it works for typical rows but silently changes results for rows whose entries are all below it.
- Producing NaN for dead rows via $-\infty - (-\infty)$, often hidden until a padding-only sequence appears in a batch.

## Interview follow-ups
- FlashAttention-2 delays the division by $\ell$ to the very end. Why does that save work, and what is stored for the backward pass? (Only the log-sum-exp per row.)
- How would you merge partial results computed on different GPUs for the same query (split-K / ring attention)? (The same associative merge on $(m, \ell, \text{acc})$.)
"""
