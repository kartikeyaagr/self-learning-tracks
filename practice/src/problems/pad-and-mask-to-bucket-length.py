import numpy as np

META = dict(title='Pad a batch to a length bucket', tags=['JAX', 'Inference'], diff='Medium', runtime='jax',
            lessons=['ieng:15.03'], related=['masked-mean-without-boolean-indexing', 'padding-and-truncation'],
            tol=[1e-6, 1e-8])

STATEMENT = r"""
A `jit`-compiled function is recompiled for every new input shape. If you pad each batch to its own longest sequence, a serving loop sees dozens of lengths and spends its time compiling. The standard fix is to pad to a small set of **bucket lengths**, so only a few shapes ever reach the compiled function.

Write `bucket_batch(seqs, buckets, pad_id, left)`:

- `seqs`: a Python list of `B >= 1` one-dimensional int arrays (token ids). A sequence may be empty.
- `buckets`: a list of positive ints in ascending order.
- `pad_id`: int used for padding.
- `left`: bool. `False` pads on the right (tokens start at column 0). `True` pads on the left (tokens end at the last column), which is how batched decoding usually lines up prompts.

## Padded length
Let $m$ be the longest sequence length (0 if all are empty). The padded length $L$ is the smallest bucket with $L \ge m$. If $m$ is larger than the largest bucket $b_{\max}$, use the smallest multiple of $b_{\max}$ that is $\ge m$.

## Return
A tuple of three `(B, L)` arrays:

1. `tokens` (int32): the sequences placed as described, `pad_id` everywhere else.
2. `mask` (bool): `True` at real tokens.
3. `positions` (int32): position of each real token within its own sequence (`0, 1, 2, ...` from its first real token), and `0` at padding.

This is host-side preprocessing that runs before `jit`, so NumPy is fine; return NumPy or JAX arrays. This problem runs in **Colab** with the other JAX problems. The notebook includes the same tests.
"""

STARTER = '''import numpy as np

def bucket_batch(seqs, buckets, pad_id, left):
    """Return (tokens int32 (B, L), mask bool (B, L), positions int32 (B, L))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def bucket_batch(seqs, buckets, pad_id, left):
    lens = [len(s) for s in seqs]
    m = max(lens)
    fits = [b for b in buckets if b >= m]
    if fits:
        L = fits[0]                                       # buckets are ascending: first fit is the smallest
    else:
        big = buckets[-1]
        L = -(-m // big) * big                            # ceil(m / big) * big

    B = len(seqs)
    tokens = np.full((B, L), pad_id, dtype=np.int32)
    mask = np.zeros((B, L), dtype=bool)
    positions = np.zeros((B, L), dtype=np.int32)
    for i, (s, n) in enumerate(zip(seqs, lens)):          # B is small and this runs once per batch, on the host
        start = L - n if left else 0
        tokens[i, start:start + n] = s
        mask[i, start:start + n] = True
        positions[i, start:start + n] = np.arange(n)
    return tokens, mask, positions
'''

HINTS = [
    'First find $L$: scan the ascending buckets for the first one that is at least the longest length; otherwise round up to a multiple of the largest bucket.',
    'Each row is a slice assignment: the real tokens occupy `[start, start + n)`, with `start = 0` for right padding and `start = L - n` for left padding.',
    'Positions count from the first real token, so they are `arange(n)` written into the same slice, independent of the padding side.',
]

_r = np.random.default_rng(2)
TESTS = [
    ([np.array([5, 6, 7]), np.array([8])], [4, 8], 0, False),
    ([np.array([5, 6, 7]), np.array([8])], [4, 8], 0, True),
    ([np.array([1, 2, 3, 4])], [4, 8, 16], -1, False),
    ([np.array([], dtype=np.int64), np.array([3, 3])], [2, 4], 9, True),
    ([np.array([], dtype=np.int64), np.array([], dtype=np.int64)], [8, 16], 0, False),
    ([_r.integers(1, 50, size=11), _r.integers(1, 50, size=3)], [4, 8], 0, True),
    ([_r.integers(1, 50, size=n) for n in (5, 9, 2, 7)], [8, 16, 32], 0, False),
    ([_r.integers(1, 50, size=n) for n in (17, 1)], [16], 99, True),
]

EXPLANATION = r"""
## Why buckets
`jax.jit` caches one compiled executable per input shape (and dtype). Padding to the batch maximum makes the sequence dimension take almost every value from 1 to the context limit, so the cache keeps growing and the first request of each new length pays a full XLA compile. With buckets such as powers of two, at most a handful of shapes ever appear and they can all be compiled during warm-up.

## The trade-off
Padding wastes compute: on average a bucket is about 25% padding for power-of-two buckets. Finer buckets waste less but compile more. Serving systems choose a few buckets per dimension (batch and length) based on the observed length distribution.

## Left vs right padding
For training and prefill, right padding with a mask is the norm. For batched autoregressive decoding, left padding puts every prompt's last token in the final column, so the next token for every row is written at the same index. Positions must still count from each sequence's own start, which is why they are returned separately: rotary or learned position embeddings would be wrong if you used the column index.

## Complexity
$O(B \cdot L)$ host work per batch, negligible next to the model.

## Common bugs
- Using the column index as the position with left padding.
- Picking the largest bucket that fits instead of the smallest.
- Off-by-one in the overflow rule when $m$ is an exact multiple of $b_{\max}$ (it should not grow by an extra bucket).

## Interview follow-ups
- How would you choose bucket boundaries from a log of request lengths to minimise expected padding with at most 4 buckets?
- What other tools reduce recompilation? (Shape polymorphism in `jax.export`, padding batch size too, or packing several sequences into one row with segment ids.)
"""
