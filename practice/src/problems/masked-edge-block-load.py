import numpy as np

META = dict(title='Load an edge block of a matrix with pointer offsets and a mask', tags=['GPU Programming'], diff='Easy', runtime='numpy',
            related=['grouped-program-id-matmul'])

STATEMENT = r"""
In Triton a kernel sees a matrix as a flat buffer plus strides, and loads a block with `tl.load(ptr + offsets, mask=mask, other=0)`. Simulate that. Write `load_block(buf, M, K, stride_m, stride_k, pid_m, pid_k, BM, BK, other)`:

1. Row indices `rm = pid_m * BM + arange(BM)`, column indices `rk = pid_k * BK + arange(BK)`.
2. Offsets `rm[:, None] * stride_m + rk[None, :] * stride_k` into the flat array `buf`.
3. `mask = (rm < M)[:, None] & (rk < K)[None, :]`.
4. The block takes `buf[offset]` where masked in, and `other` elsewhere. Never index `buf` at a masked-out offset (it may be out of range).

Return `(block, mask, n_valid)` where `n_valid` is the number of real elements loaded.
"""

STARTER = '''import numpy as np

def load_block(buf, M, K, stride_m, stride_k, pid_m, pid_k, BM, BK, other):
    """Simulate a masked tl.load of one block."""
    buf = np.asarray(buf)
    rm = pid_m * BM + np.arange(BM); rk = pid_k * BK + np.arange(BK)
    off = rm[:, None] * stride_m + rk[None, :] * stride_k
    return buf[off], np.ones((BM, BK), bool), BM * BK
'''

SOLUTION = '''import numpy as np

def load_block(buf, M, K, stride_m, stride_k, pid_m, pid_k, BM, BK, other):
    buf = np.asarray(buf, float)
    rm = pid_m * BM + np.arange(BM)
    rk = pid_k * BK + np.arange(BK)
    off = rm[:, None] * stride_m + rk[None, :] * stride_k
    mask = (rm < M)[:, None] & (rk < K)[None, :]
    block = np.full((BM, BK), float(other))
    block[mask] = buf[off[mask]]
    return block, mask, int(mask.sum())
'''

HINTS = [
    'Build the mask from row and column bounds separately and combine them by broadcasting.',
    'Gather only at masked-in offsets; the others may point past the end of the buffer.',
    'Strides make the same code read a transposed matrix: swap stride_m and stride_k.',
]

_A = np.arange(35.0)            # a 5 x 7 matrix, row-major
TESTS = [
    (_A, 5, 7, 7, 1, 0, 0, 4, 4, 0.0),
    (_A, 5, 7, 7, 1, 1, 1, 4, 4, 0.0),
    (_A, 7, 5, 1, 7, 1, 0, 4, 4, -1.0),
    (_A, 5, 7, 7, 1, 0, 1, 8, 8, 0.0),
    (np.arange(6.0), 2, 3, 3, 1, 0, 0, 2, 2, 9.0),
]

EXPLANATION = r"""
## Pointers, strides and blocks
Triton programs compute a block of addresses with broadcasting, exactly as here, and load the whole block in one instruction. Strides decouple the logical layout from memory: the same kernel handles row-major, column-major and transposed views by passing different strides.

## Why masks
Matrix sizes are rarely multiples of the block size, so the last block in each dimension hangs over the edge. The mask stops out-of-bounds reads, and `other=0` makes the padded entries harmless for a matmul (they add nothing to the dot products). Forgetting the mask gives wrong answers at best and illegal memory accesses at worst.

## Interview follow-ups
- Why is padding with 0 safe for matmul but not for a max-reduction?
"""
