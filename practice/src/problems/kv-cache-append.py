import numpy as np

META = dict(title='Append new keys and values to a ragged KV cache', tags=['GPU Programming', 'Inference'], diff='Easy',
            runtime='triton', tol=[1e-6, 1e-7], lessons=['ieng:03.04', 'ieng:13.04'],
            related=['autoregressive-decoding-with-kv-cache', 'flashattention-forward', 'kv-cache-memory-for-mha-mqa-gqa-and-mla'])

STATEMENT = r"""
During decoding, every step appends the new token's keys and values to a preallocated cache. Sequences in a batch have different lengths, so each one writes at its own position.

Write `kv_append(k_cache, v_cache, k_new, v_new, seq_lens)`.

## Inputs
- `k_cache`, `v_cache`: contiguous `float32` CUDA tensors `(B, H, S_max, D)`. Positions `>= seq_lens[b]` of sequence `b` hold stale data.
- `k_new`, `v_new`: contiguous `float32` `(B, H, T_new, D)`, with $T_{new} \ge 0$ (1 for a decode step, more for a chunk of prompt).
- `seq_lens`: `int64` `(B,)`, the current length of each sequence. Guaranteed: `seq_lens[b] + T_new <= S_max`.

## What to do
For every `b, h`, `t < T_new`: write `k_new[b, h, t, :]` to `k_cache[b, h, seq_lens[b] + t, :]`, and the same for `v`. Touch nothing else.

## Output
The tuple `(k_cache, v_cache, seq_lens + T_new)`: the two caches updated **in place** (return the same tensors) and the new lengths as `int64`. If $T_{new} = 0$, launch nothing and return the inputs with unchanged lengths.

## Requirements
- Grid `(B * H, T_new)`: one program per (sequence, head, new token) copies one row of `D` elements for both K and V, with `BLOCK_D = max(triton.next_power_of_2(D), 16)` and a mask.
- Compute offsets in `int64`: a real cache has billions of elements.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from an equivalent PyTorch reference.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def kv_append(k_cache, v_cache, k_new, v_new, seq_lens):
    """Write k_new/v_new (B, H, T_new, D) into the caches (B, H, S_max, D) at seq_lens[b] + t. Return (k, v, lens)."""
    raise NotImplementedError


@triton.jit
def _kv_append_kernel(kc_ptr, vc_ptr, kn_ptr, vn_ptr, lens_ptr, H, S_max, T_new, D, BLOCK_D: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _kv_append_kernel(kc_ptr, vc_ptr, kn_ptr, vn_ptr, lens_ptr, H, S_max, T_new, D, BLOCK_D: tl.constexpr):
    bh = tl.program_id(axis=0).to(tl.int64)           # flattened (batch, head)
    t = tl.program_id(axis=1).to(tl.int64)            # which new token
    b = bh // H
    pos = tl.load(lens_ptr + b) + t                   # destination slot of this token (int64)
    d = tl.arange(0, BLOCK_D)
    mask = d < D
    src = (bh * T_new + t) * D + d                    # row t of k_new[b, h]
    dst = (bh * S_max + pos) * D + d                  # row pos of k_cache[b, h]
    tl.store(kc_ptr + dst, tl.load(kn_ptr + src, mask=mask), mask=mask)
    tl.store(vc_ptr + dst, tl.load(vn_ptr + src, mask=mask), mask=mask)


def kv_append(k_cache, v_cache, k_new, v_new, seq_lens):
    B, H, S_max, D = k_cache.shape
    T_new = k_new.shape[2]
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    if T_new > 0:
        k_new, v_new = k_new.contiguous(), v_new.contiguous()
        _kv_append_kernel[(B * H, T_new)](k_cache, v_cache, k_new, v_new, seq_lens, H, S_max, T_new, D,
                                          BLOCK_D=max(triton.next_power_of_2(D), 16))
    return k_cache, v_cache, seq_lens + T_new
'''

REFERENCE = '''import torch

def kv_append(k_cache, v_cache, k_new, v_new, seq_lens):
    T_new = k_new.shape[2]
    if T_new == 0:
        return k_cache, v_cache, seq_lens.clone()
    for b in range(k_cache.shape[0]):
        s = int(seq_lens[b])
        k_cache[b, :, s:s + T_new] = k_new[b]
        v_cache[b, :, s:s + T_new] = v_new[b]
    return k_cache, v_cache, seq_lens + T_new
'''

HINTS = [
    'Each program copies one `D`-long row. From `bh = program_id(0)` get `b = bh // H`; the destination slot is `seq_lens[b] + program_id(1)`.',
    'For contiguous `(B, H, S, D)` tensors, row `s` of head `bh` starts at `(bh * S + s) * D`. The source uses `S = T_new`, the destination `S = S_max`.',
    'Load from `k_new` and `v_new` with `mask = d < D` and store into the caches with the same mask. Return the caches themselves and `seq_lens + T_new`; skip the launch when `T_new == 0`.',
]

_r = np.random.default_rng(5)
_f = lambda *s: _r.normal(size=s).astype(np.float32)
TESTS = [
    (np.zeros((2, 1, 4, 2), dtype=np.float32), np.zeros((2, 1, 4, 2), dtype=np.float32),
     np.array([[[[1.0, 2.0]]], [[[3.0, 4.0]]]], dtype=np.float32),
     np.array([[[[-1.0, -2.0]]], [[[-3.0, -4.0]]]], dtype=np.float32), np.array([0, 2], dtype=np.int64)),
    (np.full((1, 2, 5, 3), 9.0, dtype=np.float32), np.full((1, 2, 5, 3), 9.0, dtype=np.float32),
     np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3), -np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3),
     np.array([1], dtype=np.int64)),
    (_f(2, 2, 6, 4), _f(2, 2, 6, 4), np.zeros((2, 2, 0, 4), dtype=np.float32), np.zeros((2, 2, 0, 4), dtype=np.float32),
     np.array([3, 1], dtype=np.int64)),
    (_f(3, 4, 16, 64), _f(3, 4, 16, 64), _f(3, 4, 1, 64), _f(3, 4, 1, 64), np.array([0, 7, 15], dtype=np.int64)),
    (_f(2, 2, 10, 5), _f(2, 2, 10, 5), _f(2, 2, 4, 5), _f(2, 2, 4, 5), np.array([6, 0], dtype=np.int64)),
    (_f(4, 3, 32, 128), _f(4, 3, 32, 128), _f(4, 3, 8, 128), _f(4, 3, 8, 128), np.array([24, 3, 0, 11], dtype=np.int64)),
    (_f(1, 1, 3, 1), _f(1, 1, 3, 1), _f(1, 1, 3, 1), _f(1, 1, 3, 1), np.array([0], dtype=np.int64)),
]

EXPLANATION = r"""
## Why a kernel for a copy
The PyTorch loop over the batch issues $2B$ small copy kernels per layer per decode step, and launch overhead (several microseconds each) dominates the few kilobytes actually moved. One kernel with a `(B * H, T_new)` grid does the whole ragged scatter at once, and it can be captured in a CUDA graph because its launch shape doesn't depend on the lengths.

## Indexing
The cache row of sequence `b`, head `h`, slot `s` starts at $((b H + h) S_{max} + s) D$. Each sequence reads its own length from device memory, so the host never synchronises to look at `seq_lens`. Test 1 writes at slot 0 for one sequence and slot 2 for the other; test 2 checks that the untouched slots (the 9s) survive; test 4 includes `seq_lens = S_max - 1`.

## Performance
Memory-bound: $2 \cdot 2 B H T_{new} D \cdot 4$ bytes (read new, write cache, for K and V). With `D = 128` each program moves 512-byte rows, which coalesce into full 128-byte transactions. For decode ($T_{new} = 1$) a production engine usually fuses this write into the attention kernel or the QKV projection epilogue, so the new K/V never makes a separate round trip.

## Common bugs
- Using `S_max` as the source stride or `T_new` as the destination stride.
- Int32 offsets: `B * H * S_max * D` overflows $2^{31}$ for real models.
- Returning new tensors (`torch.empty_like` + copy) instead of updating in place: the cache is the model's memory, it must not be reallocated each step.
- Launching a grid with a zero dimension when `T_new == 0`.

## Interview follow-ups
- How does this change for a **paged** cache, where `seq_lens[b] + t` is first mapped through a block table to a physical block and an offset inside it?
- With grouped-query attention there are fewer KV heads than query heads. What changes in the grid and the memory per token?
"""
