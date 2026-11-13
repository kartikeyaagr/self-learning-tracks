import numpy as np

META = dict(title='Vector add with an unmasked fast path for wide loads', tags=['GPU Programming'], diff='Easy',
            runtime='triton', tol=[1e-6, 1e-7], lessons=['ieng:13.01', 'ieng:13.03'],
            related=['fused-multiply-add', 'vector-addition', 'kv-cache-append'])

STATEMENT = r"""
A GPU moves memory fastest with **128-bit** accesses: 4 `float32` or 8 `float16` values per instruction per thread. Triton emits them only when it can prove that each thread's elements are contiguous, aligned, and either all in bounds or all out. A mask like `offs < n` with an `n` that is not a multiple of 16 breaks that proof, and the compiler falls back to one scalar access per element **for the whole tensor**, not just the tail.

The fix is to launch the bulk of the work without a mask and handle the ragged end separately.

Write `vec_add(x, y)`.

## Inputs
- `x`, `y`: contiguous CUDA tensors of the same shape and dtype, either `float32` or `float16`; any number of elements $n$ (including 0), $n < 2^{31}$.

## Output
- A new tensor of the same shape and dtype, `x + y` (computed in that dtype, as `torch` does).

## Requirements
- One kernel with constexpr parameters `BLOCK` (use 4096) and `EVEN`, and a runtime `start` offset. Program `pid` covers `start + pid * BLOCK + [0, BLOCK)`.
- **Bulk launch**: `n // BLOCK` programs, `start = 0`, `EVEN=True`: loads and stores without a mask. Tell the compiler what you know with `tl.multiple_of` and `tl.max_contiguous` on the offsets.
- **Tail launch**: if `n % BLOCK != 0`, one program with `start = (n // BLOCK) * BLOCK`, `EVEN=False`, masked by `offs < n`.
- Skip any launch that would have zero programs.

This problem runs in **Colab with a GPU runtime**. Expected outputs come from the same addition in PyTorch.
"""

STARTER = '''import torch
import triton
import triton.language as tl

def vec_add(x, y):
    """x, y: float32 or float16 CUDA tensors of one shape. Return x + y: unmasked bulk launch + masked tail."""
    raise NotImplementedError


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, start, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pass
'''

SOLUTION = '''import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, start, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = start + pid * BLOCK + tl.arange(0, BLOCK)
    # each program's block starts at a multiple of BLOCK and is BLOCK long and contiguous
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    if EVEN:                                         # compile-time branch: no mask, 128-bit accesses
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        tl.store(out_ptr + offs, x + y)
    else:                                            # one program for the ragged end
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)


def vec_add(x, y):
    assert x.shape == y.shape and x.dtype == y.dtype and x.dtype in (torch.float32, torch.float16)
    x, y = x.contiguous(), y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 4096
    n_full = n // BLOCK
    if n_full > 0:
        _add_kernel[(n_full,)](x, y, out, 0, n, BLOCK=BLOCK, EVEN=True)
    if n % BLOCK:
        _add_kernel[(1,)](x, y, out, n_full * BLOCK, n, BLOCK=BLOCK, EVEN=False)
    return out
'''

REFERENCE = '''import torch

def vec_add(x, y):
    return x + y
'''

HINTS = [
    'Use one kernel and a constexpr `EVEN` flag: `if EVEN:` is resolved at compile time, so the bulk launch compiles to a kernel with no mask at all.',
    'Offsets are `start + pid * BLOCK + tl.arange(0, BLOCK)`. Wrap them as `tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)`; both claims are true because every block starts at a multiple of `BLOCK`.',
    'On the host: `n_full = n // BLOCK`; launch `(n_full,)` programs with `start=0, EVEN=True` if `n_full > 0`, then `(1,)` program with `start=n_full * BLOCK, EVEN=False` if `n % BLOCK`.',
]

_r = np.random.default_rng(19)
_f = lambda n, dt=np.float32: _r.normal(size=n).astype(dt)
TESTS = [
    (np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([10.0, 20.0, 30.0], dtype=np.float32)),
    (np.array([0.5, -1.25], dtype=np.float16), np.array([0.25, 1.0], dtype=np.float16)),
    (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)),
    (_f(4096), _f(4096)),
    (_f(4097), _f(4097)),
    (_f(8197, np.float16), _f(8197, np.float16)),
    (_f(12288), _f(12288)),
    (_f((3, 1500)), _f((3, 1500))),
    (_f(5000, np.float16) * 1000, _f(5000, np.float16) * 1000),
]

EXPLANATION = r"""
## Why the mask matters for speed
Triton's compiler tracks, for every tensor of offsets, how many consecutive elements are contiguous and what they are divisible by. From that it decides each thread's access width. `offs < n` is constant over runs of `gcd(BLOCK, divisibility of n)` lanes; when $n$ is odd that's 1, so every access must be scalar in case the boundary falls between any two lanes. On a memory-bound kernel that can cost a large fraction of bandwidth, for a mask that only matters in the last block. (Triton specialises integer arguments divisible by 16, which is why sizes that are multiples of 16 already vectorise.)

## The split
- The bulk launch has no mask, so contiguity and alignment (`tl.multiple_of`, `tl.max_contiguous`, plus 16-byte-aligned PyTorch allocations) are all the compiler needs for `ld.global.v4` loads.
- The tail is at most `BLOCK - 1` elements in one program: its scalar accesses are irrelevant.

Test 4 ($n = 4096$) uses only the bulk launch; test 1 and test 3 only the tail (or nothing); tests 5, 6, 8 and 9 use both.

## Performance
3 accesses of 4 bytes (or 2) per element for one add: pure bandwidth. A well-vectorised version reaches 85 to 90 percent of peak DRAM bandwidth for large $n$; the gain over the masked version depends on the GPU and is largest for `float16`, where one 128-bit access covers 8 elements.

## Common bugs
- Claiming `tl.multiple_of(offs, BLOCK)` when `start` is not a multiple of `BLOCK`: lying to the compiler gives misaligned vector accesses and wrong results or faults.
- Launching the tail kernel with `start = n - n % BLOCK` but forgetting the mask.
- A grid of 0 programs for `n < BLOCK` or `n = 0`.
- Allocating the output as `torch.empty(x.shape, device=...)`, which is `float32` even for `float16` inputs.

## Interview follow-ups
- How would you confirm the vectorisation actually happened? (Inspect the generated PTX via the kernel's `asm` dictionary and look for `ld.global.v4` instructions.)
- Persistent grid alternative: launch one program per SM and loop over blocks. When is that better than one program per block?
"""
