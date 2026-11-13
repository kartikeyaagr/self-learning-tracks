import numpy as np

META = dict(title='ReLU on tensors of any shape', tags=['GPU Programming', 'Activation Functions'], diff='Easy',
            runtime='cuda', lessons=['ieng:13.04'], related=['leaky-relu', 'vector-addition'])

STATEMENT = r"""
Element-wise kernels don't care about shape: an `(N, C, H, W)` activation in C-contiguous memory is just one long row of floats. Write `relu(x)` that applies

$$\mathrm{ReLU}(v) = \max(v, 0)$$

to every element of a `float32` array of **any** shape and returns a new `float32` array of the **same shape**.

## Requirements
- Flatten on the host (`x.ravel()` on a contiguous array is free: no copy), launch a 1-D kernel over all `x.size` elements, and reshape the result back to `x.shape`.
- Write the result **in place** in the device buffer: copy the input to the GPU once, let the kernel overwrite it, copy it back. No second device array is needed.
- Use 64 threads per block and $\lceil \text{size} / 64 \rceil$ blocks. An empty array (size 0) returns an empty array of the same shape without launching.
- `ReLU(-0.0)` is `0.0` or `-0.0`; either is accepted. Inputs contain no NaN.

## Runtime
GPU in Colab; checked here on Numba's CUDA simulator.
"""

STARTER = '''import numpy as np
from numba import cuda

TPB = 64

def relu(x):
    """x: float32 array of any shape. Returns max(x, 0) with the same shape."""
    # TODO: flatten, copy to the device, launch relu_kernel in place, copy back, reshape
    return x.copy()

@cuda.jit
def relu_kernel(buf):
    # TODO: one thread per element, overwrite buf[i] with max(buf[i], 0)
    pass
'''

SOLUTION = '''import numpy as np
from numba import cuda

TPB = 64

def relu(x):
    flat = np.ascontiguousarray(x, dtype=np.float32).ravel()  # 1-D view, no copy when already contiguous
    if flat.size == 0:
        return flat.reshape(x.shape)
    d_buf = cuda.to_device(flat)
    relu_kernel[(flat.size + TPB - 1) // TPB, TPB](d_buf)    # overwrite in place on the device
    return d_buf.copy_to_host().reshape(x.shape)

@cuda.jit
def relu_kernel(buf):
    i = cuda.grid(1)
    if i < buf.shape[0]:
        v = buf[i]
        if v < 0:
            buf[i] = 0.0
'''

HINTS = [
    'An element-wise op on a contiguous array is the same as on its flattened 1-D view. Do the shape bookkeeping on the host.',
    'In place: `d_buf = cuda.to_device(flat)`, launch with `d_buf` as the only argument, then `d_buf.copy_to_host()`.',
    'In the kernel, only negative values need a write: `if buf[i] < 0: buf[i] = 0.0`. Remember `.reshape(x.shape)` on the way out.',
]

_r = np.random.default_rng(3)
TESTS = [
    (np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32),),
    (np.array([[1.0, -1.0], [-3.0, 4.0]], dtype=np.float32),),
    (np.zeros(0, dtype=np.float32),),
    (np.full((3, 5), -1.0, dtype=np.float32),),
    (_r.standard_normal((2, 3, 4, 5)).astype(np.float32),),     # NCHW-shaped, 120 elements
    (_r.standard_normal(65).astype(np.float32),),               # one element past a full block
    (_r.standard_normal((16, 40)).astype(np.float32) * 100,),
]

EXPLANATION = r"""
## Why it works
For a C-contiguous array, element `x[n, c, h, w]` lives at flat offset $((n C + c) H + h) W + w$. An element-wise op applied to every flat offset therefore touches every logical element exactly once, and `reshape` restores the logical view. The kernel never needs to know the shape.

## In-place on the device
Writing back into the input buffer halves device memory for the op and skips one allocation. Frameworks do the same with `relu_` / `inplace=True`. The catch: in a training graph, autograd may need the original input for the backward pass. ReLU's backward only needs the sign of the output, which is why in-place ReLU is safe in PyTorch, but in-place on an arbitrary op is not.

## Performance
Reading and writing 4 bytes per element with one comparison: purely memory-bound. Fusing ReLU into the preceding kernel (bias add, convolution, matmul epilogue) eliminates the extra round trip through global memory, which is most of what compilers like `torch.compile` gain on element-wise chains.

## Common bugs
- Using `x.flatten()` on the device side or forgetting `.reshape(x.shape)`: shape mismatch.
- Passing a non-contiguous view (e.g. `x.T`): `ravel()` then copies in a different order. `np.ascontiguousarray` makes the layout explicit.
- Launching for `size == 0`.

## Interview follow-ups
- The kernel only writes negative elements. Does skipping writes save bandwidth on a GPU? (Partly: the reads still happen, and a warp with mixed signs still issues the store instruction for its active lanes.)
- How would you handle a non-contiguous strided tensor without copying? (Pass shape and strides and compute the offset per thread, or launch a multi-dimensional grid.)
"""
