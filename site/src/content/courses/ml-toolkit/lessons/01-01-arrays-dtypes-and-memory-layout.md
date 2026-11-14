---
id: "01.01"
course: "ml-toolkit"
module: "01"
slug: "arrays-dtypes-and-memory-layout"
title: "Arrays, dtypes and memory layout"
keyIdeas: ["An `ndarray` is one flat block of memory plus a small header: dtype, shape, strides and an offset. Most NumPy operations that look like they move data only rewrite that header.", "The dtype fixes how many bytes each element takes and what it can hold: float32 keeps about 7 significant digits, float64 about 16, and integer arrays wrap around silently when they overflow.", "Strides are the byte step per axis. In C order the last axis is contiguous, so for a `(3, 4)` float64 array the strides are `(32, 8)`. A transpose just swaps them.", "Vectorised code is fast because one C loop runs over a contiguous typed buffer. A Python loop pays interpreter dispatch and object boxing on every element, which costs 10 to 100 times more."]
question: "X is a C-ordered float32 array of shape (100000, 256). Give X.strides and X.T.strides. Why does X.T cost almost nothing, while X.T.reshape(-1) allocates about 100 MB?"
answer: "X.strides is (1024, 4): one row is 256 x 4 = 1024 bytes and neighbours within a row are 4 bytes apart. X.T has shape (256, 100000) and strides (4, 1024). Creating it only writes a new header with the axes and strides swapped, over the same buffer, so the cost does not depend on the array size. reshape(-1) has to return elements in C order of X.T, which means walking down the columns of X: element 0 of row 0, then element 0 of row 1, and so on. In the buffer those elements are 1024 bytes apart, then the next run starts 4 bytes after the first. A single stride cannot describe that sequence for a 1-D result, so NumPy must copy the data into a new contiguous buffer: 100000 x 256 x 4 bytes is about 102 MB."
sources: [{"title": "NumPy reference: The N-dimensional array", "url": "https://numpy.org/doc/stable/reference/arrays.ndarray.html", "note": "The official account of memory layout, strides and C vs Fortran order."}, {"title": "NumPy user guide: Data types", "url": "https://numpy.org/doc/stable/user/basics.types.html", "note": "Every dtype, its range, and the overflow behaviour of integer arrays."}, {"title": "NEP 50: Promotion rules for Python scalars", "url": "https://numpy.org/neps/nep-0050-scalar-promotion.html", "note": "Why NumPy 2 changed how Python scalars affect result dtypes."}, {"title": "van der Walt et al.: The NumPy array (2011)", "url": "https://arxiv.org/abs/1102.1523", "note": "The original paper on the buffer-plus-strides design and why vectorisation is fast."}]
practice: ["create-arrays-from-lists", "int8-symmetric-quantize", "arange-and-linspace", "zeros-and-ones"]
tags: ["memory layout", "dtypes", "strides"]
tools: ["NumPy"]
---

## What an ndarray actually is

A NumPy array is not a list of numbers. It is a pointer to one contiguous block of raw bytes (the **data buffer**) plus a small header that says how to read those bytes:

- **dtype**: what each element is (`float32`, `int64`, `bool`, ...) and so how many bytes it takes (`itemsize`).
- **shape**: the number of elements along each axis, e.g. `(1000, 784)`.
- **strides**: how many bytes to step to move one position along each axis.
- **offset and flags**: where in the buffer element `[0, 0]` starts, and whether the memory is C-contiguous, F-contiguous, writeable, or owned by this array.

Everything else follows from this split. Slicing, transposing, adding an axis and most reshapes produce a new header over the *same* buffer. That is why they cost microseconds whatever the array size, and also why they can surprise you when you write through them (next lesson).

~~~dg-layers One ndarray = a fixed header that describes a flat byte buffer.
dtype | float32, itemsize 4 bytes
shape | (2, 3, 4)
*strides | (48, 16, 4) bytes per step on each axis
data buffer | 24 x 4 = 96 contiguous bytes
~~~

## Strides: from an index to an address

For an element at index $(i_0, i_1, \dots)$ the byte address is

$$\text{addr} = \text{offset} + \sum_k i_k \cdot \text{stride}_k$$

In **C order** (row-major, NumPy's default) the last axis is contiguous: for shape `(2, 3, 4)` and float32 the strides are `(3*4*4, 4*4, 4) = (48, 16, 4)`. In **Fortran order** (column-major, `order='F'`) the first axis is contiguous and the strides would be `(4, 8, 24)`.

~~~dg-flow How NumPy finds element [1, 2, 3] of a (2, 3, 4) float32 array.
Index (1, 2, 3) -(x strides 48, 16, 4)-> 48 + 32 + 12 -(+ offset)-> *Byte 92 of buffer
~~~

A transpose `a.T` reverses shape and strides and touches no data. The result is F-contiguous, not C-contiguous, and that is fine for most operations. It matters when a function needs a flat, ordered layout: `a.T.reshape(-1)` must copy, and libraries that hand the buffer to C or GPU code (a BLAS call, a PyTorch `from_numpy` followed by a kernel) may need `np.ascontiguousarray(a)` first. Check `a.flags['C_CONTIGUOUS']` when in doubt.

## dtypes and precision

| dtype | bytes | precision / range | typical use
| --- | --- | --- | --- |
| float64 | 8 | about 16 digits, eps 2.2e-16 | NumPy default, statistics, linear algebra
| float32 | 4 | about 7 digits, eps 1.2e-7, max 3.4e38 | model weights, large feature matrices
| float16 | 2 | about 3 digits, max 65504 | storage and inference; risky to accumulate in
| int64 | 8 | plus or minus 9.2e18 | ids, counts, default integer
| int8 / uint8 | 1 | -128..127 / 0..255 | quantised weights, image pixels
| bool | 1 | True / False | masks

Four failure modes come up again and again:

- **Silent integer wrap-around.** Array arithmetic does not check overflow. A `uint8` image minus 128 wraps: `np.array([100], np.uint8) - 128` gives `228`. Cast first: `img.astype(np.int16) - 128` or `img / 255.0`.
- **Lost increments in low precision.** float32 represents integers exactly only up to $2^{24} = 16{,}777{,}216$; adding 1 to that number in float32 returns the same number. Long running sums and counters belong in float64 or int64.
- **Unexpected promotion.** Combining arrays promotes to a type that can hold both: `int64` with `float32` gives `float64`, which doubles memory. Since NumPy 2 (NEP 50) a plain Python scalar no longer upcasts an array, so `float32_array * 2.0` stays float32, but `float32_array * np.float64(2.0)` becomes float64.
- **Default dtypes.** `np.array([1, 2])` is int64 and `np.zeros(n)` is float64. Pass `dtype=` explicitly when memory or a downstream library cares.

## Why vectorised code is fast

A Python list of a million floats is a million pointers to separately allocated float objects (24 bytes each plus an 8-byte pointer), scattered across the heap. A loop over it does, for every element: fetch the object, check its type, look up `__add__`, allocate a new result object. That is tens of nanoseconds of interpreter work per element.

`x.sum()` on a float64 array does one type dispatch, then runs a compiled loop over 8 MB of contiguous memory. The hardware prefetcher streams it in 64-byte cache lines (8 float64s each), and SIMD instructions add several elements per cycle. On a typical laptop summing $10^7$ floats takes several hundred milliseconds as a Python loop and a few milliseconds in NumPy. The same contiguity argument explains why memory layout matters: traversing along a large stride wastes most of every cache line fetched.

```python
import numpy as np
a = np.zeros((2, 3, 4), dtype=np.float32)
a.strides, a.T.strides            # (48, 16, 4) (4, 16, 48)
a.T.flags['C_CONTIGUOUS']         # False: same buffer, new header
np.shares_memory(a, a.T)          # True
x = np.float32(2**24); x + 1 == x # True: the increment is lost
```

<div class="ex">
<div class="lab">Worked example</div>

**memory budget for a feature matrix**

5 million rows by 300 features. In float64 this is $5 \times 10^6 \times 300 \times 8$ bytes = 12 GB, too large for many machines. In float32 it is 6 GB; ~7 significant digits is plenty for features that came from noisy measurements.

Watch for silent upcasts: one `X - X.mean(axis=0)` where the mean was computed in float64 (the `dtype=np.float64` argument, or a float64 array of statistics loaded from disk) creates a fresh 12 GB float64 result. Keep the statistics in float32 or cast with `.astype(np.float32, copy=False)`.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

think of every array as a dtype, a shape and strides over one buffer. The dtype decides precision and memory, the strides decide whether an operation is free or a copy, and contiguous typed memory is the whole reason NumPy beats Python loops.

</div>
