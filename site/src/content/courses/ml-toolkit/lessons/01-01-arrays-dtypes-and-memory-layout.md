---
id: "01.01"
course: "ml-toolkit"
module: "01"
slug: "arrays-dtypes-and-memory-layout"
title: "Arrays, dtypes and memory layout"
keyIdeas: ["An `ndarray` is one flat block of memory plus a small header: dtype, shape, strides and an offset. Most NumPy operations that look like they move data (slicing, transposing, many reshapes) only write a new header.", "Strides are the bytes to step along each axis, and an element's address is the sum of index times stride. In C order the last axis is contiguous, so a `(3, 4)` float64 array has strides `(32, 8)`; a transpose just swaps them.", "The dtype fixes bytes per element and precision: float32 keeps about 7 significant digits and cannot add 1 beyond $2^{24}$, float64 keeps about 16. Choose float32 for big matrices and float64 for accumulation.", "Integer arrays wrap around silently on overflow (`uint8` 100 − 128 = 228), and mixing dtypes can promote float32 to float64 and double memory. Cast deliberately and check `x.dtype` and `x.nbytes`.", "Vectorised code is fast because one compiled loop runs over contiguous typed memory. A Python loop pays interpreter work and object boxing per element, typically tens of times slower, and a list of floats uses 4 times the memory of a float64 array."]
question: "You load a training set as a C-ordered float32 array X of shape (100000, 256). A teammate's feature pipeline calls X.T several times (fast) and then X.T.reshape(-1) once, after which memory use jumps by about 100 MB. Give X.strides and X.T.strides, explain why X.T costs almost nothing while X.T.reshape(-1) allocates a new buffer, and say what you would check or change."
answer: "X.strides is (1024, 4): one row holds 256 values of 4 bytes, so moving one row down skips 1024 bytes, and neighbours within a row are 4 bytes apart. X.T has shape (256, 100000) and strides (4, 1024). Creating it only writes a new header with the shape and strides reversed over the same buffer, so its cost does not depend on the array's size, and np.shares_memory(X, X.T) is True. X.T is Fortran-contiguous rather than C-contiguous. reshape(-1) must return the elements in C order of X.T, that is, row 0 of X.T (column 0 of X) first: X[0,0], X[1,0], X[2,0], ... Those elements are 1024 bytes apart in the buffer, and after 100000 of them the next run starts back at byte 4. A 1-D result has a single stride, and no single stride describes that sequence, so NumPy has to gather the values into a new contiguous buffer: 100000 x 256 x 4 bytes = 102.4 MB. To check, look at X.T.flags (C_CONTIGUOUS False) and at whether the result's .base is None or np.shares_memory returns False. Fixes depend on the intent: if the flat order of X is what was wanted, use X.reshape(-1) (a free view); if column-major order really is needed and it is used repeatedly, make one deliberate copy with np.ascontiguousarray(X.T) and reuse it, or keep the data in Fortran order from the start; and if the copy is only feeding a reduction, reduce along the right axis of X instead of flattening."
sources: [{"title": "NumPy reference: The N-dimensional array", "url": "https://numpy.org/doc/stable/reference/arrays.ndarray.html", "note": "The official account of memory layout, strides and C vs Fortran order."}, {"title": "NumPy user guide: Data types", "url": "https://numpy.org/doc/stable/user/basics.types.html", "note": "Every dtype, its range, and the overflow behaviour of integer arrays."}, {"title": "NEP 50: Promotion rules for Python scalars", "url": "https://numpy.org/neps/nep-0050-scalar-promotion.html", "note": "Why NumPy 2 changed how Python scalars affect result dtypes."}, {"title": "van der Walt et al.: The NumPy array (2011)", "url": "https://arxiv.org/abs/1102.1523", "note": "The original paper on the buffer-plus-strides design and why vectorisation is fast."}]
practice: ["create-arrays-from-lists", "int8-symmetric-quantize", "arange-and-linspace", "zeros-and-ones"]
tags: ["memory layout", "dtypes", "strides"]
tools: ["NumPy"]
---

## What you will learn

- What a NumPy array really is: one flat block of bytes plus a small header that says how to read it.
- How **dtype**, **shape** and **strides** work together, and how to turn an index like `[1, 2]` into a byte address by hand.
- The difference between C order and Fortran order, and why a transpose costs nothing.
- How to choose a dtype: memory, precision, and the silent failures (integer wrap-around, lost increments, surprise upcasts).
- Why vectorised NumPy code is tens of times faster than a Python loop, measured rather than asserted.

*Before you start:* you need basic Python (lists, loops, functions) and the idea that a computer stores everything as bytes. No earlier lesson is required; this is the first lesson of the toolkit.

## Where this module starts

Almost every ML library speaks NumPy's language: pandas columns are typed arrays underneath, scikit-learn takes and returns arrays, and PyTorch and JAX copy NumPy's dtype, shape and strides almost one to one. Learn the model once here and you will recognise it everywhere.

This module builds that model from the bottom up. This lesson is about what an array *is*: memory, types and layout. Then 01.02 covers indexing and the view-versus-copy question, 01.03 broadcasting (combining arrays of different shapes), 01.04 vectorised functions, 01.05 reductions along an axis, 01.06 reshaping and `einsum`, and 01.07 linear algebra and random numbers.

## The idea in one picture

Make a small two-dimensional array and ask it to describe itself:

```python
import numpy as np
a = np.arange(12, dtype=np.float64).reshape(3, 4)
print(a)
# [[ 0.  1.  2.  3.]
#  [ 4.  5.  6.  7.]
#  [ 8.  9. 10. 11.]]
print(a.dtype, a.itemsize, a.shape, a.strides, a.nbytes)
# float64 8 (3, 4) (32, 8) 96
```

You see a grid with 3 rows and 4 columns. The computer sees something simpler: 12 numbers of 8 bytes each, laid end to end in one block of 96 bytes. The grid is an interpretation, written in a few fields of a header.

~~~dg-fig 01.01-layout.svg
Top: the array as you think of it, 3 rows by 4 columns. Bottom: the same twelve numbers as they sit in memory, one row after another. Element [1, 2] is found by jumping 1 row (32 bytes) and 2 columns (2 × 8 bytes) to byte 48.
~~~

Read the printed header like a sentence:

- **dtype** `float64`: every element is a 64-bit floating-point number, so the **itemsize** is 8 bytes.
- **shape** `(3, 4)`: 3 positions along axis 0 (rows) and 4 along axis 1 (columns).
- **strides** `(32, 8)`: to move one row down, skip 32 bytes; to move one column right, skip 8 bytes.
- **nbytes** `96`: the size of the data, $12 \times 8$.

## Step 1: an array is a header over a buffer

A NumPy array (type `ndarray`) is not a list of numbers. It is a pointer to one contiguous block of raw bytes, the **data buffer**, plus a small header that says how to read those bytes:

- **dtype**: what each element is (`float32`, `int64`, `bool`, ...) and therefore how many bytes it takes.
- **shape**: the number of elements along each axis, for example `(1000, 784)` for 1000 flattened images.
- **strides**: how many bytes to step to move one position along each axis.
- **offset and flags**: where in the buffer element `[0, 0]` starts, and facts such as whether the memory is C-contiguous, Fortran-contiguous, writeable, or owned by this array.

~~~dg-layers One ndarray = a fixed header that describes a flat byte buffer.
dtype | float32, itemsize 4 bytes
shape | (2, 3, 4)
*strides | (48, 16, 4) bytes per step on each axis
data buffer | 24 x 4 = 96 contiguous bytes
~~~

Everything else follows from this split. Slicing, transposing, adding an axis and most reshapes produce a *new header over the same buffer*: they cost microseconds at any size, and writes through them reach the original (lesson 01.02).

Contrast a Python list: a million separate float objects, each with its own type tag, reached through a million pointers. An array stores the type once and packs the raw numbers together.

## Step 2: strides turn an index into an address

In words: to find an element, start at the beginning of the data, then for each axis take "how far along this axis" times "bytes per step on this axis", and add it all up.

As a formula, for an element at index $(i_0, i_1, \dots, i_{d-1})$:

$$\text{addr} = \text{offset} + \sum_{k=0}^{d-1} i_k \cdot s_k$$

Here $d$ is the number of axes (`a.ndim`), $i_k$ is the index along axis $k$, $s_k$ is the stride of axis $k$ in bytes, and offset is where element zero sits (0 for a freshly created array).

Number check with the picture above: for `a[1, 2]` the address is $0 + 1 \times 32 + 2 \times 8 = 48$. Byte 48 divided by 8 bytes per element is slot 6, and indeed `a[1, 2]` is `6.0`.

Where do the strides come from? In the default layout the last axis is packed tightly, so its stride is the itemsize. Each earlier axis's stride is the next axis's stride times the next axis's length. For shape `(3, 4)` and 8-byte elements: $s_1 = 8$, $s_0 = 4 \times 8 = 32$.

<div class="ex">
<div class="lab">Worked example</div>

**strides and an address for a 3-D float32 array**

Take shape `(2, 3, 4)` with dtype float32, so the itemsize is 4 bytes and there are $2 \times 3 \times 4 = 24$ elements, 96 bytes in all.

Last axis: $s_2 = 4$ bytes (neighbours are adjacent).

Middle axis: one step skips a whole row of 4 elements, so $s_1 = 4 \times 4 = 16$ bytes.

First axis: one step skips a whole $3 \times 4$ block, so $s_0 = 3 \times 16 = 48$ bytes. The strides are `(48, 16, 4)`.

Address of element `[1, 2, 3]`: $1 \times 48 + 2 \times 16 + 3 \times 4 = 48 + 32 + 12 = 92$. Divide by the itemsize: $92 / 4 = 23$, the last of the 24 slots, as it should be for the last element.

The same array in Fortran order packs the *first* axis tightly: $s_0 = 4$, $s_1 = 2 \times 4 = 8$, $s_2 = 3 \times 8 = 24$, strides `(4, 8, 24)`. Element `[1, 2, 3]` is then at $4 + 16 + 72 = 92$ too (it is still the last element), but element `[1, 0, 0]` moves from byte 48 to byte 4.

</div>

~~~dg-flow How NumPy finds element [1, 2, 3] of a (2, 3, 4) float32 array.
Index (1, 2, 3) -(x strides 48, 16, 4)-> 48 + 32 + 12 -(+ offset 0)-> *Byte 92 of buffer
~~~

## Step 3: C order, Fortran order and the free transpose

There are two natural ways to lay a grid out in a line. **C order** (row-major, NumPy's default) stores row 0, then row 1, and so on: the last axis is contiguous. **Fortran order** (column-major, `order='F'`) stores column 0, then column 1: the first axis is contiguous.

~~~dg-fig 01.01-order.svg
Each number is the element's position in the buffer. C order fills rows first, Fortran order fills columns first. The transpose of a C array reads the original buffer with swapped strides, which is exactly a Fortran-ordered layout.
~~~

Now the useful trick. A transpose `a.T` reverses the shape and the strides and touches no data. For our `(3, 4)` array, `a.T` has shape `(4, 3)` and strides `(8, 32)`: moving down a row of `a.T` is moving right along a row of `a`. No byte moved, so the cost is the same for a 12-element array and a 12-billion-element array.

The result is Fortran-contiguous, not C-contiguous, and for most operations that does not matter. It matters when a function needs one flat, ordered run of memory:

- `a.T.reshape(-1)` must return elements in C order of `a.T`, which means walking down the columns of `a`. No single stride describes that sequence, so NumPy copies.
- Libraries that hand the buffer to C, Fortran or GPU code may require a particular layout. `np.ascontiguousarray(x)` returns `x` unchanged if it is already C-contiguous and makes a C-ordered copy otherwise.

Check with `a.flags['C_CONTIGUOUS']` and `a.flags['F_CONTIGUOUS']` when in doubt.

## Step 4: choosing a dtype

The dtype decides two things: how many bytes each element takes, and what values it can hold exactly. Here are the ones you will meet most:

| dtype | bytes | precision / range | typical use
| --- | --- | --- | --- |
| float64 | 8 | about 16 digits, eps 2.2e-16 | NumPy default, statistics, linear algebra
| float32 | 4 | about 7 digits, eps 1.2e-7, max 3.4e38 | model weights, large feature matrices
| float16 | 2 | about 3 digits, max 65504 | storage and inference; risky to accumulate in
| int64 | 8 | plus or minus 9.2e18 | ids, counts, default integer
| int8 / uint8 | 1 | -128..127 / 0..255 | quantised weights, image pixels
| bool | 1 | True / False | masks

**eps** (machine epsilon) is the gap between 1.0 and the next representable number of that type. It is the best mental unit for precision: float32 can tell 1.0 from $1 + 1.2 \times 10^{-7}$ but nothing finer.

~~~dg-fig 01.01-dtypes.svg
Left: memory for a 5 million by 300 matrix depends only on bytes per element. Right: the gap between neighbouring floats grows in proportion to the number itself. Once float32 passes 2²⁴ the gap is 2, so adding 1 does nothing; float16 runs out entirely at 65504.
~~~

The right panel is the key fact about floating point. Floats have a fixed number of significant digits, not a fixed number of decimal places, so the absolute gap between neighbours grows with the size of the number. Near 1000, float32 neighbours are about $6 \times 10^{-5}$ apart. Near 16.8 million ($2^{24}$) they are 1 apart, and above it they are 2 apart: float32 can no longer represent every integer.

A practical default:

- Use **float32** for large feature matrices and model weights: half the memory, often twice the speed.
- Use **float64** for statistics, small linear-algebra problems, and anything that accumulates many values.
- Use **int64** for counts and ids, **uint8** for raw pixels, **bool** for masks.
- Use **float16** only for storage or inference, never for running sums.

## Step 5: why contiguous, typed memory is fast

Summing a Python list means, per element: follow a pointer, check the type, look up addition, allocate a new float for the total. Tens of nanoseconds of interpreter work per number.

`x.sum()` on a float64 array checks the dtype once, then hands the whole buffer to a compiled loop. The loop reads memory front to back, the processor's prefetcher streams it in 64-byte **cache lines** (8 float64s per line), and **SIMD** instructions (one instruction acting on several numbers) add several elements per cycle.

~~~dg-fig 01.01-speed.svg
Left: measured time to sum n floats, log scale on both axes. All three grow linearly, but NumPy sits more than an order of magnitude lower at every size. Right: one million floats take 32 MB as a list and 8 MB as a float64 array. Timings are from one run here and are machine-dependent.
~~~

On the machine that drew this figure, summing a million floats took around 14 ms as a Python `for` loop, about 4 ms with the built-in `sum`, and well under 1 ms with `x.sum()`. The ratio depends on your hardware, but "tens of times faster" is typical.

Memory differs too: a list spends an 8-byte pointer plus a 24-byte float object per number (32 bytes); a float64 array spends 8.

## Arrays in code

Creating arrays and choosing the dtype explicitly:

```python
import numpy as np
np.array([1, 2]).dtype          # int64   (Python ints -> int64 on Linux/macOS)
np.array([1, 2.0]).dtype        # float64 (one float makes it all float)
np.zeros(3).dtype               # float64 (the default for zeros, ones, empty)
x = np.zeros((1000, 784), dtype=np.float32)
x.nbytes / 1e6                  # 3.136 (MB)
```

Strides, order and the free transpose:

```python
a = np.zeros((2, 3, 4), dtype=np.float32)
a.strides, a.T.strides            # (48, 16, 4) (4, 16, 48)
np.zeros((2, 3, 4), np.float32, order='F').strides   # (4, 8, 24)
a.T.flags['C_CONTIGUOUS']         # False: same buffer, new header
np.shares_memory(a, a.T)          # True
```

Precision limits you can ask NumPy for directly:

```python
np.finfo(np.float32).eps          # 1.1920929e-07
np.finfo(np.float16).max          # 6.55e+04 (exactly 65504)
np.iinfo(np.uint8).max            # 255
x = np.float32(2**24); x + 1 == x # True: the increment is lost
```

Converting between types:

```python
img = np.array([100, 200], dtype=np.uint8)
img.astype(np.int16) - 128        # [-28  72]: cast first, then subtract
img / 255.0                       # [0.392 0.784]: true division gives float64
w = np.ones(3, np.float64)
w.astype(np.float32, copy=False).dtype   # float32 (a new array is still needed here)
w.astype(np.float64, copy=False) is w    # True: no copy when nothing changes
```

## Checking what you have

When an array misbehaves, print its header before you look at its values. Five attributes answer most questions:

| question | ask | example answer
| --- | --- | --- |
| what type, how big per element? | `x.dtype`, `x.itemsize` | `float32`, `4`
| how many elements per axis? | `x.shape`, `x.ndim` | `(100000, 256)`, `2`
| how is it laid out? | `x.strides`, `x.flags` | `(1024, 4)`, `C_CONTIGUOUS : True`
| how much memory? | `x.nbytes` | `102400000`
| is it a view of something else? | `x.base`, `np.shares_memory(x, y)` | `None` means it owns its data

Two sanity rules: in a C-contiguous array the last stride equals the itemsize (larger means a strided view), and `x.nbytes` should match rows × columns × itemsize (double means a stray float64).

## What goes wrong

Four failure modes recur, and all are silent: no error, just wrong numbers or wasted memory.

**Integer wrap-around.** Array arithmetic does not check for overflow. Subtract 128 from a `uint8` pixel of 100 and you get 228, because the result must stay in 0..255 and arithmetic wraps modulo 256.

~~~dg-fig 01.01-wrap.svg
Subtracting 128 from every possible uint8 pixel. In uint8 the result wraps from 0 back up to 255, so dark pixels become bright. Casting to int16 first gives the straight line you expected.
~~~

**Lost increments in low precision.** float32 represents integers exactly only up to $2^{24} = 16{,}777{,}216$. A float32 counter that has reached that value stops moving when you add 1. Long running totals belong in float64 or int64.

**Unexpected promotion.** Combining two arrays promotes to a type that can hold both: `int64` with `float32` gives `float64`, which doubles memory. Since NumPy 2 (the rules in NEP 50), a plain Python scalar no longer upcasts an array, so `float32_array * 2.0` stays float32, but `float32_array * np.float64(2.0)` becomes float64 because a NumPy scalar carries its dtype with it.

**Default dtypes.** `np.array([1, 2])` is int64 and `np.zeros(n)` is float64; pass `dtype=` when it matters.

| symptom | likely cause | fix
| --- | --- | --- |
| image looks inverted or noisy after arithmetic | uint8 wrap-around | `img.astype(np.int16)` or `img / 255.0` before arithmetic
| counter or running sum stops growing | float32 past $2^{24}$ (or float16 past 2048) | accumulate in float64 / int64, or `x.sum(dtype=np.float64)`
| memory doubles after one line of code | float32 combined with float64 or int64 data | keep statistics in float32; `.astype(np.float32, copy=False)`
| a C extension or GPU call complains about layout | array is a transpose or strided slice | `np.ascontiguousarray(x)`
| `OverflowError: Python integer 300 out of bounds for uint8` | NumPy 2 refuses a Python int that does not fit the array's dtype | cast the array to a wider dtype first

<div class="ex">
<div class="lab">Worked example</div>

**memory budget for a feature matrix**

You have 5 million rows and 300 features. Number of values: $5 \times 10^6 \times 300 = 1.5 \times 10^9$.

In float64: $1.5 \times 10^9 \times 8 = 1.2 \times 10^{10}$ bytes = 12 GB, too large for many machines.

In float32: $1.5 \times 10^9 \times 4 = 6 \times 10^9$ bytes = 6 GB. About 7 significant digits is plenty for features that came from noisy measurements.

Now the trap. You centre the features with `X - mu`, where `mu` was computed with `X.mean(axis=0, dtype=np.float64)` or loaded from disk as float64. Promotion makes the result float64, so this one line allocates a fresh 12 GB array on top of the 6 GB you already hold: 18 GB at peak.

Fix: keep `mu` in float32 (`mu = mu.astype(np.float32)`, only 300 values, so the cast is free) and subtract in place with `X -= mu`, which writes into the existing buffer. Peak memory stays at about 6 GB.

</div>

## Going deeper

**Cache lines and stride cost.** Memory is fetched in 64-byte cache lines, never one number at a time. Reading contiguous float64 uses all 8 values in each line. Reading every second value uses 4 of 8, and from a stride of 64 bytes upward each value costs a whole line. Larger strides also cross 4 KB memory pages more often, which adds address-translation misses, and the compiled loop can no longer use the fast contiguous SIMD path.

~~~dg-fig 01.01-stride.svg
Summing the same one million float64 values, read at growing byte strides. Time climbs steadily as each fetched cache line carries fewer useful values, and keeps climbing past 64 bytes as page crossings add up. Measured here; machine-dependent.
~~~

So "iterate along the contiguous axis" is real advice. NumPy's own reductions cope reasonably with either axis, but hand-written loops (Numba, C) and copies such as `a.T.copy()` pay the full price of the wrong order.

**Summation is smarter than you think.** `np.ones(20_000_000, np.float32).sum()` returns exactly `2e+07`, even though a naive float32 running total would get stuck at 16,777,216. NumPy uses **pairwise summation**: it adds in blocks and then adds the block totals, so the rounding error grows roughly like $\log n$ instead of $n$. A cumulative sum cannot do this: `np.cumsum` of the same array in float32 ends at `16777216.0`. When the total matters, pass `dtype=np.float64` to the reduction.

**The full promotion rule.** For two arrays NumPy picks a type that can hold both (`int64` with `float32` gives `float64`, since float32 cannot hold every int64). NEP 50 made Python scalars "weak": they adapt to the array's dtype, and if the value does not fit (300 into uint8) NumPy raises. `np.result_type(a, b)` tells you the answer without computing anything.

**Strides can be anything.** `a[::-1]` has a negative stride and an offset pointing at the last row. `np.broadcast_to(v, (1000, 3))` has stride 0 on the first axis, so 1000 "rows" are the same 3 numbers in memory (how broadcasting in 01.03 avoids copies). `sliding_window_view` builds overlapping windows the same way. Writing through such views would change several logical elements at once, so most are read-only.

## Key terms

| Term | Meaning
| --- | --- |
| ndarray | NumPy's array: a header (dtype, shape, strides, offset, flags) over one data buffer
| Data buffer | The contiguous block of raw bytes that holds the elements
| dtype | The element type, which fixes its size in bytes and the values it can represent
| Shape | The number of elements along each axis
| Strides | Bytes to step to move one position along each axis
| C order / F order | Last axis contiguous (row-major) / first axis contiguous (column-major)
| Machine epsilon | Gap between 1.0 and the next representable float; the unit of precision
| Type promotion | The rule that picks the result dtype when different dtypes are combined
| Cache line | The 64-byte chunk in which memory is fetched; why contiguous access is fast

<div class="take">
<div class="lab">Key takeaway</div>

think of every array as a dtype, a shape and strides over one buffer. The dtype decides precision and memory, the strides decide whether an operation is free or a copy, and contiguous typed memory is the whole reason NumPy beats Python loops.

</div>
