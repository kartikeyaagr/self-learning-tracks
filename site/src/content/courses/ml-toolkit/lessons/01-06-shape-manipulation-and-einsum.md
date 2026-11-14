---
id: "01.06"
course: "ml-toolkit"
module: "01"
slug: "shape-manipulation-and-einsum"
title: "Shape manipulation and einsum"
keyIdeas: ["`reshape` reads the elements in C order (last axis fastest) and pours them into the new shape in that same order. It only moves the line breaks, never an element's place in the sequence. It is a view when the strides allow and a copy otherwise, for example after a transpose.", "`transpose`, `swapaxes` and `moveaxis` reorder whole axes by permuting strides, at no cost. Using `reshape` where you meant one of these gives the right shape with scrambled data, and only a value check catches it.", "To cut an array into patches, windows or heads: reshape to split each axis into 'which block' and 'inside the block', transpose the block indices to the front, then reshape to merge. The final merge usually copies.", "`concatenate` lengthens an existing axis; `stack` adds a new one. Concatenating inside a loop is quadratic, so collect pieces in a list and join once, or preallocate.", "`tile` and `repeat` allocate; `broadcast_to` and `sliding_window_view` give read-only, zero-copy views built from strides.", "`einsum` names each axis with a letter: letters in the output are kept, letters missing from it are summed. `'bhqd,bhkd->bhqk'` is attention scores in one line; use `optimize=True` for three or more operands."]
question: "You have token embeddings `X` of shape (B, T, D) and want multi-head form (B, H, T, D/H). Explain why `X.reshape(B, H, T, D // H)` is wrong, give the correct sequence of calls, and say which step copies memory. Then write the einsum that computes attention scores of shape (B, H, T, T) from queries and keys in that layout, and describe a quick test that would have caught the reshape bug."
answer: "`reshape` regroups elements in C order. For each batch item, the T x D block is read row by row (token by token) and cut into H consecutive chunks of T x D/H values. That mixes tokens: the first 'head' ends up holding the full embeddings of the first T/H tokens rather than the first D/H features of every token, so the array has the right shape and the wrong meaning. The heads must split the feature axis D, so the correct sequence is `X.reshape(B, T, H, D // H)`, which is a view because it only splits the last axis of a contiguous array, then `.transpose(0, 2, 1, 3)` (or `np.moveaxis(..., 2, 1)`) to get (B, H, T, D/H), also a view since it only permutes strides. Nothing has copied yet. A copy happens only when a later operation needs contiguous memory, for example `np.ascontiguousarray`, a reshape that merges the transposed axes (such as merging heads back to (B, T, D) after a transpose), or handing the array to code that requires C order. With queries Q and keys K both of shape (B, H, T, D/H), the scores are `np.einsum('bhqd,bhkd->bhqk', Q, K)`: b and h are kept, q and k become the two output token axes, and d is summed, giving (B, H, T, T); it equals `Q @ K.swapaxes(-1, -2)`. To catch the bug, build `X = np.arange(B*T*D).reshape(B, T, D)` so every entry is unique, then assert that `heads[0, 1, 0]` equals `X[0, 0, D//H : 2*D//H]` (head 1 of token 0 is the second chunk of token 0's features). The wrong reshape fails this assertion even though its shape is correct."
sources: [{"title": "numpy.reshape", "url": "https://numpy.org/doc/stable/reference/generated/numpy.reshape.html", "note": "Element order, the -1 dimension, and when a copy is made."}, {"title": "NumPy reference: Array manipulation routines", "url": "https://numpy.org/doc/stable/reference/routines.array-manipulation.html", "note": "One page listing transpose, stack, split, tile, repeat and friends."}, {"title": "numpy.einsum", "url": "https://numpy.org/doc/stable/reference/generated/numpy.einsum.html", "note": "Subscript rules, implicit vs explicit mode, and the optimize argument."}, {"title": "sliding_window_view", "url": "https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html", "note": "Zero-copy windows and the warnings about writing to them."}]
practice: ["reshaping-arrays", "window-partition-and-reverse", "tile-and-diff", "concat-and-correlate", "outer-product"]
tags: ["reshape", "einsum"]
tools: ["NumPy"]
---

## What you will learn

- What `reshape` really does (it regroups a fixed sequence of numbers) and how to predict its output before you run it.
- How `transpose`, `swapaxes` and `moveaxis` move whole axes, and why confusing them with `reshape` gives the right shape with scrambled data.
- How to combine reshape and transpose to cut an image into patches, the core step of vision transformers.
- How `concatenate`, `stack`, `split`, `tile`, `repeat` and `sliding_window_view` join, cut and repeat arrays, and which of them copy memory.
- How to read and write `einsum` strings, so that one line states the full shape contract of a product.

*Before you start:* you should know shapes, dtypes and strides (01.01), views versus copies (01.02), broadcasting (01.03) and the `axis` argument of reductions (01.05). Nothing else is needed.

## From reductions to rearrangements

Lesson 01.05 collapsed axes with reductions. This lesson covers operations that keep every number but change how they are arranged: converting `(N, C, H, W)` images to `(N, H, W, C)`, splitting embeddings into heads, joining vectors into a matrix. Each is a one-line call, and each has a version that runs without error and silently produces garbage.

## The idea in one picture

Take twelve numbers and ask NumPy for three different shapes:

```python
import numpy as np
a = np.arange(12)            # [ 0  1  2 ... 11], shape (12,)
a.reshape(3, 4)              # rows: [0 1 2 3], [4 5 6 7], [8 9 10 11]
a.reshape(4, 3)              # rows: [0 1 2], [3 4 5], [6 7 8], [9 10 11]
a.reshape(2, -1).shape       # (2, 6): the -1 is worked out for you
```

~~~dg-fig 01.06-reshape.svg
The same twelve numbers in three shapes. Read any of them left to right, top to bottom and you get 0, 1, 2, ..., 11 in that order. Reshape only decides where each row ends.
~~~

Two terms will appear throughout:

- **C order** (also called row-major order) is the reading order NumPy uses by default: the last axis changes fastest. For a `(3, 4)` array that means `[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], ...`.
- An **axis permutation** is a reordering of the axes themselves, such as turning `(rows, columns)` into `(columns, rows)`. This is what `transpose` does, and it really does change which number sits where.

## Step 1: reshape regroups, it never reorders

The rule, in words: flatten the array in C order, then pour the sequence into the new shape, again in C order. In symbols, an element at position $(i, j)$ of an array with $W$ columns sits at flat position $f = iW + j$. After reshaping to $W'$ columns it lands at row $\lfloor f / W' \rfloor$ and column $f \bmod W'$.

Number check: element 7 of the `(3, 4)` array is at $(1, 3)$, so $f = 1 \cdot 4 + 3 = 7$. In the `(2, 6)` shape it lands at row $\lfloor 7/6 \rfloor = 1$, column $7 \bmod 6 = 1$, which matches the figure.

Some practical rules:

- The element count must not change: `np.arange(12).reshape(5, 3)` raises `ValueError`.
- At most one dimension can be `-1`. `x.reshape(-1)` flattens; `x.reshape(len(x), -1)` keeps the first axis and flattens the rest.
- Adding or removing length-1 axes is also a reshape: `v[:, None]` or `np.expand_dims(v, 1)` turns `(3,)` into `(3, 1)` (the broadcasting trick from 01.03), and `np.squeeze` removes them.

**View or copy?** An array is a buffer plus **strides**, the bytes to step per axis (01.01). `reshape` returns a view whenever some strides describe the new shape over the same buffer, which is always true for a contiguous array:

```python
b = a.reshape(3, 4)
print(b.strides, np.shares_memory(a, b))   # (32, 8) True   (int64: 8 bytes per step)
t = b.T                                    # transpose: strides (8, 32)
print(t.flags['C_CONTIGUOUS'])             # False
r = t.reshape(12)
print(r)                                   # [ 0  4  8  1  5  9  2  6 10  3  7 11]
print(np.shares_memory(r, b))              # False: this reshape had to copy
```

The transposed array reads its elements in the order 0, 4, 8, 1, ... and no single stride can walk the buffer that way, so NumPy copies. So reshape can be silently expensive on non-contiguous input, and you should not write through a reshape result without checking `np.shares_memory`.

## Step 2: transpose moves axes

`transpose` leaves the numbers in memory and permutes the strides, so indexing walks the buffer in a different order. `x.transpose(p)` is a view whose axis `i` is the input's axis `p[i]`; with no argument (or `x.T`) it reverses all axes.

Two readable helpers:

- `np.swapaxes(x, a, b)` swaps two axes and leaves the rest alone.
- `np.moveaxis(x, src, dst)` moves one axis to a new position and keeps the others in order. `np.moveaxis(x, 1, -1)` turns `(N, C, H, W)` into `(N, H, W, C)`, which is easier to read than `x.transpose(0, 2, 3, 1)`.

```python
np.moveaxis(np.zeros((2, 3, 4, 5)), 1, -1).shape   # (2, 4, 5, 3)
np.swapaxes(np.zeros((2, 3, 4)), 0, 2).shape       # (4, 3, 2)
```

All three are free: a new header with permuted shape and strides, no data touched. The cost arrives later, if something needs C-contiguous memory and copies.

Now the most common shape bug in numerical code. Compare a transpose and a reshape that produce the same shape:

~~~dg-fig 01.06-transpose-vs-reshape.svg
Cells are coloured by the column they came from. The transpose keeps each original column together; the reshape mixes columns in every row. Both results have shape (3, 2).
~~~

```python
x = np.arange(6).reshape(2, 3)   # [[0 1 2], [3 4 5]]
x.T                              # [[0 3], [1 4], [2 5]]  each row = an original column
x.reshape(3, 2)                  # [[0 1], [2 3], [4 5]]  just new line breaks
```

At full scale this is the classic batch bug: `x.reshape(N, H, W, C)` on an `(N, C, H, W)` batch has the correct shape and scrambled pixels. The correct call is `x.transpose(0, 2, 3, 1)`. No shape check catches this; only a value check does, such as asserting `out[0, 0, 0, :]` equals `x[0, :, 0, 0]`.

## Step 3: reshape and transpose together cut a grid into blocks

Vision transformers split images into patches, Swin attention into windows, multi-head attention splits features into heads. All use the same three-move recipe:

1. Reshape to split each axis you want to cut into two axes: "which block" and "position inside the block". This is a view.
2. Transpose so that all "which block" axes come first and all "inside the block" axes come last. Also a view.
3. Reshape to merge each group. This usually copies, because the pixels of one patch are not adjacent in memory.

~~~dg-flow Patchify: split each spatial axis in two, bring the block indices together, flatten each patch.
(H, W) -(reshape)-> (H/p, p, W/p, p) -(transpose 0,2,1,3)-> (H/p, W/p, p, p) -(reshape)-> *(H/p * W/p, p*p)
~~~

```python
def patchify(img, p):
    H, W = img.shape
    return (img.reshape(H // p, p, W // p, p)   # [block_row, row_in, block_col, col_in]
               .transpose(0, 2, 1, 3)            # [block_row, block_col, row_in, col_in]
               .reshape(-1, p * p))              # one row per patch
```

<div class="ex">
<div class="lab">Worked example</div>

**patchify a 4 x 4 image with p = 2**

Start with `img = np.arange(16).reshape(4, 4)`, so row $r$ holds $4r, 4r+1, 4r+2, 4r+3$. The top-left 2 x 2 patch covers rows 0-1 and columns 0-1, so it must contain pixels 0, 1, 4, 5.

First reshape to `(2, 2, 2, 2)`. Index `[a, b, c, d]` means block row $a$, row inside the block $b$, block column $c$, column inside the block $d$, and the pixel value is $8a + 4b + 2c + d$. Check: `[0, 1, 0, 1]` gives $0 + 4 + 0 + 1 = 5$, the pixel at row 1, column 1. Correct.

Transpose with `(0, 2, 1, 3)`: the new index is `[a, c, b, d]`, still with value $8a + 4b + 2c + d$. Nothing has moved in memory; only the walking order changed.

Final reshape to `(4, 4)` reads `[a, c, b, d]` in C order. For patch $(a, c) = (0, 0)$ the four values are $b, d \in \{(0,0), (0,1), (1,0), (1,1)\}$, giving $0, 1, 4, 5$. Patch $(0, 1)$ adds 2: $2, 3, 6, 7$. Patch $(1, 0)$ adds 8: $8, 9, 12, 13$. Patch $(1, 1)$ adds 10: $10, 11, 14, 15$.

Skipping the transpose and reshaping `(4, 4)` straight to `(4, 4)` returns the image rows, `[0, 1, 2, 3]` first. That row straddles two patches: the right shape with the wrong answer.

</div>

~~~dg-fig 01.06-patchify.svg
Pixels are coloured by the 2×2 patch they belong to. With the transpose each output row has a single colour (one patch). Without it each row mixes two patches.
~~~

`np.shares_memory(step, img)` is true after the first reshape and the transpose, false after the final reshape. Multi-head attention is the same pattern: `X.reshape(B, T, H, D // H).transpose(0, 2, 1, 3)` turns `(B, T, D)` into `(B, H, T, D/H)` with no copy.

## Step 4: joining and splitting

~~~dg-fig 01.06-concat-stack.svg
Concatenate makes an existing axis longer. Stack inserts a new axis whose length is the number of arrays.
~~~

- `np.concatenate([a, b], axis=k)`: shapes must match on every axis except `k`, and the result is as long as both inputs together along `k`.
- `np.stack([a, b], axis=k)`: shapes must be identical, and a new axis of length 2 (the number of inputs) is inserted at position `k`. Stacking 32 images of shape `(H, W, 3)` gives `(32, H, W, 3)`.
- `np.vstack` and `np.hstack` are shortcuts with one surprise: `vstack` treats 1-D inputs as rows, so two `(3,)` vectors become a `(2, 3)` array, while `hstack` joins them end to end into one `(6,)` vector. `np.column_stack` makes them the columns of a `(3, 2)` array.

```python
A, B = np.ones((2, 3)), np.zeros((2, 3))
np.concatenate([A, B], axis=0).shape   # (4, 3)
np.concatenate([A, B], axis=1).shape   # (2, 6)
np.stack([A, B], axis=0).shape         # (2, 2, 3)
np.stack([A, B], axis=2).shape         # (2, 3, 2)
u, v = np.array([1, 2, 3]), np.array([4, 5, 6])
np.vstack([u, v]).shape, np.hstack([u, v]).shape, np.column_stack([u, v]).shape
# ((2, 3), (6,), (3, 2))
```

Splitting is the reverse: `np.split(x, 4)` needs equal parts, `np.array_split(x, 4)` allows uneven ones (10 elements give 3, 3, 2, 2), and `np.split(x, [10, 20])` cuts at those indices. The pieces are views.

**Never grow an array in a loop.** Every `np.concatenate` allocates a new buffer and copies all inputs into it. Calling it repeatedly to grow a result copies everything accumulated so far each time. After $n$ pieces of size $m$ you have copied $m(1 + 2 + \dots + n) \approx m n^2 / 2$ numbers, which is $O(n^2)$. The fix is to collect pieces in a Python list (appending to a list is cheap) and call `np.concatenate` once, or, if you know the final size, to preallocate with `np.empty` and fill slices.

~~~dg-fig 01.06-append-timing.svg
Measured on this machine (timings are machine-dependent; the slopes are what matter). Growing with concatenate in the loop has slope 2 on log-log axes, which is quadratic; the other two have slope 1.
~~~

With 8000 pieces of 100 floats the loop took about 3 seconds here and the list about 1 millisecond. `np.append` wraps `concatenate` and has the same problem.

## Step 5: repeating and windows

Two functions repeat data, and they repeat different things:

- `np.tile([1, 2], 3)` gives `[1, 2, 1, 2, 1, 2]`: the whole block is repeated.
- `np.repeat([1, 2], 3)` gives `[1, 1, 1, 2, 2, 2]`: each element is repeated in place.

Both allocate the full repeated array. If it is only an operand of arithmetic, broadcasting (01.03) avoids building it, and `np.broadcast_to(v, (1000, 3))` is a read-only view that looks like 1000 copies of `v` but uses the memory of one (stride 0 on the new axis).

For overlapping windows over a signal, use `np.lib.stride_tricks.sliding_window_view(x, w)`. On a length-$n$ signal it returns an $(n - w + 1, w)$ view in which window $i$ is `x[i:i+w]`. It costs no memory because consecutive windows share elements: both strides point into the same buffer, one element apart.

```python
from numpy.lib.stride_tricks import sliding_window_view
s = np.arange(8.)                  # [0. 1. ... 7.]
win = sliding_window_view(s, 3)
print(win.shape, win.strides)      # (6, 3) (8, 8): both axes step one float
print(win.mean(axis=1))            # [1. 2. 3. 4. 5. 6.]  a moving average
print(win.flags.writeable)         # False
```

It is read-only, because one cell belongs to several windows. Arithmetic on `win` or `win.copy()` materialises all $(n - w + 1) \cdot w$ values, so reduce first (as with `mean`) on long signals.

## Step 6: einsum names every axis

`np.einsum` replaces combinations of `@`, `transpose` and `sum` with one string: a letter for each axis of each input, then the letters the output keeps. The rule:

- A letter that appears in the output is kept as an output axis.
- A letter that appears in the inputs but not in the output is multiplied across the inputs and **summed** away.
- A letter that appears in several inputs must have the same length in each.

For a matrix product the string is `'ij,jk->ik'`. In words: for every `i` and `k`, multiply `M[i, j] * N[j, k]` and sum over `j`. The formula is $\text{out}_{ik} = \sum_j M_{ij} N_{jk}$.

~~~dg-fig 01.06-einsum.svg
einsum 'ij,jk->ik'. The letter j is missing from the output, so for out[0, 1] row 0 of M and column 1 of N are multiplied entry by entry and summed: 1·8 + 2·10 + 3·12 = 64.
~~~

```python
M = np.array([[1, 2, 3], [4, 5, 6]])
N = np.array([[7, 8], [9, 10], [11, 12]])
np.einsum('ij,jk->ik', M, N)       # [[ 58  64] [139 154]], same as M @ N
S = np.array([[1, 2], [3, 4]])
np.einsum('ii->', S)               # 5     trace: diagonal, summed
np.einsum('ij->ji', S)             # [[1 3] [2 4]]  transpose: nothing summed
np.einsum('ij->j', S)              # [4 6] column sums, like S.sum(axis=0)
np.einsum('i,j->ij', [1, 2], [10, 20, 30])   # [[10 20 30] [20 40 60]] outer product
```

The common patterns are worth knowing by sight:

| subscripts | operation
| --- | --- |
| `'ij,jk->ik'` | matrix product
| `'bij,bjk->bik'` | batched matrix product (`b` kept, `j` summed)
| `'bi,bi->b'` | row-wise dot products, same as `(X * Y).sum(1)`
| `'i,j->ij'` | outer product (nothing summed)
| `'ii->'` | trace
| `'ij->'` | sum of all entries
| `'bhqd,bhkd->bhqk'` | attention scores per batch and head

The last row is why einsum is popular in deep learning: queries `(B, H, Tq, D)` and keys `(B, H, Tk, D)` give scores `(B, H, Tq, Tk)` by summing over `d`. `Q @ K.swapaxes(-1, -2)` does the same but hides which axis is contracted.

```python
B, H, T, D = 2, 4, 5, 8
rng = np.random.default_rng(0)
Q, K = rng.normal(size=(B, H, T, D)), rng.normal(size=(B, H, T, D))
scores = np.einsum('bhqd,bhkd->bhqk', Q, K)
print(scores.shape)                                      # (2, 4, 5, 5)
print(np.allclose(scores, Q @ K.swapaxes(-1, -2)))       # True
```

The string states the shape contract: which axes are kept, matched and summed. Mismatched lengths for a shared letter raise an error.

## Checking the result

Shape bugs rarely crash; they put numbers in the wrong places. A few habits catch nearly all of them:

- **Check one value, not just the shape.** After a layout change, assert that a known element ended up where it should: `assert out[0, 0, 0, 1] == x[0, 1, 0, 0]` after converting NCHW to NHWC.
- **Use distinct test data.** `np.arange(np.prod(shape)).reshape(shape)` makes every entry unique; an array of ones hides everything.
- **Round-trip.** If you patchify, write the inverse (reshape to `(H/p, W/p, p, p)`, transpose `(0, 2, 1, 3)`, reshape to `(H, W)`) and assert you get the original back.
- **Check memory when it matters.** `np.shares_memory(a, b)` tells you whether an operation copied; `x.flags['C_CONTIGUOUS']` tells you whether a later reshape will have to.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Right shape, images look like noise or stripes | `reshape` used where `transpose` or `moveaxis` was needed | Move axes with `transpose`; reshape only to split or merge adjacent axes
| Patches or heads contain data from the wrong place | Final reshape done without the middle transpose | Split, transpose block indices to the front, then merge
| Loop that builds an array gets slower and slower | `np.concatenate` or `np.append` inside the loop, $O(n^2)$ copying | Collect in a list and concatenate once, or preallocate
| `vstack` of two vectors gives `(2, n)` but you wanted `(n, 2)` | `vstack` treats 1-D inputs as rows | `np.column_stack` or `np.stack(..., axis=1)`
| Writing through a reshape result does not change the original | The reshape had to copy (non-contiguous input) | Write to the original, or check `np.shares_memory` first
| Memory spikes after `tile` or `repeat` | Full repeated array allocated | Broadcast or use `np.broadcast_to`
| `ValueError: assignment destination is read-only` | Writing into `sliding_window_view` or `broadcast_to` output | Compute a new array instead, or `.copy()` deliberately
| einsum is slow with three or more operands | Contraction done in a bad order | Pass `optimize=True`

## Going deeper

**When exactly is reshape a view?** NumPy tries to find strides for the new shape over the old buffer. Splitting an axis is always possible: an axis of length $ab$ with stride $s$ becomes axes $a$ and $b$ with strides $bs$ and $s$. Merging two adjacent axes of lengths $a$ and $b$ with strides $s_1$ and $s_2$ is possible only when $s_1 = b \cdot s_2$, meaning the outer axis steps exactly over one full run of the inner axis. After a transpose that condition fails, which is why the last step of patchify copies. Axes of length 1 can be added or removed freely since their stride never matters.

**C versus F order.** `reshape(..., order='F')` reads in column-major order (first axis fastest). Mixing orders in one pipeline reliably causes the scrambled-data bug; prefer explicit transposes.

**Undoing a partition.** The inverse of a split-transpose-merge applies the inverse permutation, `np.argsort(p)`; `(0, 2, 1, 3)` is its own inverse.

**How einsum computes.** Without `optimize`, NumPy runs one nested loop over all letters: `'ij,jk,kl->il'` costs $O(n^4)$, while two matrix products cost $O(n^3)$. `optimize=True` searches for a good pairwise order and uses BLAS where possible; `np.einsum_path` shows the plan. For a plain two-operand product, `@` is at least as fast.

**Implicit mode.** Without `->`, the output is the once-used letters in alphabetical order, so `np.einsum('ba', x)` is a transpose. Always write the arrow.

## Key terms

| Term | Meaning
| --- | --- |
| C order (row-major) | Reading order in which the last axis changes fastest; the order `reshape` uses by default
| Reshape | Regroup the same sequence of elements into a new shape; a view when strides allow, a copy otherwise
| Transpose / axis permutation | Reorder the axes by permuting shape and strides; changes which element sits at each index, copies nothing
| `moveaxis` / `swapaxes` | Readable helpers to move one axis or swap two
| Concatenate vs stack | Lengthen an existing axis vs insert a new axis indexing the inputs
| Contiguous | Elements laid out in memory in exactly C order with no gaps; needed for a free merge of axes
| Sliding window view | Zero-copy, read-only array of overlapping windows built from strides
| einsum subscripts | Letters naming each axis; letters kept in the output stay, letters left out are summed

<div class="take">
<div class="lab">Key takeaway</div>

reshape only regroups elements in C order, transpose and moveaxis move whole axes, and the split-transpose-merge recipe cuts arrays into blocks; collect pieces in a list instead of concatenating in a loop, and let einsum state the shape contract in one string. When a shape looks right, still check one value.

</div>
