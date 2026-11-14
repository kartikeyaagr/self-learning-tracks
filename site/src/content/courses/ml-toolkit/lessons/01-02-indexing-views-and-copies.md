---
id: "01.02"
course: "ml-toolkit"
module: "01"
slug: "indexing-views-and-copies"
title: "Indexing, views and copies"
keyIdeas: ["Basic indexing (integers, slices, `...`, `None`) returns a **view**: a new offset, shape and strides over the same memory. Advanced indexing (integer arrays, lists or boolean masks) always returns a **copy**, because arbitrary positions cannot be described by fixed strides.", "Writing through a view changes the original, and a small view keeps its whole parent buffer alive; call `.copy()` when you keep a small piece of a big array. Check what you hold with `np.shares_memory`, not by guessing from `.base`.", "Chained indexing like `a[mask][0] = 1` writes into a temporary copy and changes nothing, with no warning. Assign through one combined index, such as `a[mask] = 1` or `a[mask, 0] = 1`.", "`a[idx] += 1` with repeated indices adds only once per unique index; use `np.add.at` (or `np.bincount` for counts) when duplicates must accumulate.", "Index arrays on two axes are paired element by element; use `np.ix_` for a rectangular block. A full-shape boolean mask always returns a flat 1-D result.", "`np.where(cond, x, y)` evaluates both `x` and `y` in full before choosing, so it cannot prevent `log(0)` or division by zero; use a ufunc's `where=` argument together with `out=`."]
question: "You run `X = data[data[:, 0] > 0]` and then `X[:, 1] = 0`. Does `data` change? What changes if you instead write `X = data[5:100]`? And why does `data[data[:, 0] > 0, 1] = 0` behave differently from both? Finally, a teammate counts word frequencies with `freq[word_ids] += 1` and reports that every word in the batch appeared exactly once. What is going on?"
answer: "With the boolean mask, `X` is a copy because the mask is an advanced index: NumPy gathered the selected rows into a new buffer. Setting `X[:, 1] = 0` writes into that new buffer and `data` is unchanged; `np.shares_memory(data, X)` would print False.\n\nWith `X = data[5:100]`, `X` is a basic slice: a view with a new offset (5 rows in) over the same buffer and the same strides. `X[:, 1] = 0` therefore zeroes column 1 of rows 5 to 99 of `data` itself, and `np.shares_memory(data, X)` is True.\n\nThe single expression `data[data[:, 0] > 0, 1] = 0` is one `__setitem__` call on `data`, with the mask selecting rows and `1` selecting the column. Assignment writes straight into `data`'s buffer whatever the index type, so exactly the masked rows of `data` get column 1 set to 0, and no intermediate copy exists to absorb the write. The general rule: indexing and then assigning is two operations (the first may produce a throwaway copy), while assigning through one combined index is one operation on the original.\n\nThe word-count bug is the buffered-update trap. `freq[word_ids] += 1` is evaluated as `freq[word_ids] = freq[word_ids] + 1`: the right side is computed once from the old counts, and a word id that appears several times is written several times with the same value, so it gains 1 no matter how often it occurs. Use `np.add.at(freq, word_ids, 1)`, or, faster for plain counts, `freq += np.bincount(word_ids, minlength=len(freq))`. A quick test on `[3, 3, 5]` (expected counts 2 and 1) would have caught it."
sources: [{"title": "NumPy user guide: Indexing on ndarrays", "url": "https://numpy.org/doc/stable/user/basics.indexing.html", "note": "Basic vs advanced indexing, how index arrays combine, and the result shapes."}, {"title": "NumPy user guide: Copies and views", "url": "https://numpy.org/doc/stable/user/basics.copies.html", "note": "Which operations return views, and how to check with .base."}, {"title": "numpy.ufunc.at", "url": "https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html", "note": "Unbuffered in-place updates for repeated indices."}, {"title": "Python Data Science Handbook: Fancy indexing", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html", "note": "Worked examples, including the repeated-index gotcha."}]
practice: ["mutation-trap", "fancy-indexing", "boolean-masking", "basic-indexing", "filter-and-extract"]
tags: ["views", "fancy indexing", "aliasing"]
tools: ["NumPy"]
---

## What you will learn

- The three ways to index a NumPy array (basic, integer-array, boolean mask) and the shape each one returns.
- Why basic indexing gives a **view** that shares memory, while integer arrays and masks always give a **copy**, explained with offsets and strides.
- How to check what you are holding with `np.shares_memory` instead of guessing.
- The four classic aliasing traps: chained assignment, repeated indices in `+=`, functions that mutate their inputs, and paired index arrays.
- How `np.where` works in its two forms, and why the ufunc `where=` argument is the safe way to skip bad values.

*Before you start:* you need lesson 01.01 (arrays, dtypes and memory layout), especially the idea that an array is a header (shape, strides, offset) over a flat data buffer. Python list slicing should be familiar.

Lesson 01.01 showed that an ndarray is a small header over a flat block of bytes. This lesson asks the next question: when you pull part of an array out, do you get a new header over the *same* bytes, or new bytes? The answer decides whether later writes change the original, how much memory you keep, and whether some innocent lines silently do nothing.

## The idea in one picture

Take a small integer array and slice it:

```python
import numpy as np
x = np.arange(20).reshape(5, 4)   # values 0..19, 5 rows of 4
v = x[1:4, ::2]                   # rows 1-3, every second column
print(v)            # [[ 4  6] [ 8 10] [12 14]]
print(v.strides)    # (32, 16)   x.strides is (32, 8)
v[0, 0] = -1
print(x[1, 0])      # -1   <- the original changed
```

~~~dg-fig 01.02-slice-strides.svg
The slice picks six elements (blue). In memory they are not next to each other, but they sit at regular distances: start 32 bytes in, then step 16 bytes along a row and 32 bytes down a column. That regularity is what lets NumPy describe the slice without copying anything.
~~~

Nothing was copied. NumPy made a new header: "start 32 bytes in, step 32 bytes per row and 16 per column, shape (3, 2)". Because `v` uses the same bytes as `x`, setting `v[0, 0]` changed `x[1, 0]`.

Two words carry the whole lesson:

- A **view** is an array whose header points into another array's buffer. It owns no data. Creating it is instant, whatever the size, and writes through it land in the original.
- A **copy** is an array with its own freshly allocated buffer. Creating it costs time and memory in proportion to its size, and writes to it affect nobody else.

The rule: if the selection can be described by an offset plus one fixed step per axis, you get a view. If not, NumPy gathers the elements into new memory and you get a copy.

## Step 1: basic indexing returns views

**Basic indexing** is everything built from integers, slices (`start:stop:step`), the ellipsis `...` (meaning "all remaining axes") and `None` (also spelled `np.newaxis`, which inserts an axis of length 1). Each of these changes the header only:

- A slice `i:j:s` moves the offset to element `i` and multiplies that axis's stride by `s`; a negative step gives a negative stride, so `x[::-1]` is a view too.
- An integer drops the axis and moves the offset; `None` adds a length-1 axis; `...` fills in `:` for unmentioned axes.

The address of element $(i, j)$ of a 2-D view is

$$\text{addr}(i, j) = \text{offset} + i \cdot s_0 + j \cdot s_1,$$

where offset is the byte position of the view's first element inside the shared buffer, and $s_0$, $s_1$ are the view's strides in bytes along axis 0 and axis 1. This is the same formula as in lesson 01.01; a view just has different numbers in it.

<div class="ex">
<div class="lab">Worked example</div>

**find v[2, 1] from the strides alone**

`x = np.arange(20).reshape(5, 4)` holds int64, so each element is 8 bytes and `x.strides` is `(32, 8)`: one row is 4 elements x 8 bytes = 32 bytes.

`v = x[1:4, ::2]`. The slice `1:4` on axis 0 starts at row 1, so the offset is 1 x 32 = 32 bytes. Its step is 1, so $s_0$ stays 32. The slice `::2` on axis 1 starts at column 0 and has step 2, so $s_1$ = 2 x 8 = 16 bytes. Shape: rows 1, 2, 3 give 3; columns 0, 2 give 2. So `v` is (3, 2) with strides (32, 16).

Address of `v[2, 1]`: 32 + 2 x 32 + 1 x 16 = 32 + 64 + 16 = 112 bytes.

Element number in the flat buffer: 112 / 8 = 14. Since `x` holds 0..19 in order, `v[2, 1]` is 14.

Check against the 2-D picture: `v[2, 1]` is row 1 + 2 = 3 and column 0 + 2 x 1 = 2 of `x`, and `x[3, 2]` = 3 x 4 + 2 = 14. Both routes agree, and Python prints `v[2, 1]` as 14.

</div>

The same reasoning tells you which other operations can be views. `x.T` swaps the strides, so it is a view. `reshape` and `ravel` return a view when the new shape can be described by strides on the old buffer and silently fall back to a copy when it cannot: `x.ravel()` shares memory with `x`, but `x.T.ravel()` does not, because reading the transpose in row order jumps around memory. `flatten()` always copies.

Note that an integer on *every* axis, like `x[0, 1]`, returns a scalar, not a view.

## Step 2: advanced indexing returns copies

**Advanced indexing** (often called fancy indexing) happens whenever an index is an integer array, a list, or a boolean array. It lets you pick arbitrary positions in arbitrary order, with repeats. No single stride can describe "rows 0, 2, 2", so NumPy gathers the selected elements into a new buffer.

| index | example | result shape for a (5, 4) array | view or copy
| --- | --- | --- | --- |
| basic: ints and slices | `a[1:4, ::2]` | (3, 2) | view
| basic: new axis, ellipsis | `a[:, None, :]`, `a[..., 0]` | (5, 1, 4), (5,) | view
| advanced: integer arrays | `a[[0, 2, 2]]` | (3, 4) | copy
| advanced: boolean mask | `a[a > 5]` | (k,) for k True entries | copy

~~~dg-fig 01.02-view-vs-copy.svg
Left: a slice is a window onto `a`, so writing 99 through it changes `a[2]`. Right: an index list gathers the values into a separate buffer, so writing 99 there leaves `a` exactly as it was.
~~~

A **boolean mask** is a True/False array matching the array's shape (or one axis). `a[mask]` keeps the True positions. A full-shape mask always gives 1-D: rows may keep different numbers of elements, so NumPy lists them in row order.

~~~dg-fig 01.02-mask.svg
A full-shape mask selects 2, 2 and 3 elements from the three rows. Ragged rows cannot form a 2-D array, so the seven values come back as a flat copy in row order.
~~~

A mask on one axis only keeps the other axes intact: `a[a[:, 0] > 0]` selects whole rows (shape `(k, 4)`), and `a[a[:, 0] > 0, 1]` selects column 1 of those rows. Mixing a mask or index array with slices is allowed and still gives a copy.

## Step 3: check, do not guess

Two tools tell you what you are holding:

- `np.shares_memory(a, b)` returns True when the two arrays overlap in memory. This is the reliable test. (`np.may_share_memory` is a cheaper check that can report false positives.)
- `b.base` is the array that owns the memory a view points into, or `None` for an array that owns its data.

Be careful with `.base`: it points to the *owner*, not to the array you sliced. In the example above `x` is itself a view (`reshape` of the `arange` result), so `v.base is x` is False while `np.shares_memory(x, v)` is True. Prefer `shares_memory` in tests and assertions.

```python
x = np.arange(20).reshape(5, 4)
v = x[1:4, ::2]
c = x[[0, 2]]
print(np.shares_memory(x, v), np.shares_memory(x, c))   # True False
print(v.base is x, v.base is x.base)                    # False True
print(np.shares_memory(x, x.ravel()))                   # True  (view)
print(np.shares_memory(x, x.T.ravel()))                 # False (had to copy)
print(np.shares_memory(x, x.flatten()))                 # False (always a copy)
```

## Step 4: views share memory, on purpose

Views are a feature: slicing is free, and you can modify part of a big array in place. Two consequences follow:

- **Writes go through.** `col = X[:, 0]; col -= col.mean()` centres column 0 of `X` itself. With `X = [[1, 2], [3, 4], [5, 6]]` the first column of `X` becomes `[-2, 0, 2]`. Useful when you mean it, a nasty bug when you do not.
- **A view keeps its whole parent alive.** The view holds a reference to the owner of the buffer, so the buffer cannot be freed while the view exists.

~~~dg-fig 01.02-keepalive.svg
Keeping just ten values sliced from a 10-million-element float64 array keeps all 80 MB allocated, because the slice's base is the full buffer. Keeping a copy of the same ten values costs 80 bytes.
~~~

The fix for the second point is a single call: `first_rows = huge[:10].copy()`. Use it whenever you keep a small piece of a large array for long. To hand out a read-only view, set `v.flags.writeable = False`.

## Step 5: the aliasing traps

**Aliasing** means two names refer to the same memory. Most indexing bugs come from forgetting which side of the view/copy line an expression is on.

~~~dg-flow Chained indexing writes into a temporary copy, so the original never changes.
a[mask][0] = 9 -(a[mask] is advanced)-> temporary copy -(item 0 set)-> *copy discarded, a unchanged
~~~

1. **Chained assignment.** `a[mask] = 0` works, because it is a single `__setitem__` call on `a`: Python asks `a` itself to write into the masked positions. `a[mask][0] = 0` is two operations: `a[mask]` builds a copy, then `[0] = 0` writes into that temporary copy, which is thrown away. No error, no warning, no change. Combine the steps into one index instead: `a[np.flatnonzero(mask)[0]] = 0`.
2. **Repeated indices in in-place updates.** `counts[ids] += 1` with `ids = [3, 3, 5]` behaves like `counts[ids] = counts[ids] + 1`: the right side is computed once from the old values, then position 3 is written twice with the same value. It gains 1, not 2. This is called a **buffered** update, and it silently corrupts histograms and embedding gradients. `np.add.at(counts, ids, 1)` performs an **unbuffered** accumulation that visits each index in turn, and `np.bincount(ids, minlength=n)` is faster still for plain counts.
3. **Functions that mutate their inputs.** A helper that does `x[x < 0] = 0` or `x /= x.sum()` changes the caller's array, because the argument is the same object. Copy at the top (`x = x.copy()`) or build a new array (`np.maximum(x, 0)`, `x / x.sum()`).
4. **Paired advanced indices.** `a[[0, 1], [1, 2]]` does not give a 2x2 block; index arrays on different axes are paired element by element, so it returns the two elements `a[0, 1]` and `a[1, 2]`. For the block use `a[np.ix_([0, 1], [1, 2])]` or `a[[0, 1]][:, [1, 2]]`.

~~~dg-fig 01.02-pairing.svg
Index arrays on two axes are zipped together: [0, 1] with [1, 2] means the cells (0, 1) and (1, 2). np.ix_ instead forms every row-column combination and returns a 2x2 block.
~~~

The repeated-index trap fails quietly on real data: with 40 random ids over 10 bins, the buffered update reports 1 in every bin.

~~~dg-fig 01.02-repeated.svg
Left: buffered `counts[ids] += 1` caps every bin at 1, while `np.add.at` returns the true counts. Right: measured cost of counting one million ids (timings are machine-dependent; the Python loop was timed on 100,000 ids and scaled up).
~~~

Here `np.add.at` took a few tens of milliseconds for a million ids, the loop several times longer, and `np.bincount` about one millisecond. Use `bincount` for counting or summing weights into bins, and `add.at` for anything more general, such as adding whole rows.

<div class="ex">
<div class="lab">Worked example</div>

**embedding gradient with repeated tokens**

A batch has token ids `[7, 2, 7, 7]`. To keep the arithmetic visible, let each token's upstream gradient be a single number: `G = [0.5, 1.0, 0.25, 2.0]`. The gradient table `dE` starts at zeros, one entry per vocabulary word.

Correct answer: word 7 appears at positions 0, 2 and 3, so its gradient is 0.5 + 0.25 + 2.0 = 2.75. Word 2 appears once, so its gradient is 1.0.

Buffered `dE[ids] += G`: the right side is computed first as `dE[ids] + G` = `[0 + 0.5, 0 + 1.0, 0 + 0.25, 0 + 2.0]`. Then the assignments run in order: `dE[7] = 0.5`, `dE[2] = 1.0`, `dE[7] = 0.25`, `dE[7] = 2.0`. The last write wins, so `dE[7]` ends at 2.0, not 2.75. Word 7 silently loses 0.75 of its gradient.

Unbuffered `np.add.at(dE, ids, G)`: each addition reads the current value, so `dE[7]` goes 0 → 0.5 → 0.75 → 2.75. Correct.

With real embeddings `G` has shape (4, 16) and the same happens row by row. For large batches, sorting ids and using `np.add.reduceat` on contiguous runs is faster.

</div>

The PyTorch indexing lesson in the AI Research course covers the same view and copy rules for tensors, where `index_add_` plays the role of `np.add.at`.

## Indexing in code

The traps in one runnable script; each comment is the printed output.

```python
import numpy as np

a = np.array([3, -1, 4, -2, 5])
a[a < 0][0] = 0;  print(a)                 # [ 3 -1  4 -2  5]  (chained: no effect)
a[a < 0] = 0;     print(a)                 # [3 0 4 0 5]

counts = np.zeros(6, int); ids = [3, 3, 5]
counts[ids] += 1;        print(counts)     # [0 0 0 1 0 1]
counts = np.zeros(6, int)
np.add.at(counts, ids, 1); print(counts)   # [0 0 0 2 0 1]
print(np.bincount(ids, minlength=6))       # [0 0 0 2 0 1]
```

## np.where and its two forms

`np.where(cond, x, y)` builds a new array taking `x` where `cond` is True and `y` elsewhere, broadcasting all three. It is the vectorised `if`. With one argument, `np.where(cond)` equals `np.nonzero(cond)`: a tuple of index arrays, one per axis.

The trap is evaluation order. Python evaluates every argument first, so in `np.where(x > 0, np.log(x), 0.0)` all of `np.log(x)` is computed, including entries where `x` is 0. You get a `divide by zero` warning and `-inf` in an intermediate array. The final answer happens to be right here, but with `0/0`, gradients, or `np.errstate(all='raise')`, the bad values can leak or crash.

Prefer the ufunc `where=` argument, which never computes the masked entries:

```python
x = np.array([2.0, 0.0, 0.5])
out = np.zeros_like(x)
np.log(x, out=out, where=x > 0)
print(out)                       # [ 0.69314718  0.  -0.69314718], no warning
print(np.where(x > 0))           # (array([0, 2]),)
```

Always pass `out=` with `where=`; otherwise the skipped entries hold uninitialised memory.

## Reading and checking your indexing

When an indexing line writes data or touches a large array, check it the same way every time:

- **Predict the shape.** Each integer removes an axis, each slice keeps one, a full-shape mask gives 1-D, index arrays contribute their own shape. Print `.shape` and compare.
- **Ask view or copy.** Only ints, slices, `...` and `None` means a view; any list, integer array or mask means a copy. Confirm with `np.shares_memory`.
- **Test the write on a tiny array** like `np.arange(12).reshape(3, 4)` and inspect the original. This catches chained assignment at once.
- **Test repeats** on a case like `[3, 3, 5]` against `np.bincount`.
- **Watch memory.** If memory stays high after deleting a big array, look for kept slices with a large `kept.base.nbytes`.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Assignment runs, data unchanged | Chained index like `a[mask][0] = v` or `a[[1, 2]][:, 0] = v` | One combined index: `a[mask, 0] = v` or `a[np.flatnonzero(mask)[0]] = v`
| Original data changed unexpectedly | Wrote through a slice or `.T`, or a function mutated its argument | `.copy()` before editing, or return a new array
| Counts or gradients too small | `arr[ids] += v` with repeated ids | `np.add.at`, `np.bincount`, or `index_add_` in PyTorch
| Expected a block, got a short 1-D array | Index arrays on two axes are paired | `np.ix_` or index one axis at a time
| Result unexpectedly 1-D | Full-shape boolean mask flattens | Mask one axis (`a[row_mask]`) or use `np.where(mask, a, fill)`
| Memory never released | A small view keeps a huge base alive | Keep `huge[:k].copy()` instead
| `divide by zero` warnings despite `np.where` | Both branches evaluated in full | Ufunc `where=` with `out=`

## Going deeper

**Why advanced indexing cannot be a view.** A view is fully described by (offset, shape, strides), so the addresses it reaches form a regular lattice. Lists like `[0, 2, 2]` (repeat), `[3, 0, 1]` (reorder) or an irregular mask are not lattices. Even a regular list like `[0, 2, 4]` gives a copy: the rule depends on the *type* of index, not its values. Use the slice `0:5:2` for the view.

**Mixed basic and advanced indices.** The advanced indices are broadcast together and their shape replaces the indexed axes. If they are adjacent, that block stays in place; if a slice separates them, as in `a[[0, 1], :, [0, 1]]`, the block moves to the front. For `a` of shape (2, 5, 3) that gives (2, 5), a common source of surprising shapes.

**Assignment is not indexing.** `a[idx] = v` calls `__setitem__` on `a` and writes directly into `a`'s buffer, whatever kind of index `idx` is. This is why `a[mask] = 0` works even though `a[mask]` on its own would be a copy. The view/copy distinction only matters for what you get *back* from `a[idx]` when you read it.

**Overlapping in-place operations.** Since NumPy 1.13, ufuncs detect overlapping input and output (such as `a[1:] += a[:-1]`) and copy internally, so the result is as if there were no aliasing. For a running sum use `np.cumsum`.

**Cost model.** A view costs a constant, tiny amount whatever the size. A copy costs time proportional to the bytes moved, and random gathers are slower than contiguous copies because they miss the CPU cache.

## Key terms

| Term | Meaning
| --- | --- |
| View | An array that reads and writes another array's buffer through its own offset, shape and strides
| Copy | An array with its own newly allocated buffer; changes do not affect the source
| Basic indexing | Integers, slices, `...` and `None`; always produces a view (or a scalar)
| Advanced (fancy) indexing | Integer arrays, lists or boolean masks; always produces a copy when read
| Boolean mask | A True/False array that selects the elements where it is True
| `.base` | The array that owns the memory a view uses; `None` for an owner
| Aliasing | Two names referring to the same memory, so a write through one is seen by the other
| Buffered update | `a[idx] += v` computes the right side once, so repeated indices add only once
| `np.add.at` | Unbuffered in-place add that accumulates correctly over repeated indices

<div class="take">
<div class="lab">Key takeaway</div>

slices give views and index arrays or masks give copies. Before every in-place write, ask which one you are holding (check with `np.shares_memory`), assign through one combined index rather than a chain, and use `np.add.at` or `np.bincount` whenever indices can repeat.

</div>
