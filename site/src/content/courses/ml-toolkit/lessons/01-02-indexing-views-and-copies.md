---
id: "01.02"
course: "ml-toolkit"
module: "01"
slug: "indexing-views-and-copies"
title: "Indexing, views and copies"
keyIdeas: ["Basic indexing (integers, slices, `...`, `None`) returns a **view** that shares memory with the original. Advanced indexing (integer arrays or boolean masks) always returns a **copy**.", "Writing through a view changes the original. Writing to a copy produced by chained indexing like `a[mask][0] = 1` changes nothing, and NumPy gives no warning.", "`a[idx] += 1` with repeated indices adds only once per unique index; use `np.add.at` when duplicates must accumulate.", "`np.where(cond, x, y)` selects element by element but evaluates both `x` and `y` in full first, so it cannot protect you from `log(0)` or division by zero."]
question: "You run `X = data[data[:, 0] > 0]` and then `X[:, 1] = 0`. Does `data` change? What changes if you instead write `X = data[5:100]`? And why does `data[data[:, 0] > 0, 1] = 0` behave differently from both?"
answer: "With the boolean mask, `X` is a copy because the mask is an advanced index. Setting `X[:, 1] = 0` writes into that new buffer and `data` is unchanged. With `X = data[5:100]`, `X` is a basic slice: a view with a new offset over the same buffer, so `X[:, 1] = 0` zeroes column 1 of rows 5 to 99 of `data`. The single expression `data[data[:, 0] > 0, 1] = 0` is one `__setitem__` call on `data` itself, with the mask selecting rows and `1` selecting the column, so it modifies exactly the masked rows of `data`. No intermediate copy is created. The rule is that indexing a view or copy and then assigning is two operations, while assigning through one combined index is one."
sources: [{"title": "NumPy user guide: Indexing on ndarrays", "url": "https://numpy.org/doc/stable/user/basics.indexing.html", "note": "Basic vs advanced indexing, how index arrays combine, and the result shapes."}, {"title": "NumPy user guide: Copies and views", "url": "https://numpy.org/doc/stable/user/basics.copies.html", "note": "Which operations return views, and how to check with .base."}, {"title": "numpy.ufunc.at", "url": "https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html", "note": "Unbuffered in-place updates for repeated indices."}, {"title": "Python Data Science Handbook: Fancy indexing", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html", "note": "Worked examples, including the repeated-index gotcha."}]
practice: ["mutation-trap", "fancy-indexing", "boolean-masking", "basic-indexing", "filter-and-extract"]
tags: ["views", "fancy indexing", "aliasing"]
tools: ["NumPy"]
---

## Three kinds of indexing

NumPy has three indexing mechanisms, and the one you use decides whether you get a view or a copy.

| index | example | result shape for a (5, 4) array | view or copy
| --- | --- | --- | --- |
| basic: ints and slices | `a[1:4, ::2]` | (3, 2) | view
| basic: new axis, ellipsis | `a[:, None, :]`, `a[..., 0]` | (5, 1, 4), (5,) | view
| advanced: integer arrays | `a[[0, 2, 2]]` | (3, 4) | copy
| advanced: boolean mask | `a[a > 0]` | (k,) for k True entries | copy

Basic indexing can always be expressed as a new offset plus new strides on the same buffer: `a[::2]` just doubles the stride of axis 0. Advanced indexing picks arbitrary positions in arbitrary order, which no single stride can describe, so NumPy gathers the elements into a new buffer. A boolean mask with the same shape as the array flattens the result to 1-D because the number of selected elements per row varies.

Two tools tell you what you have: `np.shares_memory(a, b)` and `b.base is a` (a view keeps a reference to the array that owns the memory).

## Views share memory, on purpose

Views make slicing free, and that is usually what you want. Two consequences to keep in mind:

- **Writes go through.** `col = X[:, 0]; col -= col.mean()` centres column 0 of `X` itself.
- **A view keeps its whole parent alive.** `first_rows = huge[:10]` holds the entire multi-gigabyte buffer in memory for as long as `first_rows` exists. Use `huge[:10].copy()` when you keep a small piece of a big array.

The same rules apply to `reshape` and `ravel` (views when the strides allow it) versus `flatten` (always a copy), and to `x.T`.

## The aliasing traps

~~~dg-flow Chained indexing writes into a temporary copy, so the original never changes.
a[mask][0] = 9 -(a[mask] is advanced)-> temporary copy -(item 0 set)-> *copy discarded, a unchanged
~~~

1. **Chained assignment.** `a[mask] = 0` works, because it is a single `__setitem__` call on `a`. `a[mask][0] = 0` first builds a copy with `a[mask]`, then writes into the copy. Combine the steps into one index instead: `a[np.flatnonzero(mask)[0]] = 0`.
2. **Repeated indices in in-place updates.** `counts[ids] += 1` with `ids = [3, 3, 5]` behaves like `counts[ids] = counts[ids] + 1`: the right side is computed once, and position 3 is written twice with the same value, so it gains 1, not 2. This silently corrupts histograms and embedding gradients. `np.add.at(counts, ids, 1)` does an unbuffered accumulation, and `np.bincount(ids, minlength=n)` is faster for plain counts.
3. **Functions that mutate their inputs.** A helper that does `x[x < 0] = 0` or `x /= x.sum()` changes the caller's array. Copy at the top (`x = x.copy()`) or return a new array (`np.maximum(x, 0)`).
4. **Mixing advanced indices.** `a[[0, 1], [1, 2]]` does not give a 2x2 block; it pairs the indices and returns the 2 elements `a[0, 1]` and `a[1, 2]`. For the block use `a[np.ix_([0, 1], [1, 2])]` or `a[[0, 1]][:, [1, 2]]`.

```python
import numpy as np
x = np.arange(12).reshape(3, 4)
v = x[:, 1]          # view: v[0] = 100 changes x[0, 1]
c = x[[0, 2]]        # copy: rows 0 and 2
x[x % 2 == 0] = -1   # one __setitem__ call: modifies x
g = np.zeros(5)
np.add.at(g, [3, 3, 1], 1.0)   # g = [0, 1, 0, 2, 0]
```

## np.where and its two forms

`np.where(cond, x, y)` builds an array that takes `x` where `cond` is True and `y` elsewhere, broadcasting all three. It is the vectorised `if`. With only one argument, `np.where(cond)` is `np.nonzero(cond)`: a tuple of index arrays, one per axis.

The trap is evaluation order. Python evaluates `np.log(x)` for **every** element before `where` chooses, so `np.where(x > 0, np.log(x), 0.0)` still triggers divide-by-zero warnings and can create `nan` or `inf` intermediates. It produces the right answer for `log`, but with gradients or `0/0` the bad values can leak into later steps. Prefer the ufunc `where=` argument, which never computes the masked entries:

```python
out = np.zeros_like(x, dtype=float)
np.log(x, out=out, where=x > 0)   # entries where x <= 0 stay 0
```

Always pass `out=` together with `where=`; without it the unselected entries are uninitialised memory.

<div class="ex">
<div class="lab">Worked example</div>

**embedding gradient with repeated tokens**

A batch has token ids `[7, 2, 7, 7]` and upstream gradients `G` of shape (4, 16). The gradient for the embedding table `E` (shape (vocab, 16)) must add all three rows that came from token 7.

`dE[ids] += G` gives row 7 only the *last* gradient written. `np.add.at(dE, ids, G)` gives row 7 the sum of three gradients, which is the correct value. On large batches `np.add.at` is slow; sorting the ids and using `np.add.reduceat` on contiguous runs is the fast vectorised alternative.

</div>

The PyTorch indexing lesson in the AI Research course covers the same view and copy rules for tensors, where `index_add_` plays the role of `np.add.at`.

<div class="take">
<div class="lab">Key takeaway</div>

slices give views and index arrays or masks give copies. Before every in-place write, ask which one you are holding, and use `np.add.at` whenever indices can repeat.

</div>
