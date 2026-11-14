---
id: "01.06"
course: "ml-toolkit"
module: "01"
slug: "shape-manipulation-and-einsum"
title: "Shape manipulation and einsum"
keyIdeas: ["`reshape` keeps the elements in C order and only regroups them; it is a view when the strides allow and a copy otherwise. It never moves an element to a different logical position.", "`transpose`, `swapaxes` and `moveaxis` reorder axes by permuting strides. Reshape-then-transpose-then-reshape is how you cut images into patches or windows.", "`concatenate` joins along an existing axis; `stack` creates a new one. Appending to an array inside a loop is quadratic, so collect pieces in a list and concatenate once.", "`einsum` names every axis with a letter: letters missing from the output are summed. `'bhqd,bhkd->bhqk'` is attention scores in one line."]
question: "You have token embeddings `X` of shape (B, T, D) and want multi-head form (B, H, T, D/H). Explain why `X.reshape(B, H, T, D // H)` is wrong, give the correct sequence of calls, and say which step copies memory."
answer: "`reshape` regroups elements in C order. For each batch item, the T x D block is read row by row and cut into H consecutive chunks of T x D/H values. That mixes tokens: the first `head` ends up holding the first T/H tokens' full embeddings rather than the first D/H features of every token. The heads must split the feature axis D, so: `X.reshape(B, T, H, D // H)`, which is a view because it only splits the last axis of a contiguous array, then `.transpose(0, 2, 1, 3)` to get (B, H, T, D/H), also a view since it only permutes strides. Nothing has copied yet. A copy happens only when a later operation needs contiguous memory, for example `np.ascontiguousarray`, a reshape that merges the transposed axes, or handing the array to code that requires C order."
sources: [{"title": "numpy.reshape", "url": "https://numpy.org/doc/stable/reference/generated/numpy.reshape.html", "note": "Element order, the -1 dimension, and when a copy is made."}, {"title": "NumPy reference: Array manipulation routines", "url": "https://numpy.org/doc/stable/reference/routines.array-manipulation.html", "note": "One page listing transpose, stack, split, tile, repeat and friends."}, {"title": "numpy.einsum", "url": "https://numpy.org/doc/stable/reference/generated/numpy.einsum.html", "note": "Subscript rules, implicit vs explicit mode, and the optimize argument."}, {"title": "sliding_window_view", "url": "https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html", "note": "Zero-copy windows and the warnings about writing to them."}]
practice: ["reshaping-arrays", "window-partition-and-reverse", "tile-and-diff", "concat-and-correlate", "outer-product"]
tags: ["reshape", "einsum"]
tools: ["NumPy"]
---

## reshape regroups, it does not reorder

`reshape` reads the elements in C order (last axis fastest) and writes them into the new shape in the same order. `np.arange(6).reshape(2, 3)` is `[[0, 1, 2], [3, 4, 5]]`. One dimension may be `-1` and is inferred. The result is a view when a single set of strides can describe the new shape over the existing buffer, which is always true for a contiguous array. Otherwise NumPy copies, which happens, for instance, when you reshape a transposed array.

The classic bug is using reshape when you meant transpose. An `(N, C, H, W)` batch converted to channels-last with `x.reshape(N, H, W, C)` has the right shape and scrambled pixels: the numbers were regrouped, not moved. The correct call is `x.transpose(0, 2, 3, 1)`. Shapes cannot catch this bug; only looking at values can.

## transpose and moveaxis

`x.transpose(p)` returns a view whose axis `i` is the input's axis `p[i]`, by permuting shape and strides. `np.swapaxes(x, a, b)` swaps two axes, and `np.moveaxis(x, src, dst)` moves one axis and keeps the others in order, which is easier to read for long shapes. All three are free; the cost comes later, when a function needs contiguous memory and copies.

Combining reshape and transpose cuts a grid into blocks. For an image of shape `(H, W)` and patch size `p`:

~~~dg-flow Patchify: split each spatial axis in two, bring the block indices together, flatten each patch.
(H, W) -(reshape)-> (H/p, p, W/p, p) -(transpose 0,2,1,3)-> (H/p, W/p, p, p) -(reshape)-> *(H/p * W/p, p*p)
~~~

```python
def patchify(img, p):
    H, W = img.shape
    return (img.reshape(H // p, p, W // p, p)
               .transpose(0, 2, 1, 3)
               .reshape(-1, p * p))
```

The first reshape is a view; the final reshape copies, because after the transpose the patch pixels are not adjacent in memory. Vision transformers and Swin's window attention use exactly this pattern.

## Joining and splitting

- `np.concatenate([a, b], axis=k)`: shapes must match on every axis except `k`, and the result is as long as both together on `k`.
- `np.stack([a, b], axis=k)`: shapes must be identical, and a new axis of length 2 is inserted at position `k`. Stacking 32 images of shape (H, W, 3) gives (32, H, W, 3).
- `np.vstack` and `np.hstack` are shortcuts with one surprise: `vstack` treats 1-D inputs as rows, so two `(3,)` vectors become a `(2, 3)` array, while `hstack` joins them into one `(6,)` vector.
- `np.split(x, 4)` needs equal parts; `np.array_split(x, 4)` allows uneven ones; `np.split(x, [10, 20])` cuts at the given indices.

Every `np.concatenate` allocates a new buffer and copies both inputs. Calling it in a loop to grow an array copies everything accumulated so far each time, $O(n^2)$ in total. Append the pieces to a Python list and concatenate once at the end, or preallocate with `np.empty` and fill slices.

## tile, repeat and windows

`np.tile([1, 2], 3)` gives `[1, 2, 1, 2, 1, 2]` (repeat the whole block); `np.repeat([1, 2], 3)` gives `[1, 1, 1, 2, 2, 2]` (repeat each element). Both allocate. If you only need the repeated array as an operand, broadcasting or `np.broadcast_to` gives the same values at no memory cost.

For overlapping windows, `np.lib.stride_tricks.sliding_window_view(x, w)` returns an `(n - w + 1, w)` view of a length-`n` signal. It costs no memory because consecutive windows share elements through the strides. It is read-only, and any operation that writes the windows out (an arithmetic result, a copy) materialises all `(n - w + 1) * w` values.

## einsum: name the axes

`np.einsum` takes a subscript string with one letter per axis of each operand. Letters that appear in the output are kept; letters that appear only in the inputs are multiplied and summed.

| subscripts | operation
| --- | --- |
| `'ij,jk->ik'` | matrix product
| `'bij,bjk->bik'` | batched matrix product
| `'bi,bi->b'` | row-wise dot products
| `'i,j->ij'` | outer product
| `'ii->'` | trace
| `'bhqd,bhkd->bhqk'` | attention scores per batch and head

`einsum` makes the shape contract explicit, which is its main value. For performance, `np.einsum(..., optimize=True)` chooses a good contraction order for three or more operands and can dispatch to BLAS. For simple products, `@` (`np.matmul`) is just as fast and more familiar.

<div class="ex">
<div class="lab">Worked example</div>

**patchify a 4 x 4 image with p = 2**

`img = np.arange(16).reshape(4, 4)`. The top-left patch should be pixels 0, 1, 4, 5.

The reshape to (2, 2, 2, 2) gives index `[block_row, row_in_block, block_col, col_in_block]`; transposing to `(0, 2, 1, 3)` puts both block indices first. The final reshape yields 4 rows: `[0, 1, 4, 5]`, `[2, 3, 6, 7]`, `[8, 9, 12, 13]`, `[10, 11, 14, 15]`.

Skipping the transpose and reshaping (4, 4) straight to (4, 4) returns the rows of the image, `[0, 1, 2, 3]` first, which is the wrong answer with the right shape.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

reshape only regroups C-ordered elements, transpose moves axes, and einsum documents the whole shape contract in one string. When a shape looks right, still check one value.

</div>
