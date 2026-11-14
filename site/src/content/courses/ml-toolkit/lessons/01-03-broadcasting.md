---
id: "01.03"
course: "ml-toolkit"
module: "01"
slug: "broadcasting"
title: "Broadcasting"
keyIdeas: ["Align shapes from the **right**. Two dimensions are compatible when they are equal or one of them is 1; a missing leading dimension counts as 1. The result takes the larger size on each axis.", "Broadcasting never copies data: a stretched axis gets stride 0, so `(N, 1) - (1, M)` reads N + M numbers to build an N x M result.", "Use `None` (`np.newaxis`), `keepdims=True` or `reshape(-1, 1)` to put an axis where the rule needs it.", "The worst bugs are the silent ones: `(n, 1) - (n,)` gives an `(n, n)` matrix instead of an error, and on square data `X / X.sum(axis=1)` divides by the wrong axis."]
question: "A softmax over each row of `S` (shape (B, T)) is written as `np.exp(S) / np.exp(S).sum(axis=1)`. When does this raise, when does it silently give the wrong answer, and what is the fix?"
answer: "`np.exp(S).sum(axis=1)` has shape (B,). Broadcasting aligns it with the last axis of (B, T), so it lines up with the T columns, not the B rows. If B != T the shapes (B, T) and (B,) are incompatible on the last axis and NumPy raises an error. If B == T (for example a square attention matrix) the shapes match, and element (i, j) is divided by the sum of row j instead of row i. No error is raised, and the rows no longer sum to 1. The fix is `keepdims=True`, giving shape (B, 1), which broadcasts across each row: `e = np.exp(S - S.max(axis=1, keepdims=True)); e / e.sum(axis=1, keepdims=True)`. The max shift also prevents overflow."
sources: [{"title": "NumPy user guide: Broadcasting", "url": "https://numpy.org/doc/stable/user/basics.broadcasting.html", "note": "The official rules with diagrams and a worked distance example."}, {"title": "Python Data Science Handbook: Computation on arrays, broadcasting", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html", "note": "Visual walk-through of the three rules and centring and plotting examples."}, {"title": "Rougier: From Python to NumPy", "url": "https://www.labri.fr/perso/nrougier/from-python-to-numpy/", "note": "Free book; the vectorisation chapters show broadcasting replacing nested loops."}]
practice: ["pairwise-differences", "outer-sum", "row-scaling", "column-scaling", "normalize-columns"]
tags: ["broadcasting", "shapes"]
tools: ["NumPy"]
---

## The rule

When two arrays meet in an elementwise operation, NumPy compares their shapes starting from the **last** axis and moving left. For each pair of dimensions:

- equal sizes are fine;
- if one size is 1, that axis is stretched to match the other;
- if one array has fewer dimensions, it is padded with 1s on the left;
- anything else raises `ValueError: operands could not be broadcast together`.

~~~dg-flow Shapes are compared right-aligned; each size-1 or missing axis stretches to fit.
(8, 1, 6, 1) and (7, 1, 5) -(pad left)-> (8,1,6,1) and (1,7,1,5) -(max per axis)-> *(8, 7, 6, 5)
~~~

| A | B | result
| --- | --- | --- |
| (5, 3) | (3,) | (5, 3): B is added to every row
| (5, 3) | (5,) | error: 3 vs 5 on the last axis
| (5, 3) | (5, 1) | (5, 3): B is added to every column
| (5, 1) | (1, 4) | (5, 4): an outer operation
| (2, 3) | (3, 2) | error

The stretched axis is not materialised. NumPy gives it a stride of 0, so every step along that axis reads the same memory. `np.broadcast_to(v, (1000, 3))` returns a read-only view of 3 numbers that looks like 3000. The memory cost of a broadcast operation is the size of its **output**, not of a stretched input.

## Putting axes where you need them

Broadcasting only stretches size-1 axes, so most of the work is inserting them in the right place:

- `x[:, None]` turns shape `(n,)` into `(n, 1)` (a column); `x[None, :]` gives `(1, n)` (a row).
- `X.mean(axis=1, keepdims=True)` returns `(n, 1)` instead of `(n,)`, ready to broadcast back against `X`.
- `w.reshape(1, -1, 1, 1)` lines a per-channel vector up with an `(N, C, H, W)` image batch.

A good habit is to write the intended shapes next to the line: `# (B, T, 1) * (1, 1, D) -> (B, T, D)`. If the comment and the code disagree, the code is wrong.

## Outer operations and their memory cost

Adding a new axis to each operand gives every pairwise combination. Pairwise differences of `x` (N, D) and `y` (M, D) are `x[:, None, :] - y[None, :, :]`, of shape (N, M, D). This is concise but easy to make too large. With N = M = 10,000 and D = 128 in float32, the difference tensor is $10^8 \times 128 \times 4$ bytes, about 51 GB.

The squared Euclidean distance can be expanded instead:

$$\lVert x_i - y_j \rVert^2 = \lVert x_i \rVert^2 + \lVert y_j \rVert^2 - 2\, x_i \cdot y_j$$

```python
sq = (x**2).sum(1)[:, None] + (y**2).sum(1)[None, :] - 2 * x @ y.T   # (N, M)
d2 = np.maximum(sq, 0)    # cancellation can make tiny values negative
```

The largest intermediate is now the (N, M) matrix, 400 MB, and the matrix product runs in optimised BLAS. The price is precision: when two points are almost equal the subtraction cancels and can go slightly negative, hence the clamp. For very small distances in float32, compute the exact difference for the few pairs that matter.

## The silent shape bugs

Broadcasting errors are helpful. The dangerous cases are the shapes that broadcast when you did not mean them to:

1. **Column vs flat vector.** A model returns predictions of shape (n, 1) and labels are (n,). `pred - y` broadcasts to (n, n), and `np.square(pred - y).mean()` returns a plausible-looking but meaningless loss. Flatten one side, or assert `pred.shape == y.shape`.
2. **Wrong axis on square data.** To make each row sum to 1 you need `X / X.sum(axis=1, keepdims=True)`. Writing `X / X.sum(axis=1)` divides column $j$ by the sum of row $j$. On a non-square array this raises an error; on a square one (a 512 x 512 attention matrix, a confusion matrix) it runs and gives wrong numbers.
3. **Accidental huge outputs.** `a[:, None] == b` for two id arrays of a million elements tries to allocate a $10^{12}$-element boolean matrix. Use `np.isin` or a sort-based approach.

The AI Research course's tensor broadcasting lesson shows the same rules in PyTorch, which follows NumPy exactly.

<div class="ex">
<div class="lab">Worked example</div>

**standardise features with training statistics**

`X_train` is (50000, 20), `X_test` is (10000, 20). `mu = X_train.mean(axis=0)` and `sd = X_train.std(axis=0)` both have shape (20,).

`(X_test - mu) / sd` aligns (10000, 20) with (20,) on the right, so every row is shifted and scaled per column. No loop and no copies of `mu`.

Guard constant columns where `sd == 0`: `sd = np.where(sd == 0, 1.0, sd)`. Otherwise those columns become `nan` (0/0) or `inf`.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

right-align the shapes, stretch only the 1s, and write the expected shape of every broadcast in a comment. The bugs that hurt are the ones that broadcast without an error.

</div>
