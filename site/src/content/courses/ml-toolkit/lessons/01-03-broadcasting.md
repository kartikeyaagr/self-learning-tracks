---
id: "01.03"
course: "ml-toolkit"
module: "01"
slug: "broadcasting"
title: "Broadcasting"
keyIdeas: ["**Broadcasting** lets an elementwise operation combine different shapes: `(4, 3) * (3,)` multiplies every row of the table by the same 3-element vector, with no loop and no copy.", "The rule: align shapes from the **right**, pad missing leading axes with 1, and on each axis the sizes must be equal or one of them must be 1. The result takes the larger size on each axis.", "Stretching never copies data: a stretched axis gets stride 0, so `(N, 1) - (1, M)` reads N + M numbers to build an N x M result. The cost is the size of the output.", "Use `None` (`np.newaxis`), `keepdims=True` or `reshape` to put a size-1 axis where the rule needs it; a `(n,)` vector always lines up with the last axis, never with the rows.", "Outer operations such as all pairwise distances can create huge intermediates; the expansion $\\lVert x\\rVert^2 + \\lVert y\\rVert^2 - 2x\\cdot y$ replaces an (N, M, D) tensor with one matrix product.", "The worst bugs are silent: `(n, 1) - (n,)` gives an `(n, n)` matrix instead of an error, and on square data `X / X.sum(axis=1)` divides by the wrong axis. Assert shapes and test with distinct sizes."]
question: "A softmax over each row of `S` (shape (B, T)) is written as `np.exp(S) / np.exp(S).sum(axis=1)`. When does this raise, when does it silently give the wrong answer, and what is the fix?"
answer: "`np.exp(S).sum(axis=1)` sums over the T columns and leaves one number per row, so it has shape (B,). Broadcasting right-aligns (B, T) with (B,): the vector is compared with the last axis, which is T, not with the B rows. If B != T, the last axis compares T with B, neither is 1, and NumPy raises `operands could not be broadcast together`. If B == T (for example a square attention matrix, or a batch size that happens to equal the sequence length), the shapes are compatible, the vector is stretched down the rows, and element (i, j) is divided by the sum of row j instead of row i. No error is raised, the values look like probabilities, but the rows no longer sum to 1, which you can confirm with `np.allclose(out.sum(axis=1), 1)`. The fix is `keepdims=True`, which gives the sums shape (B, 1); right-aligned against (B, T), the 1 stretches across the columns, so each row is divided by its own sum. Subtract the row maximum first, also with keepdims, so that `exp` cannot overflow for large scores: `e = np.exp(S - S.max(axis=1, keepdims=True)); out = e / e.sum(axis=1, keepdims=True)`. The shift does not change the result because it cancels between numerator and denominator. To catch this class of bug early, test with distinct sizes such as B = 3, T = 5 and assert the output shape."
sources: [{"title": "NumPy user guide: Broadcasting", "url": "https://numpy.org/doc/stable/user/basics.broadcasting.html", "note": "The official rules with diagrams and a worked distance example."}, {"title": "Python Data Science Handbook: Computation on arrays, broadcasting", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html", "note": "Visual walk-through of the three rules and centring and plotting examples."}, {"title": "Rougier: From Python to NumPy", "url": "https://www.labri.fr/perso/nrougier/from-python-to-numpy/", "note": "Free book; the vectorisation chapters show broadcasting replacing nested loops."}]
practice: ["pairwise-differences", "outer-sum", "row-scaling", "column-scaling", "normalize-columns"]
tags: ["broadcasting", "shapes"]
tools: ["NumPy"]
---

## What you will learn

- What broadcasting is: how NumPy combines arrays of *different* shapes in one elementwise operation, without loops and without copying data.
- The three-part shape rule (right-align, pad with 1, stretch only the 1s), and how to predict the output shape before you run the line.
- How to put a size-1 axis exactly where you need it with `None`, `keepdims=True` and `reshape`.
- How "outer" operations such as all pairwise distances work, and how to keep their memory under control.
- The silent shape bugs that run without an error and give wrong numbers, and the habits that catch them.

*Before you start:* you should know what an array's shape, dtype and strides are (01.01) and the difference between a view and a copy (01.02). Basic arithmetic on arrays (`a + b`, `a * 2`) is assumed.

In 01.02 you saw that slicing returns a **view**: new shape and strides over the same memory. Broadcasting uses that trick: NumPy views the smaller array with some strides set to 0, so the same numbers are re-read and the arrays behave as if they had equal shapes, with nothing copied.

## The idea in one picture

A small shop sells three products. You have a table of units sold over four days (rows are days, columns are products) and a price list with one price per product:

```python
import numpy as np
units = np.array([[3, 0, 2],      # day 0
                  [1, 4, 0],      # day 1
                  [5, 2, 1],      # day 2
                  [0, 1, 3]])     # day 3      shape (4, 3)
price = np.array([2.5, 4.0, 10.0])  #          shape (3,)

revenue = units * price           # (4, 3) * (3,) -> (4, 3)
print(revenue)
# [[ 7.5  0.  20. ]
#  [ 2.5 16.   0. ]
#  [12.5  8.  10. ]
#  [ 0.   4.  30. ]]
print(revenue.sum(axis=1))        # [27.5 18.5 30.5 34. ]   revenue per day
```

You never asked NumPy to repeat the price list. It saw that the last sizes agree (3 and 3) and treated the prices as a `(4, 3)` table with the same row four times.

~~~dg-fig 01.03-idea.svg
The price row (orange) is stored once. The dashed rows below it are not real memory: they are the same three numbers read again, which is what a stride of 0 means. The result (green) is a normal (4, 3) array.
~~~

**Broadcasting** is this automatic stretching of arrays so that an elementwise operation can combine different shapes. The stretched copies are **virtual**: never written to memory, so `units * price` costs no more than multiplying equal-sized arrays.

The alternatives, a Python loop over days or `np.tile(price, (4, 1))`, are slower (01.04) or waste memory.

## Step 1: line the shapes up from the right

NumPy decides whether two shapes can be combined by writing them one above the other, **aligned on the right**, and comparing them one axis at a time from the last axis leftwards.

In the shop example:

| | axis -2 | axis -1
| --- | --- | --- |
| units | 4 | 3
| price | (missing) | 3

Last axis: 3 against 3, fine. Axis -2: 4 against nothing. Right-alignment means a vector of per-column values (prices, means, weights) lines up with the columns of a table.

## Step 2: pad missing axes with 1

If one shape is shorter, NumPy pretends it has extra axes of size 1 on the **left**. So `(3,)` is treated as `(1, 3)`, and `(7, 1, 5)` next to a four-axis array is treated as `(1, 7, 1, 5)`. Padding only happens on the left: NumPy never turns `(3,)` into `(3, 1)` for you (step 4 shows how).

## Step 3: each axis must match or be 1

Now both shapes have the same number of axes. For each axis, NumPy applies one test:

- the two sizes are **equal**: fine, that size goes into the result;
- one of them is **1**: that array is stretched along this axis to the other size;
- anything else: `ValueError: operands could not be broadcast together with shapes ...`.

The output shape is simply the larger size on each axis. In symbols, for two shapes padded to the same length $d$,

$$\text{out}_k = \max(a_k, b_k) \quad \text{allowed only if } a_k = b_k \text{ or } \min(a_k, b_k) = 1,$$

where $a_k$ and $b_k$ are the sizes of the two arrays on axis $k$.

~~~dg-fig 01.03-align.svg
Shapes written right-aligned in slots. Dotted slots are padding (size 1). Each output slot takes the larger size; the middle pair fails because 3 and 5 differ and neither is 1.
~~~

~~~dg-flow Shapes are compared right-aligned; each size-1 or missing axis stretches to fit.
(8, 1, 6, 1) and (7, 1, 5) -(pad left)-> (8,1,6,1) and (1,7,1,5) -(max per axis)-> *(8, 7, 6, 5)
~~~

Check each line yourself before reading the result column:

| A | B | result
| --- | --- | --- |
| (5, 3) | (3,) | (5, 3): B is added to every row
| (5, 3) | (5,) | error: 3 vs 5 on the last axis
| (5, 3) | (5, 1) | (5, 3): B is added to every column
| (5, 1) | (1, 4) | (5, 4): an outer operation
| (2, 3) | (3, 2) | error
| (8, 1, 6, 1) | (7, 1, 5) | (8, 7, 6, 5)

`np.broadcast_shapes` applies the rule without creating data:

```python
np.broadcast_shapes((8, 1, 6, 1), (7, 1, 5))   # (8, 7, 6, 5)
np.broadcast_shapes((5, 3), (5, 1))            # (5, 3)
np.ones((5, 3)) + np.ones(5)
# ValueError: operands could not be broadcast together with shapes (5,3) (5,)
```

Note the second row: a `(5,)` vector does **not** line up with the 5 rows of a `(5, 3)` matrix; right-alignment compares it with the 3 columns. This one fact explains most beginner confusion.

## Step 4: put axes where you need them

Since only size-1 axes stretch, most of the work is inserting a size-1 axis in the right place. Three tools do this without copying:

- **`None`** (also spelled `np.newaxis`) in an index inserts a new axis of size 1. `x[:, None]` turns shape `(n,)` into a column `(n, 1)`; `x[None, :]` gives a row `(1, n)`.
- **`keepdims=True`** on a reduction (`sum`, `mean`, `max`, `std`) keeps the reduced axis with size 1, so `X.mean(axis=1, keepdims=True)` has shape `(n, 1)` instead of `(n,)` and broadcasts straight back against `X`.
- **`reshape`** places axes anywhere: `w.reshape(1, -1, 1, 1)` turns a per-channel vector of length C into shape `(1, C, 1, 1)`, ready to multiply an `(N, C, H, W)` image batch.

```python
X = np.array([[1., 2, 3],
              [4,  5, 6]])                 # (2, 3)
X.mean(axis=1).shape                       # (2,)   lines up with columns: wrong for rows
X.mean(axis=1, keepdims=True).shape        # (2, 1) lines up with rows
X - X.mean(axis=1, keepdims=True)          # centre each row
# [[-1.  0.  1.]
#  [-1.  0.  1.]]

imgs = np.ones((8, 3, 32, 32))             # (N, C, H, W)
w = np.array([0.5, 1.0, 2.0])              # one scale per channel
(imgs * w.reshape(1, -1, 1, 1))[0, :, 0, 0]  # [0.5 1.  2. ]
```

Without `keepdims`, `X - X.mean(axis=1)` combines `(2, 3)` with `(2,)` and fails. That error is lucky; step 6 shows unlucky cases. A good habit: write intended shapes beside every broadcast, e.g. `# (B, T, 1) * (1, 1, D) -> (B, T, D)`. If comment and code disagree, the code is wrong.

<div class="ex">
<div class="lab">Worked example</div>

**standardise test features with training statistics**

`X_train` has shape (50000, 20) and `X_test` has shape (10000, 20). Compute `mu = X_train.mean(axis=0)` and `sd = X_train.std(axis=0)`. Reducing over axis 0 removes the rows, so both have shape (20,): one number per column.

Shape check for `(X_test - mu) / sd`: right-align (10000, 20) with (20,). Last axis: 20 vs 20, equal. Axis -2: 10000 vs a padded 1, so `mu` is stretched down all 10000 rows. Result: (10000, 20).

So element $(i, j)$ becomes $(x_{ij} - \mu_j) / \sigma_j$, using its own column's statistics, with no loop or copy.

A trap: if column 3 is constant in the training data (say always 7.0), then `sd[3]` is 0 and the test column becomes $\pm\infty$ (or `nan` where the value is exactly 7, since 0/0). Guard it first: `sd = np.where(sd == 0, 1.0, sd)`. Here `np.where` itself broadcasts a scalar 1.0 against the (20,) array.

</div>

## Step 5: outer operations, every pair at once

Insert a new axis into *each* operand, at different positions, and broadcasting builds every combination of their elements. A column of 4 numbers plus a row of 3 numbers gives a 4 by 3 table of all sums:

```python
x = np.array([0, 10, 20, 30])
y = np.array([1, 2, 3])
x[:, None] + y[None, :]          # (4, 1) + (1, 3) -> (4, 3)
# [[ 1  2  3]
#  [11 12 13]
#  [21 22 23]
#  [31 32 33]]
```

~~~dg-fig 01.03-outer.svg
In an outer operation both inputs stretch: the column across, the row down. Only 4 + 3 numbers are stored, but the result has 4 x 3 = 12 entries.
~~~

The same pattern computes all pairwise differences between two point sets. If `x` has shape (N, D) (N points with D coordinates) and `y` has shape (M, D), then `x[:, None, :] - y[None, :, :]` has shape (N, M, D): entry $(i, j)$ is the vector from point $j$ of `y` to point $i$ of `x`. Squaring and summing over the last axis gives all squared distances, shape (N, M).

This is correct, but the (N, M, D) intermediate can be enormous. With N = M = 10,000 points of D = 128 float32 numbers, it holds $10^8 \times 128$ values at 4 bytes each: about 51 GB. The final answer is only (N, M), 400 MB.

The fix is algebra. Expanding the square gives

$$\lVert x_i - y_j \rVert^2 = \lVert x_i \rVert^2 + \lVert y_j \rVert^2 - 2\, x_i \cdot y_j ,$$

where $\lVert x_i \rVert^2$ is the sum of squares of point $i$'s coordinates and $x_i \cdot y_j$ is a dot product. The first term is a column (N, 1), the second a row (1, M), and all the dot products at once are the matrix product `x @ y.T`, shape (N, M). Broadcasting adds the three pieces:

```python
def sq_dists(x, y):                               # x (N, D), y (M, D)
    sq = (x**2).sum(1)[:, None] + (y**2).sum(1)[None, :] - 2 * x @ y.T   # (N, M)
    return np.maximum(sq, 0)    # rounding can make tiny values slightly negative
```

<div class="ex">
<div class="lab">Worked example</div>

**pairwise squared distances by hand, both ways**

Points `x = [[0, 0], [3, 4]]` (N = 2) and `y = [[0, 0], [1, 1], [6, 8]]` (M = 3), so D = 2 and the answer has shape (2, 3).

Direct way: `x[:, None, :]` is (2, 1, 2) and `y[None, :, :]` is (1, 3, 2), so the difference is (2, 3, 2). For $i = 1, j = 1$: $(3, 4) - (1, 1) = (2, 3)$, squared and summed $4 + 9 = 13$. For $i = 1, j = 2$: $(3, 4) - (6, 8) = (-3, -4)$, giving $9 + 16 = 25$.

Expanded way. Squared norms: $\lVert x \rVert^2 = (0, 25)$ as a column, $\lVert y \rVert^2 = (0, 2, 100)$ as a row. Dot products `x @ y.T` $= [[0, 0, 0], [0, 7, 50]]$, because $(3,4)\cdot(1,1) = 7$ and $(3,4)\cdot(6,8) = 18 + 32 = 50$.

Row 0: $0 + (0, 2, 100) - 2 \times (0, 0, 0) = (0, 2, 100)$. Row 1: $25 + (0, 2, 100) - 2 \times (0, 7, 50) = (25, 13, 25)$.

Both ways give $[[0, 2, 100], [25, 13, 25]]$. The expanded way never built the (2, 3, 2) tensor.

</div>

The next figure times both methods and a Python double loop with D = 64.

~~~dg-fig 01.03-distance.svg
Left: measured time for all pairwise squared distances (single-threaded BLAS on this course's build machine; your numbers will differ, the gaps will not). Right: the largest array each method allocates. At N = 1600 the difference tensor is 1311 MB against 20 MB.
~~~

On the machine used for this lesson, at N = 400 the double loop took about 280 ms, the broadcast difference tensor about 45 ms and the expanded form about 1 ms. At N = 1600 the broadcast version took about 670 ms and the expanded one about 16 ms. The broadcast version is limited by its huge intermediate; the expanded one is a single optimised (BLAS) matrix product. Broadcasting expresses an operation; it does not guarantee the cheapest one.

## Step 6: the silent bugs

The dangerous cases are shapes that broadcast when you did not intend them to, so the code runs and returns a plausible number.

**Column against flat vector.** A model returns predictions of shape (n, 1) and the labels have shape (n,). Right-align (n, 1) with (n,): last axis 1 vs n, stretch; axis -2 n vs padded 1, stretch. The result is an (n, n) table of every prediction minus every label.

```python
pred = np.array([[2.0], [3.0], [5.0], [6.0], [4.0]])   # (5, 1)
y    = np.array([2.1, 2.9, 5.2, 5.8, 4.1])             # (5,)
(pred - y).shape                    # (5, 5)  not (5,)
np.mean((pred - y) ** 2)            # 3.902   meaningless
np.mean((pred.ravel() - y) ** 2)    # 0.022   the real MSE
```

**Wrong axis on square data.** To make each row of a matrix sum to 1 you need `X / X.sum(axis=1, keepdims=True)`. Writing `X / X.sum(axis=1)` lines the (n,) vector of row sums up with the *columns*, so column $j$ is divided by the sum of row $j$. On a non-square matrix this raises an error. On a square one (a confusion matrix, an attention matrix, a transition matrix) it runs.

~~~dg-fig 01.03-silent.svg
Left: a (5, 1) minus (5,) subtraction silently becomes a 5 x 5 table; only the diagonal holds the residuals you wanted, and the MSE over the whole table is 3.9 instead of 0.022. Right: normalising a 5 x 5 matrix without keepdims leaves row sums between 0.85 and 1.56 instead of 1.
~~~

**Accidentally huge outputs.** `a[:, None] == b` for two million-id arrays asks for a $10^{12}$-element table (a terabyte). Use `np.isin(a, b)`.

## Broadcasting in code

The rules hold for every elementwise operation: arithmetic, comparisons, `np.where`, `np.maximum` and all ufuncs (01.04). Three habits in code:

```python
assert pred.shape == y.shape, (pred.shape, y.shape)   # one line, catches the (n,1) vs (n,) bug
np.broadcast_shapes(a.shape, b.shape)                  # predict the result before allocating it

def softmax(S):                                         # S (B, T), softmax over each row
    e = np.exp(S - S.max(axis=1, keepdims=True))        # (B, T) - (B, 1)
    return e / e.sum(axis=1, keepdims=True)             # (B, T) / (B, 1)
```

In-place operators are stricter: `a += b` writes into `a`, so the result must already have `a`'s shape. `z = np.zeros(3); z += np.ones((2, 3))` fails; `z + np.ones((2, 3))` is fine.

## Checking the result

- **Print shapes, not just values.** A result with more axes, or a bigger axis, than your shape comment predicted is the signature of an unintended broadcast.
- **Test with distinct sizes** like (5, 3), never (4, 4): a wrong alignment then raises instead of running.
- **Check an invariant.** After row-normalising, `np.allclose(P.sum(axis=1), 1)`; after standardising, training column means near 0 and standard deviations near 1.
- **Compare with a tiny loop** on two or three entries, as the distance example did.
- **Watch memory.** If a line takes seconds or gigabytes, multiply out `np.broadcast_shapes` of the operands: the output size is the cost.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| `operands could not be broadcast together` | a vector meant for rows met the column axis | `keepdims=True`, or `v[:, None]`
| loss is plausible but far too large, training does not improve | (n, 1) minus (n,) became (n, n) | flatten one side; `assert pred.shape == y.shape`
| rows of a "normalised" square matrix do not sum to 1 | reduction without `keepdims` divided the wrong axis | `X / X.sum(axis=1, keepdims=True)`
| `MemoryError` or the machine starts swapping | an outer operation built an (N, M, D) or (N, M) intermediate that is too big | expand algebraically, process in chunks, or use `np.isin` / sorting
| `nan` or `inf` after standardising | a constant column has standard deviation 0 | replace zero `sd` with 1 before dividing
| `ValueError: assignment destination is read-only` | writing into `np.broadcast_to` output | call `.copy()` first if you really need a writable array
| "non-broadcastable output operand" | `a += b` where the result is larger than `a` | use `a = a + b`

## Going deeper

**How stretching works in memory.** Recall from 01.01 that an array's strides say how many bytes to jump to move one step along each axis. A stretched axis gets stride 0: stepping along it moves zero bytes, so every step reads the same element. You can see this directly:

```python
v = np.broadcast_to(price, (4, 3))
v.strides, price.strides          # (0, 8)  (8,)
v.flags.writeable                 # False
np.shares_memory(v, price)        # True
```

The view is read-only on purpose: writing one dashed cell in the first figure would change all four rows, since they share memory. `np.broadcast_arrays` returns such views of all operands at the common shape.

**Cost model.** Inputs are never materialised, so memory cost is the size of the *output*: (N, 1) and (1, M) read N + M numbers to write N x M, in roughly one pass. But a chain like `((x[:, None] - y[None]) ** 2).sum(-1)` creates a full-size temporary at each step. When the output is the bottleneck, the options are a matrix product (as in the distance trick), `np.einsum` (01.06), which can fuse a multiply and a sum, or processing the first axis in chunks of a few thousand rows.

**Precision of the distance trick.** When $x_i$ and $y_j$ are close, the expanded formula subtracts nearly equal large numbers, and in float32 tiny distances lose most digits. The clamp hides negative values, not the error; recompute the exact difference for the few pairs where small distances matter.

**Broadcasting beyond elementwise operations.** `np.matmul` (the `@` operator) treats the last two axes as matrices and broadcasts all axes before them: `(B, n, k) @ (k, m)` gives `(B, n, m)`, multiplying every matrix in the batch by the same weight matrix. NumPy calls this "core dimensions plus broadcast batch dimensions" pattern a generalised ufunc. PyTorch and JAX follow the same rules; the AI Research course's tensor broadcasting lesson shows the PyTorch version.

## Key terms

| Term | Meaning
| --- | --- |
| Broadcasting | Combining arrays of different shapes elementwise by virtually stretching size-1 axes
| Right-alignment | Comparing shapes from the last axis leftwards
| Padding | Treating missing leading axes as size 1
| Stride 0 | A step size of zero bytes, so an axis re-reads the same memory; how stretching is implemented
| `None` / `np.newaxis` | Index that inserts a new axis of size 1
| `keepdims=True` | Keep a reduced axis with size 1 so the result broadcasts back against the input
| Outer operation | Broadcasting a column against a row to get every pairwise combination
| Silent broadcast | A shape combination that runs without error but computes something unintended

<div class="take">
<div class="lab">Key takeaway</div>

right-align the shapes, pad missing axes with 1, and stretch only the 1s; the output takes the larger size on each axis and costs memory only for itself. Insert axes with `None` or `keepdims=True`, write the expected shape of every broadcast in a comment, and test with distinct sizes, because the bugs that hurt are the ones that broadcast without an error.

</div>
