---
id: "01.05"
course: "ml-toolkit"
module: "01"
slug: "reductions-and-the-axis-argument"
title: "Reductions and the axis argument"
keyIdeas: ["`axis=k` names the axis that **disappears**: for `X` of shape `(N, D)`, `X.sum(axis=0)` has shape `(D,)` (one value per column) and `X.sum(axis=1)` has shape `(N,)` (one per row). A tuple such as `axis=(0, 1, 2)` collapses several axes at once.", "`keepdims=True` keeps the reduced axis as length 1 so the result broadcasts straight back against the input, as in `X - X.mean(axis=1, keepdims=True)`. Without it, square data can give wrong numbers with no error.", "`argmax` returns positions, not values, and breaks ties by taking the first occurrence; `np.take_along_axis` turns per-row positions back into values, and `argpartition` finds the top k in linear time but unordered.", "Cumulative operations (`cumsum`, `np.maximum.accumulate`) keep the axis and return running results. In float32 they drift on long series, unlike `np.sum`, which uses pairwise summation; accumulate in float64 instead.", "NaN propagates through `sum` and `mean`. The `nan*` variants skip it, but an all-NaN row gives `nan` with a warning from `nanmean`, `0` from `nansum` and an error from `nanargmax`. `np.std` uses `ddof=0`, pandas uses `ddof=1`."]
question: "Sensor data `X` has shape (days, sensors, 24) with hourly readings and some NaNs. Write the calls for (a) each sensor's mean over all days and hours, and (b) for each day, the hour at which the average over sensors peaked. What happens in (b) if one day is entirely NaN, and how would you handle it?"
answer: "(a) Collapse days and hours and keep sensors: `np.nanmean(X, axis=(0, 2))`, shape (sensors,). Cross out positions 0 and 2 of (days, sensors, 24) and only sensors is left, which confirms the axis choice. (b) First average over sensors for each day and hour: `H = np.nanmean(X, axis=1)`, shape (days, 24). Then collapse the hour axis to a position: `np.nanargmax(H, axis=1)`, shape (days,). If one day is entirely NaN, `nanmean` returns a NaN row for it and emits a RuntimeWarning ('Mean of empty slice'), and `nanargmax` then raises ValueError ('All-NaN slice encountered') because that row has no valid entry, so the whole call fails. Handle it explicitly: compute `ok = ~np.isnan(H).all(axis=1)`, replace NaNs with `-np.inf` via `np.where(np.isnan(H), -np.inf, H)`, take an ordinary `argmax(axis=1)`, and set the result to a sentinel such as -1 wherever `ok` is False (`np.where(ok, peak, -1)`). That way the missing day is visible in the output and is never confused with a genuine peak at hour 0, which is what a plain argmax over an all `-inf` row would report because ties go to the first position. Also note that a day where only some sensors are missing still gets a mean from the remaining sensors; if that is not acceptable, check the per-hour count `(~np.isnan(X)).sum(axis=1)` and require a minimum before trusting the value."
sources: [{"title": "numpy.sum", "url": "https://numpy.org/doc/stable/reference/generated/numpy.sum.html", "note": "The axis, keepdims, dtype and where parameters, plus the pairwise summation note."}, {"title": "NumPy reference: Statistics", "url": "https://numpy.org/doc/stable/reference/routines.statistics.html", "note": "All reductions, including the nan-aware versions and ddof."}, {"title": "numpy.take_along_axis", "url": "https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html", "note": "Pairing argmax or argsort results with values along an axis."}, {"title": "Python Data Science Handbook: Aggregations", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/02.04-computation-on-arrays-aggregates.html", "note": "Short tour of reductions with the axis convention explained."}]
practice: ["row-extremes", "aggregation-functions", "sample-variance-and-standard-deviation", "missing-value-imputation"]
tags: ["reductions", "axis", "missing values"]
tools: ["NumPy"]
---

## What you will learn

- What a **reduction** is (sum, mean, max, argmax and friends) and how the `axis` argument picks which direction gets squashed.
- A rule that ends almost every axis bug: the axis you name is the one that **disappears** from the shape.
- How `keepdims=True` lets a reduced result broadcast straight back onto the original array (centring, normalising, softmax).
- How to use `argmax`, `take_along_axis` and `argpartition` to get positions, values and top-k per row, and the tie rule that can quietly bias results.
- Running (cumulative) reductions, why long float32 running sums drift, and how NaN and empty rows change every answer.

*Before you start:* you should be comfortable with array shapes and dtypes (01.01), indexing (01.02), broadcasting (01.03) and ufuncs (01.04).

In lesson 01.04 every operation was **elementwise**: `np.exp(X)` or `X + Y` returns an array of the same shape, one output per input. Analysis also needs the opposite move: one number per column, per row, or for the whole array (the overall loss). Those are reductions, and nearly all take an `axis` argument.

## The idea in one picture

Three students each took four quizzes. Rows are students, columns are quizzes:

```python
import numpy as np
X = np.array([[7, 9, 6, 8],     # student 0
              [5, 6, 9, 4],     # student 1
              [8, 8, 7, 9]])    # student 2
X.shape          # (3, 4)
X.sum()          # 86              one number for everything
X.sum(axis=0)    # [20 23 22 21]   one number per quiz (column)
X.sum(axis=1)    # [30 24 32]      one number per student (row)
```

~~~dg-fig 01.05-axis.svg
Left: axis=0 runs down the rows, so the three rows are squashed into a single row of four column totals. Right: axis=1 runs across the columns, so the four columns are squashed into one column of three row totals.
~~~

Read the shapes, not the words. `X` has shape `(3, 4)`. `X.sum(axis=0)` has shape `(4,)`: the 3 is gone, the 4 survived. `X.sum(axis=1)` has shape `(3,)`: the 4 is gone. In plain words, **the axis you pass is the axis that vanishes**. Everything else in this lesson builds on that one sentence.

Two terms used throughout:

- An **axis** is one of the array's dimensions, numbered from 0. For a table of shape `(N, D)`, axis 0 runs down the N rows and axis 1 runs across the D columns. Negative numbers count from the end, so `axis=-1` is always the last axis.
- A **reduction** is a function that combines the values along one or more axes into a single value per remaining position.

## Step 1: what a reduction computes

Start with the whole array. Without `axis`, a reduction treats the array as one long list and returns a scalar (an array of shape `()`). The common ones:

| Function | Combines by | Returns
| --- | --- | --- |
| `sum`, `prod` | adding / multiplying | a value
| `mean`, `median`, `percentile` | averaging / ordering | a value
| `std`, `var` | spread around the mean | a value
| `min`, `max` | comparing | a value
| `argmin`, `argmax` | comparing | a **position**
| `any`, `all` | logical or / and | a boolean

Now add `axis`. Write the entry in row $i$ and column $j$ as $X_{ij}$. The column sums and row sums are

$$s_j = \sum_{i} X_{ij} \quad (\texttt{axis=0}), \qquad t_i = \sum_{j} X_{ij} \quad (\texttt{axis=1}).$$

Here $s_j$ is the total of column $j$, so the index $i$ (the row number) is the one being summed away, which is why it is axis 0. $t_i$ is the total of row $i$, summing away the column index $j$. Check on the quiz table: $s_0 = 7 + 5 + 8 = 20$ and $t_1 = 5 + 6 + 9 + 4 = 24$, matching the printed output.

"Sum over rows" can mean "add the rows together" (axis 0) or "sum each row" (axis 1). Skip the words: decide what shape you want back and name the axis that must vanish.

## Step 2: the shape rule in higher dimensions

The rule scales to any number of axes, and you can pass a tuple to collapse several at once. Take a batch of colour images `X` with shape `(N, H, W, C)`: N images, H by W pixels, C colour channels.

| Call | Result shape | Meaning
| --- | --- | --- |
| `X.mean()` | `()` | one global mean
| `X.mean(axis=0)` | `(H, W, C)` | the average image
| `X.mean(axis=(0, 1, 2))` | `(C,)` | per-channel mean, used for normalisation
| `X.mean(axis=(1, 2, 3))` | `(N,)` | mean brightness of each image
| `X.max(axis=-1)` | `(N, H, W)` | brightest channel at each pixel

To read each row, cross out the named positions in `(N, H, W, C)`; what is left is the result's shape.

~~~dg-flow axis names the dimension that is collapsed, not the one that is kept.
X (N, D) -(sum axis=0)-> *(D,) per column
X (N, D) -(sum axis=1)-> *(N,) per row
X (N, D) -(sum axis=1, keepdims)-> *(N, 1) broadcastable
~~~

## Step 3: keepdims, so the result can broadcast back

Centring, scaling and softmax all reduce and then combine the result with the original array. Suppose we want each student's scores relative to their own average. The row means have shape `(3,)`, and subtracting that from a `(3, 4)` array fails, because broadcasting lines shapes up from the **right**: the 3 meets the 4.

`keepdims=True` keeps the reduced axis in place with length 1. The row means come back as shape `(3, 1)`, a column vector, and broadcasting stretches that column across all four quizzes.

~~~dg-fig 01.05-keepdims.svg
Left: with keepdims the row means have shape (3, 1) and stretch across the columns, so each student is compared with their own average. Right: without it the shape is (3,), which lines up with the length-4 last axis and raises an error.
~~~

In symbols, row centring is $Z_{ij} = X_{ij} - \bar X_i$, where $\bar X_i = \frac{1}{D}\sum_j X_{ij}$ is the mean of row $i$ and $D$ is the number of columns. The `(3, 1)` shape is exactly "one $\bar X_i$ per row, reused for every $j$".

<div class="ex">
<div class="lab">Worked example</div>

**centring the quiz table by hand**

Row means: student 0 has $(7 + 9 + 6 + 8)/4 = 30/4 = 7.5$, student 1 has $24/4 = 6$, student 2 has $32/4 = 8$. With keepdims they form the column $[7.5, 6, 8]^\top$ of shape (3, 1).

Subtract row by row. Student 0: $(7 - 7.5, 9 - 7.5, 6 - 7.5, 8 - 7.5) = (-0.5, 1.5, -1.5, 0.5)$. Student 1: $(-1, 0, 3, -2)$. Student 2: $(0, 0, -1, 1)$.

Check: each centred row sums to zero, for example $-0.5 + 1.5 - 1.5 + 0.5 = 0$. That is always true after subtracting a row's own mean, and it is a cheap test that you used the right axis.

Now the trap. If the table had been square, say 4 students by 4 quizzes, `X - X.mean(axis=1)` would not raise anything. The four row means would line up with the four columns, and student $i$'s mean would be subtracted from quiz $i$ for every student. No error, wrong answer. keepdims makes the intent explicit.

</div>

The same pattern appears in the three most common normalisation lines in ML code:

```python
mu = X.mean(axis=1, keepdims=True)               # (N, 1)
Z  = (X - mu) / X.std(axis=1, keepdims=True)     # (N, D) standardised rows
E  = np.exp(S - S.max(axis=-1, keepdims=True))   # stable softmax numerator
P  = E / E.sum(axis=-1, keepdims=True)           # each row sums to 1
```

The softmax line subtracts each row's maximum before `exp`. That changes nothing mathematically (the factor $e^{-m}$ cancels in the ratio) but stops `exp(1000)` from overflowing to `inf`.

## Step 4: argmax, ties and getting values back

`argmax` is a reduction that returns **where** the maximum is, not what it is. For a score matrix of shape `(N, K)` (N examples, K classes), `S.argmax(axis=1)` collapses K and returns shape `(N,)`: the predicted class for each example.

~~~dg-fig 01.05-argmax.svg
argmax returns the position of the highest score. When several scores tie it always returns the first one, which is why an untrained model that outputs all zeros predicts class 0 for everything.
~~~

The tie rule matters. NumPy returns the **first** position of the maximum, so an untrained model with all-zero outputs predicts class 0 everywhere; if class 0 is common, its accuracy looks decent before training has started.

To turn per-row positions back into values, index each row at its own column:

```python
S = np.array([[2.1, 0.3, 3.4, 1.0],
              [1.2, 2.5, 2.5, 0.1],
              [0.0, 0.0, 0.0, 0.0]])
idx = S.argmax(axis=1)                                 # [2 1 0]
np.take_along_axis(S, idx[:, None], axis=1).ravel()    # [3.4 2.5 0. ]
S[np.arange(3), idx]                                   # same: [3.4 2.5 0. ]
```

`take_along_axis` needs an index with as many dimensions as `S`, hence `idx[:, None]` of shape `(3, 1)`: keepdims again.

For the top k entries per row, sorting the whole row with `np.argsort` costs $O(D \log D)$ per row. `np.argpartition(-S, k, axis=1)[:, :k]` only guarantees that the k largest come first, in **arbitrary order**, which is $O(D)$ work. On a 2000 by 1000 array, top-5 by full sort took about 24 ms and by `argpartition` about 11 ms on the machine used to write this lesson (timings are machine-dependent). If order matters, sort just those k afterwards with a second `argsort` on the k values.

## Step 5: running reductions along an axis

A **cumulative** operation keeps the axis instead of removing it: element $i$ of the output is the reduction of everything up to and including position $i$. `np.cumsum`, `np.cumprod` and `np.maximum.accumulate` are the common ones, and every binary ufunc has an `.accumulate`.

$$c_i = \sum_{j \le i} x_j \qquad m_i = \max_{j \le i} x_j$$

Here $c_i$ is the running total and $m_i$ the best value seen so far. Three patterns worth knowing:

- **Range sums in O(1).** Build `P = np.concatenate([[0], np.cumsum(x)])` once. The sum of `x[i:j]` is `P[j] - P[i]`. With `x = [3, 1, 4, 1, 5, 9, 2, 6]`, `P = [0, 3, 4, 8, 9, 14, 23, 25, 31]` and `x[2:6].sum()` is `P[6] - P[2] = 23 - 4 = 19`.
- **Running maximum and drawdown.** `x / np.maximum.accumulate(x) - 1` is how far each point sits below the best value so far, the standard risk measure for a price series or a validation curve.

~~~dg-fig 01.05-cumulative.svg
Top: a simulated price and its running maximum, which can only stay flat or rise. Bottom: drawdown, price divided by the running maximum minus one; it is zero at each new high and its lowest point is the worst loss from a peak.
~~~

Running sums have a precision trap. float32 carries about 7 significant digits, so once a running total is large, each small new term is partly rounded away. Adding 0.1 ten million times with `np.cumsum` in float32 ends at 1,087,937 instead of 1,000,000, almost 9% off. Plain `np.sum` on the same float32 array returns 1,000,000.1, because it uses **pairwise summation**: it adds in a tree of partial sums so no term is ever added to a huge running total.

~~~dg-fig 01.05-float32.svg
Measured relative error after adding 0.1 n times. The float32 running sum gets steadily worse; np.sum in float32 and cumsum with dtype=float64 stay near the tiny error of storing 0.1 itself.
~~~

`cumsum` cannot use the tree trick, because it must report every intermediate total in order. The fix is `np.cumsum(x, dtype=np.float64)`, which accumulates in double precision even when the input is float32.

## Step 6: missing values and empty rows

A single NaN makes `sum`, `mean`, `max` and `std` of its slice NaN, because any arithmetic with NaN gives NaN. When you do want to skip missing entries, use the `nan*` family: `np.nansum`, `np.nanmean`, `np.nanmax`, `np.nanstd`, `np.nanpercentile`, `np.nanargmax`.

~~~dg-fig 01.05-nan.svg
Row reductions on a table with gaps. One NaN makes the plain sum NaN; the nan-variants skip it. A row that is entirely NaN gives 0 from nansum but NaN (with a warning) from nanmean and nanmax.
~~~

Their edge cases differ, and you need to know them:

- `np.nanmean` of an all-NaN slice returns `nan` and emits `RuntimeWarning: Mean of empty slice`.
- `np.nansum` of an all-NaN slice returns `0.0`, which looks exactly like a real zero.
- `np.nanargmax` of an all-NaN slice raises `ValueError: All-NaN slice encountered`.

When missingness is stored as a separate boolean mask `m` (common in PyTorch and in JAX, where boolean indexing is not allowed under `jit`), compute masked statistics explicitly: `(x * m).sum(1) / np.maximum(m.sum(1), 1)`. The `np.maximum(..., 1)` guard stops division by zero for empty rows; decide deliberately whether an empty row should then read 0, NaN or a sentinel.

There is also a quiet convention difference. The standard deviation is

$$\sigma = \sqrt{\frac{1}{n - \text{ddof}} \sum_i (x_i - \bar x)^2},$$

where $n$ is the number of values and **ddof** ("delta degrees of freedom") adjusts the divisor. `np.std` defaults to `ddof=0`, dividing by $n$. pandas defaults to `ddof=1`, dividing by $n - 1$ (the sample estimate). For `[1, 2, 3, 4]`, NumPy gives 1.118 and pandas gives 1.291. Set `ddof` explicitly whenever the two libraries meet.

## Reductions in code: per-class accuracy

Here is the module's classic grouped reduction, with every intermediate printed.

<div class="ex">
<div class="lab">Worked example</div>

**per-class accuracy by hand**

Five examples, three classes. True labels `y = [0, 1, 2, 1, 0]`. The model's scores have shape (5, 3); `argmax(axis=1)` collapses the class axis and gives predictions `pred = [0, 1, 1, 1, 2]`, shape (5,).

Compare: `correct = (pred == y) = [T, T, F, T, F]`, which counts as `[1, 1, 0, 1, 0]`.

Count examples per class with `np.bincount(y)`: class 0 appears at positions 0 and 4, class 1 at 1 and 3, class 2 at 2, giving `[2, 2, 1]`.

Count correct per class with `np.bincount(y, weights=correct)`: class 0 gets $1 + 0 = 1$, class 1 gets $1 + 1 = 2$, class 2 gets $0$, giving `[1, 2, 0]`.

Divide: per-class accuracy is $[1/2, 2/2, 0/1] = [0.5, 1.0, 0.0]$.

Overall accuracy is `correct.mean()` $= 3/5 = 0.6$. The unweighted mean of the per-class accuracies is $(0.5 + 1 + 0)/3 = 0.5$. They differ because the classes have different sizes; they agree only when every class has the same count.

</div>

```python
y    = np.array([0, 1, 2, 1, 0])
pred = np.array([0, 1, 1, 1, 2])          # would be logits.argmax(axis=1)
correct = (pred == y)                     # [ True  True False  True False]
K = 3
hits  = np.bincount(y, weights=correct, minlength=K)   # [1. 2. 0.]
count = np.bincount(y, minlength=K)                    # [2 2 1]
print(hits / np.maximum(count, 1))                     # [0.5 1.  0. ]
print(correct.mean())                                  # 0.6
```

`minlength=K` matters: if the last class is absent from a batch, `bincount` would return a shorter array. The `np.maximum(count, 1)` guard reports an empty class as 0 instead of NaN; decide whether that is what you want.

## Checking the result

Reductions fail silently more often than they crash. A few cheap checks catch nearly all of it:

- **Assert the shape.** After every reduction you care about, `assert out.shape == (D,)` (or whatever you expect). If the shape is right, the axis almost certainly is too.
- **Use an identity.** `X.sum(axis=0).sum() == X.sum()`; centred rows sum to zero; softmax rows sum to 1 (`np.allclose(P.sum(-1), 1)`).
- **Count NaNs first.** `np.isnan(X).sum(axis=0)` tells you how many values each column is missing before you decide between plain and `nan*` reductions.
- **Look for ties.** `(S == S.max(axis=1, keepdims=True)).sum(axis=1) > 1` flags every row where `argmax` had to break a tie.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Result has the wrong length | Named the axis you wanted to keep | Name the axis that should disappear
| "could not be broadcast" after a reduction | Reduced shape `(N,)` meets `(N, D)` | `keepdims=True`
| No error, but numbers are subtly wrong on square data | `(N,)` silently aligned with the last axis when N equals D | `keepdims=True` and a shape assert
| Whole column becomes `nan` | One missing value in it | Use `nan*` variants or clean first
| `ValueError: All-NaN slice` | `nanargmax` on a fully missing row | Fill with `-np.inf`, take `argmax`, then mark bad rows
| Every prediction is class 0 | Ties at initialisation, first index wins | Check the score spread; break ties deliberately
| Running total drifts in long series | float32 `cumsum` | `dtype=np.float64`

## Going deeper

**Every reduction is a ufunc's `reduce`.** `np.sum(x, axis=k)` is `np.add.reduce(x, axis=k)`, `np.prod` is `np.multiply.reduce`, and `np.max` is `np.maximum.reduce`. That means any binary ufunc can reduce: `np.logical_xor.reduce`, `np.gcd.reduce`. The same objects provide `.accumulate` (the running version) and `.reduceat`, which reduces over segments. `np.add.reduceat(x, [0, 3, 5])` returns the sums of `x[0:3]`, `x[3:5]` and `x[5:]`, which for `x = [3, 1, 4, 1, 5, 9, 2, 6]` is `[8, 6, 17]`.

**Identities and empty arrays.** A reduction over zero elements returns the ufunc's identity: the sum of nothing is 0 and the product of nothing is 1. `maximum` has no identity, so `np.zeros(0).max()` raises "zero-size array to reduction operation maximum which has no identity". Pass `initial=-np.inf` to give it one. Reductions also take `where=`, a boolean mask of which elements to include: `v.sum(axis=1, where=m)` is a masked sum with no multiplication by zero, which also stops a masked-out NaN from leaking in.

**Why pairwise summation helps.** Adding $n$ numbers left to right, each addition can introduce a relative rounding error of about $\varepsilon$ (machine precision, about $6 \times 10^{-8}$ for float32), and these errors can add up to roughly $n\varepsilon$ in the worst case. Summing in a balanced tree means each value passes through only about $\log_2 n$ additions, so the worst-case error grows like $\varepsilon \log_2 n$. For $n = 10^7$ that is about 23 additions per value instead of ten million. NumPy only does this along the fast (contiguous) axis, in blocks; `math.fsum` gives an exactly rounded sum when you need it.

**Numerically stable reductions are built from reductions.** The stable softmax above is one example of a general trick, the log-sum-exp:

$$\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}, \qquad m = \max_j z_j.$$

The max is itself a reduction with keepdims, then an elementwise shift, then a second reduction. `scipy.special.logsumexp` and `torch.logsumexp` do exactly this.

**Other libraries.** PyTorch spells it `dim=` and `keepdim=`; JAX follows NumPy exactly; pandas skips NaN by default (`skipna=True`).

## Key terms

| Term | Meaning
| --- | --- |
| Reduction | A function that combines many values into fewer, such as sum, mean, max, argmax
| Axis | One dimension of an array, numbered from 0; negative numbers count from the end
| Collapsed axis | The axis named in `axis=`; it disappears from the result's shape
| keepdims | Keep the collapsed axis with length 1 so the result broadcasts back
| argmax | The position of the maximum along an axis; ties go to the first occurrence
| take_along_axis | Gather values using per-row index arrays such as argmax output
| Cumulative reduction | A running reduction (cumsum, accumulate) that keeps the axis
| Pairwise summation | Adding in a tree of partial sums to limit rounding error
| nan-aware reduction | A `nan*` function that skips NaN instead of propagating it
| ddof | Delta degrees of freedom: the variance divides by $n - \text{ddof}$

<div class="take">
<div class="lab">Key takeaway</div>

name the axis you want to collapse and check the result's shape, keep it with `keepdims=True` when you will broadcast back, and decide explicitly what ties, NaNs, empty groups and long float32 running sums should do.

</div>
