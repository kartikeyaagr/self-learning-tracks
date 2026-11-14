---
id: "01.05"
course: "ml-toolkit"
module: "01"
slug: "reductions-and-the-axis-argument"
title: "Reductions and the axis argument"
keyIdeas: ["`axis=k` names the axis that **disappears**: for `X` of shape `(N, D)`, `X.sum(axis=0)` has shape `(D,)` (one value per column) and `X.sum(axis=1)` has shape `(N,)` (one per row).", "`keepdims=True` keeps the reduced axis as size 1 so the result broadcasts straight back against the input, as in `X - X.mean(axis=1, keepdims=True)`.", "`argmax` returns positions, not values, and breaks ties by taking the first occurrence; `np.take_along_axis` turns per-row positions back into values.", "NaN propagates through `sum` and `mean`. The `nan*` variants skip it, but an all-NaN row gives `nan` with a warning from `nanmean`, and `0` from `nansum`."]
question: "Sensor data `X` has shape (days, sensors, 24) with hourly readings and some NaNs. Write the calls for (a) each sensor's mean over all days and hours, and (b) for each day, the hour at which the average over sensors peaked. What happens in (b) if one day is entirely NaN, and how would you handle it?"
answer: "(a) Collapse days and hours, keep sensors: `np.nanmean(X, axis=(0, 2))`, shape (sensors,). (b) First average over sensors for each day and hour: `H = np.nanmean(X, axis=1)`, shape (days, 24). Then collapse the hour axis to a position: `np.nanargmax(H, axis=1)`, shape (days,). If one day is entirely NaN, `nanmean` gives a NaN row with a RuntimeWarning, and `nanargmax` raises ValueError because that row has no valid entry. Handle it explicitly: compute `ok = ~np.isnan(H).all(axis=1)`, fill the bad rows with `-np.inf` (`np.where(np.isnan(H), -np.inf, H)`), take `argmax`, and then set the result for days where `ok` is False to a sentinel such as -1. That way the missing day is visible and is never confused with a real peak at hour 0."
sources: [{"title": "numpy.sum", "url": "https://numpy.org/doc/stable/reference/generated/numpy.sum.html", "note": "The axis, keepdims, dtype and where parameters, plus the pairwise summation note."}, {"title": "NumPy reference: Statistics", "url": "https://numpy.org/doc/stable/reference/routines.statistics.html", "note": "All reductions, including the nan-aware versions and ddof."}, {"title": "numpy.take_along_axis", "url": "https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html", "note": "Pairing argmax or argsort results with values along an axis."}, {"title": "Python Data Science Handbook: Aggregations", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/02.04-computation-on-arrays-aggregates.html", "note": "Short tour of reductions with the axis convention explained."}]
practice: ["row-extremes", "aggregation-functions", "sample-variance-and-standard-deviation", "missing-value-imputation"]
tags: ["reductions", "axis", "missing values"]
tools: ["NumPy"]
---

## The axis is the one that goes away

A reduction combines many numbers into one: `sum`, `mean`, `std`, `min`, `max`, `argmax`, `any`, `all`, `prod`, `median`, `percentile`. Without an `axis` it reduces the whole array to a scalar. With `axis=k` it reduces along axis `k` only, and **that axis vanishes from the result**.

For a batch of images `X` of shape `(N, H, W, C)`:

| call | result shape | meaning
| --- | --- | --- |
| `X.mean()` | () | one global mean
| `X.mean(axis=0)` | (H, W, C) | the average image
| `X.mean(axis=(0, 1, 2))` | (C,) | per-channel mean, used for normalisation
| `X.mean(axis=(1, 2, 3))` | (N,) | mean brightness of each image
| `X.max(axis=-1)` | (N, H, W) | brightest channel at each pixel

The phrase "sum over rows" is ambiguous in English and causes most axis bugs. Translate it into shapes: if you want one number per column of an `(N, D)` table you are collapsing the N axis, so `axis=0`.

~~~dg-flow axis names the dimension that is collapsed, not the one that is kept.
X (N, D) -(sum axis=0)-> *(D,) per column
X (N, D) -(sum axis=1)-> *(N,) per row
X (N, D) -(sum axis=1, keepdims)-> *(N, 1) broadcastable
~~~

## keepdims and broadcasting back

Many computations reduce and then combine the result with the original array: centring, normalising, softmax. `keepdims=True` leaves the reduced axis in place with size 1, which is exactly the shape broadcasting needs:

```python
mu = X.mean(axis=1, keepdims=True)              # (N, 1)
Z = (X - mu) / X.std(axis=1, keepdims=True)     # (N, D)
P = E / E.sum(axis=-1, keepdims=True)           # rows sum to 1
```

Without it, an `(N,)` result aligns with the last axis of `(N, D)`, which either errors or (when N equals D) silently uses the wrong statistics, as in the broadcasting lesson.

## argmax, top-k and getting values back

`X.argmax(axis=1)` gives, for each row, the column index of the maximum. Ties go to the **first** occurrence, which can bias metrics when a model outputs many identical scores (for example all zeros at initialisation, where every prediction becomes class 0).

To fetch values at per-row positions, use `np.take_along_axis(X, idx[:, None], axis=1)`, or the explicit form `X[np.arange(N), idx]`. For top-k, sorting the whole row is $O(D \log D)$; `np.argpartition(-X, k, axis=1)[:, :k]` finds the k largest in $O(D)$ but in arbitrary order, so sort just those k afterwards if order matters.

## Cumulative operations

`np.cumsum`, `np.cumprod` and `np.maximum.accumulate` keep the axis and return the running result along it. Useful patterns:

- **Range sums in O(1)**: with `S = np.concatenate([[0], np.cumsum(x)])`, the sum of `x[i:j]` is `S[j] - S[i]`.
- **Running maximum and drawdown**: `x / np.maximum.accumulate(x) - 1`.
- **Inverse pair**: `np.diff` undoes `np.cumsum` (up to the first element).

Long cumulative sums in float32 drift: after millions of additions each new term is small relative to the total and loses digits. Pass `dtype=np.float64` to accumulate in higher precision. Plain `np.sum` is more forgiving because it uses pairwise summation internally, but `cumsum` cannot.

## Missing values: NaN-aware reductions

A single `nan` turns `sum`, `mean`, `max` and `std` of its slice into `nan`. The `nan*` family (`np.nanmean`, `np.nanmax`, `np.nanstd`, `np.nanpercentile`, `np.nanargmax`) ignores NaNs instead. Their edge cases differ, and they do matter:

- `np.nanmean` of an all-NaN slice returns `nan` and emits a `RuntimeWarning`.
- `np.nansum` of an all-NaN slice returns `0.0`, which can look like a genuine zero.
- `np.nanargmax` of an all-NaN slice raises `ValueError`.

When missingness is a separate mask (or you work in JAX, where boolean indexing is not allowed under `jit`), compute masked statistics explicitly: `(x * m).sum(1) / np.maximum(m.sum(1), 1)`, with the guard deciding what an empty row means.

Also note the variance convention: `np.std` and `np.var` default to `ddof=0`, the population formula dividing by $n$. pandas defaults to `ddof=1`, dividing by $n - 1$. The same data gives slightly different numbers in the two libraries unless you set `ddof` explicitly.

<div class="ex">
<div class="lab">Worked example</div>

**per-class accuracy from predictions**

`logits` has shape (N, K) and `y` has shape (N,). `pred = logits.argmax(axis=1)` collapses K, giving (N,).

`correct = (pred == y)` is a boolean (N,). Per-class accuracy is `np.bincount(y, weights=correct, minlength=K) / np.maximum(np.bincount(y, minlength=K), 1)`, a grouped mean without a loop.

Overall accuracy is `correct.mean()`. It equals the unweighted mean of the per-class accuracies only when every class has the same number of examples.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

name the axis you want to collapse, keep it with `keepdims=True` when you will broadcast back, and decide explicitly what NaNs and empty groups should produce.

</div>
