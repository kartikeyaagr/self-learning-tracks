---
id: "01.04"
course: "ml-toolkit"
module: "01"
slug: "vectorisation-and-ufuncs"
title: "Vectorisation and ufuncs"
keyIdeas: ["A ufunc is a compiled elementwise function (`np.add`, `np.exp`, `np.maximum`, ...) that broadcasts its inputs and also has `reduce`, `accumulate`, `outer` and `at` methods.", "Most loops vectorise into a few patterns: masks and `np.where` for branches, `cumsum` or `accumulate` for running state, `bincount` or `add.at` for group-by, `searchsorted` for lookups.", "Each expression like `a*b + c` allocates temporaries. `out=` and in-place operators reuse buffers, which matters once arrays reach hundreds of megabytes.", "Floating point overflows long before maths does: `exp(89)` is already `inf` in float32. Shift by the maximum ($\\log\\sum e^{x_i} = m + \\log\\sum e^{x_i - m}$) and use `log1p`, `expm1` and `logaddexp`."]
question: "A colleague computes binary cross-entropy as `-(y*np.log(sigmoid(z)) + (1-y)*np.log(1-sigmoid(z))).mean()` on float32 logits. For which values of z does this break, how, and how would you rewrite it stably?"
answer: "For large positive z (above about 17 in float32) sigmoid(z) rounds to exactly 1.0, so `np.log(1 - sigmoid(z))` is `log(0) = -inf`. If the label is 0, the loss is infinite, and its gradient is `nan`. For large negative z, sigmoid(z) underflows towards 0 (the naive `1/(1+np.exp(-z))` returns exactly 0 below about -89 in float32, when the exponential overflows), giving the same problem on the other term when y = 1. The stable form works in logit space: $\\log\\sigma(z) = -\\log(1+e^{-z})$ and $\\log(1-\\sigma(z)) = -\\log(1+e^{z})$, so the loss is `np.logaddexp(0, z) - y*z`, equivalently `np.maximum(z, 0) - y*z + np.log1p(np.exp(-np.abs(z)))`. No exponent is ever large and positive, nothing rounds to 0 inside a log, and the result is accurate for any z."
sources: [{"title": "NumPy user guide: Universal functions basics", "url": "https://numpy.org/doc/stable/user/basics.ufuncs.html", "note": "Ufunc methods, out= and where=, casting rules for in-place operations."}, {"title": "Higham: What is the log-sum-exp function?", "url": "https://nhigham.com/2021/01/05/what-is-the-log-sum-exp-function/", "note": "A numerical analyst's short derivation of the shift trick and its error bounds."}, {"title": "scipy.special.logsumexp", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html", "note": "A reference implementation with weights, signs and axis handling."}, {"title": "Python Data Science Handbook: Computation on arrays, ufuncs", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/02.03-computation-on-arrays-ufuncs.html", "note": "Timing of loops vs ufuncs and the ufunc method zoo."}]
practice: ["softmax-numerical-stability-and-log-sum-exp", "softmax-rows", "normalized-difference", "angle-features", "vector-norms"]
tags: ["ufuncs", "vectorisation", "numerical stability"]
tools: ["NumPy"]
---

## What a ufunc is

A **universal function** applies a scalar operation to every element of its broadcast inputs, using a compiled inner loop chosen for the input dtypes. `np.add`, `np.multiply`, `np.exp`, `np.maximum`, `np.isnan` and the comparison operators are all ufuncs; `a + b` is literally `np.add(a, b)`.

Binary ufuncs also come with methods that turn the same inner loop into other patterns:

| method | example | meaning
| --- | --- | --- |
| reduce | `np.add.reduce(x, axis=0)` | fold along an axis (this is `np.sum`)
| accumulate | `np.maximum.accumulate(x)` | running result (running max)
| outer | `np.subtract.outer(a, b)` | all pairs, shape `a.shape + b.shape`
| at | `np.add.at(g, idx, v)` | unbuffered in-place update, duplicates accumulate
| reduceat | `np.add.reduceat(x, starts)` | reduce contiguous segments

Functions such as `np.sort`, `np.linalg.solve` or `np.median` are not ufuncs, but they are vectorised in the same sense: one Python call, one compiled loop.

## Replacing loops

Vectorising is mostly pattern matching. The common translations:

- **Branch per element** (`if x > 0: ... else: ...`) becomes a boolean mask and `np.where`, or `np.clip`, `np.maximum`.
- **Running state** (`total += x[i]`) becomes `np.cumsum`, `np.cumprod` or `ufunc.accumulate`.
- **Group by key** (`sums[key[i]] += v[i]`) becomes `np.bincount(key, weights=v)` or `np.add.at`.
- **Lookup in a sorted table** (find the bin for each value) becomes `np.searchsorted`.
- **Pairwise loops** become broadcasting or a matrix product (previous lesson).

~~~dg-flow A nested Python loop turns into one broadcast expression plus one reduction.
for i, for j loop -(add axes)-> x[:, None] op y[None, :] -(reduce axis)-> *one C call
~~~

Some loops do not vectorise. A recurrence where step $t$ depends nonlinearly on step $t-1$ (an RNN, a simulation with branching state) has to run sequentially. Linear recurrences can sometimes be rewritten in closed form: an exponential moving average is a weighted cumulative sum. For the truly sequential ones, compile the loop (Numba, `jax.lax.scan`) rather than fighting NumPy.

## Temporaries, out= and in-place operations

`y = a * b + c` on arrays of $10^8$ float64 values allocates `a * b` (800 MB), then the sum (another 800 MB), then frees the first. Memory bandwidth, not arithmetic, dominates this kind of code, so every temporary costs time as well as memory.

```python
y = np.empty_like(a)
np.multiply(a, b, out=y)   # one output buffer
y += c                     # in place, no new allocation
np.exp(y, out=y)
```

Two caveats. In-place operations obey casting rules: `ints += 0.5` raises `UFuncTypeError` because a float result cannot be stored in an int array under the default `same_kind` casting. And `where=` without `out=` leaves the unselected entries uninitialised. Do this only on hot paths with large arrays; for small arrays readability wins.

## Numerical stability tricks

Floating point has a finite range: float64 overflows above about $1.8 \times 10^{308}$, so `np.exp(710.0)` is `inf`; float32 overflows at about $3.4 \times 10^{38}$, so `np.exp(np.float32(89))` is `inf`. Logits from a model easily reach these values. The fix for sums of exponentials is to factor out the maximum $m = \max_i x_i$:

$$\log \sum_i e^{x_i} = m + \log \sum_i e^{x_i - m}$$

Every shifted exponent is at most $e^0 = 1$, so nothing overflows, and at least one term equals 1, so the sum cannot underflow to 0 and produce `log(0)`. Softmax uses the same shift.

```python
def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)        # rows that are all -inf
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)
```

Other tools worth knowing:

- `np.log1p(x)` and `np.expm1(x)` stay accurate for tiny `x`, where `1 + x` rounds to 1: in float64, `np.log(1 + 1e-17)` is 0 while `np.log1p(1e-17)` is `1e-17`.
- `np.logaddexp(a, b)` computes $\log(e^a + e^b)$ stably, useful for combining log-probabilities.
- A stable sigmoid computes `exp(-x)` only for $x \ge 0$ and `exp(x) / (1 + exp(x))` for $x < 0$, so no exponent is ever large and positive.
- Variance as $E[x^2] - E[x]^2$ cancels catastrophically when the mean is large relative to the spread; `np.var` subtracts the mean first.

<div class="ex">
<div class="lab">Worked example</div>

**logsumexp of [1000, 1001, 1002]**

Naively, `np.exp(1000.0)` is `inf` in float64, so the result is `inf`.

Shifted, $m = 1002$ and the terms are $e^{-2} + e^{-1} + e^{0} = 0.1353 + 0.3679 + 1 = 1.5032$.

The answer is $1002 + \ln 1.5032 = 1002.4076$, computed with no overflow. The softmax probabilities are the same three terms divided by 1.5032: 0.090, 0.245, 0.665.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

write the loop body as whole-array ufunc calls, reuse buffers with `out=` only where size demands it, and never exponentiate a raw logit: subtract the maximum first.

</div>
