---
id: "01.04"
course: "ml-toolkit"
module: "01"
slug: "vectorisation-and-ufuncs"
title: "Vectorisation and ufuncs"
keyIdeas: ["**Vectorisation** means replacing a per-element Python loop with whole-array calls; the looping then happens in compiled code, which is typically tens of times faster for large arrays (but not for a handful of elements, and never with `np.vectorize`).", "A **ufunc** is a compiled elementwise function (`np.add`, `np.exp`, `np.maximum`, ...) that broadcasts its inputs and also has `reduce`, `accumulate`, `outer`, `at` and `reduceat` methods; `np.sum` is `np.add.reduce`.", "Most loops vectorise into a few patterns: masks and `np.where` for branches, `cumsum` or `accumulate` for running state, `bincount` or `add.at` for group-by (never `g[idx] += v` with repeated indices), `searchsorted` for lookups, broadcasting for pairs.", "Each expression like `a*b + c` can allocate temporaries. `out=` and in-place operators reuse buffers, which matters once arrays reach many megabytes; mind the casting rules and always pair `where=` with `out=`.", "Floating point overflows long before maths does: `exp(89)` is already `inf` in float32. Shift by the maximum ($\\log\\sum e^{x_i} = m + \\log\\sum e^{x_i - m}$) and use `log1p`, `expm1` and `logaddexp`."]
question: "A colleague computes binary cross-entropy as `-(y*np.log(sigmoid(z)) + (1-y)*np.log(1-sigmoid(z))).mean()` on float32 logits, with `sigmoid(z) = 1/(1+np.exp(-z))`. For which values of z does this break, how, and how would you rewrite it stably and without any Python loop?"
answer: "For large positive z (above about 17 in float32) `sigmoid(z)` rounds to exactly 1.0, because float32 cannot represent a number between 1 - 6e-8 and 1 closer than that, so `np.log(1 - sigmoid(z))` is `log(0) = -inf`. If the label is 0, the loss is infinite and its gradient becomes `nan`, which then poisons every weight it touches. For large negative z, `np.exp(-z)` overflows to `inf` below about -89 in float32 (with a RuntimeWarning), so sigmoid returns exactly 0 and `np.log(sigmoid(z))` is `-inf`, the same problem on the other term when y = 1. Even before these hard limits, precision is lost because the code subtracts nearly equal numbers. The stable form works in logit space: $\\log\\sigma(z) = -\\log(1+e^{-z})$ and $\\log(1-\\sigma(z)) = -\\log(1+e^{z})$, so the per-example loss is $\\log(1+e^{z}) - yz$, which is `np.logaddexp(0, z) - y*z`, or equivalently `np.maximum(z, 0) - y*z + np.log1p(np.exp(-np.abs(z)))`. In the second form the only exponent is $-|z| \\le 0$, so nothing overflows, `log1p` keeps precision when the exponential is tiny, and nothing inside a log can round to 0. Both are pure ufunc expressions over the whole batch, so the `.mean()` stays vectorised. Check: for z = 20 and y = 0 the stable loss is 20.0 (the naive one is inf); for z = 0 it is ln 2 = 0.693 for either label. In practice use the library's fused version (`torch.nn.functional.binary_cross_entropy_with_logits`, `optax.sigmoid_binary_cross_entropy`)."
sources: [{"title": "NumPy user guide: Universal functions basics", "url": "https://numpy.org/doc/stable/user/basics.ufuncs.html", "note": "Ufunc methods, out= and where=, casting rules for in-place operations."}, {"title": "Higham: What is the log-sum-exp function?", "url": "https://nhigham.com/2021/01/05/what-is-the-log-sum-exp-function/", "note": "A numerical analyst's short derivation of the shift trick and its error bounds."}, {"title": "scipy.special.logsumexp", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html", "note": "A reference implementation with weights, signs and axis handling."}, {"title": "Python Data Science Handbook: Computation on arrays, ufuncs", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/02.03-computation-on-arrays-ufuncs.html", "note": "Timing of loops vs ufuncs and the ufunc method zoo."}]
practice: ["softmax-numerical-stability-and-log-sum-exp", "softmax-rows", "normalized-difference", "angle-features", "vector-norms"]
tags: ["ufuncs", "vectorisation", "numerical stability"]
tools: ["NumPy"]
---

## What you will learn

- What "vectorising" a loop means, and why one NumPy call can be tens of times faster than the same Python loop.
- What a **ufunc** is, and how its methods (`reduce`, `accumulate`, `outer`, `at`, `reduceat`) reuse one compiled loop for several jobs.
- The five loop patterns you will meet again and again, and the NumPy call that replaces each one.
- How temporary arrays cost memory and time, and when `out=` and in-place operators are worth the extra lines.
- Why `exp`, `log` and `sigmoid` break on real model outputs, and the standard stable rewrites (`logsumexp`, `log1p`, `logaddexp`).

*Before you start:* you should be comfortable with dtypes and memory layout (01.01), boolean masks and fancy indexing (01.02), and broadcasting (01.03). Nothing here needs maths beyond $e^x$ and $\log x$.

## From broadcasting to whole-array thinking

Lesson 01.03 showed how NumPy lines up arrays of different shapes; this lesson is about the operation that runs once they line up. The habit to build: instead of "for each element, do this", say "do this to the whole array", and let NumPy run the "for each" part in compiled C. That habit is **vectorisation**, the biggest speed lever in NumPy, pandas, PyTorch and JAX alike.

## The idea in one picture

A shop applies a pricing rule to every order: orders above 100 get 10% off, everything else pays a shipping fee of 5. Here is the loop most people write first, and the vectorised version:

```python
import numpy as np
prices = np.array([80., 150., 40., 120.])

out = np.empty_like(prices)                 # loop version
for i in range(len(prices)):
    out[i] = prices[i] * 0.9 if prices[i] > 100 else prices[i] + 5.0

out2 = np.where(prices > 100, prices * 0.9, prices + 5.0)   # vectorised
print(out2)                                  # [ 85. 135.  45. 108.]
```

Both give the same four numbers. The difference is who runs the loop. In the first version Python, for every element, fetches `prices[i]`, wraps it in a Python float object, checks types, multiplies and stores. In the second, Python makes four calls (`>`, `*`, `+`, `np.where`), and each runs a tight C loop over raw memory.

~~~dg-fig 01.04-speed.svg
Measured time for the pricing rule written four ways (log scale; timings are machine-dependent). The three Python-level versions rise together; the ufunc version is roughly 20 times faster at a million prices and about 45 times faster at ten thousand.
~~~

With 10 prices the loop even wins, because each NumPy call has a fixed start-up cost of a few microseconds. From about a thousand elements on, the gap opens and stays open. Notice that `np.vectorize` is *not* fast: it still calls your Python function once per element. Despite its name, it gives you broadcasting, not speed.

An **elementwise operation** applies the same scalar rule to each position independently (`a + b`, `np.exp(a)`, `a > 0`). A **ufunc** (universal function) is NumPy's object for such an operation, backed by one compiled loop per supported dtype.

## Step 1: what a ufunc actually is

`np.add`, `np.multiply`, `np.exp`, `np.sqrt`, `np.maximum`, `np.isnan` and every comparison operator are ufuncs. The arithmetic operators are just shorthand: `a + b` calls `np.add(a, b)` and `a > b` calls `np.greater(a, b)`. You can check: `isinstance(np.exp, np.ufunc)` is `True`, while `isinstance(np.sum, np.ufunc)` is `False`.

When you call a ufunc, NumPy does three things:

1. **Broadcast** the inputs to a common shape (the rules from 01.03).
2. **Pick an inner loop** for the input dtypes. `np.add.types` lists them: `'ff->f'` means "two float32 in, one float32 out". If no loop matches exactly, inputs are cast to a type that does.
3. **Run the loop** over the whole output, in C, usually using SIMD instructions that process several numbers per CPU instruction.

Functions such as `np.sort`, `np.median` or `np.linalg.solve` are not ufuncs, because they are not elementwise, but they are vectorised in the same useful sense: one Python call, one compiled loop.

## Step 2: one inner loop, five jobs

Every binary ufunc (one with two inputs) carries methods that run the same compiled operation in a different pattern. This is why NumPy feels consistent: `np.sum` is really `np.add.reduce`, and `np.cumsum` is `np.add.accumulate`.

~~~dg-fig 01.04-methods.svg
Four ufunc methods applied to the same small array. reduce folds everything into one number, accumulate keeps every intermediate result, and outer combines every pair.
~~~

| method | example | what it does
| --- | --- | --- |
| reduce | `np.add.reduce(x, axis=0)` | fold along an axis (this is `np.sum`)
| accumulate | `np.maximum.accumulate(x)` | keep each running result (running max)
| outer | `np.subtract.outer(a, b)` | all pairs, shape `a.shape + b.shape`
| at | `np.add.at(g, idx, v)` | unbuffered in-place update; repeated indices accumulate
| reduceat | `np.add.reduceat(x, starts)` | reduce contiguous segments that begin at `starts`

Small number check with `a = [3, 1, 4, 1, 5]`:

- `np.add.reduce(a)` is $3+1+4+1+5 = 14$; `np.multiply.reduce(a)` is $3 \cdot 1 \cdot 4 \cdot 1 \cdot 5 = 60$; `np.maximum.reduce(a)` is 5.
- `np.add.accumulate(a)` is `[3, 4, 8, 9, 14]`: each entry is the sum so far.
- `np.maximum.accumulate(a)` is `[3, 3, 4, 4, 5]`: the highest value seen so far. Handy for "best validation score so far" or drawdown curves.
- `np.add.reduceat([3, 1, 4, 1, 5, 9, 2], [0, 2, 5])` sums the segments `[3, 1]`, `[4, 1, 5]` and `[9, 2]`, giving `[4, 10, 11]`.

`np.add.outer(a, b)` equals the broadcast `a[:, None] + b[None, :]` from 01.03.

## Step 3: translate the loop, pattern by pattern

Vectorising is mostly pattern matching. Look at the body of your loop and ask which of these shapes it has.

~~~dg-flow A nested Python loop turns into one broadcast expression plus one reduction.
for i, for j loop -(add axes)-> x[:, None] op y[None, :] -(reduce axis)-> *one C call
~~~

**Pattern 1: a branch per element.** `if x > 0: ... else: ...` becomes a boolean mask plus `np.where(cond, a, b)`. Note that `np.where` computes *both* branches for every element and then picks, so both must be safe everywhere (see "What goes wrong"). Simple branches have dedicated ufuncs: `np.maximum(x, 0)` is ReLU, `np.clip(x, -1, 1)` caps values, `np.abs`, `np.sign`.

**Pattern 2: running state.** `total += x[i]` inside a loop becomes `np.cumsum`, `np.cumprod` or `ufunc.accumulate`. Differences of a cumulative sum give window sums, so a moving average needs no loop:

```python
x = np.array([2., 5., 1., 4.]); w = 3
c = np.cumsum(np.r_[0, x])        # [0, 2, 7, 8, 12]
print((c[w:] - c[:-w]) / w)       # [2.667 3.333]  = mean(2,5,1), mean(5,1,4)
```

**Pattern 3: group by key.** `sums[key[i]] += v[i]` becomes `np.bincount(key, weights=v)` when keys are small non-negative integers, or `np.add.at(sums, key, v)` for any ufunc and shape. There is a trap here that catches almost everyone, shown in the next figure.

~~~dg-fig 01.04-groupby.svg
Top: bincount sums the values of each key in one call. Bottom: with a repeated index, g[idx] += 1 adds only once per distinct index, because each slot is written, not accumulated; np.add.at counts every occurrence.
~~~

`g[idx] += 1` is evaluated as `g[idx] = g[idx] + 1`: NumPy reads the old values, adds 1, and writes them back. Slot 2 is written twice with the same value 1, so the second write does not see the first. `np.add.at` is **unbuffered**: it applies each update one after another, so duplicates accumulate. Prefer the faster `bincount` when it fits.

<div class="ex">
<div class="lab">Worked example</div>

**group sums by hand with bincount**

Keys `[0, 2, 1, 0, 2, 2]` and values `[5, 3, 4, 1, 2, 6]`. We want one total per key.

Key 0 appears at positions 0 and 3, with values 5 and 1: total $5 + 1 = 6$.

Key 1 appears only at position 2, with value 4: total 4.

Key 2 appears at positions 1, 4 and 5, with values 3, 2 and 6: total $3 + 2 + 6 = 11$.

So `np.bincount(key, weights=value)` returns `[6., 4., 11.]`, and `np.bincount(key)` without weights returns the counts `[2, 1, 3]`. Dividing gives group means: `[3., 4., 3.667]`.

</div>

**Pattern 4: lookup in a sorted table.** "Which bin does each value fall into?" becomes `np.searchsorted`. With age-band edges `[0, 18, 35, 65]` and ages `[4, 22, 35, 70, 64]`, `np.searchsorted(edges, ages, side='right') - 1` gives bands `[0, 1, 2, 3, 2]`. It runs a binary search per value, in C.

**Pattern 5: pairs of items.** A double loop over $i$ and $j$ becomes broadcasting (`x[:, None] - y[None, :]`), `ufunc.outer`, or a matrix product, as in 01.03.

**When a loop will not vectorise.** A recurrence where step $t$ depends *nonlinearly* on step $t-1$ (an RNN cell, a simulation with branching state) must run in order. Linear recurrences can often be rewritten: an exponential moving average $s_t = \alpha x_t + (1-\alpha) s_{t-1}$ unrolls into a weighted sum of past values. For the truly sequential cases, compile the loop with Numba, or use `jax.lax.scan`, rather than fighting NumPy.

## Step 4: temporaries, out= and in-place operations

Each operator in an expression produces a new array. On $10^7$ float64 values (80 MB each), `np.exp(a * b + c) / 2` creates intermediate results that must be allocated, filled, read back and freed. For such large arrays the CPU spends most of its time waiting for memory, not doing arithmetic, so every extra pass over memory costs time as well as space.

NumPy lets you choose where results go. Every ufunc accepts `out=`, and the augmented operators (`+=`, `*=`, `/=`) write into the left-hand array:

```python
y = np.empty_like(a)       # allocate the result once
np.multiply(a, b, out=y)   # y = a * b
y += c                     # in place, no new allocation
np.exp(y, out=y)
y /= 2
```

~~~dg-fig 01.04-memory.svg
The same computation on ten million float64 values, measured here with tracemalloc and timeit (machine-dependent). Writing into one buffer halves the peak allocation; reusing a buffer you already own allocates nothing.
~~~

In the run behind the figure, the plain expression peaked at 160 MB of new memory and took about 51 ms; the `out=` version allocated only its 80 MB result and took about 36 ms; reusing a preallocated buffer allocated nothing and took about 32 ms. The plain version does better than a naive count suggests because NumPy quietly reuses some temporaries for you (see "Going deeper").

Three caveats:

- In-place operations obey casting rules. `ints = np.arange(3); ints += 0.5` raises `UFuncTypeError` (cannot cast float64 output to int64 under `same_kind`), because the float result has nowhere to go in an int array.
- `where=` without `out=` leaves the unselected positions uninitialised: they contain whatever bytes were in memory. Always pair them, as in `np.sqrt(x, out=np.full_like(x, np.nan), where=x >= 0)`.
- Writing in place destroys the input, the view-versus-copy issue from 01.02 again.

Use these tools only on hot paths with arrays of many megabytes; otherwise the clear one-liner is better code.

## Step 5: numbers that are too big or too small

Floating point has a finite range. float64 overflows above about $1.8 \times 10^{308}$, so `np.exp(710.0)` is `inf`. float32, the default in deep learning, overflows above about $3.4 \times 10^{38}$, so `np.exp(np.float32(89))` is already `inf` while `np.exp(np.float32(88))` is about $1.65 \times 10^{38}$. Model logits reach these values easily.

The cure for a sum of exponentials is to factor out the largest exponent. Let $m = \max_i x_i$. Since $e^{x_i} = e^{m} e^{x_i - m}$,

$$\log \sum_i e^{x_i} = m + \log \sum_i e^{x_i - m}.$$

Here $x_i$ are the inputs (for example logits) and $m$ is their maximum. Every shifted exponent $x_i - m$ is at most 0, so each term $e^{x_i - m}$ is at most 1 and nothing overflows. At least one term equals exactly $e^0 = 1$, so the sum is at least 1 and the log never sees 0. Softmax uses the same shift, because the factor $e^m$ cancels between numerator and denominator.

<div class="ex">
<div class="lab">Worked example</div>

**logsumexp and softmax of [1000, 1001, 1002]**

Naively, `np.exp(1000.0)` is `inf` in float64, so the sum is `inf` and so is its log.

Shift by the maximum $m = 1002$: the shifted inputs are $(-2, -1, 0)$.

Exponentiate: $e^{-2} = 0.1353$, $e^{-1} = 0.3679$, $e^{0} = 1$. Their sum is $0.1353 + 0.3679 + 1 = 1.5032$.

Log and add back: $1002 + \ln 1.5032 = 1002 + 0.4076 = 1002.4076$.

Softmax is each term divided by the sum: $0.1353/1.5032 = 0.090$, $0.3679/1.5032 = 0.245$, $1/1.5032 = 0.665$. The three probabilities add to 1, and they are exactly the softmax of $(1, 2, 3)$, because softmax ignores a constant shift.

</div>

The same idea gives stable versions of other common expressions:

- `np.log1p(x)` computes $\log(1+x)$ and `np.expm1(x)` computes $e^x - 1$ accurately for tiny $x$. In float64, `1 + 1e-17` rounds to exactly 1, so `np.log(1 + 1e-17)` is 0, while `np.log1p(1e-17)` is `1e-17`.
- `np.logaddexp(a, b)` computes $\log(e^a + e^b)$ stably: `np.logaddexp(1000, 1000)` is $1000 + \ln 2 = 1000.693$.
- A stable sigmoid uses $1/(1+e^{-x})$ for $x \ge 0$ and $e^{x}/(1+e^{x})$ for $x < 0$, so the exponent is never large and positive.
- Variance computed as $E[x^2] - E[x]^2$ cancels catastrophically when the mean is large compared with the spread: for `[1e9+1, 1e9+2, 1e9+3]` in float64 it returns 0.0, while `np.var` (which subtracts the mean first) returns the correct 0.667.

~~~dg-fig 01.04-stability.svg
Left: in float32 the naive binary cross-entropy for a label of 0 matches the stable formula until sigmoid rounds to exactly 1.0 near z = 16.7, after which it is inf. Right: the relative error of log(1 + x) climbs to 100% as x shrinks, while log1p stays at rounding level.
~~~

## Vectorised code, all together

A small, realistic block that uses every idea above: per-row softmax of logits, a stable loss, and per-class accuracy with `bincount`.

```python
import numpy as np
logits = np.array([[1., 2., 3.], [1000., 1001., 1002.]])
m = logits.max(axis=1, keepdims=True)          # (2, 1), ready to broadcast
e = np.exp(logits - m)
probs = e / e.sum(axis=1, keepdims=True)
print(probs.round(3))      # [[0.09  0.245 0.665]
                           #  [0.09  0.245 0.665]]

def bce_with_logits(z, y):                      # binary cross-entropy, stable
    return np.logaddexp(0, z) - y * z
print(bce_with_logits(np.array([-30., 0., 20.]), np.array([1., 0., 0.])))
                           # [30.  0.693  20.]

pred = np.array([0, 2, 1, 0, 2, 1]); true = np.array([0, 2, 1, 1, 2, 2])
hits = np.bincount(true, weights=(pred == true), minlength=3)
counts = np.bincount(true, minlength=3)
print(hits / counts)       # [1.  0.5  0.667]  accuracy per class
```

There is no `for` anywhere, yet the code reads like the maths: shorter code, fewer index bugs, and speed as a bonus.

## Checking that a vectorised rewrite is right

A rewrite that is fast but wrong is worse than the loop. A short routine:

1. Keep the loop version as a reference and compare on random inputs with `np.testing.assert_allclose(fast(x), slow(x))`. Include edge cases: empty arrays, a single element, repeated keys, negative and huge values, NaN.
2. Check shapes explicitly with `assert out.shape == (n, k)`. Silent broadcasting (01.03) is the most common way a vectorised rewrite goes wrong while still running.
3. Turn warnings into errors while testing: `with np.errstate(over='raise', invalid='raise', divide='raise'):`. An overflow inside `exp` then stops the program instead of quietly producing `inf`.
4. Time it with `timeit` on realistic sizes, repeating each run; if the vectorised version is not faster, the array is probably small.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| `g[idx] += v` with repeated indices | group totals too small, no error | `np.add.at(g, idx, v)` or `np.bincount(idx, weights=v)`
| `np.vectorize(f)` used "for speed" | no speed-up at all | rewrite the body with ufuncs, or compile with Numba
| `np.where(x > 0, np.log(x), 0)` | RuntimeWarning, NaN in a branch you discarded | both branches are computed: use `np.log(x, out=np.zeros_like(x), where=x > 0)`
| `ints += 0.5` | `UFuncTypeError` about `same_kind` casting | make the array float first, or assign `ints = ints + 0.5`
| `where=` without `out=` | random-looking garbage in some positions | always pass an initialised `out=` array
| `np.exp(logits)` on raw logits | `inf`, then `nan` after division | subtract the max (softmax, logsumexp)
| `np.log(1 - sigmoid(z))` | loss is `inf` for confident predictions | work in logits: `np.logaddexp(0, z) - y*z`
| vectorising tiny arrays in a hot loop | slower than plain Python | batch the work into one big array, or leave it as Python

The `np.where` row surprises people: Python evaluates all arguments before the call, so `np.log(x)` runs on the negative elements too. The answer is right, but you get warnings, and in autodiff libraries such as JAX the discarded NaN can still leak into gradients.

## Going deeper

**Why the speed-up is so large.** A Python loop pays per element for bytecode dispatch, boxing each value as a Python object, type checks and reference counting. A ufunc pays once per call, then streams through memory, often with SIMD instructions (AVX2, NEON) handling several elements at once. Once arrays outgrow the CPU cache, memory bandwidth sets the speed, which is why fewer passes over memory (fewer temporaries) pays off at that scale.

**Temporary elision.** On CPython, when NumPy sees an operation whose left operand is a large temporary that nothing else references (for example the result of `a * b` inside `a * b + c`), it reuses that buffer for the result instead of allocating a new one. This is why the plain expression in the memory figure allocated 160 MB rather than 320 MB. It does not cover function calls like `np.exp` or small arrays, so explicit `out=` remains the tool for guaranteed memory use. Libraries such as `numexpr`, and compilers in JAX (`jit`) and PyTorch (`torch.compile`), go further by **fusing** a whole expression into one loop that reads each input once.

**Casting.** Each ufunc call resolves an output dtype from its inputs: adding two int8 arrays stays int8 and wraps silently (100 + 100 gives -56). The `casting=` keyword (`'safe'`, `'same_kind'`, `'unsafe'`, ...) controls what `out=` may receive; the in-place error above is the `same_kind` rule.

**Fused losses.** Deep-learning libraries ship fused `log_softmax` and `cross_entropy` functions that never form the probabilities at all, which is both faster and more accurate than softmax followed by log.

**Summation order.** `np.sum` uses **pairwise summation** (adding partial sums of blocks), which has smaller rounding error than a left-to-right loop. So the vectorised result can differ from your loop in the last digits, and is usually the more accurate one. Compare with `assert_allclose`, never `==`.

## Key terms

| Term | Meaning
| --- | --- |
| Vectorisation | Replacing a per-element Python loop with whole-array operations that loop in compiled code
| Ufunc | A NumPy elementwise function with compiled loops per dtype, broadcasting, and methods such as `reduce`
| reduce / accumulate | Fold an axis to one value / keep every running value of that fold
| `ufunc.at` | Unbuffered in-place update where repeated indices each apply
| Temporary | An intermediate array created by one step of an expression and then thrown away
| `out=` | Argument telling a ufunc to write into an existing array instead of allocating
| Overflow / underflow | A result too large (becomes `inf`) or too small (becomes 0) for the dtype
| Log-sum-exp trick | Subtracting the maximum before exponentiating so sums of exponentials stay finite

<div class="take">
<div class="lab">Key takeaway</div>

write the loop body as whole-array ufunc calls (masks, cumulative sums, bincount, searchsorted, broadcasting), test the rewrite against the loop, reuse buffers with `out=` only where array size demands it, and never exponentiate a raw logit: subtract the maximum first.

</div>
