---
id: "05.01"
course: "ml-toolkit"
module: "05"
slug: "jax-as-numpy"
title: "JAX as NumPy"
keyIdeas: ["`jax.numpy` mirrors the NumPy API closely enough that most array code ports by changing the import, but every result is a `jax.Array` that lives on a device (CPU, GPU or TPU) and is produced by XLA-compiled kernels.", "JAX arrays are **immutable** because the transformations (`grad`, `jit`, `vmap`) need pure functions. `x[i] = v` raises a `TypeError`; `x.at[i].set(v)` returns a new array, and under `jit` the compiler turns it into an in-place write whenever the old value is no longer needed.", "`.at[]` also provides `add`, `multiply`, `min`, `max` and `get`. Repeated indices in `.at[idx].add(1)` all accumulate, unlike NumPy's `a[idx] += 1`, which applies each repeated index only once.", "Defaults differ from NumPy: floats are float32 and integers int32 unless 64-bit mode is enabled, out-of-bounds reads clamp instead of raising, out-of-bounds updates are silently dropped, and boolean-mask indexing fails under `jit`, so mask with `jnp.where` instead.", "Dispatch is asynchronous. A JAX call returns as soon as the work is queued, so time code with `.block_until_ready()` after a warm-up call, and remember that `np.asarray(x)` or `float(x)` copies the result back to the host and waits for it. Eager `jnp` is slower than NumPy on small arrays; speed comes from `jit`."]
question: "You port a NumPy preprocessing function to JAX. It builds `out = np.zeros(n)` and then loops `for i, j in enumerate(idx): out[j] += w[i]`, where `idx` has repeated entries and occasionally contains `n` (one past the end) as a sentinel. Rewrite it in JAX and explain what the rewritten version does with the repeats and the sentinel, compared with the original loop. What would change if the sentinel were -1 instead of n?"
answer: "The idiomatic rewrite is `out = jnp.zeros(n).at[idx].add(w)`: one scatter-add with no Python loop, which also compiles under `jit`. Repeats behave like the original loop, because `.at[].add` accumulates every occurrence (unlike the vectorised NumPy `out[idx] += w`, which would keep only one write per position; `np.add.at` is NumPy's accumulating equivalent). The sentinel differs. The NumPy loop raises `IndexError` on `out[n]`, whereas JAX's scatter drops out-of-bounds updates by default, so sentinel entries are silently ignored. That happens to be the desired behaviour here, but it is better to make it explicit, for example `.at[idx].add(w, mode='drop')`, or to zero the weights of sentinel entries with `jnp.where(idx < n, w, 0)`, so a future change to the indexing mode or to the sentinel value cannot quietly change results. If the sentinel were -1, the NumPy loop would not raise either: -1 wraps to the last position, so the original code would add the sentinel weights into `out[n-1]`, and JAX does the same, because negative indices are normalised before the bounds rule applies (even `mode='drop'` keeps them). The only safe fix is the explicit mask, `w_eff = jnp.where(idx >= 0, w, 0)`. Finally, the result is float32 unless 64-bit mode is enabled, so if the NumPy version relied on float64 accumulation for very long sums, compare the two with a tolerance (for example `np.testing.assert_allclose(..., rtol=1e-5)`) rather than exact equality."
sources: [{"title": "JAX docs: Quickstart", "url": "https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html", "note": "jax.numpy as a NumPy replacement and a first look at jit, grad and vmap."}, {"title": "JAX docs: Common gotchas in JAX", "url": "https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html", "note": "Pure functions, in-place updates with .at[], out-of-bounds indexing and the 64-bit default."}, {"title": "JAX docs: JAX 101", "url": "https://docs.jax.dev/en/latest/101/index.html", "note": "jax.Array, transformations, tracing and how jax.numpy relates to jax.lax."}]
practice: ["functional-array-update", "masked-mean-without-boolean-indexing", "pad-and-mask-to-bucket-length"]
tags: ["jax.numpy", "immutability", "functional updates"]
tools: ["JAX", "NumPy"]
---

## What you will learn

- What JAX adds on top of NumPy, and why almost every difference you meet comes from one rule: functions must be pure.
- How to port NumPy code by swapping the import, and what kind of object (`jax.Array`) you get back.
- How to replace in-place writes with `.at[]` updates, and why repeated indices behave differently from NumPy.
- Which defaults change: float32 instead of float64, clamped reads, dropped out-of-bounds writes.
- How asynchronous dispatch works, and how to time JAX code honestly.

*Before you start:* you should be comfortable with NumPy arrays, indexing, broadcasting and dtypes (module 01, especially 01.01 to 01.03). Lessons 04.01 to 04.03 on PyTorch tensors are useful for contrast but not required.

## Where this module starts

JAX is NumPy plus a set of **composable function transformations**. A transformation takes a Python function and gives you back a new function: `grad` returns its gradient, `jit` returns a compiled version, `vmap` returns a batched version, and the sharding tools return a version that runs across many devices. "Composable" means you can stack them: `jit(vmap(grad(loss)))` is a perfectly normal thing to write.

The transformations have one requirement. They only work reliably on **pure functions**: functions whose output depends only on their inputs, which do not read hidden global state, and which do not modify anything. Almost every surprise you will meet when moving from NumPy or PyTorch to JAX follows from that rule.

This lesson covers the array layer. Lessons 05.02 to 05.04 cover the three core transformations, 05.05 stores models and optimiser states as nested containers called pytrees, 05.06 covers explicit random keys and numerics, and 05.07 spreads work over many devices with sharding.

~~~dg-layers The pieces of JAX this module covers, from the bottom up.
jax.numpy and jax.lax | array operations, lowered to XLA
*Transformations | grad, jit, vmap, compose freely
Pytrees | parameters and optimiser state as nested containers
Sharding | NamedSharding, shard_map, collectives
~~~

## The idea in one picture

Here is a small function written once and run with both libraries. It standardises four numbers (subtract the mean, divide by the standard deviation).

```python
import numpy as np
import jax.numpy as jnp

def standardise(x, xp):            # xp is either np or jnp
    return (x - xp.mean(x)) / xp.std(x)

data = [3.0, 5.0, 2.0, 8.0]
a = standardise(np.array(data), np)
b = standardise(jnp.array(data), jnp)
print(a, a.dtype)   # [-0.65465367  0.21821789 -1.09108945  1.52752523] float64
print(b, b.dtype)   # [-0.65465367  0.21821788 -1.0910894   1.5275252 ] float32
```

The code is identical and the answers agree to about seven digits. The visible differences are the result type (`ndarray` versus `jax.Array`) and the precision (float64 versus float32); the largest gap, about $8 \times 10^{-8}$, is float32 rounding.

The difference you cannot see yet is what happens when you try to change an element.

~~~dg-fig 05.01-immutable.svg
NumPy writes into the existing buffer, so x itself changes. JAX never changes x: x.at[2].set(9) builds a new array y and the old one is still there.
~~~

Reading the picture: in NumPy, `x[2] = 9` overwrites memory, and every variable that points at that buffer sees the 9. In JAX the same line raises a `TypeError`. You write `y = x.at[2].set(9)` instead, which returns a new array and leaves `x` untouched. Two terms to fix now:

- An **immutable** array is one whose contents can never change after it is created. All JAX arrays are immutable.
- A **functional update** is an operation that returns a modified copy instead of modifying in place. `.at[...]` is JAX's functional update syntax.

Why would a library forbid mutation? Because `grad`, `jit` and `vmap` work by running your function on stand-in values and recording the operations (lesson 05.03 calls this **tracing**). If the function could quietly change its inputs or some global array, the recorded program would not describe what the function really does. Immutability keeps every function a clean "inputs in, outputs out" box.

## Step 1: swap the import and meet `jax.Array`

Most array code ports by replacing `np.` with `jnp.`. The names, argument orders and broadcasting rules follow NumPy closely.

```python
import jax
import jax.numpy as jnp

x = jnp.linspace(0.0, 1.0, 5)
y = jnp.sin(x) @ jnp.ones(5)        # same names as NumPy
isinstance(x, jax.Array)            # True
x.dtype                             # float32, not float64
x.devices()                         # {CpuDevice(id=0)} here; {CudaDevice(id=0)} on a GPU machine
```

Every result is a **`jax.Array`**: an array that lives on a **device** (CPU, GPU or TPU) and is produced by kernels compiled by **XLA**, the compiler JAX hands its work to. `jnp` functions accept NumPy arrays, Python lists and Python scalars as inputs, and always return `jax.Array`s.

A new array goes on the default device (the first accelerator, if any). `jax.device_put(x, device)` moves it explicitly. There is no `.to(device)` on every operation as in PyTorch: data stays where JAX put it until you ask for a transfer. `np.asarray(x)` copies it back to the host as a NumPy array, and that copy is read-only, because it may share memory with the JAX buffer.

Under `jnp` sits **`jax.lax`**, the lower-level primitives (`lax.dot_general`, the control-flow primitives of lesson 05.03) that `jnp` is a thin layer over. `lax` is strict: `lax.add(jnp.float32(1), jnp.int32(2))` raises a `TypeError` about mismatched dtypes, while `jnp.add` promotes the int to float32 for you, NumPy-style. Stay in `jnp` unless you need a primitive it does not expose.

## Step 2: replace in-place writes with `.at[]`

In PyTorch you can write `x[0] = 1` or `x.add_(1)` and autograd tracks tensor versions to catch problems. JAX forbids mutation entirely and gives you a small family of functional updates instead:

```python
x = jnp.zeros(5)
x[0] = 1.0                          # TypeError: JAX arrays are immutable
x = x.at[0].set(1.0)                # new array; rebinding the name is fine
x = x.at[1:3].add(2.0)              # also: multiply, divide, power, min, max
v = x.at[jnp.array([0, 4])].get()   # same as x[[0, 4]], with extra options
print(x, v)                         # [1. 2. 2. 0. 0.] [1. 0.]
```

The pattern is always the same: `array.at[where].operation(values)`. The `where` part accepts everything ordinary indexing accepts (integers, slices, integer arrays, tuples for several axes). Rebinding `x = x.at[...]...` only moves the name; any other variable holding the old array still sees the old values.

Outside `jit`, each `.at[]` call allocates a whole new array, so a Python loop of single-element updates on a big array is slow. Inside a `jit`-compiled function, XLA sees the entire computation, notices that the old `x` is never used again after the update, and performs the write in place. That is one reason hot loops belong inside `jit` (lesson 05.03).

**Repeated indices accumulate.** The `.at[]` updates are **scatter** operations: each listed position receives its update, and if a position is listed twice it receives two updates. NumPy's fancy-index shorthand behaves differently.

~~~dg-fig 05.01-scatter.svg
Adding 1 at indices [1, 1, 3]. NumPy's a[idx] += 1 reads, adds and writes back once per distinct position, so index 1 ends at 1. np.add.at and JAX's .at[idx].add both apply every occurrence, so index 1 ends at 2.
~~~

```python
idx = jnp.array([1, 1, 3])
jnp.zeros(5).at[idx].add(1.0)       # [0. 2. 0. 1. 0.]

a = np.zeros(5); a[[1, 1, 3]] += 1  # [0. 1. 0. 1. 0.]   (NumPy: one write per position)
```

NumPy needs the special `np.add.at` to accumulate; in JAX it is the default. That makes histograms, segment sums ("add each row to its group's total") and the gradient of an embedding lookup into one-liners.

## Step 3: know the defaults that differ

A handful of defaults differ on purpose. They cause most silent porting bugs, so learn them as a list.

| behaviour | NumPy | JAX
| --- | --- | --- |
| default float / int | float64 / int64 | float32 / int32 (64-bit only with `jax_enable_x64`)
| `x[i] = v` | in place | `TypeError`; use `x.at[i].set(v)`
| index out of bounds on read | `IndexError` | clamped to the last valid index
| index out of bounds on update | `IndexError` | update silently dropped (configurable via `mode=`)
| boolean mask `x[mask]` | data-dependent shape, fine | fine eagerly, fails under `jit`
| execution | synchronous | asynchronous dispatch

**32-bit by default.** Accelerators are much faster in 32-bit (and 16-bit) arithmetic, so JAX makes float32 and int32 the defaults. Converting a float64 NumPy array with `jnp.asarray` silently produces float32. Asking for `dtype=jnp.float64` explicitly only gets you a warning and a float32 array. To get real 64-bit arrays, set `jax.config.update("jax_enable_x64", True)` at program start (or the environment variable `JAX_ENABLE_X64=1`).

~~~dg-fig 05.01-float32.svg
The gap between neighbouring representable numbers grows with the number's size. float32 gaps are about 500 million times wider than float64 gaps; past 2²⁴ (about 16.8 million) the gap exceeds 1, so float32 cannot even store every whole number.
~~~

The figure shows what float32 costs. Near 1.0 the next float32 is about $1.2 \times 10^{-7}$ away, so you get roughly seven significant digits. That is plenty for neural-network weights, but it matters for long sums, for timestamps in seconds, and for integer counts above 16.8 million: `jnp.float32(16_777_217)` comes back as `16777216.0`.

**Out-of-bounds indices do not raise.** An accelerator kernel cannot stop halfway and throw a Python exception, so JAX picks a safe behaviour instead.

~~~dg-fig 05.01-bounds.svg
Three behaviours for index 7 on a length-3 array: a read clamps to the last element, an update to a non-existent slot is dropped, and get(mode='fill') returns a value you choose.
~~~

So `jnp.arange(3)[10]` returns `2`, and `x.at[7].set(99)` returns `x` unchanged. Negative indices still wrap from the end, exactly as in NumPy. Clamping is convenient for padded lookups and dangerous for indexing bugs, because a wrong index produces a plausible number instead of an error. Validate index ranges in tests, and pass `mode=` explicitly when out-of-range indices are expected.

## Step 4: keep shapes independent of the data

Boolean-mask indexing `x[x > 0]` returns an array whose length depends on the values inside `x`. Eagerly that works. Under `jit` it fails with `NonConcreteBooleanIndexError`, because the compiler must know every array's shape before it sees any values (lesson 05.03 explains why). The JAX habit is to keep the shape fixed and do the masking with arithmetic:

```python
def masked_mean(x, mask):
    return jnp.sum(jnp.where(mask, x, 0.0)) / jnp.maximum(mask.sum(), 1)

jax.jit(masked_mean)(jnp.array([1., 2., 3., 4.]),
                     jnp.array([True, False, True, False]))   # 2.0
```

`jnp.where(mask, x, 0.0)` keeps the full length and zeroes the unwanted entries; the `maximum(..., 1)` guard avoids dividing by zero when nothing is selected. Variable-length sequences use the same idea: pad to a few fixed **bucket** lengths and carry a mask, so a handful of compiled shapes cover the dataset.

## Step 5: asynchronous dispatch and honest timing

When you call a JAX function, Python does not wait for the answer. JAX puts the work in the device's queue and immediately hands you a `jax.Array` that will hold the result once it is computed. This is **asynchronous dispatch** (the same model CUDA uses in PyTorch): Python runs ahead and queues more work while the device is busy.

~~~dg-fig 05.01-async.svg
Measured here on a CPU (timings are machine-dependent). The call itself returns in about 0.1 ms whatever the size; the matmul only finishes later, about 115 ms for n = 2048.
~~~

Anything that needs real values waits: printing, `float(x)`, `x.item()`, `np.asarray(x)`. So if you time a JAX call without waiting, you measure only the queuing. Always call `.block_until_ready()` on the result, and exclude the first call of a jitted function, which includes tracing and compilation:

```python
import timeit
f = jax.jit(lambda a: a @ a.T)
a = jnp.ones((2048, 2048))
f(a).block_until_ready()                                   # warm-up: trace + compile + run
timeit.timeit(lambda: f(a).block_until_ready(), number=10) # steady state only
```

~~~dg-fig 05.01-speed.svg
Measured on a 2-thread CPU for sum(tanh(x)*2 + x*x - 1); timings are machine-dependent. Eager jax.numpy pays a fixed dispatch cost per operation; jit fuses the five operations into one kernel and wins clearly on large arrays.
~~~

On tiny arrays NumPy takes about 4 µs, while eager `jnp`, which dispatches each of the five operations separately, takes about 50 µs. On 4 million elements, eager `jnp` and NumPy are similar (about 11-12 ms), and the jitted version is roughly three times faster because XLA fuses the operations and makes one pass over memory. The lesson: eager `jnp` is for convenience and debugging; speed comes from `jit`.

<div class="ex">
<div class="lab">Worked example</div>

**a token-count histogram, by hand and in JAX**

A batch of token ids has shape (2, 3): `tokens = [[2, 0, 2], [1, 2, -1]]`, vocabulary size $V = 4$, and -1 marks padding. We want `counts[v]` = how many times id $v$ appears, ignoring padding.

By hand: the flattened ids are 2, 0, 2, 1, 2, -1. Dropping the padding leaves 2, 0, 2, 1, 2. Id 0 appears once, id 1 once, id 2 three times, id 3 zero times. Target: `[1, 1, 3, 0]`.

NumPy reflex `counts[ids] += 1`: one write per distinct position, so id 2 is counted once, and -1 wraps to the last index 3. Result `[1, 1, 1, 1]`, wrong twice.

JAX `jnp.zeros(4, jnp.int32).at[ids].add(1)`: repeats accumulate, so id 2 gets $1 + 1 + 1 = 3$. But -1 still wraps to index 3 (negative indices are normalised before any bounds check, so even `mode='drop'` keeps it). Result `[1, 1, 3, 1]`, wrong once.

Fix: make the padding add nothing. `.at[ids].add(jnp.where(ids >= 0, 1, 0))` adds 1, 1, 1, 1, 1 and finally 0 at index 3. Result `[1, 1, 3, 0]`, matching the hand count. The shape never depends on the data, so this compiles under `jit` as a single scatter kernel.

</div>

## JAX as NumPy in code

A small cleaning pipeline that uses every idea above: conversion, a fixed-shape mask, a functional update, a scatter-add and the trip back to NumPy.

```python
import numpy as np
import jax, jax.numpy as jnp

# a batch of 6 sensor readings, with -1 meaning "missing"
raw = np.array([4.0, -1.0, 7.0, 5.0, -1.0, 8.0])
x = jnp.asarray(raw)                       # NumPy -> jax.Array (float64 -> float32)
print(x.dtype, x.devices())                # float32 {CpuDevice(id=0)}

valid = x >= 0
mean = jnp.sum(jnp.where(valid, x, 0.0)) / jnp.maximum(valid.sum(), 1)
print(mean)                                # 6.0

filled = jnp.where(valid, x, mean)         # replace missing by the mean
print(filled)                              # [4. 6. 7. 5. 6. 8.]

clipped = filled.at[jnp.argmax(filled)].set(7.5)   # cap the largest reading
print(clipped, filled[5])                  # [4.  6.  7.  5.  6.  7.5] 8.0

bins = jnp.array([0, 1, 1, 0, 1, 2])       # which station each reading came from
per_station = jnp.zeros(3).at[bins].add(clipped)
print(per_station)                         # [ 9.  19.   7.5]

back = np.asarray(per_station)             # host copy, read-only
print(type(back).__name__, back.flags.writeable)   # ndarray False
```

`filled[5]` is still 8.0 after the update, because `clipped` is a new array, and station 1 correctly accumulates $6 + 7 + 6 = 19$.

## Checking a port

Check a port the boring way before trusting it.

1. **Compare against the original** on the same inputs with a tolerance suited to float32: `np.testing.assert_allclose(np_out, np.asarray(jax_out), rtol=1e-5, atol=1e-6)`. Exact equality will fail for honest reasons.
2. **Print dtypes** of inputs and outputs. A surprise float32 is the most common source of a mismatch larger than about $10^{-6}$ relative.
3. **Test edge indices**: empty masks, the last index, one past the end, negative ids. JAX will not raise on these, so your tests must look at the values.
4. **Run the function under `jax.jit`** once. If it fails with a concretisation or boolean-index error, a shape still depends on data (step 4).

## What goes wrong

| Symptom | Cause | Fix
| --- | --- | --- |
| `TypeError: JAX arrays are immutable` | `x[i] = v` or `x += ...` on a slice | `x = x.at[i].set(v)` or `.add(v)`
| Counts too small after porting | NumPy code relied on `a[idx] += 1` | nothing to do in JAX; if checking against NumPy, use `np.add.at`
| Plausible but wrong values, no error | out-of-bounds index clamped on read | validate indices; use `.at[i].get(mode='fill', fill_value=...)`
| Updates silently missing | out-of-bounds update dropped | check index ranges; make the mode explicit
| Results differ from NumPy at the 7th digit | float32 default | compare with tolerance, or enable `jax_enable_x64`
| Large integer counts or timestamps off by one | float32 cannot represent them exactly | keep them in int32, or enable 64-bit
| `NonConcreteBooleanIndexError` under `jit` | `x[mask]` gives a data-dependent shape | `jnp.where(mask, x, 0)` and masked reductions
| Benchmarks look impossibly fast | asynchronous dispatch; timer stops before work ends | `.block_until_ready()`, warm-up call first
| Eager JAX slower than NumPy | per-operation dispatch on small arrays | wrap the function in `jax.jit`
| `ValueError: assignment destination is read-only` | writing into `np.asarray(jax_array)` | `np.array(jax_array)` for a writable copy

## Going deeper

**Why purity is the price of the transformations.** `jit`, `grad` and `vmap` run your Python function once with abstract **tracer** values that carry only shape and dtype, and record every `jnp` or `lax` call into an intermediate program (a *jaxpr*). The transformation then rewrites that program: `grad` adds the backward pass, `vmap` adds a batch dimension, `jit` hands it to XLA. Side effects such as appending to a list, mutating a global or printing happen once during tracing and are absent from the recorded program.

**In-place after all: buffer donation.** Because the program is a pure function, XLA is free to reuse memory. Inside one compiled function it turns dead-after-update arrays into in-place writes. Across calls, `jax.jit(step, donate_argnums=0)` tells JAX that the caller will not use the input buffer again, so the output can overwrite it. Training loops use it to avoid holding two copies of large parameters.

**Scatter semantics.** `.at[idx].add` lowers to an XLA scatter. When indices repeat, every update is applied, but on parallel hardware the order of floating-point additions is not guaranteed, so sums of floats into the same slot may differ in the last bits from run to run on a GPU. Integer scatters are exact. For `.set` with repeated indices, which of the competing values wins is unspecified; do not rely on it. If you can promise the indices are sorted or unique, the `indices_are_sorted` and `unique_indices` arguments let XLA pick faster kernels.

**Type promotion and weak types.** JAX has its own promotion table, designed so that mixing in Python scalars never widens your arrays. A Python `2.5` is **weakly typed**, so `jnp.ones(3, jnp.float16) * 2.5` stays float16; two arrays of different dtypes still promote (int32 with float32 gives float32).

**The array model.** One `jax.Array` type covers single- and multi-device data; its `.sharding` attribute says which devices hold which pieces (lesson 05.07).

## Key terms

| Term | Meaning
| --- | --- |
| `jax.numpy` (`jnp`) | NumPy-compatible API whose functions return `jax.Array`s computed by XLA
| `jax.lax` | Lower-level, stricter primitives that `jnp` is built on
| `jax.Array` | JAX's immutable array type; lives on a device and carries a sharding
| Pure function | Output depends only on inputs; no hidden state, no mutation
| Functional update | `x.at[idx].set/add/...`: returns a modified copy, leaves `x` unchanged
| Scatter | An update applied at listed positions; repeated positions all receive updates
| `jax_enable_x64` | Config flag that turns on float64/int64; off by default
| Asynchronous dispatch | Calls return once work is queued; reading values waits for it
| `block_until_ready()` | Waits until an array's computation has finished; needed for timing

<div class="take">
<div class="lab">Key takeaway</div>

write JAX as NumPy without mutation. Replace in-place writes with `.at[]` (and expect repeated indices to accumulate), keep shapes independent of data, expect float32 and clamped indices instead of errors, and block on results before you time them.

</div>
