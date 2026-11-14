---
id: "05.01"
course: "ml-toolkit"
module: "05"
slug: "jax-as-numpy"
title: "JAX as NumPy"
keyIdeas: ["`jax.numpy` mirrors the NumPy API closely enough that most array code ports by changing the import, but every result is a `jax.Array` that lives on a device (CPU, GPU or TPU) and is produced by XLA-compiled kernels.", "JAX arrays are **immutable**. `x[i] = v` raises a `TypeError`; the functional form `x.at[i].set(v)` returns a new array, and under `jit` the compiler turns it into an in-place write whenever the old value is no longer needed.", "`.at[]` also provides `add`, `multiply`, `min`, `max` and `get`. Repeated indices in `.at[idx].add(1)` all accumulate, unlike NumPy's `a[idx] += 1`, which applies each repeated index only once.", "Defaults differ from NumPy: floats are float32 and integers int32 unless 64-bit mode is enabled, out-of-bounds reads clamp instead of raising, and out-of-bounds updates are silently dropped.", "Dispatch is asynchronous. A JAX call returns as soon as the work is queued, so time code with `.block_until_ready()`, and remember that `np.asarray(x)` or `float(x)` copies the result back to the host and waits for it."]
question: "You port a NumPy preprocessing function to JAX. It builds `out = np.zeros(n)` and then loops `for i, j in enumerate(idx): out[j] += w[i]`, where `idx` has repeated entries and occasionally contains `n` (one past the end) as a sentinel. Rewrite it in JAX and explain what the rewritten version does with the repeats and the sentinel, compared with the original loop."
answer: "The idiomatic rewrite is `out = jnp.zeros(n).at[idx].add(w)`: one scatter-add with no Python loop, which also compiles under `jit`. Repeats behave like the original loop, because `.at[].add` accumulates every occurrence (unlike the vectorised NumPy `out[idx] += w`, which would keep only one write per position). The sentinel differs. The NumPy loop raises `IndexError` on `out[n]`, whereas JAX's scatter drops out-of-bounds updates by default, so sentinel entries are silently ignored. That happens to be the desired behaviour here, but it is better to make it explicit, for example `.at[idx].add(w, mode='drop')`, or to zero the weights of sentinel entries with `jnp.where(idx < n, w, 0)`, so a future change to the indexing mode or to the sentinel value cannot quietly change results. Note also that the result is float32 unless 64-bit mode is enabled, so if the NumPy version relied on float64 accumulation for very long sums, compare the results with a tolerance."
sources: [{"title": "JAX docs: Quickstart", "url": "https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html", "note": "jax.numpy as a NumPy replacement and a first look at jit, grad and vmap."}, {"title": "JAX docs: Common gotchas in JAX", "url": "https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html", "note": "Pure functions, in-place updates with .at[], out-of-bounds indexing and the 64-bit default."}, {"title": "JAX docs: JAX 101", "url": "https://docs.jax.dev/en/latest/101/index.html", "note": "jax.Array, transformations, tracing and how jax.numpy relates to jax.lax."}]
practice: ["functional-array-update", "masked-mean-without-boolean-indexing", "pad-and-mask-to-bucket-length"]
tags: ["jax.numpy", "immutability", "functional updates"]
tools: ["JAX", "NumPy"]
---

## Orientation: what JAX adds to NumPy

JAX is NumPy plus a set of **composable function transformations**: `grad` turns a function into its gradient, `jit` compiles it, `vmap` batches it, and the sharding tools spread it over many devices. The transformations only work on *pure* functions (output depends only on inputs, no hidden state, no mutation), and almost every difference you will notice from NumPy or PyTorch follows from that one requirement. This module assumes the NumPy module (01) and contrasts with PyTorch (04) where it helps. Lessons 2 to 4 cover the three core transformations, lesson 5 shows how whole models and optimisers are expressed as nested containers called pytrees, lesson 6 covers explicit randomness and numerics, and lesson 7 scales out with meshes and sharding.

~~~dg-layers The pieces of JAX this module covers, from the bottom up.
jax.numpy and jax.lax | array operations, lowered to XLA
*Transformations | grad, jit, vmap, compose freely
Pytrees | parameters and optimiser state as nested containers
Sharding | NamedSharding, shard_map, collectives
~~~

## Same API, different array

```python
import jax
import jax.numpy as jnp

x = jnp.linspace(0.0, 1.0, 5)
y = jnp.sin(x) @ jnp.ones(5)        # same names as NumPy
type(x)                             # jax.Array
x.dtype                             # float32, not float64
x.devices()                         # {CudaDevice(id=0)} on a GPU machine
```

`jnp` functions accept NumPy arrays and Python scalars and return `jax.Array`s. The array is allocated on the default device (the first accelerator if there is one), and `jax.device_put(x, device)` moves it explicitly. There is no `.to(device)` per operation as in PyTorch; data simply stays where JAX put it unless you ask for a transfer. `np.asarray(x)` brings it back as a NumPy array on the host.

Lower-level operations live in `jax.lax` (`lax.dot_general`, `lax.conv_general_dilated`, the control-flow primitives of lesson 3). `jnp` is a thin layer of NumPy-style conveniences built on top of `lax`.

## Immutability and functional updates

In PyTorch you write `x[0] = 1` or `x.add_(1)` and autograd tracks versions to catch problems. JAX forbids mutation entirely:

```python
x = jnp.zeros(5)
x[0] = 1.0                      # TypeError: JAX arrays are immutable
x = x.at[0].set(1.0)            # new array; the old one is unchanged
x = x.at[1:3].add(2.0)          # also: multiply, min, max, divide, power
v = x.at[jnp.array([0, 4])].get()   # same as x[...], with indexing options
```

Copying on every update sounds expensive, but inside a `jit`-compiled function XLA sees the whole computation, knows the old `x` is dead after the update, and performs the write in place. Outside `jit`, each `.at[]` call does allocate a new array, which is one reason to compile hot loops.

The `.at[]` scatter has a semantic advantage too. Repeated indices accumulate:

```python
idx = jnp.array([1, 1, 3])
jnp.zeros(5).at[idx].add(1.0)   # [0., 2., 0., 1., 0.]
```

The NumPy equivalent `a[idx] += 1` gives `[0, 1, 0, 1, 0]`, because it reads, adds and writes back once per unique position. NumPy needs `np.add.at` for the accumulating behaviour; in JAX it is the default, which makes histograms, embedding-gradient scatters and segment sums one-liners.

## Defaults that differ from NumPy

| behaviour | NumPy | JAX
| --- | --- | --- |
| default float / int | float64 / int64 | float32 / int32 (64-bit only with `jax_enable_x64`)
| `x[i] = v` | in place | `TypeError`; use `x.at[i].set(v)`
| index out of bounds on read | `IndexError` | clamped to the last valid index
| index out of bounds on update | `IndexError` | update silently dropped (configurable via `mode=`)
| boolean mask `x[mask]` | data-dependent shape, fine | fine eagerly, fails under `jit`
| execution | synchronous | asynchronous dispatch

Out-of-bounds reads clamp because raising an error from inside an accelerator kernel is not possible; `jnp.arange(3)[10]` returns `2`. That is convenient for padded gathers and dangerous for indexing bugs, so validate indices in tests.

Boolean-mask indexing produces an array whose length depends on the data, which `jit` cannot compile because shapes must be known at trace time (lesson 3). The JAX idiom is to keep the shape fixed and mask arithmetically:

```python
def masked_mean(x, mask):
    return jnp.sum(jnp.where(mask, x, 0.0)) / jnp.maximum(mask.sum(), 1)
```

The same idea drives padding variable-length sequences to a few fixed bucket lengths, so that a handful of compiled shapes cover the whole dataset.

## Asynchronous dispatch

JAX returns from a call as soon as the work is enqueued on the device, like CUDA in PyTorch. Python keeps running ahead while the accelerator computes. Anything that needs the actual values blocks: printing, `float(x)`, `np.asarray(x)`, `x.item()`. To time code honestly, call `.block_until_ready()` on the output, and exclude the first call of a jitted function, which includes compilation:

```python
f = jax.jit(lambda a: a @ a.T)
a = jnp.ones((4096, 4096))
f(a).block_until_ready()             # warm-up: trace + compile + run
%timeit f(a).block_until_ready()     # steady-state time only
```

<div class="ex">
<div class="lab">Worked example</div>

**a token-count histogram**

You need counts of each token id in a batch of shape (B, T) over a vocabulary of size V. The NumPy reflex `counts[tokens.ravel()] += 1` undercounts in NumPy and raises in JAX.

The JAX version is `jnp.zeros(V, jnp.int32).at[tokens.ravel()].add(1)`. Repeated ids accumulate, shapes are static, so it compiles under `jit` and runs as a single scatter kernel on the device.

If some ids can be a padding value of -1, note that negative indices wrap as in NumPy, so -1 would count into the last bucket. Mask first with `jnp.where(tokens >= 0, 1, 0)` as the update values, or remap padding to an id you later ignore.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

write JAX as NumPy without mutation. Replace in-place writes with `.at[]`, keep shapes independent of data, expect float32 and clamped indices, and block on results before you time them.

</div>
