---
id: "05.04"
course: "ml-toolkit"
module: "05"
slug: "vectorising-with-vmap"
title: "Vectorising with vmap"
keyIdeas: ["`jax.vmap(f)` turns a function written for one example into one that works on a batch. It pushes the batch dimension through every primitive with batching rules, so the result is the same vectorised code you would write by hand, not a Python loop; measured here it is about 700 times faster than a loop at 10,000 examples.", "`in_axes` says which axis of each positional argument is mapped (`0`, another integer, or `None` to share it), and can be a pytree matching the argument; `out_axes` places the batch axis in the outputs. Keyword arguments are always mapped on axis 0.", "Nesting `vmap` maps over several axes, for example `vmap(vmap(dist, (None, 0)), (0, None))` for all pairwise distances, or single-head attention mapped over heads and then over the batch, without writing any broadcasting by hand.", "`vmap(grad(loss), in_axes=(None, 0, 0))` gives per-example gradients: one gradient pytree per example, each leaf with a leading batch axis. This is what DP-SGD clipping and gradient-noise diagnostics need.", "Under `vmap`, a branch on batched data runs both sides and selects, so unsafe branches need guarded inputs, and memory grows with the batch: per-example gradients cost one copy of the parameters per example."]
question: "For DP-SGD you compute `jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))(params, X, Y)` on a 350M-parameter model with batch 256 in float32, and the step runs out of memory on an 80 GB GPU even though a normal training step at batch 256 fits easily. Explain the memory difference and give two ways to make it fit while keeping exact per-example clipping."
answer: "A normal step reduces over the batch inside the backward pass, so it only ever holds one gradient pytree (350M floats, about 1.4 GB) plus activations. The vmapped version materialises one gradient per example: 256 x 1.4 GB is about 358 GB, far beyond 80 GB, before counting activations. Clipping only needs each example's norm and the clipped sum, so the full per-example tensor never has to exist at once. First option: split the batch into micro-batches of, say, 16 and loop over them with `lax.scan` (or `lax.map`), computing per-example gradients for 16 examples (about 22 GB), clipping them, and adding their sum into a running accumulator in the scan carry; at the end divide by 256 and add noise. Second option: a two-pass approach, computing per-example gradient norms cheaply (for dense layers the norm of an outer-product gradient factorises into norms of activations and output gradients, the 'ghost clipping' trick) and then running one ordinary backward pass on a loss reweighted by each example's clip factor, which gives the clipped sum with a single gradient-sized buffer. Reducing precision or using `jax.checkpoint` addresses activations, not the per-example gradient buffer, so they do not fix this on their own."
sources: [{"title": "JAX docs: Transformations: grad and vmap", "url": "https://docs.jax.dev/en/latest/101/transformations.html", "note": "vmap, in_axes and out_axes, with the manual-batching comparison."}, {"title": "JAX docs: The autodiff cookbook", "url": "https://docs.jax.dev/en/latest/301/cookbook.html", "note": "Per-example gradients by composing vmap and grad, and how jacfwd and jacrev use vmap."}, {"title": "JAX API: jax.vmap", "url": "https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html", "note": "The full signature, including axis_name, axis_size and pytree in_axes."}]
practice: ["per-example-gradient-clipping", "batched-attention-with-vmap", "model-ensemble-with-in-axes", "pairwise-distances-with-nested-vmap"]
tags: ["vmap", "batching", "per-example gradients"]
tools: ["JAX"]
---

## What you will learn

- What `jax.vmap` does: take a function written for one example and turn it into one that handles a whole batch, with no Python loop and no hand-written broadcasting.
- How `in_axes` and `out_axes` say which axis of each input is the batch axis, which inputs are shared, and where the batch axis goes in the result.
- How nesting `vmap` builds all-pairs computations (distance matrices, batched multi-head attention) from a two-argument function.
- How `vmap(grad(loss))` gives one gradient per example, and how to clip those gradients for differentially private training.
- Where `vmap` behaves differently from a loop (`lax.cond`, while loops, `where` traps) and what it costs in memory.

*Before you start:* you should be comfortable with `jax.numpy` arrays and shapes (05.01), `jax.grad` (05.02) and `jax.jit` (05.03). Pytrees appear briefly; lesson 05.05 covers them in full.

Lesson 05.03 made one function fast by compiling it. This lesson covers the other half of speed: doing the same work for many examples at once. A Python loop over examples wastes the hardware, and hand-written batched code is easy to get wrong. `vmap` gives you the batched version automatically.

## The idea in one picture

Here is a tiny linear scorer, written for **one** input vector $x$ of length 3:

```python
import jax, jax.numpy as jnp

def predict(w, b, x):          # w: (3,), b: scalar, x: (3,)
    return w @ x + b           # one number

w = jnp.array([2., -1., 0.5]); b = 1.0
X = jnp.array([[1., 0., 2.],   # three examples, one per row
               [0., 3., 1.],
               [2., 1., 0.]])

batched = jax.vmap(predict, in_axes=(None, None, 0))
print(batched(w, b, X))        # [ 4.  -1.5  4. ]
```

You never touched `predict` to make it batch-aware. The call `jax.vmap(predict, in_axes=(None, None, 0))` reads: "keep `w` and `b` the same for everyone, and feed `X` in one row at a time along axis 0". The result has one entry per row.

~~~dg-fig 05.04-idea.svg
Left: predict is written for one input vector and returns a scalar. Right: vmap keeps w shared (None) and maps over the rows of X (axis 0), giving one output per row with matching colours.
~~~

Three words to fix now:

- **Vectorising** (or **batching**) means rewriting per-example work as a few big array operations on the whole batch.
- The **batch axis** (or **mapped axis**) is the axis that indexes examples. For `X` above it is axis 0, of size 3.
- `jax.vmap` is a **function transformation**, like `grad` and `jit`: it takes a function and returns a new function.

Check the first output by hand: $2 \times 1 + (-1) \times 0 + 0.5 \times 2 + 1 = 4$. The second: $0 - 3 + 0.5 + 1 = -1.5$. Exactly what a loop would give, but computed as one matrix-vector product.

## Step 1: what vmap actually does

The obvious way to batch is a loop: `jnp.stack([predict(w, b, x) for x in X])`. It is correct, but each iteration is a separate trip from Python to the device.

`vmap` takes a different route. It traces `predict` once (the same tracing you met with `jit` in 05.03), but with a tracer that carries an extra, hidden batch dimension. Each primitive operation in the function is then replaced by its **batching rule**: the batched form of that operation. A dot product of two vectors becomes a matrix-vector product, an elementwise `tanh` just applies to the bigger array, and a `sum` over an axis shifts its axis number by one so it skips the batch axis.

You can see the rewrite with `jax.make_jaxpr`, which prints the traced program:

```python
print(jax.make_jaxpr(predict)(w, b, X[0]))
#   d:f32[]  = dot_general[dimension_numbers=(([0], [0]), ([], []))] a c
print(jax.make_jaxpr(batched)(w, b, X))
#   d:f32[3] = dot_general[dimension_numbers=(([0], [1]), ([], []))] a c
```

The single-example program contracts two vectors into a scalar `f32[]`; the vmapped one contracts `w` against axis 1 of the `(3, 3)` matrix and returns `f32[3]`. No loop appears: it is the same code as a hand-written `X @ w + b`.

The result is a normal JAX function, so it composes with `jit` and `grad` in any order.

## Step 2: measure the difference

How much does this matter? Below is a measurement on this course's CPU container for a slightly bigger function, `tanh(W @ x + b)` with $W$ of shape $(32, 64)$, at batch sizes from 10 to 10,000.

~~~dg-fig 05.04-timing.svg
Measured time per call on a 2-thread CPU (machine-dependent). The Python loop grows in step with the batch; jit(vmap(predict)) sits on top of the hand-batched version and is about 700 times faster than the loop at 10,000 examples.
~~~

Three things to read off the plot:

1. The loop's time grows linearly with batch size: each example pays a fixed dispatch overhead (about 60 microseconds here).
2. `jit(vmap(predict))` and the hand-written `jnp.tanh(X @ W.T + b)` lie on top of each other: `vmap` produces the program a careful person would write.
3. `vmap` without `jit` beats the loop but still dispatches operations one by one; wrap it in `jit` to fuse them.

On a GPU or TPU the gap is usually larger, because one big matrix multiply keeps thousands of arithmetic units busy (no GPU numbers are shown; this container has none).

## Step 3: in_axes and out_axes

`in_axes` has one entry per positional argument. Each entry says where that argument's batch axis is:

- an integer `0`, `1`, ... means "map over this axis";
- `None` means "do not map; give every example the same full value" (we say the argument is **broadcast** or shared).

All mapped arguments must have the same size along their mapped axis. If `X` has 3 rows and `y` has 4 entries, you get `ValueError: vmap got inconsistent sizes for array axes to be mapped`.

The picture below shows the same $2 \times 3$ matrix $M$ handed to `jnp.sum` three ways.

~~~dg-fig 05.04-in-axes.svg
Same matrix, three in_axes. Axis 0 gives two calls, each seeing one row; axis 1 gives three calls, each seeing one column; None gives every call the full matrix.
~~~

```python
M = jnp.arange(6.).reshape(2, 3)          # [[0,1,2],[3,4,5]]
print(jax.vmap(jnp.sum)(M))               # [ 3. 12.]   rows
print(jax.vmap(jnp.sum, in_axes=1)(M))    # [3. 5. 7.]  columns

double = lambda r: r * 2
print(jax.vmap(double, in_axes=1)(M).shape)              # (3, 2)  batch axis first
print(jax.vmap(double, in_axes=1, out_axes=1)(M).shape)  # (2, 3)  batch axis kept at 1
```

`out_axes` says where to put the batch axis in each output. The default `0` puts it first, which is why mapping over columns gives `(3, 2)` unless you ask for `out_axes=1` (handy for time-major data).

`in_axes` can also be a **pytree** (a nested tuple, list or dict) that mirrors the argument. This lets you map over some leaves of a parameter dict and share others:

```python
p = {"w": jnp.ones((4, 3)), "b": jnp.zeros(())}
f = lambda p, x: p["w"] @ x + p["b"]
jax.vmap(f, in_axes=({"w": 0, "b": None}, None))(p, jnp.ones(3)).shape   # (4,)
```

The four rows of `w` are mapped; `b` and `x` are shared. Summary:

| call | meaning
| --- | --- |
| `vmap(f)` | map axis 0 of every argument and every output
| `vmap(f, in_axes=(None, 0))` | first argument shared, second batched along axis 0
| `vmap(f, in_axes=1, out_axes=1)` | batch lives on axis 1 on both sides, e.g. time-major data
| `vmap(f, in_axes=({"w": 0, "b": None}, 0))` | map only some leaves of a dict argument
| `vmap(f, axis_name="batch")` | name the axis so collectives like `lax.psum(x, "batch")` can reduce over it

Two details catch people out. First, `vmap` controls only **positional** arguments with `in_axes`; keyword arguments are always mapped along axis 0. So `jax.vmap(lambda a, s=1.: a * s)(jnp.ones(3), s=jnp.arange(3.))` returns `[0., 1., 2.]`: `s` was mapped, not shared. Pass everything positionally and say what you mean in `in_axes`. Second, with `axis_name` a function can talk across the batch: `jax.vmap(lambda x: x - jax.lax.pmean(x, "i"), axis_name="i")(jnp.array([1., 2., 6.]))` returns `[-2., -1., 3.]`, each value minus the batch mean. This is the same collective vocabulary used for multi-device code in 05.07.

## Step 4: nesting vmap for all pairs

Each `vmap` adds one batch dimension, so nesting two of them maps over two. The classic use is a table of pairwise distances between point sets $A$ (shape $n \times d$) and $B$ (shape $m \times d$). Write the distance for **two single points**, then map twice:

```python
def sqdist(a, b):                     # two vectors -> scalar
    return jnp.sum((a - b) ** 2)

pairwise = jax.vmap(jax.vmap(sqdist, in_axes=(None, 0)), in_axes=(0, None))

A = jnp.array([[0., 0.], [1., 2.], [3., 1.]])
B = jnp.array([[1., 1.], [3., 0.]])
D = pairwise(A, B)
print(D)                        # [[2. 9.] [1. 8.] [4. 1.]]   shape (3, 2)
print(jnp.argmin(D, axis=1))    # [0 0 1]
```

Read the nesting inside out. The inner `vmap(sqdist, in_axes=(None, 0))` takes **one** point $a$ and all of $B$, returning a row of $m$ distances. The outer `vmap(..., in_axes=(0, None))` repeats that for every row of $A$, giving the $n \times m$ table.

~~~dg-fig 05.04-pairwise.svg
Left: three points in A and two in B, with a dotted line from each A point to its nearest B point. Right: the 3 by 2 table produced by nested vmap; the highlighted cell in each row is the argmin.
~~~

Check one entry by hand: $a_1 = (1, 2)$ and $b_0 = (1, 1)$ differ by $(0, 1)$, so the squared distance is $0^2 + 1^2 = 1$. And $a_2 = (3, 1)$ to $b_1 = (3, 0)$ is also 1, which is why $a_2$'s nearest neighbour is $b_1$.

The same recipe builds batched attention. Write attention for one head and one sequence, with `(T, d)` queries, keys and values and a `(T,)` padding mask. Then `vmap` over heads, and `vmap` again over the batch, passing the mask as another mapped argument.

~~~dg-flow Building a batched attention function from the single-example version.
attend(q, k, v, mask) -(vmap over heads)-> multi-head -(vmap over batch)-> *batched attention -(jit)-> fused kernels
~~~

You never write `[:, None, :]`, and the single-head function stays easy to test alone.

## Step 5: per-example gradients

Ordinary training averages gradients over the batch inside the backward pass, so individual examples' gradients are never seen. Some methods need them:

- **DP-SGD** (differentially private SGD) clips each example's gradient to a maximum norm $C$ before averaging, so no single person's data can move the model much.
- Gradient-noise and influence analyses look at how spread out the per-example gradients are.

In JAX this is one line. Write the loss for a single example, take `grad`, then `vmap` with the parameters shared:

```python
def loss(w, x, y):                     # one example, squared error
    return (w @ x - y) ** 2

y = jnp.array([4., -1., 2.])
per_ex = jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))
g = per_ex(w, X, y)                    # (3, 3): one gradient row per example
print(g)       # [[-2. -0. -4.] [-0. -9. -3.] [ 4.  2.  0.]]
```

The order matters: `grad` first makes "gradient of one example's loss", and `vmap` then runs it for every example.

<div class="ex">
<div class="lab">Worked example</div>

**per-example gradients and clipping by hand**

Use the same $w = (2, -1, 0.5)$ and rows of $X$, with targets $y = (4, -1, 2)$. For squared error $(w \cdot x - y)^2$ the gradient with respect to $w$ is $2\,(w \cdot x - y)\,x$.

Example 1: $w \cdot x = 2 + 0 + 1 = 3$, residual $3 - 4 = -1$, gradient $2 \times (-1) \times (1, 0, 2) = (-2, 0, -4)$.

Example 2: $w \cdot x = 0 - 3 + 0.5 = -2.5$, residual $-2.5 - (-1) = -1.5$, gradient $-3 \times (0, 3, 1) = (0, -9, -3)$.

Example 3: $w \cdot x = 4 - 1 + 0 = 3$, residual $3 - 2 = 1$, gradient $2 \times (2, 1, 0) = (4, 2, 0)$. These match the printed `g`.

Norms: $\sqrt{4 + 16} = \sqrt{20} \approx 4.47$, $\sqrt{81 + 9} = \sqrt{90} \approx 9.49$, and again $\sqrt{20} \approx 4.47$.

Clip to $C = 4$: each gradient is multiplied by $\min(1, C / \text{norm})$, giving scales $4/4.47 \approx 0.894$, $4/9.49 \approx 0.422$ and $0.894$. Example 2, the outlier, is shrunk the most.

Clipped gradients: $(-1.789, 0, -3.578)$, $(0, -3.795, -1.265)$, $(3.578, 1.789, 0)$. Their mean is $(0.596, -0.669, -1.614)$.

Compare the unclipped mean $(2/3, -7/3, -7/3) \approx (0.667, -2.333, -2.333)$, which is what `jax.grad` of the batch-mean loss returns. Clipping cut the second and third components sharply because example 2 no longer dominates.

</div>

For a real model the parameters are a pytree, so each leaf of `g` gets a leading batch axis. To clip, compute each example's norm across all leaves, scale, then average:

```python
sq = sum(jnp.sum(l.reshape(l.shape[0], -1) ** 2, axis=1) for l in jax.tree.leaves(g))
scale = jnp.minimum(1.0, C / (jnp.sqrt(sq) + 1e-12))          # (B,)
clipped = jax.tree.map(lambda l: jnp.mean(l * scale.reshape(-1, *[1] * (l.ndim - 1)), 0), g)
```

~~~dg-fig 05.04-clip.svg
Left: per-example gradient norms for 24 logistic-regression examples; anything above C = 1 is scaled down to exactly 1. Right: holding every per-example gradient takes batch size × parameters × 4 bytes, which passes 80 GB quickly for large models.
~~~

The right panel is the price: a vmapped per-example step holds $B$ gradients instead of one. For a 350M-parameter model in float32 that is 1.4 GB per example, about 358 GB at batch 256. For large models, process micro-batches with `lax.scan`, clip each, and add the clipped sums into a running total.

## Ensembles: in_axes is not only for data

The mapped argument need not be data. Stack the parameters of $E$ models along a new leading axis with `jax.tree.map(lambda *ps: jnp.stack(ps), *param_list)`. Then:

- `jax.vmap(apply, in_axes=(0, None))(stacked, x)` runs every member on the **same** batch and returns `(E, B, ...)`;
- `jax.vmap(apply, in_axes=(0, 0))` gives each member **its own** batch, which is how bootstrapped ensembles are trained.

Initialisation is a `vmap` over random keys (05.06).

<div class="ex">
<div class="lab">Worked example</div>

**an ensemble of five MLPs in one call**

Initialise five parameter sets with `jax.vmap(init_params)(jax.random.split(key, 5))`. This returns a single pytree whose every leaf has a leading axis of 5.

Training uses `jax.vmap(train_step, in_axes=(0, 0, None))(params, opt_state, batch)`: every member sees the same batch but has its own parameters and optimiser state. One compiled step updates all five, and on a GPU it runs close to the speed of one larger model because the matrix multiplies are batched.

At evaluation, `preds = jax.vmap(apply, in_axes=(0, None))(params, x)` has shape (5, B, classes). The mean over axis 0 is the ensemble prediction and the variance over axis 0 a cheap uncertainty estimate.

</div>

## Checking that vmap did what you meant

Batching bugs are usually silent shape bugs. Check:

1. **Compare against a loop on a few examples.** `jnp.allclose(jax.vmap(f, ...)(X[:4]), jnp.stack([f(x) for x in X[:4]]))` should be true. This one check catches most `in_axes` mistakes.
2. **Print shapes, not just values.** The output's leading axis should be the batch size. If you see `(d, B)` instead of `(B, d)`, you mapped the wrong axis or need `out_axes`.
3. **Test the single-example function first.** Batching will not fix a broken `f(X[0])`.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| `inconsistent sizes for array axes to be mapped` | two mapped arguments have different batch sizes | check shapes; mark shared arguments `None`
| Output has the wrong shape, e.g. `(d, B)` | mapped the wrong axis, or forgot `out_axes` | set `in_axes`/`out_axes` explicitly; compare with a loop
| A keyword argument that should be shared varies per example | keyword arguments are always mapped on axis 0 | pass it positionally with `None` in `in_axes`
| Code already written for batches gets an extra axis | `vmap` over an already-batched function | write the function for one example, or drop the `vmap`
| NaN gradients from code with a branch on the data | the masked-out branch is still computed (e.g. `log` of 0 inside `jnp.where`) | make the input safe before the risky op, then select
| Out-of-memory with `vmap(grad(...))` | one gradient copy per example | micro-batch with `lax.scan`, accumulate clipped sums
| Slower than expected | not jitted, so each batched op dispatches separately | wrap in `jax.jit`

The NaN row needs an example. Under `vmap`, data-dependent branches run as a **select**: both sides are computed for every example and the right one is picked. The gradient of `jnp.where(x > 0, jnp.log(x), 0.)` at `x = 0` comes out as `nan`: the unused `log` branch has an infinite derivative, and infinity times the zero from the mask is NaN. The standard fix is the "double where":

```python
def safe_log(x):
    x_safe = jnp.where(x > 0, x, 1.0)          # never feed 0 to log
    return jnp.where(x > 0, jnp.log(x_safe), 0.0)
print(jax.vmap(jax.grad(safe_log))(jnp.array([2., 0.])))   # [0.5 0. ]
```

## Going deeper

**How batching rules work.** Every JAX primitive has a batching rule: given inputs that each carry an optional batch-axis position, it returns the batched output and its batch-axis position. `vmap` applies these rules through the program, moving axes only when it must. For `dot_general` the batch axis becomes a batch dimension of the matrix multiply, which maps onto efficient batched GEMM kernels.

**Control flow.** A `lax.cond` whose predicate is the same for the whole batch stays a real branch. If the predicate is batched, no single branch is right for everyone, so `vmap` converts the `cond` into a select that evaluates **both** branches; expensive or unsafe branches then run for every example. A `lax.while_loop` with a batched condition keeps running until every example is finished, carrying finished examples along unchanged, so the batch costs as much as its slowest member.

**Jacobians are vmaps.** `jax.jacrev` maps a vector-Jacobian product over the rows of an identity matrix with `vmap`, and `jax.jacfwd` maps a Jacobian-vector product over its columns.

**Memory and devices.** `vmap` materialises the whole batch at once. When that does not fit, `lax.map` processes chunks in sequence, trading speed for memory; to spread a batch across devices, combine `vmap` with sharding (05.07). PyTorch's `torch.func.vmap` was modelled on JAX's, so these patterns carry over almost unchanged.

## Key terms

| Term | Meaning
| --- | --- |
| Vectorising / batching | Doing per-example work as a few large array operations over the whole batch
| `jax.vmap` | Transformation that turns a single-example function into a batched one
| Batch (mapped) axis | The array axis that indexes examples
| `in_axes` | Per-argument batch axis: an integer to map, `None` to share; may be a pytree
| `out_axes` | Where the batch axis goes in each output (default 0)
| Batching rule | The batched version of one primitive, used by `vmap` when rewriting a function
| Nested vmap | Two or more `vmap`s, each adding a batch dimension, e.g. pairwise tables
| Per-example gradients | One gradient per training example, from `vmap(grad(loss))`
| `axis_name` | A name for the mapped axis so collectives like `lax.psum` can reduce over it

<div class="take">
<div class="lab">Key takeaway</div>

write functions for one example and let `vmap` add batch axes. Use `in_axes` to say what is shared and what is mapped, nest it for multiple axes, and combine it with `grad` for per-example gradients, keeping an eye on the memory that a batch of gradients costs.

</div>
