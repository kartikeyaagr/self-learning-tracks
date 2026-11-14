---
id: "05.04"
course: "ml-toolkit"
module: "05"
slug: "vectorising-with-vmap"
title: "Vectorising with vmap"
keyIdeas: ["`jax.vmap(f)` turns a function written for one example into one that works on a batch, by pushing the batch dimension through every primitive. The result is the same vectorised code you would write by hand, not a Python loop.", "`in_axes` says which axis of each argument is mapped (`0`, another integer, or `None` for broadcast), and can be a pytree matching the arguments; `out_axes` places the batch axis in the outputs.", "Nesting `vmap` maps over several axes, for example `vmap(vmap(dist, (None, 0)), (0, None))` for all pairwise distances, without writing a single broadcast by hand.", "`vmap(grad(loss), in_axes=(None, 0, 0))` gives per-example gradients: one gradient pytree per example, each with a leading batch axis. This is what DP-SGD clipping and gradient-noise diagnostics need.", "Under `vmap`, `lax.cond` on a batched predicate executes both branches and selects, and memory grows with the batch: per-example gradients cost one copy of the parameters per example."]
question: "For DP-SGD you compute `jax.vmap(jax.grad(loss), in_axes=(None, 0, 0))(params, X, Y)` on a 350M-parameter model with batch 256 in float32, and the step runs out of memory on an 80 GB GPU even though a normal training step at batch 256 fits easily. Explain the memory difference and give two ways to make it fit while keeping exact per-example clipping."
answer: "A normal step reduces over the batch inside the backward pass, so it only ever holds one gradient pytree (350M floats, about 1.4 GB) plus activations. The vmapped version materialises one gradient per example: 256 x 1.4 GB is about 358 GB, far beyond 80 GB, before counting activations. Clipping only needs each example's norm and the clipped sum, so the full per-example tensor never has to exist at once. First option: split the batch into micro-batches of, say, 16 and loop over them with `lax.scan` (or `lax.map`), computing per-example gradients for 16 examples (about 22 GB), clipping them, and adding their sum into a running accumulator in the scan carry; at the end divide by 256 and add noise. Second option: a two-pass approach, computing per-example gradient norms cheaply (for dense layers the norm of an outer-product gradient factorises into norms of activations and output gradients, the 'ghost clipping' trick) and then running one ordinary backward pass on a loss reweighted by each example's clip factor, which gives the clipped sum with a single gradient-sized buffer. Reducing precision or using `jax.checkpoint` addresses activations, not the per-example gradient buffer, so they do not fix this on their own."
sources: [{"title": "JAX docs: Transformations: grad and vmap", "url": "https://docs.jax.dev/en/latest/101/transformations.html", "note": "vmap, in_axes and out_axes, with the manual-batching comparison."}, {"title": "JAX docs: The autodiff cookbook", "url": "https://docs.jax.dev/en/latest/301/cookbook.html", "note": "Per-example gradients by composing vmap and grad, and how jacfwd and jacrev use vmap."}, {"title": "JAX API: jax.vmap", "url": "https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html", "note": "The full signature, including axis_name, axis_size and pytree in_axes."}]
practice: ["per-example-gradient-clipping", "batched-attention-with-vmap", "model-ensemble-with-in-axes", "pairwise-distances-with-nested-vmap"]
tags: ["vmap", "batching", "per-example gradients"]
tools: ["JAX"]
---

## Write it for one, run it for many

Batched code in NumPy or PyTorch is written with the batch dimension in mind from the start: you think about which axis to sum over, insert `[:, None]` for broadcasting, and use `einsum` strings with an extra `b`. JAX lets you write the function for a single example and add the batch afterwards:

```python
def predict(w, b, x):            # x: (d,)  one example
    return jnp.tanh(w @ x + b)

batched = jax.vmap(predict, in_axes=(None, None, 0))
batched(w, b, X)                 # X: (B, d)  ->  (B, k)
```

`vmap` is a transformation like `grad`: it traces `predict` with a batched tracer and rewrites each primitive into its batched form (a matrix-vector product becomes a matrix-matrix product, a sum gains an axis offset, and so on). The output is ordinary vectorised XLA code, as fast as the hand-written version, and it composes with `jit` and `grad` in any order. PyTorch now offers the same idea as `torch.func.vmap`, which was modelled on JAX.

## in_axes and out_axes

`in_axes` has one entry per positional argument. An integer names the axis to map over, and `None` means the argument is shared by every example and broadcast. The entry can itself be a pytree that mirrors the argument, so you can map over some leaves of a dict and not others. All mapped axes must have the same size. `out_axes` does the same for outputs; the default `0` puts the batch axis first.

| call | meaning
| --- | --- |
| `vmap(f)` | map axis 0 of every argument and output
| `vmap(f, in_axes=(None, 0))` | first argument shared, second batched along axis 0
| `vmap(f, in_axes=1, out_axes=1)` | batch lives on axis 1 on both sides, e.g. time-major data
| `vmap(f, in_axes=({"w": 0, "b": None}, 0))` | map only some leaves of a dict argument
| `vmap(f, axis_name="batch")` | name the axis so collectives like `lax.psum(x, "batch")` can reduce over it

`vmap` maps over **positional** arguments; keyword arguments are always mapped along axis 0, so it is clearest to pass everything positionally and control it with `in_axes`.

## Nested vmap

Each `vmap` adds one batch dimension, so nesting maps over several. Pairwise squared distances between point sets $A$ ($n \times d$) and $B$ ($m \times d$):

```python
def sqdist(a, b):                          # two vectors -> scalar
    return jnp.sum((a - b) ** 2)

pairwise = jax.vmap(jax.vmap(sqdist, in_axes=(None, 0)), in_axes=(0, None))
D = pairwise(A, B)                         # (n, m)
nearest = jnp.argmin(D, axis=1)
```

The inner `vmap` maps one point of $A$ against every row of $B$; the outer maps that over every row of $A$. The same pattern gives batched attention: write single-head, single-sequence attention over `(T, d)` arrays, then `vmap` over heads and again over the batch, with key-padding masks passed as another mapped argument.

~~~dg-flow Building a batched attention function from the single-example version.
attend(q, k, v, mask) -(vmap over heads)-> multi-head -(vmap over batch)-> *batched attention -(jit)-> fused kernels
~~~

## Per-example gradients

Ordinary training averages gradients over the batch before you ever see them. Some methods need each example's gradient separately: differentially private SGD clips each one to a norm bound before averaging, and gradient-noise or influence analyses look at their spread. In PyTorch this takes hooks or `torch.func`; in JAX it is one line, because a loss written for a single example can be differentiated and then batched:

```python
def example_loss(params, x, y):          # one example
    return cross_entropy(model_apply(params, x[None]), y[None])

per_ex_grads = jax.vmap(jax.grad(example_loss), in_axes=(None, 0, 0))
g = per_ex_grads(params, X, Y)           # each leaf: (B, *leaf.shape)
```

To clip each example to norm $C$, compute each example's norm across all leaves, scale, then average:

```python
sq = sum(jnp.sum(l.reshape(l.shape[0], -1) ** 2, axis=1) for l in jax.tree.leaves(g))
scale = jnp.minimum(1.0, C / (jnp.sqrt(sq) + 1e-12))           # (B,)
clipped = jax.tree.map(lambda l: jnp.mean(l * scale.reshape(-1, *[1] * (l.ndim - 1)), 0), g)
```

The cost is memory: `g` holds $B$ full copies of the parameter gradients. For large models, compute per-example gradients in micro-batches with `lax.map` or `lax.scan` and accumulate the clipped sum.

## Ensembles and other uses of in_axes

`in_axes` is not only for data. Stack the parameters of $E$ independently initialised models along a new leading axis, `stacked = jax.tree.map(lambda *ps: jnp.stack(ps), *param_list)`, and `jax.vmap(model_apply, in_axes=(0, None))(stacked, x)` evaluates all members on the same batch in one call, returning `(E, B, ...)`. With `in_axes=(0, 0)` each member gets its own batch, which is how bootstrapped ensembles are trained. Initialising the members is itself a `vmap` over a batch of random keys (lesson 6).

## What vmap changes semantically

Most code behaves exactly as if you looped. Two exceptions are worth knowing. A `lax.cond` whose predicate is batched cannot pick one branch for the whole batch, so `vmap` converts it into a select that evaluates both branches; expensive or unsafe branches (a `log` of a negative number, say) then run for every example, so guard their inputs. And a `lax.while_loop` with a batched condition keeps running until every example is done, carrying finished ones along unchanged.

<div class="ex">
<div class="lab">Worked example</div>

**an ensemble of five MLPs in one call**

Initialise five parameter sets with `jax.vmap(init_params)(jax.random.split(key, 5))`, which returns a single pytree whose every leaf has a leading axis of 5.

Training uses `jax.vmap(train_step, in_axes=(0, 0, None))(params, opt_state, batch)`: every member sees the same batch but has its own parameters and optimiser state. One compiled step updates all five, and on a GPU it runs at close to the speed of one larger model because the matmuls are batched.

At evaluation, `preds = jax.vmap(apply, in_axes=(0, None))(params, x)` has shape (5, B, classes); the mean over axis 0 is the ensemble prediction and the variance over axis 0 a cheap uncertainty estimate.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

write functions for one example and let `vmap` add batch axes. Use `in_axes` to say what is shared and what is mapped, nest it for multiple axes, and combine it with `grad` for per-example gradients, keeping an eye on the memory that a batch of gradients costs.

</div>
