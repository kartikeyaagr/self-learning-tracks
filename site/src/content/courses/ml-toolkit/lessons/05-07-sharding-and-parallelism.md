---
id: "05.07"
course: "ml-toolkit"
module: "05"
slug: "sharding-and-parallelism"
title: "Sharding and parallelism"
keyIdeas: ["JAX scales out with one program and many devices (SPMD). A **mesh** arranges the devices in a named grid, a `PartitionSpec` says which mesh axis splits each array dimension, and `NamedSharding(mesh, spec)` combines the two into a layout for a `jax.Array`.", "With `jax.jit`, computation follows data: give the inputs shardings and the compiler propagates layouts through the program and inserts the collectives (all-reduce, all-gather, reduce-scatter) that the maths requires. Your model code does not change.", "Data parallelism shards the batch and replicates parameters; FSDP also shards parameters and optimiser state over the data axis, cutting state memory by the axis size; tensor (model) parallelism shards the weight matrices themselves, as in the Megatron MLP with one all-reduce per block.", "`jax.shard_map` drops to a per-device view: you write code for one shard and call collectives such as `lax.psum`, `lax.all_gather` and `lax.ppermute` yourself. Differentiate a `pmean` of the local loss, and check against a one-device run, or gradients can come out N times too large.", "Simulate devices on a CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=8` to test layouts and correctness; `jax.pmap` is the legacy API, and new code should use `jit` with shardings or `shard_map`."]
question: "You have 16 GPUs as 2 nodes of 8 with fast NVLink inside a node and slower networking between nodes. You want tensor parallelism plus data parallelism for a model whose layers are too large for one GPU's memory with optimiser state. Propose a mesh shape and axis order, the PartitionSpecs for a dense layer's weight `(d_in, d_out)`, the activations `(B, T, d)` and the batch, and say which collectives cross nodes."
answer: "Use a mesh of shape `(2, 8)` with axes `('data', 'model')`, ordered so that the 8 devices of one node form the `'model'` axis; `jax.make_mesh` tries to match axes to the physical topology, but you should check that each `'model'` group is one node by inspecting `mesh.devices`. Tensor parallelism communicates inside every layer, so it belongs on the fast NVLink axis, while data parallelism communicates once per step for gradients, so it can cross the slower inter-node link. For a column-parallel layer the weight is `P(None, 'model')` and for the following row-parallel layer `P('model', None)`; to also reduce optimiser-state memory, add FSDP on the other dimension, for example `P('data', 'model')`, at the cost of all-gathers across nodes. The batch is `P('data', None)` for token ids `(B, T)`, and activations `(B, T, d)` are `P('data', None, None)` between blocks and `P('data', None, 'model')` inside the MLP. Communication over `'model'` (the all-reduce after each row-parallel matmul, in the forward and backward passes) stays inside a node. Only the gradient all-reduce over `'data'` (or, with FSDP, the weight all-gathers and gradient reduce-scatters, about 1.5 times the bytes of a plain all-reduce) crosses nodes, and XLA can overlap it with the backward pass. Before scaling up, verify on a small run that the sharded loss and gradients match an unsharded reference and inspect the compiled program for unexpected all-gathers of activations."
sources: [{"title": "JAX docs: Distributed arrays and automatic parallelization", "url": "https://docs.jax.dev/en/latest/parallel.html", "note": "Meshes, NamedSharding, automatic partitioning with jit and an introduction to shard_map."}, {"title": "JAX docs: Manual parallelism with shard_map", "url": "https://docs.jax.dev/en/latest/201/shard-map.html", "note": "Per-device programs, collectives and worked tensor-parallel examples."}, {"title": "How to Scale Your Model (the scaling book)", "url": "https://jax-ml.github.io/scaling-book/", "note": "Cost models for data, FSDP and tensor parallelism and the collectives each one needs."}]
practice: ["device-mesh-and-named-sharding", "data-parallel-step-with-psum", "tensor-parallel-mlp", "ring-all-reduce-with-ppermute"]
tags: ["sharding", "parallelism", "shard_map"]
tools: ["JAX", "XLA"]
---

## What you will learn

- How JAX runs one program on many devices: a **mesh**, a `PartitionSpec` per array, and `jit` doing the rest.
- How to read common layouts and predict each device's block shape.
- The four collectives that move data between devices (all-reduce, all-gather, reduce-scatter, permute), with real outputs.
- How data parallelism, FSDP and tensor parallelism differ in memory and communication.
- When to write the per-device program yourself with `jax.shard_map`, and why `pmap` is legacy.

*Before you start:* you should be comfortable with `jax.jit` (05.03), `vmap` and batch axes (05.04), and pytrees of parameters with a training step (05.05). No GPU is needed: every snippet runs on a CPU pretending to have eight devices.

So far in this module every array lived on one device. Lesson 05.04's `vmap` added a batch axis inside one program; this lesson splits arrays across several devices, so a model and its data can outgrow one accelerator. PyTorch usually starts one process per GPU and wraps the model in DDP or FSDP. JAX instead treats all devices as one machine: you write one program over **global** arrays, attach a layout, and the compiler writes the per-device program. This style is called **SPMD** (single program, multiple data).

## The idea in one picture

Take eight devices and arrange them in a 4 x 2 grid. The first grid axis is called `"data"`, the second `"model"`. Now place an 8 x 4 array on that grid in four different ways:

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"   # before importing jax
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, AxisType

mesh = jax.make_mesh((4, 2), ("data", "model"),
                     axis_types=(AxisType.Auto, AxisType.Auto))
for spec in [P(), P("data", None), P(None, "model"), P("data", "model")]:
    a = jax.device_put(jnp.zeros((8, 4)), NamedSharding(mesh, spec))
    print(spec, a.addressable_shards[0].data.shape)
# P()                 (8, 4)   every device holds everything
# P('data', None)     (2, 4)   rows split 4 ways
# P(None, 'model')    (8, 2)   columns split 2 ways
# P('data', 'model')  (2, 2)   both split: 8 distinct blocks
```

~~~dg-fig 05.07-specs.svg
The same 8 x 4 array on a 4 x 2 mesh under four PartitionSpecs. Each outlined block is one piece of the array; the labels say which devices hold it. A mesh axis that the spec does not mention means "copy along that axis", which is why rows split over "data" are held by two devices each.
~~~

Three new words carry the whole lesson:

- A **mesh** is the devices arranged in a grid with named axes, here shape `(4, 2)` with names `("data", "model")`.
- A **PartitionSpec** (`P`) has one entry per array dimension saying which mesh axis splits it, or `None` to keep that dimension whole.
- A **sharding** (`NamedSharding(mesh, spec)`) joins the two: where every element lives. Each per-device piece is a **shard**.

The second panel reads: "split rows over `data`, keep columns whole, and copy each block along `model`, which is not mentioned". Your program still sees one `(8, 4)` array; only storage is spread out.

## Step 1: build a mesh

`jax.make_mesh(shape, names)` takes the device count you want, in grid form, and the axis names. The product of the shape must equal the number of devices you use (here $4 \times 2 = 8$).

Device order matters on real hardware. `make_mesh` follows the physical topology so that devices along the last axis tend to share the fastest links (say, the GPUs of one node joined by NVLink): put the axis you communicate over most often last. On simulated CPU devices, `mesh.devices` is the plain grid `[[0 1] [2 3] [4 5] [6 7]]`.

Without accelerators, set `XLA_FLAGS=--xla_force_host_platform_device_count=8` *before* importing JAX: you get eight CPU devices that behave like real ones for layout and correctness (not speed). Recent JAX also has an **explicit** sharding mode, where shardings are part of an array's type and ambiguous layouts are errors; `make_mesh` now makes explicit axes by default. `axis_types=(AxisType.Auto, ...)` requests the **automatic** mode this lesson uses, where the compiler fills in unspecified layouts (older releases had no `axis_types` argument). On many hosts, each runs the same script after `jax.distributed.initialize()`, and `jax.devices()` lists every device in the job.

## Step 2: describe layouts with PartitionSpec

A spec entry can be a mesh axis name, a tuple of names (split over several axes at once, in order), or `None`. For an array of shape $(B, D)$ on the 4 x 2 mesh:

| Spec | Each device holds | Typical use
| --- | --- | --- |
| `P()` | the whole $(B, D)$ | replicated parameters in plain data parallelism
| `P("data", None)` | $(B/4, D)$, same block on both "model" devices | a batch of activations
| `P(None, "model")` | $(B, D/2)$ | column-parallel weights
| `P("data", "model")` | $(B/4, D/2)$ | 2-D sharded activations or FSDP + tensor-parallel weights
| `P(("data", "model"), None)` | $(B/8, D)$ | a batch split over all eight devices

The one hard rule: each split dimension must be divisible by the product of the mesh axis sizes that split it. With $B = 64$, `P("data", None)` gives each device $64 / 4 = 16$ rows. With $B = 6$ it fails, because 6 rows cannot be cut into 4 equal blocks:

```python
x = jax.device_put(jnp.ones((64, 512)), NamedSharding(mesh, P("data", None)))
print(x.addressable_shards[0].data.shape)          # (16, 512)
print(len(x.addressable_shards))                   # 8  (one shard object per device)
jax.device_put(jnp.ones((6, 512)), NamedSharding(mesh, P("data", None)))
# IndivisibleError: ... implies that array axis 0 is partitioned 4 times ...
```

To see who holds what, inspect `x.addressable_shards` (each has `.device`, `.index` and `.data`) or call `jax.debug.visualize_array_sharding(x)` (needs the `rich` package).

## Step 3: let jit follow the data

Here is the central trick: you do not rewrite your model. You give its inputs shardings and call the same `jax.jit` function; XLA's partitioner propagates layouts through every operation and inserts communication only where the maths requires it ("computation follows data"). Take a two-layer MLP block with weights split as in Megatron-LM: the first by columns, the second by rows.

```python
k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
w1 = jax.device_put(jax.random.normal(k1, (512, 2048)) * 0.02, NamedSharding(mesh, P(None, "model")))
w2 = jax.device_put(jax.random.normal(k2, (2048, 512)) * 0.02, NamedSharding(mesh, P("model", None)))
x  = jax.device_put(jax.random.normal(k3, (64, 512)),          NamedSharding(mesh, P("data", None)))

@jax.jit
def mlp(x, w1, w2):
    h = jax.nn.relu(x @ w1)      # (64, 2048) laid out P("data", "model"): no communication
    return h @ w2                # sums over the split axis: XLA adds an all-reduce

y = mlp(x, w1, w2)
print(y.shape, y.sharding.spec)  # (64, 512) P('data', None)
hlo = mlp.lower(x, w1, w2).compile().as_text()
print("all-reduce" in hlo)       # True: exactly one all-reduce in the compiled program
```

Against a one-device run the largest difference was $1.2 \times 10^{-7}$, plain float32 rounding. The figure shows why the communication lands where it does.

~~~dg-fig 05.07-megatron.svg
Tensor-parallel MLP over a 2-way "model" axis. Splitting w1 by columns means each device computes its own half of the hidden units from the full x, so nothing moves. Splitting w2 by rows means each device multiplies its half of h by its half of w2 and gets a partial output of full size. The true output is the sum of the partials, which is one all-reduce.
~~~

The first matmul needs every input feature but only some output columns, so each device has what it needs; ReLU is elementwise and stays local. The second matmul sums over the hidden dimension, which is split: device 0 adds up its 1024 terms, device 1 the other 1024. Adding the two partials, one all-reduce per block, is the only exchange.

When the compiler's choice is not what you want, you can pin layouts: `jax.jit(f, in_shardings=..., out_shardings=...)` fixes inputs and outputs, and `jax.lax.with_sharding_constraint(h, sharding)` inside the function fixes an intermediate.

## Step 4: the four collectives

A **collective** is an operation in which every device on a mesh axis takes part at once. Four of them cover almost everything. The figure shows each one run on four simulated devices, where device $i$ starts with row $i$ of the numbers 0 to 15.

~~~dg-fig 05.07-collectives.svg
Real outputs on four simulated devices. All-reduce (psum) leaves the column sums 24, 28, 32, 36 on every device. All-gather collects one value from each device onto all of them. Reduce-scatter (psum_scatter) computes the same sums but leaves each device only its own quarter. Permute (ppermute) passes each row one step around a ring.
~~~

- **All-reduce** (`lax.psum`, or `lax.pmean` for the average): every device ends with the sum over all devices. Used for gradients in data parallelism and for partial sums in tensor parallelism.
- **All-gather** (`lax.all_gather`): every device ends with all the pieces, concatenated. Used to reassemble a sharded weight just before it is needed.
- **Reduce-scatter** (`lax.psum_scatter`): sum, but each device keeps only its slice of the result. It is exactly half of an all-reduce, since all-reduce = reduce-scatter followed by all-gather.
- **Permute** (`lax.ppermute`): each device sends its block to a chosen partner; the building block of ring algorithms. A fifth, `lax.all_to_all`, changes which dimension is split (mixture-of-experts layers use it).

Inside sharded `jit` the compiler picks these for you, but you will read them in profiles and compiled programs, and they are where multi-device time goes.

## Step 5: the three parallelism strategies

All three strategies are just different PartitionSpecs on the same training step.

- **Data parallelism (DP)**: batch `P("data")`, parameters `P()`. Each device runs the full model on its slice; the mean-loss gradient needs every device's contribution, so the compiler inserts an all-reduce. Every device stores the full model and optimiser state.
- **Fully sharded data parallelism (FSDP, ZeRO-3)**: batch split as in DP, and parameters and optimiser state also split over `"data"`, e.g. `P("data", None)`. Each weight is all-gathered just before use; gradients are reduce-scattered so each device updates only its shard. State memory drops by the axis size.
- **Tensor (model) parallelism (TP)**: the weight matrices are split over `"model"`, as above. It communicates inside every layer, so it belongs on the fastest links, usually within one node.
- **Combinations**: a 2-D mesh `("data", "model")` runs DP or FSDP on one axis and TP on the other. **Pipeline parallelism** (layers on different devices, micro-batches streamed through) is built on top of these primitives.

To check this with real compiled programs, the same two-layer MLP training step (weights 256 x 1024 and 1024 x 256 in float32) was compiled for 8 simulated devices under each strategy:

~~~dg-fig 05.07-measured.svg
Measured on 8 simulated CPU devices: parameter bytes held by one device, and the collectives XLA placed in one full training step (forward, backward, update). All three runs gave the identical loss, 1.8617828. Counts come from the CPU compiler and may differ on GPU or TPU, where XLA tends to use reduce-scatter for FSDP gradients.
~~~

DP needed one all-reduce because XLA merged both gradients into one message. DP and FSDP produced bit-identical updated weights: sharding changes where numbers live, not what they are. TP needed nothing in the backward pass because the replicated input's gradient was never required.

<div class="ex">
<div class="lab">Worked example</div>

**averaging gradients across four devices by hand**

A linear model $\hat y = x w$ with $w = (0.5, 0.5)$ and loss $L = \frac{1}{n}\sum (x_i w - y_i)^2$. Eight examples are split two per device over a 4-way `"data"` axis. The gradient on one device is $\frac{2}{n_{\text{local}}} X^\top (Xw - y)$.

Device 0 holds rows $x = (1, 0), (0, 1)$ with $y = (1, 2)$. Predictions are $(0.5, 0.5)$, errors $(-0.5, -1.5)$. Gradient $= \frac{2}{2}\big(1 \cdot (-0.5) + 0 \cdot (-1.5),\; 0 \cdot (-0.5) + 1 \cdot (-1.5)\big) = (-0.5, -1.5)$.

The other devices compute in the same way: device 1 gets $(-7, -4.5)$, device 2 $(-17, -7)$, device 3 $(-12, -28.5)$.

An all-reduce with sum gives $(-0.5 - 7 - 17 - 12,\; -1.5 - 4.5 - 7 - 28.5) = (-36.5, -41.5)$ on every device.

Each device saw the same number of examples, so the gradient of the global mean loss is the average: $(-36.5/4,\; -41.5/4) = (-9.125, -10.375)$. This matches `jax.grad` on all eight rows on one device exactly. Forget the division by 4 and every step is four times too large, as if the learning rate had been multiplied by the number of devices.

</div>

## Step 6: shard_map, the per-device view

Sometimes you want to choose the collectives yourself: to overlap communication with compute, to write a ring algorithm, or because the compiler chose badly. `jax.shard_map` (formerly `jax.experimental.shard_map.shard_map`) maps a function over the shards. Inside, each array has its *local* block shape, and you call collectives by mesh axis name.

```python
def mlp_local(x, w1, w2):                 # x: (16, 512), w1: (512, 1024), w2: (1024, 512)
    h = jax.nn.relu(x @ w1)
    return jax.lax.psum(h @ w2, "model")  # sum the partial products over "model"

mlp_tp = jax.jit(jax.shard_map(
    mlp_local, mesh=mesh,
    in_specs=(P("data", None), P(None, "model"), P("model", None)),
    out_specs=P("data", None)))
print(jnp.abs(mlp_tp(x, w1, w2) - mlp(x, w1, w2)).max())   # 0.0
```

`in_specs` say how to cut global inputs into local blocks; `out_specs` how to glue local results back. The result matches exactly: you wrote by hand the all-reduce the compiler inserted before.

The data-parallel step from the worked example, written with `shard_map`, shows the one trap worth memorising:

```python
def loss(w, x, y): return jnp.mean((x @ w - y) ** 2)
mesh4 = jax.make_mesh((4,), ("data",), axis_types=(AxisType.Auto,), devices=jax.devices()[:4])
spec = dict(mesh=mesh4, in_specs=(P(), P("data"), P("data")), out_specs=P())

good = lambda w, xl, yl: jax.grad(lambda w: jax.lax.pmean(loss(w, xl, yl), "data"))(w)
bad  = lambda w, xl, yl: jax.lax.pmean(jax.grad(loss)(w, xl, yl), "data")
print(jax.jit(jax.shard_map(good, **spec))(w, x8, y8))   # [ -9.125 -10.375]  correct
print(jax.jit(jax.shard_map(bad,  **spec))(w, x8, y8))   # [-36.5   -41.5  ]  4x too big
```

Current JAX tracks which values inside `shard_map` are identical on every device. Because `w` came in replicated (`P()`), its gradient is already summed over the axis: the "bad" version gets the sum $(-36.5, -41.5)$ from the example, and `pmean` of an already-identical value changes nothing. Differentiate the global mean (`pmean` of the local loss) instead, and always compare with an unsharded reference.

## pmap is legacy

`jax.pmap`, JAX's first multi-device API, maps a function over the leading axis of its inputs, one index per device. It ties layout to that axis, compiles per-device programs separately and mixes poorly with sharded `jit`; the JAX team treats it as legacy. Use `jit` with `NamedSharding`, or `shard_map` for per-device code; old `pmap` code usually becomes `shard_map` over a 1-D mesh.

## Checking that it worked

- **Compare with one device.** Outputs and gradients should differ only by rounding ($10^{-7}$ in float32); a factor equal to an axis size means a doubled or missing reduction.
- **Look at the layouts.** Print `y.sharding.spec` and a shard's shape. `P()` where you expected `P("data")` means something forced full replication.
- **Count the collectives.** Search `f.lower(*args).compile().as_text()` for `all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all`, `collective-permute`. An unexpected all-gather of a large activation is the most common hidden cost.
- **Profile on real hardware.** Simulated devices check correctness, not speed; on GPUs or TPUs the JAX profiler shows whether communication overlaps compute.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| `IndivisibleError` when placing an array | a split dimension is not a multiple of the mesh axis size | pad the batch or choose a mesh shape that divides it
| Loss trains, but gradients are N times too large | an extra `psum` or a missing average inside `shard_map` | differentiate a `pmean` of the local loss; compare with a one-device run
| Memory per device no lower than with one device | weights left as `P()`, or the compiler gathered them and kept them | shard parameters and optimiser state (FSDP), check output shardings
| Multi-device run slower than one device | tensor parallelism across slow links, or huge all-gathers of activations | put the TP axis on the fast links; pin intermediates with `with_sharding_constraint`
| Only one CPU device appears | `XLA_FLAGS` set after `import jax` | set the variable before the first import, or in the shell
| Errors about ambiguous or mismatched shardings | mesh created in explicit mode (the new default) | pass `axis_types=(AxisType.Auto, ...)` or follow the explicit-sharding API

<div class="ex">
<div class="lab">Worked example</div>

**why a 7B model does not fit with plain data parallelism**

Training a 7-billion-parameter model with Adam on 8 GPUs of 80 GB each. bfloat16 weights take $7 \times 10^9 \times 2$ bytes $= 14$ GB, float32 master weights $7 \times 10^9 \times 4 = 28$ GB, and Adam's two float32 moments $2 \times 28 = 56$ GB.

Total: $14 + 28 + 56 = 98$ GB per replica, before gradients and activations. With parameters replicated (`P()`), every GPU needs all 98 GB, which is more than 80, so it does not fit.

Shard only the optimiser state (master weights and moments) over 8 devices: $14 + 84/8 = 14 + 10.5 = 24.5$ GB. Shard everything (FSDP over an 8-way `"data"` axis): $98 / 8 = 12.25$ GB, leaving most of each GPU for activations.

The cost is communication: an all-gather of each layer's weights in the forward and again in the backward pass, plus a reduce-scatter of gradients, which the compiler can overlap with compute.

</div>

~~~dg-fig 05.07-memory.svg
Model-state memory per GPU for the 7B example (arithmetic, not a measurement). Plain data parallelism needs 98 GB and crosses the 80 GB line; sharding only optimiser state brings it to 24.5 GB; sharding everything to 12.25 GB.
~~~

## Going deeper: what the communication costs

Collectives are usually rings. Put $N$ devices in a circle, each with $S$ bytes to sum, cut into $N$ chunks. In the **reduce-scatter** phase each device passes a chunk to its neighbour, which adds its own copy and passes the running sum on; after $N - 1$ steps each device owns the full sum of one chunk. In the **all-gather** phase the finished chunks circle for another $N - 1$ steps. Each step moves $S/N$ bytes per device, so:

$$\text{reduce-scatter} = \text{all-gather} = \frac{N-1}{N}\,S, \qquad \text{all-reduce} = 2\,\frac{N-1}{N}\,S \text{ bytes sent per device.}$$

Here $S$ is the full array size and $N$ the axis size. The cost per device barely grows with $N$: about $2S$ for an all-reduce with 8 devices or 1024. With `ppermute` and a shift-by-one permutation you can write this algorithm in `shard_map`; it is also the basis of ring attention, where key and value blocks circulate while each device attends to the block it holds.

Using these formulas for one training step with $S$ bytes of gradients:

- **DP** sends one all-reduce: $2 \frac{N-1}{N} S$.
- **FSDP** sends two all-gathers of weights (forward and backward) and one reduce-scatter of gradients: $3 \frac{N-1}{N} S$, about 1.5 times DP, in exchange for an $N$-fold cut in state memory.
- **TP** sends activations, not weights: roughly one all-reduce of a $(B, T, d)$ activation per block in the forward pass and one in the backward pass. That scales with batch size and is paid in every layer, which is why TP stays inside a node.

Whether communication hurts depends on overlap. A matmul takes time proportional to FLOPs; a collective, to bytes divided by link bandwidth. If the per-device batch is large enough that compute outlasts communication, XLA can hide layer $\ell + 1$'s all-gather behind layer $\ell$'s compute and scaling is nearly free; shrink the batch too far and the step becomes communication-bound. "How to Scale Your Model" works these ratios out for real hardware.

~~~dg-flow Choosing a strategy for a model that does not fit
Model + optimiser fit on one device? -(yes)-> Data parallel -(still too big)-> FSDP over the data axis -(single layers too big or batch too small)-> *Add tensor parallel on the fast axis
~~~

## Key terms

| Term | Meaning
| --- | --- |
| SPMD | One program, compiled into per-device copies that each work on their own part of the data
| Mesh | The devices arranged in a grid with named axes, such as `("data", "model")`
| PartitionSpec | One entry per array dimension naming the mesh axis that splits it, or `None`
| NamedSharding | A mesh plus a PartitionSpec: the full layout of an array across devices
| Shard | The block of a global array stored on one device
| Collective | A communication step that all devices on an axis join: all-reduce, all-gather, reduce-scatter, permute
| FSDP | Data parallelism that also shards parameters and optimiser state, gathering weights only when used
| Tensor parallelism | Splitting each weight matrix across devices, with an all-reduce inside every block
| shard_map | Write the per-device program yourself, with local shapes and explicit collectives

<div class="take">
<div class="lab">Key takeaway</div>

describe where data lives with a mesh and PartitionSpecs, let `jit` derive the communication for data, FSDP and tensor parallelism, use `shard_map` when you need to write the collectives yourself, check every sharded result against a one-device run, and leave `pmap` to old code.

</div>
