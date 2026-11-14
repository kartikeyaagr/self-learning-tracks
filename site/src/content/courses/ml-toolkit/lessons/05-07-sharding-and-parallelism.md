---
id: "05.07"
course: "ml-toolkit"
module: "05"
slug: "sharding-and-parallelism"
title: "Sharding and parallelism"
keyIdeas: ["JAX scales out with one program and many devices (SPMD). A **mesh** arranges the devices in a named grid, a `PartitionSpec` says which mesh axis splits each array dimension, and `NamedSharding(mesh, spec)` combines the two into a layout for a `jax.Array`.", "With `jax.jit`, computation follows data: give the inputs shardings and the compiler propagates layouts through the program and inserts the collectives (all-reduce, all-gather, reduce-scatter) that the math requires.", "Data parallelism shards the batch and replicates parameters; FSDP also shards parameters and optimiser state over the data axis; tensor (model) parallelism shards the weight matrices themselves, as in the Megatron MLP.", "`jax.shard_map` drops to a per-device view: you write code for one shard and call collectives such as `lax.psum`, `lax.all_gather` and `lax.ppermute` yourself, which is the tool for hand-tuned communication.", "`jax.pmap` is the legacy API (one device per index of a leading axis); new code should use `jit` with shardings or `shard_map`."]
question: "You have 16 GPUs as 2 nodes of 8 with fast NVLink inside a node and slower networking between nodes. You want tensor parallelism plus data parallelism for a model whose layers are too large for one GPU's memory with optimiser state. Propose a mesh shape and axis order, the PartitionSpecs for a dense layer's weight `(d_in, d_out)`, the activations `(B, T, d)` and the batch, and say which collectives cross nodes."
answer: "Use a mesh of shape `(2, 8)` with axes `('data', 'model')`, ordered so that the 8 devices of one node form the `'model'` axis; `jax.make_mesh` tries to match axes to the physical topology, but you should check that each `'model'` group is one node by inspecting `mesh.devices`. Tensor parallelism communicates inside every layer, so it belongs on the fast NVLink axis, while data parallelism communicates once per step for gradients, so it can cross the slower inter-node link. For a column-parallel layer the weight is `P(None, 'model')` and for the following row-parallel layer `P('model', None)`; to also reduce optimiser-state memory, add FSDP on the other dimension, for example `P('data', 'model')`, at the cost of all-gathers across nodes. The batch is `P('data', None)` for token ids `(B, T)`, and activations `(B, T, d)` are `P('data', None, None)` between blocks and `P('data', None, 'model')` inside the MLP. Communication over `'model'` (the all-reduce after each row-parallel matmul) stays inside a node. Only the gradient all-reduce over `'data'` (or, with FSDP, the weight all-gathers and gradient reduce-scatters) crosses nodes, and XLA can overlap it with the backward pass."
sources: [{"title": "JAX docs: Distributed arrays and automatic parallelization", "url": "https://docs.jax.dev/en/latest/parallel.html", "note": "Meshes, NamedSharding, automatic partitioning with jit and an introduction to shard_map."}, {"title": "JAX docs: Manual parallelism with shard_map", "url": "https://docs.jax.dev/en/latest/201/shard-map.html", "note": "Per-device programs, collectives and worked tensor-parallel examples."}, {"title": "How to Scale Your Model (the scaling book)", "url": "https://jax-ml.github.io/scaling-book/", "note": "Cost models for data, FSDP and tensor parallelism and the collectives each one needs."}]
practice: ["device-mesh-and-named-sharding", "data-parallel-step-with-psum", "tensor-parallel-mlp", "ring-all-reduce-with-ppermute"]
tags: ["sharding", "parallelism", "shard_map"]
tools: ["JAX", "XLA"]
---

## From one device to a mesh

Everything so far ran on one device. Lesson 4's `vmap` adds a batch axis; this lesson splits arrays across devices. PyTorch typically does this by launching one process per GPU and wrapping the model in DDP or FSDP. JAX instead treats the whole set of devices as a single machine: you write one program over global arrays, attach a sharding to them, and the compiler produces the per-device program. On multi-host clusters each host runs the same script after `jax.distributed.initialize()`, and `jax.devices()` lists every device in the job.

A mesh gives the devices a logical shape with axis names:

```python
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, AxisType

mesh = jax.make_mesh((4, 2), ("data", "model"),
                     axis_types=(AxisType.Auto, AxisType.Auto))
x = jax.device_put(jnp.ones((64, 512)), NamedSharding(mesh, P("data", None)))
x.addressable_shards[0].data.shape     # (16, 512): rows split 4 ways, replicated over "model"
jax.debug.visualize_array_sharding(x)  # prints which device holds which block
```

`jax.make_mesh` picks a device order that matches the hardware topology, so the axis you communicate over most should be the one with the fastest links. Current JAX also has an *explicit* sharding mode, where shardings are part of an array's type and ambiguous layouts are errors instead of compiler decisions, and `make_mesh` now makes every axis `AxisType.Explicit` by default; the `axis_types` argument above requests the automatic mode this lesson describes (older releases had only automatic mode and no `axis_types` argument). To experiment without accelerators, set `XLA_FLAGS=--xla_force_host_platform_device_count=8` before importing JAX to get eight CPU devices.

## PartitionSpec

A `PartitionSpec` has one entry per array dimension: a mesh axis name to split that dimension over, a tuple of names to split over several axes, or `None` to keep the dimension whole. Mesh axes not mentioned mean replication over them. For an array of shape `(B, D)` on the 4 x 2 mesh:

| spec | each device holds | used for
| --- | --- | --- |
| `P()` | the whole array | replicated parameters in plain data parallelism
| `P("data", None)` | `(B/4, D)`, same on both "model" devices | a batch of activations
| `P(None, "model")` | `(B, D/2)` | column-parallel weights
| `P("data", "model")` | `(B/4, D/2)` | 2-D sharded activations
| `P(("data", "model"), None)` | `(B/8, D)` | batch over all eight devices

Each split dimension must be divisible by the product of the mesh axes that split it.

## Automatic partitioning with jit

Pass sharded arrays to a jitted function and XLA's partitioner works out the rest:

```python
w1 = jax.device_put(w1, NamedSharding(mesh, P(None, "model")))   # (512, 2048)
w2 = jax.device_put(w2, NamedSharding(mesh, P("model", None)))   # (2048, 512)

@jax.jit
def mlp(x, w1, w2):
    h = jax.nn.relu(x @ w1)      # (B, 2048) sharded P("data", "model"): no communication
    return h @ w2                # contracts the sharded axis: XLA inserts an all-reduce
```

This is Megatron-style tensor parallelism: the first matmul is split by columns, the second by rows, and the only communication is one all-reduce over `"model"` at the end of the block. You can pin layouts with `jax.jit(..., in_shardings=..., out_shardings=...)` and constrain an intermediate with `jax.lax.with_sharding_constraint(h, sharding)` when the compiler's choice is not the one you want.

~~~dg-flow Tensor-parallel MLP on the "model" axis: where data moves.
x replicated over model -(x @ w1 column shards)-> h split by columns -(relu, local)-> h @ w2 row shards -> *partial sums -(all-reduce over model)-> y
~~~

## The parallelism strategies

- **Data parallel**: batch `P("data")`, parameters `P()`. Each device computes gradients on its slice and the gradient of the mean loss needs an all-reduce, which the compiler inserts. Memory per device is the full model plus optimiser state.
- **FSDP / ZeRO-3**: parameters and optimiser state also sharded over `"data"`, for example `P("data", None)` on each weight. The compiler all-gathers weights just before they are used and reduce-scatters gradients, so per-device state memory drops by the axis size.
- **Tensor parallel**: weights sharded over `"model"` as above, usually within a node where links are fastest, with activations communicated inside each layer.
- **Combinations**: a 2-D mesh with `("data", "model")` axes runs FSDP or data parallelism across one axis and tensor parallelism across the other. Pipeline parallelism splits layers into stages and is built by hand or by libraries on top of these primitives.

## shard_map: write the per-device program

Sometimes you want control over exactly which collectives run, for instance to overlap communication with compute or implement a ring algorithm. `jax.shard_map` (formerly `jax.experimental.shard_map.shard_map`) maps a function over the shards: inside, arrays have their local block shapes and you call collectives by mesh axis name.

```python
def mlp_local(x, w1, w2):                 # x: (B/4, 512), w1: (512, 1024), w2: (1024, 512)
    h = jax.nn.relu(x @ w1)
    return jax.lax.psum(h @ w2, "model")  # sum the partial products over "model"

mlp_tp = jax.jit(jax.shard_map(
    mlp_local, mesh=mesh,
    in_specs=(P("data", None), P(None, "model"), P("model", None)),
    out_specs=P("data", None)))
```

The collectives available inside are `psum`, `pmean`, `all_gather`, `psum_scatter` (reduce-scatter), `all_to_all`, and `ppermute`, which sends each shard to another device along a permutation and is the building block of ring all-reduce and ring attention. If you differentiate code inside `shard_map`, check against an unsharded reference: recent versions track which values are replicated and insert the gradient reduction for replicated parameters themselves, so adding your own `psum` on top would multiply gradients by the axis size.

## pmap is legacy

`jax.pmap` was JAX's original multi-device API: it maps a function over the leading axis of its inputs, one index per device, with collectives named by `axis_name`. It ties array layout to that leading axis, compiles per-device programs separately, and composes poorly with sharded `jit` code. The JAX team now treats it as legacy; use `jit` with `NamedSharding` for most work and `shard_map` where you need explicit per-device code. Old `pmap` code usually translates directly to `shard_map` with a one-dimensional mesh.

<div class="ex">
<div class="lab">Worked example</div>

**why a 7B model does not fit with plain data parallelism**

Training a 7-billion-parameter model with Adam on 8 GPUs of 80 GB: bfloat16 parameters take 14 GB, float32 master weights 28 GB and the two Adam moments 56 GB, about 98 GB per replica before activations. With parameters replicated (`P()`), every GPU needs all of it, so it does not fit.

Shard parameters and optimiser state over an 8-way `"data"` axis (FSDP) and each device holds about 12 GB of state, leaving most of the memory for activations. The cost is an all-gather of each layer's weights in the forward and backward passes and a reduce-scatter of gradients, which the compiler can overlap with compute.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

describe where data lives with a mesh and PartitionSpecs, let `jit` derive the communication for data, FSDP and tensor parallelism, use `shard_map` when you need to write the collectives yourself, and leave `pmap` to old code.

</div>
