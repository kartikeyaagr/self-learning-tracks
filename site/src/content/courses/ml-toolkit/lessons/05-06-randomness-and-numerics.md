---
id: "05.06"
course: "ml-toolkit"
module: "05"
slug: "randomness-and-numerics"
title: "Randomness and numerics"
keyIdeas: ["JAX has no global random state. Every random function takes an explicit **key**, the same key always gives the same numbers, and you make new keys with `jax.random.split` (or `fold_in`) rather than reusing one.", "`jax.random.key(seed)` returns a typed key array (dtype `key<fry>`); the older `jax.random.PRNGKey(seed)` returns a raw `uint32[2]` array. Both work with every `jax.random` function, but new code should use typed keys.", "Arrays are float32 by default and float64 requests are silently downcast unless `jax_enable_x64` is set at startup. On TPUs (and with TF32 on recent NVIDIA GPUs) float32 matmuls may use reduced-precision passes by default; request higher precision when it matters.", "On GPU, JAX preallocates most of device memory at startup. Inside a training step, buffer **donation** (`donate_argnums`) lets XLA write the new parameters and optimiser state into the old ones' memory, roughly halving the step's peak for that state.", "`jax.checkpoint` (remat) trades compute for activation memory by recomputing intermediates in the backward pass, and `jax_debug_nans` finds the first operation that produced a NaN."]
question: "A colleague's data-augmentation function is `vmap`ped over the batch as `jax.vmap(augment, in_axes=(None, 0))(key, images)` and the key is split once per step. They report that augmentations look random across steps but the model ignores them. What is going on, and what is the correct call?"
answer: "With `in_axes=None` the key is broadcast, so every image in the batch receives the same random crop, flip and colour jitter within a step. Across steps the key changes, so the augmentation looks random when you inspect one image over time, but within a batch all examples are transformed identically. That greatly reduces the diversity of augmentations per gradient step (and with batch-level statistics such as batch norm, an identical transform is partly normalised away). The fix is to give each example its own key: `keys = jax.random.split(step_key, images.shape[0])` and `jax.vmap(augment)(keys, images)`, with `in_axes=0` for both. A useful check is to augment a batch made of one image repeated B times and confirm that the outputs differ."
sources: [{"title": "JAX docs: Pseudorandom numbers", "url": "https://docs.jax.dev/en/latest/101/random.html", "note": "Explicit keys, split and fold_in, and why JAX avoids global random state."}, {"title": "JAX docs: Common gotchas in JAX", "url": "https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html", "note": "Random numbers, double (64-bit) precision and debugging NaNs."}, {"title": "JAX docs: GPU memory allocation", "url": "https://docs.jax.dev/en/latest/201/gpu-memory.html", "note": "Preallocation and the XLA_PYTHON_CLIENT_* environment variables."}, {"title": "JAX docs: Just-in-time compilation (buffer donation section)", "url": "https://docs.jax.dev/en/latest/201/jit.html", "note": "How donate_argnums works and when donated buffers cannot be reused."}]
practice: ["prng-key-splitting", "mixed-precision-with-loss-scaling", "rematerialized-mlp-block", "int8-matmul-with-per-channel-scales"]
tags: ["PRNG keys", "precision", "memory"]
tools: ["JAX"]
---

## Why keys are explicit

NumPy and PyTorch keep a hidden global generator that every call advances. That is impure: the result of a function depends on how many random calls happened before it, which breaks `jit` caching, makes `vmap` ambiguous (should every example get the same sample?) and makes parallel execution order-dependent. JAX replaces the hidden state with a value you pass around:

```python
import jax, jax.numpy as jnp

key = jax.random.key(0)                 # typed key, dtype key<fry>
key, k_init, k_drop = jax.random.split(key, 3)
w = jax.random.normal(k_init, (784, 256)) * 0.02
mask = jax.random.bernoulli(k_drop, 0.9, (32, 256))
```

A key is consumed by the function you give it to. Calling `jax.random.normal(k, ...)` twice with the same `k` returns identical arrays, so reuse produces correlated "random" numbers without any error. The discipline is: split before each use, keep one key as the carry for later, and never pass the same key to two consumers.

~~~dg-flow Splitting a root key into independent streams.
key(0) -(split)-> *carry key -(split)-> next carry + step key
key(0) -(split)-> init key -(split per layer)-> layer keys
key(0) -(split)-> data key -(fold_in step)-> per-step shuffle key
~~~

`jax.random.fold_in(key, i)` derives a new key from a key and an integer, which is handy for per-step or per-device keys without carrying state: `step_key = jax.random.fold_in(base_key, step)`. The default generator is counter-based Threefry, so splitting is cheap and results are reproducible across devices. Alternative implementations (such as `"rbg"`) can be selected for speed on TPU through the `jax_default_prng_impl` option, at the cost of different streams.

## Keys under jit and vmap

Keys are ordinary arguments, so a jitted function takes a key and returns a new one if it needs to keep drawing. Under `vmap`, give each example its own key by splitting into a batch: `keys = jax.random.split(key, B)` returns a key array of shape `(B,)`, and `jax.vmap(f)(keys, xs)` gives each example independent noise. Passing one unbatched key with `in_axes=None` gives every example the same noise, which is occasionally what you want (the same augmentation for an ensemble) and usually a bug.

Typed keys (`jax.random.key`) carry their implementation in the dtype and refuse arithmetic, so mistakes like adding a key to an array fail loudly. Legacy keys from `jax.random.PRNGKey` are plain `uint32` arrays of shape `(2,)`, which is why old code batches them with a trailing axis of 2. Convert with `jax.random.wrap_key_data` and `jax.random.key_data` when you meet old checkpoints.

## Precision

| setting | default | how to change
| --- | --- | --- |
| float dtype | float32 | `jax.config.update("jax_enable_x64", True)` before creating arrays, or `JAX_ENABLE_X64=1`
| integer dtype | int32 | same flag
| float32 matmul on TPU | reduced-precision passes | `precision=jax.lax.Precision.HIGHEST`, or `with jax.default_matmul_precision("float32"):`
| Python scalars | weakly typed | `bf16_array * 2.0` stays bfloat16

Without x64 mode, `jnp.array(1.0, dtype=jnp.float64)` gives float32 with only a warning, so scientific code that needs double precision must enable the flag at program start. Mixed dtypes promote along JAX's type lattice; `bfloat16 + float16` gives float32, and a float32 array mixed into a bfloat16 model upcasts everything downstream, just as in PyTorch.

The matmul default matters in practice. On TPUs, a float32 `jnp.dot` by default uses bfloat16 multiplication passes, which is fast and usually fine for training, but can make a numerical test against NumPy fail at the third or fourth significant digit. Set the precision per call or with the context manager when you need full float32 accuracy. The same `jnp.dot` takes `preferred_element_type` to choose the accumulator: an int8 quantised matmul written as `jnp.dot(a_q, w_q, preferred_element_type=jnp.int32)` accumulates in int32, and you then multiply by the per-channel scales in float32.

Mixed-precision training in JAX is explicit: keep float32 master parameters, cast them to bfloat16 inside the forward pass, and compute the loss and the optimiser update in float32. bfloat16 has float32's exponent range and needs no loss scaling. float16 does: multiply the loss by a scale $S$ before differentiating, divide the gradients by $S$, skip the step and lower $S$ if any gradient is non-finite, and raise $S$ after a run of clean steps.

## Memory and donation

On GPUs, JAX preallocates a large fraction of device memory (75% by default) the first time it runs, to avoid fragmentation. This surprises people sharing a GPU or running a PyTorch process alongside. `XLA_PYTHON_CLIENT_PREALLOCATE=false` allocates on demand and `XLA_CLIENT_MEM_FRACTION=0.5` (older name `XLA_PYTHON_CLIENT_MEM_FRACTION`) changes the fraction; both must be set before JAX initialises.

Because arrays are immutable, a training step `state = step(state, batch)` would normally hold the old state and the new state at the same time. For a large model with Adam, the state is parameters plus two moment trees, so this doubles the biggest allocation in the program. Donation tells XLA it may reuse the input buffers for outputs:

```python
from functools import partial

@partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch):
    ...
    return new_state, metrics

state, metrics = train_step(state, batch)   # old state buffers are reused
```

After the call, the donated arrays are deleted, and touching the old `state` raises `RuntimeError: Array has been deleted`. Only donate arguments you rebind immediately, and keep a separate copy if you need the old values (for an EMA or a rollback). If a backend cannot use a donation, JAX warns and carries on.

Activations are the other big consumer. `jax.checkpoint` (alias `jax.remat`) wraps a function so its intermediates are not saved for the backward pass and are recomputed instead; applied to each block of a transformer inside a `lax.scan` over layers, it cuts activation memory to roughly one block's worth plus the block inputs, at about one extra forward pass of compute.

<div class="ex">
<div class="lab">Worked example</div>

**dropout that was not random**

A jitted training step takes `key` as an argument and uses it for dropout, but the loop calls `train_step(params, batch, key)` with the same `key` every step. Training works, yet the model overfits as if dropout were a fixed mask.

It is a fixed mask: the same key produces the same Bernoulli draws every step, so the same units are dropped forever. Inside the loop, write `key, step_key = jax.random.split(key)` and pass `step_key`, or derive `jax.random.fold_in(base_key, step)` inside the step.

A quick test for any stochastic function is to call it twice with different keys and assert the outputs differ, and twice with the same key and assert they match.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

treat random keys like any other data and split them before every use, know which precision your hardware gives you by default, and use donation and rematerialisation to keep immutable arrays from doubling your memory.

</div>
