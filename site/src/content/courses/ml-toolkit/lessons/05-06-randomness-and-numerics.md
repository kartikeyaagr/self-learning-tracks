---
id: "05.06"
course: "ml-toolkit"
module: "05"
slug: "randomness-and-numerics"
title: "Randomness and numerics"
keyIdeas: ["JAX has no global random state. Every random function takes an explicit **key**, the same key always gives the same numbers, and you make new keys with `jax.random.split` (or `fold_in`) rather than reusing one.", "`jax.random.key(seed)` returns a typed key array (dtype `key<fry>`); the older `jax.random.PRNGKey(seed)` returns a raw `uint32[2]` array. Both work with every `jax.random` function, but new code should use typed keys. Under `vmap`, split into one key per example; a broadcast key gives every example the same noise.", "Arrays are float32 by default and float64 requests are silently downcast unless `jax_enable_x64` is set at startup. On TPUs (and with TF32 on recent NVIDIA GPUs) float32 matmuls may use reduced-precision passes by default; request higher precision when it matters.", "bfloat16 keeps float32's range and needs no loss scaling; float16 underflows below about 6e-8, so multiply the loss by a scale before differentiating and divide the gradients afterwards.", "On GPU, JAX preallocates most of device memory at startup. Inside a training step, buffer **donation** (`donate_argnums`) lets XLA write the new parameters and optimiser state into the old ones' memory, roughly halving the step's peak for that state.", "`jax.checkpoint` (remat) trades compute for activation memory by recomputing intermediates in the backward pass, and `jax_debug_nans` finds the first operation that produced a NaN."]
question: "A colleague's data-augmentation function is `vmap`ped over the batch as `jax.vmap(augment, in_axes=(None, 0))(key, images)` and the key is split once per step. They report that augmentations look random across steps but the model ignores them. What is going on, and what is the correct call? How would you write a test that would have caught it?"
answer: "With `in_axes=None` the key is broadcast, so every image in the batch receives the same random crop, flip and colour jitter within a step. Across steps the key changes, so the augmentation looks random when you inspect one image over time, but within a batch all examples are transformed identically. That greatly reduces the diversity of augmentations per gradient step (and with batch-level statistics such as batch norm, an identical transform is partly normalised away). The fix is to give each example its own key: `keys = jax.random.split(step_key, images.shape[0])` and `jax.vmap(augment)(keys, images)`, with `in_axes=0` for both. Keep the outer loop as it is: `key, step_key = jax.random.split(key)` each step, so the per-example keys also change from step to step. A test that catches this: build a batch made of one image repeated B times, augment it, and assert that the outputs are not all equal (for example, that the per-example crop offsets or pixel means differ). Add two more cheap assertions for any stochastic function: the same key twice gives identical output, and two different keys give different output."
sources: [{"title": "JAX docs: Pseudorandom numbers", "url": "https://docs.jax.dev/en/latest/101/random.html", "note": "Explicit keys, split and fold_in, and why JAX avoids global random state."}, {"title": "JAX docs: Common gotchas in JAX", "url": "https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html", "note": "Random numbers, double (64-bit) precision and debugging NaNs."}, {"title": "JAX docs: GPU memory allocation", "url": "https://docs.jax.dev/en/latest/201/gpu-memory.html", "note": "Preallocation and the XLA_PYTHON_CLIENT_* environment variables."}, {"title": "JAX docs: Just-in-time compilation (buffer donation section)", "url": "https://docs.jax.dev/en/latest/201/jit.html", "note": "How donate_argnums works and when donated buffers cannot be reused."}]
practice: ["prng-key-splitting", "mixed-precision-with-loss-scaling", "rematerialized-mlp-block", "int8-matmul-with-per-channel-scales"]
tags: ["PRNG keys", "precision", "memory"]
tools: ["JAX"]
---

## What you will learn

- Why JAX has no hidden global random generator, and what a **key** is instead.
- The one habit that keeps random numbers correct: split a key before every use, with `jax.random.split` or `jax.random.fold_in`.
- How to pass keys through `jit` and `vmap` so every step and every example gets its own noise.
- What precision JAX gives you by default (float32, reduced-precision matmuls on some hardware) and how to ask for more.
- How float16 gradients vanish, and how loss scaling or bfloat16 prevents it.
- How donation and rematerialisation save memory, and how to find a NaN.

*Before you start:* you need `jax.numpy` and immutability (05.01), `jit` (05.03), `vmap` and `in_axes` (05.04), and pytrees (05.05). PyTorch mixed precision (module 04) helps but is optional.

## Picking up from pytrees

Lesson 05.05 turned a model into plain data plus a pure `step` function. This lesson covers three things that purity makes awkward: random numbers need state, JAX's precision defaults differ from NumPy's, and immutable arrays turn "update in place" into "make a new copy", which costs memory.

## The idea in one picture

In NumPy you write `np.random.normal(size=3)` twice and get two different answers, because a hidden generator object advanced in between. In JAX you write:

```python
import jax, jax.numpy as jnp

key = jax.random.key(0)
jax.random.normal(key, (3,))   # [ 1.6226  2.0253 -0.4336]
jax.random.normal(key, (3,))   # [ 1.6226  2.0253 -0.4336]   same key, same numbers
k1, k2 = jax.random.split(key)
jax.random.normal(k1, (3,))    # [ 1.0040 -0.9063 -0.7482]
jax.random.normal(k2, (3,))    # [-2.4425 -2.0357  0.2055]   new keys, new numbers
```

A **key** is a small value that fully determines the random numbers drawn from it: same key in, same numbers out, on every machine. For fresh randomness you **split** a key into new keys, which are statistically independent of each other and of the parent. Forget to split and nothing crashes; you silently get the same numbers again. The figure shows what that does to dropout.

~~~dg-fig 05.06-reuse.svg
Dropout masks for 16 hidden units over six training steps. Left: the same key is passed every step, so the same units are dropped forever. Right: the key is split every step, so each step drops a different random half.
~~~

The left grid is a real bug: training still works, but dropout has become a fixed pruning of half the network.

## Step 1: why JAX makes keys explicit

NumPy and PyTorch keep a global generator that every random call advances, so a function's output depends on how many random calls ran before it. JAX's transformations cannot live with that:

- `jit` traces once and replays compiled code, so a generator read during tracing would be baked in as a constant.
- `vmap` would be ambiguous: which example advances the generator first?
- Parallel devices would make results depend on scheduling order.

So JAX replaces the hidden state with an ordinary value you pass around, and sampling becomes a pure function: $\text{sample} = f(\text{key}, \text{shape})$. The default generator, **Threefry**, is counter-based (essentially a hash of key and position), so splitting is cheap and results are reproducible across devices.

## Step 2: split before every use

The discipline has three rules:

1. A key is consumed by the function you give it to. Never pass the same key to two consumers.
2. Before drawing, split: keep one key as the carry for later, use the other now.
3. To make many keys at once, split into `n` pieces.

```python
key = jax.random.key(0)                       # typed key, dtype key<fry>
key, k_init, k_drop = jax.random.split(key, 3)
w = jax.random.normal(k_init, (784, 256)) * 0.02
mask = jax.random.bernoulli(k_drop, 0.9, (32, 256))
# `key` is the carry: split it again next time you need randomness
```

~~~dg-flow Splitting a root key into independent streams.
key(0) -(split)-> *carry key -(split)-> next carry + step key
key(0) -(split)-> init key -(split per layer)-> layer keys
key(0) -(split)-> data key -(fold_in step)-> per-step shuffle key
~~~

The second tool is `jax.random.fold_in(key, i)`, which derives a new key from a key and an integer. `step_key = jax.random.fold_in(base_key, step)` gives each step its own key from the step number alone (the same trick gives per-device or per-layer keys), so a run restarted from a checkpoint at step 5000 reproduces exactly the randomness it would have had.

**Typed keys versus legacy keys.** `jax.random.key(seed)` returns a typed key array whose dtype is `key<fry>` and whose shape is `()`. The older `jax.random.PRNGKey(seed)` returns a raw `uint32` array of shape `(2,)`. Every `jax.random` function accepts both, but new code should use typed keys, because they refuse arithmetic:

```python
jax.random.key(0) + 1
# TypeError: add does not accept dtypes key<fry>, int32.
jax.random.PRNGKey(0)                     # Array([0, 0], dtype=uint32)
jax.random.key_data(jax.random.key(0))    # Array([0, 0], dtype=uint32)
```

Legacy keys explain why older code batches keys with a trailing axis of 2: a batch has shape `(B, 2)`. Convert old checkpoints with `jax.random.key_data` and `jax.random.wrap_key_data`.

The generator implementation can be changed through the `jax_default_prng_impl` option; for example `"rbg"` is faster on TPU, at the cost of producing different streams from Threefry.

## Step 3: keys under jit and vmap

Keys are ordinary arguments, so a jitted function simply takes one (and returns a new carry key if it must keep drawing). The training loop splits a fresh `step_key` each iteration, as in the full example below.

Under `vmap`, the question is whether every example should get its own randomness. Usually yes. Split once into a batch of keys and map over them:

```python
def noise(k, x):
    return x + jax.random.normal(k, x.shape)

xs = jnp.zeros((4, 3))
jax.vmap(noise, in_axes=(None, 0))(jax.random.key(1), xs)
# every row is [-0.1544  0.0847 -0.1360]          one key broadcast: identical rows
jax.vmap(noise)(jax.random.split(jax.random.key(1), 4), xs)
# rows differ: [ 0.53 -0.72  0.91], [-0.24  0.12 -0.76], ...
```

`jax.random.split(key, B)` returns a key array of shape `(B,)`, which `vmap` slices like any other array. Passing one unbatched key with `in_axes=None` gives every example the *same* noise. That is occasionally intended (the same augmentation applied across an ensemble) and usually a bug.

~~~dg-fig 05.06-vmap.svg
Random crop offsets for twelve images in one batch. With a single broadcast key (left) all twelve images get exactly the same crop; with one split key per image (right) the crops spread over the allowed range.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**dropout that was not random**

A jitted training step takes `key` as an argument and uses it for dropout, but the loop calls `train_step(params, batch, key)` with the same `key` every step. Training runs fine, yet the model overfits as if dropout were switched off.

In fact dropout is a fixed mask. The same key produces the same Bernoulli draws every step, so the same units are dropped forever (the left grid in the first figure). The network simply learns to live without those units.

Fix: inside the loop write `key, step_key = jax.random.split(key)` and pass `step_key`, or compute `jax.random.fold_in(base_key, step)` inside the step.

Check: log the fraction of units that were dropped at least once over the first 10 steps. With dropout rate 0.5 and fresh keys it should be about $1 - 0.5^{10} \approx 99.9\%$; with a reused key it stays at 50%.

</div>

## Step 4: know your default precision

Floating-point numbers store a fixed number of **significant bits**, so the gap between neighbouring numbers grows with their size: precision is *relative*, as the flat lines show.

~~~dg-fig 05.06-formats.svg
The relative gap between neighbouring representable numbers for four formats. bfloat16 keeps float32's range but has only 8 significant bits; float16 has 11 bits but stops at 65,504; float32 has 24; float64 has 53.
~~~

For float32 the relative gap is about $2^{-23} \approx 1.2 \times 10^{-7}$, so you get roughly 7 decimal digits. Near $10^8$ the gap is exactly 8: float32 cannot tell 100,020,001 from 100,020,000.

JAX's defaults differ from NumPy's:

| setting | default | how to change
| --- | --- | --- |
| float dtype | float32 | `jax.config.update("jax_enable_x64", True)` before creating arrays, or `JAX_ENABLE_X64=1`
| integer dtype | int32 | same flag
| float32 matmul on TPU | reduced-precision passes | `precision=jax.lax.Precision.HIGHEST`, or `with jax.default_matmul_precision("float32"):`
| Python scalars | weakly typed | `bf16_array * 2.0` stays bfloat16

Without x64 mode, `jnp.array(1.0, dtype=jnp.float64)` gives float32 with only a warning, so set the flag at program start. Why it can matter: a variance computed as $\overline{x^2} - \bar x^2$ subtracts two huge, nearly equal numbers.

~~~dg-fig 05.06-cancel.svg
Relative error of a variance computed three ways, for data with spread 1 and a growing mean. The one-pass float32 formula is useless by a mean of about 1000 (it can even go negative); subtracting the mean first, or using float64, stays accurate.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a variance of zero**

Take $x = (10000, 10001, 10002)$ in float32. The true variance is $\frac{1 + 0 + 1}{3} = 0.667$.

One-pass formula, first term: $x^2 = (100{,}000{,}000,\ 100{,}020{,}001,\ 100{,}040{,}004)$, sum $300{,}060{,}005$, mean $100{,}020{,}001.67$.

Near $10^8$ float32 numbers are 8 apart, so the mean is stored as the nearest multiple of 8: $100{,}020{,}000$.

Second term: $\bar x = 10001$, $\bar x^2 = 100{,}020{,}001$, which float32 also stores as $100{,}020{,}000$.

Difference: $100{,}020{,}000 - 100{,}020{,}000 = 0$. The answer is 0 instead of 0.667, a 100% error.

Two-pass: deviations from the mean are $(-1, 0, 1)$, squares $(1, 0, 1)$, mean $0.667$. Exact. In JAX: `jnp.mean(x*x) - jnp.mean(x)**2` prints `0.0`, `jnp.var(x)` prints `0.6666667`.

</div>

The lesson is not "always use float64": prefer the stable formula, and turn on x64 when a problem truly needs 15 digits (ill-conditioned optimisation, long simulations, finite-difference checks).

**Promotion.** Mixed dtypes promote along JAX's type lattice: `bfloat16 + float16` gives float32, and a float32 array mixed into a bfloat16 model upcasts everything downstream, just as in PyTorch. Python scalars are **weakly typed**, so `x_bf16 * 2.0` stays bfloat16 instead of upcasting.

**Matmul precision.** On TPUs, a float32 `jnp.dot` by default uses bfloat16 multiplication passes: fast and fine for training, but a test against NumPy can fail at the third or fourth significant digit. Recent NVIDIA GPUs can similarly use TF32. Use the table's settings when you need full accuracy (on CPU, float32 matmuls are already exact float32).

**Accumulator dtype.** `jnp.dot` takes `preferred_element_type` to choose the type of the accumulator. An int8 quantised matmul accumulates in int32 and is then rescaled in float32 with the per-channel scales:

```python
a = jnp.array([[1, 2], [3, 4]], jnp.int8)
jnp.dot(a, a, preferred_element_type=jnp.int32)   # [[ 7 10] [15 22]], dtype int32
# then: y = acc.astype(jnp.float32) * x_scale * w_scale[None, :]
```

## Step 5: mixed precision and loss scaling

Mixed precision in JAX is explicit (there is no autocast): keep **master parameters** in float32, cast them to bfloat16 inside the forward pass with `jax.tree.map(lambda p: p.astype(jnp.bfloat16), params)`, and compute the loss and optimiser update in float32.

bfloat16 has float32's exponent range, so small gradients do not underflow and no loss scaling is needed. float16 has a much smaller range: its smallest positive (subnormal) value is about $6 \times 10^{-8}$, and anything smaller becomes exactly zero.

~~~dg-fig 05.06-lossscale.svg
A synthetic but typical spread of gradient magnitudes. In float16, 12% of them fall below the smallest representable value and become 0. Multiplying the loss by 1024 shifts every gradient right by about three powers of ten, so only 0.02% are lost.
~~~

**Loss scaling** fixes this: multiply the loss by a scale $S$ before differentiating (the chain rule multiplies every gradient by $S$), then divide the gradients by $S$ in float32 before the update. With dynamic scaling, if any gradient is non-finite you skip the step and lower $S$; after a run of clean steps you raise $S$ again.

<div class="ex">
<div class="lab">Worked example</div>

**one gradient through float16**

A gradient of $2 \times 10^{-8}$ cast to float16 becomes `0.0`: it is below the smallest subnormal, $2^{-24} \approx 5.96 \times 10^{-8}$.

With $S = 1024$ the scaled gradient is $2.048 \times 10^{-5}$. float16 stores it as $2.0504 \times 10^{-5}$ (344 steps of $2^{-24}$).

Unscaled in float32: $2.0504 \times 10^{-5} / 1024 = 2.0023 \times 10^{-8}$, a relative error of $0.12\%$ instead of $100\%$.

</div>

## Step 6: memory, donation and rematerialisation

**Preallocation.** On GPUs, JAX preallocates 75% of device memory on first use, to avoid fragmentation, which surprises people sharing a GPU with another process. `XLA_PYTHON_CLIENT_PREALLOCATE=false` allocates on demand, and `XLA_CLIENT_MEM_FRACTION=0.5` (older name `XLA_PYTHON_CLIENT_MEM_FRACTION`) changes the fraction. Both must be set before JAX initialises.

**Donation.** Because arrays are immutable, `state = step(state, batch)` normally holds the old state and the new state at the same time. With Adam, the state is parameters plus two moment trees, so this doubles the biggest allocation in the program. For a one-billion-parameter model in float32 that is $3 \times 4$ GB $= 12$ GB of state, and 24 GB at the peak of the step. **Donation**, written `@partial(jax.jit, donate_argnums=(0,))` on the step function, tells XLA it may write the outputs into the input buffers of argument 0. After the call the donated arrays are deleted. Touching the old value raises `RuntimeError: Array has been deleted with shape=float32[1000].` (that message is from a real run here). Only donate arguments you rebind immediately, and keep a separate copy if you need the old values, for an EMA or a rollback. If a backend cannot use a donation, JAX warns and carries on.

**Rematerialisation.** Activations are the other big consumer: the backward pass needs intermediate values from the forward pass, so they are saved. `jax.checkpoint` (alias `jax.remat`) wraps a function so its intermediates are *not* saved and are recomputed in the backward pass instead. Applied to each block of a transformer inside a `lax.scan` over layers, it cuts activation memory to roughly the block inputs plus one block's worth, at the cost of about one extra forward pass.

~~~dg-fig 05.06-memory.svg
Left: arithmetic for a one-billion-parameter float32 model with Adam; donation removes the second copy of the state. Right: activation memory saved for the backward pass, measured with `print_saved_residuals` on an MLP stack; with `jax.checkpoint` per block it grows about twenty times more slowly.
~~~

## The whole pattern in code

Here is everything together: a key carried through the loop, per-example keys under `vmap`, donated state, and a rematerialised block scanned over layers. It runs on CPU as written.

```python
from functools import partial
from jax.ad_checkpoint import print_saved_residuals

L, D = 8, 256
key = jax.random.key(0)
key, k1, k2 = jax.random.split(key, 3)
params = (jax.random.normal(k1, (L, D, 4 * D)) * 0.02,
          jax.random.normal(k2, (L, 4 * D, D)) * 0.02)

def block(x, w):
    return x + jax.nn.gelu(x @ w[0]) @ w[1]

def loss(params, x, keys):
    x = jax.vmap(lambda k, xi: xi + 0.1 * jax.random.normal(k, xi.shape))(keys, x)
    y, _ = jax.lax.scan(lambda c, w: (jax.checkpoint(block)(c, w), None), x, params)
    return jnp.mean(y ** 2)

@partial(jax.jit, donate_argnums=(0,))
def step(params, x, key):
    keys = jax.random.split(key, x.shape[0])          # one key per example
    l, g = jax.value_and_grad(loss)(params, x, keys)
    return jax.tree.map(lambda p, gi: p - 0.1 * gi, params, g), l

x = jnp.ones((64, D))
for i in range(3):
    key, step_key = jax.random.split(key)
    params, l = step(params, x, step_key)
print_saved_residuals(loss, params, x, jax.random.split(key, 64))
print(l)                 # 0.3152
# saved: the two parameter stacks, one f32[8,64,256] stack of block inputs and
# the f32[64,256] output; no per-layer f32[64,1024] GELU intermediates
```

## Checking that it worked

- **Randomness:** the same key gives the same output, different keys give different outputs, and a batch of one example repeated $B$ times gives different outputs per example. Make these three assertions unit tests.
- **Reproducibility:** rerun with the same seed and compare a hash of the final parameters. Different XLA versions can still change the last bits of float results.
- **Precision:** print `x.dtype` at the boundaries of your model. If you expected float64 and see float32, x64 mode is off.
- **NaNs:** set `jax.config.update("jax_debug_nans", True)`. JAX then checks every result, and when a NaN appears inside a jitted function it re-runs that function op by op and raises at the first operation that produced it. It is slow, so use it for debugging only.
- **Memory:** `print_saved_residuals` shows what the backward pass keeps.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Dropout or augmentation has no regularising effect | same key reused every step | split the key each step, or `fold_in(base_key, step)`
| All examples in a batch get identical noise | key broadcast with `in_axes=None` | `split(key, B)` and map over the keys
| float64 code gives float32 results | x64 mode not enabled at startup | set `jax_enable_x64` before creating any array
| Test against NumPy fails at the 4th digit on TPU | default reduced-precision matmul | `Precision.HIGHEST` or `default_matmul_precision`
| float16 training stalls; many gradients exactly 0 | gradient underflow | dynamic loss scaling, or switch to bfloat16
| `RuntimeError: Array has been deleted` | used an argument after donating it | rebind immediately; copy what you still need
| Out of memory on a GPU shared with PyTorch | JAX preallocated 75% | `XLA_PYTHON_CLIENT_PREALLOCATE=false` before import
| Loss turns NaN after hours | overflow, log of 0 | `jax_debug_nans` finds the first bad op

## Going deeper

**Why counter-based generators fit JAX.** A stateful generator (like NumPy's PCG64) is a state machine stepped forward by each draw. A counter-based generator such as Threefry computes block $i$ directly as a keyed hash of $i$, so any slice of a random array can be computed independently, on any device, in any order. Splitting is itself a hash of the parent key, so the whole key tree is determined by the root seed. The cost is a few cipher rounds per draw, which is why faster options such as `"rbg"` exist for TPUs.

**Floating point arithmetic is not associative.** $(a + b) + c$ can differ from $a + (b + c)$ in the last bits, and XLA reorders reductions freely, so jitted and un-jitted code, or 1 and 8 devices, can disagree slightly. Compare with `jnp.allclose`, never `==`.

**Remat policies.** `jax.checkpoint` takes a `policy` that decides what to keep. `jax.checkpoint_policies.dots_saveable` saves matmul outputs and recomputes cheap elementwise ops like GELU, often the best balance. Between "save everything" and "save only block inputs", $\sqrt{L}$ checkpointing needs $O(\sqrt{L})$ memory for $L$ layers.

**Donation and aliasing.** Donation only helps when an output has the same shape and dtype (and, for sharded arrays, sharding) as a donated input, so XLA can alias the buffers. `p - lr * g` fits perfectly; otherwise JAX warns that the donation was unused.

## Key terms

| Term | Meaning
| --- | --- |
| PRNG key | A value that fully determines the random numbers drawn from it
| split | Make several new, independent keys from one key
| fold_in | Derive a new key from a key and an integer, such as a step number
| Typed key | Key with dtype `key<fry>` from `jax.random.key`; refuses arithmetic
| x64 mode | The `jax_enable_x64` flag that allows float64 and int64 arrays
| Matmul precision | How many bfloat16 passes a float32 matmul uses on TPU or GPU
| Loss scaling | Multiplying the loss so small float16 gradients do not underflow
| Donation | Letting XLA reuse an input buffer for an output with `donate_argnums`
| Rematerialisation | Recomputing activations in the backward pass instead of storing them

<div class="take">
<div class="lab">Key takeaway</div>

treat random keys like any other data and split them before every use, know which precision your hardware gives you by default, and use donation and rematerialisation to keep immutable arrays from doubling your memory.

</div>
