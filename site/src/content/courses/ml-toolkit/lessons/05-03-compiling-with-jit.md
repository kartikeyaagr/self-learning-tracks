---
id: "05.03"
course: "ml-toolkit"
module: "05"
slug: "compiling-with-jit"
title: "Compiling with jit"
keyIdeas: ["`jax.jit` traces a function once per input signature with abstract values (shape and dtype, no data), records the operations as a jaxpr, compiles that with XLA, and caches the executable. Later calls with the same signature skip Python entirely.", "Python code runs only during tracing. `print` and other side effects fire once, values are tracers, and Python `if`/`while` on a traced value raises an error because the value is not known at trace time.", "Arguments marked with `static_argnums` or `static_argnames` are treated as compile-time constants: they must be hashable, can drive Python control flow and shapes, and every new value triggers a recompilation.", "Data-dependent control flow uses `jnp.where`, `lax.cond`, `lax.switch`, `lax.fori_loop`, `lax.while_loop` and `lax.scan`. Python loops are unrolled at trace time, which makes long loops slow to compile.", "Recompilation is the main jit performance bug. Watch for changing shapes, new static values, Python scalars standing in for traced values, and `jax.config.update('jax_log_compiles', True)` to see when it happens."]
question: "A jitted beam-search decoder takes `max_len` and `beam_size` as arguments and loops `for t in range(max_len)` in Python, breaking out early when all beams have emitted an end token. It fails with a tracer error. A colleague deletes the early break and marks both arguments static; it now runs, but compiling takes four minutes and every new `max_len` compiles again. Explain both symptoms and how you would restructure it."
answer: "The early `break` needs a Python boolean computed from traced data (whether all beams finished), so under `jit` the comparison is a tracer and converting it to `bool` raises a `TracerBoolConversionError`. Deleting the break removes the error, but the Python loop over a static `max_len` is unrolled: the jaxpr contains `max_len` copies of the decoder step, so compile time grows with the length, and every new `max_len` value is a new static value and therefore a new compilation. The fix is to keep `max_len` and `beam_size` static (they determine shapes, so they should be), and replace the Python loop with a compiled loop. `lax.while_loop` with a condition `(t < max_len) & ~all_done` stops early on the device, with a carry holding fixed-shape buffers of size `(beam_size, max_len)` updated with `.at[:, t].set(...)`. Alternatively `lax.scan` over `max_len` steps with a `done` mask keeps going but freezes finished beams with `jnp.where`; scan is simpler when you might later differentiate through the loop, which `while_loop` does not support in reverse mode. Either way the body is traced once, so compile time no longer grows with `max_len`."
sources: [{"title": "JAX docs: Just-in-time compilation", "url": "https://docs.jax.dev/en/latest/201/jit.html", "note": "Tracing, jaxprs, static arguments and the caching rules."}, {"title": "JAX docs: Control flow and logical operators with JIT", "url": "https://docs.jax.dev/en/latest/201/control-flow.html", "note": "cond, switch, fori_loop, while_loop and scan, and which are differentiable."}, {"title": "JAX docs: Common gotchas in JAX", "url": "https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html", "note": "Pure functions, side effects under jit and control-flow errors."}]
practice: ["static-arguments-in-jit", "traced-learning-rate-schedule", "exponential-moving-average-with-scan", "spectral-norm-by-power-iteration"]
tags: ["jit", "tracing", "lax control flow"]
tools: ["JAX", "XLA"]
---

## What jit actually does

PyTorch runs eagerly and `torch.compile` (lesson 04.07) captures graphs as an optimisation. In JAX, compilation is the normal way to run anything non-trivial, and its model is simple to state. When you call a jitted function, JAX replaces each array argument with a **tracer** that carries only a shape and dtype, runs your Python function once with those tracers, and records every primitive operation into an intermediate representation called a jaxpr. XLA compiles the jaxpr into fused device kernels, and the result is cached under a key made of the argument shapes, dtypes, pytree structure, static argument values and sharding.

~~~dg-flow The life of a jitted call.
call f(x) -(cache hit)-> run compiled executable
call f(x) -(new signature)-> *trace with tracers -> jaxpr -> XLA compile -> cache -> run compiled executable
~~~

You can inspect the trace with `jax.make_jaxpr(f)(x)`, and the compiled program with `jax.jit(f).lower(x).compile().as_text()`. The jaxpr of a small function is worth reading once: you will see that `jnp.where` became a `select_n`, that Python arithmetic on constants has been folded, and that nothing of your Python control flow survives.

## Side effects and tracers

Because Python runs only during tracing, side effects behave unexpectedly:

```python
@jax.jit
def f(x):
    print("tracing with", x)          # prints a tracer, once per new signature
    jax.debug.print("x = {}", x)      # prints the runtime value, every call
    return x * 2

f(1.0); f(2.0)            # the first print fires only on the first call
f(jnp.ones(2))            # new shape: traces and prints again
```

The same rule makes global state dangerous. A jitted function that reads a global variable bakes its trace-time value into the compiled code; changing the global later has no effect until something forces a retrace. Appending to a list inside a jitted function appends tracers that are useless outside it. Write jitted functions as pure functions of their arguments.

## Static arguments

Some arguments are configuration rather than data: a window size, a number of layers, an activation name, a `training` flag. If they decide shapes or Python control flow they must be known at trace time, so mark them static:

```python
from functools import partial

@partial(jax.jit, static_argnames="window")
def pool(x, window):
    return x.reshape(-1, window).mean(axis=1)    # shape depends on window

pool(jnp.arange(8.0), window=2)     # compiles for window=2
pool(jnp.arange(8.0), window=4)     # compiles again for window=4
```

Static values must be hashable (ints, strings, tuples, frozen dataclasses; not arrays or lists), and each distinct value is a separate compilation. That is the trade-off: static arguments buy flexibility at trace time and cost a compile per value, so use them for a handful of configurations, never for something that changes every step.

## Control flow on traced values

Python `if x > 0:` needs a concrete boolean, but under `jit` `x > 0` is a tracer, so JAX raises a `TracerBoolConversionError` (a kind of concretisation error). There are three fixes, in order of preference:

| need | tool | notes
| --- | --- | --- |
| choose elementwise | `jnp.where(c, a, b)` | computes both sides, then selects
| choose one branch | `lax.cond(pred, f_true, f_false, *ops)` / `lax.switch` | both branches traced; must return same shapes and dtypes
| loop with known trip count | `lax.fori_loop(lo, hi, body, init)` | compiled as a loop, not unrolled
| loop until a condition | `lax.while_loop(cond, body, init)` | not reverse-differentiable
| loop that emits per-step outputs | `lax.scan(f, carry, xs)` | the workhorse: RNNs, EMAs, layer stacks, accumulation

Python loops over a static range are legal: JAX unrolls them, so a 1,000-step Python loop becomes 1,000 copies of the body in the jaxpr and compiles slowly. `lax.scan` compiles the body once:

```python
def ema(xs, decay=0.99):
    def step(m, x):
        m = decay * m + (1 - decay) * x
        return m, m                      # (new carry, per-step output)
    _, ms = jax.lax.scan(step, jnp.zeros(xs.shape[1:]), xs)
    return ms
```

`scan` is also how JAX code expresses gradient accumulation over micro-batches and applies a stack of identical layers whose parameters are stacked along a leading axis.

## Avoiding recompilation

Every new input shape is a new compilation, which can take seconds to minutes for a large model. Common causes:

- Variable-length batches. Pad to a few bucket lengths (lesson 1) so only a few shapes ever occur.
- A changing value passed as a static argument, such as the step number or the learning rate.
- Switching between Python scalars and arrays of different dtypes for the same argument.
- Calling `jax.jit(f)` inside a loop on a fresh lambda, which defeats the cache because the function object is new each time.

Turn on `jax.config.update("jax_log_compiles", True)` while developing to log every compilation.

<div class="ex">
<div class="lab">Worked example</div>

**the learning-rate schedule that recompiled every step**

A training step is written `@partial(jax.jit, static_argnames="step")` so that a warmup-cosine schedule can use Python `if step < warmup:`. Throughput is terrible, and the compile log shows a new compilation for every step.

Each step value is a new static value, so each call traces and compiles from scratch. The fix is to pass `step` as an ordinary traced integer array and write the schedule with array operations: `lr = jnp.where(step < warmup, peak * step / warmup, cosine(step))`.

Now one compilation serves the whole run. Optax schedules are written this way for the same reason: they are pure functions of a traced step count.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

jit traces once per signature and compiles the trace. Keep functions pure, make only true configuration static, express data-dependent branches and loops with `jnp.where` and `lax`, and keep the set of input shapes small.

</div>
