---
id: "05.03"
course: "ml-toolkit"
module: "05"
slug: "compiling-with-jit"
title: "Compiling with jit"
keyIdeas: ["`jax.jit` runs your Python function once per input **signature** (shapes, dtypes, structure, static values) using **tracers**, records the operations as a **jaxpr**, compiles it with XLA and caches the program. Later calls with the same signature skip Python entirely, which is why the first call is slow and the rest are fast.", "XLA **fuses** many small operations into one kernel, so jitted code avoids per-operation dispatch and repeated passes over memory. Time it after a warm-up call and with `.block_until_ready()`, because JAX dispatches asynchronously.", "Python code runs only during tracing: `print` fires once and shows tracers, globals are baked in as constants, and Python `if`/`while` on a traced value raises a `TracerBoolConversionError`. Keep jitted functions pure and use `jax.debug.print` for runtime values.", "Arguments marked with `static_argnums` or `static_argnames` are compile-time constants: they must be hashable, may drive shapes and Python control flow, and every new value triggers a recompilation.", "Data-dependent control flow uses `jnp.where`, `lax.cond`, `lax.switch`, `lax.fori_loop`, `lax.while_loop` and `lax.scan`. Python loops are unrolled at trace time, so long ones make compile time grow with their length.", "Recompilation is the main jit performance bug: watch for changing shapes, changing static values, mixed Python scalars and arrays, and fresh lambdas, and turn on `jax.config.update('jax_log_compiles', True)` to see each compile."]
question: "A jitted beam-search decoder takes `max_len` and `beam_size` as arguments and loops `for t in range(max_len)` in Python, breaking out early when all beams have emitted an end token. It fails with a tracer error. A colleague deletes the early break and marks both arguments static; it now runs, but compiling takes four minutes and every new `max_len` compiles again. Explain both symptoms and how you would restructure it."
answer: "The early `break` needs a Python boolean computed from traced data (whether all beams have finished). Under `jit` that comparison is a tracer whose value is unknown at trace time, so converting it to `bool` raises a `TracerBoolConversionError`. Deleting the break removes the error, but the Python loop over a static `max_len` is unrolled at trace time: the jaxpr contains `max_len` copies of the whole decoder step, so trace and compile time grow linearly with the length (hence four minutes), and every new `max_len` is a new static value, which is part of the cache key and therefore forces a fresh compilation. The fix keeps `max_len` and `beam_size` static, because they determine buffer shapes and so genuinely are configuration, but replaces the Python loop with a compiled loop whose body is traced once. Option 1: `lax.while_loop` with a carry of fixed-shape buffers (tokens of shape `(beam_size, max_len)`, scores, a `done` mask, and the step `t`), a condition `(t < max_len) & ~jnp.all(done)` so it stops early on the device, and a body that writes the new tokens with `.at[:, t].set(...)`. Option 2: `lax.scan` over `max_len` steps that always runs to the end but freezes finished beams with `jnp.where(done, old, new)`; it is simpler when you may later want to differentiate through decoding, which `while_loop` does not support in reverse mode. Either way compile time no longer depends on `max_len`. To avoid recompiling for every length, pick a few bucketed `max_len` values (for example 32, 64, 128) or compile once for the largest and stop early. Finally, confirm the fix with `jax.config.update('jax_log_compiles', True)`: you should see one compile per (bucket, beam_size) pair and none during decoding."
sources: [{"title": "JAX docs: Just-in-time compilation", "url": "https://docs.jax.dev/en/latest/201/jit.html", "note": "Tracing, jaxprs, static arguments and the caching rules."}, {"title": "JAX docs: Control flow and logical operators with JIT", "url": "https://docs.jax.dev/en/latest/201/control-flow.html", "note": "cond, switch, fori_loop, while_loop and scan, and which are differentiable."}, {"title": "JAX docs: Common gotchas in JAX", "url": "https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html", "note": "Pure functions, side effects under jit and control-flow errors."}]
practice: ["static-arguments-in-jit", "traced-learning-rate-schedule", "exponential-moving-average-with-scan", "spectral-norm-by-power-iteration"]
tags: ["jit", "tracing", "lax control flow"]
tools: ["JAX", "XLA"]
---

## What you will learn

- What `jax.jit` does on a call: trace with stand-in values, record a jaxpr, compile with XLA, cache the program.
- Why the first call is slow, later calls are fast, and fusion makes the compiled code quicker.
- Why `print`, globals and Python `if` behave strangely under jit, and how to write jit-friendly code.
- How static arguments work and what they cost.
- How to write data-dependent branches and loops with `jnp.where` and `jax.lax`.
- How to spot and remove recompilation, the most common jit performance bug.

*Before you start:* you should be comfortable with `jax.numpy` arrays and pure functions (05.01) and with `jax.grad` (05.02). Knowing what `torch.compile` does (04.07) helps for contrast but is not required.

In lesson 05.02 you turned a loss into a gradient function with `jax.grad`, but each call still ran one small operation at a time from Python. This lesson adds `jax.jit`, which turns the same pure function into a compiled program. In PyTorch, compiling is an optimisation you add later; in JAX almost every serious piece of code (a training step, an evaluation loop, a sampler) runs under `jit`, so its rules shape how you write everything else.

## The idea in one picture

A tiny function with a few element-wise operations:

```python
import jax, jax.numpy as jnp

def f(x):
    y = jnp.sin(x) * 2.0
    return jnp.where(y > 0, y, 0.0) + x ** 2

fj = jax.jit(f)                         # same maths, compiled
print(fj(jnp.array([-1.0, 0.5, 2.0])))  # [1.        1.2088511 5.818595 ]
```

`jax.jit` does not change the answer, only how it is produced. Below, the jitted function is called eight times: four times on one million numbers, twice on half a million, then twice on a million again.

~~~dg-fig 05.03-calls.svg
Measured wall time of eight calls to the jitted function (median of 7 runs from an empty cache; timings are machine-dependent). Call 1 and call 5 are about ten times slower because they see a new input shape and must trace and compile. Every other call reuses a cached program.
~~~

On call 1, JAX has never seen this function with a float32 array of length 1,000,000, so it runs your Python once to learn what the function does, compiles that recipe, and stores the result: about 30 ms here. Calls 2 to 4 match, so JAX skips Python and runs the stored program in about 3 ms. Call 5 brings a new length, a new signature, so the process repeats. Calls 7 and 8 find the old program still in the cache.

Four terms carry the rest of the lesson:

- A **tracer** is a stand-in object JAX passes to your function instead of real data. It knows the shape and dtype (for example `float32[1000000]`) but not the values.
- A **jaxpr** (JAX expression) is the list of primitive operations recorded while the tracers flowed through your code.
- **XLA** is the compiler that turns the jaxpr into fast code for CPU, GPU or TPU, merging operations where it can.
- The **compilation cache** stores one compiled program per **signature**: the shapes, dtypes and structure of the inputs plus the values of any static arguments.

## Step 1: tracing, or what JAX sees

On a new signature, JAX replaces each array argument with a tracer and runs your Python once. A `jnp` call on a tracer computes nothing; it appends a line to the jaxpr, which you can print:

```python
print(jax.make_jaxpr(f)(jnp.ones(3)))
# { lambda ; a:f32[3]. let
#     b:f32[3] = sin a
#     c:f32[3] = mul b 2.0:f32[]
#     d:bool[3] = gt c 0.0:f32[]
#     e:f32[3] = jit[name=_where ...  select_n d h c ...] d c 0.0:f32[]
#     i:f32[3] = integer_pow[y=2] a
#     j:f32[3] = add e i
#   in (j,) }
```

Each line is one primitive: output variable, type (`f32[3]` is a float32 vector of length 3), operation and inputs. There is no Python left, only *what* was computed. `jnp.where` became a `select_n`, which picks element-wise between two arrays, and constants are folded: `x * (3.0 * 2.0)` appears as a single `mul a 6.0`, because Python computed `3.0 * 2.0` while tracing.

<div class="ex">
<div class="lab">Worked example</div>

**run the jaxpr by hand**

Take $x = (-1,\ 0.5,\ 2)$ and follow the jaxpr line by line, rounding to four decimals.

Line b, `sin a`: $\sin(-1) = -0.8415$, $\sin(0.5) = 0.4794$, $\sin(2) = 0.9093$.

Line c, `mul b 2.0`: $(-1.6829,\ 0.9589,\ 1.8186)$.

Line d, `gt c 0.0`: $(\text{False},\ \text{True},\ \text{True})$.

Line e, `select_n`: where d is False take 0, otherwise take c, giving $(0,\ 0.9589,\ 1.8186)$.

Line i, `integer_pow[y=2] a`: $(1,\ 0.25,\ 4)$.

Line j, `add e i`: $(0 + 1,\ 0.9589 + 0.25,\ 1.8186 + 4) = (1,\ 1.2089,\ 5.8186)$, which matches the printed output `[1. 1.2088511 5.818595]`.

The jaxpr is a complete recipe: for this signature JAX never needs your Python again.

</div>

## Step 2: compile once, then reuse

XLA now compiles the jaxpr. Its key trick is **fusion**: instead of six separate passes over memory (`sin`, `mul`, `gt`, `select`, `pow`, `add`), it emits one loop that reads each element once, does all six steps in registers, and writes one result. The program is cached under the signature and reused directly on the next matching call.

~~~dg-flow The life of a jitted call: a cache hit skips straight to running compiled code.
call f(x) -(cache hit)-> run compiled executable
call f(x) -(new signature)-> *trace with tracers -> jaxpr -> XLA compile -> cache -> run compiled executable
~~~

How much does fusion buy? The next figure times the same function eagerly (no `jit`, one operation at a time) and jitted, after compilation has already happened.

~~~dg-fig 05.03-fusion.svg
Steady-state time per call for the same function, eager versus jitted, on a 2-thread CPU (best of 25 calls; numbers are machine-dependent). The label on each point is the jit speed-up. For small arrays the eager cost is mostly per-operation dispatch from Python; for large arrays it is the extra passes over memory.
~~~

On small arrays eager mode spends its time on Python and dispatch overhead (six round trips for six operations), so jit wins by roughly 8x. On large arrays memory traffic dominates; the fused kernel touches the data once instead of six times and wins by roughly 3x. On a GPU both effects are usually larger.

One timing catch: JAX dispatches **asynchronously**, so a call returns once the work is queued. Finish timings with `.block_until_ready()` and time the first call separately, or you measure either nothing or mostly compilation.

## Step 3: Python runs only while tracing

Here is the rule that explains most jit surprises: **your Python code runs during tracing and never again for that signature.** Anything Python does that is not a `jnp` operation on the inputs happens once, at trace time.

```python
@jax.jit
def g(x):
    print("tracing with", x)          # Python print: runs at trace time only
    jax.debug.print("x = {}", x)      # staged into the program: runs every call
    return x * 2

g(1.0)          # tracing with JitTracer(~float32[])   then   x = 1.0
g(2.0)          # x = 2.0                 (no trace: cache hit)
g(jnp.ones(2))  # tracing with JitTracer(float32[2])   then   x = [1. 1.]
```

The Python `print` fires only when tracing and shows a tracer; `jax.debug.print` is recorded in the jaxpr and prints the runtime value every call. Globals follow the same rule, and here the result is silently wrong:

```python
scale = 2.0
@jax.jit
def gs(x): return x * scale
print(gs(1.0))   # 2.0
scale = 10.0
print(gs(1.0))   # 2.0   <- still the old value: it was baked in at trace time
```

During tracing `scale` was 2.0, so the jaxpr holds the constant 2.0. Changing the global does nothing until a new signature forces a retrace, and then behaviour changes without warning. Appending to a list or reading a config object inside a jitted function has the same problem. The fix: make the function **pure**, with every input an argument and every output a return value.

## Step 4: static arguments

Some arguments are configuration rather than data: a window size, a number of layers, a `training` flag. If such a value decides a shape or a Python branch, it must be a real Python value at trace time, so you mark it **static**:

```python
from functools import partial

@partial(jax.jit, static_argnames="window")
def pool(x, window):
    return x.reshape(-1, window).mean(axis=1)   # output shape depends on window

pool(jnp.arange(8.0), window=2)   # compiles:  [0.5 2.5 4.5 6.5]
pool(jnp.arange(8.0), window=4)   # compiles again: [1.5 5.5]
pool(jnp.arange(8.0), window=2)   # cache hit, no compile
```

A static value is part of the cache key. So it must be **hashable** (ints, strings, tuples, frozen dataclasses; a list raises `ValueError: Non-hashable static arguments are not supported`), it can be used freely in Python (`if window > 2:`, `range(window)`), and every new value triggers a fresh compile. `static_argnums=(1,)` marks arguments by position instead.

Static arguments buy Python flexibility and cost one compilation per value: a good deal for a few fixed configurations, a terrible one for anything that changes every step.

## Step 5: branches and loops on traced values

A Python `if` needs a real `True` or `False`. Under `jit`, `x > 0` is a tracer whose value is unknown at trace time, so JAX raises an error:

```python
@jax.jit
def h(x):
    if x > 0:          # TracerBoolConversionError:
        return x       # "Attempted boolean conversion of traced array with shape bool[]"
    return -x
```

Instead, move the decision into the compiled program:

| need | tool | notes
| --- | --- | --- |
| choose element by element | `jnp.where(c, a, b)` | computes both `a` and `b`, then selects
| run one of two functions | `lax.cond(pred, f_true, f_false, *ops)` | both branches are traced; outputs must match in shape and dtype
| run one of many functions | `lax.switch(index, [f0, f1, ...], *ops)` | same rules as `cond`
| loop a known number of times | `lax.fori_loop(lo, hi, body, init)` | compiled as a real loop, not unrolled
| loop until a condition fails | `lax.while_loop(cond, body, init)` | not reverse-mode differentiable
| loop and collect per-step outputs | `lax.scan(f, carry, xs)` | the workhorse: RNNs, moving averages, layer stacks

In code:

```python
from jax import lax

lax.cond(True, lambda v: v**2, lambda v: jnp.maximum(v, 0.), jnp.array([-2., 3.]))   # [4. 9.]
lax.fori_loop(0, 5, lambda i, acc: acc + i * i, 0)                                  # 30
lax.while_loop(lambda c: c[0] > 1.0, lambda c: (c[0] / 2, c[1] + 1), (100.0, 0))      # (0.78125, 7)

def ema(xs, decay=0.9):
    def step(m, x):
        m = decay * m + (1 - decay) * x
        return m, m                    # (new carry, output for this step)
    _, ms = lax.scan(step, jnp.zeros(xs.shape[1:]), xs)
    return ms

jax.jit(ema)(jnp.ones(4))   # [0.1    0.19   0.271  0.3439]
```

A **carry** is the state passed between iterations; it must keep its shape and dtype, so JAX loops use fixed-size buffers updated with `.at[i].set(...)`, never growing lists.

Python loops over a plain int are allowed, but they run at trace time and **unroll**: the body is copied into the jaxpr once per iteration. A 400-step Python loop over a two-line body gives a 1,602-line jaxpr; `fori_loop` gives 16 lines.

~~~dg-fig 05.03-unroll.svg
First-call (trace plus compile) time for a jitted loop, best of 5 cold compiles; timings are machine-dependent. The unrolled Python loop gets slower in proportion to its length; `fori_loop` traces its body once and stays flat at about 20 ms.
~~~

Unrolling a short fixed loop (say 4 iterations) is fine and can even run faster; for long loops it makes compile time grow with the length.

## Step 6: keep the cache warm

A program is reused only on an exact signature match, so the commonest jit performance bug is **recompilation**: rebuilding the program again and again. Typical causes:

- Batches of varying length. Pad to a few bucket sizes (lesson 05.01) so only a few shapes ever occur.
- A value that changes every step passed as a static argument, such as a step counter or a learning rate.
- Mixing Python scalars and arrays: `dbl(1.0)`, `dbl(jnp.float32(1.0))` and `dbl(1)` make three cache entries (weak float32, strong float32, int).
- Calling `jax.jit(lambda ...)` inside a loop: each lambda is a new function, so the cache never hits (20 calls took about 420 ms here, against 19 ms reusing one jitted function).

The figure shows the classic case: a training step with the step number static, then passed as an array.

~~~dg-fig 05.03-recompile.svg
Forty steps of the same update, computed two ways (timings are machine-dependent). Left: with a static step, every step pays a fresh compilation of tens of milliseconds; with a traced step only the first does. Right: the static version spends about 1.3 s in total against under 0.1 s, and both produce the same weights.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the learning-rate schedule that recompiled every step**

A training step uses `@partial(jax.jit, static_argnames="step")` so the schedule can say `if step < warmup:`. Training is slow, and `jax_log_compiles` prints a compile every step.

Diagnosis: each new `step` is a new static value, so each call compiles from scratch. Suppose compiling costs 50 ms and running a step 0.5 ms.

Static step, 1,000 steps: $1000 \times (50 + 0.5) = 50{,}500$ ms, about 50 s.

Traced step: one compile plus 1,000 runs, $50 + 1000 \times 0.5 = 550$ ms, about 0.55 s. That is $50{,}500 / 550 \approx 92$ times faster, and the gap grows with the number of steps.

Fix: pass `step` as an array and write the schedule with array operations: `lr = jnp.where(step < warmup, peak * step / warmup, cosine(step))`. One compilation now serves the whole run; Optax schedules work the same way.

</div>

## A jitted training step in code

A pure step, the schedule as array maths, the step counter passed as data:

```python
key = jax.random.key(0)
kx, kn = jax.random.split(key)
X = jax.random.normal(kx, (1000, 3))
y = X @ jnp.array([2.0, -1.0, 0.5]) + 0.1 * jax.random.normal(kn, (1000,))

def loss(w, X, y):
    return jnp.mean((X @ w - y) ** 2)

@jax.jit
def train_step(w, X, y, step):
    lr = jnp.where(step < 10, 0.1 * (step + 1) / 10, 0.1)   # warm-up, traced
    l, g = jax.value_and_grad(loss)(w, X, y)
    return w - lr * g, l

w = jnp.zeros(3)
for s in range(200):
    w, l = train_step(w, X, y, s)
print(w, float(l))                 # [ 1.998221  -1.0019286  0.5008956] 0.0110...
print(train_step._cache_size())    # 1   (one compilation for all 200 steps)
```

`jit` composes with `grad`: the gradient is traced into the same jaxpr, so the whole step is one compiled program. Put `jit` on the outermost function you call per step; nested `jit` calls simply become part of it.

## Checking that jit is doing what you think

- **Count compilations.** `jax.config.update("jax_log_compiles", True)` logs a line such as `Compiling jit(pool) with global shapes and types (ShapedArray(float32[8]),)` per build. A healthy loop shows it once (or once per bucket). `f._cache_size()` is a private debugging helper.
- **Read the trace.** In `jax.make_jaxpr(f)(x)`, far more lines than expected means a loop is unrolling; a constant where you expected a variable means a captured global or static value.
- **Read the compiled program.** `jax.jit(f).lower(x).compile().as_text()` shows the optimised program with its fused computations.
- **Time it honestly.** Warm up once, time with `.block_until_ready()`, compare with the un-jitted function.

## What goes wrong

| Symptom | Cause | Fix
| --- | --- | --- |
| `TracerBoolConversionError` or `ConcretizationTypeError` | Python `if`, `while`, `int()` or `bool()` on a traced value | use `jnp.where`, `lax.cond`, `lax.while_loop`; or make the argument static if it is true configuration
| A `print` shows `JitTracer(...)` and fires only once | Python side effects run at trace time | use `jax.debug.print` for runtime values
| Changing a global or config has no effect | value baked into the trace as a constant | pass it as an argument
| Every call is slow; compile log fires every step | new shapes, or a changing static argument | bucket shapes, pass changing values as arrays
| First call takes minutes | long Python loop unrolled into the trace | `lax.fori_loop` or `lax.scan`
| Timings look impossibly fast | async dispatch: you timed the queueing | call `.block_until_ready()`
| Gradient is NaN even though the forward value is fine | `jnp.where` ran the unused branch, e.g. `log(0)` | sanitise the input inside the branch (see below)

## Going deeper

**The cache key** holds the function object, array shapes and dtypes (including weak typing), the pytree structure (a dict with different keys is a new signature), static values, and sharding. Traced values are never in the key, which is why the program can be reused.

**Shapes must be static.** Tracers carry a `ShapedArray`: shape and dtype only. So a function whose *output shape* depends on input *values*, such as `x[x > 0]`, cannot be jitted as written; pass a static `size=` (as in `jnp.nonzero(x, size=10)`) or rewrite it with masks.

**`jnp.where` and gradients.** `where` evaluates both branches on every element. In the forward pass the unused branch is discarded, but in the backward pass its gradient is multiplied by zero, and $0 \times \infty$ is NaN. So `jnp.where(x > 0, jnp.log(x), 0.0)` has gradient NaN at $x = 0$. The standard fix is a double `where`: `jnp.log(jnp.where(x > 0, x, 1.0))` inside the outer `where`, which gives gradient 0.0 there.

**`cond` versus `where`.** `lax.cond` with a scalar predicate runs only one branch, which pays off for expensive branches. Under `vmap` (05.04) a batched predicate turns it into a `select`, so both run anyway.

**Gradients through loops.** `scan` and `fori_loop` with static bounds support reverse-mode autodiff; `while_loop` does not, since its trip count is unknown. For gradients through early stopping, `scan` over the maximum length with a `done` mask.

**Ahead of time.** `jax.jit(f).lower(x).compile()` returns a callable compiled object, separating compile time from run time. `jax.config.update("jax_compilation_cache_dir", path)` keeps programs on disk across restarts, and `donate_argnums` lets XLA reuse an input buffer for the output.

## Key terms

| Term | Meaning
| --- | --- |
| `jax.jit` | Transformation that traces a function once per signature and runs a compiled XLA program
| Tracer | Stand-in value with shape and dtype but no data, passed through your code during tracing
| jaxpr | The recorded list of primitive operations produced by tracing
| XLA | The compiler that turns a jaxpr into fused machine code for CPU, GPU or TPU
| Signature / cache key | Shapes, dtypes, pytree structure, static values and placement that select a compiled program
| Fusion | Merging several operations into one kernel so data is read and written once
| Static argument | An argument treated as a compile-time constant; hashable, usable in Python, one compile per value
| Unrolling | A Python loop under jit copied into the trace once per iteration
| `lax.scan` | Compiled loop with a carry that also stacks per-step outputs

<div class="take">
<div class="lab">Key takeaway</div>

`jit` runs your Python once per input signature to record a jaxpr, compiles it with XLA into fused code, and reuses that program for every later call with the same shapes and static values. Keep functions pure, make only true configuration static, express data-dependent branches and loops with `jnp.where` and `jax.lax`, and keep the set of input shapes small so you compile once and run many times.

</div>
