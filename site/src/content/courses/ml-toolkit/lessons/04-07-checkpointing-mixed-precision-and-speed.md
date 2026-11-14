---
id: "04.07"
course: "ml-toolkit"
module: "04"
slug: "checkpointing-mixed-precision-and-speed"
title: "Checkpointing, mixed precision and speed"
keyIdeas: ["A resumable checkpoint is more than weights: model, optimiser, scheduler and `GradScaler` state dicts, the step and epoch counters, and RNG states. Missing any one makes the resumed run diverge from the uninterrupted one; write it to a temporary file and rename so a crash mid-save cannot corrupt it.", "float16 and bfloat16 spend 16 bits differently: float16 keeps more precision but its range ends at 65504 and tiny values round to zero below about $3 \\times 10^{-8}$; bfloat16 keeps float32's range with only about 2 to 3 significant digits.", "`torch.autocast` runs matmuls and convolutions in 16 bits while keeping precision-sensitive ops (softmax, norms, losses) in float32. The weights stay float32 (master weights) so small updates are not rounded away.", "float16 needs `GradScaler`: it multiplies the loss by a scale $S$ (starting at $2^{16}$) so small gradients survive, skips steps whose gradients overflow and halves $S$, and doubles $S$ after 2000 clean steps. Call `unscale_` before clipping. bfloat16 does not need it.", "`torch.compile(model)` captures the forward into graphs and generates fused kernels. The first call is slow while it compiles; new input shapes or untraceable Python cause recompiles or graph breaks, so benchmark only after warmup."]
question: "You switch a working fp32 training script to fp16 autocast with `GradScaler`, and add gradient clipping at max_norm 1.0. Training is now much slower to converge, and the logged gradient norm before clipping is around 30,000 every step. What is the bug, why does it slow convergence, and why would the same code be fine with bfloat16 and the scaler disabled?"
answer: "The gradients are clipped while still multiplied by the loss scale. `scaler.scale(loss).backward()` produces gradients $S$ times too large ($S$ is about $2^{15}$ to $2^{16}$ here, consistent with a logged norm near 30,000 for a true norm below 1). `clip_grad_norm_` sees a norm of 30,000, far above 1.0, and multiplies every gradient by about 1/30,000 to bring the (scaled) norm down to 1. Then `scaler.step()` divides by $S$ as usual, so the update is computed from a gradient of norm about $1/S \\approx 1.5 \\times 10^{-5}$ instead of the true norm (say 0.5, which would not have been clipped at all). Every update is shrunk by a factor of tens of thousands, exactly as if the learning rate had been cut by four orders of magnitude, so convergence crawls. The fix is to call `scaler.unscale_(opt)` before `clip_grad_norm_`, so clipping sees true gradient magnitudes; `scaler.step` remembers that the gradients were already unscaled and does not divide again. The same fix applies to any code that logs or inspects gradient norms. With bfloat16 and `GradScaler(enabled=False)`, `scale()` returns the loss unchanged and `unscale_()` does nothing, so the gradients are never scaled and clipping already sees true values; bfloat16's float32-sized exponent range means small gradients do not underflow, which is why no scaling is needed in the first place."
sources: [{"title": "PyTorch docs: Automatic mixed precision (torch.amp)", "url": "https://pytorch.org/docs/stable/amp.html", "note": "Autocast op lists (which ops run in 16-bit, which in fp32) and GradScaler's parameters."}, {"title": "PyTorch docs: AMP examples", "url": "https://pytorch.org/docs/stable/notes/amp_examples.html", "note": "Correct patterns for clipping, accumulation and multiple losses with a GradScaler."}, {"title": "PyTorch tutorial: Saving and loading a general checkpoint", "url": "https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html", "note": "Bundling model, optimiser and counters into one resumable file."}, {"title": "PyTorch tutorial: Introduction to torch.compile", "url": "https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html", "note": "Usage, modes, graph breaks and how to read compile behaviour."}]
practice: ["mixed-precision-training-memory-accountant", "mixed-precision-with-loss-scaling", "rmsnorm-forward-pass", "mini-batch-training-loop"]
tags: ["checkpointing", "mixed precision", "torch.compile"]
tools: ["PyTorch"]
---

## What you will learn

- What a checkpoint must contain so a resumed run continues exactly, and how to write it safely.
- How float32, float16 and bfloat16 differ, and why range and precision are separate problems.
- What `torch.autocast` does: which ops drop to 16 bits, which stay in float32, and why weights stay float32.
- Why float16 needs a `GradScaler`, how dynamic loss scaling works, and why bfloat16 does not.
- What `torch.compile` does, what it costs up front, and how to avoid recompiles.

*Before you start:* you should be comfortable with the training loop from lesson 04.06 (zero grads, forward, backward, step), with optimisers and schedulers from 04.04, and with dtypes and devices from 04.01.

Lesson 04.06 built a correct training loop. This one makes it survive crashes (checkpoints), fit in memory and run fast (mixed precision), and waste less time on overhead (compilation). None of these changes what the model learns; all of them change whether you can afford to train it.

## The idea in one picture

Imagine a 400-step training run that crashes at step 200. You restart it from a saved file. Three things could happen:

| What you saved | What happens after the restart
| --- | --- |
| nothing | start again from step 0; 200 steps of work lost
| model weights + optimiser state | training continues, but the learning-rate schedule and data order restart
| everything the next step reads | the run continues as if the crash never happened

We ran this on a small network with a warmup-then-cosine schedule:

~~~dg-fig 04.07-resume.svg
Top: the learning rate. Bottom: the training loss. The full-checkpoint resume (dashed) lies exactly on top of the uninterrupted run; the run that saved only weights and optimiser re-warms its learning rate to the peak and its loss drifts above the reference.
~~~

In the top panel, the run that saved only weights and optimiser built a fresh scheduler on restart, which believes it is at step 0: the learning rate drops to almost nothing, then climbs back through warmup to its peak, where the uninterrupted run had already decayed it to half. Below, that run's loss sits above the reference. The dashed full-checkpoint run is not just close: its final weights are bit-for-bit equal (`torch.equal` returns `True`).

Two terms for the rest of the lesson:

- A **state dict** is a plain Python dictionary of tensors and numbers that describes an object's state. Models, optimisers, schedulers and gradient scalers all have `state_dict()` and `load_state_dict()`.
- A **checkpoint** is one file bundling several state dicts plus counters, saved periodically so training can resume.

## Step 1: list everything the next step reads

To resume exactly, ask: "what does the next training step read?" Every answer must be in the checkpoint.

- **Model weights and buffers** (`model.state_dict()`): the parameters, plus buffers such as BatchNorm running means.
- **Optimiser state** (`opt.state_dict()`): for Adam, two moment tensors per parameter and a step count. Without them Adam restarts with zero momentum and a burst of oversized updates.
- **Scheduler state** (`sched.state_dict()`): the position in the warmup and decay curve. This is what the figure above was missing.
- **GradScaler state** (only for float16): the current loss scale, discussed in step 4.
- **Counters**: global step, epoch, best validation score so far (for early stopping).
- **Random number generator states**: the global torch RNG drives dropout and random augmentations, and a `torch.Generator` (or the sampler's seed) drives the data order. On a GPU, also the CUDA RNG states.

The optimiser dominates the size: 100 million float32 parameters are 400 MB, Adam's two moments add 800 MB, so the checkpoint is about 1.2 GB, three times the weights-only file.

## Step 2: write it atomically, load it in the right order

A crash *during* saving is common (time limit, full disk), and writing straight over your only checkpoint leaves a half-written file. Write to a temporary file and rename it: a rename within one filesystem is **atomic**, so readers see either the old file or the new one, never a mixture.

```python
import os, torch

def save_checkpoint(path, model, opt, sched, step, gen):
    state = {
        "model": model.state_dict(),
        "optim": opt.state_dict(),
        "sched": sched.state_dict(),
        "step": step,
        "rng": {"torch": torch.get_rng_state(), "data": gen.get_state()},
    }
    if torch.cuda.is_available():
        state["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)          # atomic swap

def load_checkpoint(path, model, opt, sched, gen):
    ck = torch.load(path, map_location="cpu")  # weights_only=True by default
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optim"])
    sched.load_state_dict(ck["sched"])
    torch.set_rng_state(ck["rng"]["torch"])
    gen.set_state(ck["rng"]["data"])
    return ck["step"]
```

Build the model and optimiser exactly as before, move the model to its device, and only then call each `load_state_dict`. Optimiser state is matched to parameters by position, so a changed model definition silently pairs the wrong moments with the wrong weights. `map_location="cpu"` opens a GPU checkpoint on a laptop.

Since PyTorch 2.6, `torch.load` defaults to `weights_only=True`. A checkpoint is a pickle, and unpickling can run arbitrary code, so the default only rebuilds tensors, numbers, strings and plain containers. Keep checkpoints in that vocabulary (convert NumPy RNG states to plain ints and lists) rather than switching the guard off.

Here is the check we ran: train 100 steps straight, versus 60 steps, save, rebuild everything from scratch, load, and finish.

```python
m, opt, sched = build(); g = torch.Generator().manual_seed(1)
print(train(m, opt, sched, g, 0, 100))      # 0.6388404369354248
ref = m[0].weight.clone()

m, opt, sched = build(); g = torch.Generator().manual_seed(1)
train(m, opt, sched, g, 0, 60)
save_checkpoint("ckpt.pt", m, opt, sched, 60, g)
m, opt, sched = build(); g = torch.Generator()    # pretend: new process
start = load_checkpoint("ckpt.pt", m, opt, sched, g)
print(start, train(m, opt, sched, g, start, 100))  # 60 0.6388404369354248
print(torch.equal(m[0].weight, ref))                # True
```

The model includes dropout, so the identical result depends on the restored RNG. Keep the last two or three checkpoints plus the best; for a model wrapped by `torch.compile` or `DistributedDataParallel`, save the *inner* module so keys have no `_orig_mod.` or `module.` prefix.

## Step 3: understand the number formats

Most training time goes into matrix multiplications, which GPU tensor cores run several times faster in 16-bit formats; 16-bit numbers also halve memory and bandwidth. The catch: 16 bits must be shared between two jobs.

~~~dg-fig 04.07-formats.svg
How each format spends its bits. Exponent bits decide how large and how small a number can be; mantissa bits decide how many significant digits it keeps.
~~~

A floating-point number is stored as sign, **exponent** (which power of two) and **mantissa** (the significant digits). More exponent bits give a wider **range**; more mantissa bits give finer **precision**. `torch.finfo` reports the consequences:

```python
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(dt, fi.max, fi.tiny, fi.eps)
# torch.float32   3.4028e+38  1.1755e-38  1.1921e-07
# torch.float16   65504.0     6.1035e-05  0.0009765625
# torch.bfloat16  3.3895e+38  1.1755e-38  0.0078125
```

`max` is the largest finite value, `tiny` the smallest *normal* positive value, and `eps` the gap between 1 and the next number (relative precision). float16 keeps about 3 decimal digits but has a small range: above 65504 is `inf`, below $6.1 \times 10^{-5}$ come **subnormal** numbers that lose digits, and below about $3 \times 10^{-8}$ (half the smallest subnormal) everything rounds to zero. bfloat16 is the opposite trade: float32's range, only 8 significant bits. Cast a few values:

```python
t = torch.tensor([1e-5, 2e-8, 7e4, 1.001])
print(t.half().tolist())      # [1.00136e-05, 0.0, inf, 1.0009765625]
print(t.bfloat16().tolist())  # [1.00136e-05, 2.00234e-08, 70144.0, 1.0]
```

float16 loses the tiny value and overflows the large one; bfloat16 keeps both but cannot tell 1.001 from 1.

## Step 4: autocast, and why weights stay in float32

**Automatic mixed precision** (AMP) uses 16 bits where it is safe and fast, and 32 bits where it is not. `torch.autocast` is a context manager that, for each operation, looks up an op list:

- **Low-precision list**: `matmul`, `linear`, convolutions and attention run in float16 or bfloat16, because they are where the time goes and they tolerate rounding.
- **float32 list**: on CUDA, numerically sensitive ops such as `softmax`, `log_softmax`, `layer_norm`, `exp`, sums and losses are computed in float32, because they add up many numbers or exaggerate small differences.
- Everything else runs in whatever dtype its inputs already have.

The op lists differ by device; on CPU, for example, `log_softmax` stays in bfloat16 while losses are float32:

```python
x, y = torch.randn(8, 4), torch.randint(0, 3, (8,))
lin = torch.nn.Linear(4, 3)
with torch.autocast("cpu", dtype=torch.bfloat16):
    out = lin(x)
    loss = torch.nn.functional.cross_entropy(out, y)
print(out.dtype, lin.weight.dtype, loss.dtype)
# torch.bfloat16 torch.float32 torch.float32
```

The parameters stay float32: these are the **master weights**. Autocast casts a 16-bit copy on the fly for each matmul, but the optimiser updates the float32 originals, because a typical update is tiny compared with the weight it changes.

~~~dg-fig 04.07-stagnation.svg
A weight of 1.0 receiving 500 small updates. Stored in bfloat16 it never moves, because each update is smaller than half the gap between 1 and the next bfloat16 number. With the smaller update, float16 stalls too. float32 tracks the true sum.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**an update that rounds away**

A weight is $w = 1.0$ and the optimiser wants to add $\Delta = 0.001$ (learning rate times gradient).

In bfloat16, numbers between 1 and 2 are spaced $2^{-7} = 0.0078125$ apart, so the two neighbours of the true answer 1.001 are 1.0 and 1.0078125.

1.001 is $0.001$ from 1.0 and $0.0068$ from 1.0078, so it rounds to 1.0. The update is lost every step: after 500 steps the weight is still 1.0 (left panel).

In float16 the spacing is $2^{-10} \approx 0.000977$, so 1.001 rounds to 1.000977: the update survives almost intact, and after 500 steps the weight is 1.488 instead of 1.5.

With $\Delta = 0.0002$, below half of float16's spacing ($0.000488$), float16 also stays at 1.0 while float32 reaches 1.1 (right panel).

Conclusion: compute in 16 bits, but accumulate updates in float32.

</div>

Only the forward pass (and the loss) belongs inside the `autocast` block. Call `backward()` outside it; each backward op automatically runs in the dtype its forward op used.

## Step 5: float16 needs loss scaling

Range is float16's problem. Gradients in deep networks are often far smaller than $6 \times 10^{-5}$. We measured every weight gradient of an 8-layer sigmoid network:

~~~dg-fig 04.07-underflow.svg
Real gradient sizes on a log scale. A quarter of them sit in the region that float16 rounds to zero. Multiplying the loss by S = 65536 multiplies every gradient by the same factor and slides the whole histogram into the safe range.
~~~

Cast to float16, 25% of those gradients become exactly 0 and most others are subnormal. The fix uses linearity: multiply the loss by a constant $S$ and the chain rule multiplies every gradient by $S$, lifting them into range. Before the optimiser step, divide by $S$ in float32. With $S = 65536$, only 0.003% still underflow.

The difficulty is choosing $S$: too small and gradients underflow, too large and some overflow to `inf`. **Dynamic loss scaling**, which `torch.amp.GradScaler` implements, finds it automatically:

~~~dg-flow One GradScaler step with float16.
loss -(multiply by S)-> scaled backward -(divide grads by S)-> *any inf or nan? -(no)-> optimiser step; double S after 2000 clean steps
~~~

1. Start with $S = 2^{16} = 65536$ (`init_scale`).
2. If any gradient is `inf` or `nan`, **skip** the step and halve $S$.
3. Otherwise unscale and step; after 2000 clean steps (`growth_interval`), double $S$.

<div class="ex">
<div class="lab">Worked example</div>

**one overflow and one rescue**

Take $S = 65536$. A gradient element whose true value is $2 \times 10^{-8}$ would be 0 in float16. Scaled, it is $2 \times 10^{-8} \times 65536 = 1.31 \times 10^{-3}$, comfortably representable. After the backward pass the scaler divides by 65536 in float32 and recovers $1.9994 \times 10^{-8}$, within 0.03% of the truth.

Now take a gradient element whose true value is 2.0. Scaled, it is $131072$, above float16's maximum of 65504, so it becomes `inf`. The scaler sees the `inf`, skips this step and sets $S = 32768$.

Next step: $2 \times 32768 = 65536$, still above 65504, skip again, $S = 16384$. Then $32768$ fits. Two skipped steps, no damage.

</div>

Here is the scale from a real `GradScaler` run on CPU float16 (we set `growth_interval=20` instead of 2000 so the pattern fits in 300 steps):

~~~dg-fig 04.07-scaler.svg
Left: a healthy run. The scale drops fast in the first steps, then climbs in a staircase as gradients shrink, with an occasional skip when a doubling overshoots. Right: a run with too high a learning rate. Every step overflows, so the scale halves forever.
~~~

In the healthy run 16 of 300 steps were skipped, most in the first dozen while the scale fell to a size that fits, a few later when a doubling overshot. In the diverging run the weights themselves blow up, so the overflows never stop and the scale collapses. The rule: occasional skips are normal; skips that never stop mean the forward pass itself produces `inf`, which no scaler can fix.

**bfloat16 does not need any of this.** With float32's 8 exponent bits, tiny gradients stay representable; its coarse rounding is handled by master weights. On GPUs that support it (NVIDIA A100 and later), it is the default choice.

## Mixed precision in code

The full pattern for a GPU. It handles both dtypes: with bfloat16 the scaler is disabled, and `scale`, `unscale_` and `step` become plain pass-throughs.

```python
use_fp16 = False                          # True on GPUs without bfloat16
dtype = torch.float16 if use_fp16 else torch.bfloat16
scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

for x, y in loader:
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=dtype):
        loss = loss_fn(model(x), y)       # matmuls 16-bit, softmax/norm/loss fp32
    scaler.scale(loss).backward()         # gradients multiplied by S
    scaler.unscale_(opt)                  # divide by S BEFORE clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                      # skipped if any grad is inf/nan
    scaler.update()                       # halve S, or double after 2000 clean steps
```

`unscale_` must come before anything that looks at gradient sizes (clipping, logging the norm). And the scaler's `state_dict()` belongs in the checkpoint: restart at $S = 65536$ when the run had settled at 1024, and the first several steps are skipped.

## Step 6: torch.compile

In eager mode PyTorch launches one kernel per operation (matmul, bias add, GELU, dropout), and each elementwise kernel reads its whole input from memory and writes its output back. `torch.compile` removes much of that overhead:

1. **TorchDynamo** reads your forward function's Python bytecode and captures the tensor operations into a graph.
2. **AOTAutograd** derives the matching backward graph ahead of time.
3. **TorchInductor** generates **fused kernels**: one kernel that does bias, GELU and dropout in a single pass over memory (Triton code on GPU, C++ on CPU).

```python
model = torch.compile(model)              # returns a wrapped module
for step in range(25):
    t0 = time.perf_counter()
    model(xb).sum().backward()
    print(time.perf_counter() - t0)
# first call ~12 s (compiling), later calls ~6 ms here; machine-dependent
```

~~~dg-fig 04.07-compile.svg
Measured on this shared 2-thread CPU for a small MLP: the first compiled call spends about 12 seconds compiling forward and backward; after that the compiled version runs about 6 ms per call, slightly faster than eager. Timings are machine-dependent.
~~~

On our small CPU model the steady-state gain was modest (about 5.7 ms versus 6.5 ms); on GPUs gains of 1.2 to 2 times are typical for Transformer training. The shape is always the same: pay once, then benefit, so benchmark only after warmup.

Things that interrupt compilation:

- A **graph break** happens at code Dynamo cannot trace (`.item()`, `print`, Python `if`s on tensor values): that part runs eagerly and fusion stops there.
- A **recompile** happens when an input shape changes; padding to a few fixed lengths keeps them rare.
- `mode="reduce-overhead"` adds CUDA graphs to cut launch cost for small batches; `mode="max-autotune"` searches kernel configurations longer at compile time.

| Lever | Typical gain on GPU | Catch
| --- | --- | --- |
| bf16 autocast | 2x or more on matmul-heavy models | coarse rounding; keep fp32 master weights
| `torch.compile` | 1.2x to 2x | compile time, graph breaks, recompiles
| `channels_last` memory format | 1.1x to 1.3x | conv nets only
| larger batch, fewer CPU-GPU syncs | varies | memory, changes optimisation dynamics

## Checking that it worked

- **Checkpoints**: test resumption once, early: N steps straight versus N/2, save, load, N/2. Compare loss curves and the learning rate just before and after loading.
- **Mixed precision**: the loss should track a float32 run with the same seed closely (not identically). Log `scaler.get_scale()`: it should settle or rise, not keep halving.
- **Speed**: measure steps per second after warmup; on a GPU call `torch.cuda.synchronize()` before reading the clock, since kernels run asynchronously.
- **Compile**: run with `TORCH_LOGS="graph_breaks,recompiles"` to list every break and recompile with its reason.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| loss jumps right after a resume | scheduler or scaler state not saved | save and load every state dict
| resumed run sees the same batches as epoch 0 | data RNG or sampler position not restored | save the generator state or seed per epoch
| `load_state_dict` complains of `_orig_mod.` or `module.` keys | saved the compiled or DDP wrapper | save `model._orig_mod` / `model.module`
| loss becomes nan with fp16, scale keeps halving | overflow in the forward pass | use bf16, lower the LR, check for large logits
| fp16 run converges far slower, grad norm about 30,000 | clipping before `unscale_` | call `scaler.unscale_(opt)` first
| weights stop changing in a pure bf16 model | updates rounded away | keep float32 master weights and optimiser state
| compiled model is slow every step | recompiling on new shapes | pad to fixed shapes, `dynamic=True`

## Going deeper

**Why the scaler's division is safe.** $S$ is a power of two, so multiplying by $S$ or $1/S$ only changes the exponent, never the mantissa: it adds no rounding error as long as nothing overflows or underflows.

**Where the memory saving comes from.** With float32 master weights, parameter memory does not shrink: still 4 bytes of weight, 4 of gradient and 8 of Adam moments per parameter. The saving is in **activations** (outputs stored for backward), which halve and dominate for large batches. fp8 and sharded optimiser state are covered in the LLM-scale module of the AI Research course.

**Where precision errors accumulate.** Summing many numbers in 16 bits loses digits quickly, so tensor cores multiply in 16 bits but accumulate dot products in float32, and autocast keeps reductions in float32. Rounding is harmless once, harmful when its error is added up thousands of times.

**Determinism.** Even with every RNG restored, GPU kernels using atomic adds can sum in a different order each run, so resumed runs may differ in the last bits; `torch.use_deterministic_algorithms(True)` forces reproducible kernels at some speed cost.

**What Dynamo guards on.** Each compiled graph carries **guards**: cheap checks on input shapes, dtypes, devices and specialised Python values. If a guard fails, Dynamo recompiles, up to a limit, then falls back to eager, so many recompiles show up as a slow model, not an error.

## Key terms

| Term | Meaning
| --- | --- |
| State dict | A dictionary of tensors and numbers capturing one object's state (model, optimiser, scheduler, scaler)
| Checkpoint | One file bundling all state dicts, counters and RNG states needed to resume training
| Atomic save | Write to a temporary file, then rename, so a crash never leaves a half-written checkpoint
| Exponent / mantissa | The bits that set a float's range / its precision
| Autocast | A context that runs each op in 16 or 32 bits according to an op list
| Master weights | The float32 copy of parameters that the optimiser updates
| Loss scaling | Multiplying the loss by $S$ so small float16 gradients do not round to zero
| Graph break | A point where `torch.compile` stops tracing and falls back to eager Python

<div class="take">
<div class="lab">Key takeaway</div>

checkpoint everything the next step reads and write it atomically; compute in 16 bits with `autocast` but keep float32 master weights; use bfloat16 when the hardware has it and float16 plus `GradScaler` (with `unscale_` before clipping) otherwise; compile once the model is correct, and judge speed only after warmup.

</div>
