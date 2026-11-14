---
id: "04.07"
course: "ml-toolkit"
module: "04"
slug: "checkpointing-mixed-precision-and-speed"
title: "Checkpointing, mixed precision and speed"
keyIdeas: ["A resumable checkpoint is more than weights: model, optimiser, scheduler and `GradScaler` state dicts, the step and epoch counters, and RNG states. Missing any one makes the resumed run diverge from the uninterrupted one.", "`torch.autocast` runs matmuls and convolutions in float16 or bfloat16 while keeping precision-sensitive ops (softmax, norms, losses) in float32. The weights themselves stay float32.", "float16 has a narrow range (largest finite value 65504, smallest normal about $6 \\times 10^{-5}$), so small gradients underflow; `GradScaler` multiplies the loss by a large factor and skips steps that overflow. bfloat16 has float32's range and does not need it.", "`torch.compile(model)` captures the forward into graphs and generates fused kernels. The first call is slow while it compiles; new input shapes or Python-side changes trigger recompiles or graph breaks."]
question: "You switch a working fp32 training script to fp16 autocast with `GradScaler`, and add gradient clipping at max_norm 1.0. Training is now much slower to converge, and the logged gradient norm before clipping is around 30,000 every step. What is the bug, why does it slow convergence, and why would the same code be fine with bfloat16 and the scaler disabled?"
answer: "The gradients are clipped while still multiplied by the loss scale. `scaler.scale(loss).backward()` produces gradients S times too large (S is about 2^15 to 2^16 here, consistent with a logged norm near 30,000 for a true norm below 1), and clipping them to norm 1 then shrinks every update by a factor of roughly S/true-norm. `scaler.step()` later divides by S again, so the effective step is tiny: convergence slows as if the learning rate were cut by orders of magnitude. The fix is `scaler.unscale_(opt)` before `clip_grad_norm_`, so clipping sees true gradient magnitudes, and `scaler.step` knows not to unscale twice. With bfloat16 and `enabled=False`, `scale()` and `unscale_()` are identities, so the gradients are never scaled and clipping already sees true values."
sources: [{"title": "PyTorch docs: Automatic mixed precision (torch.amp)", "url": "https://pytorch.org/docs/stable/amp.html", "note": "Autocast op lists (which ops run in 16-bit, which in fp32) and GradScaler's parameters."}, {"title": "PyTorch docs: AMP examples", "url": "https://pytorch.org/docs/stable/notes/amp_examples.html", "note": "Correct patterns for clipping, accumulation and multiple losses with a GradScaler."}, {"title": "PyTorch tutorial: Saving and loading a general checkpoint", "url": "https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html", "note": "Bundling model, optimiser and counters into one resumable file."}, {"title": "PyTorch tutorial: Introduction to torch.compile", "url": "https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html", "note": "Usage, modes, graph breaks and how to read compile behaviour."}]
practice: ["mixed-precision-training-memory-accountant", "mixed-precision-with-loss-scaling", "rmsnorm-forward-pass", "mini-batch-training-loop"]
tags: ["checkpointing", "mixed precision", "torch.compile"]
tools: ["PyTorch"]
---

## Checkpoints you can actually resume from

Saving `model.state_dict()` is enough to *use* a model. To *resume training* so that the result matches an uninterrupted run, you need every piece of state that the next step reads:

```python
ckpt = {
    "model": model.state_dict(),
    "optim": opt.state_dict(),        # Adam moments and step counts
    "sched": sched.state_dict(),      # position in warmup/cosine
    "scaler": scaler.state_dict(),    # current loss scale (fp16 only)
    "step": step, "epoch": epoch, "best_val": best,
    "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
}
torch.save(ckpt, "ckpt.tmp"); os.replace("ckpt.tmp", "ckpt.pt")   # atomic swap
```

Loading reverses it: build the model and optimiser exactly as before, move the model to its device, then call each `load_state_dict`. Use `map_location` to load a GPU checkpoint on a CPU machine. Since PyTorch 2.6 `torch.load` defaults to `weights_only=True`, which unpickles only tensors and plain containers; that is a security feature (a pickle can run arbitrary code), so keep checkpoints to tensors, numbers and strings rather than lowering the guard.

Practical habits: write to a temporary file and rename, so a crash mid-save never corrupts the last good checkpoint; keep the last two or three plus the best; and for a model wrapped by `torch.compile` or DDP, save the underlying module's state dict so key names have no `_orig_mod.` or `module.` prefix. Resuming mid-epoch exactly also requires restoring the sampler's position, which most loops approximate by restarting the epoch.

## Mixed precision with autocast

Most training time goes into matmuls, and tensor cores run them several times faster in 16-bit formats. Automatic mixed precision gets that speed without converting the whole model:

```python
use_fp16 = False                         # True on GPUs without bfloat16 support
dtype = torch.float16 if use_fp16 else torch.bfloat16
scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
for x, y in loader:
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=dtype):
        loss = loss_fn(model(x), y)      # matmuls in 16-bit, softmax/norm/loss in fp32
    scaler.scale(loss).backward()        # identity when the scaler is disabled
    scaler.unscale_(opt)                 # so clipping sees true gradient sizes
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                     # skips the step if grads contain inf/nan
    scaler.update()                      # adjust the scale for next time
```

Autocast keeps an op list: `matmul`, `linear`, `conv` and attention run in the low-precision dtype; reductions and numerically sensitive ops such as `softmax`, `log_softmax`, `layer_norm` and losses run in float32. Parameters stay float32 (the "master weights"); autocast casts them on the fly. Only the forward pass goes inside the context; backward ops run in the dtype their forward ops used.

~~~dg-flow Dynamic loss scaling with float16.
loss -(times S)-> scaled backward -(unscale by 1/S)-> *finite? -(yes)-> step, grow S every 2000 steps
~~~

## Why float16 needs a scaler and bfloat16 does not

float16 has 5 exponent bits: its largest finite value is 65504 and its smallest normal value about $6.1 \times 10^{-5}$ (subnormals reach about $6 \times 10^{-8}$). Activation gradients in deep networks are often smaller than that, so they flush to zero. `GradScaler` multiplies the loss by a scale $S$ (initially $2^{16}$), which multiplies every gradient by $S$ and lifts them into range; it divides by $S$ before the step. If any gradient overflowed to inf or nan, it skips the step and halves $S$; after 2,000 consecutive good steps it doubles $S$. A few skipped steps early in training are normal; persistent skipping signals a real numerical problem.

bfloat16 keeps float32's 8 exponent bits (range up to about $3.4 \times 10^{38}$) and gives up mantissa: 8 significant bits, roughly 2 to 3 decimal digits. Underflow is not the issue, so no scaler is needed; the cost is coarse rounding, which is why weights and optimiser state stay in float32. On GPUs that support it (A100 and later), bfloat16 is the default choice. The LLM-scale module in the AI Research course goes further, into fp8 and memory budgets.

## torch.compile basics

`model = torch.compile(model)` hands the forward pass to a compiler stack: TorchDynamo captures the Python code into FX graphs by inspecting bytecode, AOTAutograd derives the backward graph ahead of time, and TorchInductor generates fused kernels (Triton on GPU). Fusing chains of elementwise ops (bias, activation, dropout, residual add) removes round trips to memory, which is where much of the speed-up comes from: typically tens of percent for Transformer training, more for models dominated by small ops.

What to expect:

- The first call compiles and can take tens of seconds to minutes; benchmark only after warmup.
- Code Dynamo cannot trace (data-dependent Python control flow, `.item()`, printing, some third-party calls) causes a **graph break**: the model still runs, with the untraceable part in eager mode, but fusion stops at the break.
- A new input shape triggers a recompile; after it sees varying shapes Dynamo tries a dynamic-shape graph. Padding to a few fixed lengths keeps recompiles rare.
- `mode="reduce-overhead"` adds CUDA graphs to cut kernel-launch cost for small batches; `mode="max-autotune"` searches kernel configurations for longer at compile time.

| lever | typical gain | catch
| --- | --- | --- |
| bf16 autocast | 2x or more on matmul-heavy models | coarse rounding; keep fp32 master weights
| `torch.compile` | 1.2x to 2x | compile time, graph breaks, recompiles
| `channels_last` for CNNs | 1.1x to 1.3x | conv nets only
| larger batch, fewer syncs | varies | memory, optimisation dynamics

<div class="ex">
<div class="lab">Worked example</div>

**the resumed run that drifted**

A run is stopped at step 40,000 and resumed from a checkpoint containing only model and optimiser state.

The scheduler restarts at step 0, so the learning rate jumps back through warmup to its peak: the loss spikes. The fp16 loss scale restarts at 65536 when it had settled at 1024, so the first few steps overflow and are skipped. The data order also restarts from epoch 0.

Saving and restoring the scheduler, scaler, step counter and RNG states makes the resumed loss curve sit on top of the original.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

checkpoint everything the next step reads, write it atomically, use bfloat16 autocast with float32 master weights where hardware allows (fp16 plus GradScaler otherwise), and compile once the model is correct, not before.

</div>
