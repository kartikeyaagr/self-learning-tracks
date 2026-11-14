---
id: "04.06"
course: "ml-toolkit"
module: "04"
slug: "the-training-loop"
title: "The training loop"
keyIdeas: ["Every PyTorch training step is the same five lines: zero gradients, forward, loss, backward, step. Everything else in a real loop (clipping, accumulation, scheduling, logging, evaluation, checkpoints) is inserted at fixed points around them.", "`model.train()` and `model.eval()` switch layer behaviour (dropout on or off, batch norm using batch statistics or running ones). They do not touch autograd, so evaluation also needs `torch.no_grad()`.", "Clip after `backward()` and before `step()` with `clip_grad_norm_`, and log the norm it returns: the gradient norm is one of the most informative curves you can plot.", "Validate on a fixed held-out set at a fixed cadence, keep the best checkpoint by validation metric, and stop when it has not improved for `patience` evaluations."]
question: "Your validation accuracy changes by about 1 point every time you re-run evaluation on the same checkpoint, and after the first evaluation the training loss starts falling faster than before. The model uses dropout and batch norm. What is wrong with the loop and what do you change?"
answer: "Evaluation is running in training mode. With dropout active, every evaluation samples a different subnetwork, so accuracy varies between identical re-runs; in eval mode the forward pass is deterministic. Batch norm in training mode also normalises with the validation batches' own statistics and updates its running averages with validation data. The faster-falling training loss after evaluation is the second half of the bug: either evaluation switched to eval mode and never switched back, so training continues with dropout disabled and fits the training data more easily, or the running statistics were contaminated. The fix is a helper that calls `model.eval()` and runs under `torch.no_grad()`, then restores `model.train()` before returning, as in the evaluate function in this lesson."
sources: [{"title": "PyTorch tutorial: Optimizing model parameters", "url": "https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html", "note": "The official minimal train and test loops, with the step order spelled out."}, {"title": "PyTorch docs: clip_grad_norm_", "url": "https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html", "note": "The global-norm definition, the returned value and the error_if_nonfinite option."}, {"title": "Karpathy: A recipe for training neural networks (2019)", "url": "https://karpathy.github.io/2019/04/25/recipe/", "note": "A practitioner's sequence for building and checking a training loop step by step."}]
practice: ["mini-training-loop", "early-stopping", "gradient-clipping-global-norm-rescaling", "mini-batch-training-loop"]
tags: ["training loop", "gradient clipping", "early stopping"]
tools: ["PyTorch"]
---

## The canonical step

Strip any PyTorch training script down and you find this:

```python
model.train()
for step, (x, y) in enumerate(train_loader):
    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    sched.step()                       # if the schedule is per step
```

The order matters. Zeroing after `backward()` throws the gradients away; clipping before `backward()` clips the previous step's gradients; stepping the scheduler before the optimiser skips the first learning rate. The Deep Learning course discusses how to choose batch size, warmup and clipping thresholds; this lesson is about wiring them correctly.

~~~dg-flow One training step, with the optional pieces in their only correct positions.
zero_grad -> forward -> loss -> backward -(optional)-> unscale + *clip -> optimizer.step -> scheduler.step
~~~

## Train and eval modes

`model.train()` and `model.eval()` set a `training` flag on every submodule. Only layers that read it change: dropout is active in training and the identity in eval; batch norm normalises with the current batch's statistics and updates running averages in training, and uses the stored running averages in eval. Neither mode affects gradient recording.

```python
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        total += loss_fn(model(x), y).item() * len(y)   # undo the mean
        n += len(y)
    model.train()                                      # restore for training
    return total / n
```

Two classic bugs: forgetting `model.eval()` (validation scores fluctuate run to run because dropout is on, and batch norm's running stats get updated with validation data) and forgetting to switch back with `model.train()` (training silently continues without dropout). Weighting each batch's mean loss by its size gives the true dataset mean even when the last batch is short.

## Gradient clipping

`clip_grad_norm_(params, max_norm)` computes the global L2 norm of all gradients concatenated, $\|g\| = \sqrt{\sum_p \|g_p\|^2}$, and if it exceeds `max_norm` rescales every gradient by `max_norm / ||g||`. The direction is preserved; only the step length is capped. It returns the norm **before** clipping, which is what to log. A healthy run shows a gradient norm that settles into a band; spikes that coincide with loss spikes point to bad batches, a learning rate that is too high, or numerical problems. `clip_grad_value_` clips each element independently and changes the direction, so it is rarely what you want.

With mixed-precision `GradScaler` (next lesson), call `scaler.unscale_(opt)` before clipping, or you clip scaled gradients against an unscaled threshold. With gradient accumulation, clip once, after the last micro-batch.

## Logging without stalling the GPU

Each `.item()` forces a host-device sync. Logging every step is fine for small models, but for fast steps accumulate on the device and sync every N steps:

```python
running = torch.zeros((), device=dev)
running += loss.detach()
if step % 100 == 0:
    log(step=step, loss=(running / 100).item(), lr=sched.get_last_lr()[0], gnorm=gnorm.item())
    running.zero_()
```

Worth logging: training loss, validation loss and metric, learning rate, gradient norm, throughput (examples or tokens per second) and peak memory (`torch.cuda.max_memory_allocated()`). Fix all seeds and log the config so a run can be reproduced.

## Validation cadence and early stopping

Evaluate on a fixed validation set every epoch, or every K steps for long runs. Early stopping keeps the best checkpoint and stops when the validation metric has not improved by at least `min_delta` for `patience` consecutive evaluations:

```python
if val < best - min_delta:
    best, bad = val, 0
    torch.save(model.state_dict(), "best.pt")
else:
    bad += 1
    if bad >= patience:
        break
```

Patience should cover the noise in the validation metric and any planned learning-rate drops; a run with a step schedule often looks flat just before a drop. Early stopping is model selection on the validation set, so report final numbers on a separate test set.

<div class="ex">
<div class="lab">Worked example</div>

**reading a run from its logs**

Step 0: loss 2.31 (close to ln 10 = 2.303 for 10 balanced classes, as expected at initialisation). Steps 0 to 2,000: loss falls to 0.6, gradient norm settles between 0.8 and 1.5 under a clip of 1.0, so clipping fires on some steps.

Step 2,400: gradient norm jumps to 40 and loss to 1.9, then recovers over 300 steps. The spike coincides with the end of warmup, when the learning rate peaks. Lowering the peak learning rate or lengthening warmup is the first thing to try.

Validation loss bottoms out at epoch 7 and rises while training loss keeps falling: overfitting. With patience 3, training stops at epoch 10 and `best.pt` holds epoch 7.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

keep the five-line core intact, put clipping, scheduling and logging in their fixed slots, evaluate in eval mode under `no_grad`, and keep the best checkpoint rather than the last.

</div>
