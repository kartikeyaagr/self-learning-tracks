---
id: "04.06"
course: "ml-toolkit"
module: "04"
slug: "the-training-loop"
title: "The training loop"
keyIdeas: ["Every PyTorch training step is the same five lines: zero gradients, forward, loss, backward, step. Everything else in a real loop (clipping, accumulation, scheduling, logging, evaluation, checkpoints) is inserted at fixed points around them.", "The order is not optional: forgetting `zero_grad()` makes gradients pile up until training explodes, and zeroing between `backward()` and `step()` means nothing is ever learned.", "`model.train()` and `model.eval()` switch layer behaviour (dropout on or off, batch norm using batch statistics or running ones). They do not touch autograd, so evaluation also needs `torch.no_grad()`.", "Clip after `backward()` and before `step()` with `clip_grad_norm_`, which shrinks the step but keeps its direction, and log the norm it returns: the gradient norm is one of the most informative curves you can plot.", "Validate on a fixed held-out set at a fixed cadence, keep the best checkpoint by validation metric, and stop when it has not improved for `patience` evaluations."]
question: "Your validation accuracy changes by about 1 point every time you re-run evaluation on the same checkpoint, and after the first evaluation the training loss starts falling faster than before. The model uses dropout and batch norm. What is wrong with the loop and what do you change?"
answer: "Evaluation is running in training mode. With dropout active, every evaluation samples a different random subnetwork, so accuracy varies between identical re-runs (and is usually lower than the true score); in eval mode the forward pass is deterministic. Batch norm in training mode also normalises with the validation batches' own statistics and updates its running averages with validation data, so evaluation silently changes the model. The faster-falling training loss after evaluation is the second half of the bug: either the evaluation code switched to eval mode and never switched back, so training continues with dropout disabled and fits the training data more easily (it looks better but is now overfitting without the regulariser), or the running statistics were contaminated. The fix is a single evaluate helper that records `model.training`, calls `model.eval()`, runs the whole loop under `torch.no_grad()` (or `torch.inference_mode()`), weights each batch's loss by its size, and restores the previous mode with `model.train(was_training)` before returning. Then re-run evaluation twice on the same checkpoint and confirm the numbers match exactly; if batch norm statistics may already be contaminated, reload the last clean checkpoint."
sources: [{"title": "PyTorch tutorial: Optimizing model parameters", "url": "https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html", "note": "The official minimal train and test loops, with the step order spelled out."}, {"title": "PyTorch docs: clip_grad_norm_", "url": "https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html", "note": "The global-norm definition, the returned value and the error_if_nonfinite option."}, {"title": "Karpathy: A recipe for training neural networks (2019)", "url": "https://karpathy.github.io/2019/04/25/recipe/", "note": "A practitioner's sequence for building and checking a training loop step by step."}]
practice: ["mini-training-loop", "early-stopping", "gradient-clipping-global-norm-rescaling", "mini-batch-training-loop"]
tags: ["training loop", "gradient clipping", "early stopping"]
tools: ["PyTorch"]
---

## What you will learn

- The five lines of every PyTorch training step, and why their order cannot change.
- What `model.train()` and `model.eval()` switch, and why evaluation also needs `torch.no_grad()`.
- How gradient clipping by global norm works, where it goes, and why its return value is worth logging.
- How to log loss and gradient norm without slowing training, and how to read the curves.
- How to validate on a schedule, keep the best checkpoint, and stop early with a patience counter.

*Before you start:* you should know tensors and devices (04.01), how `backward()` fills `.grad` (04.02), `nn.Module` (04.03), losses, optimisers and schedulers (04.04), and `DataLoader` (04.05). This lesson glues those pieces together.

The last five lessons gave you every part of a training run: a model, a loss, an optimiser, a scheduler and a data loader. The **training loop** is the short code you write yourself that feeds batches through the model, turns the loss into gradients and lets the optimiser move the weights. PyTorch leaves it to you, so every bug in it is yours.

## The idea in one picture

Strip any PyTorch training script down and this is what is left:

```python
for x, y in train_loader:
    opt.zero_grad()            # 1. forget the previous step's gradients
    out = model(x)             # 2. forward pass
    loss = loss_fn(out, y)     # 3. one number: how wrong we are on this batch
    loss.backward()            # 4. fill p.grad for every parameter p
    opt.step()                 # 5. move the weights a little downhill
```

One pass of this body over one mini-batch is a **step**; one pass over the whole training set is an **epoch**. After each epoch you evaluate on held-out data.

We ran exactly that on a small 3-class problem (300 training examples with 20 features, 20% of the labels deliberately randomised, 2,000 validation examples) with a two-hidden-layer network and Adam:

~~~dg-fig 04.06-curves.svg
Training loss keeps falling for 30 epochs, but validation loss is lowest at epoch 7 and then climbs. Early stopping with patience 3 halts at epoch 10 and keeps the epoch-7 weights.
~~~

At first both curves fall together. From epoch 7 the **validation loss** (loss on data the model never trains on) turns upward while the **training loss** keeps dropping: the network has started fitting the noisy labels, which is **overfitting**. Saving the best weights and stopping when validation stops improving is called **early stopping**, and it is the last slot we add to the loop.

## Step 1: the five-line core, and why order matters

Each line has one job.

1. `opt.zero_grad()` clears `.grad` on every parameter. Gradients in PyTorch **accumulate**: `backward()` adds into `.grad` rather than overwriting it (04.02). Since PyTorch 2.0 the default `set_to_none=True` sets `.grad` to `None` rather than zeros, saving memory.
2. `model(x)` runs the forward pass and records the graph autograd needs.
3. `loss_fn(out, y)` reduces the batch to one scalar, by default the **mean** over the batch.
4. `loss.backward()` walks the graph backwards and writes $\partial L / \partial p$ into `p.grad`.
5. `opt.step()` updates each parameter using its `.grad`; for plain SGD that is $p \leftarrow p - \eta\, \partial L/\partial p$, where $\eta$ is the learning rate.

The order is not a style choice. To see it, we trained the same network on the same 400 batches three ways with plain SGD at learning rate 0.05:

~~~dg-fig 04.06-order.svg
Left: correct order descends smoothly; forgetting zero_grad looks fast at first, then the loss explodes past 10; zeroing between backward and step never learns. Right: without zero_grad, the gradient the optimiser uses grows from about 0.6 to over 300.
~~~

- **Correct order**: loss falls steadily from 1.11 to 0.26.
- **`zero_grad()` forgotten**: each step uses the sum of every gradient so far. Early on that acts like a bigger learning rate, so the bug can survive a quick glance. By step 200 the accumulated gradient has a norm of about 56 (right panel), the steps overshoot, and the loss jumps above 10.
- **`zero_grad()` between `backward()` and `step()`**: the optimiser always sees empty gradients, so nothing moves. The loss is flat at 1.11 for all 400 steps.

Two more rules follow. Clipping comes after `backward()`, since before it `.grad` holds the previous step's values. The scheduler steps after the optimiser; calling `sched.step()` first skips the first learning rate, and PyTorch warns about it.

## Step 2: train mode and eval mode

`model.train()` and `model.eval()` set a boolean `training` flag on the module and every submodule inside it. Only layers that read the flag change behaviour:

- **Dropout** zeroes a random fraction of activations in training mode (scaling up the rest); in eval mode it does nothing.
- **Batch norm** in training mode uses the current batch's mean and variance and updates running averages; in eval mode it uses the stored averages.

The flag does **not** touch autograd: in eval mode PyTorch still records the graph and keeps activations for a backward pass that never comes. So evaluation also runs under `torch.no_grad()` (or the slightly faster `torch.inference_mode()`). Eval mode means "behave like the deployed model"; `no_grad` means "do not prepare for backward".

What if you evaluate in training mode? We trained a network with dropout 0.5, then measured validation accuracy 20 times on the same weights and examples:

~~~dg-fig 04.06-evalmode.svg
Twenty re-runs of evaluation on one checkpoint. With dropout left on, accuracy wanders between 62.7% and 66.1% because each run samples a different subnetwork; in eval mode it is 70.2% every time.
~~~

In training mode each run drops a different random half of the units, so the score is noisy and too low. A small helper makes this hard to get wrong:

```python
@torch.no_grad()
def evaluate(model, loader):
    was_training = model.training
    model.eval()
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        total += loss_fn(out, y).item() * len(y)    # undo the per-batch mean
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    model.train(was_training)                       # restore whatever mode we came from
    return total / n, correct / n
```

<div class="ex">
<div class="lab">Worked example</div>

**why the batch-size weighting matters**

A validation set of 70 examples is split into batches of 32, 32 and 6. The mean losses of the three batches are 0.50, 0.60 and 1.20.

Averaging the three means gives $(0.50 + 0.60 + 1.20) / 3 = 2.30 / 3 = 0.767$. That treats the 6-example batch as if it were as large as the others.

Weighting by size: $32 \times 0.50 = 16.0$, $32 \times 0.60 = 19.2$, $6 \times 1.20 = 7.2$. The total is $16.0 + 19.2 + 7.2 = 42.4$, and $42.4 / 70 = 0.606$.

The naive mean overstates the true 0.606 by about 27%, purely because of how the data was split.

</div>

## Step 3: gradient clipping

Sometimes one batch produces a huge gradient (a corrupted example, a learning rate at its peak), and one huge step can undo hundreds of good ones. **Gradient clipping** caps the length of the step without changing its direction.

`torch.nn.utils.clip_grad_norm_(params, max_norm)` first computes the **global norm** of all gradients, as if every parameter's gradient were glued into one long vector:

$$\|g\| = \sqrt{\sum_{p} \|g_p\|^2}$$

Here $\|g_p\|^2$ is the sum of squared entries of parameter $p$'s gradient. If $\|g\|$ is larger than `max_norm`, every gradient is multiplied by the same factor $\text{max\_norm} / \|g\|$; otherwise nothing changes. The function returns the norm **before** clipping, which is the number to log.

<div class="ex">
<div class="lab">Worked example</div>

**clip two gradients by hand**

A model has two parameters. After `backward()`, the first has gradient $(3, 4)$ and the second has gradient $(12)$. We clip with `max_norm = 1.0`.

Squared norms: $3^2 + 4^2 = 9 + 16 = 25$ for the first, $12^2 = 144$ for the second. Global norm: $\sqrt{25 + 144} = \sqrt{169} = 13$.

13 is larger than 1.0, so every entry is multiplied by $1/13 \approx 0.0769$. The first gradient becomes $(3/13, 4/13) = (0.231, 0.308)$ and the second becomes $12/13 = 0.923$.

Check: $0.231^2 + 0.308^2 + 0.923^2 = 0.053 + 0.095 + 0.852 = 1.000$, exactly the cap, and the ratios $3 : 4 : 12$ (the direction) are unchanged.

PyTorch agrees: `clip_grad_norm_` returns `tensor(13.)` and leaves gradients 0.2308, 0.3077 and 0.9231.

</div>

To test it, we trained with SGD plus momentum and at step 91 fed one bad batch: inputs times 30, labels shuffled.

~~~dg-fig 04.06-clip.svg
Left: the gradient norm stays in a band around 1, then spikes to about 54 on the bad batch. Right: without clipping, validation loss jumps from 0.70 to 1.22 and needs dozens of steps to recover; clipped at 1.0, it barely moves.
~~~

A healthy gradient-norm log is a noisy band, here roughly 0.5 to 1.5; the spike stands out by a factor of 50. In the unclipped run that one step (plus the momentum it leaves behind) wrecks the model: 90 steps later its validation loss (0.77) is still above where it was before the spike (0.70). With clipping the bad step is scaled down to length 1, the same as an ordinary step, and the run continues.

Clipping also fired on 48% of ordinary steps here, because the normal band sits near 1. If it fires on nearly every step it acts as a smaller learning rate; set `max_norm` a little above a healthy run's band.

`clip_grad_value_` clamps each entry to $[-c, c]$ independently, which changes the direction; it is rarely what you want.

## Step 4: logging without slowing down

Log training and validation loss, the metric, learning rate, gradient norm, throughput and, on a GPU, peak memory (`torch.cuda.max_memory_allocated()`).

Per-batch loss is noisy because each batch is a different sample; average over a window before logging:

~~~dg-fig 04.06-smooth.svg
Per-batch training loss jumps around by ±0.2; the mean of each 20-step window shows the real trend: a fast drop, then a slow plateau near 0.64.
~~~

There is also a speed reason. On a GPU, Python queues kernels and moves on. Calling `.item()` or `print` on a GPU tensor makes Python wait until all queued work finishes, a **host-device sync**, which can leave a fast GPU idle between steps. Accumulate on the device and sync only when you log:

```python
running = torch.zeros((), device=dev)
for step, (x, y) in enumerate(train_loader):
    ...                                    # the five-line core
    running += loss.detach()               # stays on the device, no sync
    if (step + 1) % 100 == 0:
        print(step + 1, (running / 100).item(), sched.get_last_lr()[0])   # one sync per 100 steps
        running.zero_()
```

Without `.detach()`, `running` would keep every step's graph alive and memory would grow. On a CPU there is nothing to sync, so the pattern only pays off on accelerators. Also fix all seeds and log the configuration so a run can be reproduced.

## Step 5: validation cadence and early stopping

Evaluate on a **fixed** validation set (the same examples every time) at a fixed cadence: once per epoch for small data, every K steps for long runs.

Early stopping needs three numbers:

- `best`: the best validation loss seen so far.
- `patience`: how many evaluations in a row without improvement you tolerate.
- `min_delta`: how much better a value must be to count as improvement, so noise does not reset the counter.

The rule: if the new value beats `best - min_delta`, save the weights and reset the counter; otherwise add one to the counter and stop when it reaches `patience` (the full code is in the loop below).

<div class="ex">
<div class="lab">Worked example</div>

**run the patience counter by hand**

Validation losses from the run in the first figure, epochs 1 to 10: 1.016, 0.925, 0.841, 0.788, 0.778, 0.775, 0.772, 0.783, 0.784, 0.794. Use `patience = 3` and `min_delta = 0.001`.

Epochs 1 to 7: each value is more than 0.001 below the previous best (for example at epoch 7, $0.772 < 0.775 - 0.001 = 0.774$), so `best` is updated and `best.pt` overwritten. After epoch 7, `best = 0.772`.

Epoch 8: $0.783$ is not below $0.771$, so `bad = 1`. Epoch 9: $0.784$, `bad = 2`. Epoch 10: $0.794$, `bad = 3`, which reaches the patience, so the loop stops.

`best.pt` holds epoch 7. Training to epoch 30 would have reached validation loss 1.25, worse than the untrained model.

</div>

Patience should cover the noise of your validation metric and any planned learning-rate drop: with a step schedule the curve often looks flat just before the drop. Early stopping uses the validation set to choose a model, so that score is slightly optimistic; report final numbers on a separate test set.

## Step 6: the optional pieces and their only correct slots

Real loops add a few more operations, each with exactly one place.

~~~dg-flow One training step, with the optional pieces in their only correct positions.
zero_grad -> forward -> loss -> backward -(optional)-> unscale + *clip -> optimizer.step -> scheduler.step
~~~

- **Mixed precision** (04.07) multiplies the loss by a scale factor before `backward()`. Call `scaler.unscale_(opt)` before clipping, or you compare scaled gradients with an unscaled threshold.
- **Gradient accumulation** simulates a large batch with several small ones: run forward and backward on $k$ micro-batches, dividing each loss by $k$, and only then clip, step and zero. Four micro-batches of 16 gave the same gradient as one batch of 64 (max difference $6 \times 10^{-8}$, rounding). Clip once, after the last micro-batch.
- **Per-step schedulers** (warmup, cosine) call `sched.step()` after every `opt.step()`; per-epoch schedulers call it once after the epoch. Mixing the two up makes the schedule run 100 times too fast or too slow.

## The whole loop in code

Everything assembled, run on CPU on the data of the first figure (only `dev` changes on a GPU):

```python
model.train()
best, bad, patience, min_delta = float("inf"), 0, 3, 1e-3
print(evaluate(model, val_loader))          # (1.110, 0.306) before training; ln 3 = 1.099
for epoch in range(1, 31):
    model.train()
    run_loss, run_n, gnorms = 0.0, 0, []
    for x, y in train_loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        gnorms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        run_loss += loss.detach() * len(y); run_n += len(y)
    val_loss, val_acc = evaluate(model, val_loader)
    gn = torch.stack(gnorms)
    print(f"epoch {epoch}  train {run_loss.item()/run_n:.3f}  val {val_loss:.3f}  "
          f"acc {val_acc:.3f}  gnorm median {gn.median():.2f} max {gn.max():.2f}")
    if val_loss < best - min_delta:
        best, bad = val_loss, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        bad += 1
        if bad >= patience:
            break
model.load_state_dict(torch.load("best.pt"))
# epoch 1  train 1.087  val 1.012  acc 0.606  gnorm median 0.75 max 1.09
# epoch 5  train 0.683  val 0.771  acc 0.709  gnorm median 0.97 max 1.21
# epoch 6  train 0.651  val 0.760  acc 0.722  gnorm median 1.06 max 1.41
# epoch 9  train 0.530  val 0.773  acc 0.718  gnorm median 1.05 max 1.59  -> stop
# after reload: val 0.760, acc 0.722 (the epoch-6 weights)
```

The best epoch is 6 here, not 7, because clipping changed the updates slightly: trust the shape of the curves, not the exact epoch.

## Reading a run from its logs

The evaluation before training is a free sanity check. A classifier that knows nothing spreads probability evenly, so its cross-entropy is about $\ln C$ for $C$ classes: $\ln 3 = 1.099$ here (we measured 1.110). A much larger starting loss means a bad initialisation or a label bug.

<div class="ex">
<div class="lab">Worked example</div>

**diagnosing a longer run from its logs**

Step 0: loss 2.31 on a 10-class problem, close to $\ln 10 = 2.303$, as expected at initialisation.

Steps 0 to 2,000: loss falls to 0.6; the gradient norm settles between 0.8 and 1.5 under a clip of 1.0, so clipping fires on some steps.

Step 2,400: the gradient norm jumps to 40 and the loss to 1.9, then recovers over 300 steps. The spike coincides with the end of warmup, when the learning rate peaks. Lowering the peak learning rate or lengthening warmup is the first thing to try.

Validation loss bottoms out at epoch 7 while training loss keeps falling: overfitting. Patience 3 stops the run at epoch 10 with `best.pt` holding epoch 7.

</div>

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Validation score changes on re-runs of the same checkpoint | evaluating in training mode, dropout active | `model.eval()` inside the evaluate helper
| Training gets easier right after the first evaluation | eval mode never switched back, dropout now off | restore with `model.train()` before the next epoch
| Loss drops fast, then explodes | `zero_grad()` missing, gradients pile up | zero once per step, before forward
| Loss perfectly flat from step 0 | gradients zeroed after `backward()`, or optimiser built on the wrong parameters | fix the order; check `opt.param_groups`
| Memory grows every step until out of memory | logging `loss` instead of `loss.detach()` or `loss.item()` | detach before accumulating
| Final model worse than mid-run | saved the last weights, not the best | save on improvement, reload `best.pt`

## Going deeper

**Why clipping keeps the direction.** Scaling every gradient by the same factor $c = \text{max\_norm}/\|g\|$ scales the whole update vector by $c$. For SGD the update is $-\eta c\, g$, the same direction as $-\eta g$, just shorter: clipping acts as a learning rate that shrinks only on steps with a large gradient. Adam divides each coordinate by a running estimate of its gradient size, so there clipping mainly caps how much one batch can corrupt those estimates and the momentum. In code, PyTorch actually multiplies by $\text{max\_norm}/(\|g\| + 10^{-6})$ to avoid dividing by zero, and passing `error_if_nonfinite=True` raises an error if the norm is `inf` or `nan` instead of silently writing `nan` into every gradient.

**Accumulation is not quite a big batch.** Summing the gradients of $k$ micro-batches equals the large-batch gradient only when the loss is a mean over independent examples. Batch norm breaks this: its statistics come from each micro-batch of 16, not the virtual batch of 64 (layer norm does not have this problem). With distributed data parallel training, wrap all but the last micro-batch in `model.no_sync()` so gradients are only averaged across devices once.

**Where the time goes.** A step is a pipeline: loader workers prepare the next batch (04.05), `non_blocking=True` copies it from pinned memory while the GPU still runs queued kernels. Anything that makes Python wait, such as `.item()` or an `if` on a tensor value, drains that pipeline.

## Key terms

| Term | Meaning
| --- | --- |
| Step / epoch | One weight update from one mini-batch / one full pass over the training set
| `zero_grad` | Clears `.grad` so the next backward pass starts from nothing
| Train / eval mode | A flag that switches dropout and batch norm between training and inference behaviour
| `no_grad` | Turns off graph recording, so evaluation uses less memory and time
| Global gradient norm | Length of all gradients joined into one vector; what `clip_grad_norm_` measures and returns
| Gradient clipping | Rescaling gradients so their global norm is at most `max_norm`, keeping the direction
| Gradient accumulation | Summing gradients over several micro-batches before one optimiser step
| Early stopping | Stopping when validation has not improved for `patience` evaluations, keeping the best checkpoint
| Host-device sync | Python waiting for the GPU to finish, triggered by `.item()` and similar calls

<div class="take">
<div class="lab">Key takeaway</div>

keep the five-line core intact (zero_grad, forward, loss, backward, step), put clipping, scheduling and logging in their fixed slots, evaluate in eval mode under `no_grad`, and keep the best checkpoint rather than the last.

</div>
