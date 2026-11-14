---
id: "04.04"
course: "ml-toolkit"
module: "04"
slug: "losses-and-optimisers"
title: "Losses and optimisers"
keyIdeas: ["Most classification losses take **raw logits**, not probabilities: `CrossEntropyLoss` applies log-softmax itself and `BCEWithLogitsLoss` applies a stable log-sigmoid. Applying softmax first trains a different, worse objective without any error.", "`CrossEntropyLoss` expects class scores on dimension 1 and int64 class indices: shape `(N, C)` with targets `(N,)`, or `(N, C, d1, ...)` with targets `(N, d1, ...)`. Language-model logits `(B, T, V)` must be flattened or transposed first.", "An optimiser holds references to parameters plus per-parameter state: SGD with momentum keeps one buffer, Adam keeps two (`exp_avg`, `exp_avg_sq`), so Adam's state alone is twice the model size in float32.", "Parameter groups give different subsets their own learning rate and weight decay; schedulers rewrite `group['lr']` and are stepped after `optimizer.step()`, per batch or per epoch depending on how they were configured."]
question: "A sequence tagger produces logits of shape `(B, T, C)` with padded positions labelled -100. A teammate computes `nn.CrossEntropyLoss()(torch.softmax(logits, -1).permute(0, 2, 1), y)`. It runs. List what is right, what is wrong, and what the training curves will look like."
answer: "The permute is correct: it puts classes on dimension 1, giving `(B, C, T)` against targets `(B, T)`, and the default `ignore_index=-100` correctly drops padded positions from both the sum and the averaging count. The bug is the softmax: `CrossEntropyLoss` applies log-softmax itself, so it now treats probabilities in [0, 1] as logits. Softmax of values in [0, 1] is nearly uniform: even a perfect one-hot prediction gives a logit gap of only 1, so the loss is bounded below by ln(1 + (C-1)/e), about 1.46 for 10 classes instead of 0. Gradients pass through two softmaxes and are small. Curves show the loss starting near ln(C) and flattening well above zero while accuracy creeps up slowly, with no error raised. Remove the softmax and pass the raw logits."
sources: [{"title": "PyTorch docs: CrossEntropyLoss", "url": "https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html", "note": "Exact input/target shapes, ignore_index, label smoothing and how mean reduction counts."}, {"title": "PyTorch docs: BCEWithLogitsLoss", "url": "https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html", "note": "Why the fused logit form is numerically stable, and pos_weight."}, {"title": "PyTorch docs: torch.optim", "url": "https://pytorch.org/docs/stable/optim.html", "note": "Parameter groups, per-parameter options, scheduler ordering and the full scheduler list."}, {"title": "Loshchilov and Hutter: Decoupled weight decay regularization (2017)", "url": "https://arxiv.org/abs/1711.05101", "note": "Why AdamW's decay differs from L2 regularisation inside Adam."}]
practice: ["optimizer-scheduler", "learning-rate-warmup-scheduler", "custom-sgd-with-momentum", "manual-weight-update", "cross-entropy-logits"]
tags: ["loss functions", "optimisers", "lr schedules"]
tools: ["PyTorch"]
---

## Losses and their input conventions

PyTorch losses are modules (`nn.CrossEntropyLoss()`) with functional twins (`F.cross_entropy`). The maths is covered in the Deep Learning course; what bites in practice is the input contract. Getting it wrong rarely raises an error. It silently optimises the wrong thing.

| loss | input | target | notes
| --- | --- | --- | --- |
| `CrossEntropyLoss` | logits `(N, C)` or `(N, C, d...)` | int64 indices `(N)` or `(N, d...)`, or float probabilities shaped like input | log-softmax inside; `ignore_index=-100`, `label_smoothing`, `weight`
| `NLLLoss` | log-probabilities | int64 indices | pair with `log_softmax`, never `softmax`
| `BCEWithLogitsLoss` | logits, any shape | float, same shape | stable; `pos_weight` for imbalance
| `MSELoss` / `L1Loss` / `HuberLoss` | any shape | same shape | broadcasting mismatches only warn
| `KLDivLoss` | log-probabilities | probabilities (or log with `log_target=True`) | use `reduction="batchmean"` for the true KL

Three common failures:

- **Softmax before cross-entropy.** The loss becomes cross-entropy of softmax(softmax(z)), whose gradients are squashed; training is slow and accuracy plateaus low.
- **Class dimension in the wrong place.** For a language model with logits `(B, T, V)` and targets `(B, T)`, write `F.cross_entropy(logits.view(-1, V), targets.view(-1))` or `F.cross_entropy(logits.transpose(1, 2), targets)`. Passing `(B, T, V)` directly fails only when shapes happen to disagree.
- **Broadcasting in regression.** `MSELoss` of predictions `(N, 1)` against targets `(N,)` broadcasts to `(N, N)`, comparing every prediction with every target. PyTorch prints a warning that is easy to miss; squeeze the prediction.

With `reduction="mean"` and `ignore_index`, cross-entropy divides by the number of non-ignored targets, which is what you want for padded sequences within one batch. Across accumulated micro-batches with different numbers of real tokens, sum the losses and divide by the global token count yourself.

## What an optimiser holds

An optimiser is constructed from an iterable of parameters (or of parameter groups) and keeps two things: `param_groups`, a list of dicts with hyperparameters like `lr`, `weight_decay` and `betas`, and `state`, a per-parameter dict. For `torch.optim.AdamW` the state of each parameter is a step count plus `exp_avg` and `exp_avg_sq`, both the parameter's shape and dtype. For a 1-billion-parameter float32 model that is 8 GB of optimiser state on top of 4 GB of weights and 4 GB of gradients, which is why the LLM-scale module counts bytes per parameter.

~~~dg-seq One optimisation step: who reads and writes what.
Loop > Model : forward(x)
Model -> Loop : logits
Loop > Loss : loss(logits, y)
Loop > Autograd : loss.backward()
Autograd -> Model : p.grad filled
Loop > Optimiser : step() reads p.grad, updates state and p
Loop > Scheduler : step() rewrites lr in param_groups
Loop > Optimiser : zero_grad()
~~~

Build the optimiser **after** moving the model to its device; the optimiser keeps references to the parameter tensors, and its state is created lazily on the first `step()`, on the same device as each parameter.

~~~dg-anim gradient-descent | What `optimizer.step()` does on the simplest loss there is. Drag the learning rate past 0.5 to see oscillation, and past 1 to see divergence.
~~~

## Parameter groups

Groups let one optimiser treat subsets differently. The standard Transformer recipe applies weight decay to matrices but not to biases, norm weights or (often) embeddings:

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (decay if p.ndim >= 2 else no_decay).append(p)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4, betas=(0.9, 0.95))
```

Fine-tuning often adds a group with a smaller learning rate for the pretrained backbone and a larger one for the new head. Note the difference between `Adam(weight_decay=...)`, which adds an L2 term to the gradient and so gets rescaled by Adam's normaliser, and `AdamW`, which decays the weights directly; the ML Maths course explains why the decoupled version regularises as intended.

## Schedulers

A scheduler wraps an optimiser and rewrites `group["lr"]` each time you call `scheduler.step()`. Call it after `optimizer.step()`; the reverse order skips the first value of the schedule and PyTorch warns about it.

- **Per-epoch** schedules: `StepLR`, `MultiStepLR`, `CosineAnnealingLR(T_max=epochs)`.
- **Per-step** schedules: `OneCycleLR`, or `LambdaLR` with a warmup-then-cosine lambda over total steps, standard for Transformers. Configure `T_max` or the lambda in the same unit you step in; mixing epochs and steps is a common silent bug.
- **Metric-driven**: `ReduceLROnPlateau` takes the validation loss, `scheduler.step(val_loss)`.
- `SequentialLR` and `ChainedScheduler` compose schedules, for example linear warmup then cosine.

The scheduler's `state_dict()` (current step, base learning rates) must be checkpointed along with the optimiser's, or a resumed run restarts its warmup.

<div class="ex">
<div class="lab">Worked example</div>

**warmup plus cosine with LambdaLR**

Total 10,000 steps, 500 warmup steps, peak learning rate 3e-4 set on the optimiser.

`lam = lambda s: (s + 1) / 500 if s < 500 else 0.5 * (1 + math.cos(math.pi * (s - 500) / 9500))`

At step 250 the multiplier is 0.502 (lr about 1.5e-4); at step 5,250 it is 0.5 (1.5e-4 on the way down); at step 9,999 it is about 0, so the run ends with a near-zero learning rate. Step it once per batch.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

feed losses logits in the layout they expect, remember that the optimiser carries state as large as the model, and treat learning rate, groups and schedule as part of the checkpointed training state.

</div>
