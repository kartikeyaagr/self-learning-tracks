---
id: "04.04"
course: "ml-toolkit"
module: "04"
slug: "losses-and-optimisers"
title: "Losses and optimisers"
keyIdeas: ["Most classification losses take **raw logits**, not probabilities: `CrossEntropyLoss` applies log-softmax itself and `BCEWithLogitsLoss` applies a stable log-sigmoid. Applying softmax first trains a different, worse objective (for ten classes the loss can never drop below 1.46) without any error.", "`CrossEntropyLoss` expects class scores on dimension 1 and int64 class indices: shape `(N, C)` with targets `(N,)`, or `(N, C, d1, ...)` with targets `(N, d1, ...)`. Sequence logits `(B, T, V)` must be flattened or transposed first, and positions labelled -100 are ignored.", "Regression losses broadcast: predictions `(N, 1)` against targets `(N,)` compare every pair, and the model learns to predict the mean. Squeeze the prediction.", "An optimiser holds references to parameters plus per-parameter state: SGD with momentum keeps one buffer, Adam keeps two (`exp_avg`, `exp_avg_sq`), so Adam's state alone is twice the model size in float32.", "Parameter groups give subsets their own learning rate and weight decay; `AdamW` decays weights directly, unlike L2 inside `Adam`. Schedulers rewrite `group['lr']`, are stepped after `optimizer.step()` in the same unit they were configured in, and must be checkpointed with the optimiser."]
question: "A sequence tagger produces logits of shape `(B, T, C)` with padded positions labelled -100. A teammate computes `nn.CrossEntropyLoss()(torch.softmax(logits, -1).permute(0, 2, 1), y)`. It runs. List what is right, what is wrong, and what the training curves will look like."
answer: "The permute is correct: it puts classes on dimension 1, giving `(B, C, T)` against targets `(B, T)`, and the default `ignore_index=-100` correctly drops padded positions from both the sum and the averaging count. The bug is the softmax: `CrossEntropyLoss` applies log-softmax itself, so it now treats probabilities in [0, 1] as logits. Softmax of values in [0, 1] is nearly uniform: even a perfect one-hot prediction gives a logit gap of only 1, so the loss is bounded below by ln(1 + (C-1)/e), about 1.46 for 10 classes instead of 0. Gradients pass through two softmaxes and are small (the softmax Jacobian has entries below 0.25 and the outer softmax is nearly flat on [0, 1]). Curves show the loss starting near ln(C) and flattening well above zero while accuracy creeps up slowly, with no error raised; a quick check is to overfit one batch, which will stall near the floor instead of reaching zero. Remove the softmax and pass the raw logits (keeping the permute, or equivalently flattening to `(B*T, C)` and `(B*T,)`). Apply softmax only at inference time when probabilities are needed."
sources: [{"title": "PyTorch docs: CrossEntropyLoss", "url": "https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html", "note": "Exact input/target shapes, ignore_index, label smoothing and how mean reduction counts."}, {"title": "PyTorch docs: BCEWithLogitsLoss", "url": "https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html", "note": "Why the fused logit form is numerically stable, and pos_weight."}, {"title": "PyTorch docs: torch.optim", "url": "https://pytorch.org/docs/stable/optim.html", "note": "Parameter groups, per-parameter options, scheduler ordering and the full scheduler list."}, {"title": "Loshchilov and Hutter: Decoupled weight decay regularization (2017)", "url": "https://arxiv.org/abs/1711.05101", "note": "Why AdamW's decay differs from L2 regularisation inside Adam."}]
practice: ["optimizer-scheduler", "learning-rate-warmup-scheduler", "custom-sgd-with-momentum", "manual-weight-update", "cross-entropy-logits"]
tags: ["loss functions", "optimisers", "lr schedules"]
tools: ["PyTorch"]
---

## What you will learn

- What a loss and an optimiser each do in one training step, and how they meet through `.grad`.
- The input contract of common PyTorch losses: logits or probabilities, where the classes go, which target dtype.
- Three silent bugs: softmax before cross-entropy, a misplaced class dimension, broadcasting in regression.
- What SGD, momentum and Adam compute, and how much memory their state costs.
- How parameter groups and schedulers work, and what must go in a checkpoint.

*Before you start:* shapes and dtypes (04.01), how `backward()` fills `.grad` (04.02), and module parameters (04.03). The maths of cross-entropy and gradient descent lives in the Deep Learning course; here the focus is how PyTorch expects you to use them.

## From a model to a training signal

Lesson 04.03 built a model that turns inputs into outputs. To make it learn you need two more pieces. A **loss function** compares outputs with labels and returns one number, how wrong the batch was. An **optimiser** reads the gradient of that number for every parameter and nudges the parameters to lower it.

Both are one line of code, and that is the danger: a loss given the wrong kind of input rarely crashes. It quietly optimises a different objective, and the only symptom is a curve that looks a bit worse than it should.

## The idea in one picture

A classifier for three animals sees a batch of three images. For each image it outputs three **logits**: raw, unbounded scores, one per class, where higher means "more likely". They are not probabilities: they can be negative and they do not sum to 1.

```python
import torch, torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1],    # example 1, true class: cat  (0)
                       [0.5, 2.5, 0.3],    # example 2, true class: dog  (1)
                       [1.2, 0.4, 0.9]])   # example 3, true class: bird (2)
target = torch.tensor([0, 1, 2])           # int64 class indices, shape (3,)

print(logits.softmax(dim=1))
# tensor([[0.6590, 0.2424, 0.0986],
#         [0.1086, 0.8025, 0.0889],
#         [0.4566, 0.2052, 0.3383]])
print(F.cross_entropy(logits, target, reduction="none"))  # tensor([0.4170, 0.2200, 1.0840])
print(F.cross_entropy(logits, target))                    # tensor(0.5737)
```

~~~dg-fig 04.04-picture.svg
Left: the raw scores (logits) the model produces. Right: the same scores turned into probabilities by softmax. The outlined bar is the true class; each example's loss is minus the log of that bar's height, and the batch loss is their mean.
~~~

The model is fairly sure of the cat (0.659) and very sure of the dog (0.802), so those losses are small. For the bird it gives the true class only 0.338 and prefers "cat", so that loss is large (1.084). The batch loss, 0.574, is the average.

Two words to keep straight:

- **Softmax** turns a row of logits into probabilities: exponentiate each, then divide by the row's sum.
- **Cross-entropy** for one example is $-\ln p_{\text{true}}$: minus the log of the probability given to the correct class. It is 0 when the model is certain and right, and it grows without limit as the true class's probability goes to 0.

The crucial fact: `F.cross_entropy` takes the **logits** and does the softmax itself, so you never apply softmax before it.

## Step 1: what a loss computes, by hand

For a row of logits $z_1, \dots, z_C$ (one per class, $C$ classes) with true class $y$, cross-entropy is

$$\ell(z, y) = -\ln \frac{e^{z_y}}{\sum_{j=1}^{C} e^{z_j}} = \ln \sum_{j=1}^{C} e^{z_j} - z_y .$$

Here $e^{z_j}$ is the exponential of logit $j$, the fraction is the softmax probability of the true class, and the right-hand form ("log-sum-exp minus the true logit") is what PyTorch actually evaluates, because it never needs to form a probability that could round to zero.

<div class="ex">
<div class="lab">Worked example</div>

**cross-entropy for example 1 by hand**

Logits $z = (2.0, 1.0, 0.1)$, true class cat (index 0).

Exponentials: $e^{2.0} = 7.389$, $e^{1.0} = 2.718$, $e^{0.1} = 1.105$. Their sum is $11.212$.

Probability of cat: $7.389 / 11.212 = 0.659$. Loss: $-\ln 0.659 = 0.417$.

Same answer the stable way: $\ln 11.212 - 2.0 = 2.417 - 2.0 = 0.417$.

The gradient of the loss with respect to the logits is simply "probabilities minus the one-hot target": $(0.659 - 1,\ 0.242,\ 0.099) = (-0.341,\ 0.242,\ 0.099)$. Gradient descent therefore raises the cat logit and lowers the other two, in proportion to how wrong each one is. `z.grad` after `backward()` prints exactly `[-0.3410, 0.2424, 0.0986]`.

</div>

## Step 2: know each loss's input contract

PyTorch losses come as modules (`nn.CrossEntropyLoss()`) and as functions (`F.cross_entropy`); they compute the same thing. What differs between losses is what they expect to receive:

| Loss | Input | Target | Notes
| --- | --- | --- | --- |
| `CrossEntropyLoss` | logits `(N, C)` or `(N, C, d...)` | int64 indices `(N)` or `(N, d...)`, or float probabilities shaped like the input | log-softmax inside; `ignore_index=-100`, `label_smoothing`, `weight`
| `NLLLoss` | log-probabilities | int64 indices | pair with `log_softmax`, never `softmax`
| `BCEWithLogitsLoss` | logits, any shape | float, same shape | stable; `pos_weight` for imbalance
| `MSELoss` / `L1Loss` / `HuberLoss` | any shape | same shape | broadcasting mismatches only warn
| `KLDivLoss` | log-probabilities | probabilities (or log with `log_target=True`) | use `reduction="batchmean"` for the true KL

Here `N` is the batch size and `C` the number of classes. Three rules cover most of the table: classification losses want **raw logits** (or log-probabilities when the name says so), class scores live on **dimension 1**, and class labels are **int64** while binary and regression targets are **float**.

**Binary and multi-label problems** use `BCEWithLogitsLoss`: one logit per yes/no question, sigmoid applied inside. The fused version matters more than it looks. In float32, `torch.sigmoid(30.0)` rounds to exactly 1.0, so a separate `BCELoss` must take the log of 0 for a confidently wrong example; PyTorch clamps that log at -100, so the loss is stuck at 100 with no useful gradient:

```python
import torch.nn as nn
z = torch.tensor([30., -30.]); t = torch.tensor([0., 1.])   # two confidently wrong logits
print(nn.BCELoss()(torch.sigmoid(z), t))   # tensor(65.)  the first term is clamped at 100
print(nn.BCEWithLogitsLoss()(z, t))        # tensor(30.)  exact: each term is 30
```

Cross-entropy also accepts float targets shaped like the logits (for distillation or mixup), and `label_smoothing=0.1` softens one-hot labels. `KLDivLoss` is the odd one out: input log-probabilities, target probabilities, and only `reduction="batchmean"` gives the true KL divergence.

## Step 3: avoid the three silent bugs

**Bug 1: softmax before cross-entropy.** `F.cross_entropy(logits.softmax(-1), y)` runs without complaint, but the loss now treats probabilities as logits and applies its own softmax on top. Even a perfect prediction becomes a logit gap of only 1, so the loss can never approach zero.

~~~dg-fig 04.04-margin.svg
Loss for one example with ten classes, as the true class's logit rises above the others. The correct loss falls towards zero. With softmax applied first it flattens at ln(1 + 9/e) = 1.46 and stops rewarding further confidence.
~~~

The floor is $\ln(1 + (C-1)/e)$: 1.46 for ten classes. The gradient also shrinks: for example 1, the push on the cat logit drops from -0.341 to -0.188. Here is a measured run of the same network trained both ways.

~~~dg-fig 04.04-double.svg
Measured on CPU: a 20-64-10 network, SGD with learning rate 0.1, 30 epochs. With logits the validation accuracy reaches 68% by epoch 5. With softmax first the loss crawls down towards its floor and accuracy is only 57% after 30 epochs.
~~~

Accuracy still rises, which is why this bug survives code review. The fingerprint is a loss that refuses to go much below the floor.

**Bug 2: classes in the wrong dimension.** `CrossEntropyLoss` reads classes from dimension 1, but a sequence model's logits `(B, T, V)` (batch, time steps, vocabulary) have time there. Flatten, or move the class axis:

```python
B, T, V = 2, 4, 5
logits = torch.randn(B, T, V)
y = torch.tensor([[1, 2, -100, -100],      # -100 marks padding
                  [0, 3, 4, -100]])
a = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))   # (B*T, V) vs (B*T,)
b = F.cross_entropy(logits.transpose(1, 2), y)              # (B, V, T) vs (B, T)
print(torch.allclose(a, b))   # True
```

Passing `(B, T, V)` directly fails only when the shapes happen to disagree. If $T$ equals $V$ it runs and computes nonsense.

The `-100` entries are skipped (`ignore_index=-100` is the default), and the mean divides by the 5 real targets, not 8.

**Bug 3: broadcasting in regression.** A head `nn.Linear(d, 1)` outputs `(N, 1)`. With targets `(N,)`, `pred - target` broadcasts to `(N, N)`: every prediction against every target.

PyTorch only prints a `UserWarning` ("Using a target size ... that is different to the input size").

~~~dg-fig 04.04-broadcast.svg
Left: subtracting a column from a row gives an N-by-N grid; only the diagonal holds the pairs you meant. Right: a linear model trained for 300 steps with the bug learns a flat line at the mean of y (slope 0.00), while squeezing the prediction recovers the true slope of about 2.
~~~

The loss over all pairs is minimised by predicting the overall mean for every input. The warning scrolls away; the fix is `pred.squeeze(-1)`.

## Step 4: what an optimiser does

An optimiser is built from the parameters, `torch.optim.SGD(model.parameters(), lr=0.1)`, and keeps references to them. `opt.step()` reads each `.grad` and updates the parameter in place; `opt.zero_grad()` clears gradients, because autograd adds into `.grad` (04.02).

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

**Plain SGD** subtracts the gradient times the **learning rate** $\eta$: $p \leftarrow p - \eta\, g$, where $p$ is a parameter and $g$ its gradient. `p - lr * p.grad` computed by hand matches `opt.step()` exactly.

**Momentum** keeps a velocity buffer $b$: $b \leftarrow \mu b + g$, then $p \leftarrow p - \eta b$, with $\mu$ (typically 0.9) saying how much past direction survives. With a constant gradient of 1, $\eta = 0.1$, $\mu = 0.9$, the buffer grows 1, 1.9, 2.71 and the parameter moves to -0.1, -0.29, -0.561.

**Adam** keeps two buffers per parameter: a running mean of gradients $m$ and a running mean of squared gradients $v$, with decay rates $\beta_1 = 0.9$ and $\beta_2 = 0.999$. After correcting both for their start at zero, it steps by $\eta\, \hat m / (\sqrt{\hat v} + \epsilon)$, where the hats mark the corrected averages and $\epsilon$ is a tiny constant against division by zero. So each coordinate steps by roughly $\eta$, whatever the gradient's scale.

<div class="ex">
<div class="lab">Worked example</div>

**Adam ignores gradient scale on its first step**

Two weights start at 1.0. The loss $1000\,w_1 + 0.001\,w_2$ gives gradients 1000 and 0.001. Adam, $\eta = 0.1$.

Step 1: $m = 0.1g$ and $v = 0.001g^2$. Bias correction divides them by $1 - 0.9 = 0.1$ and $1 - 0.999 = 0.001$, giving $\hat m = g$ and $\hat v = g^2$.

So the step is $0.1 \times g / |g| = 0.1$ for both: PyTorch prints `tensor([0.9000, 0.9000])`.

Plain SGD with the same learning rate would move $w_1$ by 100 and $w_2$ by 0.0001.

</div>

The next figure runs the three real optimisers on a bowl that is 50 times steeper in one direction.

~~~dg-fig 04.04-paths.svg
Three torch.optim optimisers for 60 steps on loss = ½(w₁² + 50 w₂²). SGD's learning rate is capped by the steep direction, so it zig-zags and then crawls along the gentle one. Momentum builds speed along the valley and reaches the lowest loss. Adam takes near-diagonal steps because it normalises each coordinate, then hovers near the minimum at a constant learning rate.
~~~

Momentum SGD is the classic choice for vision CNNs; AdamW is the default for Transformers and most new work.

~~~dg-anim gradient-descent | What `optimizer.step()` does on the simplest loss there is. Drag the learning rate past 0.5 to see oscillation, and past 1 to see divergence.
~~~

## Step 5: optimiser state costs memory

An optimiser keeps `param_groups` (dicts of `lr`, `weight_decay`, `betas`...) and `state`, a per-parameter dict created lazily on the first `step()`, on the parameter's device. Measure it:

```python
model = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 10))
n = sum(p.numel() for p in model.parameters())
print(n, round(n * 4 / 1e6, 2), "MB")           # 535562 2.14 MB of float32 weights
for opt in [torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
            torch.optim.AdamW(model.parameters())]:
    model(torch.randn(8, 512)).sum().backward(); opt.step()
    st = opt.state[model[0].weight]
    mb = sum(v.numel() * v.element_size() for s in opt.state.values()
             for v in s.values() if torch.is_tensor(v) and v.ndim > 0) / 1e6
    print(type(opt).__name__, list(st), round(mb, 2), "MB")
# SGD ['momentum_buffer'] 2.14 MB
# AdamW ['step', 'exp_avg', 'exp_avg_sq'] 4.28 MB
```

Momentum adds one extra copy of the model; Adam adds two. For a 1-billion-parameter float32 model that is 8 GB of Adam state on top of 4 GB of weights and 4 GB of gradients, which is why the LLM-scale module counts bytes per parameter.

Build the optimiser **after** moving the model to its device with `model.to(device)`. The PyTorch docs recommend this order, so that the referenced tensors and the state built from them live on the training device.

## Step 6: parameter groups

A **parameter group** is a subset of parameters with its own hyperparameters. The common Transformer recipe applies weight decay to weight matrices but not to biases, normalisation weights or (often) embeddings. A simple rule of thumb is "decay tensors with two or more dimensions":

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
print([len(g["params"]) for g in opt.param_groups])   # [2, 2]: two weight matrices, two biases
```

Keys a group does not set, like `lr` here, come from the constructor defaults. Fine-tuning often gives the pretrained backbone a smaller learning rate than the new head.

Note the difference between `Adam(weight_decay=...)` and `AdamW`. The first adds an L2 term $\lambda p$ to the gradient, which then gets divided by Adam's normaliser, so heavily updated weights are barely decayed. `AdamW` shrinks the weights directly, $p \leftarrow p - \eta \lambda p$, separately from the gradient step. This is called **decoupled weight decay**, and it regularises as intended.

## Step 7: learning-rate schedules

A **scheduler** wraps an optimiser and rewrites `group["lr"]` each time you call `scheduler.step()`. Call it after `optimizer.step()`; the reverse order skips the first value of the schedule and PyTorch warns about it.

- **Per-epoch**: `StepLR`, `MultiStepLR`, `CosineAnnealingLR(T_max=epochs)`.
- **Per-step**: `OneCycleLR`, or `LambdaLR` with warmup then cosine over total steps, standard for Transformers.
- **Metric-driven**: `ReduceLROnPlateau`, stepped with `scheduler.step(val_loss)`. `SequentialLR` and `ChainedScheduler` compose schedules.

<div class="ex">
<div class="lab">Worked example</div>

**warmup plus cosine with LambdaLR**

Total 10,000 steps, 500 warmup steps, peak learning rate 3e-4 set on the optimiser. The lambda returns a multiplier on that peak:

`lam = lambda s: (s + 1) / 500 if s < 500 else 0.5 * (1 + math.cos(math.pi * (s - 500) / 9500))`

Step 250 is in warmup: $(250 + 1)/500 = 0.502$, so lr $= 0.502 \times 3\text{e-}4 \approx 1.5\text{e-}4$.

Step 5,250 is halfway through the decay: $(5250 - 500)/9500 = 0.5$, $\cos(\pi/2) = 0$, so the multiplier is $0.5 \times (1 + 0) = 0.5$, again 1.5e-4 but on the way down.

Step 9,999: $\cos(\pi \times 9499/9500) \approx -1$, so the multiplier is about $3 \times 10^{-8}$ and the run ends with a near-zero learning rate. Step it once per batch.

</div>

~~~dg-fig 04.04-schedule.svg
Left: the schedule read back from a real LambdaLR, with the three hand-checked points. Right: the same scheduler stepped once per epoch (20 epochs of 500 batches) instead of once per batch. It is still in warmup at the end and never exceeds 1.2e-5, under 5% of the intended peak.
~~~

The right panel is the most common schedule bug, and it raises no error: configure the schedule in the unit you step in.

## Losses and optimisers in code

Putting the pieces together for one classification epoch:

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(0)
X = torch.randn(3000, 20); y = X[:, :10].argmax(1)                     # toy 10-class labels, int64
loader = DataLoader(TensorDataset(X, y), batch_size=100, shuffle=True)  # 30 batches per epoch
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))  # .to(device) here, before the optimiser
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
epochs = 10
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * len(loader))

for epoch in range(epochs):
    for xb, yb in loader:
        logits = model(xb)                 # (B, 10) raw scores, no softmax
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()                       # per batch, because total_steps counts batches
    if epoch in (0, epochs - 1):
        print(epoch, round(loss.item(), 3), f"{sched.get_last_lr()[0]:.2e}")
# 0 2.278 8.55e-04
# 9 0.338 1.80e-07
```

Softmax appears only when you need probabilities to show someone; `argmax` on logits already gives the class. To resume training, save all three state dicts: model, optimiser (buffers and groups) and scheduler (current step). Forget the last and a resumed run restarts its warmup.

## Checking that it works

Cheap checks that catch nearly every bug above:

- **Loss at initialisation** should be close to $\ln C$, because a fresh classifier is roughly uniform. The ten-class network above starts at 2.32 against $\ln 10 = 2.30$. Far above means badly scaled logits; far below suggests a label leak.
- **Overfit one batch** for a hundred steps: the loss should reach almost zero. Stalling near 1.46 (ten classes) points to a softmax before the loss.
- **Print** `logits.shape, y.shape, y.dtype` once.
- **Log the learning rate** with `sched.get_last_lr()`. A flat line means the scheduler is not stepped, or is stepped in the wrong unit.
- **Turn warnings into errors** while developing (`python -W error`), so the broadcasting warning stops the run.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Loss plateaus well above zero, accuracy creeps up | softmax (or sigmoid) applied before a logits loss | pass raw logits to `CrossEntropyLoss` / `BCEWithLogitsLoss`
| Shape error, or nonsense loss on sequences | classes not on dimension 1 | `logits.transpose(1, 2)` or flatten to `(B*T, V)`
| Regression predicts a constant | `(N, 1)` predictions vs `(N,)` targets broadcast | `pred.squeeze(-1)`
| Loss stuck at exactly 100 for some samples | `BCELoss` on saturated sigmoid outputs | `BCEWithLogitsLoss`
| Loss becomes NaN after a few steps | learning rate too high, or log of zero in a hand-written loss | lower lr, use fused losses, clip gradients (04.08)
| Resumed run spikes in loss | optimiser or scheduler state not restored | save and load all three state dicts
| Learning rate barely moves | schedule configured in steps, stepped per epoch | match the units

## Going deeper

**Why fused losses are stable.** In float32, $e^{z}$ overflows beyond $z \approx 88$, and tiny probabilities underflow to 0 with log $-\infty$. The fused form $\text{logsumexp}(z) - z_y$ subtracts the maximum logit before exponentiating, so every exponential is at most 1 and nothing overflows. `BCEWithLogitsLoss` uses the analogous identity $\ln(1 + e^{-|z|}) + \max(z, 0) - zt$ for logit $z$ and target $t$.

**The gradient of the double softmax.** With the correct loss, $\partial \ell / \partial z = p - \text{onehot}(y)$. Adding a softmax in front multiplies this by the softmax's Jacobian $\text{diag}(p) - pp^\top$, whose entries are all below 0.25, and the inner probabilities are already in $[0, 1]$, so the outer softmax is nearly flat. Both effects shrink the signal, which explains the slow curve in the measured run.

**Reduction across micro-batches.** `reduction="mean"` with `ignore_index` divides by the number of real targets *in that call*. If you accumulate gradients over micro-batches with different numbers of real tokens, averaging the per-micro-batch means weights each micro-batch equally rather than each token. With 6 and 1 real tokens, the true loss was 1.939 but the mean of means 2.247. Use `reduction="sum"` in each micro-batch and divide by the global token count yourself.

## Key terms

| Term | Meaning
| --- | --- |
| Logits | Raw class scores from the model, before any softmax
| Cross-entropy | Minus the log of the true class's probability, computed from logits
| `ignore_index` | A target value (default -100) excluded from both the loss and its averaging count
| Learning rate | The multiplier on each update step; the most important optimiser hyperparameter
| Momentum | A running sum of past gradients that keeps steps moving in a consistent direction
| Optimiser state | Per-parameter buffers (momentum, `exp_avg`, `exp_avg_sq`) that must be saved with checkpoints
| Parameter group | A subset of parameters with its own `lr`, `weight_decay` and other settings
| Decoupled weight decay | AdamW's shrinking of weights directly, instead of adding an L2 term to the gradient
| Scheduler | An object that rewrites the learning rate in `param_groups` each time it is stepped

<div class="take">
<div class="lab">Key takeaway</div>

feed losses raw logits in the layout they expect (classes on dimension 1, int64 targets, matching shapes for regression), remember that the optimiser carries state as large as one or two copies of the model, and treat the learning rate, parameter groups and schedule as part of the checkpointed training state.

</div>
