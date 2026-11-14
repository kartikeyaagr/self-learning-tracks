---
id: "04.08"
course: "ml-toolkit"
module: "04"
slug: "debugging-models"
title: "Debugging models"
keyIdeas: ["Before training for real, overfit one small batch: a correct model and loop drive the loss on 2 to 16 examples to near zero within a few hundred steps. If it cannot, the bug is in the code, not the data or the hyperparameters. Watch the loss, not just accuracy: softmax before `cross_entropy` still reaches 100 percent accuracy but stalls at a loss of 1.46 for 10 classes.", "Check the loss at initialisation against theory: with $C$ balanced classes and small initial logits, cross-entropy should start near $\\ln C$ (2.30 for 10 classes). A much larger value means overconfident initial logits or a mismatched target format.", "After one backward, list parameters with no gradient, and after one step, list parameters that did not change. These catch detached tensors, frozen layers and a zero learning rate that the overfit test can miss.", "Most shape bugs that do not crash come from broadcasting: an `(N, 1)` prediction minus an `(N,)` target becomes an `(N, N)` table. Assert shapes at module boundaries and be suspicious of any loss whose inputs have different numbers of dimensions.", "NaNs come from a short list: log or sqrt of zero, division by a zero norm, softmax over a fully masked row, fp16 overflow, or a learning rate that blows up. `torch.isfinite` checks in forward hooks and `torch.autograd.detect_anomaly` localise them.", "Exploding gradients show up as gradient-norm spikes before or with loss spikes; vanishing gradients as per-layer norms that shrink toward the input. Log the global norm every step (clipping returns it for free) and per-layer norms occasionally."]
question: "A Transformer language model (vocabulary 32,000) trains fine for 3,000 steps, then the loss becomes nan at step 3,140 and stays nan. The gradient norm log shows values around 1 to 2, then 900 at step 3,138 and inf at 3,139. Describe a systematic way to find the cause, and name two plausible root causes and their fixes."
answer: "First reproduce cheaply: reload the last checkpoint before the spike with the same seeds and data order (seeded workers, deterministic algorithms on), and replay steps 3,130 to 3,140 with per-layer gradient-norm logging, forward hooks asserting finite outputs, and anomaly detection on. That identifies the first layer and op that produce huge or non-finite values, and the exact batch; logging batch indices lets you pull out and inspect those examples. Inspect the batch: an all-padding, extremely long or garbage sequence points at the data. Note that the norm rose to 900 a step before going infinite, so the problem started in the update at step 3,138, not at the NaN. Plausible causes: (1) attention logits growing until softmax saturates or fp16 overflows (values above 65,504); fixes are bfloat16, query-key normalisation or logit capping, a lower peak learning rate or longer warmup. (2) A fully masked row producing softmax over all -inf, which gives nan in the forward pass; fix the masking or filter empty sequences. A third candidate is a corrupted or unnormalised batch producing a huge gradient, which clipping would have absorbed. In every case add global-norm clipping (for example max_norm 1.0) with `error_if_nonfinite=True`, or skip and count non-finite steps, so one bad batch cannot destroy the weights. As a sanity check, the loss at step 0 should have been near ln 32,000 = 10.4; if it was, initialisation is not the issue."
sources: [{"title": "Karpathy: A recipe for training neural networks (2019)", "url": "https://karpathy.github.io/2019/04/25/recipe/", "note": "The overfit-one-batch, check-init-loss and visualise-the-data habits in detail."}, {"title": "PyTorch docs: torch.autograd (anomaly detection and gradcheck)", "url": "https://pytorch.org/docs/stable/autograd.html", "note": "detect_anomaly, set_detect_anomaly and gradcheck usage."}, {"title": "PyTorch docs: Frequently asked questions", "url": "https://pytorch.org/docs/stable/notes/faq.html", "note": "Out-of-memory causes such as accumulating history across iterations, and other common errors."}, {"title": "PyTorch docs: Reproducibility", "url": "https://pytorch.org/docs/stable/notes/randomness.html", "note": "Seeding, deterministic algorithms and DataLoader worker seeding, needed to replay a failing step."}]
practice: ["finite-difference-gradient-checking", "gradient-clipping", "softmax-numerical-stability-and-log-sum-exp", "mutation-trap", "tensor-shape-manipulation"]
tags: ["debugging", "NaNs", "exploding gradients"]
tools: ["PyTorch"]
---

## What you will learn

- Why model bugs are usually silent, and how to make them fail loudly and early.
- A five-rung checklist for every new model: inspect the data, check the step-0 loss, overfit one batch, check gradients reach every parameter, check batch independence.
- How silent broadcasting corrupts a loss, worked by hand on three numbers.
- Where NaNs come from, and how `detect_anomaly`, forward hooks and `error_if_nonfinite` find the exact operation.
- How gradient norms reveal exploding and vanishing gradients, and how `gradcheck` tests a hand-written backward.

*Before you start:* autograd and `.grad` (04.02), `nn.Module` (04.03), `cross_entropy` and optimisers (04.04), the training loop with clipping (04.06); mixed precision (04.07) appears briefly.

## Neural network bugs are quiet

The previous lesson (04.07) made training fast and resumable. This last lesson of the module is about making it *correct*.

A typo in a web service throws an exception. A typo in a model usually produces a model that trains a bit worse. Softmax over the wrong axis, dropout left on at evaluation, a frozen layer, a loss computed on probabilities: all of them run, and most still make the loss go down. So debugging a model means building checks that **fail loudly**: small tests with a known right answer, run before the expensive job. The Deep Learning course covers how to read loss curves and diagnose dead or saturated units; this lesson is the PyTorch-side toolkit.

## The idea in one picture

The single most useful check, on a tiny classifier (20 inputs, 64 hidden units, 10 classes): train on one batch of 16 examples over and over. A working model can easily memorise 16 examples, so its loss must fall to nearly zero. Four versions, 200 Adam steps each:

| Version | Loss at step 0 | Loss at step 200 | Verdict
| --- | --- | --- | --- |
| Correct model | 2.33 | 0.0002 | passes
| Backbone output wrapped in `.detach()` | 2.33 | 0.024 | passes (slowly), needs another check
| `Softmax` layer before `cross_entropy` | 2.30 | 1.461 | stuck: bug in the loss wiring
| LR schedule that returns 0 | 2.33 | 2.33 | flat: nothing is learning

~~~dg-fig 04.08-overfit.svg
Training on the same 16 examples every step. Only the correct model drops to zero quickly; softmax-before-cross-entropy stalls at exactly 1.46, and a zero learning rate leaves the loss flat. The detached model still memorises the batch, so this test alone does not catch it.
~~~

A flat line means no parameter is moving. A line that hits a hard floor means the loss cannot reach zero by construction. A line that falls, only more slowly, is the hardest case, and the reason the checklist has more than one rung.

Two terms used throughout: a **sanity check** is a cheap test whose correct outcome you know in advance, so any other outcome proves a bug. The **gradient norm** is the length of the vector of all gradients, $\|g\| = \sqrt{\sum_j g_j^2}$: one number per step saying how hard the optimiser is being pushed.

~~~dg-flow The debugging ladder: each rung is cheap and rules out a whole class of bugs before you pay for a long run.
Inspect data and labels -> Loss at init = ln C? -> *Overfit one batch -> Every parameter gets a gradient -> Batch independence -> Scale up
~~~

## Step 1: look at the data after the loader

Decode one batch exactly as the model sees it, after the `DataLoader`, augmentation and normalisation, back into images or text, and print each label beside it. This catches labels shuffled separately from inputs, an augmentation that turns a 6 into a 9, images normalised twice, or batches that are mostly padding. None of these crash or show up in the model code. Also print `x.dtype`, `x.shape`, `x.min()`, `x.max()` and the label counts: integer pixels in `[0, 255]` where the model expects floats near `[-1, 1]` are a classic.

## Step 2: check the loss at initialisation

A freshly initialised classifier knows nothing, so it should spread its probability evenly over the $C$ classes: each class gets $1/C$. Cross-entropy for one example is minus the log of the probability given to the true class, so the expected starting loss is

$$L_0 \approx -\ln \frac{1}{C} = \ln C.$$

Here $L_0$ is the step-0 loss and $C$ the number of classes. Number check: 10 classes give $\ln 10 = 2.30$; a 50,000-token vocabulary gives $\ln 50000 = 10.8$.

```python
import math, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
x, y = torch.randn(512, 20), torch.randint(0, 10, (512,))
print(F.cross_entropy(model(x), y).item(), math.log(10))   # 2.3205  2.3026
with torch.no_grad():
    model[2].weight.mul_(30)                                # simulate a bad final-layer init
print(F.cross_entropy(model(x), y).item())                  # 11.33
```

Default initialisation gives 2.32, a hair from theory. Scale the final layer by 30 and the loss starts at 11.3: the model is confidently wrong on most examples.

~~~dg-fig 04.08-init-loss.svg
Step-0 loss as the final layer's initial weights are scaled. Anywhere near the default scale the loss sits on the ln 10 line; once the logits grow large, the starting loss climbs fast. A starting loss far above ln C is a bug signal before you have trained a single step.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the starting loss by hand**

Take 3 classes, so theory says $\ln 3 = 1.10$. Suppose a badly initialised model outputs logits $(4, 0, -4)$ and the true class is the third one.

Softmax turns logits into probabilities: $e^4 = 54.60$, $e^0 = 1$, $e^{-4} = 0.0183$. Their sum is $55.62$.

The probability of the true class is $0.0183 / 55.62 = 0.000329$.

The loss is $-\ln 0.000329 = 8.02$, more than seven times the healthy $1.10$.

With small logits such as $(0.04, 0, -0.04)$ every probability is near $1/3$ and the loss near $1.10$. Large initial logits are the whole difference.

</div>

Fix a high starting loss by initialising the final layer smaller (or at zero). A starting loss far *below* $\ln C$ suggests a leak: the label is encoded in the input.

## Step 3: overfit one batch

Take 2 to 16 examples, switch off augmentation, dropout and weight decay, and train on them for a few hundred steps. Loss should approach zero and accuracy reach 100 percent. If not, the bug is in the model, loss or optimiser wiring, since data quality and generalisation are out of the picture.

```python
def overfit_one_batch(model, x, y, steps=300, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    acc = (model(x).argmax(1) == y).float().mean().item()
    return round(loss.item(), 4), acc

xb, yb = x[:16], y[:16]
torch.manual_seed(0)
good = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
print(overfit_one_batch(good, xb, yb))      # (0.0001, 1.0)
torch.manual_seed(0)
bad = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10), nn.Softmax(dim=1))
print(overfit_one_batch(bad, xb, yb))       # (1.4612, 1.0)
```

The buggy model still reaches 100 percent accuracy; only the loss reveals the bug, stalling at a very specific number.

<div class="ex">
<div class="lab">Worked example</div>

**why softmax before cross_entropy stalls at 1.46**

`F.cross_entropy` expects raw **logits** and applies softmax itself. If the model already ends in `Softmax`, its outputs are probabilities between 0 and 1, and the loss applies softmax a second time.

The best the model can do is output 1 for the true class and 0 for the other 9. The second softmax then gives the true class $e^1 / (e^1 + 9 e^0) = 2.718 / 11.718 = 0.232$.

So the lowest possible loss is $-\ln 0.232 = \ln(1 + 9/e) = \ln 4.311 = 1.461$, exactly the floor in the figure.

Fix: delete the `Softmax` layer; apply `softmax` only when you need probabilities for reporting.

</div>

Other common causes of a failed overfit test: frozen parameters, a scheduler that sets the learning rate to zero, an optimiser holding the wrong parameters, and targets in the wrong format.

## Step 4: check that every parameter gets a gradient and moves

The detached model passed the overfit test because its last layer alone could memorise 16 examples. To catch it, list the parameters whose `.grad` is missing or all zero after one `backward()`:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(20, 64), nn.ReLU())
        self.head = nn.Linear(64, 10)
    def forward(self, x):
        feats = self.backbone(x).detach()     # leftover from a frozen-feature experiment
        return self.head(feats)

torch.manual_seed(0); net = Net()
F.cross_entropy(net(xb), yb).backward()
for name, p in net.named_parameters():
    if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0):
        print("no gradient:", name)
# no gradient: backbone.0.weight
# no gradient: backbone.0.bias
```

A gradient can exist and still never be applied (an optimiser built for an older copy of the model, or a zero learning rate), so also clone the parameters, take one `opt.step()`, and print the largest change per parameter. Here `backbone.0.weight` changes by exactly 0.0 while `head.weight` moves by 0.013.

## Step 5: check that examples in a batch stay independent

Example 3's prediction should depend only on example 3's input (BatchNorm in training mode is the deliberate exception, so test in `model.eval()`). A reshape that mixes the batch axis with another axis breaks this, yet the model still trains, just worse. The test: make the input require a gradient, backpropagate the loss of one output row, and see which input rows got a gradient.

```python
def good_flatten(x): return x.reshape(x.shape[0], -1)                   # (B, T, D) -> (B, T*D)
def bad_flatten(x):  return x.transpose(0, 1).reshape(x.shape[0], -1)   # mixes B and T

for fn in (good_flatten, bad_flatten):
    xi = torch.randn(4, 3, 5, requires_grad=True)       # batch of 4
    fn(xi)[2].sum().backward()                          # loss of example 2 only
    print(fn.__name__, (xi.grad.abs().sum((1, 2)) > 0).nonzero().flatten().tolist())
# good_flatten [2]
# bad_flatten [0, 2, 3]
```

## Shape errors and silent broadcasting

Crashing shape errors are the easy kind: read both shapes in the message (`mat1 and mat2 shapes cannot be multiplied (64x512 and 256x10)`) and trace the wrong dimension. The dangerous ones do not crash, because **broadcasting** stretches size-1 axes to make shapes match.

The classic: a regression head `nn.Linear(h, 1)` returns shape `(N, 1)`, the targets are `(N,)`, and `pred - target` silently becomes an `(N, N)` table.

<div class="ex">
<div class="lab">Worked example</div>

**one broadcasting bug, computed by hand**

Predictions $(1, 2, 3)$ stored as a column of shape $(3, 1)$; targets $(1, 2, 4)$ of shape $(3,)$.

Intended: subtract pairwise, $(0, 0, -1)$, square, $(0, 0, 1)$, average: $1/3 = 0.333$.

What happens: both are stretched into the $3 \times 3$ table of $p_i - t_j$. Row 1 is $(0, -1, -3)$, row 2 is $(1, 0, -2)$, row 3 is $(2, 1, -1)$.

Squares: $(0, 1, 9)$, $(1, 0, 4)$, $(4, 1, 1)$. Their sum is $10 + 5 + 6 = 21$, and the mean over 9 cells is $21/9 = 2.333$.

The "loss" is seven times the true value, and minimising it pulls every prediction toward the *mean* of all targets.

</div>

~~~dg-fig 04.08-broadcast.svg
Left: what you meant, three pairs and mean 0.333. Right: what (3,1) minus (3,) computes, a 3 by 3 table of every prediction against every target, mean 2.333. Only the diagonal cells are the comparisons you wanted.
~~~

```python
pred = torch.tensor([[1.], [2.], [3.]])            # (3, 1)
target = torch.tensor([1., 2., 4.])                # (3,)
print((pred - target).shape, ((pred - target) ** 2).mean().item())   # torch.Size([3, 3]) 2.3333
print(F.mse_loss(pred.squeeze(1), target).item())                    # 0.3333
F.mse_loss(pred, target)   # returns 2.3333 and warns: "Using a target size (torch.Size([3]))
                           # that is different to the input size (torch.Size([3, 1]))..."
```

A hand-written loss does not even warn. Similar silent cases: a mean over the wrong `dim`, or a `(B, T)` attention mask applied to `(B, H, T, T)` scores without reshaping it to `(B, 1, 1, T)`. Defences: assert shapes at module boundaries (`assert x.shape == (B, T, D), x.shape`), name each axis in a comment, and prefer `einsum`-style code that names axes.

## Finding NaNs and infinities

Any arithmetic with a NaN (not a number) gives NaN, so one bad value can spread to every weight within a step. The sources form a short list:

| Source | Typical trigger | Fix
| --- | --- | --- |
| `log(0)` or log of a negative | probabilities computed, then logged | work in log space: `log_softmax`, `logsigmoid`, `cross_entropy` on logits
| `sqrt(0)` in backward | norm of an all-zero vector | `torch.linalg.vector_norm`, or add epsilon inside the sqrt
| division by zero | normalising by a sum or std that can be 0 | clamp the denominator
| softmax over all `-inf` | a fully masked row, such as an all-padding sequence | keep one unmasked position, or use a large finite negative
| fp16 overflow | values above 65,504 | bfloat16, or `GradScaler` for gradients (04.07)
| optimisation blow-up | learning rate too high, no warmup, a bad batch | lower LR, warmup, clipping

To find the first bad backward operation, turn on **anomaly detection** for a short run (it records a traceback per op, so it is slow):

```python
v = torch.zeros(3, requires_grad=True)
with torch.autograd.detect_anomaly():
    torch.sqrt((v ** 2).sum()).backward()
# RuntimeError: Function 'PowBackward0' returned nan values in its 0th output.
# (plus a warning showing the forward line that created the op)
```

Read it carefully: the square root's backward produced `inf` (the slope of $\sqrt{u}$ at 0), and the first actual NaN came from the backward of `v ** 2`, multiplying that `inf` by $2v = 0$. The error names the op *next to* the cause, so look one step upstream. `torch.linalg.vector_norm(v)` handles zero correctly.

For forward NaNs, register hooks that check each module's output and report its name:

```python
def add_finite_hooks(model):
    def make_hook(name):
        def hook(module, inputs, output):
            if not torch.isfinite(output).all():
                raise RuntimeError(f"non-finite output from {name} ({type(module).__name__})")
        return hook
    for name, m in model.named_modules():
        if name:                                    # skip the root
            m.register_forward_hook(make_hook(name))
# on a model whose layer "2" holds an inf weight:
# RuntimeError: non-finite output from 2 (Linear)
```

Finally, `clip_grad_norm_(..., error_if_nonfinite=True)` turns a silent NaN step into an exception.

## Exploding and vanishing gradients

`clip_grad_norm_` returns the global norm *before* clipping, so log it every step, plus per-layer norms now and then:

```python
total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
per_layer = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
# fresh seed-0 model: total 0.198; '0.weight' 0.074, '2.weight' 0.171
```

**Exploding gradients** appear as spikes in this log. Below, a small regression network trains normally until step 150, where 8 rows of one batch skipped input normalisation.

~~~dg-fig 04.08-spike.svg
One bad batch at step 150. Without clipping, the huge gradient wrecks the weights; the loss explodes and is nan from step 153 on. With clip_grad_norm_(max_norm=5) the same batch costs one noisy step and training carries on. The gradient-norm log points straight at the step and batch to inspect.
~~~

The unclipped run shows the typical cascade: norm 0.5 at step 149, 8,800 at 150, 37 million at 151, infinity at 152, NaN loss at 153. In Transformers the spike often starts in a few layers (last layers, embeddings, attention logits) before the loss moves. Remedies, roughly in order: lower the peak learning rate or lengthen warmup; clip the global norm; check initialisation and normalisation placement (pre-norm blocks are far more stable than post-norm); for attention, normalise queries and keys or cap the logits.

**Vanishing gradients** are the opposite: per-layer norms shrinking by orders of magnitude toward the input.

~~~dg-fig 04.08-layer-norms.svg
Weight-gradient norm per layer for two 10-layer networks on the same batch. With sigmoid and default init, the layer next to the input receives a gradient tens of millions of times smaller than the last layer; with ReLU and Kaiming init all layers sit within a factor of about 2.
~~~

## Testing a hand-written backward

For a custom `torch.autograd.Function`, `torch.autograd.gradcheck` nudges each input slightly, measures how the output changes, and compares that with your backward. Use small float64 inputs; float32 is too noisy.

```python
class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x); return x ** 2
    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return g * x                     # bug: the derivative of x**2 is 2*x

xt = torch.randn(4, dtype=torch.float64, requires_grad=True)
print(torch.autograd.gradcheck(MySquare.apply, (xt,), raise_exception=False))   # False
```

<div class="ex">
<div class="lab">Worked example</div>

**a classifier stuck at 10 percent**

A 10-class image classifier trains for one epoch (about 1,560 steps) and stays at 10 percent accuracy, loss flat at 2.30.

Rung 2 passes: 2.30 is $\ln 10$, so initialisation is fine. Rung 3 fails: on 16 examples the loss also stays at 2.30, so the bug is in the wiring. Rung 4: every parameter has a non-zero gradient, yet none changes after `opt.step()`.

`opt.param_groups[0]["lr"]` prints `0.0`. The warmup was `LambdaLR(opt, lambda s: min(1, s // warmup))` with `warmup = 2000`: integer division gives 0 for every step below 2,000, so the learning rate is zero for the whole first epoch.

Fix: `lambda s: min(1.0, (s + 1) / warmup)`, which ramps 0.0005, 0.001, ... up to 1. The single batch now drops to near zero and the full run trains. Minutes of checks, instead of days of hyperparameter search that could never have worked.

</div>

## What goes wrong

| Symptom | Likely cause | Check or fix
| --- | --- | --- |
| Loss flat from step 0 | LR is 0, frozen or detached parameters, optimiser holds the wrong parameters | print the LR; parameter-change check
| Loss stalls at a suspicious constant | softmax or sigmoid before a loss that expects logits | feed raw logits to `cross_entropy` or `BCEWithLogitsLoss`
| Evaluation noisy and worse than training | `model.eval()` forgotten: dropout and BatchNorm still in training mode | `model.eval()` plus `torch.no_grad()`
| `RuntimeError: ... modified by an inplace operation` | an in-place op (`x.mul_()`, `x += 1`) overwrote a tensor saved for backward | use the out-of-place version there
| Loss several times too large | `(N, 1)` against `(N,)` broadcasting | match shapes; assert them
| NaN after thousands of good steps | bad batch, LR too high, fp16 overflow | replay with hooks and anomaly mode; clip; bfloat16

## Going deeper

**Why $\ln C$ exactly.** With small logits $z_k$, a first-order expansion gives $p_k \approx \frac{1}{C}(1 + z_k - \bar z)$, with $\bar z$ the mean logit, so the expected loss is $\ln C$ plus a term of the order of the logit variance. Once one logit dominates, the loss on a wrong example is roughly the gap between the largest and the true logit, which grows linearly with the weights: the right side of the init figure.

**Replaying a failure.** Spikes usually come from one batch, so reproduce the exact step: checkpoint often (04.07), seed Python, NumPy, torch and the DataLoader workers, and enable `torch.use_deterministic_algorithms(True)`. Reload the checkpoint before the spike and replay a few steps with hooks, per-layer norms and anomaly detection on. Logging batch indices makes the bad examples easy to find.

**How `gradcheck` decides.** It uses central differences, $\frac{f(x + h) - f(x - h)}{2h}$, with default $h = 10^{-6}$; their error shrinks like $h^2$. Float32 rounding (about $10^{-7}$ relative) divided by $2h$ swamps the answer; float64's $10^{-16}$ does not. `gradgradcheck` does the same for second derivatives.

**Cost of the tools.** Forward hooks add a Python call and an `isfinite` reduction per module per step, and get in the way of `torch.compile` fusion. Anomaly mode stores a traceback per op and can slow training several-fold. The global norm is free because clipping computes it. A good default for long runs: always log loss and global norm, sample per-layer norms, keep cheap shape and `isfinite(loss)` asserts, and enable the heavy tools only during a replay.

## Key terms

| Term | Meaning
| --- | --- |
| Sanity check | A cheap test with a known correct outcome, run before an expensive job
| Loss at initialisation | The step-0 loss; about $\ln C$ for $C$ balanced classes when logits start small
| Overfit one batch | Train on a few fixed examples until the loss is near zero; failure means a wiring bug
| Batch independence | Each example's output depends only on its own input (in eval mode)
| Silent broadcasting | Size-1 axes stretched to make shapes match, producing a wrong result with no error
| Anomaly detection | `detect_anomaly()`: raises at the first backward op returning NaN, with its forward traceback
| Gradient norm | Length of the gradient vector; spikes mean explosion, shrinking per-layer values mean vanishing
| `gradcheck` | Compares an analytic backward with central finite differences, in float64

<div class="take">
<div class="lab">Key takeaway</div>

make bugs loud. Before every new model, look at decoded batches, check that the first loss is near ln C, overfit one batch, and confirm every parameter gets a gradient and moves; then assert shapes, log gradient norms, and localise NaNs with hooks and anomaly detection instead of guessing hyperparameters.

</div>
