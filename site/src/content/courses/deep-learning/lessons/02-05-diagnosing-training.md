---
id: "02.05"
course: "deep-learning"
module: "02"
slug: "diagnosing-training"
title: "Diagnosing training"
keyIdeas: ["Neural networks **fail silently**: a buggy pipeline usually still trains a little, so run a fixed routine of cheap checks before spending compute on tuning.", "Check the loss at step 0 before anything else: a $C$-class classifier with sensible init should start near $\\ln C$ (2.30 for 10 classes). Far above means the logits are too large; far below means label leakage or a wrong loss.", "Overfitting a single small batch to near-zero loss is the fastest end-to-end test. If the model cannot memorise 16 examples, the bug is in the pipeline (double softmax, detached tensors, misaligned labels), not the hyperparameters.", "Read loss curves by shape: flat, early divergence, a jump followed by a flat line at $\\ln C$ (a dead network), oscillation, a train/validation gap and a high plateau each point to a different cause.", "Log per-layer gradient norms, update-to-weight ratios (around $10^{-3}$ is healthy for SGD) and the fraction of dead ReLUs; vanishing and exploding gradients show up there long before they show up in accuracy."]
question: "Your new model starts at loss 0.01 on a 100-class classification problem and quickly goes to 0.0001, but validation accuracy is 1%. Walk through what the initial loss tells you, the most likely class of bug, and two concrete checks you would run."
answer: "With small initial logits a 100-class model predicts roughly uniform probabilities, so the loss at step 0 should be near $\\ln 100 \\approx 4.6$. A starting loss of 0.01 means the model assigns probability $e^{-0.01} \\approx 0.99$ to the correct label before it has learned anything, so either the label is effectively available in the input or the loss is not measuring what you think. Likely bugs: label leakage (a feature derived from the target, such as the class index encoded in a filename or ID column, or a one-hot target accidentally concatenated into the input), comparing predictions against themselves, or a loss with the wrong reduction or axis (for example cross-entropy taken over the batch dimension, so each 'class' is an example). Validation accuracy of 1% is exactly chance for 100 classes, which confirms the model learned nothing that transfers: the shortcut exists in the training inputs but not in the validation inputs (or the validation path is built differently). Checks: (1) decode one batch exactly as it is fed to the model and confirm the targets are not present in the inputs, and that training and validation batches are built by the same code; (2) compute the loss by hand for one example from its logits and label (log-sum-exp of the logits minus the true-class logit) and compare it with the framework's value, checking the shapes of the logits (batch, classes) and the targets (batch,); (3) shuffle the labels relative to the inputs and confirm the initial loss becomes about 4.6 and cannot fall quickly. If shuffling does not change the loss, the loss is not using the labels correctly; if it does, look for the leaking feature."
sources: [{"title": "Karpathy: A Recipe for Training Neural Networks", "url": "https://karpathy.github.io/2019/04/25/recipe/", "note": "The step-by-step routine: inspect data, verify init loss, overfit a batch, then regularise and tune."}, {"title": "CS231n: Neural Networks Part 3", "url": "https://cs231n.github.io/neural-networks-3/", "note": "Gradient checks, loss-curve shapes, update/weight ratios and activation histograms."}, {"title": "Google Research: Deep Learning Tuning Playbook", "url": "https://github.com/google-research/tuning_playbook", "note": "A systematic process for choosing what to tune and reading training curves."}]
practice: ["learning-rate-sweep", "finite-difference-gradient-checking", "early-stopping", "lr-range-test-and-one-cycle-policy", "gradient-clipping"]
tags: ["debugging", "vanishing gradients", "loss curves"]
tools: ["PyTorch"]
---

## What you will learn

- Why neural networks "fail silently", and why a fixed routine of checks beats guessing at hyperparameters.
- The two cheapest tests in deep learning: the loss at step 0 (it should be about $\ln C$) and overfitting one small batch.
- How to read a loss curve by its shape: flat, diverging, dying, plateauing, or splitting into a train/validation gap.
- How to look inside the network with per-layer gradient norms, update-to-weight ratios, activation statistics and dead-unit counts.
- Where NaNs come from and how to find the first operation that produced one.

*Before you start:* cross-entropy and softmax (01.03), backpropagation (01.04), initialisation (01.05), and this module's knobs: normalisation (02.01, 02.02), regularisation (02.03), learning rate and clipping (02.04).

## Training fails silently

The previous lesson (02.04) set up a training run: batch size, learning rate, warmup, schedules, clipping. This one is about what to do when the run does not work, which is most runs the first time.

The hard part is that a buggy network usually still trains. A bug in ordinary code raises an exception; a bug in a training pipeline (misaligned labels, a softmax applied twice, a frozen layer) just makes the model somewhat worse. Andrej Karpathy's summary is that neural nets **fail silently**.

The defence is a fixed routine of cheap checks, run cheapest first, each ruling out a whole class of bugs before you spend compute on tuning:

~~~dg-flow A debugging order that rules out cheap bugs before expensive tuning.
Inspect batch -(inputs ok)-> Initial loss ≈ ln C -(yes)-> Overfit 16 examples -(reaches ≈ 0)-> Full run + monitors -(gap or plateau)-> *Tune regularisation, lr, capacity
~~~

Every figure here comes from real runs of a small multi-layer perceptron (MLP) on scikit-learn's 8×8 handwritten digits (1,797 images, 10 classes).

## The idea in one picture

Before training, a 10-class classifier knows nothing, so the honest prediction is "each class with probability 1/10". The cross-entropy loss for one example is $-\ln p_{\text{true}}$, so a model that guesses uniformly has loss

$$-\ln \tfrac{1}{10} = \ln 10 \approx 2.303.$$

With standard initialisation the output **logits** (raw scores before softmax) are small, softmax turns them into nearly equal probabilities, and the loss starts close to $\ln C$ for $C$ classes:

| Number of classes $C$ | 2 | 10 | 100 | 1000
| --- | --- | --- | --- | --- |
| Expected loss at step 0, $\ln C$ | 0.693 | 2.303 | 4.605 | 6.908

The figure checks this on the digits network, multiplying the last layer's initial weights by a scale factor and recording the loss before any training.

~~~dg-fig 02.05-initloss.svg
Loss before training against the scale of the last layer's initial weights. At the default scale it is 2.31, right on ln 10. Make the output weights 100 times too large and the untrained network is confidently wrong, starting at 18.
~~~

Read it like this:

- With PyTorch's default initialisation the starting loss was 2.309, within 0.01 of $\ln 10$: a healthy start.
- At 10× scale it is 2.89, at 100× it is 18.0. Large logits make softmax bet almost everything on one essentially random class, and a confident wrong bet costs a lot.
- A starting loss *far below* $\ln C$ is just as suspicious: the model "knows" the answer before learning, which usually means the label leaked into the input.

## Step 1: look at the data the model actually sees

The most common bugs are in the data, not the model. Decode a batch *exactly as the model receives it*: after loading, augmentation, normalisation, padding and batching. Plot the images, print the token ids back as text, and print the labels next to them.

This catches, in minutes, labels shuffled apart from inputs, normalisation applied twice or not at all, augmentation that destroys the signal (a flip that turns a "6" into a "9"), and padding tokens counted in the loss.

Also set a **baseline**: a simple model whose score tells you what "working" looks like. On the digits data plain logistic regression reaches 98.2% validation accuracy in our split, so a deep network at 60% is broken, not "still tuning".

## Step 2: check the loss at step 0

Before the first update, compute the loss on a batch and compare it with the uniform-guess value: $\ln C$ for classification, or about 1 for regression with standardised targets and outputs near zero. What the deviations mean:

- **Much higher** (say 40 for 10 classes): logits too large, from an over-scaled output layer or unnormalised inputs.
- **Much lower** (say 0.01): label leakage, or a loss computed against the wrong thing.
- **Right at $\ln C$ but it never moves**: gradients are not reaching the weights (see the "network dies" curve below).

A useful trick: initialise the last layer's weights small and set its bias to the log of the class frequencies, so the model starts by predicting the base rates.

<div class="ex">
<div class="lab">Worked example</div>

**the loss at step 0, by hand**

A 3-class model gives logits $z = (0.1, -0.2, 0.05)$ for an example whose true class is the first one.

Exponentiate: $e^{0.1} = 1.1052$, $e^{-0.2} = 0.8187$, $e^{0.05} = 1.0513$. Their sum is $2.9752$.

Softmax: $p = (1.1052, 0.8187, 1.0513) / 2.9752 = (0.3715, 0.2752, 0.3533)$.

Loss: $-\ln 0.3715 = 0.990$. The uniform-guess value is $\ln 3 = 1.099$.

Small, nearly equal logits give a loss close to $\ln 3$; averaged over many examples it comes out closer still.

</div>

## Step 3: overfit one small batch

Take 8 to 32 examples, switch off augmentation, dropout and weight decay, and train on that same tiny batch over and over. A network with thousands of parameters can memorise 16 examples, so the loss should fall to almost zero.

If it does not, the problem is wiring (or an architecture that cannot pass signal, as in the diagnosis below), not hyperparameters. The test exercises the whole pipeline in under a minute.

~~~dg-fig 02.05-overfit.svg
Training on the same 16 digit images for 300 Adam steps. The correct pipeline reaches a loss of 0.0003. A double softmax gets stuck at exactly 1.46. Labels shuffled independently of the inputs get stuck near 1.75, because there is nothing left to memorise.
~~~

The two bugs in the figure are real and common:

- **Softmax fed into `CrossEntropyLoss`.** That loss expects raw logits and applies log-softmax itself, so softmax gets applied twice and the loss has a floor (worked out below).
- **Labels shuffled apart from inputs.** The best the model can do is predict how often each label appears; for this batch that has entropy 1.77, matching the plateau.

<div class="ex">
<div class="lab">Worked example</div>

**why the double-softmax loss cannot go below 1.46**

Ten classes. After the first softmax the best possible output is $(1, 0, 0, \dots, 0)$ with the 1 on the true class.

The loss applies softmax again to these numbers: the true class gets $e^1 = 2.718$, each of the 9 others gets $e^0 = 1$.

Probability of the true class: $2.718 / (2.718 + 9) = 2.718 / 11.718 = 0.232$.

Loss: $-\ln 0.232 = \ln(1 + 9/e) = \ln(1 + 3.311) = \ln 4.311 = 1.461$.

The buggy run ends at 1.4612. When a loss plateaus at a suspiciously exact value, work out which bug would produce it.

</div>

The test also catches a loss over the wrong axis, tensors detached from the graph (`.item()` or `.numpy()` inside the forward pass), parameters never passed to the optimiser and a forgotten `zero_grad()`.

## Step 4: run for real and read the loss curve

Now train on the full data and plot training and validation loss together. Shapes say more than single numbers.

~~~dg-fig 02.05-curves.svg
Four real runs of the same network on the digits data. Learning rate 1e-5: the loss barely moves from 2.31. Learning rate 0.01: both curves fall smoothly. Learning rate 0.5: the loss jumps around, then the network dies and sits at ln 10 forever. 60 training examples: training loss reaches zero while validation loss bottoms out at epoch 7 and then rises.
~~~

In the "network dies" panel, after the jumps, 97.7% of the units in the second hidden layer were **dead**: their ReLU output was zero for every training example. A dead ReLU passes no gradient, so nothing revives it, and the network falls back to a constant prediction with loss $\ln 10$. The flat, calm curve fools people.

| Curve shape | Likely cause | First thing to try
| --- | --- | --- |
| Flat from step 0 | lr far too low, dead network, frozen params, data bug | check gradients are non-zero; lr range test
| Decreases, then NaN or spike | lr too high, no warmup, fp16 overflow, log(0) | lower lr, add warmup and clipping
| Jumps, then flat at $\ln C$ | lr so high that ReLUs died | lower lr, warmup; check dead-unit fraction
| Noisy, oscillating, not improving | lr too high for batch size | lower lr or increase batch
| Train falls, validation rises | overfitting | augmentation, dropout, weight decay, early stopping
| Both plateau high | underfitting or too-small lr late in training | bigger model, longer schedule, check capacity
| Sudden drops at fixed epochs | step lr schedule | expected, not a bug

Use a log scale when the loss spans orders of magnitude, smooth per-step loss with a moving average, and compute validation loss in eval mode (02.01).

## Step 5: look inside the network

A loss curve tells you *that* something is wrong; per-layer statistics tell you *where*. Four are worth logging.

**Per-layer gradient norms.** In a healthy network they are within an order of magnitude or two of each other. Early-layer gradients a thousand times smaller than late-layer ones are **vanishing gradients**; norms that grow toward the input are **exploding gradients**.

~~~dg-fig 02.05-gradnorms.svg
Three 12-layer networks at initialisation, one batch of 256 digits. With sigmoid activations and default init, gradient norms fall from 3e-2 at the last hidden layer to 1e-10 at the first: eight orders of magnitude. ReLU with He init keeps every layer between 0.18 and 0.30, and adding BatchNorm also keeps the activation spread constant.
~~~

The cause is the one from 01.02: a sigmoid's slope is at most 0.25, so every layer shrinks the backward signal and twelve layers compound it. Fixes: non-saturating activations (01.02), variance-preserving init (01.05), normalisation (02.01), residual connections (03.04). Clipping (02.04) caps exploding gradients but does not fix their cause.

**Update-to-weight ratio.** For each layer, compare the size of one update with the size of the weights:

$$\text{ratio} = \frac{\|\eta\, g\|}{\|W\|}$$

where $\eta$ is the learning rate, $g$ the layer's gradient, $W$ its weights, and $\|\cdot\|$ the square root of the sum of squared entries. For Adam, measure the actual change $\|W_{t+1} - W_t\|$ instead. A rule of thumb from Stanford's CS231n notes: about $10^{-3}$ is healthy; $10^{-1}$ means the layer is rewritten every step; $10^{-6}$ means it is effectively frozen.

<div class="ex">
<div class="lab">Worked example</div>

**an update-to-weight ratio by hand**

Fresh digits MLP, plain SGD with learning rate $\eta = 0.1$, one batch of 256 examples. For the second weight matrix the gradient norm is $\|g\| = 0.272$ and the weight norm is $\|W\| = 6.52$.

Size of the update: $\eta\,\|g\| = 0.1 \times 0.272 = 0.0272$.

Ratio: $0.0272 / 6.52 = 0.0042$, or $4.2 \times 10^{-3}$.

Within a factor of a few of $10^{-3}$: a plausible learning rate. With $\eta = 0.001$ the layer would sit at $4 \times 10^{-5}$.

</div>

~~~dg-fig 02.05-ratio.svg
Update-to-weight ratio of one hidden layer during 300 SGD steps. Learning rate 0.001 gives about 5e-5 and the loss stays near 2.27. Learning rate 0.1 starts near 5e-3, drifts toward 1e-3 and trains well. Learning rate 3 starts between 0.15 and 0.9, the loss explodes, and after step 66 the ratio is exactly zero: every unit is dead.
~~~

**Activation statistics.** Log each layer's output mean and standard deviation. Shrinking spreads mean the signal is dying; in the gradient-norm figure the sigmoid activations have a spread of only about 0.07 in every layer, all sitting near 0.5.

**Dead units.** The fraction of ReLUs that are zero for every example in a large batch. A few percent is normal; a jump after a learning-rate spike, like the 97.7% above, is capacity lost for good.

## Diagnosing training in code

The four checks fit in a page of PyTorch and run in seconds.

```python
import math, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
torch.manual_seed(0)
d = load_digits()
X = torch.tensor(d.data, dtype=torch.float32); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
y = torch.tensor(d.target)
def make():
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))
model = make()

# Check 1: loss at step 0 should be close to ln(10)
with torch.no_grad():
    print(f"initial loss {F.cross_entropy(model(X), y):.3f}  vs ln(10) = {math.log(10):.3f}")
# initial loss 2.309  vs ln(10) = 2.303

# Check 2: overfit 16 examples (no augmentation, no regularisation)
xb, yb = X[:16], y[:16]
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(300):
    loss = F.cross_entropy(model(xb), yb)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"loss after 300 steps on 16 examples: {loss.item():.4f}")
# loss after 300 steps on 16 examples: 0.0005

# Check 3: per-layer gradient norm and update/weight ratio (plain SGD step = lr * grad)
def grad_report(model, x, y, lr):
    model.zero_grad(); F.cross_entropy(model(x), y).backward()
    for name, p in model.named_parameters():
        if p.ndim > 1:                                     # weight matrices only
            g, w = p.grad.norm().item(), p.norm().item()
            print(f"{name:9s} grad {g:.2e}  update/weight {lr * g / w:.1e}")
fresh = make()
grad_report(fresh, X[:256], y[:256], lr=0.1)
# 0.weight  grad 1.87e-01  update/weight 2.9e-03
# 2.weight  grad 2.72e-01  update/weight 4.2e-03
# 4.weight  grad 2.37e-01  update/weight 1.3e-02

# Check 4: fraction of dead ReLU units (zero on every example), via a forward hook
acts = {}
h = fresh[3].register_forward_hook(lambda m, i, o: acts.__setitem__("relu2", o.detach()))
with torch.no_grad(): fresh(X)
h.remove()
print(f"dead units in layer 2: {(acts['relu2'] <= 0).all(0).float().mean():.1%}")
# dead units in layer 2: 0.0%
```

In a real project, log these every few hundred steps (TensorBoard or similar) so you can scroll back to the first layer that went wrong.

## Reading the result: a worked diagnosis

<div class="ex">
<div class="lab">Worked example</div>

**reading the symptoms**

A 12-layer sigmoid MLP on the digits data starts at loss 2.32 (fine: close to $\ln 10$). It fails the 16-example test: after 300 Adam steps it is stuck at 1.77, the entropy of the label mix in that batch, so it has learned only how often each class appears. On the full data with SGD (learning rate 0.1, momentum 0.9) the loss is still 2.30 after 30 epochs.

Gradient norms per layer at initialisation: $3.1 \times 10^{-2}$ at the last hidden layer, $6.5 \times 10^{-4}$ two layers earlier, and $1.0 \times 10^{-10}$ at the first layer.

Activation standard deviation is about 0.07 in every layer: the sigmoids all output nearly 0.5.

Diagnosis: the signal vanishes with depth; this is not a learning-rate problem. A learning rate large enough to move the first layer (millions of times larger) would blow up the last layers.

Fixes in order of effort: ReLU with He init (every layer's gradient norm then sits between 0.18 and 0.30), BatchNorm after each linear layer, then residual connections.

</div>

The general rule: find the *first* layer or step where a statistic leaves its healthy range, and fix the cause there.

## NaNs and infinities

A NaN ("not a number") spreads to everything it touches, so within one step the whole model is NaN. Usual sources:

- Learning rate too high: the loss spikes, weights overflow, then NaN.
- `log(0)` or division by zero: hand-written `log(softmax(z))` instead of the logits-based losses (01.03), or a normalisation variance without its $\epsilon$.
- **fp16 overflow** above 65504: use bf16 or loss scaling.
- Bad data, such as an empty mask making a mean divide by zero.

`torch.autograd.set_detect_anomaly(True)` raises an error at the first backward operation that produced a NaN and names the forward operation responsible. It is slow, so use it only to locate a bug; a cheap everyday guard is to check `torch.isfinite(loss)` and skip and log the batch when it fails.

## What goes wrong

| Symptom | Likely cause | Check or fix
| --- | --- | --- |
| Starting loss far above $\ln C$ | output weights too large, unnormalised inputs | shrink last-layer init; set bias to log class frequencies
| Starting loss far below $\ln C$ | label leakage, wrong loss target | decode a batch; shuffle labels and confirm loss rises to $\ln C$
| Cannot overfit 16 examples | wiring bug: double softmax, detached tensor, params missing from optimiser, wrong axis | compute the loss by hand for one example; check every parameter's `.grad` is non-zero
| Early layers have tiny gradients | vanishing gradients | ReLU-type activations, He init, normalisation, residuals
| Many dead ReLUs after a spike | lr spike killed units | lower lr; leaky ReLU or GELU; restart from a checkpoint
| NaN after hours | rare overflow, bad sample, fp16 | `isfinite` guard, bf16, loss scaling, anomaly detection

## Going deeper

**Gradient checking.** If you write a backward pass by hand (01.04, 02.02), compare it with **central finite differences**: nudge one parameter by $\pm h$, rerun the forward pass, and estimate the derivative as

$$\frac{\partial f}{\partial w} \approx \frac{f(w + h) - f(w - h)}{2h}.$$

Its error shrinks like $h^2$. Compare analytic gradient $a$ and numeric gradient $n$ with the **relative error** $|a - n| / \max(|a|, |n|)$: in float64, about $10^{-6}$ or smaller means correct code, around $10^{-2}$ means a bug. Kinks (ReLU at 0) give false alarms, and float32 has too few digits. Check: for $f(w) = w^3$ at $w = 2$, $h = 10^{-3}$ gives $(8.012006 - 7.988006)/0.002 = 12.000001$ against the exact 12, a relative error of $8 \times 10^{-8}$. `torch.autograd.gradcheck` automates this.

**Why ratios, not raw gradients.** A gradient of 0.1 is huge for a weight of size 0.01 and negligible for one of size 100; dividing by the weight norm makes layers comparable. Adam rescales each coordinate, so its update size is set mostly by the learning rate, which is one reason Adam learning rates around $10^{-4}$ to $10^{-3}$ transfer across architectures.

**Why dead ReLUs stay dead.** A ReLU with pre-activation $a = w \cdot x + b$ passes gradient only when $a > 0$. If one large update makes $a < 0$ for every input, the loss gradients for $w$ and $b$ are exactly zero, so gradient steps can no longer move them. Leaky ReLU and GELU keep a small slope for $a < 0$, leaving a path back.

## Key terms

| Term | Meaning
| --- | --- |
| Fail silently | A bug that makes training worse without crashing or raising an error
| Initial loss check | Comparing the step-0 loss with $\ln C$, the uniform-guess loss
| Overfit one batch | Training on 8 to 32 examples until the loss is near zero, to test the pipeline end to end
| Vanishing / exploding gradients | Gradient norms that shrink or grow by orders of magnitude from layer to layer
| Update-to-weight ratio | Size of one update divided by the size of the weights; about $10^{-3}$ is a healthy rule of thumb
| Dead ReLU | A unit whose output is zero for every input, so it receives no gradient and cannot recover
| Gradient check | Comparing a hand-written gradient with a finite-difference estimate via the relative error

<div class="take">
<div class="lab">Key takeaway</div>

before tuning anything, check that the loss starts near $\ln C$ and that the model can overfit one small batch; then read loss curves by shape, and log per-layer gradient norms, update-to-weight ratios and dead units to see where the signal vanishes, explodes or dies.

</div>
