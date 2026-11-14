---
id: "02.01"
course: "deep-learning"
module: "02"
slug: "batch-normalisation"
title: "Batch normalisation"
keyIdeas: ["BatchNorm takes each unit's values across the current mini-batch, subtracts their mean and divides by their standard deviation, then applies a learned scale $\\gamma$ and shift $\\beta$. Because $\\gamma$ and $\\beta$ can undo the normalisation, nothing is lost in what the network can represent.", "For an MLP the statistics are per feature over the $N$ examples; for a CNN they are per channel over $N \\cdot H \\cdot W$ values. A layer directly before BatchNorm needs no bias, since $\\beta$ replaces it.", "In `model.train()` the layer uses batch statistics and updates running averages (PyTorch momentum 0.1 = weight on the new value); in `model.eval()` it uses the running averages, so outputs no longer depend on the other examples in the batch.", "It keeps activations at a steady scale through depth and makes a much wider range of learning rates and initialisations work. The modern explanation is a smoother loss landscape, more than the original \\\"internal covariate shift\\\" story.", "It breaks when batches are tiny (noisy statistics), when you forget `model.eval()` at test time, when test data differ from training data, and when examples in a batch must not influence each other; LayerNorm or GroupNorm (02.02) avoid the batch entirely."]
question: "A model scores 92% validation accuracy when evaluated with batch size 256 but 61% when the same weights serve requests one at a time. The serving code never calls `model.eval()`. Explain what happens inside each BatchNorm layer in both settings, why calling `eval()` fixes serving, and what test you would add so this cannot happen again."
answer: "Without `eval()` every BatchNorm layer stays in training mode and normalises each channel with the statistics of the current batch rather than with the running averages learned during training.\n\nWith 256 validation examples, the per-channel batch mean and variance over $N \\cdot H \\cdot W$ values are close to the population statistics the network was trained against, so each layer's outputs look almost normal and accuracy is close to the eval-mode figure. (A quick experiment on the digits dataset shows the same pattern: 98.1% in eval mode, 97.7% in train mode with batches of 128, but 20% with batches of 2.)\n\nWith one request at a time, the statistics come from a single example. In a convolutional layer the mean and variance are taken only over that image's spatial positions, so every channel is re-centred on its own average: the information \"this channel is unusually active for this image\" is erased at every layer. In a fully connected layer with $N = 1$ the variance is zero; PyTorch's BatchNorm1d refuses to run and raises an error, and a hand-written version would output $\\beta$ for every input. Either way, the features the classifier sees bear little relation to what it saw during training, so accuracy collapses. Training mode also keeps updating the running averages with serving traffic, slowly corrupting them.\n\n`model.eval()` makes each BatchNorm layer use its stored running mean and variance: a fixed per-channel affine map $\\gamma (x - \\hat\\mu)/\\sqrt{\\hat\\sigma^2 + \\epsilon} + \\beta$ that does not depend on which other examples share the batch, and does not update anything. A single request then gets exactly the output it would get inside any batch. Also wrap inference in `torch.no_grad()` (or `torch.inference_mode()`), and consider folding BatchNorm into the preceding convolutions for deployment, which removes the mode question entirely.\n\nTo prevent a regression, add a test that runs one example alone and the same example inside a batch of random others through the serving path and asserts the outputs are equal up to floating-point tolerance; it fails immediately if any BatchNorm (or dropout) layer is left in training mode."
sources: [{"title": "Ioffe & Szegedy: Batch Normalization", "url": "https://arxiv.org/abs/1502.03167", "note": "Algorithms 1 and 2 give the training transform and the inference procedure with population statistics."}, {"title": "Santurkar et al.: How Does Batch Normalization Help Optimization?", "url": "https://arxiv.org/abs/1805.11604", "note": "Experiments against the covariate-shift explanation and the smoothness argument."}, {"title": "PyTorch docs: BatchNorm2d", "url": "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html", "note": "The momentum convention, biased vs unbiased variance, and track_running_stats."}, {"title": "Dive into Deep Learning: Batch Normalization", "url": "https://d2l.ai/chapter_convolutional-modern/batch-norm.html", "note": "From-scratch implementation and discussion of train vs predict mode."}]
practice: ["batch-normalization", "apply-alphago-residual-block", "resnet-residual-block"]
tags: ["batch norm", "normalisation"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- What a batch normalisation (BatchNorm) layer computes, step by step, with numbers you can check by hand.
- Why it carries two learned parameters, $\gamma$ and $\beta$, and why a layer feeding into it needs no bias.
- Which axes the statistics cover for an MLP $(N, D)$ and a CNN $(N, C, H, W)$.
- Why the layer behaves differently in `model.train()` and `model.eval()`, and how the running averages are kept.
- What it buys in practice (a wide range of usable learning rates, robustness to initialisation) and why.
- Where it breaks: tiny batches, forgotten mode switches, shifted test data, examples that must stay independent.

*Before you start:* layers and activations (01.01), backpropagation (01.04), and why initialisation keeps activations at a sensible scale (01.05). You also need the mean and variance of a list of numbers.

## Where this module starts

Module 01 gave you a network that can, in principle, learn. This module is about making deep networks train **reliably**: normalisation layers (this lesson and 02.02), regularisation (02.03), the mechanics of mini-batch training (02.04), and how to read the signs when training goes wrong (02.05). The optimiser update rules themselves live in the ML Maths course.

Lesson 01.05 showed the core difficulty: a slightly wrong weight scale compounds layer after layer. Careful initialisation sets the scale right *at the start*, but nothing keeps it right once the weights move. BatchNorm measures the scale of each unit on every mini-batch and resets it, at every step.

## The idea in one picture

Take one hidden unit in the middle of a network and look at the value it produces for each of the 8 examples in a mini-batch:

| Example | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raw value $x$ | 52 | 61 | 47 | 58 | 55 | 49 | 63 | 55
| Normalised $\hat x$ | -0.57 | 1.15 | -1.53 | 0.57 | 0.00 | -1.15 | 1.53 | 0.00
| Output $y = 1.5\hat x + 0.2$ | -0.66 | 1.92 | -2.10 | 1.06 | 0.20 | -1.52 | 2.50 | 0.20

The raw values sit around 55 with a spread of about 5: a ReLU after this unit would be "on" for every example, and the next layer sees inputs far larger than its initialisation assumed.

~~~dg-fig 02.01-idea.svg
Left: the unit's eight raw values, centred at 55 with standard deviation 5.2. Right: after subtracting the batch mean and dividing by the batch standard deviation they are centred at 0 with spread 1 (circles); the learned scale and shift then move them to wherever training finds useful (diamonds).
~~~

BatchNorm does two things to these numbers:

1. It **normalises** them with the statistics of the current batch: subtract the mean (55) and divide by the standard deviation (5.22). Now the values have mean 0 and standard deviation 1, whatever they were before.
2. It applies a learned **scale** $\gamma$ (here 1.5) and **shift** $\beta$ (here 0.2), so the network can still choose a different mean and spread if that helps.

The **batch statistics** are the mean and variance across the examples of the current mini-batch, separately for each unit. That "across the batch" gives the layer its name and causes every pitfall below.

## Step 1: standardise each unit over the batch

For one unit (one feature), call its values across a mini-batch of $m$ examples $x_1, \dots, x_m$. Compute the batch mean and the batch variance:

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i, \qquad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2.$$

Then normalise each value:

$$\hat x_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}.$$

Symbols: $\mu_B$ is the batch mean, $\sigma_B^2$ the batch variance (divided by $m$, the "biased" version), $\hat x_i$ the normalised value, and $\epsilon$ a tiny constant (PyTorch default $10^{-5}$) that stops a division by zero when all values happen to be equal.

Number check with the table: the deviations from 55 are $(-3, 6, -8, 3, 0, -6, 8, 0)$, their squares sum to $9+36+64+9+0+36+64+0 = 218$, so $\sigma_B^2 = 218/8 = 27.25$ and $\sigma_B = 5.22$. The first value becomes $-3/5.22 = -0.57$, as in the table.

## Step 2: give the layer its freedom back

Forcing every unit to mean 0 and variance 1 forever would be a restriction (a ReLU unit may work best "on" for most inputs). So each unit gets two learned numbers:

$$y_i = \gamma\,\hat x_i + \beta.$$

$\gamma$ (gamma) is a learned scale and $\beta$ (beta) a learned shift, trained by gradient descent like any weight. They start at $\gamma = 1$, $\beta = 0$. If training sets $\gamma = \sigma_B$ and $\beta = \mu_B$, the layer gives back exactly its input, so adding BatchNorm never makes a function impossible to represent. What changes is that each unit's mean and spread are now set by two dedicated parameters instead of emerging from all the weights below it.

A practical consequence: the mean subtraction cancels any constant the previous layer adds, and $\beta$ supplies one anyway, so a layer followed directly by BatchNorm should use `bias=False`.

## Step 3: which numbers count as "the batch"

The statistics are taken per unit, over every value that unit produced in the batch.

- **MLP activations, shape $(N, D)$:** $N$ examples, $D$ features. Each of the $D$ features gets its own $\mu_B$ and $\sigma_B^2$, computed over the $N$ examples. So $m = N$.
- **CNN feature maps, shape $(N, C, H, W)$:** $C$ channels, each an $H \times W$ image. A convolution applies the same filter at every position, so BatchNorm treats all positions of a channel as samples of the same unit and pools over the batch *and* the spatial positions: $m = N \cdot H \cdot W$.

For a CNN the formulas become

$$\mu_c = \frac{1}{NHW}\sum_{n,h,w} x_{n,c,h,w}, \qquad \sigma_c^2 = \frac{1}{NHW}\sum_{n,h,w} (x_{n,c,h,w} - \mu_c)^2, \qquad y = \gamma_c \frac{x - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c.$$

Number check: a batch of 32 maps with 64 channels of size $14 \times 14$ gives each channel $32 \cdot 14 \cdot 14 = 6272$ values to average, and the layer has just $2 \times 64 = 128$ learned parameters.

~~~dg-flow Where BatchNorm sits in a typical CNN block, and the two different statistics it uses.
Conv (no bias) -(N×C×H×W)-> *BatchNorm -(γ, β per channel)-> ReLU -> Next conv
[Train] Batch stats μ_B, σ²_B -(normalise)-> Output -(EMA update)-> Running μ, σ²
[Eval] Running μ, σ² -(normalise)-> *Deterministic output
~~~

## Step 4: two modes, and the running averages

At prediction time you may receive a single example, whose "batch mean" is just itself. So BatchNorm keeps **running statistics**: estimates of each unit's mean and variance over the training data, updated after every training batch as an **exponential moving average (EMA)**:

$$\hat\mu \leftarrow (1 - \alpha)\,\hat\mu + \alpha\,\mu_B, \qquad \hat\sigma^2 \leftarrow (1 - \alpha)\,\hat\sigma^2 + \alpha\,\frac{m}{m-1}\sigma_B^2.$$

$\hat\mu$ and $\hat\sigma^2$ are the running mean and variance (starting at 0 and 1); $\alpha$ is PyTorch's `momentum`, default 0.1. Two details trip people up:

- PyTorch's "momentum" weights the *new* value, the opposite of the optimiser convention (Keras uses the optimiser-style one, default 0.99).
- The batch is normalised with the biased variance (divide by $m$), but the running variance stores the unbiased one (divide by $m-1$), hence the factor $\frac{m}{m-1}$.

~~~dg-fig 02.01-running.svg
A simulation of one unit whose true mean is 4: batch means scatter around it, the running mean with momentum 0.1 settles in a few dozen steps, and with momentum 0.01 it is still at 3.5 after 200 steps.
~~~

The two modes are:

- **`model.train()`:** normalise with the current batch's $\mu_B, \sigma_B^2$, and update the running statistics.
- **`model.eval()`:** normalise with the stored $\hat\mu, \hat\sigma^2$ and change nothing. Each unit then applies a fixed $y = \gamma\frac{x - \hat\mu}{\sqrt{\hat\sigma^2+\epsilon}} + \beta$, so one example gives the same answer whatever else is in the batch.

A small momentum averages more batches (less noise) but lags behind, which matters while the weights, and hence the true statistics, are still changing.

<div class="ex">
<div class="lab">Worked example</div>

**one channel, one training step, then eval**

A channel holds the values $[1, 3, 5, 7]$ across a batch of 4 (spatial size 1, so $m = 4$). Learned parameters: $\gamma = 2$, $\beta = 0.5$. Running statistics start at $\hat\mu = 0$, $\hat\sigma^2 = 1$.

Batch mean: $\mu_B = (1+3+5+7)/4 = 4$. Deviations: $(-3, -1, 1, 3)$. Biased variance: $\sigma_B^2 = (9+1+1+9)/4 = 20/4 = 5$, so $\sqrt{5} = 2.236$.

Normalised: $\hat x = (-3, -1, 1, 3)/2.236 = (-1.342, -0.447, 0.447, 1.342)$.

Training-mode output: $y = 2\hat x + 0.5 = (-2.18, -0.39, 1.39, 3.18)$, with mean $\beta = 0.5$ and spread $\gamma = 2$.

Running mean update: $0.9 \times 0 + 0.1 \times 4 = 0.4$. Unbiased variance: $20/3 = 6.667$. Running variance update: $0.9 \times 1 + 0.1 \times 6.667 = 1.567$.

Eval mode on the same four values uses $0.4$ and $\sqrt{1.567} = 1.252$: $\hat x = (1 - 0.4)/1.252 = 0.479$ for the first value, and so on, giving $y = (1.46, 4.65, 7.85, 11.05)$. Same input, very different output: after one step the running statistics are far from the true ones. After many steps the two modes agree closely.

</div>

## Why it helps

Two figures from runs on scikit-learn's digits dataset (8×8 images of handwritten digits, 1257 training and 540 test examples) show what BatchNorm buys.

First, activation scale. The next figure pushes 256 images through a fresh 20-layer ReLU network of width 64 and measures the spread of each layer's output.

~~~dg-fig 02.01-depth.svg
Spread of activations through 20 layers at initialisation. With PyTorch's default init it shrinks to about 0.05 within three layers; with weights of standard deviation 0.25 it grows to about 960 by layer 20; the same large weights followed by BatchNorm keep it near 0.6 at every depth.
~~~

Without BatchNorm the scale is the product of all the weight scales below (the 01.05 problem); with it, each layer's scale is reset, so the weight scale stops mattering.

Second, learning rates. A 10-layer MLP was trained for 20 epochs with SGD (momentum 0.9, batch 64) at six learning rates, in three versions.

~~~dg-fig 02.01-lr.svg
Left: final test accuracy against learning rate. The default-init network never learns (10% is chance); with Kaiming init it works only up to 0.03; with BatchNorm it reaches 96-98% from 0.003 up to 0.3. Right: training loss at learning rate 0.1, where only the BatchNorm network learns.
~~~

The default-init network is stuck at chance because its signal fades by the last layer. Kaiming init reaches 98.0% at learning rate 0.01 but fails at 0.1 and gives `nan` losses from 0.3. BatchNorm reaches 98.1% at 0.1 and 97.0% at 0.3, breaking down only at 1.0 (51.7%). The benefit is the wide safe *range*: far less tuning of learning rate and init.

Why does it work? The original paper (Ioffe and Szegedy, 2015) said BatchNorm reduces **internal covariate shift**: each layer's input distribution keeps moving as earlier layers update, and normalising pins it down. Santurkar et al. (2018) added random, changing noise *after* each BatchNorm layer, bringing the shift back, and training was barely hurt. Their explanation: BatchNorm makes the loss and its gradients **smoother**, so larger steps are safe. Three further effects are well established:

- **Scale invariance.** Multiply the weights before a BatchNorm layer by 10 and its output does not change, because the normalisation divides the factor out. Bad weight scales cannot blow the network up.
- **Robustness to initialisation**, as the depth figure shows.
- **Mild regularisation.** Each example is normalised with slightly different statistics depending on its batch-mates, which acts like noise injection. BatchNorm networks often need less dropout (02.03).

## Batch normalisation in code

A from-scratch training-mode forward pass in NumPy:

```python
import numpy as np

def batchnorm_train(x, gamma, beta, run_mu, run_var, momentum=0.1, eps=1e-5):
    # x: (N, C, H, W); one mean and variance per channel, over N, H and W
    mu = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)           # biased: used to normalise
    m = x.size // x.shape[1]                             # values per channel = N*H*W
    run_mu[:] = (1 - momentum) * run_mu + momentum * mu.ravel()
    run_var[:] = (1 - momentum) * run_var + momentum * var.ravel() * m / (m - 1)  # unbiased
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma[None, :, None, None] * xhat + beta[None, :, None, None]

rng = np.random.default_rng(0)
x = rng.normal(3.0, 2.0, size=(8, 4, 5, 5)).astype(np.float32)
g, b = np.ones(4, np.float32), np.zeros(4, np.float32)
rm, rv = np.zeros(4, np.float32), np.ones(4, np.float32)
y = batchnorm_train(x, g, b, rm, rv)
print(y.mean(axis=(0, 2, 3)).round(4), y.std(axis=(0, 2, 3)).round(4))   # [0 0 0 0] [1 1 1 1]
# matches torch.nn.BatchNorm2d(4) to within 2.4e-07 (outputs) and 1.2e-07 (running_var)
```

And the hand example in PyTorch, showing the two modes:

```python
import torch, torch.nn as nn

bn = nn.BatchNorm1d(1)                        # one feature; gamma=1, beta=0 at start
with torch.no_grad():
    bn.weight.fill_(2.0); bn.bias.fill_(0.5)  # gamma = 2, beta = 0.5
x = torch.tensor([[1.], [3.], [5.], [7.]])

bn.train()
print(bn(x).squeeze().detach())     # tensor([-2.1833, -0.3944,  1.3944,  3.1833])
print(bn.running_mean, bn.running_var)        # tensor([0.4000]) tensor([1.5667])

bn.eval()
print(bn(x).squeeze().detach())     # tensor([ 1.4587,  4.6545,  7.8502, 11.0459])
print(bn(x[:1]).squeeze().detach()) # tensor(1.4587): one example is fine in eval mode

bn.train()
bn(x[:1])   # ValueError: Expected more than 1 value per channel when training
```

In a real model: `nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU()`, with `model.train()` before training and `model.eval()` before validation, testing or export.

## Checking that it works

**Look at the train/eval gap on the same data.** Score a training subset in train mode and in eval mode; late in training they should be close. A large gap means running and batch statistics disagree: momentum too small, or a pipeline that differs between the two.

**Look at the running statistics.** They should be finite and plausible; a running variance still exactly 1.0 means the layer never ran in train mode.

**Test the mode switch directly.** The commonest deployed BatchNorm bug is serving in train mode. Below, the trained network from the sweep (98.1% in eval mode) is scored in train mode with the test set split into batches of different sizes.

~~~dg-fig 02.01-trainmode.svg
The same trained weights, scored two ways. In eval mode accuracy is 98.1% regardless of batching. In train mode each batch is normalised with its own statistics: 20% with batches of 2, 72% with 8, 94% with 32; only from 128 up does it come close.
~~~

With large batches the bug hides during offline evaluation and appears only when traffic arrives one request at a time. A cheap guard is a unit test that feeds the same example alone and inside a batch and asserts identical outputs.

## What goes wrong

| Problem | Symptom | Fix
| --- | --- | --- |
| Serving or validating without `model.eval()` | Accuracy depends on batch size; same input gives different outputs; running stats drift with test data | Call `model.eval()` (and use `torch.no_grad()`); test single-example vs batched outputs
| Very small batches (1-4 per device) | Noisy statistics, worse accuracy, train/eval gap | Synchronised BN across devices, GroupNorm, or LayerNorm (02.02); freeze BN when fine-tuning
| Test data from a different distribution | Accuracy drops on the new domain only | Recompute running statistics on unlabelled target data (no gradient steps needed)
| Fine-tuning a pretrained model with tiny batches | Pretrained features degrade early | Keep BN layers in eval mode ("frozen BN") while training the rest
| Sequence models with padding or one-token generation | Statistics mix padding and examples; generation sees a batch of one | Use LayerNorm or RMSNorm (02.02)
| Contrastive or self-supervised training | Model "cheats" by detecting which examples share a batch | Shuffle BN across devices, or use a different normaliser

On placement: the original paper put BatchNorm before the nonlinearity, still the CNN default; after the ReLU also works, and neither is universally better.

## Going deeper

**The backward pass.** $\mu_B$ and $\sigma_B$ depend on every example, so each example's gradient depends on the whole batch. Write $s = \sqrt{\sigma_B^2 + \epsilon}$ and $g_i = \gamma\,\partial L/\partial y_i$. Differentiating through the mean and the variance gives

$$\frac{\partial L}{\partial x_i} = \frac{1}{s}\Big(g_i - \frac{1}{m}\sum_j g_j - \hat x_i \cdot \frac{1}{m}\sum_j g_j \hat x_j\Big).$$

In words: remove the incoming gradient's batch mean and its component along $\hat x$, then divide by the batch standard deviation. So the weights below can never change a unit's mean or spread; only $\beta$ and $\gamma$ can.

**Scale invariance and the effective learning rate.** If $\text{BN}(a W x) = \text{BN}(W x)$ for any $a > 0$, then the loss is constant along the direction of $W$. Its gradient is therefore perpendicular to $W$, and its size shrinks like $1/\|W\|$. A step of fixed learning rate thus turns $W$ by an angle of roughly $\eta/\|W\|^2$. Plain gradient steps make $\|W\|$ grow, quietly lowering the effective learning rate; weight decay pulls it back, which is why weight decay still matters on layers whose output scale it cannot affect. In our check, multiplying a convolution's weights by 10 changed the BatchNorm output by at most $7 \times 10^{-5}$, a difference that comes only from $\epsilon$.

**Folding for deployment.** In eval mode BatchNorm is a fixed per-channel affine map, so it can be merged into the preceding convolution: scale each output filter by $s_c = \gamma_c/\sqrt{\hat\sigma_c^2 + \epsilon}$ and set the bias to $\beta_c - s_c\,\hat\mu_c$. Our check gave a maximum difference of $10^{-6}$ between the fused convolution and conv-then-BN. Deployment toolchains do this automatically, so BatchNorm costs nothing at inference.

**Variants and alternatives.**

- **Synchronised BN** pools statistics over all devices' batches, at the cost of communication per layer.
- **Ghost BatchNorm** computes statistics on small virtual sub-batches of a large batch, to keep the regularising noise.
- **Batch Renormalisation** corrects batch statistics towards the running ones during training, shrinking the train/eval mismatch.
- **GroupNorm** and **LayerNorm** normalise within one example, so batch size does not matter (02.02).
- **Normaliser-free networks** train strong image models with no normalisation, using careful initialisation, scaled residual branches and gradient clipping: BatchNorm's benefits can be had other ways.

## Key terms

| Term | Meaning
| --- | --- |
| Batch normalisation | Standardise each unit with the current mini-batch's mean and variance, then apply a learned scale and shift
| Batch statistics | Per-unit mean $\mu_B$ and variance $\sigma_B^2$ over the batch ($N \cdot H \cdot W$ values for images)
| $\gamma$, $\beta$ | Learned per-unit scale and shift; they let the layer represent the identity
| $\epsilon$ | Small constant added to the variance (PyTorch default $10^{-5}$)
| Running statistics | Moving averages of batch mean and unbiased variance, used in eval mode
| Momentum (BatchNorm) | PyTorch: weight on the new batch value in the running average (default 0.1)
| Train vs eval mode | Batch statistics plus running updates, versus fixed running statistics
| Internal covariate shift | The original motivation: layer inputs drift as earlier layers train
| Folding | Merging eval-mode BatchNorm into the previous layer's weights and bias

<div class="take">
<div class="lab">Key takeaway</div>

BatchNorm standardises each unit with the current batch's mean and variance, then rescales it with learned $\gamma$ and $\beta$; this keeps activations at a steady scale and makes a wide range of learning rates and initialisations work. It uses batch statistics in training and running averages in eval, so always switch modes correctly, and reach for LayerNorm or GroupNorm when batches are tiny or examples must not influence each other.

</div>
