---
id: "02.03"
course: "deep-learning"
module: "02"
slug: "dropout-and-regularisation"
title: "Dropout and regularisation"
keyIdeas: ["Dropout zeroes each activation with probability $p$ during training; inverted dropout divides survivors by $1-p$, so the expected activation is unchanged and the layer is simply the identity at test time (call `model.eval()`).", "Dropout approximately trains an ensemble of thinned subnetworks that share weights, and stops units relying on specific partners (co-adaptation). Typical rates: 0.5 for big fully connected layers, 0.1 in Transformers, often none in CNNs with BatchNorm.", "With SGD, L2 regularisation and weight decay are the same thing; with Adam they are not, because the penalty gradient gets divided by the adaptive denominator, so rarely updated weights are over-regularised. AdamW decouples the decay.", "Data augmentation is the strongest regulariser for vision: label-preserving transforms (crops, flips, colour jitter) and label-mixing ones (mixup, CutMix) encode invariances the data alone doesn't show. Augment training data only.", "Judge regularisation by held-out curves: training loss falling while held-out loss rises means overfitting; both high means too much regularisation. Watch held-out loss as well as accuracy, since an overfit model can keep its accuracy but become badly overconfident."]
question: "You fine-tune a model with Adam plus an L2 penalty and find that the embedding rows for rare tokens shrink much faster than the rows for frequent ones, hurting rare-token performance. Explain the mechanism, and how AdamW changes it."
answer: "With L2 inside Adam, the penalty gradient $\\lambda w$ is added to the data gradient and the sum is divided by $\\sqrt{\\hat v}$, the running RMS of that weight's gradients. Frequent tokens' embedding rows receive large, frequent data gradients, so $\\hat v$ is large and the decay term is heavily damped: the step from the penalty is roughly $\\eta\\lambda w/\\sqrt{\\hat v}$, small. Rare tokens' rows receive data gradients only occasionally, so $\\hat v$ is small (it decays between appearances and ends up dominated by the penalty itself), and the same $\\lambda w$ becomes a large normalised step that pulls the row toward zero on every step, even when the token is not in the batch. For example, with $\\lambda w = 0.005$, a row with $\\sqrt{\\hat v} = 1$ moves by $\\eta \\times 0.005$ while one with $\\sqrt{\\hat v} = 0.02$ moves 50 times further. The effective regularisation strength is therefore inversely tied to gradient history, the opposite of what you want. AdamW applies $w \\leftarrow w - \\eta\\lambda w$ outside the adaptive scaling, so every row shrinks by the same fraction per step regardless of its gradient statistics, and Adam's normalisation is applied only to the data gradient. In practice you would switch to AdamW, put biases and normalisation parameters (and often the embedding table) in a parameter group with `weight_decay=0`, and retune $\\lambda$, since the meaningful range changes (0.01 to 0.1 is typical for AdamW). Then check rare-token validation loss to confirm the fix."
sources: [{"title": "Srivastava et al.: Dropout (JMLR 2014)", "url": "https://jmlr.org/papers/v15/srivastava14a.html", "note": "The ensemble interpretation, test-time scaling, and the co-adaptation experiments."}, {"title": "Loshchilov & Hutter: Decoupled Weight Decay Regularization", "url": "https://arxiv.org/abs/1711.05101", "note": "Why L2 and weight decay differ for adaptive optimisers; the AdamW update."}, {"title": "Zhang et al.: mixup", "url": "https://arxiv.org/abs/1710.09412", "note": "The mixup recipe and its effect on generalisation and calibration."}, {"title": "Dive into Deep Learning: Dropout", "url": "https://d2l.ai/chapter_multilayer-perceptrons/dropout.html", "note": "Inverted dropout from scratch and where to apply it."}]
practice: ["dropout", "dropout-2", "adamw-decoupled-weight-decay", "mixup-augmentation", "cutmix-augmentation"]
tags: ["dropout", "weight decay", "augmentation"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- Why a large network can memorise its training set, and what **regularisation** means for deep nets in practice.
- How **dropout** works step by step, why survivors are scaled by $1/(1-p)$, and where to put it in a model.
- Why L2 regularisation and weight decay are the same for SGD but different for Adam, and what **AdamW** changes.
- How **data augmentation** teaches a model invariances, from flips and crops to mixup and CutMix.
- How to read train and held-out curves to tell whether a regulariser is helping, and the common ways each tool goes wrong.

*Before you start:* the training loop and cross-entropy (module 01), overfitting and bias-variance (ML course), and ideally Adam (ML Maths). Lessons 02.01 and 02.02 made deep networks *trainable*; this one makes them *generalise*.

## From trainable to trustworthy

Lessons 02.01 and 02.02 were about getting the training loss down. Once that works, a new problem appears: a modern network has far more parameters than training examples, so it can drive the training loss to zero by memorising, even on completely random labels.

**Regularisation** is any change to training that steers the network toward solutions that generalise, usually by making memorisation harder or less attractive. Beyond the classical view from the ML course, three tools do most of the work in deep learning: dropout, weight decay done correctly, and data augmentation.

## The idea in one picture

Take a tiny problem that is easy to overfit: 80 noisy points from the two-moons dataset (two interleaved half-circles, two classes), and a network with two hidden layers of 256 units each, about 67,000 weights. That is over 800 weights per training point. Train it three ways for 2,000 full-batch steps with AdamW (learning rate 0.003) and test on 2,000 fresh points:

| Setting | Train accuracy | Test accuracy | Final held-out loss
| --- | --- | --- | --- |
| No regularisation | 100% | 83.9% | 3.48
| Dropout, $p = 0.5$ | 98.8% | 81.2% | 2.75
| Weight decay $\lambda = 1.0$ | 100% | 81.3% | 1.04

~~~dg-fig 02.03-boundaries.svg
Solid line: where the model says 50/50. Dashed lines: where it says 10% and 90%. All three boundaries bend around noisy points, but the unregularised model jumps from 10% to 90% almost instantly (only 4% of test points fall between the dashed lines), while weight decay keeps a wider band of honest uncertainty (16%).
~~~

Two things stand out, and both are worth remembering.

- The unregularised model is perfect on training data and much worse on new data: it has memorised noise. Its decision boundary makes little detours around individual points of the wrong colour, and it is near-certain almost everywhere.
- Regularisation did *not* raise test accuracy here; it was a couple of points lower. What it changed dramatically is the **held-out loss**, the cross-entropy on unseen points. The unregularised model is wildly overconfident when it is wrong (an average cross-entropy of 3.48 corresponds to a geometric-mean probability of about 3% for the true class, dragged down by confident mistakes), while weight decay brings that down to 1.04. On a tiny 2-D problem the boundary has little room to improve; on big image and language tasks, the same effect usually also shows up as better accuracy.

## Step 1: what dropout does

**Dropout** randomly switches off units during training. For each training example and each step, every activation in a dropout layer is independently set to zero with probability $p$ (the **dropout rate**) and kept otherwise. The kept ones are scaled up:

$$\tilde h = \frac{m \odot h}{1 - p}, \qquad m_i \sim \mathrm{Bernoulli}(1-p).$$

Here $h$ is the layer's activation vector, $m$ is a random **mask** of 0s and 1s (each entry is 1 with probability $1-p$), $\odot$ means element-wise multiplication, and $\tilde h$ is what the next layer receives. A new mask is drawn for every example and every step.

At test time the layer does nothing: $\tilde h = h$. Because survivors were scaled up during training, the next layer sees the same *average* input in both modes. This version is called **inverted dropout**. The original paper did it the other way round (no scaling in training, multiply weights by $1-p$ at test time); inverted dropout puts all the cost in training so serving code needs no change.

The backward pass uses the same mask: gradients flow only through kept units and are multiplied by the same $1/(1-p)$. Dropped units get no gradient on that step.

## Step 2: why the scaling keeps the average right

Each unit is either $h_i/(1-p)$ (with probability $1-p$) or $0$ (with probability $p$). Its expected value is

$$\mathbb{E}[\tilde h_i] = (1-p)\cdot\frac{h_i}{1-p} + p \cdot 0 = h_i.$$

So dropout adds noise but no bias. The noise is large, though: the variance of one unit is $h_i^2\, p/(1-p)$, which equals $h_i^2$ when $p = 0.5$.

<div class="ex">
<div class="lab">Worked example</div>

**inverted dropout on four numbers**

Take $h = [2, 4, 6, 8]$ and $p = 0.5$, so survivors are multiplied by $1/(1-0.5) = 2$. The plain sum is $2 + 4 + 6 + 8 = 20$.

Mask $[1, 0, 0, 1]$ keeps units 1 and 4: $\tilde h = [4, 0, 0, 16]$, sum $= 20$.

Mask $[0, 0, 1, 0]$ keeps only unit 3: $\tilde h = [0, 0, 12, 0]$, sum $= 12$.

There are $2^4 = 16$ equally likely masks. Their sums range from 0 (all dropped) to 40 (all kept, each doubled). Averaging all 16 sums gives exactly 20.

Per unit: unit 1 is 4 half the time and 0 otherwise, mean $0.5 \times 4 = 2$. Each unit's variance is $h_i^2$, so the sum has variance $4 + 16 + 36 + 64 = 120$, standard deviation $\sqrt{120} \approx 10.95$. Same mean, lots of noise.

</div>

~~~dg-fig 02.03-masks.svg
Left: the summed output for all 16 masks of the example. The spread is wide, but the average is exactly the no-dropout value of 20. Right: averaging each unit over more and more random masks converges to its original value.
~~~

## Step 3: why noise helps

Injecting noise sounds like it should make learning worse. Two complementary explanations say why it helps.

- **An ensemble for free.** A layer with $n$ units has $2^n$ possible masks, so each step trains a different "thinned" subnetwork, all sharing the same weights. The full network at test time approximates averaging all their predictions. Averaging many models (an **ensemble**) reliably reduces variance, and dropout gets an approximation of it for the price of one model.
- **No co-adaptation.** Without dropout, unit A can learn to patch unit B's specific mistakes on specific training examples. With dropout any partner may be missing, so each unit must learn features useful on their own, which transfer better to new data.

How strong should it be? The next figure sweeps $p$ on the moons task.

~~~dg-fig 02.03-sweep.svg
Dropout rate swept on the moons task (mean of two seeds). Accuracy stays between 81% and 86%, while held-out loss falls from about 3.6 to 0.64. For this oversized network on 80 points, a very high rate (0.9) was best on both measures.
~~~

The best rate here, $p = 0.9$, is far above the usual defaults. That is not a contradiction: two layers of 256 units for 80 points is absurdly oversized, so very strong noise suits it. The general rule is that the right $p$ grows with how oversized the model is relative to the data, and has to be tuned on held-out data.

## Step 4: where dropout goes, and how much

Dropout goes on hidden activations, usually after the non-linearity, never on the output logits. Typical settings:

| Where | Typical $p$ | Notes
| --- | --- | --- |
| Large fully connected layers (AlexNet, VGG heads) | 0.5 | where the original results came from
| Transformers | 0.1 | on attention weights and on each residual branch before the add
| Input features | 0.1 to 0.2 | gentler, because inputs carry real information
| CNNs with BatchNorm | often 0 | BN and augmentation already regularise
| Large pre-training runs with huge data | often 0 | the data itself prevents memorisation

On convolutional feature maps, dropping single pixels barely regularises because neighbours are so correlated; **channel dropout** (`nn.Dropout2d`) drops whole channels instead.

## Step 5: weight decay, and why Adam needs AdamW

**Weight decay** shrinks every weight toward zero a little at each step, which favours small weights and smoother functions. There are two ways to write it.

**L2 regularisation** adds a penalty to the loss: $L_{\text{total}} = L + \frac{\lambda}{2}\|w\|^2$. Its gradient is $g + \lambda w$, where $g$ is the data gradient and $\lambda$ (lambda) is the strength.

**Decoupled weight decay** shrinks the weights directly: $w \leftarrow w - \eta\lambda w - \eta g$, where $\eta$ (eta) is the learning rate.

For plain SGD these are the same thing: both give $w - \eta(g + \lambda w)$.

<div class="ex">
<div class="lab">Worked example</div>

**SGD sees no difference**

Let $w = 2$, data gradient $g = 0.5$, $\eta = 0.1$, $\lambda = 0.01$.

L2: the total gradient is $0.5 + 0.01 \times 2 = 0.52$, so $w \leftarrow 2 - 0.1 \times 0.52 = 1.948$.

Decoupled: $w \leftarrow 2 - 0.1 \times 0.01 \times 2 - 0.1 \times 0.5 = 2 - 0.002 - 0.05 = 1.948$. Identical.

</div>

Adam breaks this. Adam divides each weight's gradient by $\sqrt{\hat v}$, a running root-mean-square of that weight's past gradients ($\hat m$ is the running mean of gradients, $\epsilon$ a tiny constant to avoid division by zero):

$$\text{Adam + L2:}\quad w \leftarrow w - \eta \frac{\hat m(g + \lambda w)}{\sqrt{\hat v} + \epsilon}$$
$$\text{AdamW:}\quad w \leftarrow w - \eta\left(\frac{\hat m(g)}{\sqrt{\hat v} + \epsilon} + \lambda w\right)$$

With L2 inside Adam, the decay term is divided by $\sqrt{\hat v}$ too. Weights with large past gradients are barely regularised and weights with rare gradients are regularised hard.

<div class="ex">
<div class="lab">Worked example</div>

**the same penalty, two very different pulls**

Two weights, both $w = 0.5$, with $\lambda = 0.01$ and $\eta = 0.001$, so the penalty gradient is $\lambda w = 0.005$ for both.

Weight A gets a gradient every step; its $\sqrt{\hat v} = 1$. Weight B is, say, a rare token's embedding; its $\sqrt{\hat v} = 0.02$.

Adam + L2, decay part only: A moves by $0.001 \times 0.005 / 1 = 5 \times 10^{-6}$. B moves by $0.001 \times 0.005 / 0.02 = 2.5 \times 10^{-4}$, 50 times more.

AdamW: both move by $\eta\lambda w = 0.001 \times 0.01 \times 0.5 = 5 \times 10^{-6}$, the same relative shrink.

</div>

~~~dg-fig 02.03-adamw.svg
Simulation: Adam, learning rate 0.001, lambda 0.1, 3,000 steps, two weights whose data gradients are pure noise (one every step, one on 1% of steps). With L2 inside Adam the rare weight collapses from 1 to 0.04 while the frequent one ends at 0.81. With AdamW they end at 0.81 and 0.75: the same steady shrink for both.
~~~

Loshchilov and Hutter's AdamW applies the decay outside the adaptive scaling; in their experiments it generalised better and made the best $\lambda$ less tied to the learning rate. Common practice: AdamW with $\lambda$ between 0.01 and 0.1, and no decay on biases, normalisation gains ($\gamma$, $\beta$) and often embeddings.

## Step 6: data augmentation

**Data augmentation** creates new training examples by applying transforms that should not change the label. It tells the model what does not matter: a cat flipped left-right is still a cat. For vision it is usually the strongest regulariser, because it adds information rather than only constraints.

- **Geometric:** random resized crops (the single most effective ImageNet augmentation), horizontal flips, small rotations and translations.
- **Photometric and erasing:** colour jitter, brightness, blur; Cutout blanks a patch so the model cannot rely on one region.
- **Label mixing:** **mixup** blends two images and their labels, $\tilde x = \lambda x_i + (1-\lambda)x_j$ and $\tilde y = \lambda y_i + (1-\lambda) y_j$ on one-hot labels, with $\lambda$ drawn from a Beta($\alpha$, $\alpha$) distribution (here $\lambda$ is a mixing weight, not the decay strength). **CutMix** pastes a rectangle from one image into another and mixes the labels by area.

To see the effect, train a small network on just 300 of scikit-learn's 8×8 digit images, then test on the rest, both as given and shifted by one pixel.

~~~dg-fig 02.03-augment.svg
A network trained on 300 original digits loses accuracy when test digits move by one pixel. Training on randomly shifted copies of the same 300 images keeps accuracy high on both versions of the test set.
~~~

Averaged over two seeds, the plain model scores 95% on the test digits as given but only 15% when every test digit is moved one pixel right and one pixel down: it learned *where* the ink is, not the shape. The model trained on random shifts of up to one pixel scores 94% and 92%. Same 300 images, same network; the only difference is that augmentation told the model which changes do not matter.

Augmentation must respect the task: flips are fine for ImageNet and wrong for digits or text, and for detection the boxes must move with the image. Augment only the training data.

~~~dg-flow Where each regulariser acts in one training step.
Raw batch -(augment: crop, flip, mixup)-> Model forward -(dropout masks)-> Loss -(backward)-> Gradients -(AdamW: decay applied outside Adam)-> *Updated weights
~~~

## Dropout, weight decay and augmentation in code

First, inverted dropout from scratch in NumPy, checked against the hand example:

```python
import numpy as np

def dropout(h, p, training, rng):
    if not training or p == 0:
        return h, None
    mask = (rng.random(h.shape) >= p) / (1 - p)   # keep with prob 1-p, pre-scaled
    return h * mask, mask                         # backward: dh = dout * mask

rng = np.random.default_rng(0)
h = np.array([2., 4., 6., 8.])
print(dropout(h, 0.5, True, rng)[0])              # [4. 0. 0. 0.]  one random draw
outs = np.stack([dropout(h, 0.5, True, rng)[0] for _ in range(100_000)])
print(outs.mean(0).round(2))                      # [2.   3.99 5.97 8.  ]  means match h
print(outs.sum(1).std().round(2))                 # 10.93  (theory: sqrt(120) = 10.95)
```

In PyTorch, the layer behaves differently in `train()` and `eval()` mode, and AdamW takes parameter groups so that biases and norm parameters skip decay:

```python
import torch, torch.nn as nn
torch.manual_seed(0)
drop = nn.Dropout(p=0.5)
x = torch.tensor([2., 4., 6., 8.])
drop.train(); print(drop(x))      # tensor([ 0.,  0., 12.,  0.])
drop.eval();  print(drop(x))      # tensor([2., 4., 6., 8.])  identity at test time

model = nn.Sequential(nn.Linear(64, 256), nn.LayerNorm(256), nn.ReLU(),
                      nn.Dropout(0.1), nn.Linear(256, 10))
decay    = [p for n, p in model.named_parameters() if p.ndim >= 2]   # weight matrices
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]    # biases, LN gain/shift
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                         {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)
print(len(decay), len(no_decay))  # 2 4
```

## Reading the curves

The most useful diagnostic is the pair of training and held-out loss curves plotted on the same axes.

~~~dg-fig 02.03-curves.svg
Dashed lines are training loss, solid lines held-out loss (log scale). Without regularisation the training loss heads to zero while the held-out loss reaches its lowest point within the first hundred steps and then climbs. Weight decay keeps the held-out loss far lower.
~~~

How to read them:

- **Training loss falling, held-out loss rising:** overfitting. Add or strengthen regularisation, get more data or augmentation, or stop earlier.
- **Both high and close together:** underfitting. Regularisation is too strong (or the model too small); reduce $p$ or $\lambda$.
- **Training loss higher with dropout:** expected; compare models on held-out numbers, evaluated in `eval()` mode.
- **Held-out minimum very early:** in the figure, all three runs reach their best held-out loss within the first 50 to 100 steps. Keeping the checkpoint with the best validation score, **early stopping**, is the cheapest regulariser of all and works alongside the others.
- Watch held-out *loss* as well as accuracy: a model can keep its accuracy while becoming badly overconfident.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Forgot `model.eval()` at test time | predictions change from run to run; accuracy lower than expected | call `eval()` before validation, `train()` after
| Dropout placed before BatchNorm | train/eval gap; worse validation than without dropout | BN's running variance was measured with dropout noise that vanishes at eval; put dropout after BN or drop it
| L2 penalty used with Adam | rare features or embeddings shrink, best $\lambda$ changes with learning rate | use AdamW (decoupled decay)
| Decay on biases and norm gains | slightly worse results, collapsed scales | parameter groups with `weight_decay=0` for them
| $p$ too high | training loss stays high, underfitting | lower $p$; 0.1 is a safer start in modern architectures
| Augmentation that changes the label | accuracy ceiling on certain classes (6 vs 9, b vs d) | only use transforms that preserve the label for your task
| Pixel dropout on conv maps | no regularisation effect | use `Dropout2d` (channel dropout) or rely on augmentation

## Going deeper

**Dropout as approximate averaging.** For a purely linear layer, the test-time network computes exactly the average over all masks, because expectation passes through linear maps. With non-linearities in between it is only an approximation, since the average of $f(\tilde h)$ is not $f$ of the average.

**Dropout as adaptive L2.** For linear and generalised linear models, the expected loss under dropout noise can be expanded to second order: it equals the clean loss plus a penalty roughly of the form $\frac{p}{1-p}\sum_j w_j^2 \sum_i x_{ij}^2$ (for squared loss), an L2 penalty scaled per feature by how large that feature's inputs are. So dropout is a data-dependent cousin of weight decay.

**Monte Carlo dropout.** Keeping dropout on at test time and averaging several stochastic forward passes gives a prediction plus a spread, a cheap and rough uncertainty estimate.

**Weight decay with normalisation.** In a layer followed by BatchNorm or LayerNorm, scaling the weights by any constant does not change the output, so weight decay cannot be acting as a capacity limit there. Instead it shrinks the weight norm, which raises the *effective* learning rate of that layer ($\eta / \|w\|^2$ for scale-invariant weights). This is why decay still matters in normalised networks, and why $\lambda$ and the learning rate interact.

**Implicit regularisation.** Small batches add gradient noise and early stopping limits how far weights travel from initialisation; both regularise without any explicit term. Label smoothing (a target of, say, 0.9 for the true class) directly targets the overconfidence seen in the opening table.

## Key terms

| Term | Meaning
| --- | --- |
| Regularisation | Any change to training that favours solutions that generalise over ones that memorise
| Dropout rate $p$ | Probability that a unit is zeroed on a given training step
| Inverted dropout | Scaling kept units by $1/(1-p)$ during training so the test-time layer is the identity
| Co-adaptation | Units relying on specific partners being present; dropout breaks it
| Weight decay | Shrinking each weight toward zero by a fixed fraction every step
| AdamW | Adam with weight decay applied outside the adaptive scaling
| Data augmentation | Training on label-preserving transforms of the inputs
| Early stopping | Keeping the checkpoint with the best validation score

<div class="take">
<div class="lab">Key takeaway</div>

regularisation stops a large network from memorising. Use inverted dropout (and remember `eval()`), use AdamW rather than an L2 penalty inside Adam and skip decay on biases and norm parameters, treat label-preserving augmentation as the main way to inject what you know about invariance, and judge every choice by held-out loss and accuracy, not training loss.

</div>
