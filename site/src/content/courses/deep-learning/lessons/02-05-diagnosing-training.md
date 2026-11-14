---
id: "02.05"
course: "deep-learning"
module: "02"
slug: "diagnosing-training"
title: "Diagnosing training"
keyIdeas: ["Check the loss at step 0 before anything else: a $C$-class classifier with sensible init should start near $\\ln C$ (2.30 for 10 classes); anything far off means a bug in data, labels, init or the loss.", "Overfitting a single small batch to near-zero loss is the fastest end-to-end test; if the model cannot memorise 16 examples, the bug is in the pipeline, not the hyperparameters.", "Read loss curves by shape: a flat line, early divergence, oscillation, a train/validation gap and a plateau each point to a different cause.", "Log per-layer gradient norms, update-to-weight ratios (around $10^{-3}$ is healthy for SGD) and the fraction of dead ReLUs; vanishing and exploding gradients show up there long before they show up in accuracy."]
question: "Your new model starts at loss 0.01 on a 100-class classification problem and quickly goes to 0.0001, but validation accuracy is 1%. Walk through what the initial loss tells you, the most likely class of bug, and two concrete checks you would run."
answer: "With small initial logits a 100-class model should start near $\\ln 100 \\approx 4.6$. A starting loss of 0.01 means the model is almost certain of the correct label before learning anything, so the label is effectively available in the input or the loss is computed wrongly. Likely bugs: label leakage (a feature derived from the target, e.g. the class index encoded in the filename or a one-hot target concatenated into the input), comparing predictions against themselves, or a loss with the wrong reduction or axis (e.g. cross-entropy over the batch dimension so each \"class\" is an example). Validation at 1% (chance) confirms the model learned nothing transferable. Checks: decode one batch exactly as fed to the model and confirm the targets are not present in the inputs; compute the loss by hand for one example from the logits and label and compare with the framework value; shuffle the labels and confirm the initial loss becomes about 4.6 and cannot fall quickly."
sources: [{"title": "Karpathy: A Recipe for Training Neural Networks", "url": "https://karpathy.github.io/2019/04/25/recipe/", "note": "The step-by-step routine: inspect data, verify init loss, overfit a batch, then regularise and tune."}, {"title": "CS231n: Neural Networks Part 3", "url": "https://cs231n.github.io/neural-networks-3/", "note": "Gradient checks, loss-curve shapes, update/weight ratios and activation histograms."}, {"title": "Google Research: Deep Learning Tuning Playbook", "url": "https://github.com/google-research/tuning_playbook", "note": "A systematic process for choosing what to tune and reading training curves."}]
practice: ["learning-rate-sweep", "finite-difference-gradient-checking", "early-stopping", "lr-range-test-and-one-cycle-policy", "gradient-clipping"]
tags: ["debugging", "vanishing gradients", "loss curves"]
tools: ["PyTorch"]
---

## Training fails silently

A buggy network usually still trains: the loss goes down a bit, accuracy is mediocre, and nothing crashes. Karpathy's summary is that neural nets "fail silently". The defence is a fixed routine of checks that rule out whole classes of bugs before you spend compute on tuning. This lesson collects them and ties each back to the earlier lessons in this module.

## Checks before training

1. **Look at the data after the pipeline.** Decode a batch exactly as the model receives it: images after augmentation and normalisation, tokens after padding. Mislabelled or wrongly normalised inputs are the most common bug.
2. **Check the initial loss.** With small output logits, a $C$-class classifier predicts roughly uniform probabilities, so the loss should be about $\ln C$: 2.303 for 10 classes, 6.908 for 1000. A starting loss of 40 means exploding init or missing softmax scaling; a loss of 0 means label leakage.
3. **Overfit one batch.** Take 8-32 examples, turn off augmentation and regularisation, and train until loss is near zero. Failure means a wiring bug: wrong axis in the loss, detached tensors, frozen parameters, labels shuffled independently of inputs.
4. **Gradient check custom code.** Any hand-written backward (01.04, 02.02) should match central finite differences to a relative error of about $10^{-6}$ in float64.
5. **Set a baseline.** A linear model or a known recipe tells you what "working" looks like.

## Reading loss curves

| Curve shape | Likely cause | First thing to try |
| --- | --- | --- |
| Flat from step 0 | lr far too low, dead network, frozen params, data bug | check gradients are non-zero; lr range test |
| Decreases, then NaN or spike | lr too high, no warmup, fp16 overflow, log(0) | lower lr, add warmup and clipping |
| Noisy, oscillating, not improving | lr too high for batch size | lower lr or increase batch |
| Train falls, validation rises | overfitting | augmentation, dropout, weight decay, early stop |
| Both plateau high | underfitting or too-small lr late in training | bigger model, longer schedule, check capacity |
| Sudden drops at fixed epochs | step lr schedule | expected, not a bug |

Plot the loss on a log scale and smooth it; raw per-step loss is noisy enough to hide trends. Always plot validation alongside training, and measure validation in eval mode (02.01).

~~~dg-flow A debugging order that rules out cheap bugs before expensive tuning.
Inspect batch -(inputs ok)-> Initial loss ≈ ln C -(yes)-> Overfit 16 examples -(reaches ≈ 0)-> Full run + monitors -(gap or plateau)-> *Tune regularisation, lr, capacity
~~~

## Monitoring inside the network

Loss curves tell you something is wrong; internal statistics tell you where.

- **Per-layer gradient norms.** In a healthy network they are within an order of magnitude or two of each other. Early-layer norms 1000× smaller than late-layer norms is vanishing gradients (saturating activations, 01.02; bad init, 01.05; missing residuals, 03.04). Norms that grow over training toward the input are exploding gradients; clipping (02.04) is the bandage, normalisation and init are the fix.
- **Update-to-weight ratio.** $\|\eta\,\Delta w\| / \|w\|$ per layer. Around $10^{-3}$ is a useful rule of thumb for SGD (CS231n); $10^{-1}$ means the layer is being rewritten each step, $10^{-6}$ means it is effectively frozen.
- **Activation statistics.** Mean and standard deviation per layer at init and during training. Collapsing standard deviations mean signal is dying; tanh/sigmoid outputs piled at the extremes mean saturation.
- **Dead units.** Fraction of ReLU units that are zero for every example in a large batch. A few percent is normal; 40% after a learning-rate spike means you lost capacity permanently.

```python
import torch

def grad_report(model, x, y, loss_fn, lr, relu_layers):
    acts = {}
    hooks = [m.register_forward_hook(lambda m, i, o, n=n: acts.__setitem__(n, o.detach()))
             for n, m in relu_layers.items()]
    model.zero_grad(); loss_fn(model(x), y).backward()
    for h in hooks: h.remove()
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"{name:30s} grad {p.grad.norm():.2e}  update/weight {lr * p.grad.norm() / p.norm():.1e}")
    for n, a in acts.items():                       # unit dead if zero on every example
        dead = (a.flatten(1) <= 0).all(0).float().mean()
        print(f"{n}: {dead:.1%} dead units")
```

## NaNs and infinities

A NaN anywhere poisons everything downstream within one step. Usual sources: learning rate too high, `log` or division of zero (use the logits-based losses from 01.03, add $\epsilon$ to normalisation variances), fp16 overflow above 65504 (use bf16 or loss scaling), and bad data such as an empty mask making a mean divide by zero. `torch.autograd.set_detect_anomaly(True)` finds the first operation that produced a NaN, at a large speed cost, so use it only to localise the bug.

<div class="ex">
<div class="lab">Worked example</div>

**reading the symptoms**

A 10-class CNN starts at loss 2.31 (good), drops to 1.2 by step 2000, then stops. Per-layer gradient norms: last layer $3 \times 10^{-2}$, first conv layer $4 \times 10^{-7}$. Activation standard deviation falls from 1.0 at the input to 0.003 by layer 15.

Diagnosis: vanishing signal through depth, not a learning-rate problem. Raising the learning rate would make the last layers unstable while the first layers stay frozen.

Fixes in order of effort: He init for the ReLU layers, BatchNorm after each conv, then residual connections. After adding BN the first-layer gradient norm rises to about $10^{-3}$ and the loss keeps improving.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

verify the initial loss and that the model can overfit one batch before tuning anything, read loss curves by shape, and monitor per-layer gradient norms, update ratios and dead units to locate vanishing or exploding signal.

</div>
