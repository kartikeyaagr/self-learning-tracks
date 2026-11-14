---
id: "02.01"
course: "deep-learning"
module: "02"
slug: "batch-normalisation"
title: "Batch normalisation"
keyIdeas: ["BatchNorm standardises each channel using the mean and variance of the current mini-batch, then applies a learned scale $\\gamma$ and shift $\\beta$ so the layer can undo the normalisation if that helps.", "In training it uses batch statistics and updates running averages; in eval it uses the running averages, so `model.train()` versus `model.eval()` changes the output.", "It lets you use much larger learning rates and makes training far less sensitive to initialisation; the modern explanation is a smoother loss landscape, not the original \"internal covariate shift\" story.", "It breaks with small batches (noisy statistics), with train/eval distribution mismatch, and whenever examples in a batch should not influence each other."]
question: "A model scores 92% validation accuracy when evaluated with batch size 256 but 61% when the same weights serve requests one at a time. The serving code never calls `model.eval()`. Explain what happens inside each BatchNorm layer in both settings, and why calling `eval()` fixes serving."
answer: "Without `eval()` every BatchNorm layer stays in training mode and normalises with the statistics of the current batch. With 256 validation examples those statistics are close to the population statistics the network was trained against, so outputs look normal. With one example, each channel's batch mean over $N \\cdot H \\cdot W$ comes from a single image, and in an MLP layer with $N=1$ the variance is zero, so $\\hat x$ is roughly zero (or depends only on spatial variation within one image). The features the classifier sees bear little relation to training, so accuracy collapses. Training mode also keeps updating running averages with serving data. `model.eval()` makes each BN layer use the running mean and variance accumulated during training: a fixed per-channel affine transform independent of batch composition, so a single request produces the same output it would inside any batch."
sources: [{"title": "Ioffe & Szegedy: Batch Normalization", "url": "https://arxiv.org/abs/1502.03167", "note": "Algorithms 1 and 2 give the training transform and the inference procedure with population statistics."}, {"title": "Santurkar et al.: How Does Batch Normalization Help Optimization?", "url": "https://arxiv.org/abs/1805.11604", "note": "Experiments against the covariate-shift explanation and the smoothness argument."}, {"title": "PyTorch docs: BatchNorm2d", "url": "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html", "note": "The momentum convention, biased vs unbiased variance, and track_running_stats."}, {"title": "Dive into Deep Learning: Batch Normalization", "url": "https://d2l.ai/chapter_convolutional-modern/batch-norm.html", "note": "From-scratch implementation and discussion of train vs predict mode."}]
practice: ["batch-normalization", "apply-alphago-residual-block", "resnet-residual-block"]
tags: ["batch norm", "normalisation"]
tools: ["NumPy", "PyTorch"]
---

## Why this module exists

Module 01 gave you a network that can, in principle, train. This module covers what makes deep networks train **reliably**: normalisation layers (this lesson and 02.02), regularisation (02.03), the mechanics of mini-batch optimisation (02.04), and how to read the signs when training goes wrong (02.05). The optimiser update rules themselves are in ML Maths.

## What BatchNorm computes

For a convolutional feature map of shape $(N, C, H, W)$, BatchNorm treats each channel $c$ separately and computes statistics over the batch **and** spatial dimensions, $N \cdot H \cdot W$ values per channel:

$$\mu_c = \frac{1}{NHW}\sum x_{n,c,h,w}, \qquad \sigma_c^2 = \frac{1}{NHW}\sum (x_{n,c,h,w} - \mu_c)^2$$
$$y = \gamma_c \frac{x - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c$$

For an MLP activation of shape $(N, D)$ the statistics are over the batch only, one pair per feature. $\gamma$ and $\beta$ are learned per channel, so the layer can represent the identity if normalisation turns out to be unhelpful. Because $\beta$ absorbs any constant offset, a convolution followed directly by BatchNorm needs no bias: set `bias=False`.

~~~dg-flow Where BatchNorm sits in a typical CNN block, and the two different statistics it uses.
Conv (no bias) -(N×C×H×W)-> *BatchNorm -(γ, β per channel)-> ReLU -> Next conv
[Train] Batch stats μ_B, σ²_B -(normalise)-> Output -(EMA update)-> Running μ, σ²
[Eval] Running μ, σ² -(normalise)-> *Deterministic output
~~~

## Train mode versus eval mode

At inference you may see a single example, so batch statistics are unavailable or meaningless. BatchNorm therefore keeps **running estimates** updated during training as an exponential moving average. In PyTorch, with `momentum=0.1`:

$$\hat\mu \leftarrow 0.9\,\hat\mu + 0.1\,\mu_B$$

(PyTorch's "momentum" is the weight on the new value, the opposite of the optimiser convention.) Normalisation in training uses the biased batch variance; the running variance stores the unbiased one. In `model.eval()` the layer uses the running estimates and becomes a fixed affine map, which can be folded into the preceding convolution's weights for deployment.

The most common BatchNorm bug is forgetting the mode switch. Evaluating in train mode makes predictions depend on which other examples share the batch, and updates the running stats with test data. Training in eval mode by accident freezes the statistics. Fine-tuning with tiny batches often deliberately keeps BN in eval mode ("frozen BN") for this reason.

```python
import numpy as np

def batchnorm_train(x, gamma, beta, run_mu, run_var, momentum=0.1, eps=1e-5):
    # x: (N, C, H, W); statistics per channel over N, H, W
    mu = x.mean(axis=(0, 2, 3), keepdims=True)
    var = x.var(axis=(0, 2, 3), keepdims=True)              # biased, used to normalise
    n = x.size / x.shape[1]
    run_mu[:] = (1 - momentum) * run_mu + momentum * mu.ravel()
    run_var[:] = (1 - momentum) * run_var + momentum * var.ravel() * n / (n - 1)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma[None, :, None, None] * xhat + beta[None, :, None, None]
```

## Why it helps

The original paper argued BN reduces **internal covariate shift**, the change in each layer's input distribution as earlier layers update. Later work (Santurkar et al., 2018) showed that injecting noise to increase that shift after BN barely hurts, and argued the real benefit is that BN makes the loss and its gradients smoother (smaller Lipschitz constants), so larger steps are safe. Other effects that are well established:

- **Scale invariance.** The output of conv-then-BN doesn't change if you multiply the conv weights by a constant. The effective learning rate then depends on the weight norm, and weight decay interacts with it.
- **Higher learning rates and robust init.** ResNets with BN train at learning rate 0.1 with SGD; without it they need careful init and smaller steps.
- **Mild regularisation.** Each example is normalised with noisy batch statistics, a form of noise injection; BN networks often need less dropout.

## Where it breaks

- **Small batches.** With batch 2 per GPU the statistics are so noisy that accuracy drops sharply. Fixes: synchronised BN across devices, GroupNorm, or LayerNorm (02.02).
- **Distribution mismatch.** If test data differ from training data, running statistics are wrong. Recomputing them on target data is a cheap adaptation trick.
- **Sequences and autoregressive models.** Batch statistics mix information across examples and padding positions, and at generation time the batch is one token. This is why Transformers use LayerNorm.
- **Information leakage.** In contrastive learning, BN can let the model cheat by detecting which examples share a batch.

<div class="ex">
<div class="lab">Worked example</div>

**batch statistics for one channel**

A channel holds values $[1, 3, 5, 7]$ across a batch of 4 (spatial size 1). $\mu = 4$, biased $\sigma^2 = (9+1+1+9)/4 = 5$, so $\hat x = [-1.342, -0.447, 0.447, 1.342]$.

With $\gamma = 2$, $\beta = 0.5$: $y = [-2.18, -0.39, 1.39, 3.18]$.

Running mean starting from 0 with momentum 0.1 becomes 0.4; running variance from 1 becomes $0.9 + 0.1 \cdot 5 \cdot 4/3 = 1.567$, using the unbiased $20/3$. In eval mode, the same input would be normalised with these values, giving very different outputs until the running stats converge.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

BatchNorm normalises each channel with batch statistics during training and running averages at eval, which buys larger learning rates and robustness to init; respect the train/eval switch and avoid it when batches are tiny or examples must stay independent.

</div>
