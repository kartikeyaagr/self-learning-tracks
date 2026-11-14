---
id: "02.03"
course: "deep-learning"
module: "02"
slug: "dropout-and-regularisation"
title: "Dropout and regularisation"
keyIdeas: ["Dropout zeroes each activation with probability $p$ during training; inverted dropout divides survivors by $1-p$ so the expected activation is unchanged and inference needs no rescaling.", "Dropout approximately trains an ensemble of thinned subnetworks that share weights, and stops units relying on specific partners (co-adaptation).", "With SGD, L2 regularisation and weight decay are the same thing; with Adam they are not, because the penalty gradient gets divided by the adaptive denominator. AdamW decouples them.", "Data augmentation is the strongest regulariser for vision: label-preserving transforms (crops, flips, colour jitter) and label-mixing ones (mixup, CutMix) encode invariances the data alone doesn't show."]
question: "You fine-tune a model with Adam plus an L2 penalty and find that the embedding rows for rare tokens shrink much faster than the rows for frequent ones, hurting rare-token performance. Explain the mechanism, and how AdamW changes it."
answer: "With L2 inside Adam, the penalty gradient $\\lambda w$ is added to the data gradient and the sum is divided by $\\sqrt{\\hat v}$, the running RMS of that weight's gradients. Frequent tokens' embedding rows receive large, frequent data gradients, so $\\hat v$ is large and the decay term is heavily damped. Rare tokens' rows receive data gradients only occasionally, so $\\hat v$ is small and the same $\\lambda w$ becomes a large normalised step that pulls the row toward zero every step, even when the token isn't in the batch. The effective regularisation strength is therefore inversely tied to gradient history. AdamW applies $w \\leftarrow w - \\eta\\lambda w$ outside the adaptive scaling, so every row decays by the same fraction per step regardless of its gradient statistics, and Adam's normalisation is applied only to the data gradient. Many setups also exclude embeddings from decay entirely."
sources: [{"title": "Srivastava et al.: Dropout (JMLR 2014)", "url": "https://jmlr.org/papers/v15/srivastava14a.html", "note": "The ensemble interpretation, test-time scaling, and the co-adaptation experiments."}, {"title": "Loshchilov & Hutter: Decoupled Weight Decay Regularization", "url": "https://arxiv.org/abs/1711.05101", "note": "Why L2 and weight decay differ for adaptive optimisers; the AdamW update."}, {"title": "Zhang et al.: mixup", "url": "https://arxiv.org/abs/1710.09412", "note": "The mixup recipe and its effect on generalisation and calibration."}, {"title": "Dive into Deep Learning: Dropout", "url": "https://d2l.ai/chapter_multilayer-perceptrons/dropout.html", "note": "Inverted dropout from scratch and where to apply it."}]
practice: ["dropout", "dropout-2", "adamw-decoupled-weight-decay", "mixup-augmentation", "cutmix-augmentation"]
tags: ["dropout", "weight decay", "augmentation"]
tools: ["NumPy", "PyTorch"]
---

## Regularisation in deep networks

A modern network has enough parameters to memorise its training set, even with random labels. Regularisation is anything that steers it toward solutions that generalise. The ML course covers the classical view (bias-variance, L1/L2 penalties, early stopping); this lesson covers the three tools that matter most for deep nets: dropout, weight decay done right, and data augmentation.

## Dropout

During training, each activation is kept with probability $1-p$ and zeroed otherwise, with a fresh random mask per example and per step. **Inverted dropout** rescales the survivors:

$$\tilde h = \frac{m \odot h}{1 - p}, \qquad m_i \sim \mathrm{Bernoulli}(1-p)$$

so $\mathbb{E}[\tilde h] = h$ and at inference the layer is the identity. (The original paper scaled weights by $1-p$ at test time instead; inverted dropout puts the cost in training so serving code doesn't change.) The backward pass uses the same mask: gradients flow only through kept units, multiplied by $1/(1-p)$.

```python
import numpy as np

def dropout(h, p, training, rng):
    if not training or p == 0:
        return h, None
    mask = (rng.random(h.shape) >= p) / (1 - p)   # keep with prob 1-p, pre-scaled
    return h * mask, mask                         # backward: dh = dout * mask
```

Why it works, in two views:

- **Ensemble.** Each mask selects one of $2^n$ thinned networks that share weights; training samples them, and the scaled full network at test time approximates averaging their predictions.
- **No co-adaptation.** A unit cannot rely on any specific other unit being present, so features must be useful in many contexts.

Practical defaults: $p = 0.5$ on large fully connected layers (AlexNet, VGG), $0.1$ in Transformers (attention weights and residual branches), often none in CNNs with BatchNorm. For feature maps, **channel dropout** (Dropout2d) drops entire channels, since neighbouring pixels are so correlated that dropping single ones barely regularises. Dropout lengthens training (noisier gradients) and interacts badly with BatchNorm if placed before it, because BN's running variance is estimated under dropout noise that disappears at eval.

## Weight decay and L2 are not the same in Adam

L2 regularisation adds $\frac{\lambda}{2}\|w\|^2$ to the loss, adding $\lambda w$ to the gradient. Weight decay shrinks the weights directly each step: $w \leftarrow w - \eta\lambda w - \eta g$. For plain SGD these are identical. For Adam they are not:

$$\text{Adam + L2:}\quad w \leftarrow w - \eta \frac{\hat m(g + \lambda w)}{\sqrt{\hat v} + \epsilon}$$
$$\text{AdamW:}\quad w \leftarrow w - \eta\left(\frac{\hat m(g)}{\sqrt{\hat v} + \epsilon} + \lambda w\right)$$

With L2 inside Adam, the decay term is divided by $\sqrt{\hat v}$, so weights with large historical gradients are barely regularised and weights with tiny gradients are regularised heavily, the opposite of what you want. Loshchilov and Hutter's AdamW applies the decay outside the adaptive scaling, so every weight shrinks at the same relative rate; it generalises better and makes the best $\lambda$ largely independent of the learning rate. Typical: AdamW with $\lambda = 0.01$ to $0.1$, and no decay on biases and normalisation parameters. ML Maths covers Adam itself.

~~~dg-flow Where each regulariser acts in one training step.
Raw batch -(augment: crop, flip, mixup)-> Model forward -(dropout masks)-> Loss -(backward)-> Gradients -(AdamW: decay applied outside Adam)-> *Updated weights
~~~

## Data augmentation

Augmentation creates new training examples by applying transforms that should not change the label. It is how you tell a vision model about invariances: a cat flipped left-right is still a cat.

- **Geometric**: random resized crops (the single most effective ImageNet augmentation), horizontal flips, small rotations and translations.
- **Photometric**: colour jitter, brightness, contrast, blur.
- **Erasing**: Cutout and random erasing blank a patch so the model can't rely on one region.
- **Label mixing**: mixup blends two images and their labels, $\tilde x = \lambda x_i + (1-\lambda)x_j$ with the same $\lambda$ on the one-hot labels; CutMix pastes a patch from one image into another and mixes labels by area.

Augmentations must respect the task. Horizontal flips are fine for ImageNet and wrong for digit or text recognition; random crops can cut the object out entirely when it is small, and for detection the boxes must be transformed with the image. Augmentation is applied only to training data.

| Tool | Acts on | Train/eval difference | Typical setting |
| --- | --- | --- | --- |
| Dropout | activations | masks only in training | 0.1 to 0.5 |
| Weight decay | weights | none | AdamW, 0.01 to 0.1 |
| Augmentation | inputs | training data only | crops + flips baseline |
| Early stopping | training length | none | stop at best validation |

<div class="ex">
<div class="lab">Worked example</div>

**what inverted dropout preserves**

A hidden vector $h = [2, 4, 6, 8]$ with $p = 0.5$. One sampled mask keeps units 1 and 4: $\tilde h = [4, 0, 0, 16]$, sum 20, the same as the original sum of 20.

Another mask keeps only unit 3: $\tilde h = [0, 0, 12, 0]$, sum 12. Individual draws vary, but averaged over masks each unit's expected value is exactly its original value, so the next layer sees the same mean at train and test time.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

use inverted dropout where the model is over-parameterised and not already normalised, use AdamW rather than L2-in-Adam for weight decay, and treat augmentation as the main way to inject what you know about invariance.

</div>
