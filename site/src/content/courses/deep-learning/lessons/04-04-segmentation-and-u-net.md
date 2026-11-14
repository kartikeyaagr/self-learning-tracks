---
id: "04.04"
course: "deep-learning"
module: "04"
slug: "segmentation-and-u-net"
title: "Segmentation and U-Net"
keyIdeas: ["Semantic segmentation predicts a class for every pixel; the output is a $C \\times H \\times W$ logit map at input resolution.", "Encoder-decoder networks downsample to gain context and upsample to recover resolution. Detail lost in downsampling cannot be recovered by upsampling alone.", "U-Net's skip connections concatenate encoder features into the decoder at each resolution, so the decoder gets both coarse 'what' and fine 'where'.", "Pixel cross-entropy is dominated by the majority class when foreground is tiny. Dice loss measures overlap directly, so it is insensitive to how many background pixels there are; most recipes combine the two.", "Evaluate with per-class IoU averaged over classes (mIoU), not pixel accuracy."]
question: "You train a U-Net to segment small lesions that cover about 0.3% of pixels, using pixel-wise cross-entropy. Validation pixel accuracy is 99.7%, but almost every predicted mask is empty. Explain why, and what you would change in the loss and the evaluation."
answer: "Predicting background everywhere is already 99.7% accurate, and with plain cross-entropy the gradient is averaged over all pixels, so the few lesion pixels contribute almost nothing; the model settles at the trivial solution. Change the loss so overlap with the rare class drives training: add soft Dice loss (whose value ignores true-negative background pixels), or use class-weighted or focal cross-entropy, typically CE plus Dice. Sampling crops that contain lesions also raises the foreground fraction. Change evaluation from pixel accuracy to per-class IoU or Dice on the lesion class (and mIoU), accumulated over the validation set; an all-background predictor scores 0 there, which exposes the failure immediately."
sources: [{"title": "U-Net (Ronneberger et al., 2015)", "url": "https://arxiv.org/abs/1505.04597", "note": "The original architecture; figure 1 is the canonical diagram and section 3 covers the overlap-tile and weighting tricks."}, {"title": "Fully Convolutional Networks for Semantic Segmentation", "url": "https://arxiv.org/abs/1411.4038", "note": "How classifiers become dense predictors, and the skip-fusion idea U-Net builds on."}, {"title": "V-Net (Milletari et al., 2016)", "url": "https://arxiv.org/abs/1606.04797", "note": "Introduced the soft Dice loss for volumetric medical segmentation; section 3 derives the gradient."}, {"title": "Dive into Deep Learning: Fully Convolutional Networks", "url": "https://d2l.ai/chapter_computer-vision/fcn.html", "note": "Hands-on FCN with transposed-convolution upsampling."}]
practice: ["tiny-u-net-forward-pass", "u-net-skip-connection", "dice-loss", "mean-iou-for-segmentation", "transposed-convolution"]
tags: ["segmentation", "U-Net", "Dice"]
tools: ["NumPy"]
---

## Three segmentation tasks

- **Semantic segmentation**: one class per pixel. Two touching cars are one "car" region.
- **Instance segmentation**: separate masks for each object (Mask R-CNN adds a mask head to the detector from the previous lessons).
- **Panoptic segmentation**: both at once: instances for countable things, semantic regions for stuff like road and sky.

This lesson covers the semantic case and the U-Net architecture, which remains the default for medical imaging and is also the backbone of the diffusion models in module 07.

## From classifier to dense predictor

A classification CNN ends with global pooling and a fully connected layer, throwing away spatial layout. The fully convolutional network (FCN) idea is to keep everything convolutional so the output is a spatial map, then upsample it to input size. A ResNet with total stride 32 turns a 512×512 image into a 16×16 map; upsampling that 32× gives blobby masks, because fine boundaries were lost when the features were downsampled. Upsampling alone cannot recover information that is no longer there.

Two ways to upsample, both from module 03:

- **Transposed convolution**: learned, but prone to checkerboard artefacts when kernel size is not divisible by stride.
- **Bilinear or nearest resize followed by a 3×3 convolution**: fixed interpolation, then learned refinement; fewer artefacts and the common modern choice.

## U-Net

U-Net (2015) has a contracting path and a symmetric expanding path. At each resolution the encoder applies two 3×3 conv layers and then downsamples 2×, doubling channels. The decoder upsamples 2×, **concatenates** the encoder feature map from the same resolution, and applies two more 3×3 convolutions. The final 1×1 convolution maps to $C$ class logits per pixel.

~~~dg-layers A four-level U-Net on a 256×256 input. Each encoder level's output is copied across to the decoder level at the same resolution.
Level 1 | 256×256, 64 channels, skip to decoder level 1
Level 2 | 128×128, 128 channels, skip to decoder level 2
Level 3 | 64×64, 256 channels, skip to decoder level 3
*Bottleneck | 32×32, 512 channels: widest context, coarsest position
Decoder | upsample, concatenate skip, two 3×3 convs, repeat to 256×256, then 1×1 conv to C logits
~~~

Why concatenation and not addition? Concatenation lets the decoder learn how to combine the two sources; addition (as in FPN or ResNets) forces them into the same feature space. The cost is more channels in the first decoder convolution: at level 3 it sees 512 upsampled plus 256 skip = 768 input channels.

~~~dg-flow One decoder step.
Decoder 32×32×512 -(upsample ×2)-> 64×64×512 -(concat skip)-> 64×64×768 -(two 3×3 convs)-> *64×64×256
~~~

A practical detail: the original paper used unpadded convolutions, so a 572×572 input produced a 388×388 output and skips had to be centre-cropped to align. Modern implementations use `padding=1`, but if the input side is not divisible by $2^{\text{levels}}$ (say 250 with four levels), the upsampled map and the skip differ by a pixel. Pad the input to a multiple of 16, or pad or crop the upsampled tensor before concatenating.

## Losses for imbalanced masks

The natural loss is cross-entropy at every pixel, averaged. It fails when foreground is rare: if a lesion covers 0.3% of pixels, predicting "background" everywhere gets 99.7% pixel accuracy and a low loss. Class-weighted cross-entropy helps; Dice loss attacks the problem directly.

The Dice coefficient between predicted set $P$ and ground truth $G$ is

$$D = \frac{2\lvert P \cap G\rvert}{\lvert P\rvert + \lvert G\rvert}$$

The soft version replaces sets with probabilities: $D = (2\sum p_i g_i + \epsilon)/(\sum p_i + \sum g_i + \epsilon)$, and the loss is $1 - D$, averaged over classes. Background pixels where both $p$ and $g$ are near zero contribute nothing to either numerator or denominator, so the loss is scale-free with respect to the amount of background. The $\epsilon$ (around 1) keeps the loss defined for images where a class is absent. Dice gradients can be noisy when objects are tiny, so a common recipe is cross-entropy plus Dice with equal weight.

```python
import numpy as np
def soft_dice_loss(probs, onehot, eps=1.0):        # both (C, H, W)
    inter = (probs * onehot).sum(axis=(1, 2))
    denom = probs.sum(axis=(1, 2)) + onehot.sum(axis=(1, 2))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()
```

<div class="ex">
<div class="lab">Worked example</div>

**Dice versus IoU on one mask**

Ground truth has 100 foreground pixels, the prediction has 80, and 60 overlap.

Dice $= 2 \times 60 / (80 + 100) = 0.667$. IoU (Jaccard) $= 60 / (80 + 100 - 60) = 0.5$.

They are monotonically related, $D = 2J/(1+J)$: $2 \times 0.5 / 1.5 = 0.667$. Dice is always at least IoU, so do not compare a Dice number from one paper with an IoU number from another.

</div>

## Evaluation

Report mean IoU: build a $C \times C$ confusion matrix over all pixels of the validation set, compute per class $\text{IoU}_c = TP_c / (TP_c + FP_c + FN_c)$, then average over classes. Accumulating over the whole set (rather than averaging per-image IoUs) avoids undefined values on images where a class is absent; state which convention you use.

<div class="take">
<div class="lab">Key takeaway</div>

segmentation needs both context and resolution. Encoders provide context, skip connections restore spatial detail, and overlap-based losses like Dice keep tiny foreground classes from being drowned out by background.

</div>
