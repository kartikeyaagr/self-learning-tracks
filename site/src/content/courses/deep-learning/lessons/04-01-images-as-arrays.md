---
id: "04.01"
course: "deep-learning"
module: "04"
slug: "images-as-arrays"
title: "Images as arrays"
keyIdeas: ["An image is a tensor with conventions: `uint8` HWC on disk, `float32` NCHW in most training code. Most silent bugs are convention mismatches (RGB vs BGR, HWC vs CHW, 0-255 vs 0-1), not model bugs.", "Normalise with the statistics the model was trained with: divide by 255, then subtract the per-channel mean and divide by the per-channel std, computed on the training set only.", "Classic filters (box, Gaussian, Sobel) are fixed convolutions. They explain what early CNN layers learn and are still useful for preprocessing and debugging.", "Augmentation must preserve the label. Geometric transforms must also move the boxes or masks, and some (horizontal flip on text, rotation on digits like 6/9) change the label itself.", "Train-time augmentation is random; eval-time preprocessing is deterministic (resize, centre crop, normalise) and must match what the model saw in training."]
question: "A ResNet pretrained on ImageNet reports 76% top-1 accuracy, but in your pipeline (images loaded with OpenCV, converted with `reshape`, normalised with mean 0.5 and std 0.5) it gets 61%. List the likely causes, in the order you would check them, and how you would confirm each."
answer: "First, axis order: `reshape` to (3, H, W) scrambles pixels without an error; confirm by converting a tensor back to an image and viewing it, then switch to `transpose(2, 0, 1)`. Second, channel order: OpenCV gives BGR, the model expects RGB; confirm by swapping channels (`img[..., ::-1]`) and re-running a small validation slice. Third, normalisation statistics: the model was trained with ImageNet means and stds, not 0.5/0.5; the shift is small per pixel but systematic. Fourth, eval resizing: check you resize the short side to 256 and centre crop 224 with the same interpolation as the reference. Change one thing at a time on a fixed 1,000-image slice so each fix's effect is visible."
sources: [{"title": "torchvision: Transforming and augmenting images", "url": "https://pytorch.org/vision/stable/transforms.html", "note": "Reference for the standard train and eval pipelines, including how boxes and masks are transformed alongside images."}, {"title": "Dive into Deep Learning: Image Augmentation", "url": "https://d2l.ai/chapter_computer-vision/image-augmentation.html", "note": "Runnable walkthrough of flips, crops and colour jitter and their effect on training."}, {"title": "RandAugment (Cubuk et al., 2019)", "url": "https://arxiv.org/abs/1909.13719", "note": "How a two-parameter augmentation policy matches searched policies; read the method section."}, {"title": "mixup: Beyond Empirical Risk Minimization", "url": "https://arxiv.org/abs/1710.09412", "note": "The original mixup paper; short, and explains why mixing labels regularises."}]
practice: ["image-normalize", "per-channel-mean-and-std", "rgb-to-grayscale", "gaussian-blur-2d", "transform-pipeline"]
tags: ["preprocessing", "augmentation"]
tools: ["NumPy", "torchvision"]
---

## Where this module sits

Module 03 built the convolutional network itself. This module is about what goes in and out of it for real vision tasks: how pixels become tensors, how detectors and segmenters frame their outputs, and how transformers replaced convolutions in many backbones. This first lesson is the unglamorous part, and it is where a surprising share of production accuracy gets lost.

## An image is a tensor with conventions

A decoded JPEG is an array of shape $(H, W, 3)$ with dtype `uint8`, values 0 to 255. PyTorch layers expect `float32` in $(N, C, H, W)$ order; TensorFlow and JAX code often keeps channels last, $(N, H, W, C)$. A single 224×224 RGB image is 150,528 numbers; a batch of 256 in `float32` is about 154 MB before the network touches it, which is why data loaders keep images as `uint8` until the last moment.

Three conventions cause most silent failures:

- **Channel order.** OpenCV decodes to BGR; PIL and torchvision use RGB. Feeding BGR to an RGB-trained model does not crash; it just loses several points of accuracy because red and blue statistics are swapped.
- **Axis order.** Converting HWC to CHW needs `transpose(2, 0, 1)`. Using `reshape(3, H, W)` has the right shape but scrambles pixels into stripes. Nothing errors.
- **Value range.** Some pipelines expect 0-1 floats, some expect 0-255, some expect -1 to 1. Normalising twice or not at all both "work" and both hurt.

## Colour spaces

RGB is what cameras and screens use, but it entangles brightness with colour. Other spaces separate them:

| Space | Channels | Typical use |
| --- | --- | --- |
| RGB | red, green, blue | model input, display |
| Grayscale | luma $Y = 0.299R + 0.587G + 0.114B$ | edge detection, cheap models |
| HSV | hue, saturation, value | colour jitter (shift hue without changing brightness) |
| YCbCr | luma plus two chroma | JPEG compression, which stores chroma at lower resolution |

The grayscale weights are not equal because the eye is most sensitive to green. A second subtlety: stored pixel values are gamma-encoded (sRGB), so averaging two pixel values is not the same as averaging light intensities. For learning this rarely matters; for resizing and blending high-contrast edges it can.

## Normalisation

The standard recipe is: convert to float, divide by 255, then for each channel $c$ compute $x' = (x - \mu_c)/\sigma_c$. The ImageNet statistics, $\mu = (0.485, 0.456, 0.406)$ and $\sigma = (0.229, 0.224, 0.225)$, are baked into almost every pretrained torchvision model. If you train from scratch on your own data, compute $\mu$ and $\sigma$ over the training split only; using validation images leaks information, mildly but measurably.

```python
import numpy as np
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def to_model_input(img_hwc_uint8):           # (H, W, 3) RGB uint8
    x = img_hwc_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD                      # broadcasts over the last axis
    return x.transpose(2, 0, 1)               # (3, H, W), never reshape
```

Why normalise at all? Zero-centred, unit-scale inputs keep the first layer's pre-activations in a sensible range at initialisation, so the variance-preserving init from module 01 actually holds.

<div class="ex">
<div class="lab">Worked example</div>

**normalising one pixel**

A red channel value of 200 becomes $200/255 = 0.784$, then $(0.784 - 0.485)/0.229 = 1.31$.

A pure black pixel (0) in the blue channel becomes $(0 - 0.406)/0.225 = -1.80$. So normalised inputs live roughly in $[-2.1, 2.6]$, not $[0, 1]$.

</div>

## Filtering

Before CNNs, vision was hand-designed filters. They are worth knowing because first-layer CNN kernels learn very similar shapes:

- **Box blur**: every weight $1/k^2$. Cheap, but produces ringing artefacts.
- **Gaussian blur**: weights from $e^{-(x^2+y^2)/2\sigma^2}$. It is separable, so a $k \times k$ blur becomes a $k$-tap horizontal pass then a $k$-tap vertical pass: $2k$ multiplies per pixel instead of $k^2$.
- **Sobel**: two 3×3 kernels that approximate $\partial I/\partial x$ and $\partial I/\partial y$; the gradient magnitude $\sqrt{G_x^2 + G_y^2}$ highlights edges.

Boundary handling (zero, reflect, replicate padding) changes the outputs at the border; this matters more for small images. Module 03 covers learned convolution, stride and padding in detail.

## Augmentation

Augmentation creates new training examples that the label should be invariant to. Common families:

- **Geometric**: random resized crop (sample 8% to 100% of the area, then resize to 224), horizontal flip, small rotations and translations.
- **Photometric**: brightness, contrast, saturation and hue jitter; blur; noise.
- **Mixing**: mixup blends two images and their labels with weight $\lambda \sim \text{Beta}(\alpha, \alpha)$; CutMix pastes a patch from one image into another and mixes labels by the pasted area.
- **Policies**: RandAugment picks $N$ random operations at a shared magnitude $M$, reducing a large search space to two knobs.

~~~dg-flow One image's path from disk to the network; augmentation runs on the cheap uint8 form, normalisation last.
JPEG bytes -(decode)-> uint8 HWC -(random crop, flip, jitter)-> augmented -(to float, /255)-> *normalise -(transpose, batch)-> NCHW float32
~~~

The rule that matters is label preservation. Horizontal flips are fine for cats and wrong for text or for left-vs-right medical scans. For detection, every geometric transform must also transform the boxes; for segmentation it must transform the mask with nearest-neighbour interpolation so class ids stay integers. At evaluation time, use a fixed pipeline, typically resize the short side to 256 and centre crop 224, so results are reproducible.

<div class="take">
<div class="lab">Key takeaway</div>

treat preprocessing as part of the model. Channel order, axis order, value range, normalisation statistics and eval-time resizing must exactly match training, and every augmentation must keep the label true.

</div>
