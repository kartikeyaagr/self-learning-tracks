---
id: "03.01"
course: "deep-learning"
module: "03"
slug: "convolution"
title: "Convolution"
keyIdeas: ["A convolutional layer slides a small kernel of shape $C_{in} \\times k \\times k$ over the input and takes a dot product at every position; $C_{out}$ such kernels give $C_{out}$ output channels.", "Weight sharing and locality cut parameters from millions to thousands and make the layer translation-equivariant: shift the input, and the feature map shifts with it.", "Output size is $\\lfloor (H + 2p - d(k-1) - 1)/s \\rfloor + 1$; padding controls border behaviour, stride downsamples, dilation spreads the kernel.", "The receptive field grows with depth: each layer adds $(k-1)$ times the product of all earlier strides, so stacks of small kernels see large regions.", "Parameters are $C_{out} C_{in} k^2$ but compute is that times the number of output positions, so early high-resolution layers dominate FLOPs while late layers dominate parameters."]
question: "A network has two layers: a 3×3 conv with stride 2, then a 3×3 conv with stride 1. A colleague claims swapping their order (stride-1 first, then stride-2) leaves both the receptive field and the compute unchanged. Check both claims for a 64-channel to 64-channel network on a 128×128 input with same padding."
answer: "Receptive field: with stride 2 first, $r_1 = 3$, $j_1 = 2$, then $r_2 = 3 + 2 \\cdot 2 = 7$. With stride 1 first, $r_1 = 3$, $j_1 = 1$, then $r_2 = 3 + 2 \\cdot 1 = 5$. So the claim is false: putting the stride first makes the later layer's kernel cover a wider input region. Compute: each layer has $64 \\cdot 64 \\cdot 9 = 36{,}864$ weights. Stride 2 first: layer 1 outputs 64×64 (4,096 positions), layer 2 also at 64×64, total $2 \\times 36{,}864 \\times 4{,}096 \\approx 302$M MACs. Stride 1 first: layer 1 at 128×128 (16,384 positions), layer 2 at 64×64, total $36{,}864 \\times (16{,}384 + 4{,}096) \\approx 755$M MACs. Parameters are identical, but downsampling early gives a larger receptive field for about 40% of the compute, at the cost of losing high-resolution detail sooner. This is the trade-off every CNN stem makes."
sources: [{"title": "CS231n: Convolutional Neural Networks", "url": "https://cs231n.github.io/convolutional-networks/", "note": "The conv layer, output-size arithmetic, parameter sharing and im2col."}, {"title": "Dumoulin & Visin: A guide to convolution arithmetic for deep learning", "url": "https://arxiv.org/abs/1603.07285", "note": "Every padding/stride/dilation case drawn out, with exact size relations."}, {"title": "Luo et al.: Understanding the Effective Receptive Field", "url": "https://arxiv.org/abs/1701.04128", "note": "Why the effective receptive field is Gaussian-shaped and much smaller than the theoretical one."}, {"title": "Dive into Deep Learning: Padding and Stride", "url": "https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html", "note": "Shape calculations with runnable code."}]
practice: ["2d-convolution", "multichannel-2d-convolution", "conv2d", "dilated-2d-convolution"]
tags: ["convolution", "receptive field", "im2col"]
tools: ["NumPy", "PyTorch"]
---

## From dense layers to convolutions

Module 01's MLP showed the cost of dense layers on images: a 224×224×3 input flattened to 150,528 numbers, connected to just 1000 hidden units, needs 150 million weights, and the network has to learn separately that a cat in the top-left and a cat in the bottom-right are both cats. Convolutional networks build in two assumptions about images:

- **Locality**: useful features (edges, textures) depend on nearby pixels, so each output looks at a small window.
- **Stationarity**: a feature detector useful in one place is useful everywhere, so the same weights are reused at every position.

This module applies the training machinery of Modules 01-02 to convolutional networks; Module 04 then uses them for detection and segmentation.

## The operation

For an input of shape $(C_{in}, H, W)$ and a weight tensor $(C_{out}, C_{in}, k, k)$:

$$y_{o,i,j} = b_o + \sum_{c=1}^{C_{in}} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} w_{o,c,u,v}\; x_{c,\; s i + d u - p,\; s j + d v - p}$$

Every output channel looks at **all** input channels. (Deep learning libraries compute cross-correlation, without flipping the kernel; since kernels are learned the distinction doesn't matter.) The output size per spatial dimension, with padding $p$, stride $s$, dilation $d$:

$$H_{out} = \left\lfloor \frac{H + 2p - d(k-1) - 1}{s} \right\rfloor + 1$$

- **Padding** $p = (k-1)/2$ with $s = 1$ keeps the size ("same" padding) and lets border pixels influence as many outputs as central ones.
- **Stride** $s = 2$ halves the resolution, a learned alternative to pooling (03.02).
- **Dilation** $d$ inserts gaps in the kernel, enlarging its footprint without adding weights; used in segmentation to keep resolution.

~~~dg-flow One conv layer as tensor shapes: the kernel's channel depth always equals the input's channel count.
Input 3×224×224 -(64 kernels of 3×7×7, s=2, p=3)-> *Output 64×112×112 -(BN, ReLU)-> Next layer
~~~

## Translation equivariance and its limits

Because weights are shared, shifting the input by one pixel shifts the output by one pixel: convolution is **translation-equivariant**. Classification wants **invariance** (same label regardless of position), which comes later from pooling and global averaging. Equivariance is exact only for stride 1; stride and pooling make outputs sensitive to sub-stride shifts, which is why CNN predictions can flicker when an image moves by a pixel.

## Receptive field

The receptive field of a unit is the input region that can affect it. With $r_0 = 1$ and jump $j_0 = 1$:

$$r_l = r_{l-1} + (k_l - 1)\, j_{l-1}, \qquad j_l = j_{l-1}\, s_l$$

Three stacked 3×3 stride-1 convs see $1 + 2 + 2 + 2 = 7$ pixels: the same as one 7×7 kernel (the VGG argument in 03.03). Strides multiply the growth of everything after them, which is how a 50-layer ResNet reaches a theoretical receptive field larger than the image. The **effective** receptive field is much smaller: influence is roughly Gaussian-weighted toward the centre, so the outer region of the theoretical field contributes little.

## Cost: parameters versus compute

Parameters: $C_{out} C_{in} k^2$ (+ $C_{out}$ biases), independent of image size. Multiply-adds: that number times $H_{out} W_{out}$. Early layers have few channels but huge resolution, late layers many channels but tiny resolution, so early layers dominate compute and late layers dominate parameters.

Implementations turn convolution into matrix multiplication. **im2col** unrolls every $C_{in} k^2$ input patch into a column, giving a $(C_{in} k^2) \times (H_{out} W_{out})$ matrix; the weights reshape to $C_{out} \times (C_{in} k^2)$; one GEMM gives the output. cuDNN also uses implicit-GEMM, FFT and Winograd algorithms, picking per layer shape.

```python
import numpy as np

def conv2d(x, w, b, stride=1, pad=0):
    # x: (C_in, H, W), w: (C_out, C_in, k, k) -> (C_out, H_out, W_out), via im2col
    C, H, W = x.shape; O, _, k, _ = w.shape
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    cols = np.stack([x[:, i * stride:i * stride + k, j * stride:j * stride + k].ravel()
                     for i in range(Ho) for j in range(Wo)], axis=1)   # (C*k*k, Ho*Wo)
    return (w.reshape(O, -1) @ cols + b[:, None]).reshape(O, Ho, Wo)
```

<div class="ex">
<div class="lab">Worked example</div>

**sizing a ResNet stem**

Input 3×224×224, 64 kernels of 7×7, stride 2, padding 3. $H_{out} = \lfloor (224 + 6 - 6 - 1)/2 \rfloor + 1 = 112$, so the output is 64×112×112.

Parameters: $64 \cdot 3 \cdot 49 = 9{,}408$ weights. Compute: $9{,}408 \times 112 \times 112 \approx 118$ million multiply-adds per image.

Compare a late ResNet layer, 512→512 channels with 3×3 kernels at 7×7: 2.36 million weights (250× more) but $2.36\text{M} \times 49 \approx 116$ million MACs, about the same compute.

</div>

| Hyperparameter | Effect on output size | Effect on parameters |
| --- | --- | --- |
| Kernel size $k$ | shrinks by $k-1$ without padding | grows as $k^2$ |
| Padding $p$ | grows by $2p$ | none |
| Stride $s$ | divides by about $s$ | none |
| Dilation $d$ | shrinks by $d(k-1)$ without padding | none |
| Channels $C_{out}$ | sets output depth | grows linearly |

<div class="take">
<div class="lab">Key takeaway</div>

a conv layer is a small dense layer applied at every position with shared weights; know the output-size formula, how receptive fields compound, and that compute scales with resolution while parameters scale with channels.

</div>
