---
id: "03.02"
course: "deep-learning"
module: "03"
slug: "pooling-and-downsampling"
title: "Pooling and downsampling"
keyIdeas: ["Max pooling takes the largest value in each window (2×2, stride 2 is standard), giving small-shift invariance and halving resolution with no parameters; its gradient goes only to the arg-max position.", "Global average pooling collapses each channel to one number, replacing huge flatten-then-dense heads and letting the network accept any input size.", "Strided convolution is learned downsampling; most modern CNNs use it, or pooling, at a few stage boundaries while doubling channels.", "Transposed convolution upsamples by scattering each input value through the kernel; its output size is $(H-1)s - 2p + k$, and kernels not divisible by the stride cause checkerboard artifacts."]
question: "A segmentation decoder uses `ConvTranspose2d(k=3, stride=2, padding=1, output_padding=1)` and its outputs show a faint grid pattern even after long training. Explain the cause in terms of how many input contributions each output pixel receives, and give two alternative designs that keep the same output size."
answer: "A transposed conv scatters each input value through the 3×3 kernel at stride 2. Along one dimension, output positions alternate between being covered by one kernel tap and two overlapping taps, since a width-3 kernel placed every 2 pixels overlaps by one. In 2-D this gives a periodic pattern of 1, 2 or 4 contributions per pixel, so even with identical weights the output has a built-in 2-pixel grid, and the network must learn weights that exactly cancel it, which it rarely does. Alternatives with the same 2× upsampling: (1) a kernel divisible by the stride, `ConvTranspose2d(k=4, stride=2, padding=1)`, which gives uniform overlap and also doubles the size; (2) nearest-neighbour or bilinear upsampling by 2 followed by a 3×3 conv with padding 1, which separates resizing from learning and is the standard U-Net choice; (3) a conv producing $4C$ channels followed by pixel shuffle with factor 2."
sources: [{"title": "Dumoulin & Visin: A guide to convolution arithmetic", "url": "https://arxiv.org/abs/1603.07285", "note": "Pooling arithmetic and the transposed-convolution chapter with output-size relations."}, {"title": "Odena, Dumoulin, Olah: Deconvolution and Checkerboard Artifacts", "url": "https://distill.pub/2016/deconv-checkerboard/", "note": "Interactive explanation of uneven overlap and the resize-convolution fix."}, {"title": "Springenberg et al.: Striving for Simplicity: The All Convolutional Net", "url": "https://arxiv.org/abs/1412.6806", "note": "Replacing max pooling with strided convolution without accuracy loss."}, {"title": "Dive into Deep Learning: Pooling", "url": "https://d2l.ai/chapter_convolutional-neural-networks/pooling.html", "note": "Max and average pooling with padding and stride, from scratch."}]
practice: ["max-pool-2d", "pooling-layers", "transposed-convolution", "2d-transposed-convolution", "adaptive-average-pool-2d"]
tags: ["pooling", "transposed convolution", "global average pooling"]
tools: ["NumPy"]
---

## Why downsample

A classifier must turn a 224×224 image into one label. Reducing resolution step by step does three things: it cuts compute in later layers (cost scales with $H \times W$), it grows the receptive field quickly (strides multiply it, 03.01), and it builds invariance to small shifts. The standard pattern is a few **stages**: at each boundary, halve the resolution and double the channels, so compute per layer stays roughly constant while features become more abstract.

## Max and average pooling

Pooling applies a fixed reduction to each window per channel, with no parameters and no channel mixing:

- **Max pooling** keeps the strongest response, which reads as "is this feature present anywhere in the window". 2×2 with stride 2 is the default; AlexNet used overlapping 3×3 windows with stride 2.
- **Average pooling** keeps the mean, smoother and preserving more information, but diluting a strong local response.

The backward passes differ in a useful way. Average pooling spreads the upstream gradient equally over the window ($1/k^2$ each). Max pooling routes all of it to the position that held the maximum and gives zero to the rest, so implementations store the arg-max indices from the forward pass.

```python
import numpy as np

def maxpool2x2(x):                      # x: (C, H, W), H and W even
    C, H, W = x.shape
    win = x.reshape(C, H // 2, 2, W // 2, 2).transpose(0, 1, 3, 2, 4).reshape(C, H // 2, W // 2, 4)
    idx = win.argmax(-1)                # remember winners for backward
    return win.max(-1), idx

def maxpool2x2_backward(dout, idx, shape):
    C, H, W = shape
    g = np.zeros((C, H // 2, W // 2, 4))
    np.put_along_axis(g, idx[..., None], dout[..., None], axis=-1)
    return g.reshape(C, H // 2, W // 2, 2, 2).transpose(0, 1, 3, 2, 4).reshape(C, H, W)
```

## Global average pooling

At the end of the network, **global average pooling** (GAP) reduces each $C \times h \times w$ feature map to a length-$C$ vector by averaging each channel over all positions. Compare heads:

- VGG-16 flattens 512×7×7 = 25,088 values into a 4096-unit dense layer: about 103 million weights, three quarters of the whole model.
- ResNet-50 applies GAP to 2048×7×7, giving 2048 values, then one 2048→1000 dense layer: about 2 million weights.

GAP also makes the network size-agnostic (any input resolution gives a $C$-vector) and acts as a structural regulariser. **Adaptive** pooling generalises it: pick the output size (say 7×7) and the window adapts to the input.

## Strided convolution instead of pooling

Springenberg et al. showed you can replace every pooling layer with a stride-2 convolution and lose nothing: the network learns how to downsample. ResNets downsample with stride-2 convs inside blocks; many modern designs use a stride-2 conv or a "patchify" stem. One caveat applies to both: subsampling without low-pass filtering aliases, so outputs can change noticeably under one-pixel shifts. Blurring before subsampling (anti-aliased pooling) restores much of the shift stability.

~~~dg-flow A classic stage layout: resolution halves at each boundary while channels double, keeping compute per stage roughly flat.
224×224×3 -(stem, s=2 + pool)-> 56×56×64 -(stage, s=2)-> 28×28×128 -(s=2)-> 14×14×256 -(s=2)-> 7×7×512 -(global avg pool)-> *512 -(dense)-> 1000 logits
~~~

## Upsampling: transposed convolution

Segmentation, generators and decoders need to go the other way. **Transposed convolution** (sometimes misnamed deconvolution) is the gradient of a convolution with respect to its input, used as a forward operation. Each input value multiplies the whole kernel and is added into the output at stride-spaced positions; overlapping contributions sum. Output size:

$$H_{out} = (H - 1)s - 2p + d(k - 1) + p_{out} + 1$$

which with $d = 1$ and no output padding is $(H-1)s - 2p + k$. The extra `output_padding` resolves the ambiguity that several input sizes map to the same strided-conv output size.

**Checkerboard artifacts**: when $k$ is not divisible by $s$ (say $k = 3, s = 2$), some output pixels receive contributions from more input positions than their neighbours, imprinting a grid pattern. Fixes: use $k$ divisible by $s$ (4 with stride 2), or upsample with nearest-neighbour or bilinear interpolation and then apply a normal convolution, which U-Nets and many generators do. **Pixel shuffle** (sub-pixel convolution) is a third option: a conv produces $C \cdot r^2$ channels and a reshape rearranges them into an $r$-times larger image.

<div class="ex">
<div class="lab">Worked example</div>

**one DCGAN-style upsampling layer**

Input 512×4×4, transposed conv to 256 channels with $k = 4$, $s = 2$, $p = 1$: $H_{out} = (4-1) \cdot 2 - 2 + 4 = 8$, so 256×8×8. Four such layers take 4×4 to 64×64.

Because $4$ is divisible by $2$, every output pixel receives exactly $(k/s)^2 = 4$ kernel contributions, so overlap is even and no grid pattern is built in.

With $k = 3$, $s = 2$, $p = 1$, $p_{out} = 1$ the output is also 8×8, but output pixels alternate between 1, 2 and 4 contributions: the checkerboard.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

downsample a few times with pooling or strided convs while doubling channels, finish with global average pooling instead of a giant dense head, and upsample with transposed convs whose kernel is divisible by the stride, or with resize-then-convolve.

</div>
