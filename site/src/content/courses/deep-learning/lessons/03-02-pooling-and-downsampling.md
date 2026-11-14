---
id: "03.02"
course: "deep-learning"
module: "03"
slug: "pooling-and-downsampling"
title: "Pooling and downsampling"
keyIdeas: ["Max pooling keeps the largest value in each window (2×2 with stride 2 is standard), halving height and width with no parameters; its gradient goes entirely to the arg-max position, while average pooling splits the gradient evenly.", "CNNs work in stages: at each boundary they halve the resolution and double the channels, which keeps the cost per layer roughly constant while the receptive field grows faster.", "Global average pooling collapses each channel to one number, replacing huge flatten-then-dense heads (VGG-16: about 103 million weights in one layer, versus about 2 million for ResNet-50's head) and letting the network accept any input size.", "A stride-2 convolution is learned downsampling and is what most modern CNNs use; both it and pooling alias, so outputs can flicker under one-pixel shifts unless you blur before subsampling.", "Transposed convolution upsamples by scattering each input value through the kernel; its output size is $(H-1)s - 2p + k$, and a kernel not divisible by the stride causes checkerboard artifacts (use $k=4, s=2$, resize-then-conv, or pixel shuffle)."]
question: "A segmentation decoder uses `ConvTranspose2d(k=3, stride=2, padding=1, output_padding=1)` and its outputs show a faint grid pattern even after long training. Explain the cause in terms of how many input contributions each output pixel receives, and give two alternative designs that keep the same output size."
answer: "A transposed conv scatters each input value through the 3×3 kernel, moving 2 output pixels per input pixel. Along one dimension a width-3 patch placed every 2 pixels overlaps its neighbour by exactly one pixel, so output positions alternate between receiving one contribution and two. In 2-D the counts form a periodic pattern of 1, 2 and 4 contributions per pixel (you can verify this by setting all weights to 1 and feeding an input of ones). So even before training the layer has a built-in 2-pixel grid, and the network would have to learn weights that cancel it exactly, which it rarely does; the residue shows up as the checkerboard. Alternatives with the same 2× upsampling and the same output size: (1) a kernel divisible by the stride, `ConvTranspose2d(k=4, stride=2, padding=1)`, which gives $(4-1) \\cdot 2 - 2 + 4 = 8$ from 4 and uniform overlap of 4 contributions per interior pixel; (2) nearest-neighbour or bilinear upsampling by 2 followed by a 3×3 conv with padding 1, which separates resizing from learning and is the standard U-Net choice; (3) a conv producing $4C$ channels followed by `PixelShuffle(2)`, which rearranges channels into a twice-larger image with no overlap at all."
sources: [{"title": "Dumoulin & Visin: A guide to convolution arithmetic", "url": "https://arxiv.org/abs/1603.07285", "note": "Pooling arithmetic and the transposed-convolution chapter with output-size relations."}, {"title": "Odena, Dumoulin, Olah: Deconvolution and Checkerboard Artifacts", "url": "https://distill.pub/2016/deconv-checkerboard/", "note": "Interactive explanation of uneven overlap and the resize-convolution fix."}, {"title": "Springenberg et al.: Striving for Simplicity: The All Convolutional Net", "url": "https://arxiv.org/abs/1412.6806", "note": "Replacing max pooling with strided convolution without accuracy loss."}, {"title": "Dive into Deep Learning: Pooling", "url": "https://d2l.ai/chapter_convolutional-neural-networks/pooling.html", "note": "Max and average pooling with padding and stride, from scratch."}]
practice: ["max-pool-2d", "pooling-layers", "transposed-convolution", "2d-transposed-convolution", "adaptive-average-pool-2d"]
tags: ["pooling", "transposed convolution", "global average pooling"]
tools: ["NumPy"]
---

## What you will learn

- Why a CNN shrinks its feature maps step by step, and what that buys: less compute, a faster-growing receptive field, and tolerance to small shifts.
- How max pooling and average pooling work, forward and backward, on a 4×4 example you can check by hand.
- How global average pooling replaces the huge dense "head" of older networks, and why it lets a network accept any image size.
- When a stride-2 convolution is used instead of pooling, and why naive subsampling makes outputs flicker under one-pixel shifts (aliasing).
- How to go the other way (upsampling) with transposed convolution, its output-size formula, and how to avoid checkerboard artifacts.

*Before you start:* you need the convolution layer, its output-size formula, stride and receptive field from lesson 03.01, and backpropagation from module 01.

## From convolution to downsampling

Lesson 03.01 built the convolution layer. With stride 1 and "same" padding it keeps the height and width of its input, so twenty stacked layers on a 224×224 image all work at 224×224: expensive, and each unit still sees only a small patch. A classifier must turn the whole image into one label. This lesson covers the layers that do that shrinking, called **downsampling**: pooling, strided convolution and global average pooling. At the end we reverse direction and look at **upsampling**, which segmentation models and image generators need.

## The idea in one picture

Imagine a convolution layer has produced one channel of a feature map: a 4×4 grid of how strongly an "edge detector" fired at each position.

| Row | col 1 | col 2 | col 3 | col 4
| --- | --- | --- | --- | --- |
| 1 | 1 | 3 | 2 | 0
| 2 | 4 | 6 | 1 | 1
| 3 | 0 | 2 | 5 | 7
| 4 | 1 | 1 | 3 | 2

Cut it into four non-overlapping 2×2 tiles and replace each tile by one number. Keep the largest value and you get **max pooling**; keep the mean and you get **average pooling**.

~~~dg-fig 03.02-pool.svg
Each coloured 2×2 window becomes one output number. Max pooling keeps the strongest response (6, 2, 2, 7); average pooling keeps the mean (3.5, 1, 1, 4.25). Both halve height and width.
~~~

Read the max-pooled result as a coarse map: "the edge fired strongly somewhere in the top-left quarter (6) and the bottom-right quarter (7)". The exact pixel is gone; presence per region is kept. Average pooling gives a softer summary: the bottom-right quarter is busy overall (4.25), the top-right quiet (1).

Three terms used throughout:

- A **pooling window** (or kernel) is the patch being summarised, here 2×2.
- The **stride** is how far the window moves each step. Stride 2 with a 2×2 window gives non-overlapping tiles.
- **Downsampling** means producing a smaller grid. A 2×2, stride-2 pool cuts the number of positions by 4.

Pooling has **no learned parameters** and works **per channel**: a feature map with 64 channels gives 64 pooled channels, each pooled on its own. Mixing information across channels is the convolution's job.

## Step 1: why shrink at all

Downsampling earns its place for three separate reasons.

**1. Compute.** The cost of a convolution layer is roughly

$$\text{multiply-adds} = H_{out} \times W_{out} \times k^2 \times C_{in} \times C_{out},$$

where $H_{out} \times W_{out}$ is the number of output positions, $k$ the kernel size, and $C_{in}, C_{out}$ the input and output channel counts. Halving height and width divides cost by 4. The standard **stage** pattern spends that saving on channels: at each stage boundary, halve the resolution and double the channels. Doubling both $C_{in}$ and $C_{out}$ multiplies cost by 4, cancelling the saving exactly.

~~~dg-fig 03.02-stages.svg
Left: multiply-adds of the 13 conv layers of VGG-16 (computed from its layer shapes). Every layer that keeps its width costs the same 1.85 billion in stages 1-4, because halving the size and doubling the channels cancel; the last stage keeps 512 channels, so its layers are cheaper. Right: the classifier head after the last stage, which global average pooling shrinks by a factor of 60 (Step 4).
~~~

Number check: VGG-16's second layer (224×224, 64→64) costs $224^2 \cdot 9 \cdot 64 \cdot 64 = 1.85$ billion; its sixth (56×56, 256→256) costs $56^2 \cdot 9 \cdot 256 \cdot 256 = 1.85$ billion.

**2. Receptive field.** Recall from 03.01 that each layer adds $(k-1) \times j$ to the receptive field, where the **jump** $j$ is the product of all earlier strides. A stride-2 layer doubles $j$, so every layer after it grows the receptive field twice as fast. Example: a 3×3 conv sees 3 pixels; a 2×2 stride-2 pool brings it to $3 + 1 \cdot 1 = 4$ with jump 2; the next 3×3 conv reaches $4 + 2 \cdot 2 = 8$ instead of 5.

**3. Tolerance to small shifts.** If the top-left 6 in our example moved one pixel right, it would still be inside the same window and max pooling would still output 6. We measure this below.

~~~dg-flow A classic stage layout: resolution halves at each boundary while channels double, keeping compute per stage roughly flat.
224×224×3 -(stem, s=2 + pool)-> 56×56×64 -(stage, s=2)-> 28×28×128 -(s=2)-> 14×14×256 -(s=2)-> 7×7×512 -(global avg pool)-> *512 -(dense)-> 1000 logits
~~~

## Step 2: max and average pooling, precisely

For one channel, a window of size $k$ and stride $s$, output position $(i, j)$ looks at the input patch starting at row $s \cdot i$ and column $s \cdot j$:

$$y_{ij} = \max_{0 \le u, v < k} x_{si+u,\; sj+v} \quad \text{(max)}, \qquad y_{ij} = \frac{1}{k^2}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} x_{si+u,\; sj+v} \quad \text{(average)}.$$

Here $x$ is the input map, $y$ the output map, and $u, v$ index positions inside the window. The output size follows the same rule as a convolution (03.01) with padding $p$:

$$H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1.$$

Check: $H = 4$, $k = 2$, $s = 2$, $p = 0$ gives $\lfloor 2/2 \rfloor + 1 = 2$. AlexNet used *overlapping* 3×3 windows with stride 2: a 55×55 map becomes $\lfloor 52/2 \rfloor + 1 = 27$. Today 2×2 with stride 2 is the default, and a 3×3, stride-2, padding-1 pool (as in the ResNet stem) maps 112 to 56.

How to choose between them:

- **Max pooling** answers "is this feature present anywhere in the window?", which suits detectors in classic classifiers.
- **Average pooling** answers "how much of this feature is in the window?". It is smoother and keeps more information, but dilutes one strong response.

## Step 3: the backward pass

Pooling has no weights, but gradients must flow through it. Given the **upstream gradient** $\partial L / \partial y_{ij}$ (how much the loss changes per unit change of each pooled output), we need $\partial L / \partial x$ for every input pixel.

- **Average pooling**: each output is $1/k^2$ times the sum of its window, so each input in the window receives $\frac{1}{k^2} \cdot \partial L/\partial y_{ij}$. The gradient is spread evenly.
- **Max pooling**: the output equals the winning input; nudging a loser changes nothing. So the winner receives the whole upstream gradient and every other position exactly zero.

That is why implementations store the **arg-max** (the position of each winner) during the forward pass: the backward pass needs it to know where to send the gradient.

<div class="ex">
<div class="lab">Worked example</div>

**pooling forward and backward by hand**

Input map (rows): $[1, 3, 2, 0]$, $[4, 6, 1, 1]$, $[0, 2, 5, 7]$, $[1, 1, 3, 2]$. Pool 2×2 with stride 2.

Top-left window $\{1, 3, 4, 6\}$: max $= 6$, mean $= 14/4 = 3.5$. Top-right $\{2, 0, 1, 1\}$: max $= 2$, mean $= 4/4 = 1$. Bottom-left $\{0, 2, 1, 1\}$: max $= 2$, mean $= 1$. Bottom-right $\{5, 7, 3, 2\}$: max $= 7$, mean $= 17/4 = 4.25$.

Max output $[[6, 2], [2, 7]]$; average output $[[3.5, 1], [1, 4.25]]$. Sixteen numbers became four.

Backward: suppose the upstream gradient is $[[1, -2], [0.5, 3]]$. Under max pooling, 1 goes to the 6 at (row 2, col 2), $-2$ to the 2 at (1, 3), 0.5 to the 2 at (3, 2), and 3 to the 7 at (3, 4). The other twelve positions get 0.

Under average pooling, each window shares its gradient in quarters: the top-left four positions get $1/4 = 0.25$, the top-right $-2/4 = -0.5$, the bottom-left $0.5/4 = 0.125$, the bottom-right $3/4 = 0.75$.

Sanity check: in both cases the input gradients of each window add up to that window's upstream gradient ($0.25 \times 4 = 1$, and so on), because each pooled output is a function of its own window only.

</div>

~~~dg-fig 03.02-backward.svg
The same upstream gradient routed back through the two pools. Max pooling sends everything to the winner (bold) and zero elsewhere; average pooling splits it evenly. Values verified against PyTorch autograd.
~~~

## Step 4: global average pooling

At the end of the convolutional stages you have a tensor such as 2048 channels × 7 × 7 positions. To classify, you need one vector. **Global average pooling** (GAP) averages each channel over all its positions:

$$z_c = \frac{1}{h\,w}\sum_{i=1}^{h}\sum_{j=1}^{w} x_{c,i,j},$$

giving one number per channel $c$, so a 2048-long vector. Compare two ways of attaching a classifier for 1000 classes:

- VGG-16 flattens 512×7×7 = 25,088 values into a 4096-unit dense layer: $25{,}088 \times 4096 + 4096 \approx 102.8$ million weights in that single layer, about three quarters of the 138-million-parameter model. With its two further dense layers the head holds 123.6 million.
- ResNet-50 applies GAP to 2048×7×7, then one dense layer $2048 \to 1000$: $2048 \times 1000 + 1000 \approx 2.05$ million weights.

GAP brings three benefits. It removes most of the parameters and the overfitting risk they carry (a "structural regulariser"). It makes the network **size-agnostic**: a 320×320 image gives a 10×10 final map instead of 7×7, but averaging still yields one 2048-vector. And each channel becomes a "how much of concept $c$ is here" detector, which class activation maps exploit.

**Adaptive pooling** generalises GAP: you fix the *output* size (say 7×7) and the layer picks the window sizes to fit whatever input arrives. `AdaptiveAvgPool2d(1)` is exactly GAP.

## Step 5: does pooling really buy shift tolerance?

Take a random smooth 64×64 image, apply one fixed conv layer (16 channels, ReLU), and compare its output with the output for the same image shifted right by 0 to 8 pixels, using cosine similarity position by position (1 means identical).

~~~dg-fig 03.02-shift.svg
Measured on a synthetic image (circular shifts). Without pooling, similarity falls to 0.62 after a 2-pixel shift; one 2×2 max pool keeps 0.74 and two keep 0.89. Global average pooling ignores position entirely and stays at 1.
~~~

Each pooling step widens the shift a feature can survive, and global pooling makes the summary fully **invariant** (unchanged by shifts). Convolution alone is only **equivariant**: the output shifts along with the input (03.01). Classification needs invariance, and the network builds it gradually.

## Step 6: strided convolution, and the aliasing catch

A **strided convolution** (stride 2) downsamples while learning *how* to summarise each window, and mixes channels at the same time. Springenberg et al. ("The All Convolutional Net") replaced every max pool with a stride-2 conv and matched the accuracy of the pooled baselines on CIFAR. ResNets downsample with stride-2 convs inside blocks (keeping one max pool in the stem); many newer models use a "patchify" stem.

Pooling and strided convolution share one flaw: they keep every second value without first removing fine detail. The result is **aliasing**: detail too fine for the new grid masquerades as a coarser pattern, which changes when the input moves by one pixel.

~~~dg-fig 03.02-alias.svg
A stripe pattern (1, 0, 1, 0, …) subsampled by keeping every second value gives all ones or all zeros depending on a one-pixel shift. Blurring with [1, 2, 1]/4 first gives 0.5 for both shifts.
~~~

This is why a CNN's prediction can flicker as an object slides by one pixel. The fix is to low-pass filter before subsampling, called **anti-aliased** or **blur pooling** (Zhang, 2019, "Making Convolutional Networks Shift-Invariant Again"): compute max with stride 1, blur with a small fixed kernel, then subsample. It restores much of the shift stability at little cost.

## Step 7: going back up with transposed convolution

Segmentation networks, decoders and image generators must turn a small map into a big one. The learnable tool is **transposed convolution** (sometimes misnamed "deconvolution"). Each input value multiplies the whole kernel, and the resulting $k \times k$ patch is added into the output at a spot that moves $s$ pixels per input pixel; overlaps sum. Formally it is the gradient of a convolution with respect to its input, used as a forward layer.

Output size, per dimension, with input size $H$, stride $s$, padding $p$, kernel $k$, dilation $d$ and output padding $p_{out}$:

$$H_{out} = (H - 1)s - 2p + d(k - 1) + p_{out} + 1,$$

which for $d = 1$ and $p_{out} = 0$ is $(H-1)s - 2p + k$. The `output_padding` term exists because a stride-2 convolution maps both 7 and 8 to 4, so going back from 4 is ambiguous; $p_{out}$ picks which size you want.

**Checkerboard artifacts.** When $k$ is not divisible by $s$, the scattered patches overlap unevenly: some output pixels collect more contributions than their neighbours. Run a transposed conv with all weights set to 1 on an input of all ones and simply count:

~~~dg-fig 03.02-overlap.svg
Contributions per output pixel for a 4×4 input upsampled to 8×8 (computed with PyTorch, all weights 1). With k = 3, stride 2 the counts alternate 1, 2, 4: a grid is built into the layer. With k = 4, stride 2 every interior pixel receives exactly 4.
~~~

Before any training, the k = 3 layer stamps a 2-pixel grid on its output that the network must learn to cancel exactly, which it rarely does. Fixes:

- Use a kernel divisible by the stride: $k = 4$ with $s = 2$.
- **Resize-then-convolve**: nearest-neighbour or bilinear upsampling by 2, then an ordinary 3×3 conv with padding 1. U-Nets and many generators do this.
- **Pixel shuffle** (sub-pixel convolution): a normal conv outputs $C \cdot r^2$ channels and a reshape rearranges each group of $r^2$ channels into an $r \times r$ block, giving an $r$-times larger image.

<div class="ex">
<div class="lab">Worked example</div>

**one DCGAN-style upsampling layer**

Input 512×4×4, transposed conv to 256 channels with $k = 4$, $s = 2$, $p = 1$: $H_{out} = (4-1) \cdot 2 - 2 \cdot 1 + 4 = 6 - 2 + 4 = 8$, so the output is 256×8×8. Four such layers take 4×4 to 8, 16, 32 and finally 64×64.

Because 4 is divisible by 2, every interior output pixel receives exactly $(k/s)^2 = 2^2 = 4$ contributions, as in the right panel of the figure, so no grid is built in.

With $k = 3$, $s = 2$, $p = 1$, $p_{out} = 1$: $(4-1) \cdot 2 - 2 + 2 + 1 + 1 = 8$, also 8×8, but the counts alternate between 1, 2 and 4: the checkerboard.

</div>

## Pooling and resampling in code

A from-scratch max pool in NumPy (forward and backward), checked against PyTorch, then the standard PyTorch layers with the shapes they produce:

```python
import numpy as np, torch, torch.nn as nn

def maxpool2x2(x):                      # x: (C, H, W), H and W even
    C, H, W = x.shape
    win = x.reshape(C, H // 2, 2, W // 2, 2).transpose(0, 1, 3, 2, 4).reshape(C, H // 2, W // 2, 4)
    idx = win.argmax(-1)                # remember the winners for backward
    return win.max(-1), idx

def maxpool2x2_backward(dout, idx, shape):
    C, H, W = shape
    g = np.zeros((C, H // 2, W // 2, 4))
    np.put_along_axis(g, idx[..., None], dout[..., None], axis=-1)
    return g.reshape(C, H // 2, W // 2, 2, 2).transpose(0, 1, 3, 2, 4).reshape(C, H, W)

x = np.array([[[1, 3, 2, 0], [4, 6, 1, 1], [0, 2, 5, 7], [1, 1, 3, 2]]], float)
y, idx = maxpool2x2(x)
print(y[0])                                   # [[6. 2.] [2. 7.]]
g = maxpool2x2_backward(np.array([[[1., -2.], [0.5, 3.]]]), idx, x.shape)
xt = torch.tensor(x).requires_grad_(True)     # same check with autograd
(nn.functional.max_pool2d(xt, 2) * torch.tensor([[[1., -2.], [0.5, 3.]]], dtype=torch.float64)).sum().backward()
print(np.allclose(xt.grad.numpy(), g))        # True

z = torch.randn(8, 64, 56, 56)                # (batch, channels, H, W)
print(nn.MaxPool2d(2)(z).shape)                              # [8, 64, 28, 28]
print(nn.MaxPool2d(3, stride=2, padding=1)(z).shape)         # [8, 64, 28, 28]
print(nn.Conv2d(64, 128, 3, stride=2, padding=1)(z).shape)   # [8, 128, 28, 28]  learned downsampling
print(nn.AdaptiveAvgPool2d(1)(z).flatten(1).shape)           # [8, 64]  global average pool
u = torch.randn(1, 512, 4, 4)
print(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)(u).shape)                    # [1, 256, 8, 8]
print(nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)(u).shape)  # [1, 256, 8, 8]
print(nn.PixelShuffle(2)(nn.Conv2d(512, 256 * 4, 3, padding=1)(u)).shape)              # [1, 256, 8, 8]
```

Print shapes after every downsampling layer while designing; off-by-one sizes come from the floor in the output formula.

## Checking your downsampling path

- **Shapes add up.** Input size divided by the product of all strides: 224 with five stride-2 steps gives 7. A 6 or 8 means a padding or `ceil_mode` choice is off.
- **Shift stability.** Shift validation images by 1 to 4 pixels and count how often the predicted class changes. Frequent flips point at aliasing.
- **Upsampled outputs.** Zoom in on generated or segmented images. A faint grid whose period equals the stride is the checkerboard signature.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Shape mismatch when concatenating encoder and decoder maps (U-Net skip) | odd input size: 7 → 4 → 8, not 7 | pad inputs to a multiple of the total stride, or set `output_padding`, or crop
| Model crashes on a different image size | hard-coded flatten size or fixed `AvgPool2d(7)` | use `AdaptiveAvgPool2d(1)` before the dense layer
| Huge parameter count, fast overfitting | flatten-then-dense head | global average pooling plus one linear layer
| Prediction flickers when the object moves a pixel | aliasing from stride or pooling without blur | blur pooling, more augmentation with small shifts
| Faint grid in generated or segmented images | transposed conv with $k$ not divisible by $s$ | $k = 4, s = 2$; resize-then-conv; pixel shuffle
| Small objects vanish from the features | too many downsampling steps, or max pooled away early | fewer stride-2 steps, dilated convs (03.01), or a feature pyramid (module 04)

## Going deeper

**Pooling as a prior.** Max pooling assumes exact position inside a window does not matter. For precise localisation that is false, so segmentation and detection networks keep high-resolution paths too (module 04).

**Why aliasing happens, in one line.** Subsampling by $s$ can only represent spatial frequencies below $1/(2s)$ cycles per pixel (the Nyquist limit of the new grid). Anything finer folds back as a false low frequency. A stride-1 max pool creates high frequencies of its own (it is a non-linear operation with sharp edges), so blur pooling places the low-pass filter after the max and before the subsampling.

**Generalised pooling.** Max and average are two ends of a family. **Lp pooling** computes $\big(\frac{1}{k^2}\sum x^p\big)^{1/p}$: for non-negative inputs $p = 1$ is average and $p \to \infty$ approaches max. Generalised-mean (GeM) pooling learns $p$ and is popular in image retrieval.

**Transposed convolution as a matrix.** Write a convolution as a sparse matrix $M$ acting on the flattened input, $y = Mx$. The transposed layer computes $M^\top y$, mapping the small space back to the big one; it is exactly the operation that backpropagates through a strided conv. It is not an inverse: $M^\top M \neq I$, so it does not undo a convolution.

## Key terms

| Term | Meaning
| --- | --- |
| Downsampling | Producing a smaller spatial grid, by pooling or by a strided convolution
| Max pooling | Keep the largest value in each window; gradient goes only to that position
| Average pooling | Keep the mean of each window; gradient is split evenly
| Global average pooling | Average each channel over all positions, giving one number per channel
| Adaptive pooling | Pooling that fixes the output size and adapts its windows to the input size
| Strided convolution | A convolution with stride greater than 1: learned downsampling
| Aliasing | False coarse patterns created by subsampling without first removing fine detail
| Transposed convolution | Learned upsampling that scatters each input through the kernel; output $(H-1)s - 2p + k$
| Checkerboard artifact | A grid pattern from uneven kernel overlap when $k$ is not divisible by $s$

<div class="take">
<div class="lab">Key takeaway</div>

downsample a few times with pooling or strided convolutions while doubling channels, finish with global average pooling instead of a giant dense head, blur before subsampling if shift stability matters, and upsample with transposed convolutions whose kernel is divisible by the stride, or with resize-then-convolve.

</div>
