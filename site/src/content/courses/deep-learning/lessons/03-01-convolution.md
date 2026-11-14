---
id: "03.01"
course: "deep-learning"
module: "03"
slug: "convolution"
title: "Convolution"
keyIdeas: ["A convolutional layer slides a small kernel of shape $C_{in} \\times k \\times k$ over the input and takes a dot product at every position; each kernel produces one output channel, and $C_{out}$ kernels give $C_{out}$ channels.", "Locality and weight sharing cut parameters from billions to thousands (64 RGB 3×3 kernels need only 1,728 weights) and make the layer translation-equivariant: shift the input, and the feature map shifts with it, exactly so only at stride 1.", "Output size is $\\lfloor (H + 2p - d(k-1) - 1)/s \\rfloor + 1$: padding keeps borders and size, stride downsamples, dilation spreads the kernel without adding weights.", "The receptive field grows with depth: each layer adds $(k-1)$ times the product of all earlier strides. The effective receptive field is a narrow, roughly Gaussian bump inside the theoretical one.", "Parameters are $C_{out} C_{in} k^2$; compute is that times the number of output positions. With the usual 'halve the resolution, double the channels' design, compute per layer stays roughly flat while parameters pile up in the late, wide layers."]
question: "A network has two layers: a 3×3 conv with stride 2, then a 3×3 conv with stride 1. A colleague claims swapping their order (stride-1 first, then stride-2) leaves both the receptive field and the compute unchanged. Check both claims for a 64-channel to 64-channel network on a 128×128 input with same padding."
answer: "Receptive field: with stride 2 first, $r_1 = 3$, $j_1 = 2$, then $r_2 = 3 + 2 \\cdot 2 = 7$. With stride 1 first, $r_1 = 3$, $j_1 = 1$, then $r_2 = 3 + 2 \\cdot 1 = 5$. So the claim is false: putting the stride first makes the later layer's kernel cover a wider input region, because its neighbouring units are already 2 pixels apart. Compute: each layer has $64 \\cdot 64 \\cdot 9 = 36{,}864$ weights (parameters are identical in both orders). With same padding ($p = 1$), a stride-2 layer on 128×128 gives $\\lfloor (128 + 2 - 2 - 1)/2 \\rfloor + 1 = 64$. Stride 2 first: layer 1 outputs 64×64 (4,096 positions), layer 2 also at 64×64, total $2 \\times 36{,}864 \\times 4{,}096 \\approx 302$M multiply-adds. Stride 1 first: layer 1 at 128×128 (16,384 positions), layer 2 at 64×64, total $36{,}864 \\times (16{,}384 + 4{,}096) \\approx 755$M multiply-adds. Downsampling early gives a larger receptive field for about 40% of the compute; the price is that only one layer ever sees the full-resolution detail, which can hurt small objects and fine texture. This is the trade-off every CNN stem makes, and it is why stems downsample aggressively (ResNet's 7×7 stride-2 conv plus pooling) while dense-prediction models keep more resolution."
sources: [{"title": "CS231n: Convolutional Neural Networks", "url": "https://cs231n.github.io/convolutional-networks/", "note": "The conv layer, output-size arithmetic, parameter sharing and im2col."}, {"title": "Dumoulin & Visin: A guide to convolution arithmetic for deep learning", "url": "https://arxiv.org/abs/1603.07285", "note": "Every padding/stride/dilation case drawn out, with exact size relations."}, {"title": "Luo et al.: Understanding the Effective Receptive Field", "url": "https://arxiv.org/abs/1701.04128", "note": "Why the effective receptive field is Gaussian-shaped and much smaller than the theoretical one."}, {"title": "Dive into Deep Learning: Padding and Stride", "url": "https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html", "note": "Shape calculations with runnable code."}]
practice: ["2d-convolution", "multichannel-2d-convolution", "conv2d", "dilated-2d-convolution"]
tags: ["convolution", "receptive field", "im2col"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- Why dense layers suit images badly, and the two assumptions convolution builds in instead.
- How one convolution is computed, by hand and in NumPy and PyTorch.
- How padding, stride and dilation set the output size, with one formula for every case.
- What translation equivariance is and how stride breaks it.
- How far a unit can "see" (its receptive field), and where parameters and compute go.

*Before you start:* you should know the dense layer and the MLP (01.01), backpropagation (01.04) and batch normalisation (02.01). Basic NumPy indexing helps for the code.

## Where this module starts

Modules 01 and 02 trained networks of dense layers, which treat the input as an unordered list of numbers. A 224×224 colour photo flattened is 150,528 numbers; connecting it to just 1000 hidden units needs about 150 million weights, and the network must learn separately that a cat top-left and a cat bottom-right are both cats. Convolutional networks (CNNs) build two facts about images into the layer:

- **Locality**: low-level patterns (edges, corners, textures) depend on a few neighbouring pixels, so each output looks at a small window.
- **Weight sharing**: an edge detector useful in one place is useful everywhere, so the same few weights are reused at every position.

The rest of the module builds on this layer: pooling (03.02), LeNet, AlexNet and VGG (03.03), residual networks (03.04) and efficient designs (03.05). Module 04 uses them for detection and segmentation.

## The idea in one picture

Take a tiny 6×6 greyscale image: the left three columns are bright (value 10) and the right three are dark (value 0). There is one vertical edge down the middle. Now take a 3×3 grid of weights called a **kernel** (or filter):

| 1 | 0 | -1
| --- | --- | --- |
| 1 | 0 | -1
| 1 | 0 | -1

Place the kernel over the top-left 3×3 window, multiply each pixel by the weight on top of it, and add the nine products. Slide one pixel and repeat at every position where it fits. The grid of results is a **feature map**.

~~~dg-fig 03.01-slide.svg
A 3×3 vertical-edge kernel slid over a 6×6 image. Inside a flat region the +1 and −1 columns cancel to 0; only windows that straddle the edge give a large value (30).
~~~

In plain words, the kernel computes "left column minus right column". Inside the bright part that is $30 - 30 = 0$, inside the dark part $0 - 0 = 0$, and only across the edge $30 - 0 = 30$. It is a **vertical edge detector**, and it works at every row with the same nine weights.

A conv layer does exactly this, except the weights are learned by backpropagation. First layers of trained CNNs end up with kernels much like these edge and colour detectors.

~~~dg-fig 03.01-edges.svg
The same idea on a square with a hole. The vertical-edge kernel fires on the left and right sides, the horizontal-edge kernel (its transpose) on the top and bottom. The sign tells you which way brightness changes.
~~~

## Step 1: one channel, one kernel

Words first: the output at position $(i, j)$ is the dot product between the kernel and the input window whose top-left corner is at $(i, j)$, plus a bias.

$$y_{i,j} = b + \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} w_{u,v}\; x_{i+u,\; j+v}$$

Here $x_{r,c}$ is the input pixel in row $r$, column $c$ (image size $H \times W$), $w_{u,v}$ is the kernel weight in row $u$, column $v$ of the $k \times k$ kernel, and $b$ is one bias shared by all outputs. Without padding the kernel fits in $H - k + 1$ rows and $W - k + 1$ columns: 6×6 with $k = 3$ gives 4×4.

Strictly, mathematical convolution flips the kernel first; the formula above is **cross-correlation**. Libraries compute cross-correlation and call it convolution. Since the kernel is learned, the flip never matters.

<div class="ex">
<div class="lab">Worked example</div>

**a full convolution by hand**

Input (4×4), rows top to bottom: $(1, 2, 0, 1)$, $(3, 1, 2, 0)$, $(0, 2, 1, 3)$, $(1, 0, 2, 2)$. Kernel: the vertical-edge kernel above, bias $b = 0$. The output is $(4 - 3 + 1) \times (4 - 3 + 1) = 2 \times 2$.

$y_{0,0}$: window = rows 0-2, columns 0-2. Left column $1 + 3 + 0 = 4$, right column $0 + 2 + 1 = 3$, so $y_{0,0} = 4 - 3 = 1$.

$y_{0,1}$: rows 0-2, columns 1-3. Left column $2 + 1 + 2 = 5$, right column $1 + 0 + 3 = 4$, so $y_{0,1} = 5 - 4 = 1$.

$y_{1,0}$: rows 1-3, columns 0-2. Left column $3 + 0 + 1 = 4$, right column $2 + 1 + 2 = 5$, so $y_{1,0} = 4 - 5 = -1$.

$y_{1,1}$: rows 1-3, columns 1-3. Left column $1 + 2 + 0 = 3$, right column $0 + 3 + 2 = 5$, so $y_{1,1} = 3 - 5 = -2$.

Output: $\begin{pmatrix} 1 & 1 \\ -1 & -2 \end{pmatrix}$. Positive means "brighter on the left". The NumPy code below prints the same matrix.

</div>

## Step 2: many channels in, many channels out

Real inputs have several **channels**: 3 for a colour image, 64 after a layer with 64 kernels. The rule to memorise: **a kernel always spans every input channel**. For a 3-channel input it is a $3 \times k \times k$ block whose dot product over all $3k^2$ numbers gives one number per position, so one kernel makes one output channel. $C_{out}$ kernels form a weight tensor of shape $(C_{out}, C_{in}, k, k)$. With stride, padding and dilation (Step 3) the full formula is

$$y_{o,i,j} = b_o + \sum_{c=1}^{C_{in}} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} w_{o,c,u,v}\; x_{c,\; s i + d u - p,\; s j + d v - p}$$

where $o$ is the output channel, $c$ the input channel, $b_o$ one bias per output channel, $s$ the stride, $p$ the padding (out-of-range pixels count as 0) and $d$ the dilation.

The weight count is $C_{out} \cdot C_{in} \cdot k^2$ plus $C_{out}$ biases. Check: 64 3×3 kernels on RGB need $64 \times 3 \times 9 = 1{,}728$ weights. A dense layer from 224×224×3 to 64×224×224 would need $150{,}528 \times 3{,}211{,}264 \approx 4.8 \times 10^{11}$.

~~~dg-layers One conv layer as tensor shapes: the kernel depth always equals the input's channel count.
Input image | 3 × 224 × 224
64 kernels | each 3 × 7 × 7, stride 2, padding 3
Feature maps | 64 × 112 × 112
BN + ReLU | same shape, next layer reads all 64 channels
~~~

## Step 3: padding, stride and dilation

Three knobs change which input pixels each output reads and therefore the output size.

- **Padding** $p$ adds a border of zeros. Without it every 3×3 layer shrinks the map by 2 and border pixels feed fewer outputs. $p = (k-1)/2$ at stride 1 keeps the size: **"same" padding**.
- **Stride** $s$ is the step between kernel positions; stride 2 halves the resolution, a learned alternative to pooling (03.02).
- **Dilation** $d$ puts $d - 1$ gaps between taps: a 3×3 kernel with $d = 2$ covers 5×5 with 9 weights, used in segmentation to see wider without losing resolution.

~~~dg-fig 03.01-shapes.svg
The four knobs in 1-D, 7 inputs and a 3-wide kernel. Lines show which inputs output y2 reads. Padding adds zero cells at the ends, stride skips positions, dilation spreads the taps.
~~~

One formula covers every combination. Per spatial dimension:

$$H_{out} = \left\lfloor \frac{H + 2p - d(k-1) - 1}{s} \right\rfloor + 1$$

Here $H$ is the input size, $d(k-1) + 1$ the kernel's dilated width, and the floor drops a final position where the kernel would hang off the padded input.

Checks against the figure (7 inputs, $k = 3$): $\lfloor 4/1 \rfloor + 1 = 5$; padding 1: $\lfloor 6/1 \rfloor + 1 = 7$; plus stride 2: $\lfloor 6/2 \rfloor + 1 = 4$; dilation 2: $\lfloor (7 - 4 - 1)/1 \rfloor + 1 = 3$.

<div class="ex">
<div class="lab">Worked example</div>

**sizing a ResNet stem**

Input 3×224×224, 64 kernels of 7×7, stride 2, padding 3. $H_{out} = \lfloor (224 + 6 - 6 - 1)/2 \rfloor + 1 = \lfloor 111.5 \rfloor + 1 = 112$, so the output is 64×112×112.

Parameters: $64 \times 3 \times 49 = 9{,}408$ weights. Compute: each of the $112 \times 112 = 12{,}544$ positions in each output channel uses every weight once, so $9{,}408 \times 12{,}544 \approx 118$ million multiply-adds per image.

Compare a late ResNet layer, 512→512 channels with 3×3 kernels at 7×7 resolution: $512 \times 512 \times 9 \approx 2.36$ million weights (250 times more) but $2.36\text{M} \times 49 \approx 116$ million multiply-adds, about the same compute.

</div>

| Hyperparameter | Effect on output size | Effect on parameters
| --- | --- | --- |
| Kernel size $k$ | shrinks by $k-1$ without padding | grows as $k^2$
| Padding $p$ | grows by $2p$ | none
| Stride $s$ | divides by about $s$ | none
| Dilation $d$ | shrinks by $d(k-1)$ without padding | none
| Channels $C_{out}$ | sets output depth | grows linearly

## Step 4: translation equivariance

Because the same weights are used at every position, if the cat moves 3 pixels right, its "whisker" responses move 3 pixels right with identical values. This is **translation equivariance**: transform the input and the output transforms the same way.

~~~dg-fig 03.01-equivariance.svg
Left: with stride 1, shifting a 1-D signal by 3 moves its feature map by exactly 3. Right: with stride 2, a 1-pixel shift changes the sampled values themselves (by up to 0.4 on a peak of 0.67 here).
~~~

Two subtleties:

- **Equivariance is not invariance.** A classifier wants **invariance** (the same label wherever the cat is). Convolutions only move evidence around; invariance comes from pooling and global averaging (03.02).
- **Stride breaks it.** It is exact only at stride 1 and away from borders. Stride 2 samples every other position, so a 1-pixel shift samples different points of the response. That is why CNN predictions can flicker when an image moves one pixel (03.02 covers the anti-aliasing fix).

## Step 5: receptive field

The **receptive field** of a unit is the input patch that can affect it. A first-layer 3×3 unit sees 3×3 pixels; a second-layer unit sees 3×3 first-layer units, so 5×5 pixels. Formally, with $r_0 = 1$ and "jump" $j_0 = 1$ (input-pixel distance between neighbouring units):

$$r_l = r_{l-1} + (k_l - 1)\, j_{l-1}, \qquad j_l = j_{l-1}\, s_l$$

where $r_l$ is the field width after layer $l$, $k_l$ and $s_l$ its kernel size and stride, and $j_l$ the product of all strides so far. Three 3×3 stride-1 layers see $1 + 2 + 2 + 2 = 7$ pixels, like one 7×7 kernel but with 27 instead of 49 weights per channel pair and two extra non-linearities: the VGG argument (03.03). Strides multiply the growth of everything after them.

~~~dg-fig 03.01-rf.svg
Left: ten 3×3 layers see 21 pixels at stride 1 but 125 pixels with a stride-2 layer every second layer. Right: measured influence of each input column on the centre output of randomly initialised 5- and 20-layer stacks; shaded bands are the theoretical fields.
~~~

The theoretical field is only an upper bound. The right panel measures the **effective receptive field**: the gradient of the centre output with respect to each input pixel, averaged over 20 random stacks of 3×3 convs with ReLU. At 20 layers theory says 41 pixels wide, yet influence above 5% of the peak covers only 25 columns, and 97% of it lies within ±10 pixels. Far more paths run through the middle of the window than to its edge, so influence is roughly Gaussian. Luo et al. report that training widens it, but it stays well inside the theoretical field. Rule: aim for a theoretical field well above the context size you need.

## Convolution in code

The NumPy function uses **im2col** ("image to columns"): copy each $C_{in} \times k \times k$ patch into a matrix column, reshape the weights to $C_{out} \times C_{in}k^2$, and multiply once. We test it on the hand example and against `nn.Conv2d`.

```python
import numpy as np, torch, torch.nn as nn

def conv2d(x, w, b, stride=1, pad=0):
    # x: (C_in, H, W), w: (C_out, C_in, k, k) -> (C_out, H_out, W_out), via im2col
    C, H, W = x.shape; O, _, k, _ = w.shape
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    cols = np.stack([x[:, i * stride:i * stride + k, j * stride:j * stride + k].ravel()
                     for i in range(Ho) for j in range(Wo)], axis=1)   # (C*k*k, Ho*Wo)
    return (w.reshape(O, -1) @ cols + b[:, None]).reshape(O, Ho, Wo)

# 1. the hand example
X = np.array([[1, 2, 0, 1], [3, 1, 2, 0], [0, 2, 1, 3], [1, 0, 2, 2]], float)
K = np.array([[1, 0, -1]] * 3, float)
print(conv2d(X[None], K[None, None], np.zeros(1))[0])   # [[ 1.  1.] [-1. -2.]]

# 2. a real layer: 3 -> 8 channels, 3x3, stride 2, same padding
torch.manual_seed(0)
layer = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
x = torch.randn(1, 3, 32, 32)                           # (batch, C, H, W)
y = layer(x)
print(y.shape, sum(p.numel() for p in layer.parameters()))  # torch.Size([1, 8, 16, 16]) 224
mine = conv2d(x[0].numpy(), layer.weight.detach().numpy(),
              layer.bias.detach().numpy(), stride=2, pad=1)
print(np.allclose(mine, y[0].detach().numpy(), atol=1e-6))   # True

# 3. equivariance: shift-then-conv equals conv-then-shift (stride 1, away from borders)
c = nn.Conv2d(3, 8, 3, padding=1)
a, b = c(torch.roll(x, 4, dims=3)), torch.roll(c(x), 4, dims=3)
print(torch.allclose(a[..., 5:-5], b[..., 5:-5]))       # True
```

Both checks match the formulas: $8 \times 3 \times 9 + 8 = 224$ parameters and $\lfloor (32 + 2 - 2 - 1)/2 \rfloor + 1 = 16$. PyTorch tensors are (batch, channels, height, width). Im2col copies each pixel up to $k^2$ times, but turns the layer into one large matrix multiply (GEMM), which hardware does very well; cuDNN also has implicit-GEMM, FFT and Winograd algorithms and picks one per layer shape.

## Checking a conv layer

- **Shapes**: compute $H_{out}$ by hand and compare with `y.shape`; an off-by-one usually comes from the floor (odd sizes with stride 2).
- **Parameters**: the total should be $\sum C_{out}(C_{in}k^2 + 1)$, without the $+1$ for `bias=False` layers (standard before batch norm).
- **First-layer kernels** after training should look like smooth oriented edges and colour blobs; pure noise suggests a bad learning rate or too little training.
- **Feature maps** that are all zero after ReLU are dead channels.
- **Custom layers**: run the shift test from the code.

## Where the cost goes

Parameters per layer are $C_{out} C_{in} k^2$, independent of image size. Compute (multiply-adds, or MACs) is that number times $H_{out} W_{out}$, the number of positions the kernel visits. These two scale very differently through a network.

~~~dg-fig 03.01-cost.svg
The 17 main conv layers of ResNet-18 (shortcut 1×1 convs and the classifier left out). Parameters grow fourfold at each stage, but compute stays near 116 million multiply-adds per layer, because resolution halves exactly when channels double.
~~~

CNNs are built in stages: at each boundary resolution halves (4 times fewer positions) and channels double (4 times more weights), so compute per layer stays flat while parameters pile up at the end. In ResNet-18 the last stage holds 75% of the conv weights (about 8.3 of 11.0 million) but only about 23% of the 1.8 billion multiply-adds. Consequences:
To shrink a model's **size**, attack the late, wide layers; to cut **latency** and memory traffic, the early layers matter just as much, since they cost as much compute and hold the largest activations.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Forgetting that kernels span all input channels | weight shape errors such as "expected input to have 3 channels" | weight is $(C_{out}, C_{in}, k, k)$; `in_channels` must equal the previous layer's `out_channels`
| Off-by-one output sizes | shape mismatch at a skip connection or concat | use the formula; odd $k$ with $p = (k-1)/2$; input sizes divisible by the total stride
| Channels-last data fed to a channels-first layer | nonsense outputs or channel-count error | PyTorch expects (N, C, H, W); permute images loaded as (H, W, C)
| No padding in a deep stack | maps shrink by $k-1$ per layer and vanish; border content ignored | "same" padding
| Downsampling too early | fine detail lost, small objects missed | keep stride 1 in the first layers or use a gentler stem
| Too small a receptive field | model misses context (large objects, layout) | more depth, stride, dilation or pooling
| Assuming shift invariance | prediction changes when the image moves 1 pixel | random shift/crop augmentation; anti-aliased pooling (03.02)

## Going deeper

**Convolution is a sparse, tied matrix.** With input and output flattened, a conv layer is $y = Wx$ where $W$ is mostly zeros (locality) and its non-zeros repeat along diagonals (sharing); in 1-D it is a banded Toeplitz matrix. A CNN is an MLP with a strong built-in constraint, which is why it learns from far fewer examples.

**The backward pass is also a convolution.** The input gradient is the upstream gradient convolved with the kernel rotated 180 degrees (a "transposed convolution", reused for upsampling in Module 04). The weight gradient correlates the input with the upstream gradient, summed over positions: each shared weight collects evidence from every place it was used.

**Why equivariance holds.** If $T$ shifts by $\delta$ pixels, $(w \star Tx)_{i} = \sum_u w_u x_{i+u-\delta} = T(w \star x)_i$. With stride $s$ the output would have to move by $\delta/s$, a whole number only when $s$ divides $\delta$: the exact source of the flicker. Pointwise ReLUs preserve equivariance, so a stride-1 stack is equivariant.

**Fast algorithms.** By the convolution theorem, FFT turns convolution into element-wise products, costing $O(HW \log HW)$ per channel pair instead of $O(HWk^2)$; it pays off only for large kernels. For 3×3, Winograd's $F(2\times2, 3\times3)$ cuts multiplications by 2.25 times (published value).

**Special kernels.** A 1×1 conv is a dense layer on each pixel's channel vector, a cheap way to mix channels. Depthwise conv gives each channel its own kernel; followed by 1×1 it is MobileNet's separable block (03.05).

## Key terms

| Term | Meaning
| --- | --- |
| Kernel / filter | Small learned weight block of shape $C_{in} \times k \times k$, slid over the input
| Feature map | The output grid of one kernel; one output channel
| Cross-correlation | Sliding dot product without flipping the kernel; what libraries call convolution
| Padding | Zeros added around the border; "same" padding keeps the size at stride 1
| Stride | Step between kernel positions; stride 2 halves the resolution
| Dilation | Gaps between kernel taps that widen the footprint without adding weights
| Translation equivariance | Shifting the input shifts the output by the same amount
| Receptive field | Input region that can affect a unit; the effective one is much smaller
| im2col | Copy patches into columns so the layer becomes one matrix multiply

<div class="take">
<div class="lab">Key takeaway</div>

a conv layer is a small dense layer applied at every position with shared weights. Know the output-size formula, remember that every kernel spans all input channels, expect equivariance only at stride 1, check receptive fields as they compound with stride, and remember that parameters scale with channels while compute scales with channels times resolution.

</div>
