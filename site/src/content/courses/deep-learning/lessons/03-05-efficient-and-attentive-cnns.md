---
id: "03.05"
course: "deep-learning"
module: "03"
slug: "efficient-and-attentive-cnns"
title: "Efficient and attentive CNNs"
keyIdeas: ["A standard conv costs $H W k^2 C_{in} C_{out}$ multiply-adds. **1×1 convolutions** mix channels at each pixel; Inception uses them to cut channel counts before expensive 3×3 and 5×5 branches, often by around 10× in compute.", "A **depthwise-separable** convolution splits a $k \\times k$ conv into a per-channel spatial filter plus a 1×1 channel mixer, reducing cost by a factor of about $1/C_{out} + 1/k^2$, roughly 8-9× for 3×3.", "MobileNetV2's **inverted residual** expands channels (about 6×), filters depthwise, then projects back with a linear (no ReLU) bottleneck, because ReLU in a low-dimensional space destroys information. The residual joins the narrow ends.", "**Squeeze-and-excitation** pools each channel to one number, passes the vector through a tiny gating MLP with a sigmoid, and rescales the channels: channel attention for about 10% more parameters and under 1% more compute in ResNet-50.", "FLOPs are not latency: depthwise convs do little arithmetic per byte moved, so they run far below peak speed. Always benchmark on the target hardware."]
question: "You replace every 3×3 conv in a ResNet-50 with a depthwise-separable equivalent. FLOPs drop about 8×, but inference on a datacenter GPU gets only about 1.5× faster, and accuracy falls 3 points. Explain both effects, and say what MobileNetV2-style design changes address each."
answer: "Speed: depthwise convs do $k^2$ multiply-adds per loaded input value per channel, very low arithmetic intensity, so on a GPU they are memory-bandwidth-bound and run at a small fraction of peak compute. FLOPs fall 8× but bytes moved fall much less, the 1×1 convs and memory traffic dominate, and the extra layers add kernel launches, so wall-clock improves only modestly. (The same thing is visible even on a CPU: in this lesson's measurement a depthwise layer achieved about 45× fewer multiply-adds per millisecond than the standard conv.) Accuracy: a depthwise-separable conv is a constrained, low-rank factorisation of the full conv, since every output channel can only mix one spatial pattern per input channel, so at the same channel width each block has far fewer parameters and less capacity; ResNet's widths were tuned for full convs. MobileNetV2 addresses capacity by expanding channels about 6× before the depthwise conv (cheap, since depthwise cost is linear in width), and by using a linear bottleneck so the projection back to the narrow width doesn't destroy information with ReLU; the residual then connects the narrow ends, keeping the tensors held between blocks small. Adding squeeze-and-excitation inside the block recovers a little more accuracy for under 1% compute. For speed, the answer is co-design: fuse depthwise and pointwise kernels, choose widths, resolutions and batch sizes by benchmarking on the target hardware, measure the median latency after warm-up, or keep plain dense 3×3 convs in the stages where the GPU is compute-rich, as later GPU-oriented designs do."
sources: [{"title": "Szegedy et al.: Going Deeper with Convolutions (GoogLeNet)", "url": "https://arxiv.org/abs/1409.4842", "note": "Section 4 on the Inception module and 1×1 dimension reductions."}, {"title": "Howard et al.: MobileNets", "url": "https://arxiv.org/abs/1704.04861", "note": "Depthwise-separable cost analysis (Section 3.1) and the width and resolution multipliers."}, {"title": "Hu et al.: Squeeze-and-Excitation Networks", "url": "https://arxiv.org/abs/1709.01507", "note": "The SE block, where to insert it, and its parameter and compute overhead."}, {"title": "Tan & Le: EfficientNet", "url": "https://arxiv.org/abs/1905.11946", "note": "Compound scaling of depth, width and resolution built on MBConv blocks with SE."}]
practice: ["depthwise-separable-convolution", "inception-module", "squeeze-and-excitation-block", "conv2d"]
tags: ["depthwise separable", "Inception", "squeeze-and-excitation"]
tools: ["PyTorch"]
---

## What you will learn

- How to count the cost of a convolution (multiply-adds and parameters) by hand.
- Why a 1×1 convolution is a cheap channel mixer, and how Inception uses it.
- How a depthwise-separable convolution splits one layer into two cheap ones, saving about 8-9× for 3×3 kernels.
- How MobileNetV2's inverted residual block is built, and why its last layer has no ReLU.
- How squeeze-and-excitation lets the network turn channels up or down for each image, for a tiny extra cost.
- Why fewer FLOPs does not automatically mean faster.

*Before you start:* you need the convolution arithmetic from 03.01 (kernels, channels, output size), pooling and global average pooling from 03.02, and the residual block from 03.04. A little PyTorch (`nn.Conv2d`, `nn.Sequential`) helps for the code.

## From accuracy to accuracy per FLOP

Lesson 03.04 ended with ResNets that could be made almost arbitrarily deep. Once depth stopped being the obstacle, the question changed from "can it train?" to "what does each multiply-add buy?". Phones, cameras, cars and serving budgets care about latency, memory and energy, so a model that matches ResNet-50 at a tenth of the cost is often worth more than one that beats it by half a point.

This lesson covers three ideas: cheap channel mixing with 1×1 convolutions (Inception), factorised convolutions (depthwise-separable, MobileNet) and channel attention (squeeze-and-excitation). EfficientNet combines all three; module 04's Vision Transformers push attention much further.

## The idea in one picture

Take one ordinary layer from the middle of a CNN: the input is a 56×56 feature map with 128 channels, and a 3×3 convolution produces 128 output channels. How much work is that, and can we get almost the same result for less?

| Version of the layer | Multiply-adds | Weights
| --- | --- | --- |
| Standard 3×3 conv | 462.4 million | 147,456
| Depthwise 3×3 (filter each channel on its own) | 3.6 million | 1,152
| Pointwise 1×1 (mix channels at each pixel) | 51.4 million | 16,384
| Depthwise + pointwise together | 55.0 million | 17,536

~~~dg-fig 03.05-cost.svg
Left: the same 3×3 layer done as a standard conv (462M multiply-adds) and as depthwise-separable (55M); nearly all of the remaining cost is the 1×1 part. Right: Inception's 5×5 branch shrinks from 120.4M to 12.4M when a 1×1 conv first cuts the channels from 192 to 16.
~~~

The standard layer does two jobs at once, looking at a 3×3 neighbourhood *and* combining all 128 channels, and pays for every combination. Doing them one after the other costs 8.4× less; the thin sliver on top is the spatial filtering, the rest is channel mixing.

Three words to fix now:

- A **multiply-add (MAC)** is one multiplication followed by one addition, the basic unit of work in a conv or a dense layer. One MAC is usually counted as 2 FLOPs.
- A **depthwise convolution** applies one small filter to each channel separately, never mixing channels.
- A **pointwise convolution** is a 1×1 convolution: at each pixel, it takes a weighted sum across channels, which is exactly a small dense layer applied at every position.

## Step 1: count what a convolution costs

Each output value of a standard convolution looks at a $k \times k$ window in each of the $C_{in}$ input channels, so it needs $k^2 C_{in}$ multiply-adds. There are $H \times W$ output positions and $C_{out}$ output channels, so the total is

$$\text{MACs} = H\,W\,k^2\,C_{in}\,C_{out}, \qquad \text{weights} = k^2\,C_{in}\,C_{out}.$$

Here $H \times W$ is the output size, $k$ the kernel width, $C_{in}$ and $C_{out}$ the channel counts. The weight count has no $H W$ because the same kernel slides over every position (the weight sharing from 03.01).

Number check for the layer above: $56 \times 56 = 3{,}136$ positions; each output value costs $9 \times 128 = 1{,}152$ MACs; times 128 output channels gives $3{,}136 \times 1{,}152 \times 128 = 462{,}422{,}016 \approx 462.4$M. The weights are $9 \times 128 \times 128 = 147{,}456$.

Notice the product $C_{in} C_{out}$: doubling the width multiplies every conv's cost by four. Every idea in this lesson attacks that term.

## Step 2: 1×1 convolutions and the Inception module

A 1×1 convolution looks at a single pixel. It cannot see shapes, but it can recombine channels: output channel $j$ at a pixel is a weighted sum of all input channels at that pixel. So it can change the channel count cheaply (192 down to 16, say).

GoogLeNet (Szegedy et al., 2014) won the ILSVRC-2014 classification task with the **Inception module**. Instead of choosing one kernel size per layer, it runs four branches in parallel (1×1, 3×3, 5×5 and a 3×3 max-pool) and concatenates their outputs along the channel axis. Each layer can then mix small and large receptive fields. Run naively, the 5×5 branch on 192 channels would be enormous, so every expensive branch starts with a 1×1 reduction.

~~~dg-flow An Inception module: four parallel branches, each 1×1-reduced where needed, concatenated along channels.
[Branch 1] Input 192ch -(1×1, 64)-> *Concat
[Branch 2] Input 192ch -(1×1, 96)-> Reduced -(3×3, 128)-> *Concat
[Branch 3] Input 192ch -(1×1, 16)-> Reduced -(5×5, 32)-> *Concat
[Branch 4] Input 192ch -(3×3 max pool)-> Pooled -(1×1, 32)-> *Concat
~~~

The concatenated output here has $64 + 128 + 32 + 32 = 256$ channels. Thanks to these reductions GoogLeNet was more accurate than VGG while using about 12× fewer parameters than AlexNet.

<div class="ex">
<div class="lab">Worked example</div>

**the Inception 5×5 branch, counted**

Input: 28×28×192. Output: 32 channels from a 5×5 conv. Positions: $28 \times 28 = 784$.

Direct: $784 \times 25 \times 192 \times 32 = 784 \times 153{,}600 = 120{,}422{,}400$, about 120.4M MACs.

With a 1×1 reduction to 16 channels first. The 1×1 costs $784 \times 1 \times 192 \times 16 = 2{,}408{,}448$. The 5×5 now sees only 16 channels: $784 \times 25 \times 16 \times 32 = 10{,}035{,}200$.

Total: $2{,}408{,}448 + 10{,}035{,}200 = 12{,}443{,}648$, about 12.4M MACs. The ratio is $120.4 / 12.4 \approx 9.7$, so roughly 10× less work.

The price: the 5×5 filters see a learned 16-channel summary instead of all 192 channels, which works because channels are highly redundant.

</div>

## Step 3: depthwise-separable convolutions

A standard conv does spatial filtering and channel mixing in one go. A **depthwise-separable convolution** factorises it into two layers:

1. **Depthwise**: one $k \times k$ filter per input channel, with no mixing across channels. In PyTorch this is a grouped conv with `groups = C_in`. Cost per position: $k^2 C_{in}$.
2. **Pointwise**: a 1×1 conv that mixes channels. Cost per position: $C_{in} C_{out}$.

Divide the new cost by the old one (the $H W$ cancels):

$$\frac{k^2 C_{in} + C_{in} C_{out}}{k^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{k^2}.$$

The first term is the depthwise part, the second the pointwise part. With hundreds of channels the first term is tiny, so the ratio settles just above $1/k^2$: about $1/9$ for 3×3 kernels.

~~~dg-fig 03.05-ratio.svg
The cost of a separable conv relative to a standard one. It falls as channels grow and levels off at 1/k² (dotted lines). At 256 channels with a 3×3 kernel the separable version costs 0.115 of the original.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**weights saved at 256 channels**

A 3×3 conv with 256 input and 256 output channels. Standard weights: $9 \times 256 \times 256 = 9 \times 65{,}536 = 589{,}824$.

Separable: depthwise $9 \times 256 = 2{,}304$ plus pointwise $256 \times 256 = 65{,}536$, total $67{,}840$.

Ratio: $67{,}840 / 589{,}824 = 0.115$, so about 8.7× fewer weights. The formula agrees: $1/256 + 1/9 = 0.0039 + 0.1111 = 0.115$.

Multiply-adds shrink by exactly the same factor, because every weight is used once per output position.

</div>

MobileNet (Howard et al., 2017) built a whole network from these blocks. As published: about 4.2M parameters and 569M multiply-adds for 70.6% ImageNet top-1, against VGG-16's 138M parameters and about 15.3 billion multiply-adds for 71.5%. Xception (Chollet, 2017) arrived at a similar design from the Inception side, viewing a depthwise-separable conv as an Inception module with one branch per channel.

The cost is capacity. A separable conv can only represent kernels whose spatial part and channel part factor apart, so at the same width each block is less expressive. The next step shows how MobileNetV2 buys that capacity back cheaply.

## Step 4: inverted residuals and linear bottlenecks

ResNet's bottleneck block (03.04) goes wide → narrow → wide: 1×1 down, 3×3, 1×1 up. MobileNetV2 (Sandler et al., 2018) inverts this to narrow → wide → narrow:

1. A 1×1 conv **expands** the channels by an expansion factor $t$, typically 6.
2. A 3×3 **depthwise** conv filters in the wide space. That is cheap, because depthwise cost grows only linearly with width.
3. A 1×1 conv **projects** back down to the narrow width, with **no activation** afterwards.
4. The residual connection joins the two narrow ends.

~~~dg-flow MobileNetV2 inverted residual: expand, filter depthwise, project linearly, add the shortcut between the narrow ends.
Input 32ch -(1×1 expand, ReLU6)-> Wide 192ch -(3×3 depthwise, ReLU6)-> Filtered 192ch -(1×1 project, no ReLU)-> Narrow 32ch -(add input)-> *Output 32ch
~~~

With the shortcut on the narrow ends, the tensors kept between blocks are the small ones; the wide tensor lives only inside the block.

Why no ReLU at the end? The paper's argument is that ReLU applied to a low-dimensional signal throws information away: any coordinate that goes negative is set to zero, and that loss cannot be undone. In a wide space the same information is spread over many coordinates, and enough of them survive. You can see this directly.

~~~dg-fig 03.05-relu.svg
A 2-D spiral is mapped into n dimensions by a random matrix, passed through ReLU, then mapped back as well as possible. At n = 2 or 3 large parts of the spiral collapse (relative error 0.57 and 0.44). At n = 15 or 30 it comes back exactly. So use ReLU in the wide layer and keep the narrow output linear.
~~~

Each panel's error is the median over 20 random matrices; the inverse uses only the coordinates ReLU kept, so once two of them survive for every point, the input is recovered exactly.

Number check for the block in the diagram, on a 28×28 map: expand $784 \times 32 \times 192 = 4.82$M MACs, depthwise $784 \times 9 \times 192 = 1.35$M, project $4.82$M, total about 11.0M. A standard 3×3 conv working at the full 192 width would cost $784 \times 9 \times 192 \times 192 \approx 260$M. As published, MobileNetV2 reaches 72.0% top-1 with about 3.4M parameters and 300M multiply-adds.

## Step 5: squeeze-and-excitation, channel attention

A convolution applies the same weights to every image, so every channel passes on with the same importance. Hu et al.'s **squeeze-and-excitation (SE)** block lets the network reweight channels per input:

1. **Squeeze**: global-average-pool each of the $C$ channels to one number. The result is a $C$-vector $z$ summarising the whole image.
2. **Excite**: a two-layer MLP $C \to C/r \to C$ with a ReLU in between and a sigmoid at the end gives a **gate** $s_c \in (0, 1)$ per channel. The reduction ratio $r$ is usually 16.
3. **Scale**: multiply every value in channel $c$ by $s_c$.

In symbols, $s = \sigma(W_2\,\text{ReLU}(W_1 z))$ and $\tilde x_c = s_c\, x_c$. Here $W_1$ is $(C/r) \times C$ and $W_2$ is $C \times (C/r)$, so the extra weights are about $2C^2/r$. For $C = 256$ and $r = 16$ that is $2 \times 65{,}536 / 16 = 8{,}192$, compared with 589,824 for one 3×3 conv at that width. The compute is negligible because the MLP runs once per image, not once per pixel.

This is a form of **attention**: global context decides which features matter for this input. We trained a small CNN with an SE block on scikit-learn's 8×8 digits and recorded the test-set gates.

~~~dg-fig 03.05-segates.svg
Average SE gate for each digit class (rows) in the 10 channels whose gates vary most. Four channels are turned up or down depending on the digit. Most others stay near fully open, and channel 4 is switched off for every input.
~~~

Two honest observations. Of the 32 channels, 27 had average gates above 0.9 and only four varied much by class: SE learned a few selective switches. And on this tiny task SE did not help: over three seeds the plain net reached 94.2% test accuracy and the SE net 92.0%. SE's published gains come from large networks on large datasets. Inserted into every ResNet-50 block (on the residual branch, before the addition), SE adds about 2.5M parameters (around 10%) and under 1% compute, and the SE-based entry won ILSVRC-2017.

EfficientNet (Tan and Le, 2019) combined MobileNetV2 blocks, SE and a **compound scaling** rule that grows depth, width and input resolution together by fixed ratios rather than one at a time. EfficientNet-B7 reached 84.3% top-1 while being 8.4× smaller than the previous best model.

## Efficient blocks in code

```python
import torch, torch.nn as nn

def n_params(m): return sum(p.numel() for p in m.parameters())

C = 256
standard = nn.Conv2d(C, C, 3, padding=1, bias=False)
separable = nn.Sequential(
    nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),   # depthwise: one 3x3 filter per channel
    nn.Conv2d(C, C, 1, bias=False))                        # pointwise: 1x1 mixes channels
print(n_params(standard), n_params(separable))             # 589824 67840

class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(c, c // r), nn.ReLU(inplace=True),
                                  nn.Linear(c // r, c), nn.Sigmoid())
    def forward(self, x):                                  # x: (N, C, H, W)
        return x * self.gate(x)[:, :, None, None]

class InvertedResidual(nn.Module):
    # MobileNetV2 block (stride 1, same width): expand -> depthwise -> linear project
    def __init__(self, c, t=6, se=False):
        super().__init__()
        h = c * t
        self.block = nn.Sequential(
            nn.Conv2d(c, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU6(inplace=True),
            nn.Conv2d(h, h, 3, padding=1, groups=h, bias=False), nn.BatchNorm2d(h), nn.ReLU6(inplace=True),
            SE(h, r=24) if se else nn.Identity(),
            nn.Conv2d(h, c, 1, bias=False), nn.BatchNorm2d(c))   # no activation: linear bottleneck
    def forward(self, x):
        return x + self.block(x)                           # shortcut joins the narrow ends

x = torch.randn(2, 32, 28, 28)
blk = InvertedResidual(32)
print(blk(x).shape, n_params(blk))                         # torch.Size([2, 32, 28, 28]) 14848
print(n_params(InvertedResidual(32, se=True)))             # 18120
```

The key detail is `groups=C`: PyTorch splits the channels into $C$ groups of one, so each filter sees only its own channel. `ReLU6` is ReLU capped at 6, which suits low-precision arithmetic. The optional SE works in the expanded space, as in MobileNetV3 and EfficientNet.

## Checking the result: FLOPs are not latency

MAC counts compare designs on paper, but what you ship is wall-clock time on the target device. We timed the three layers from the opening picture in PyTorch on this course's CPU (2 threads, batch of 8, median of 25 runs).

~~~dg-fig 03.05-speed.svg
Left: the depthwise conv does about 128× less arithmetic than the standard conv. Right: measured on a 2-thread CPU, it also performs that arithmetic about 45× less efficiently, so standard 149.8 ms became 53.3 + 64.6 = 117.9 ms, only 1.3× faster for 8.4× fewer MACs.
~~~

Why? Each depthwise output needs only 9 MACs, against $9 \times 128 = 1{,}152$ for the standard conv, yet both read and write about the same amount of data. Little arithmetic per byte moved (low **arithmetic intensity**) means the layer waits on memory, not on the multipliers; GPUs, with huge compute and comparatively limited bandwidth, show this even more strongly. Timings on a busy machine are also noisy: an earlier run under heavier load had the separable pair *slower* than the standard conv. Measure several times, on the deployment hardware.

A practical checklist:

- Count parameters and MACs (the formulas above, or a counter such as `fvcore`) to compare designs.
- Time the real model at the real batch size after warm-up runs, and report the median; check peak memory too.
- Compare accuracy at matched cost: "same FLOPs, +1%" is a fair claim, "+1%" alone is not.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| FLOPs fell 8× but the model is barely faster | Depthwise convs are memory-bound; many small layers mean many kernel launches | Benchmark on target; use fused kernels; keep dense convs where hardware is compute-rich
| Accuracy dropped after swapping in separable convs | Factorised conv has less capacity at the same width | Expand channels inside the block (inverted residual, $t \approx 6$)
| Training of a narrow network stalls or loses accuracy | ReLU after the narrow projection destroys information | Remove the activation after the projection (linear bottleneck)
| SE adds parameters and no accuracy | Task too small, or gates saturate near 1 | Expect gains on larger models and data; check the gate histogram
| Shape error in a depthwise conv | `groups` does not divide the channel counts | Use `groups=C_in` and `C_out` a multiple of `C_in`
| Out of memory despite few parameters | Expanded 6× tensors are stored for backprop | Smaller batch or resolution, activation checkpointing

## Going deeper

**The low-rank view.** A standard conv kernel is a 4-D tensor of shape $C_{out} \times C_{in} \times k \times k$. A depthwise-separable conv restricts it to $W[o, i, u, v] = P[o, i]\, D[i, u, v]$: for each input channel there is one spatial pattern, and the output channels can only take weighted mixtures of those patterns. Inception's 1×1 reductions are a different low-rank constraint: the big kernel reads from a 16-dimensional subspace of the 192 channels.

**Grouped convolutions sit in between.** With `groups = g`, channels split into $g$ blocks convolved separately, cutting cost by $g$. Depthwise is the extreme $g = C_{in}$; ResNeXt and ShuffleNet use moderate $g$, ShuffleNet shuffling channels so information still crosses groups.

**Compound scaling.** EfficientNet scales depth by $\alpha^\phi$, width by $\beta^\phi$ and resolution by $\gamma^\phi$. Because conv cost grows linearly with depth but with the square of width and of resolution, total FLOPs grow as $(\alpha \beta^2 \gamma^2)^\phi$. The paper constrains $\alpha \beta^2 \gamma^2 \approx 2$, so each step of $\phi$ roughly doubles the cost; its grid search found $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$ (published values). The dimensions are complementary: higher resolution needs more depth (receptive field) and more width (finer patterns).

**SE as a precursor of attention.** SE multiplies features by a data-dependent weight computed from global context, the pattern of self-attention with one weight per channel instead of one per pair of positions.

## Key terms

| Term | Meaning
| --- | --- |
| MAC (multiply-add) | One multiply plus one add; conv cost is $H W k^2 C_{in} C_{out}$ MACs
| 1×1 (pointwise) convolution | A per-pixel weighted sum across channels; mixes or resizes channels cheaply
| Inception module | Parallel 1×1, 3×3, 5×5 and pooling branches, concatenated, with 1×1 reductions
| Depthwise convolution | One spatial filter per channel, no channel mixing (`groups = C_in`)
| Depthwise-separable convolution | Depthwise conv followed by a pointwise conv; cost ratio $1/C_{out} + 1/k^2$
| Inverted residual | Narrow → expand → depthwise → narrow block with the shortcut on the narrow ends
| Linear bottleneck | The final projection of a block has no activation, so no information is clipped
| Squeeze-and-excitation | Pool to a channel vector, MLP plus sigmoid gives per-channel gates, rescale the channels
| Arithmetic intensity | Arithmetic done per byte moved; low intensity means memory-bound and slower than FLOPs suggest

<div class="take">
<div class="lab">Key takeaway</div>

efficient CNNs separate what a standard conv does jointly, using 1×1 convs to mix channels cheaply and depthwise convs to filter space cheaply, expand channels inside the block to recover capacity, and add channel attention like squeeze-and-excitation for a small cost; always check that FLOP savings turn into real latency on your hardware.

</div>
