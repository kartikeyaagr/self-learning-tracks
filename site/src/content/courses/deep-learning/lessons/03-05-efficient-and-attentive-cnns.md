---
id: "03.05"
course: "deep-learning"
module: "03"
slug: "efficient-and-attentive-cnns"
title: "Efficient and attentive CNNs"
keyIdeas: ["1×1 convolutions mix channels at each pixel; Inception uses them to cut channel counts before expensive 3×3 and 5×5 branches, often by around 10× in compute.", "A depthwise-separable convolution splits a $k \\times k$ conv into a per-channel spatial filter plus a 1×1 channel mixer, reducing cost by a factor of about $1/C_{out} + 1/k^2$, roughly 8-9× for 3×3.", "MobileNetV2's inverted residual expands channels, filters depthwise, then projects back with a linear (no ReLU) bottleneck, with the residual between the narrow ends.", "Squeeze-and-excitation pools each channel to one number, passes the vector through a tiny gating MLP with a sigmoid, and rescales the channels: channel attention for about 10% more parameters and under 1% more compute."]
question: "You replace every 3×3 conv in a ResNet-50 with a depthwise-separable equivalent. FLOPs drop about 8×, but inference on a datacenter GPU gets only about 1.5× faster, and accuracy falls 3 points. Explain both effects, and say what MobileNetV2-style design changes address each."
answer: "Speed: depthwise convs do $k^2$ multiply-adds per loaded input value per channel, very low arithmetic intensity, so on a GPU they are memory-bandwidth-bound and run at a small fraction of peak compute. FLOPs fall 8× but bytes moved fall much less, the 1×1 convs and memory traffic dominate, and the extra layers add kernel launches, so wall-clock improves only modestly. Accuracy: a depthwise-separable conv is a constrained, low-rank factorisation of the full conv, so at the same channel width each block has far fewer parameters and less capacity; ResNet's widths were tuned for full convs. MobileNetV2 addresses capacity by expanding channels about 6× before the depthwise conv (cheap, since depthwise cost is linear in width), and by using a linear bottleneck so the projection doesn't destroy information with ReLU. For speed, the answer is co-design: fuse depthwise and pointwise kernels, choose widths and resolutions benchmarked on the target hardware, or use plain dense convs where the GPU is compute-rich."
sources: [{"title": "Szegedy et al.: Going Deeper with Convolutions (GoogLeNet)", "url": "https://arxiv.org/abs/1409.4842", "note": "Section 4 on the Inception module and 1×1 dimension reductions."}, {"title": "Howard et al.: MobileNets", "url": "https://arxiv.org/abs/1704.04861", "note": "Depthwise-separable cost analysis (Section 3.1) and the width and resolution multipliers."}, {"title": "Hu et al.: Squeeze-and-Excitation Networks", "url": "https://arxiv.org/abs/1709.01507", "note": "The SE block, where to insert it, and its parameter and compute overhead."}, {"title": "Tan & Le: EfficientNet", "url": "https://arxiv.org/abs/1905.11946", "note": "Compound scaling of depth, width and resolution built on MBConv blocks with SE."}]
practice: ["depthwise-separable-convolution", "inception-module", "squeeze-and-excitation-block", "conv2d"]
tags: ["depthwise separable", "Inception", "squeeze-and-excitation"]
tools: ["PyTorch"]
---

## From accuracy to accuracy per FLOP

After ResNet (03.04), the question shifted from "can it train?" to "what does each multiply-add buy?". Phones, cameras and serving budgets care about latency and energy, so a model that matches ResNet-50 at a tenth of the cost is worth more than one that beats it by half a point. This lesson covers the three ideas that dominate efficient CNN design: cheap channel mixing with 1×1 convs (Inception), factorised convolutions (depthwise-separable, MobileNet), and channel attention (squeeze-and-excitation). Module 04 continues with Vision Transformers, which push the attention idea further.

## Inception and 1×1 reductions

GoogLeNet (Szegedy et al., 2014) won ILSVRC-2014 with an Inception module: parallel 1×1, 3×3 and 5×5 conv branches plus a 3×3 max-pool branch, concatenated along channels, so each layer can choose its own mix of receptive-field sizes. Run naively, the 5×5 branch is enormous. The trick is a **1×1 convolution** before it: a per-pixel dense layer across channels that cuts depth from, say, 192 to 16, so the 5×5 conv runs on 16 channels. GoogLeNet reached higher accuracy than VGG with about 12× fewer parameters than AlexNet.

~~~dg-flow An Inception module: four parallel branches, each 1×1-reduced where needed, concatenated along channels.
[Branch 1] Input 192ch -(1×1, 64)-> *Concat
[Branch 2] Input 192ch -(1×1, 96)-> Reduced -(3×3, 128)-> *Concat
[Branch 3] Input 192ch -(1×1, 16)-> Reduced -(5×5, 32)-> *Concat
[Branch 4] Input 192ch -(3×3 max pool)-> Pooled -(1×1, 32)-> *Concat
~~~

## Depthwise-separable convolutions

A standard conv does spatial filtering and channel mixing at once: cost $k^2 C_{in} C_{out}$ per output pixel. A **depthwise-separable** conv factorises it:

1. **Depthwise**: one $k \times k$ filter per input channel, no mixing across channels (a grouped conv with groups $= C_{in}$). Cost $k^2 C_{in}$.
2. **Pointwise**: a 1×1 conv mixing channels. Cost $C_{in} C_{out}$.

$$\frac{k^2 C_{in} + C_{in} C_{out}}{k^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{k^2}$$

For 3×3 kernels and hundreds of channels this is about $1/9$. MobileNet (2017) built a whole network from these blocks: about 4.2M parameters and 569M multiply-adds for 70.6% ImageNet top-1, against VGG-16's 138M parameters and about 15.3 billion for 71.5%. Xception reached a similar design from the Inception side, viewing depthwise-separable conv as an Inception module taken to the extreme.

**MobileNetV2's inverted residual** improves on it. ResNet bottlenecks go wide → narrow → wide; V2 goes narrow → wide → narrow: a 1×1 expands channels by about 6×, a 3×3 depthwise filters in the wide space (cheap, since depthwise cost is linear in channels), and a 1×1 projects back down with **no ReLU**, because ReLU in a low-dimensional space destroys information. The residual connects the narrow ends, keeping the memory-heavy tensors small.

The caveat: FLOPs are not latency. Depthwise convs do very little arithmetic per byte loaded, so on GPUs they are memory-bound and run far below peak throughput; a MobileNet can be slower than its FLOP count suggests. Always benchmark on the target hardware.

## Squeeze-and-excitation: channel attention

Convolutions treat every channel equally regardless of the image. Hu et al.'s SE block lets the network reweight channels per input:

1. **Squeeze**: global average pool each of the $C$ channels to one number, giving a $C$-vector summarising the whole image.
2. **Excite**: a two-layer MLP $C \to C/r \to C$ (with $r = 16$), ReLU in between, sigmoid at the end, giving a gate in $(0, 1)$ per channel.
3. **Scale**: multiply each channel of the feature map by its gate.

```python
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(c, c // r), nn.ReLU(inplace=True),
                                  nn.Linear(c // r, c), nn.Sigmoid())
    def forward(self, x):                       # x: (N, C, H, W)
        return x * self.gate(x)[:, :, None, None]
```

Inserted into every ResNet-50 block (on the branch, before the addition), SE adds about 2.5M parameters (10%) and under 1% compute, and the SE-based entry won ILSVRC-2017. It is a cheap form of attention: global context decides which features matter. EfficientNet (2019) combined MobileNetV2 blocks, SE and a **compound scaling** rule that grows depth, width and resolution together by fixed ratios; EfficientNet-B7 reached 84.3% top-1 while being 8.4× smaller than the previous best.

| Idea | What it factorises or adds | Typical saving or cost |
| --- | --- | --- |
| 1×1 reduction (Inception) | cuts channels before big kernels | about 10× fewer MACs in a branch |
| Depthwise-separable | spatial filter and channel mix separately | about 8-9× fewer MACs for 3×3 |
| Inverted residual | expand, depthwise, linear project | small memory footprint between blocks |
| Squeeze-and-excitation | per-image channel gates | +10% params, under 1% compute |

<div class="ex">
<div class="lab">Worked example</div>

**the savings, counted**

Inception's 5×5 branch on a 28×28×192 input, producing 32 channels. Direct: $28^2 \times 32 \times 25 \times 192 \approx 120.4$M MACs. With a 1×1 reduction to 16 channels first: $28^2 \times 16 \times 192 + 28^2 \times 32 \times 25 \times 16 \approx 2.4 + 10.0 = 12.4$M, about 10× less.

Depthwise-separable 3×3 conv with 256 in and out channels: standard $9 \times 256^2 = 589{,}824$ weights; separable $9 \times 256 + 256^2 = 2{,}304 + 65{,}536 = 67{,}840$, about 8.7× fewer, matching $1/256 + 1/9 \approx 0.115$.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

efficient CNNs separate what a standard conv does jointly, using 1×1 convs to mix channels cheaply and depthwise convs to filter space cheaply, and add channel attention like squeeze-and-excitation for a small cost; always check that FLOP savings turn into real latency on your hardware.

</div>
