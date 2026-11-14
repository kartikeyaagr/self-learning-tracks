---
id: "03.04"
course: "deep-learning"
module: "03"
slug: "residual-networks"
title: "Residual networks"
keyIdeas: ["Plain networks deeper than about 20 layers trained **worse**, with higher training error, not just test error: a degradation problem in optimisation, not overfitting.", "A residual block computes $y = x + F(x)$, so each block only has to learn a correction to the identity, and doing nothing is the easy default.", "The identity path gives backprop a direct route: $\\partial L/\\partial x_l = \\partial L/\\partial x_L\\,(1 + \\partial(\\ldots)/\\partial x_l)$, so the gradient never has to pass through every weight layer.", "Bottleneck blocks (1×1 reduce, 3×3, 1×1 expand) make 50-152 layer nets affordable; a 1×1 projection shortcut handles changes in shape."]
question: "Explain why the degradation problem cannot be overfitting, then use the residual gradient formula to argue why a 100-block ResNet still gets useful gradients at its first block even if the branch Jacobians are all small. What happens to that argument if you add a ReLU after every addition (as the original ResNet does)?"
answer: "Overfitting means low training error but high test error. In the degradation experiments the deeper plain net had higher training error than the shallower one, even though it could represent the shallower solution by setting extra layers to identity, so the failure is in optimisation. In a ResNet, $\\partial L/\\partial x_1 = \\partial L/\\partial x_{100}\\,(1 + \\partial(\\sum F_i)/\\partial x_1)$. The leading 1 passes the top gradient to the first block unchanged, so small branch Jacobians reduce the correction terms without shrinking the base signal; in a plain network the gradient is a product of 100 Jacobians and vanishes if each has norm below one. With a ReLU after each addition the shortcut is no longer a pure identity: each step multiplies the gradient by a 0/1 mask, so units that are off block their component of the shortcut. In practice most units are active and the path is still short, so it trains well to about 150 layers, but at around 1000 layers this degrades, which is why pre-activation ResNets move all nonlinearities into the branch and keep the shortcut clean."
sources: [{"title": "He et al.: Deep Residual Learning for Image Recognition", "url": "https://arxiv.org/abs/1512.03385", "note": "Figure 1 (degradation), Section 3 (residual blocks, projection shortcuts), Table 1 (ResNet-18 to 152)."}, {"title": "He et al.: Identity Mappings in Deep Residual Networks", "url": "https://arxiv.org/abs/1603.05027", "note": "The gradient decomposition and the pre-activation block that trains 1001-layer nets."}, {"title": "Li et al.: Visualizing the Loss Landscape of Neural Nets", "url": "https://arxiv.org/abs/1712.09913", "note": "Loss-surface plots with and without skip connections."}, {"title": "Dive into Deep Learning: Residual Networks", "url": "https://d2l.ai/chapter_convolutional-modern/resnet.html", "note": "Residual blocks and ResNet-18 in code, with the function-class argument."}]
practice: ["resnet-residual-block", "residual-block", "apply-alphago-residual-block", "time-conditioned-residual-block"]
tags: ["ResNet", "skip connections", "bottleneck"]
tools: ["PyTorch"]
---

## The degradation problem

By 2015, VGG (03.03), He initialisation (01.05) and BatchNorm (02.01) let 20-30 layer plain networks train. Going deeper should not hurt: a 56-layer net could copy a 20-layer net and make the extra 36 layers identity maps. In practice He et al. found the 56-layer plain network had **higher training error** than the 20-layer one on CIFAR-10, and plain-34 was worse than plain-18 on ImageNet. Since training error rose, this is not overfitting. With BN, gradients weren't vanishing either. The solver simply could not find the good solutions that exist; learning an identity mapping through a stack of conv-BN-ReLU layers is surprisingly hard.

## The residual block

The fix is to make identity the default. A residual block computes

$$y = x + F(x; W)$$

where $F$ is two or three conv layers. To act as identity the block only needs $F = 0$, which weight decay and small init make easy. Each block learns a **residual**: a correction to its input. Stacking them, the output of block $L$ is the input of block $l$ plus a sum of corrections:

$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i)$$

and the gradient is

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L}\left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F(x_i)\right)$$

The "1" means the gradient at the top reaches every earlier block directly, without being multiplied by any weight matrices. It cannot vanish just because some branches have small Jacobians. This is the same reasoning as the Pre-LN Transformer (02.02), which uses exactly this structure.

~~~dg-flow A basic residual block: the shortcut carries x unchanged and the branch learns a correction that is added before the final ReLU.
x -(conv3×3, BN, ReLU)-> h -(conv3×3, BN)-> F(x) -(+ x)-> *x + F(x) -(ReLU)-> y
~~~

## Block designs

- **Basic block** (ResNet-18/34): two 3×3 convs with BN and ReLU; the ReLU comes after the addition.
- **Bottleneck block** (ResNet-50/101/152): 1×1 conv reduces channels by 4×, a 3×3 conv works at the reduced width, a 1×1 conv expands back. The expensive 3×3 runs on a quarter of the channels.
- **Projection shortcut**: when a block changes resolution (stride 2) or channel count, $x$ no longer matches $F(x)$, so the shortcut becomes a 1×1 conv with stride 2 and BN. Elsewhere the shortcut is a free identity.
- **Pre-activation** (ResNet v2): BN and ReLU move to the start of the branch, $y = x + F(\mathrm{ReLU}(\mathrm{BN}(x)))$, so the shortcut path is a pure identity with nothing after the addition. This trains 1000-layer nets and is the form diffusion U-Nets and Transformers follow.

```python
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, c_in, c_mid, stride=1):
        super().__init__()
        c_out = 4 * c_mid
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_mid, 3, stride, 1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        nn.init.zeros_(self.f[-1].weight)          # block starts as identity
        self.short = nn.Identity() if stride == 1 and c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.f(x) + self.short(x))
```

Zero-initialising the last BN scale $\gamma$ in each branch, as above, makes every block an exact identity at step 0, so a 100-block network initially behaves like a shallow one; Goyal et al. used it for large-batch training.

## Why ResNets train

Several complementary explanations hold up:

- **Gradient highway**: the identity term above.
- **Easier targets**: learning small corrections near zero is better-conditioned than learning a full mapping.
- **Smoother loss surfaces**: Li et al.'s visualisations show deep plain nets have chaotic, non-convex landscapes, while skip connections keep them smooth and nearly convex near the minimum.
- **Implicit ensemble**: a stack of $n$ blocks contains $2^n$ paths of different lengths, and most gradient flows through relatively short ones, so a deep ResNet behaves partly like an ensemble of shallower networks.

The results: ResNet-152 had lower complexity than VGG-16 (11.3 versus 15.3 billion FLOPs as reported) and an ensemble won ILSVRC-2015 at 3.57% top-5 error. ResNet-50 (25.6M parameters, about 3.8 billion multiply-adds) remains the standard vision baseline, and the residual connection is now in almost every deep architecture.

<div class="ex">
<div class="lab">Worked example</div>

**why bottlenecks are cheaper**

At 256 channels, a basic block with two 3×3 convs needs $2 \times 9 \times 256^2 = 1{,}179{,}648$ weights.

A bottleneck 256 → 64 → 64 → 256 needs $256 \cdot 64 + 9 \cdot 64^2 + 64 \cdot 256 = 16{,}384 + 36{,}864 + 16{,}384 = 69{,}632$ weights, 17× fewer.

That saving is what let ResNet-50 use three layers per block and 16 blocks while staying cheaper than VGG-16.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

residual blocks learn corrections on top of an identity shortcut, which turns the degradation problem into a non-problem by giving gradients a direct path; use bottlenecks for depth, projection shortcuts where shapes change, and zero-init the last layer of each branch.

</div>
