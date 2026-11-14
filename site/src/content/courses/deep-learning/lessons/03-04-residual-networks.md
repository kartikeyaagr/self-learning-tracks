---
id: "03.04"
course: "deep-learning"
module: "03"
slug: "residual-networks"
title: "Residual networks"
keyIdeas: ["Plain networks deeper than about 20 layers trained **worse**, with higher *training* error, not just test error: the degradation problem is a failure of optimisation, not overfitting.", "A residual block computes $y = x + F(x)$, so each block only has to learn a correction to its input, and doing nothing ($F = 0$) is the easy default.", "The identity path gives backprop a direct route: $\\partial \\mathcal{L}/\\partial x_l = \\partial \\mathcal{L}/\\partial x_L\\,(1 + \\partial(\\sum F)/\\partial x_l)$, so the gradient never has to pass through every weight layer, and gradient sizes stay similar across layers.", "Bottleneck blocks (1×1 reduce, 3×3, 1×1 expand) make 50-152 layer nets affordable; a 1×1 projection shortcut handles changes in resolution or channel count.", "Pre-activation blocks keep the shortcut a pure identity for 1,000-layer nets, and zero-initialising the last BN scale of each branch makes every block start as an identity, which removes most depth-related training trouble."]
question: "Explain why the degradation problem cannot be overfitting, then use the residual gradient formula to argue why a 100-block ResNet still gets useful gradients at its first block even if the branch Jacobians are all small. What happens to that argument if you add a ReLU after every addition (as the original ResNet does)?"
answer: "Overfitting means low training error but high test error. In the degradation experiments the deeper plain net had higher *training* error than the shallower one, even though it could represent the shallower solution exactly by setting its extra layers to the identity. A better solution exists in its parameter space and the optimiser fails to find it, so the failure is in optimisation, not generalisation. In a ResNet with clean identity shortcuts, $\\partial \\mathcal{L}/\\partial x_1 = \\partial \\mathcal{L}/\\partial x_{100}\\,(1 + \\partial(\\sum_i F_i)/\\partial x_1)$. The leading 1 passes the top gradient to the first block unchanged, so small branch Jacobians shrink only the correction terms and never the base signal. Expanding the product of $(1 + J_i)$ factors shows $2^{99}$ paths, the shortest of which has no weight layers at all. In a plain network the gradient is a product of 100 Jacobians, so it vanishes if their gains are below one and explodes if they are above one (with BatchNorm, plain stacks tend to explode towards the input), leaving layers with gradients that differ by orders of magnitude. With a ReLU after each addition the shortcut is no longer a pure identity: each step multiplies the gradient by a 0/1 mask, so units that are off block their component of the shortcut, and the 1 in the formula becomes a diagonal 0/1 matrix. In practice most units are active and the path is still short, so post-activation ResNets train well to about 150 layers, but at around 1,000 layers the masks compound and training degrades. That is why pre-activation ResNets move BN and ReLU into the start of the branch and keep the shortcut clean; zero-initialising the last BN scale of each branch helps further by making every block an exact identity at the start."
sources: [{"title": "He et al.: Deep Residual Learning for Image Recognition", "url": "https://arxiv.org/abs/1512.03385", "note": "Figure 1 (degradation), Section 3 (residual blocks, projection shortcuts), Table 1 (ResNet-18 to 152)."}, {"title": "He et al.: Identity Mappings in Deep Residual Networks", "url": "https://arxiv.org/abs/1603.05027", "note": "The gradient decomposition and the pre-activation block that trains 1001-layer nets."}, {"title": "Li et al.: Visualizing the Loss Landscape of Neural Nets", "url": "https://arxiv.org/abs/1712.09913", "note": "Loss-surface plots with and without skip connections."}, {"title": "Dive into Deep Learning: Residual Networks", "url": "https://d2l.ai/chapter_convolutional-modern/resnet.html", "note": "Residual blocks and ResNet-18 in code, with the function-class argument."}]
practice: ["resnet-residual-block", "residual-block", "apply-alphago-residual-block", "time-conditioned-residual-block"]
tags: ["ResNet", "skip connections", "bottleneck"]
tools: ["PyTorch"]
---

## What you will learn

- What the **degradation problem** is, and why it is an optimisation failure rather than overfitting.
- What a residual block computes, $y = x + F(x)$, and why "learn a correction" is easier than "learn the whole mapping".
- How the identity shortcut changes the gradient, checked by hand on a three-block toy network.
- The block designs used in real ResNets: basic, bottleneck, projection shortcut and pre-activation.
- How to build a bottleneck block in PyTorch, check that it starts as an identity, and count its weights.
- The main explanations for why ResNets train, and the failure modes to watch for.

*Before you start:* convolution and channels (03.01), the VGG design and its depth limit (03.03), He initialisation (01.05), BatchNorm (02.01) and backpropagation as a chain of Jacobians (01.04).

Lesson 03.03 ended with VGG: stack more 3×3 convolutions and accuracy rises. By 2015 He initialisation and BatchNorm made 20-30 layer plain stacks trainable, so the obvious move was to keep stacking. It did not work, and the fix, the residual connection, is now in almost every deep network, from ResNet-50 to Transformers.

## The idea in one picture

Take the 1,797 handwritten digit images that ship with scikit-learn (8×8 pixels, 10 classes) and train fully connected networks of width 32 with BatchNorm after every layer. Dense layers instead of convolutions just keep each run to seconds on a CPU; the effect is the one He et al. reported for CNNs. Train each network for 60 epochs with SGD (learning rate 0.01, momentum 0.9) and record the error **on the training set itself**, averaged over three seeds:

| Hidden layers | 4 | 10 | 20 | 40
| --- | --- | --- | --- | --- |
| Plain stack: training error | 2.1% | 10.7% | 36.5% | 74.9%
| Residual (same layers, with shortcuts): training error | 2.3% | 3.2% | 4.0% | 7.2%

The plain 40-layer network is barely better than guessing (90% error) after 60 epochs, while the 4-layer one fits almost every example. The residual network, which has exactly the same layers plus a shortcut around every pair, keeps working as depth grows.

~~~dg-fig 03.04-idea.svg
Training loss over 60 epochs (lines are the mean of three seeds, bands the range). The 40-layer plain network stays near the loss of a random guess; the same 40 layers with residual shortcuts train almost as well as the shallow net.
~~~

Read the table carefully: these are *training* errors. The deep plain network is not memorising and failing to generalise; it cannot even fit the data it is shown. That is the **degradation problem**: adding layers to a network makes it *harder to train*, so accuracy first saturates and then drops.

The fix is the **residual block**: instead of asking a stack of layers to produce the output directly, it computes the output as "input plus a learned correction", $y = x + F(x)$. The path that carries $x$ unchanged is the **shortcut** (or skip connection), and $F$ is the **residual branch**.

## Step 1: why degradation is not overfitting

Overfitting means low training error and high test error: the model fits the training set too well. Degradation is the opposite pattern: the deeper model is worse on the training set.

And a deeper network *should* never be worse. Take a trained 20-layer network and add 20 layers that each compute the identity, $h \mapsto h$: the 40-layer network computes the same function with the same training error. A solution as good as the shallow one exists in the deeper model's parameter space; if training ends somewhere worse, the optimiser failed to find it.

He et al. (2015) saw exactly this: on CIFAR-10 a 56-layer plain network had higher training error than a 20-layer one, and on ImageNet plain-34 was worse than plain-18. Their networks used BatchNorm, so forward signals were well scaled and "vanishing gradients" was not the whole story. A stack of conv-BN-ReLU layers simply finds it hard to learn even the identity.

## Step 2: make the identity the default

If the identity is hard to learn, build it in. A residual block computes

$$y = x + F(x; W),$$

where $x$ is the block's input (a feature map with $C$ channels), $F$ is a small stack of two or three conv layers with weights $W$, and $y$ has the same shape as $x$ so the two can be added element by element.

The block is an identity when $F(x) = 0$, which is easy to reach: small weights, weight decay or a zero-initialised last layer all push $F$ towards zero. Each block only learns a **residual**: the difference between what it should output and what it receives. Deep inside a network, where each layer only refines features, that correction is small and easy to learn.

~~~dg-flow A basic residual block: the shortcut carries x unchanged and the branch learns a correction that is added before the final ReLU.
x -(conv3×3, BN, ReLU)-> h -(conv3×3, BN)-> F(x) -(+ x)-> *x + F(x) -(ReLU)-> y
~~~

Number check: if a block receives 1.80 and the ideal output is 1.95, a plain block must produce 1.95 from scratch; the residual branch only outputs 0.15.

## Step 3: what the shortcut does to the forward pass

Ignore the ReLU after the addition for a moment (the pre-activation design in Step 6 removes it). Stack blocks $l, l+1, \dots, L-1$. Each adds its correction to the running signal, so the output of block $L-1$ is

$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i),$$

where $x_i$ is the input to block $i$. The deep feature is the shallow feature plus a sum of corrections: early information is never overwritten, only added to. In a plain network $x_L = f_{L-1}(\dots f_l(x_l))$ and every layer must actively pass the signal along.

## Step 4: what the shortcut does to the gradient

Differentiate the sum with the chain rule. Writing $\mathcal{L}$ for the loss:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L}\left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F(x_i)\right).$$

The term $\partial \mathcal{L}/\partial x_L$ is the gradient arriving at the top; the bracket says how it is passed back to block $l$. The "1" (an identity matrix, for vectors) means the top gradient reaches every earlier block directly, without being multiplied by any weight matrix. Small branch Jacobians shrink only the correction terms, never the base signal.

In a plain network the same quantity is a **product** of one Jacobian per layer. If each has typical gain 0.8, forty of them give $0.8^{40} \approx 0.0001$; if each has gain 1.2, they give about 1,500. Either way the gradient at the bottom is out of scale with the gradient at the top.

~~~dg-fig 03.04-grads.svg
Gradient size of each layer's weights at initialisation, for one batch of 256 digit images. In the 40-layer plain BatchNorm network it falls from about 342 at the first layer to 0.32 at the last, a factor of about 1,100; with residual shortcuts it goes from 5.6 to 0.66, a factor of about 8.
~~~

The measured plot holds a surprise: with BatchNorm the plain network's gradients do not vanish, they *explode* towards the input, a known effect of stacking BatchNorm layers. No single learning rate suits layers whose gradients differ a thousandfold, so training stalls. The shortcut keeps every layer's gradient in a small range.

<div class="ex">
<div class="lab">Worked example</div>

**the gradient of three blocks, by hand**

Use scalars so every step is visible. Three blocks have branch slopes $a_1 = 0.2$, $a_2 = 0.5$, $a_3 = -0.3$: a small change $\delta$ in a block's input changes its branch output by $a_i \delta$.

Plain network (the output of each layer is just the branch): the gradient from top to bottom is the product $a_1 a_2 a_3 = 0.2 \times 0.5 \times (-0.3) = -0.03$. The signal has shrunk to 3% and even flipped sign.

Residual network: each block's slope is $1 + a_i$, so the gradient is $(1.2)(1.5)(0.7)$. First $1.2 \times 1.5 = 1.8$, then $1.8 \times 0.7 = 1.26$.

Expand the product to see where 1.26 comes from. Constant term: 1 (the path that skips all three branches). Single terms: $0.2 + 0.5 - 0.3 = 0.4$. Pairs: $0.2 \cdot 0.5 + 0.2 \cdot (-0.3) + 0.5 \cdot (-0.3) = 0.1 - 0.06 - 0.15 = -0.11$. Triple: $-0.03$. Total: $1 + 0.4 - 0.11 - 0.03 = 1.26$.

There are $2^3 = 8$ terms, one for every way of choosing which branches to pass through. The plain network keeps only the last one, $-0.03$; the residual network keeps all eight, and the short paths dominate.

</div>

That expansion is the seed of the "implicit ensemble" view in Going deeper.

## Step 5: blocks used in real ResNets

The same idea comes in a few standard shapes.

- **Basic block** (ResNet-18 and ResNet-34): two 3×3 convs, each followed by BN, with a ReLU between them and another after the addition.
- **Bottleneck block** (ResNet-50, 101, 152): a 1×1 conv cuts the channels by 4×, a 3×3 conv works at the reduced width, and a 1×1 conv expands back. The expensive 3×3 runs on a quarter of the channels.
- **Projection shortcut**: when a block halves the resolution (stride 2) or changes the channel count, $x$ no longer has the shape of $F(x)$. The shortcut then becomes a 1×1 conv with the same stride, followed by BN. Everywhere else the shortcut is a free identity with no parameters.

<div class="ex">
<div class="lab">Worked example</div>

**why bottlenecks are cheaper**

A 3×3 conv from $C$ channels to $C$ channels has $9C^2$ weights; a 1×1 conv from $C_1$ to $C_2$ has $C_1 C_2$.

Basic block at 256 channels: $2 \times 9 \times 256^2 = 2 \times 9 \times 65{,}536 = 1{,}179{,}648$ weights.

Bottleneck 256 → 64 → 64 → 256: $256 \cdot 64 + 9 \cdot 64^2 + 64 \cdot 256 = 16{,}384 + 36{,}864 + 16{,}384 = 69{,}632$ weights, about 17× fewer ($1{,}179{,}648 / 69{,}632 \approx 16.9$).

The fairer comparison, and the one the ResNet paper makes, is a basic block at 64 channels: $2 \times 9 \times 64^2 = 73{,}728$ weights. The bottleneck costs slightly less, yet its input and output carry 256 channels instead of 64, and it adds a third nonlinearity.

</div>

~~~dg-fig 03.04-bottleneck.svg
Conv weights per block. A bottleneck reads and writes 256 channels for about the price of a 64-channel basic block, which is how ResNet-50 got deeper and wider than ResNet-34 at a similar cost.
~~~

The networks themselves follow VGG's stage pattern: a 7×7 stride-2 stem and a max-pool, then four stages at 56×56, 28×28, 14×14 and 7×7 resolution, global average pooling and one linear layer. Each stage starts with a stride-2 block that has a projection shortcut.

~~~dg-layers ResNet-50 for a 224×224 image: bottleneck blocks per stage and the output channels of each stage.
Stem | 7×7 conv stride 2 (64 ch) and 3×3 max-pool, 224 → 56
Stage 1 | 3 bottleneck blocks, 56×56, 256 output channels
Stage 2 | 4 blocks, 28×28, 512 channels; first block strides by 2
*Stage 3 | 6 blocks, 14×14, 1024 channels; most of the compute
Stage 4 | 3 blocks, 7×7, 2048 channels
Head | global average pool, linear layer to 1000 classes
~~~

Counting layers: 16 blocks × 3 convs, plus the stem and the final linear layer, gives 50. ResNet-101 and 152 use the same stages with more blocks in stage 3 (and stage 2 for 152).

## Step 6: pre-activation and zero-initialisation

Two refinements matter in practice.

**Pre-activation** (ResNet v2, He et al. 2016) moves BN and ReLU to the *start* of the branch, $y = x + F(\mathrm{ReLU}(\mathrm{BN}(x)))$, with nothing after the addition. Now the shortcut is a pure identity from the first block to the last, exactly as the formulas in Steps 3 and 4 assume. This version trains networks of about 1,000 layers, and it is the form Transformers (the Pre-LN layout from 02.02) and diffusion U-Nets follow.

**Zero-initialising the last BN scale** $\gamma$ in each branch makes $F(x) = 0$ at step 0, so every block is an exact identity and a 100-block network starts out behaving like a shallow one. Goyal et al. used it for large-batch ImageNet training. In our digit experiment it removed degradation completely:

~~~dg-fig 03.04-depth.svg
Training error after 60 epochs against depth (mean of three seeds, bars show the range). Plain networks climb from 2% to 75%; residual networks with default initialisation reach 7.2% at 40 layers; with the last BN scale of each branch set to zero they stay between 0.9% and 1.7% at every depth.
~~~

## Residual blocks in code

Here is a bottleneck block with an optional projection shortcut and the zero-$\gamma$ trick, plus checks that it does what the text says.

```python
import torch, torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, c_in, c_mid, stride=1):
        super().__init__()
        c_out = 4 * c_mid
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_mid, 3, stride, 1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        nn.init.zeros_(self.f[-1].weight)          # last BN gamma = 0: block starts as identity
        self.short = nn.Identity() if stride == 1 and c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.f(x) + self.short(x))

torch.manual_seed(0)
same = Bottleneck(256, 64)                 # 256 -> 256 channels, identity shortcut
down = Bottleneck(256, 128, stride=2)      # 256 -> 512 channels, half resolution, projection
x = torch.relu(torch.randn(2, 256, 56, 56))
print(same(x).shape, down(x).shape)  # torch.Size([2, 256, 56, 56]) torch.Size([2, 512, 28, 28])
print(torch.allclose(same(x), x))    # True: with gamma = 0 the block is the identity
count = lambda m: sum(p.numel() for p in m.parameters())
print(count(same), count(down))      # 70400 379392
```

The 70,400 are the 69,632 conv weights from the example plus 768 BN parameters ($\gamma$, $\beta$ for 64 + 64 + 256 channels). The identity check passes only because `x` is non-negative (the final ReLU would clip negatives); inside a ResNet every block's input comes out of a ReLU, so it holds. Convs feeding BN use `bias=False` because BN's $\beta$ already shifts (02.01). The lesson's figures use the same pattern with `nn.Linear` and `nn.BatchNorm1d`.

## Checking that it works

- **Shapes:** push one dummy batch through each stage and print shapes; a mismatch at the addition is the commonest bug.
- **Gradient spread:** after one backward pass, compare the gradient norms of the first and last conv. A ratio of a few is healthy; hundreds or thousands (as in the plain net above) predicts a stall.
- **Depth sanity:** a deeper ResNet should reach equal or lower *training* loss than a shallower one on the same budget. If not, look for a broken shortcut.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Deeper model has higher training loss than shallower one | no shortcuts, or a shortcut silently missing | add residual connections; check every block's forward pass
| `RuntimeError` about sizes at the `+` | stride or channel change without a projection | use a 1×1 conv (with stride) on the shortcut where shapes change
| Loss spikes or NaN in the first steps of a very deep net | output variance grows with each added branch | zero-init the last BN $\gamma$ (or scale branches down), warm up the learning rate
| Very deep (500+ layers) post-activation net trains worse | ReLU after each addition blocks part of the shortcut | switch to pre-activation blocks
| Projection shortcuts everywhere, slow and no better | extra 1×1 convs on every block | identity shortcuts where shapes match, projection only at stage boundaries

## Going deeper

**Several explanations, all partly true.** The gradient-highway argument of Step 4 is the cleanest, but it is not the only reason ResNets work:

- **Easier targets.** Learning a small correction near zero is better conditioned than learning a full mapping, and identity is a good starting point for deep layers.
- **Smoother loss surfaces.** Li et al.'s loss-landscape visualisations show deep plain networks with chaotic, highly non-convex surfaces, while the same networks with skip connections look smooth and nearly convex around the minimum.
- **Implicit ensemble.** Expanding the product $\prod_i (1 + a_i)$ as in the hand example shows that a stack of $n$ blocks contains $2^n$ paths of different lengths.

~~~dg-fig 03.04-paths.svg
Path lengths in a 54-block residual network (the size of the CIFAR ResNet-110). The count of paths that pass through k branches is the binomial coefficient C(54, k), so 96% of all paths use between 20 and 34 branches.
~~~

Each pass through a branch usually scales the gradient down, so long paths contribute little. Veit et al. (2016) reported (published result, not rerun here) that most of the gradient in a 110-layer ResNet flows along paths only about 10-34 layers deep, and that deleting one block from a trained ResNet barely hurts accuracy while deleting a VGG layer is catastrophic. A deep ResNet behaves partly like an ensemble of shallower weight-sharing networks.

**The post-activation caveat.** With a ReLU after each addition, the shortcut is $\mathrm{ReLU}(x + F(x))$, so its Jacobian is a 0/1 mask rather than the identity. Units that are switched off block their part of the shortcut. Most units are active, so this works to about 150 layers, but by about 1,000 layers it degrades, which is the motivation for pre-activation.

**Variance growth.** In the forward direction, $x_L = x_l + \sum F(x_i)$ means variance adds up block by block. With independent branches of variance $v$, the output variance after $n$ blocks is roughly $1 + nv$, so it grows linearly with depth (much better than the exponential growth of a badly scaled plain net, but still growth). Zero-$\gamma$, scaled branches such as $x + \alpha F(x)$ with small $\alpha$, and normalisation-free schemes such as Fixup all control this term. It is also why the default-initialised residual nets in our depth sweep still lose a little accuracy at 40 layers while the zero-$\gamma$ ones do not.

**Published numbers to know (stated from the papers, not rerun here).** ResNet-152 needs about 11.3 billion FLOPs against VGG-16's 15.3 billion, yet is far more accurate; an ensemble of ResNets won ILSVRC-2015 classification at 3.57% top-5 error. ResNet-50 has 25.6 million parameters and about 3.8 billion multiply-adds per 224×224 image, and remains the standard vision baseline. In the original paper, projection shortcuts on every block (option C) helped only marginally over projections at shape changes only (option B), so B became the default.

## Key terms

| Term | Meaning
| --- | --- |
| Degradation problem | Deeper plain networks reach higher training error than shallower ones: an optimisation failure, not overfitting
| Residual block | A block that outputs its input plus a learned correction, $y = x + F(x)$
| Shortcut / skip connection | The path that carries $x$ around the branch unchanged (or through a 1×1 projection)
| Residual branch $F$ | The two or three conv layers that learn the correction
| Projection shortcut | A 1×1 conv (with stride and BN) used on the shortcut when shape changes
| Bottleneck block | 1×1 reduce, 3×3, 1×1 expand: deep and wide at low cost
| Pre-activation | BN and ReLU at the start of the branch, leaving a clean identity shortcut
| Zero-$\gamma$ init | Setting the last BN scale in each branch to zero so every block starts as an identity

<div class="take">
<div class="lab">Key takeaway</div>

a residual block learns a correction on top of an identity shortcut, $y = x + F(x)$, so doing nothing is the easy default and gradients reach every layer through the "1" in $1 + \partial F/\partial x$; use bottlenecks for depth, projection shortcuts only where shapes change, pre-activation for very deep nets, and zero-init the last BN scale of each branch.

</div>
