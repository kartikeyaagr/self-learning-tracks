---
id: "03.03"
course: "deep-learning"
module: "03"
slug: "classic-cnns-lenet-alexnet-vgg"
title: "Classic CNNs: LeNet, AlexNet, VGG"
keyIdeas: ["Every classic CNN is a **feature extractor** (conv and pool layers that halve height and width while adding channels) followed by a **dense classifier head**. LeNet-5 (1998) set this template with about 60k parameters for 32×32 digits.", "AlexNet (2012) scaled LeNet's shape to ImageNet and won with 15.3% top-5 error against 26.2% for the next entry. The layers were not new; ReLU, dropout, heavy data augmentation and two-GPU training made a 60-million-parameter network trainable.", "VGG (2014) used only 3×3 convolutions and varied depth. Two stacked 3×3 layers see a 5×5 region with 28% fewer weights than one 5×5 layer (three see 7×7 with 45% fewer), and each layer adds a ReLU.", "Parameters and compute live in different places: VGG-16's dense head holds 89% of its 138M parameters, but the convolutions do over 99% of its 15.5 billion multiply-adds. Halving resolution while doubling channels keeps each conv layer's cost the same.", "Plain stacks stop improving around 20 layers and get harder to train as they deepen. Global average pooling replaced the dense head, and residual connections (03.04) fixed the depth limit."]
question: "Someone proposes replacing each pair of 3×3 convolutions in a VGG block with a single 5×5 convolution \"since the receptive field is the same and there's one fewer layer to run\". For 256 input and output channels at 56×56, compare parameters, compute and expressiveness, and say which you would keep."
answer: "Count weights first (biases are negligible). Two 3×3 layers: $2 \\times 9 \\times 256^2 = 1{,}179{,}648$ weights. One 5×5 layer: $25 \\times 256^2 = 1{,}638{,}400$ weights, about 39% more. Compute scales the same way, because both run at 56×56 with padding (3,136 output positions) and each weight is used once per position: $3{,}136 \\times 1{,}179{,}648 \\approx 3.70$ billion multiply-adds for the pair versus $3{,}136 \\times 1{,}638{,}400 \\approx 5.14$ billion for the single layer. The receptive field is 5×5 in both cases, so the larger kernel buys no extra context. Expressiveness favours the pair: it applies two ReLUs, so it can compute functions of the 5×5 patch that one linear filter followed by one ReLU cannot, and the factorisation acts as a structured constraint that tends to generalise well (it is the argument of the VGG paper). The one real argument for the single 5×5 is one fewer kernel launch and one fewer intermediate activation map to store for backprop (56×56×256 numbers per image), which matters only if memory, not compute, is the bottleneck, and rarely outweighs 39% more FLOPs. Keep the two 3×3 convolutions; if the goal is speed, reduce channels or use a 1×1 bottleneck (03.05) instead."
sources: [{"title": "Krizhevsky, Sutskever, Hinton: ImageNet Classification with Deep Convolutional Neural Networks", "url": "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks", "note": "Section 3 on ReLU, multi-GPU training and pooling; Section 4 on augmentation and dropout."}, {"title": "Simonyan & Zisserman: Very Deep Convolutional Networks (VGG)", "url": "https://arxiv.org/abs/1409.1556", "note": "Section 2.3 gives the argument for stacks of 3×3 convolutions; Table 1 lists the configurations."}, {"title": "Dive into Deep Learning: LeNet", "url": "https://d2l.ai/chapter_convolutional-neural-networks/lenet.html", "note": "LeNet-5 layer by layer with shapes, trained on Fashion-MNIST."}, {"title": "Dive into Deep Learning: AlexNet", "url": "https://d2l.ai/chapter_convolutional-modern/alexnet.html", "note": "What changed from LeNet to AlexNet and why it mattered."}]
practice: ["lenet-forward-pass", "vgg-convolution-block", "multichannel-2d-convolution"]
tags: ["LeNet", "AlexNet", "VGG"]
tools: ["PyTorch"]
---

## What you will learn

- The **conv, nonlinearity, pool, repeat, then classify** template that LeNet-5 set in 1998, and how to trace its shapes and count its parameters by hand.
- What AlexNet (2012) added to that template (ReLU, dropout, data augmentation, GPUs) and why each piece was needed at ImageNet scale.
- Why VGG (2014) built everything from 3×3 convolutions, and the arithmetic that makes small stacked kernels cheaper than one big kernel.
- Where a classic CNN keeps its parameters (the dense head) versus where it spends its compute (the convolutions), and why that split shaped every later design.
- What limited these plain networks, which sets up residual networks (03.04) and efficient CNNs (03.05).

*Before you start:* you need convolution, padding, stride and receptive fields (03.01), and max/average pooling (03.02). ReLU and dropout (module 01 and 02) appear by name; a quick look back helps but is not required.

## From building blocks to whole networks

Lessons 03.01 and 03.02 gave you the parts (convolution and pooling). This lesson shows how three famous networks assembled them into complete image classifiers. Few new products deploy them today, but each made a design decision every later CNN inherits and hit a limit the next generation was built to fix. They are also ideal practice for counting shapes, parameters and compute.

## The idea in one picture

Every classic CNN has the same two-part shape. A **feature extractor** (also called the backbone) is a stack of conv and pool layers: it repeatedly shrinks the image's height and width while adding more channels, so each position ends up describing a bigger patch of the image in richer terms. A **classifier head** then flattens the final small feature map into one long vector and runs it through ordinary dense (fully connected) layers to produce one score per class.

Here is that pattern for VGG-16 on a 224×224 colour photo:

| Point in the network | Height × width | Channels | Numbers in the map
| --- | --- | --- | --- |
| Input image | 224 × 224 | 3 | 150,528
| After stage 1 (2 convs + pool) | 112 × 112 | 64 | 802,816
| After stage 3 (3 convs + pool) | 28 × 28 | 256 | 200,704
| After stage 5 (3 convs + pool) | 7 × 7 | 512 | 25,088
| Head output | 1 × 1 | 1000 | 1,000

~~~dg-fig 03.03-vgg-shape.svg
VGG-16 layer by layer. Each pool (vertical line) halves the side length; each new stage doubles the channels until 512. The image becomes a 7×7×512 block that the dense head reads.
~~~

Early layers see small patches in fine detail (edges, colour blobs); late layers see big patches coarsely (object parts). Every network in this lesson has this shape; what changes is the size of each piece and the tricks that make it trainable.

## Step 1: LeNet-5, the template

LeCun and colleagues built LeNet-5 (published 1998) to read handwritten digits on bank cheques. Its input is a 32×32 grayscale image. In the modern re-implementation the layers are:

1. Conv 5×5 with 6 output channels, no padding: $32 - 5 + 1 = 28$, so 28×28×6. Then 2×2 pooling to 14×14×6.
2. Conv 5×5 with 16 output channels: $14 - 5 + 1 = 10$, so 10×10×16. Then pooling to 5×5×16.
3. Flatten: $5 \times 5 \times 16 = 400$ numbers.
4. Dense layers 400 → 120 → 84 → 10, one score per digit.

To count parameters you need one formula per layer type. A conv layer with kernel $k \times k$, $C_{in}$ input channels and $C_{out}$ filters has

$$\text{params}_{conv} = C_{out}\,(k^2 C_{in} + 1),$$

where the $+1$ is each filter's bias. A dense layer from $n_{in}$ to $n_{out}$ units has $n_{out}(n_{in} + 1)$ parameters.

<div class="ex">
<div class="lab">Worked example</div>

**trace LeNet-5 and count its parameters by hand**

Conv 1: each of the 6 filters is $5 \times 5 \times 1 = 25$ weights plus 1 bias, so $6 \times 26 = 156$ parameters. Output 28×28×6.

Pool 1: no parameters. Output 14×14×6.

Conv 2: each of the 16 filters spans all 6 input channels, $5 \times 5 \times 6 = 150$ weights plus 1 bias, so $16 \times 151 = 2{,}416$. Output 10×10×16, pooled to 5×5×16 = 400.

Dense 400 → 120: $120 \times 401 = 48{,}120$. Dense 120 → 84: $84 \times 121 = 10{,}164$. Dense 84 → 10: $10 \times 85 = 850$.

Total: $156 + 2{,}416 + 48{,}120 + 10{,}164 + 850 = 61{,}706$. The three dense layers hold $59{,}134$ of them, about 96%.

Compute tells the opposite story. Conv 1 does $28 \cdot 28 \cdot 6 \cdot 25 = 117{,}600$ multiply-adds and conv 2 does $10 \cdot 10 \cdot 16 \cdot 150 = 240{,}000$, while all dense layers together do only $48{,}000 + 10{,}080 + 840 = 58{,}920$. So 4% of the parameters do 86% of the work. Keep that pattern in mind; VGG repeats it at 2,000 times the size.

</div>

The 1998 original differed in details (scaled tanh, averaging subsampling with a learned scale, a second conv wired to only some input channels, a distance-based RBF output) and has about 60,000 parameters. The ideas that survived are the ones that matter: **local receptive fields**, **shared weights**, **spatial subsampling**, and training the whole thing **end to end with backpropagation** instead of hand-designing features.

Swapping tanh for ReLU and averaging for max pooling is the standard modern update. A small test (scikit-learn's 8×8 digits upscaled to 32×32, three seeds, identical SGD settings) shows what it buys. For the first few epochs tanh is actually slightly ahead (after epoch 5 its training loss is 0.46 against 0.50 for ReLU), so do not expect miracles from one swap. After that tanh slows down while ReLU keeps going: after 20 epochs its training loss is 0.008 against 0.061, almost eight times lower.

~~~dg-fig 03.03-lenet-relu.svg
The same LeNet-5 trained two ways on 1,257 digit images. The tanh version stalls while ReLU plus max pooling keeps pushing the training loss down. On this easy task both reach about 97% test accuracy; the gap shows up in how well the network can fit, which matters far more on a hard problem like ImageNet.
~~~

## Step 2: AlexNet, the same template at scale

For fourteen years after LeNet, CNNs stayed a niche tool. Krizhevsky, Sutskever and Hinton changed that by winning the ImageNet challenge (ILSVRC-2012: 1.2 million training photos, 1000 classes) with **15.3% top-5 error** against **26.2%** for the next team. Top-5 error is the fraction of test images whose true label is not among the model's five highest-scoring guesses.

~~~dg-fig 03.03-ilsvrc.svg
ImageNet top-5 error of notable ILSVRC entries (published values, quoted from memory). AlexNet's 2012 drop of 10.5 points dwarfed the previous year's 2.4; VGG in 2014 halved it again.
~~~

The architecture is LeNet's shape, just much larger:

- Input 224×224×3 (227 makes the arithmetic of the first layer come out exactly).
- 5 conv layers. The first uses 96 filters of size 11×11 with stride 4, which shrinks the image fast; later ones are 5×5 and 3×3 with 256, 384, 384 and 256 channels.
- Overlapping max pooling (3×3 windows, stride 2) after conv 1, 2 and 5, giving a 6×6×256 map.
- Dense head 9,216 → 4096 → 4096 → 1000. About 60 million parameters in total (62.4 million if you build it as one tower; the original split some layers across two GPUs, which cut a few connections).

Five training choices, not the layers, made it work:

- **ReLU** ($\max(0, x)$) instead of tanh. It does not saturate for positive inputs, so gradients stay large; the paper reports a four-layer CNN reaching 25% training error on CIFAR-10 about six times faster than with tanh.
- **Dropout** 0.5 in the first two dense layers, which hold most of the parameters; without it the network overfit badly.
- **Data augmentation**: random 224×224 crops and flips from 256×256 images, plus PCA colour jitter.
- **GPUs**: five to six days on two 3 GB GTX 580 cards, the model split between them.
- **Local response normalisation (LRN)** across neighbouring channels; VGG later found it did not help, and BatchNorm (02.01) replaced it.

## Step 3: VGG, depth from small kernels

AlexNet mixed kernel sizes (11, 5, 3) and strides by hand. Simonyan and Zisserman asked a cleaner question: fix every conv to 3×3, stride 1, padding 1 (so it never changes height or width), use 2×2 max pooling with stride 2 to halve the size, and vary only the **depth**. The result, VGG, placed second in ILSVRC-2014 classification (first was GoogLeNet) with 7.3% top-5 error for the submitted ensemble.

The unit of design is the **VGG block**: $n$ conv layers of 3×3 with the same channel count, each followed by ReLU, then one max pool. VGG-16 uses five blocks with 2, 2, 3, 3, 3 convs and 64, 128, 256, 512, 512 channels, then the same three dense layers as AlexNet. That is 13 conv + 3 dense = 16 weight layers, hence the name.

~~~dg-layers VGG-16 as five stages: each halves resolution and (up to 512) doubles channels; the dense head holds most parameters.
Input | 224×224×3
Stages 1-2 | 2 × conv3-64, pool; 2 × conv3-128, pool → 56×56×128
Stages 3-5 | 3 × conv3-256, 3 × conv3-512, 3 × conv3-512, each then pool → 7×7×512
*Dense head | 25,088 → 4096 → 4096 → 1000, about 124M of 138M parameters
~~~

Why only 3×3? Because a stack of small kernels sees as far as one big kernel, for fewer weights. Each 3×3 layer grows the receptive field by 2 pixels (03.01), so two stacked layers cover 5×5 and three cover 7×7. Now count weights for $C$ input and $C$ output channels (ignoring biases):

$$\underbrace{2 \cdot 3^2 C^2 = 18C^2}_{\text{two } 3{\times}3} \;\text{vs}\; \underbrace{5^2 C^2 = 25C^2}_{\text{one } 5{\times}5}, \qquad \underbrace{3 \cdot 3^2 C^2 = 27C^2}_{\text{three } 3{\times}3} \;\text{vs}\; \underbrace{7^2 C^2 = 49C^2}_{\text{one } 7{\times}7}.$$

That is 28% and 45% fewer weights. The stack also applies a ReLU after every layer, so three 3×3 layers compute a more nonlinear function of the same 7×7 patch than one 7×7 filter followed by one ReLU. The gap grows with the receptive field: the stack's cost is linear in depth, a single kernel's is quadratic in width.

~~~dg-fig 03.03-stack-3x3.svg
Weights needed to cover a given receptive field. A stack of 3×3 layers grows by 9C² per layer; a single k×k kernel grows as k²C². The saving is 28% at 5×5, 45% at 7×7 and 63% at 11×11 (AlexNet's first kernel size).
~~~

## Step 4: where the parameters live and where the compute goes

Two numbers describe a network's cost. **Parameters** decide memory for weights and how much data you need to avoid overfitting. **Multiply-adds** (MACs) per image decide speed. A conv layer's MACs are

$$\text{MACs}_{conv} = H_{out} \cdot W_{out} \cdot C_{out} \cdot k^2 C_{in},$$

which is its weight count (without biases) multiplied by the number of output positions $H_{out} W_{out}$. A dense layer's MACs simply equal its weight count, because each weight is used once.

A conv weight is reused at every position, thousands of times per image; a dense weight is used once. That is the whole story of the next figure.

~~~dg-fig 03.03-params-vs-compute.svg
VGG-16 counted with PyTorch. Left: the dense head holds 123.6M of 138.4M parameters. Right: the head does only 0.12 of 15.5 billion multiply-adds; the conv stages do the rest, with stages 3 and 4 the most expensive.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**where VGG-16's parameters and compute live**

Largest conv layer, 512 → 512 at 3×3: $512 \times 512 \times 9 = 2{,}359{,}296$ weights (plus 512 biases). There are five of these (two in stage 4, three in stage 5).

All 13 conv layers together: $14{,}714{,}688$ parameters.

Dense head: $25{,}088 \times 4096 + 4096 \times 4096 + 4096 \times 1000 = 102.76\text{M} + 16.78\text{M} + 4.10\text{M} = 123.6\text{M}$ weights, $123{,}642{,}856$ with biases.

Total $138{,}357{,}544$, so the head holds $123.6 / 138.4 = 89\%$. The first dense layer alone is $102.8 / 138.4 = 74\%$, three quarters of the whole network.

Compute: one stage-1 conv (64 → 64 at 224×224) costs $224^2 \cdot 64 \cdot 64 \cdot 9 = 1.85$ billion MACs. One stage-2 conv (128 → 128 at 112×112) costs $112^2 \cdot 128 \cdot 128 \cdot 9 = 1.85$ billion too, and so does one 256 → 256 conv at 56×56. Halving the side divides positions by 4; doubling the channels multiplies $C_{in} C_{out}$ by 4; the two cancel.

Summed: convs 15.35 billion MACs, head 0.12 billion. Over 99% of the arithmetic is in the convolutions. Replacing the head by global average pooling plus one 512 → 1000 layer (03.02) would delete about 123 million parameters while saving almost no time.

</div>

Stages 3 and 4 cost the most only because they have three conv layers instead of two; stage 5's convs cost only $14^2 \cdot 512 \cdot 512 \cdot 9 = 0.46$ billion each, because channels stop doubling at 512. So compute is spread roughly evenly over stages 1-4, not concentrated in the high-resolution early layers.

## Classic CNNs in code

In PyTorch both networks are a few lines. Trace shapes with a fake input before you train anything; it catches most wiring mistakes.

```python
import torch, torch.nn as nn

lenet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),    # 32x32x1 -> 28x28x6 -> 14x14x6
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),   # -> 10x10x16 -> 5x5x16
    nn.Flatten(),                                                  # -> 400
    nn.Linear(400, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))

x = torch.zeros(1, 1, 32, 32)            # one fake 32x32 grey image: shape tracing needs no data
for layer in lenet:
    x = layer(x)
    if not isinstance(layer, nn.ReLU):
        print(f"{layer.__class__.__name__:10s} {tuple(x.shape)}")
# Conv2d (1, 6, 28, 28) / MaxPool2d (1, 6, 14, 14) / Conv2d (1, 16, 10, 10) / MaxPool2d (1, 16, 5, 5)
# Flatten (1, 400) / Linear (1, 120) / Linear (1, 84) / Linear (1, 10)
print("parameters:", sum(p.numel() for p in lenet.parameters()))    # parameters: 61706

def vgg_block(c_in, c_out, n_convs):
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.MaxPool2d(2))

features = nn.Sequential(vgg_block(3, 64, 2), vgg_block(64, 128, 2), vgg_block(128, 256, 3),
                         vgg_block(256, 512, 3), vgg_block(512, 512, 3))
head = nn.Sequential(nn.Flatten(), nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                     nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 1000))
print(features(torch.zeros(1, 3, 224, 224)).shape)                  # torch.Size([1, 512, 7, 7])
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"conv: {count(features):,}  head: {count(head):,}")          # conv: 14,714,688  head: 123,642,856
```

`torchvision.models` ships `alexnet`, `vgg11` to `vgg19` and `vgg16_bn` with pretrained weights; VGG features are still used, for example, in perceptual losses for image generation.

## Checking your own network

Three checks before training:

- **Shape trace.** Push a zero tensor through and print each shape. What reaches the first dense layer must match its `in_features` (25,088 for VGG at 224×224).
- **Parameter and compute split.** Count parameters per layer and estimate MACs with the step 4 formula; the biggest dense layer is where overfitting and checkpoint size come from.
- **Early loss.** It should fall well below $\ln(\text{classes})$ (2.30 for 10 classes, 6.91 for 1000) within an epoch or two; stuck there means uniform guessing, so check learning rate, initialisation and input scaling.

## What goes wrong

| Problem | Symptom | Cause | Fix
| --- | --- | --- | --- |
| Wrong flatten size | `mat1 and mat2 shapes cannot be multiplied` on the first dense layer | input resolution differs from the one the head was built for | trace shapes; or use global average pooling (`nn.AdaptiveAvgPool2d`) before the head
| Overfitting in the head | training accuracy near 100%, validation far lower | tens of millions of dense weights, too little data | dropout, augmentation, or replace the head with global average pooling
| Deep plain stack will not train | loss stays near $\ln(\text{classes})$ for many epochs | signals and gradients shrink through many layers | He initialisation (01.05), BatchNorm (02.01), residual connections (03.04)
| Slow training with tanh or sigmoid | loss falls very slowly, gradients tiny | saturating activations | ReLU
| Using big kernels "for context" | many parameters and FLOPs for no accuracy gain | one 7×7 costs 81% more than three 3×3 | stack 3×3 layers instead
| Huge checkpoints | VGG-16 file over 500 MB (138M float32 weights) | dense head | global average pooling head, or a modern backbone

The third row deserves a picture. VGG-style stacks of 3×3 convs (16 channels, two pools, no BatchNorm, identical SGD settings) are trained on the same digits at 16×16. With PyTorch's default initialisation, the 4-layer stack learns, but the 8-, 16- and 24-layer stacks sit at a loss of exactly $\ln 10 = 2.30$ for all 15 epochs and score 10% on test data: pure guessing. Re-running the 8- and 16-layer stacks with He initialisation (01.05) and changing nothing else, they train to 97% and 96%.

~~~dg-fig 03.03-plain-depth.svg
Plain stacks of 3×3 convolutions without BatchNorm. Solid lines: default initialisation (two seeds each); every stack of 8 or more layers stays on the dotted uniform-guess line. Dashed: the same 8- and 16-layer stacks with He initialisation (one seed) train fine, though the 16-layer one is slower than the 8-layer one.
~~~

VGG hit this limit itself: VGG-19 was barely better than VGG-16, and the authors initialised deeper configurations from layers of a trained 11-layer net, finding only later that Glorot initialisation worked without that step.

## Going deeper

**Why depth helps at all.** Each layer reuses the features of the one below: edges become textures, textures become parts, parts become objects. VGG's error fell from 11 to 16 layers and flattened at 19.

**Why it stops helping.** In a plain stack the signal and its gradient pass through every layer's weights in turn; if each layer shrinks them slightly, the product over 20 layers is tiny and early layers barely learn. He initialisation and BatchNorm keep that factor near 1, which let 20-30 layer plain nets train. Even then, He et al. found 56-layer plain nets had *higher training error* than 20-layer ones: an optimisation failure, not overfitting, which residual connections (03.04) fix.

**What changed next.** The dense head gave way to global average pooling plus one linear layer (Network in Network, GoogLeNet, ResNet); the plain stack gave way to residual blocks (03.04); the cost of full 3×3 convs over many channels led to 1×1 bottlenecks and depthwise convolutions (03.05). The 3×3 kernel itself survived as the default.

**Two-GPU grouping.** AlexNet's split across two GPUs meant some layers only connected to the channels on their own GPU. That accident is the origin of **grouped convolution**: splitting channels into $g$ groups divides a layer's weights and MACs by $g$. ResNeXt and depthwise convolution (the extreme case, one channel per group) both build on it.

## Key terms

| Term | Meaning
| --- | --- |
| Feature extractor (backbone) | The stack of conv and pool layers that turns an image into a small, many-channel feature map
| Classifier head | Layers after the backbone that map features to class scores; dense layers in classic CNNs
| Top-5 error | Fraction of images whose true label is not among the model's five best guesses
| VGG block | Several 3×3 convs with the same channel count, each followed by ReLU, then a 2×2 max pool
| Multiply-adds (MACs) | Count of multiply-then-add operations per image; the standard measure of a layer's compute
| Dropout | Randomly zeroing units during training so the network cannot rely on any single one
| Data augmentation | Training on randomly altered copies (crops, flips, colour shifts) to enlarge the dataset
| Local response normalisation | AlexNet's cross-channel normalisation, later shown unnecessary and replaced by BatchNorm

<div class="take">
<div class="lab">Key takeaway</div>

LeNet set the conv-pool-dense template, AlexNet proved it scales to ImageNet once you add ReLU, dropout, augmentation and GPUs, and VGG showed that stacks of 3×3 convolutions are the cheapest way to grow the receptive field. Their parameters sit in the dense head while their compute sits in the convolutions, and plain depth stops paying off around 20 layers; fixing those two facts produced the designs that came next.

</div>
