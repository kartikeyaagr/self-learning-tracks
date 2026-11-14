---
id: "03.03"
course: "deep-learning"
module: "03"
slug: "classic-cnns-lenet-alexnet-vgg"
title: "Classic CNNs: LeNet, AlexNet, VGG"
keyIdeas: ["LeNet-5 (1998) fixed the template still used today: conv, nonlinearity, pool, repeated, then dense layers; about 60k parameters for 32×32 digits.", "AlexNet (2012) scaled the template to ImageNet with ReLU, dropout, heavy augmentation and two GPUs, cutting top-5 error from 26.2% to 15.3% and starting the deep learning era in vision.", "VGG (2014) showed depth with only 3×3 kernels works: two 3×3 convs see a 5×5 region with 28% fewer parameters and an extra nonlinearity.", "These networks put most parameters in the dense head (about 90% in VGG-16) and most compute in the convolutions; later designs removed the head and attacked compute."]
question: "Someone proposes replacing each pair of 3×3 convolutions in a VGG block with a single 5×5 convolution \"since the receptive field is the same and there's one fewer layer to run\". For 256 input and output channels at 56×56, compare parameters, compute and expressiveness, and say which you would keep."
answer: "Two 3×3 layers: $2 \\times 9 \\times 256^2 = 1{,}179{,}648$ weights. One 5×5 layer: $25 \\times 256^2 = 1{,}638{,}400$ weights, about 39% more. Compute scales the same way since both run at 56×56 (3136 positions): about 3.70 billion versus 5.14 billion multiply-adds. The receptive field is 5×5 in both cases, so the larger kernel buys no extra context. Expressiveness favours the pair: it applies two ReLUs, so it can represent functions of the 5×5 patch that a single linear filter plus one ReLU cannot, and the factorisation acts as a structured constraint that tends to generalise well. The only argument for 5×5 is one fewer kernel launch and one fewer stored activation map, which rarely outweighs 39% more FLOPs. Keep the two 3×3 convolutions."
sources: [{"title": "Krizhevsky, Sutskever, Hinton: ImageNet Classification with Deep Convolutional Neural Networks", "url": "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks", "note": "Section 3 on ReLU, multi-GPU training and pooling; Section 4 on augmentation and dropout."}, {"title": "Simonyan & Zisserman: Very Deep Convolutional Networks (VGG)", "url": "https://arxiv.org/abs/1409.1556", "note": "Section 2.3 gives the argument for stacks of 3×3 convolutions; Table 1 lists the configurations."}, {"title": "Dive into Deep Learning: LeNet", "url": "https://d2l.ai/chapter_convolutional-neural-networks/lenet.html", "note": "LeNet-5 layer by layer with shapes, trained on Fashion-MNIST."}, {"title": "Dive into Deep Learning: AlexNet", "url": "https://d2l.ai/chapter_convolutional-modern/alexnet.html", "note": "What changed from LeNet to AlexNet and why it mattered."}]
practice: ["lenet-forward-pass", "vgg-convolution-block", "multichannel-2d-convolution"]
tags: ["LeNet", "AlexNet", "VGG"]
tools: ["PyTorch"]
---

## Why study old networks

LeNet, AlexNet and VGG are rarely deployed now, but each one introduced a design decision every later CNN inherits, and their failure modes motivated what came next: residual connections (03.04) and efficient blocks (03.05). They also make good mental benchmarks for parameter and compute counting. AI Research's TensorFlow lessons train CNNs on MNIST and CIFAR-10 in code; this lesson is about the architecture choices.

## LeNet-5: the template

LeCun et al.'s LeNet-5 read 32×32 grayscale digits for cheque processing:

- conv 5×5, 6 channels → 28×28×6, then subsample to 14×14×6
- conv 5×5, 16 channels → 10×10×16, then subsample to 5×5×16
- dense 400 → 120 → 84 → 10

About 60 thousand parameters. The key ideas were already there: local receptive fields, shared weights, spatial subsampling, and end-to-end training by backprop. It used tanh-like activations and average-style subsampling; the modern re-implementation swaps in ReLU and max pooling.

## AlexNet: scale, ReLU and regularisation

Krizhevsky, Sutskever and Hinton won ILSVRC-2012 with 15.3% top-5 error against 26.2% for the next entry. The architecture was LeNet's shape, much larger: 5 conv layers (the first 11×11 with stride 4 and 96 channels), 3 dense layers of 4096, 4096 and 1000 units, about 60 million parameters. What made it work is as important as the layers:

- **ReLU** instead of tanh: the paper reports reaching 25% training error on CIFAR-10 about six times faster.
- **Dropout** 0.5 in the first two dense layers, without which it overfit badly.
- **Data augmentation**: random 224×224 crops and flips from 256×256 images, plus PCA colour jitter.
- **GPUs**: trained for five to six days on two 3 GB GTX 580s, with the model split across them.
- **Overlapping max pooling** (3×3, stride 2) and local response normalisation; LRN was later dropped as BatchNorm superseded it.

## VGG: depth from small kernels

Simonyan and Zisserman asked what happens if you fix every conv to 3×3, stride 1, padding 1, and only vary depth. VGG-16 has 13 conv layers in 5 blocks (64, 128, 256, 512, 512 channels, each block ending in a 2×2 max pool) and the same 3-layer dense head. It placed second in ILSVRC-2014 classification. The argument for 3×3:

- Two stacked 3×3 layers have a 5×5 receptive field; three have 7×7 (03.01).
- Parameters for $C$ channels: three 3×3 layers $3 \cdot 9C^2 = 27C^2$ versus one 7×7 layer $49C^2$, 45% fewer.
- Each layer adds a ReLU, so the stack is more nonlinear than one big kernel.

The cost: VGG-16 has 138 million parameters and about 15.3 billion multiply-adds per 224×224 image. The first dense layer alone, $25{,}088 \times 4096 \approx 103$M weights, is three quarters of the parameter count.

| Model | Year | Depth (weight layers) | Parameters | ImageNet top-5 error |
| --- | --- | --- | --- | --- |
| LeNet-5 | 1998 | 5 | about 60k | (digits only) |
| AlexNet | 2012 | 8 | about 60M | 15.3% |
| VGG-16 | 2014 | 16 | about 138M | about 7.3% (ensemble entry) |

~~~dg-layers VGG-16 as five stages: each halves resolution and (up to 512) doubles channels; the dense head holds most parameters.
Input | 224×224×3
Stages 1-2 | 2 × conv3-64, pool; 2 × conv3-128, pool → 56×56×128
Stages 3-5 | 3 × conv3-256, 3 × conv3-512, 3 × conv3-512, each then pool → 7×7×512
*Dense head | 25,088 → 4096 → 4096 → 1000, about 124M of 138M parameters
~~~

```python
import torch.nn as nn

def vgg_block(c_in, c_out, n_convs):
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.MaxPool2d(2))

features = nn.Sequential(vgg_block(3, 64, 2), vgg_block(64, 128, 2), vgg_block(128, 256, 3),
                         vgg_block(256, 512, 3), vgg_block(512, 512, 3))   # -> 512×7×7
```

## What they could not do

Going deeper than about 20 layers in this plain style stopped helping: VGG-19 barely beat VGG-16, and training very deep VGG-style nets needed staged initialisation from shallower trained ones (He init, 01.05, and BatchNorm, 02.01, eased this). The next lesson shows the problem was optimisation, not capacity, and how residual connections fixed it. The huge dense heads were replaced by global average pooling (03.02), and the compute cost motivated the efficient designs of 03.05.

<div class="ex">
<div class="lab">Worked example</div>

**where VGG-16's parameters live**

Conv layers: 13 layers totalling about 14.7M parameters. The largest, 512→512 at 3×3, is $512 \cdot 512 \cdot 9 \approx 2.36$M each, and there are five of them.

Dense layers: $25{,}088 \times 4096 + 4096 \times 4096 + 4096 \times 1000 \approx 102.8 + 16.8 + 4.1 = 123.6$M.

So about 89% of parameters sit in the head, but over 99% of the multiply-adds are in the convolutions, mostly in the high-resolution early stages. Replacing the head with global average pooling plus one 512→1000 layer removes about 123M parameters and almost no compute.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

LeNet set the conv-pool-dense template, AlexNet proved it scales with ReLU, dropout, augmentation and GPUs, and VGG showed stacks of 3×3 convs are the efficient way to grow receptive fields; their dense heads and depth limits set up the designs that followed.

</div>
