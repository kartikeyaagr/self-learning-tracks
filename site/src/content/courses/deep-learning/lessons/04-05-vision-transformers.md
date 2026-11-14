---
id: "04.05"
course: "deep-learning"
module: "04"
slug: "vision-transformers"
title: "Vision Transformers"
keyIdeas: ["ViT cuts an image into non-overlapping patches (16×16 on a 224 image gives 196), flattens and linearly projects each one, and feeds the sequence to a standard transformer encoder.", "The patch projection is exactly a convolution with kernel size and stride equal to the patch size.", "A learned class token is prepended and its final state feeds the classifier; learned position embeddings are added because attention by itself ignores order. Changing resolution means interpolating them.", "With weaker built-in assumptions than CNNs, ViTs need either very large pretraining sets or strong augmentation and distillation (DeiT) to match CNNs.", "Global attention costs $O(N^2)$ in tokens. Swin restricts attention to 7×7 windows, shifts the windows every other layer, and merges patches between stages, giving linear cost and a CNN-like feature pyramid."]
question: "You have a ViT-B/16 pretrained at 224×224 and want to fine-tune it on 384×384 images. What breaks if you just feed the larger images, what do you change, and roughly how much more attention compute does each layer need?"
answer: "The patch embedding itself still works, since it is a stride-16 convolution, but the token count changes from 14×14 = 196 to 24×24 = 576 patches, and the learned position embedding table only has entries for 196 patch positions plus the class token, so the addition fails or misaligns. Fix it by reshaping the 196 patch position embeddings into a 14×14 grid, interpolating (bicubic) to 24×24, flattening, and keeping the class token's embedding as it is. Everything else (attention weights, MLPs) is resolution-agnostic. Attention cost scales with tokens squared: (577/197)² ≈ 8.6× per layer, while the MLP cost scales linearly, about 2.9×. Expect better accuracy from finer effective detail, at much higher memory and compute."
sources: [{"title": "An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)", "url": "https://arxiv.org/abs/2010.11929", "note": "The ViT paper; section 3.1 for the architecture and 3.2 for resolution changes and position-embedding interpolation."}, {"title": "Training data-efficient image transformers (DeiT)", "url": "https://arxiv.org/abs/2012.12877", "note": "How to train ViTs on ImageNet-1k alone; the ablation table shows which augmentations matter."}, {"title": "Swin Transformer (Liu et al., 2021)", "url": "https://arxiv.org/abs/2103.14030", "note": "Shifted-window attention and patch merging; figure 2 explains the shift."}, {"title": "Dive into Deep Learning: Transformers for Vision", "url": "https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html", "note": "Compact ViT implementation from patch embedding to classifier."}]
practice: ["vit-patchify", "vision-transformer-patch-embedding", "2d-sinusoidal-positional-embedding", "window-partition-and-reverse", "vit-multi-head-self-attention"]
tags: ["vision transformer", "patch embedding"]
tools: ["NumPy", "PyTorch"]
---

## Images as token sequences

Lesson 05.04 and the AI Engineering course cover how a transformer block works. The question here is narrower: how do you turn an image into the token sequence a transformer expects, and what changes because the tokens are patches rather than words?

The Vision Transformer (ViT, 2020) answer is blunt. Split a $224 \times 224 \times 3$ image into $16 \times 16$ patches: $224/16 = 14$ per side, 196 patches, each with $16 \times 16 \times 3 = 768$ values. Flatten each patch and multiply by a learned $768 \times D$ matrix. That linear map on non-overlapping patches is identical to a convolution with kernel size 16 and stride 16, and that is how it is usually implemented.

~~~dg-flow ViT-B/16 input pipeline, shapes per image.
Image 224×224×3 -(16×16 patches)-> 196 × 768 -(linear)-> 196 × D -(+ CLS, + positions)-> 197 × D -> *Encoder × 12 -(CLS state)-> Class head
~~~

```python
import numpy as np
def patchify(img, p=16):                       # img: (H, W, C)
    H, W, C = img.shape
    x = img.reshape(H // p, p, W // p, p, C)   # split both spatial axes
    x = x.transpose(0, 2, 1, 3, 4)             # (rows, cols, p, p, C)
    return x.reshape(-1, p * p * C)            # (num_patches, p*p*C), row-major order
```

The transpose is the important line: reshaping straight to `(-1, p*p*C)` would mix rows from different patches, the same class of silent bug as the HWC/CHW mix-up in lesson 04.01.

## Class token and position embeddings

ViT prepends a learned vector, the class token, giving 197 tokens. After the last layer, the class token's state goes through a small head to produce logits. It works because the class token attends to every patch in every layer, so it can act as a learned pooling. Global average pooling over the patch tokens works about as well and is common in later models.

Self-attention is permutation-equivariant: shuffle the patches and the outputs shuffle the same way. Position information must be injected, and ViT adds a learned embedding per position (197 vectors of size $D$). Fixed 2-D sine-cosine embeddings, which encode row and column separately, perform similarly and need no training.

Learned position embeddings tie the model to a grid size. Fine-tuning at 384×384 with the same patch size gives a 24×24 grid (576 patches) instead of 14×14. The standard fix is to reshape the 196 patch embeddings to a 14×14×$D$ grid, resize bilinearly or bicubically to 24×24, flatten, and keep the class token's embedding unchanged.

## Data appetite and inductive bias

A CNN builds in locality (small kernels) and translation equivariance (weight sharing). A ViT builds in very little: only the patch grid, and the MLP layers acting per token. It has to learn from data that nearby patches are related. The original paper found ViT trailed ResNets when trained on ImageNet-1k alone (1.3M images) and overtook them only with pretraining on 14M to 300M images. DeiT then showed a data-efficient recipe for ImageNet-1k alone: heavy augmentation (RandAugment, mixup, CutMix from lesson 04.01), stochastic depth, repeated augmentation, and distillation from a CNN teacher through an extra distillation token.

| Model | Width $D$ | Layers | Heads | Params |
| --- | --- | --- | --- | --- |
| ViT-S/16 | 384 | 12 | 6 | 22M |
| ViT-B/16 | 768 | 12 | 12 | 86M |
| ViT-L/16 | 1024 | 24 | 16 | 307M |

## Cost and windowed attention

Attention cost grows with the square of the token count. Halving the patch size from 16 to 8 on a 224 image quadruples the tokens (196 to 784) and multiplies attention cost by 16. Dense tasks like detection and segmentation want high resolution and fine patches, so global attention becomes the bottleneck.

The Swin Transformer's fixes:

- **Window attention**: tokens attend only within non-overlapping $M \times M$ windows ($M = 7$, so 49 tokens), making cost linear in image area.
- **Shifted windows**: every other layer shifts the window grid by $M/2$ so information crosses window borders; a cyclic shift plus a mask keeps the window count constant.
- **Patch merging**: between stages, concatenate each 2×2 group of tokens and project, halving resolution and doubling channels. The four stages have strides 4, 8, 16 and 32, the same pyramid a ResNet provides, so Swin drops into FPN-based detectors and U-Net-style segmenters directly.

<div class="ex">
<div class="lab">Worked example</div>

**global versus window attention at Swin's first stage**

A 224 image with 4×4 patches gives a 56×56 grid, 3,136 tokens. Global attention scores $3136^2 \approx 9.8$ million pairs per head per layer.

With 7×7 windows there are $8 \times 8 = 64$ windows of 49 tokens: $64 \times 49^2 = 153{,}664$ pairs, 64 times fewer. Doubling the image side multiplies the window cost by 4 but the global cost by 16.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a ViT is a standard transformer encoder on patch tokens, a class token and position embeddings. It trades CNN inductive bias for flexibility, so it needs data or strong regularisation, and windowed variants like Swin restore linear cost and a multi-scale pyramid for dense tasks.

</div>
