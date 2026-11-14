---
id: "04.05"
course: "deep-learning"
module: "04"
slug: "vision-transformers"
title: "Vision Transformers"
keyIdeas: ["ViT cuts an image into non-overlapping patches (16 × 16 on a 224 image gives 14 × 14 = 196), flattens each one and projects it with a shared learned matrix into a token, then feeds the sequence to a standard transformer encoder.", "The patch projection is exactly a convolution with kernel size and stride equal to the patch size, which is how it is implemented; flattening by hand needs a transpose or pixels from different patches get mixed.", "A learned class token is prepended and its final state feeds the classifier. Attention by itself ignores order, so learned position embeddings are added; changing resolution means interpolating them to the new grid.", "With weaker built-in assumptions than CNNs (no locality, no weight-shared kernels), ViTs need either very large pretraining sets or strong augmentation and distillation (DeiT) to match CNNs.", "Global attention costs $O(N^2)$ in tokens, so the fourth power of the image side. Swin restricts attention to 7 × 7 windows, shifts the windows every other layer, and merges patches between stages, giving linear cost and a CNN-like feature pyramid."]
question: "You have a ViT-B/16 pretrained at 224×224 and want to fine-tune it on 384×384 images of 4,000 labelled X-ray crops. What breaks if you just feed the larger images, what do you change, roughly how much more compute does each layer need, and what would you do to avoid overfitting on such a small dataset?"
answer: "The patch embedding itself still works, since it is a stride-16 convolution: it now produces a 24×24 grid, 576 patch tokens (577 with the class token) instead of 196 (197). The learned position table only has 197 entries, so adding it fails with a shape mismatch (or, if code silently truncates, misaligns positions). Fix it by taking the 196 patch position embeddings, reshaping them to a 14×14×768 grid, interpolating (bicubic or bilinear) to 24×24, flattening to 576 vectors, and keeping the class token's embedding unchanged. Everything else (attention and MLP weights, layer norms) does not depend on the number of tokens and is reused as is. Compute: the attention scores scale with tokens squared, (577/197)² ≈ 8.6× per layer, while projections and MLPs scale linearly, about 2.9×; because the linear parts dominate ViT-B at these sizes (attention scores are roughly 4% of the work at 224 and 13% at 384), total compute rises by a bit over 3×, and activation memory for attention by about 8.6×. For 4,000 images, never train from scratch: start from the pretrained weights, use a smaller learning rate with warm-up, AdamW with weight decay, augmentation suited to X-rays (small crops, flips if anatomy allows, mild intensity changes rather than colour jitter), stochastic depth, and early stopping on a validation split. Consider first fine-tuning at 224 and switching to 384 for a few final epochs, and compare against a pretrained ResNet baseline at the same resolution."
sources: [{"title": "An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)", "url": "https://arxiv.org/abs/2010.11929", "note": "The ViT paper; section 3.1 for the architecture and 3.2 for resolution changes and position-embedding interpolation."}, {"title": "Training data-efficient image transformers (DeiT)", "url": "https://arxiv.org/abs/2012.12877", "note": "How to train ViTs on ImageNet-1k alone; the ablation table shows which augmentations matter."}, {"title": "Swin Transformer (Liu et al., 2021)", "url": "https://arxiv.org/abs/2103.14030", "note": "Shifted-window attention and patch merging; figure 2 explains the shift."}, {"title": "Dive into Deep Learning: Transformers for Vision", "url": "https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html", "note": "Compact ViT implementation from patch embedding to classifier."}]
practice: ["vit-patchify", "vision-transformer-patch-embedding", "2d-sinusoidal-positional-embedding", "window-partition-and-reverse", "vit-multi-head-self-attention"]
tags: ["vision transformer", "patch embedding"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- How a Vision Transformer (ViT) turns a picture into a sequence of tokens: patches, a linear projection, a class token and position embeddings.
- Why the patch projection is exactly a convolution with kernel size and stride equal to the patch size, and how to patchify without scrambling pixels.
- Why attention needs position embeddings at all, and how to adapt them when you change the input resolution.
- Why ViTs are hungrier for data than CNNs, and what DeiT-style training does about it.
- Why global attention gets expensive on large images, and how the Swin Transformer's shifted windows and patch merging bring the cost back to linear.

*Before you start:* you need convolution and stride (03.01), residual blocks (03.04) and the image tensor layouts and augmentations of 04.01. It helps to know what self-attention computes; lesson 05.04 covers the full transformer block, and the short recap in Step 3 below is enough to follow this lesson.

## From CNNs to transformers

Every model in this module so far has been a CNN, where depth slowly grows the area each feature can see; the U-Net of 04.04 needed a whole encoder-decoder to combine local detail with global context. In a transformer every token can look at every other token in the very first layer. The question here is narrow: how do you turn an image into the token sequence a transformer expects, and what changes because the tokens are patches rather than words?

## The idea in one picture

The Vision Transformer (ViT, Dosovitskiy et al., 2020) answer is blunt: cut the image into squares and treat each square as a word.

Take a $224 \times 224$ colour image and a patch size of 16 pixels:

| Quantity | Value | How you get it
| --- | --- | --- |
| Patches per side | 14 | $224 / 16$
| Number of patches | 196 | $14 \times 14$
| Numbers in one patch | 768 | $16 \times 16 \times 3$ colours
| Tokens fed to the encoder | 197 | 196 patches + 1 class token
| Token width $D$ (ViT-Base) | 768 | chosen by the architecture

~~~dg-fig 04.05-patches.svg
A toy picture on the 14 × 14 patch grid. Patches are numbered row by row (0 to 13 in the top row, 14 starts the next row), and the sequence the transformer sees is a class token followed by patches 0 to 195.
~~~

A **patch** is one $16 \times 16$ square of the image. Each patch is flattened into 768 numbers and multiplied by a learned matrix, giving one **token**: a vector of length $D$ that the transformer treats like a word embedding. A learned extra vector, the **class token** (CLS), sits at the front, and its final state becomes the class scores. Nothing in the encoder itself knows it is looking at an image.

~~~dg-flow ViT-B/16 input pipeline, shapes per image.
Image 224×224×3 -(16×16 patches)-> 196 × 768 -(linear)-> 196 × D -(+ CLS, + positions)-> 197 × D -> *Encoder × 12 -(CLS state)-> Class head
~~~

## Step 1: cut the image into patches

Split height and width into blocks of $p$ pixels and list the blocks in reading order. For an image of height $H$, width $W$ and $C$ channels, the number of patches is

$$N = \frac{H}{p} \cdot \frac{W}{p},$$

and each patch holds $p \cdot p \cdot C$ numbers. With $H = W = 224$, $p = 16$, $C = 3$ you get $N = 196$ patches of 768 numbers each. $H$ and $W$ must be divisible by $p$; otherwise you crop or pad first.

```python
import numpy as np
def patchify(img, p=16):                       # img: (H, W, C)
    H, W, C = img.shape
    x = img.reshape(H // p, p, W // p, p, C)   # split both spatial axes
    x = x.transpose(0, 2, 1, 3, 4)             # (rows, cols, p, p, C)
    return x.reshape(-1, p * p * C)            # (num_patches, p*p*C), row-major order
```

The transpose is the important line. Reshaping straight to `(-1, p*p*C)` also gives 196 rows of 768 numbers, but each row mixes pixel strips from different patches. Nothing crashes; the model just learns worse, like the HWC/CHW mix-up in lesson 04.01.

## Step 2: project each patch to a token

Each flattened patch $x_i$ (length 768) is mapped to a token by one shared matrix:

$$z_i = E\,x_i + b, \qquad E \in \mathbb{R}^{D \times 768},\; b \in \mathbb{R}^{D}.$$

Here $E$ is the **patch embedding** matrix, $b$ its bias, and $z_i$ the resulting token. The same $E$ is used for every patch.

Now look from the CNN side. A convolution with kernel size 16 and stride 16 visits non-overlapping $16 \times 16$ windows, exactly the patches, and each of its $D$ output channels is a weighted sum of the $16 \times 16 \times 3$ inputs: one row of $E$. So the patch embedding *is* `Conv2d(3, D, kernel_size=16, stride=16)`, which is how libraries implement it, and the patch order cannot go wrong.

<div class="ex">
<div class="lab">Worked example</div>

**count everything in ViT-B/16 by hand**

Patches: $224 / 16 = 14$ per side, so $14 \times 14 = 196$ patches, each with $16 \times 16 \times 3 = 768$ values.

Patch embedding parameters: a $768 \times 768$ matrix plus 768 biases, $589{,}824 + 768 = 590{,}592$. (PyTorch reports exactly 590,592 for the `Conv2d` below.)

Sequence length: 196 patches plus the class token gives 197 tokens. The learned position table therefore has $197 \times 768 = 151{,}296$ numbers.

Attention scores: each head compares every token with every token, $197^2 = 38{,}809$ scores per head per layer. With 12 heads that is $12 \times 38{,}809 = 465{,}708$ scores in one layer.

Changing to 8 × 8 patches gives $28 \times 28 = 784$ patch tokens: 4 times as many tokens and about $4^2 = 16$ times as many scores.

</div>

## Step 3: a short recap of self-attention

A transformer layer mixes information between tokens with **self-attention**. Each token $z_i$ is turned into a query $q_i$, a key $k_i$ and a value $v_i$ by three learned matrices. Token $i$ then takes a weighted average of all values, with weights from how well its query matches each key:

$$\text{out}_i = \sum_j \text{softmax}_j\!\Big(\frac{q_i \cdot k_j}{\sqrt{d}}\Big)\, v_j,$$

where $d$ is the width of one attention head and the softmax runs over $j$ so the weights sum to 1. Attention is followed by a small per-token MLP; residual connections and layer normalisation wrap both. Several heads run in parallel.

Two facts matter for images. Every token sees every other token in every layer, so the top-left patch can use the bottom-right one in layer 1. And the formula treats the tokens as a *set*: nothing in it says which came first.

## Step 4: add a class token and position embeddings

**The class token.** ViT prepends one learned vector, giving 197 tokens. It holds no pixels and starts identical for every image, but it attends to all patches in all layers, so it becomes a learned summary that a linear head turns into logits. Averaging the patch tokens instead (**global average pooling**) works about as well and is common in later models.

**Why positions are needed.** Self-attention is **permutation-equivariant**: shuffle the input tokens and the outputs come out shuffled the same way, otherwise unchanged (the code below checks this). A ViT alone could not tell an image from its jumbled patches. So ViT adds a **position embedding**, one learned $D$-dimensional vector per slot (197 of them), to each token before the encoder.

A fixed alternative needs no training: **2-D sine-cosine embeddings**, where half the dimensions encode the row with sines and cosines of several frequencies and half the column. They perform similarly to learned ones.

~~~dg-fig 04.05-posemb.svg
Cosine similarity between one 2-D sine-cosine position embedding (outlined) and every other position. Similarity is highest at the query and its neighbours, with a cross shape because half the dimensions encode the row and half the column. After bicubic resizing to a 24 × 24 grid, the same local pattern survives.
~~~

**Changing resolution.** Learned position embeddings tie the model to one grid size. Fine-tune at $384 \times 384$ with the same patch size and you get a $24 \times 24$ grid (576 patches), but the table has only 196 patch entries. The standard fix, also shown in the figure: reshape the 196 patch embeddings into a $14 \times 14 \times D$ grid, resize it bilinearly or bicubically to $24 \times 24$, flatten back to 576 vectors, and keep the class token's embedding unchanged. Neighbouring positions have similar embeddings, so the interpolated table is a sensible start.

## Vision Transformers in code

A complete ViT-Base forward pass in PyTorch, plus two checks; printed values are from a real run.

```python
import torch, torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img=224, p=16, c=3, d=768):
        super().__init__()
        self.proj = nn.Conv2d(c, d, kernel_size=p, stride=p)   # = linear map on each patch
        self.n = (img // p) ** 2
    def forward(self, x):                      # x: (B, 3, 224, 224)
        x = self.proj(x)                       # (B, 768, 14, 14)
        return x.flatten(2).transpose(1, 2)    # (B, 196, 768): one row per patch

class ViT(nn.Module):
    def __init__(self, d=768, depth=12, heads=12, classes=1000):
        super().__init__()
        self.embed = PatchEmbed(d=d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, self.embed.n + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, classes)
    def forward(self, x):
        t = self.embed(x)                                        # (B, 196, D)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)   # (B, 197, D)
        t = self.encoder(t + self.pos)                           # (B, 197, D)
        return self.head(self.norm(t[:, 0]))                     # class token -> logits

m = ViT(); x = torch.randn(2, 3, 224, 224)
print(m.embed(x).shape)                               # torch.Size([2, 196, 768])
print(m(x).shape)                                     # torch.Size([2, 1000])
print(sum(p.numel() for p in m.parameters()) / 1e6)   # 86.57 (million)

# check 1: the conv equals "flatten each patch, multiply by a matrix"
pe = m.embed
p = x.unfold(2, 16, 16).unfold(3, 16, 16)              # (B, 3, 14, 14, 16, 16)
p = p.permute(0, 2, 3, 1, 4, 5).reshape(2, 196, 768)   # patches in row-major order
W = pe.proj.weight.reshape(768, -1)                    # (D, 3*16*16)
print(torch.allclose(p @ W.T + pe.proj.bias, pe(x), atol=1e-4))   # True

# check 2: attention alone ignores order (shuffle in = shuffle out)
enc = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True).eval()
t = torch.randn(1, 10, 64); perm = torch.randperm(10)
with torch.no_grad():
    print(torch.allclose(enc(t)[:, perm], enc(t[:, perm]), atol=1e-5))  # True
```

The 86.57 million parameters match the published ViT-B/16 size (86M). In practice you load pretrained weights (`timm`, `torchvision`) rather than train from scratch, for reasons the next step explains.

## Step 5: data appetite and inductive bias

An **inductive bias** is an assumption built into the architecture. A CNN builds in two strong ones: **locality** (a $3 \times 3$ kernel only looks at neighbours) and **translation equivariance** (the same kernel slides everywhere, so a cat shifted left gives features shifted left). A ViT builds in very little: only the patch grid and the per-token MLP. It must *learn* from data that nearby patches are related.

The ViT paper found that ViTs trailed ResNets when trained on ImageNet-1k alone (1.3M images) and overtook them only after pretraining on 14M (ImageNet-21k) to 300M (JFT) images. Weak assumptions cost data; with enough data they become flexibility.

You can see this at toy scale: a small CNN (31k parameters) against a small ViT (58k parameters, 2 × 2 pixel patches, 16 tokens) on scikit-learn's 8 × 8 digits, with growing training sets.

~~~dg-fig 04.05-data.svg
Toy experiment, two seeds per point (shaded band = min to max), 400 held-out test images. With 60 training images the CNN's built-in locality wins by 23 points; with 1,397 the gap is under 3 points.
~~~

Both models get 600 AdamW steps. With 60 training images the CNN reaches 87.4% test accuracy and the ViT, despite having twice the parameters, only 64.4%. With all 1,397 images the scores are 99.0% and 96.4%. One caution from the same runs: adding random 1-pixel shifts (half a patch) lowered the ViT to about 75% within this short budget, because augmentation makes the task harder and only pays off with long training (DeiT trains for 300 epochs).

**DeiT** (Touvron et al., 2020) showed how to train ViTs well on ImageNet-1k alone. The recipe is mostly regularisation: augmentation (RandAugment, mixup and CutMix from lesson 04.01), random erasing, stochastic depth (randomly skipping residual blocks during training), repeated augmentation, and **distillation** from a CNN teacher through an extra **distillation token** that is trained to match the teacher's predictions. 

| Model | Width $D$ | Layers | Heads | Params
| --- | --- | --- | --- | --- |
| ViT-S/16 | 384 | 12 | 6 | 22M
| ViT-B/16 | 768 | 12 | 12 | 86M
| ViT-L/16 | 1024 | 24 | 16 | 307M

"/16" is the patch size; each MLP is $4D$ wide.

## Step 6: cost and windowed attention

Attention compares every token with every other, so its cost grows with the **square** of the token count $N$. For a fixed patch size $N$ grows with the square of the image side, so attention cost grows with its **fourth power**.

~~~dg-fig 04.05-cost.svg
Attention pairs per head per layer as the image grows (log-log axes). Global attention lines climb with slope 4; Swin's window attention climbs with slope 2, so at 224 pixels with 4 × 4 patches it needs 0.15 million pairs instead of 9.8 million.
~~~

Detection and segmentation (04.02 to 04.04) want high resolution and fine patches, so global attention becomes the bottleneck. The **Swin Transformer** (Liu et al., 2021) makes three changes:

- **Window attention**: tokens attend only within non-overlapping $M \times M$ windows ($M = 7$, 49 tokens). Each window costs a fixed $49^2$ scores and the window count grows with image area, so cost is linear in area.
- **Shifted windows**: every other layer moves the window grid by $\lfloor M/2 \rfloor$ tokens so information crosses the old window borders. A cyclic shift of the token map plus an attention mask keeps the window count constant.
- **Patch merging**: between stages, concatenate each 2 × 2 group of neighbouring tokens ($4C$ channels) and project to $2C$. Resolution halves and channels double, like a strided convolution.

~~~dg-fig 04.05-swin.svg
A smaller 8 × 8 grid with 4 × 4 windows for readability. In layer l the marked token can only attend inside its own window; in layer l+1 the borders move by 2 tokens, so its new window mixes tokens from all four old windows. Stacking both lets information spread across the whole image.
~~~

Swin starts with $4 \times 4$ patches, and its four stages have strides 4, 8, 16 and 32: the same feature pyramid a ResNet provides, so Swin drops into FPN-based detectors (04.02) and U-Net-style segmenters (04.04) in place of the CNN backbone. It also uses a learned relative position bias inside each window instead of ViT's absolute table.

<div class="ex">
<div class="lab">Worked example</div>

**global versus window attention at Swin's first stage**

A 224 image with 4 × 4 patches gives a $56 \times 56$ grid, $3{,}136$ tokens. Global attention scores $3136^2 = 9{,}834{,}496 \approx 9.8$ million pairs per head per layer.

With 7 × 7 windows there are $(56/7)^2 = 8 \times 8 = 64$ windows of 49 tokens: $64 \times 49^2 = 64 \times 2{,}401 = 153{,}664$ pairs.

Ratio: $9{,}834{,}496 / 153{,}664 = 64$, which is exactly the number of windows. Doubling the image side multiplies the window cost by 4 but the global cost by 16.

</div>

## Checking that it works

- **Shapes first.** Expect `(B, N, D)` after the patch embedding, with $N = (H/p)(W/p)$, and $N + 1$ tokens after the class token.
- **Patch order.** Patchify then un-patchify one image; it must match the original exactly.
- **Position embeddings.** After training, plot similarity maps of learned position embeddings as in the figure above. Bright neighbourhoods are healthy; a random-looking map means positions were never learned.
- **Baseline.** Compare against a ResNet of similar cost with the same augmentation; losing on a small dataset is expected, not a bug.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Training works but accuracy is far below a CNN | ViT trained from scratch on a small dataset | start from pretrained weights; add DeiT-style augmentation, stochastic depth, distillation
| Shape error or nonsense after changing image size | position table built for a different grid | interpolate the patch position embeddings to the new grid; keep the class entry
| Slightly worse than expected, no error | patches flattened without the transpose | use `Conv2d` patch embedding or check with patchify/unpatchify round trip
| Out of memory at high resolution | quadratic attention in the token count | larger patches, windowed attention (Swin), or a CNN or hybrid backbone
| Loss spikes or diverges early | no warm-up, learning rate too high | linear warm-up, AdamW with weight decay, gradient clipping

## Going deeper

**Where the compute really goes.** Per layer, the four $D \times D$ projections (queries, keys, values, output) cost about $4ND^2$ multiply-adds and the MLP about $8ND^2$, while the two attention matrix products cost about $2N^2D$. The quadratic part is smaller whenever $N < 6D$. For ViT-B at 224 ($N = 197$, $D = 768$) that is about 1.39 billion versus 0.06 billion multiply-adds per layer, so attention scores are only about 4% of the work; at 384 resolution ($N = 577$: 2.9 times the tokens, 8.6 times the scores) about 13%. The quadratic term dominates only with thousands of tokens, the regime of dense prediction on large images, where its $N^2$ memory also hurts first.

**What ViTs learn.** The ViT paper reports that in early layers some heads already attend far away while others stay local, and that the learned position embeddings develop the same row-and-column structure as the sine-cosine figure, from data alone.

**Descendants.** Hybrid ViTs run a ResNet first and use its feature-map cells as patches. Later work scaled plain ViTs with self-supervised pretraining such as masked autoencoders, and patch tokens are how images enter multimodal language models.

## Key terms

| Term | Meaning
| --- | --- |
| Patch | A $p \times p$ square of the image, flattened into $p \cdot p \cdot C$ numbers
| Patch embedding | The shared linear map from a flattened patch to a $D$-dimensional token; equal to a convolution with kernel = stride = $p$
| Class token (CLS) | A learned extra token whose final state feeds the classifier
| Position embedding | A vector added to each token so the model knows where its patch came from
| Permutation-equivariant | Shuffling the input tokens shuffles the outputs the same way and changes nothing else
| Inductive bias | Assumptions built into an architecture, such as locality in CNNs
| Distillation token | DeiT's extra token trained to match a teacher network's predictions
| Shifted window attention | Swin's local attention in $M \times M$ windows whose grid moves every other layer
| Patch merging | Concatenating 2 × 2 neighbouring tokens and projecting, halving resolution

<div class="take">
<div class="lab">Key takeaway</div>

a ViT is a standard transformer encoder fed with patch tokens, a class token and position embeddings. It trades CNN inductive bias for flexibility, so it needs lots of data or strong regularisation, and windowed variants like Swin restore linear cost and a multi-scale pyramid for dense tasks.

</div>
