---
id: "04.04"
course: "deep-learning"
module: "04"
slug: "segmentation-and-u-net"
title: "Segmentation and U-Net"
keyIdeas: ["Semantic segmentation predicts a class for every pixel: the network outputs a $C \\times H \\times W$ map of logits at the input's resolution, and the label map is the arg-max at each pixel.", "Encoder-decoder networks downsample to gain context and upsample to recover resolution. Detail lost in downsampling cannot be recovered by upsampling alone: a thin structure averaged away at stride 32 is simply gone.", "U-Net's skip connections concatenate encoder features into the decoder at each resolution, so the decoder gets both coarse 'what' and fine 'where'. In a small experiment, removing the skips turned sharp masks into rounded blobs.", "Pixel cross-entropy is dominated by the majority class when foreground is tiny. Dice loss measures overlap directly and ignores true-negative background, so most recipes combine cross-entropy with Dice.", "Evaluate with per-class IoU averaged over classes (mIoU), accumulated over the whole validation set, not pixel accuracy. Dice and IoU are related by $D = 2J/(1+J)$, so never compare one with the other."]
question: "You train a U-Net to segment small lesions that cover about 0.3% of pixels, using pixel-wise cross-entropy. Validation pixel accuracy is 99.7%, but almost every predicted mask is empty. Explain why, and what you would change in the loss and the evaluation."
answer: "Predicting background everywhere is already 99.7% accurate, and with plain cross-entropy the loss and its gradient are averaged over all pixels. If the model outputs a lesion probability of 0.003 at every pixel, the average cross-entropy is only about 0.020, so the few lesion pixels contribute almost nothing and the model settles at the trivial solution. Pixel accuracy hides this, because it is dominated by true-negative background. Change the loss so that overlap with the rare class drives training: add a soft Dice loss on the lesion class, whose value ignores true-negative background pixels (an empty prediction scores a Dice loss near 1, its worst value), or use class-weighted or focal cross-entropy; the usual recipe is cross-entropy plus Dice with equal weight. Compute Dice over the whole batch and keep its smoothing constant small relative to the lesion size, otherwise the constant dominates the ratio. Sampling training crops that contain lesions also raises the foreground fraction the model sees. Change the evaluation from pixel accuracy to per-class IoU or Dice on the lesion class (and mIoU over classes), accumulated over the validation set from a confusion matrix, and also count images where a present lesion gets an empty mask. An all-background predictor scores an IoU of 0 on the lesion class, which exposes the failure immediately."
sources: [{"title": "U-Net (Ronneberger et al., 2015)", "url": "https://arxiv.org/abs/1505.04597", "note": "The original architecture; figure 1 is the canonical diagram and section 3 covers the overlap-tile and weighting tricks."}, {"title": "Fully Convolutional Networks for Semantic Segmentation", "url": "https://arxiv.org/abs/1411.4038", "note": "How classifiers become dense predictors, and the skip-fusion idea U-Net builds on."}, {"title": "V-Net (Milletari et al., 2016)", "url": "https://arxiv.org/abs/1606.04797", "note": "Introduced the soft Dice loss for volumetric medical segmentation; section 3 derives the gradient."}, {"title": "Dive into Deep Learning: Fully Convolutional Networks", "url": "https://d2l.ai/chapter_computer-vision/fcn.html", "note": "Hands-on FCN with transposed-convolution upsampling."}]
practice: ["tiny-u-net-forward-pass", "u-net-skip-connection", "dice-loss", "mean-iou-for-segmentation", "transposed-convolution"]
tags: ["segmentation", "U-Net", "Dice"]
tools: ["NumPy"]
---

## What you will learn

- What segmentation is and its three flavours (semantic, instance, panoptic).
- How a classification CNN becomes a per-pixel predictor, and why its coarse output looks blobby.
- How U-Net combines an encoder, a decoder and skip connections, with the tensor shapes at each step.
- Why pixel cross-entropy fails on tiny objects, and how the Dice loss fixes it.
- How to compute per-class IoU and mean IoU, by hand and in NumPy.

*Before you start:* you need convolution, pooling and transposed convolution (lessons 03.01 and 03.02), batch normalisation (02.01), and cross-entropy for classes (01.03). IoU was introduced for boxes in 04.02; here it is applied to pixel sets.

## From boxes to pixels

The last two lessons drew a box around each object. A box around a tumour tells a surgeon nothing about its edge. **Segmentation** gives the finest answer: a label for every pixel. It comes in three flavours:

- **Semantic segmentation**: one class per pixel. Two touching cars are one "car" region.
- **Instance segmentation**: a separate mask for each object (Mask R-CNN adds a mask head to the detector of lesson 04.02).
- **Panoptic segmentation**: both at once. Countable **things** (people, cars) get instance masks; uncountable **stuff** (road, sky, grass) gets semantic regions.

This lesson covers the semantic case and the **U-Net** architecture, still the default for medical imaging and the denoiser shape inside the diffusion models of lesson 07.04.

## The idea in one picture

Take a 48×48 grey image with bright circles and squares on a noisy background. Both shapes have the same brightness, so one pixel cannot tell you which shape it belongs to: you must look around it. The job is a map of the same size where every pixel says *background*, *circle* or *square*.

~~~dg-fig 04.04-task.svg
Left: the input, one grey channel. Middle: the target label map, one class per pixel. Right: the map predicted by a small U-Net trained for 400 steps in this lesson; it gets both the shape (circle or square) and the edges almost right.
~~~

Notice three things:

- The output has the **input's height and width**: one score vector per pixel, a $C \times H \times W$ tensor of **logits** for $C$ classes. The predicted **label map** is the arg-max over $C$ at each pixel.
- Each pixel needs **context** ("is this part of something round?") *and* **precision** ("where exactly is the edge?"). These pull in opposite directions; the architecture is about getting both.
- The training target is an integer **mask** storing each pixel's class index. Such masks are slow to label by hand, so datasets are small.

## Step 1: make the classifier fully convolutional

A classification CNN (lesson 03.03) ends with global pooling and a fully connected layer, which throw away *where* things are. The **fully convolutional network (FCN)** idea: delete those layers and keep everything convolutional, so the output stays a spatial map. A final **1×1 convolution** turns the $K$ feature channels at each location into $C$ class logits, like a fully connected layer applied at every position.

The catch is resolution. A ResNet with **total stride** 32 (five halvings) turns a 512×512 image into a 16×16 map, each cell speaking for a 32×32 patch. Resize that prediction back to 512×512 and you get blobs.

~~~dg-fig 04.04-stride.svg
A target mask (a disk, a thin bar and a small dot) squeezed to a coarse map by average pooling and then resized back with bilinear interpolation. At stride 8 all three survive; at stride 16 the 10-pixel-wide bar and the dot have vanished; at stride 32 even the disk's outline turns lumpy. No resize can bring them back.
~~~

The key fact of this lesson: **upsampling cannot recover information that downsampling destroyed**. A thin structure averaged into its neighbours is simply not in the coarse map any more. Measured against the target, the IoU (overlap score, defined in step 5) falls from 0.95 at stride 8 to 0.79 at stride 16 and 0.73 at stride 32. The fix must bring in fine detail from *before* the downsampling.

## Step 2: two ways to upsample

Lesson 03.02 gave two tools for growing a map:

- **Transposed convolution**: learned. Each input value scales the kernel, which is "stamped" onto the output at `stride` spacing; overlapping stamps add up.
- **Resize, then convolve**: a fixed bilinear or nearest resize, then a 3×3 convolution that learns to refine it.

~~~dg-fig 04.04-checker.svg
An all-ones input through each upsampler (weights all 1, or a plain average). Kernel 3, stride 2 gives a repeating 1, 2, 4 pattern: some output pixels get one stamp, some two, some four. Kernel 4 overlaps evenly; resize-then-convolve is flat.
~~~

This uneven overlap causes **checkerboard artefacts** whenever the kernel size is not divisible by the stride. The network can learn to cancel it but often does not fully succeed, so resize plus 3×3 convolution is the common modern choice, and transposed convolutions usually take kernel 4 with stride 2.

## Step 3: U-Net, an encoder, a decoder and bridges between them

U-Net (Ronneberger et al., 2015) has two halves drawn as the two arms of a letter U.

- The **encoder** (contracting path) repeats two 3×3 convolutions then a 2× max-pool, doubling channels at each level. The map shrinks but each position sees a larger area, so features answer **"what"**.
- The **decoder** (expanding path) mirrors it: upsample 2×, two 3×3 convolutions, halve the channels.
- The **skip connections** copy the encoder's feature map at each resolution and **concatenate** it (along the channel axis) onto the decoder's upsampled map. These early features still know **"where"**.
- A final **1×1 convolution** maps the last feature channels to $C$ logits per pixel.

~~~dg-layers A four-level U-Net on a 256×256 input. Each encoder level's output is copied across to the decoder level at the same resolution.
Level 1 | 256×256, 64 channels, skip to decoder level 1
Level 2 | 128×128, 128 channels, skip to decoder level 2
Level 3 | 64×64, 256 channels, skip to decoder level 3
*Bottleneck | 32×32, 512 channels: widest context, coarsest position
Decoder | upsample, concatenate skip, two 3×3 convs, repeat to 256×256, then 1×1 conv to C logits
~~~

One decoder step: the 32×32×512 bottleneck is upsampled to 64×64×512; concatenating the level-3 skip (64×64×256) gives 512 + 256 = 768 channels; two 3×3 convolutions return to 256.

~~~dg-flow One decoder step.
Decoder 32×32×512 -(upsample ×2)-> 64×64×512 -(concat skip)-> 64×64×768 -(two 3×3 convs)-> *64×64×256
~~~

**Why concatenate rather than add?** Addition (ResNets, feature pyramid networks) forces both inputs into the same feature space. Concatenation lets the next convolution learn how to mix them, at the price of a wider convolution (768 input channels instead of 512).

**Do the skips matter?** We trained two small networks on the circles-and-squares data, identical except that one has skips. Both pool four times, 48×48 down to a 3×3 bottleneck.

~~~dg-fig 04.04-skips.svg
Two validation images. Without skips, the decoder only has the 3×3 bottleneck to work from and paints vague blobs with the wrong class; with skips, the masks follow the edges. Mean IoU over 32 validation images is shown under each column.
~~~

The no-skip network reached a mean IoU of 0.43 (square IoU 0.01); the U-Net reached 0.91. Both see the same bottleneck context; the difference is the "where" information the skips carry. In a control run with only two poolings (a 12×12 bottleneck) the two scored 0.90 and 0.89: skips matter in proportion to how far you downsample.

## Step 4: a loss for every pixel, and why it fails on tiny objects

The natural loss treats every pixel as a classification: softmax over $C$ logits, cross-entropy, averaged over pixels:

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N} \log p_{i, y_i},$$

where $N$ is the number of pixels, $y_i$ the true class of pixel $i$, and $p_{i,y_i}$ the probability given to it. In PyTorch: `F.cross_entropy(logits, mask)` with logits $(B, C, H, W)$ and an integer mask $(B, H, W)$.

This fails when the foreground is rare. Suppose lesions cover 0.3% of pixels and the model predicts lesion probability 0.003 everywhere, so it never draws a lesion. The average cross-entropy is

$$-(0.997 \ln 0.997 + 0.003 \ln 0.003) \approx 0.003 + 0.017 = 0.020,$$

already small, with a flattering 99.7% pixel accuracy. The few lesion pixels' gradient is averaged with that of a huge number of easy background pixels, so it is tiny. Class weights and the focal loss (04.03) help; the Dice loss attacks the problem directly.

## Step 5: overlap scores, Dice and IoU

Treat the predicted foreground pixels as a set $P$ and the true ones as a set $G$. Two standard overlap scores compare them:

$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}, \qquad D = \frac{2\,|P \cap G|}{|P| + |G|}.$$

**IoU** (intersection over union, or **Jaccard index**) is the box measure from 04.02, now on pixel sets. The **Dice coefficient** $D$ counts the overlap twice and divides by the two sizes added. Both are 1 for a perfect mask and 0 for disjoint ones, and neither counts **true negatives** (background in both masks): adding a million empty pixels changes neither score.

<div class="ex">
<div class="lab">Worked example</div>

**Dice versus IoU on one mask**

The ground truth has 100 foreground pixels, the prediction has 80, and 60 of them overlap.

Dice: $D = 2 \times 60 / (80 + 100) = 120 / 180 = 0.667$.

IoU: the union is $80 + 100 - 60 = 120$ pixels, so $\text{IoU} = 60 / 120 = 0.5$.

The two are linked by $D = 2J/(1+J)$, where $J$ is the IoU: $2 \times 0.5 / 1.5 = 0.667$, matching.

So Dice is never below IoU: "Dice 0.67" and "IoU 0.50" describe the same mask. Never compare raw numbers across the two scores.

</div>

~~~dg-fig 04.04-dice-iou.svg
Dice as a function of IoU. The curve lies above the diagonal everywhere except at 0 and 1, so Dice always looks more generous; the worked example sits at IoU 0.50, Dice 0.67.
~~~

The differentiable **soft Dice** replaces sets with probabilities: $p_i \in [0,1]$ is the predicted foreground probability of pixel $i$ and $g_i \in \{0,1\}$ its true label:

$$D_{soft} = \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}, \qquad \mathcal{L}_{Dice} = 1 - D_{soft}.$$

For several classes, compute $D_{soft}$ per class and average. Background pixels with $p_i \approx 0$, $g_i = 0$ add almost nothing to either sum, so the amount of background does not matter. The constant $\epsilon$ (often 1) keeps the ratio defined when a class is absent. For the "never predict a lesion" model, the Dice loss is near 1, its worst value, instead of cross-entropy's comfortable 0.020.

Dice's weakness: on objects of a few pixels, one pixel changes the ratio a lot, so gradients are noisy. The common recipe is **cross-entropy plus Dice** with equal weight: smooth per-pixel gradients plus a term that will not let the rare class be ignored.

~~~dg-fig 04.04-imbalance.svg
A small U-Net trained on faint lesions covering 0.9% of pixels, with cross-entropy alone and with cross-entropy plus Dice. Left: lesion IoU on 32 validation images. Right: pixel accuracy, which reads 99.1% at step 15 when both models draw no lesion at all.
~~~

At step 15 both runs had collapsed to empty masks (IoU 0, pixel accuracy 99.1%): the trivial solution of step 4. On this clean synthetic task both recovered, ending at IoU 0.87 (cross-entropy) and 0.88 (with Dice), so Dice is no magic; it helps most when foreground is rarer and fainter, as in real scans. The lasting lesson is on the right: pixel accuracy barely moved while IoU went from 0 to 0.88.

## Step 6: evaluating with mean IoU

**Pixel accuracy** is the obvious score and the wrong one: the majority class dominates it. The standard score is **mean IoU (mIoU)**:

1. Over all pixels of the validation set, build a $C \times C$ **confusion matrix**: entry $(t, p)$ counts pixels whose true class is $t$ and predicted class is $p$.
2. For each class $c$: $TP_c$ is the diagonal entry, $FP_c$ is the rest of column $c$ (predicted $c$, wrongly), $FN_c$ is the rest of row $c$ (truly $c$, missed). Then $\text{IoU}_c = TP_c / (TP_c + FP_c + FN_c)$.
3. Average over classes, so every class counts equally.

<div class="ex">
<div class="lab">Worked example</div>

**mean IoU on a 4×4 label map**

Classes 0 (background), 1 and 2. Truth, row by row: `0 0 1 1 / 0 0 1 1 / 0 2 2 1 / 0 2 2 2`. Prediction: `0 0 1 1 / 0 1 1 1 / 0 0 2 1 / 0 2 2 1`.

Three pixels are wrong: one background pixel called 1, one class-2 pixel called 0, one class-2 pixel called 1. Pixel accuracy is $13/16 = 0.8125$.

Confusion matrix rows (truth 0, 1, 2): $[5, 1, 0]$, $[0, 5, 0]$, $[1, 1, 3]$.

Class 0: $TP = 5$, $FP = 1$ (the class-2 pixel predicted as 0), $FN = 1$, so $\text{IoU}_0 = 5/7 = 0.714$.

Class 1: $TP = 5$, $FP = 1 + 1 = 2$, $FN = 0$, so $\text{IoU}_1 = 5/7 = 0.714$.

Class 2: $TP = 3$, $FP = 0$, $FN = 2$, so $\text{IoU}_2 = 3/5 = 0.600$.

Mean IoU $= (0.714 + 0.714 + 0.600)/3 = 0.676$, below the 0.81 pixel accuracy, and it flags class 2 as the weak spot.

</div>

Accumulate the confusion matrix over the *whole* validation set, then compute IoU once. Averaging per-image IoUs hits $0/0$ when a class is absent from both truth and prediction, and libraries handle that differently; state your convention.

## Segmentation in code

A small three-level U-Net in PyTorch (the figures used the same pattern with five levels and fewer channels).

```python
import torch, torch.nn as nn, torch.nn.functional as F

def block(c_in, c_out):                       # two 3x3 convs, padding keeps H and W
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU())

class TinyUNet(nn.Module):
    def __init__(self, n_classes, c=(16, 32, 64)):
        super().__init__()
        self.enc1, self.enc2 = block(1, c[0]), block(c[0], c[1])
        self.mid = block(c[1], c[2])
        self.dec2 = block(c[2] + c[1], c[1])     # upsampled + skip channels
        self.dec1 = block(c[1] + c[0], c[0])
        self.head = nn.Conv2d(c[0], n_classes, 1) # 1x1 conv: C logits per pixel

    def forward(self, x):
        s1 = self.enc1(x)                         # full resolution
        s2 = self.enc2(F.max_pool2d(s1, 2))       # 1/2
        z = self.mid(F.max_pool2d(s2, 2))         # 1/4, the bottleneck
        u = F.interpolate(z, scale_factor=2, mode='bilinear')
        u = self.dec2(torch.cat([u, s2], dim=1))  # concatenate along channels
        u = F.interpolate(u, scale_factor=2, mode='bilinear')
        u = self.dec1(torch.cat([u, s1], dim=1))
        return self.head(u)

net = TinyUNet(n_classes=3)
x = torch.randn(8, 1, 64, 64)                     # batch of 8 grey 64x64 images
y = torch.randint(0, 3, (8, 64, 64))              # integer class per pixel
logits = net(x)
print(logits.shape)                               # torch.Size([8, 3, 64, 64])
print(sum(p.numel() for p in net.parameters()))   # 118659
loss = F.cross_entropy(logits, y)                 # averaged over all 8*64*64 pixels
pred = logits.argmax(dim=1)                       # (8, 64, 64) predicted label map
```

Dice and mIoU in NumPy; one `bincount` builds the confusion matrix by encoding each (true, predicted) pair as `C * true + pred`.

```python
import numpy as np

def soft_dice_loss(probs, onehot, eps=1.0):        # both (C, H, W)
    inter = (probs * onehot).sum(axis=(1, 2))
    denom = probs.sum(axis=(1, 2)) + onehot.sum(axis=(1, 2))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()

def confusion(gt, pred, C):                        # rows = truth, columns = prediction
    return np.bincount(C * gt.ravel() + pred.ravel(), minlength=C * C).reshape(C, C)

def iou_from_confusion(cm):
    tp = np.diag(cm)
    return tp / (cm.sum(0) + cm.sum(1) - tp)       # TP / (TP + FP + FN)

gt   = np.array([[0,0,1,1],[0,0,1,1],[0,2,2,1],[0,2,2,2]])
pred = np.array([[0,0,1,1],[0,1,1,1],[0,0,2,1],[0,2,2,1]])
cm = confusion(gt, pred, 3)
print(cm.tolist())                      # [[5, 1, 0], [0, 5, 0], [1, 1, 3]]
iou = iou_from_confusion(cm)
print(iou.round(3), iou.mean().round(3), (gt == pred).mean())  # [0.714 0.714 0.6] 0.676 0.8125
```

## Reading the result

- **Look at masks** on random and worst-IoU images. Blobby edges mean too coarse an output; speckle means too little context.
- **Read per-class IoU.** A good mIoU can hide a rare class at zero.
- **Count empty predictions** for classes that are present: a high count is the imbalance failure of step 4.
- **Split by object size.** Edges off by a pixel barely hurt large objects but ruin small ones.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| 99%+ pixel accuracy, empty masks for the rare class | plain cross-entropy on imbalanced pixels | add Dice or class weights; sample foreground crops; report per-class IoU
| Smooth, blobby masks that miss thin parts | output too coarse, no skips | add skips, downsample less, or use dilated convs
| Grid or checkerboard texture in the output | transposed conv with kernel not divisible by stride | resize then 3×3 conv, or kernel 4 with stride 2
| Shape error at `torch.cat` in the decoder | input side not divisible by $2^{\text{levels}}$ (say 250 with four levels) | pad input to a multiple of 16, or pad/crop before concatenating
| Mask misaligned, or odd labels on edges | augmentation applied to image only; mask resized bilinearly (invents labels like 1.5) | same geometric transform for both; resize masks with nearest-neighbour
| Dice loss stuck near 1 for small objects | $\epsilon$ large next to the object's pixel count; Dice used alone | sum Dice over the batch, smaller $\epsilon$, add cross-entropy

The original U-Net used **unpadded** convolutions (a 572×572 input gave a 388×388 output, and skips were centre-cropped); modern code uses `padding=1`.

## Going deeper

**Receptive field versus resolution.** A pixel's prediction can only use its **receptive field**, the input region that influences it. Each 3×3 convolution grows it by 2 pixels at the current scale, and each 2× downsample doubles the growth of later layers, so downsampling buys context cheaply and skips buy back resolution. DeepLab instead uses **dilated** (atrous) convolutions, kernels with gaps that see wide areas without downsampling, keeping stride 8 throughout at a higher memory cost.

**The gradient of soft Dice.** With $I = \sum_i p_i g_i$ and $U = \sum_i p_i + \sum_i g_i$ (ignoring $\epsilon$), $\partial D / \partial p_j = 2(g_j U - I)/U^2$. Each pixel's gradient depends on global sums, so Dice is a whole-image loss. A foreground pixel is pushed up, a background pixel down by $2I/U^2$; when $U$ is small (tiny objects) these values are large and swing between batches.

**Beyond IoU.** Masks with similar IoU can have very different edges, so medical work adds boundary metrics such as the Hausdorff distance; instance segmentation uses mask AP.

## Key terms

| Term | Meaning
| --- | --- |
| Semantic segmentation | Assigning one class to every pixel, without separating individual objects
| Instance / panoptic segmentation | One mask per object / instance masks for things plus semantic regions for stuff
| Fully convolutional network | A network with no fully connected layers, so its output is a spatial map
| Encoder / decoder | The downsampling half that builds context / the upsampling half that restores resolution
| Skip connection (U-Net) | Copying an encoder feature map across and concatenating it into the decoder at the same resolution
| Transposed convolution | A learned upsampling layer that stamps the kernel onto the output at stride spacing
| Dice coefficient | $2\lvert P \cap G\rvert / (\lvert P\rvert + \lvert G\rvert)$; its soft version is a training loss that ignores true negatives
| IoU / mIoU | Overlap over union for one class / its average over classes, from a pixel confusion matrix

<div class="take">
<div class="lab">Key takeaway</div>

segmentation needs both context and resolution. The encoder downsamples to see context, the decoder upsamples back to full size, and U-Net's skip connections carry the fine "where" detail that upsampling alone cannot recover. Train rare classes with cross-entropy plus Dice, and judge the result with per-class IoU and mIoU, never pixel accuracy.

</div>
