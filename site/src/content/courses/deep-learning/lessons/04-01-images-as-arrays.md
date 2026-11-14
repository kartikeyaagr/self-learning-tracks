---
id: "04.01"
course: "deep-learning"
module: "04"
slug: "images-as-arrays"
title: "Images as arrays"
keyIdeas: ["An image is a grid of numbers with conventions: `uint8` values 0-255 in HWC order on disk, `float32` NCHW in most PyTorch training code. Most silent bugs are convention mismatches (RGB vs BGR, HWC vs CHW, 0-255 vs 0-1), not model bugs.", "Change axis order with `transpose(2, 0, 1)` or `permute`, never `reshape(3, H, W)`: reshape gives the right shape but scrambles the pixels, and nothing errors.", "Normalise with the statistics the model was trained with: divide by 255, then subtract each channel's mean and divide by its std (ImageNet: mean 0.485, 0.456, 0.406; std 0.229, 0.224, 0.225), computed on the training set only.", "Classic filters (box, Gaussian, Sobel) are fixed convolutions. A Sobel filter is near zero on flat regions and large on edges, which is what early CNN layers learn to detect.", "Augmentation must preserve the label: geometric transforms must also move boxes and masks, and some (horizontal flip on text, 180° rotation on 6 and 9) change the label itself.", "Train-time augmentation is random; eval-time preprocessing is deterministic (resize, centre crop, normalise) and must match what the model saw in training."]
question: "A ResNet pretrained on ImageNet reports 76% top-1 accuracy, but in your pipeline (images loaded with OpenCV, converted with `reshape`, normalised with mean 0.5 and std 0.5) it gets 61%. List the likely causes, in the order you would check them, and how you would confirm each."
answer: "Work from the bugs that do the most damage and are cheapest to confirm. First, axis order: `reshape` to (3, H, W) keeps the HWC memory order and cuts it into three pieces, so each 'channel' is a strip of interleaved R, G and B values from one third of the image. Confirm by taking a tensor from your pipeline, transposing it back to HWC and displaying it (or by comparing `x[0]` with the true red plane); fix with `transpose(2, 0, 1)` / `permute(2, 0, 1)`. Second, channel order: OpenCV decodes to BGR but the model was trained on RGB, so red and blue are swapped; confirm by reversing channels (`img[..., ::-1]` or `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`) and re-running a fixed validation slice. Third, normalisation statistics: the model expects inputs scaled by 255 and normalised with the ImageNet per-channel mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225), not 0.5/0.5; the per-pixel shift is modest but systematic, so print batch means and stds and compare your tensor with the reference preprocessing using `allclose`. Fourth, evaluation resizing: check you resize the short side to 256 and centre crop 224 with the same interpolation (and antialiasing) as the reference. Change one thing at a time on a fixed slice of, say, 1,000 validation images and record accuracy after each fix, so each cause's contribution is visible; after all four, the remaining gap to 76% should be small, and any leftover points usually come from resize details."
sources: [{"title": "torchvision: Transforming and augmenting images", "url": "https://pytorch.org/vision/stable/transforms.html", "note": "Reference for the standard train and eval pipelines, including how boxes and masks are transformed alongside images."}, {"title": "Dive into Deep Learning: Image Augmentation", "url": "https://d2l.ai/chapter_computer-vision/image-augmentation.html", "note": "Runnable walkthrough of flips, crops and colour jitter and their effect on training."}, {"title": "RandAugment (Cubuk et al., 2019)", "url": "https://arxiv.org/abs/1909.13719", "note": "How a two-parameter augmentation policy matches searched policies; read the method section."}, {"title": "mixup: Beyond Empirical Risk Minimization", "url": "https://arxiv.org/abs/1710.09412", "note": "The original mixup paper; short, and explains why mixing labels regularises."}]
practice: ["image-normalize", "per-channel-mean-and-std", "rgb-to-grayscale", "gaussian-blur-2d", "transform-pipeline"]
tags: ["preprocessing", "augmentation"]
tools: ["NumPy", "torchvision"]
---

## What you will learn

- What an image really is: a grid of small integers with a shape, a data type and a memory layout.
- How colour channels are stored, and why BGR and CHW conventions cause silent bugs.
- How to normalise pixels the way pretrained models expect.
- What classic filters (box, Gaussian, Sobel) compute, worked by hand.
- Which augmentations keep the label true and which quietly change it.

*Before you start:* you should know what a tensor and its shape are (lesson 01.01) and what a convolution layer does (module 03). No image-processing background is assumed.

## Where this module starts

Module 03 built the convolutional network. This module covers what surrounds it in real vision work: pixels to tensors (this lesson), object detection (04.02, 04.03), per-pixel segmentation (04.04) and Vision Transformers (04.05).

This first lesson is the unglamorous part, and it is where a surprising share of real-world accuracy gets lost. A well-trained network still scores badly if test images are arranged, coloured or scaled differently from its training images, and none of these mistakes raise an error.

## The idea in one picture

To a computer a photo is a table of numbers. Take the pagoda photo that ships with scikit-learn, shrink it to 107 rows by 160 columns, and print the red values in a 6 × 6 square where the dark roof meets the sky:

~~~dg-fig 04.01-zoom.svg
Left: the photo, with a tiny box at the edge of the roof. Right: the 36 red-channel numbers inside that box. The darkest roof pixels are in the 20s to 40s, the sky is around 220, and the edge is just the place where the numbers jump. Photo: "china.jpg" from scikit-learn's sample images, by danielbuechele on Flickr, CC BY 2.0.
~~~

Each small square is a **pixel** (picture element) holding one number per **channel**: red, green and blue light. Each number is an **intensity** from 0 (none of that light) to 255 (the most), stored as an 8-bit unsigned integer, the `uint8` **data type**. The small photo is $107 \times 160 \times 3 = 51{,}360$ numbers; the full original ($427 \times 640 \times 3$) is 819,840, one byte each. In the grid, dark roof sits top-left (33, 28, 37...), sky bottom-right (around 222), with a diagonal edge between. Everything a vision model does starts from tables like this.

## Step 1: shape, data type and size

- **Shape.** Grayscale is $(H, W)$: height (rows) by width (columns). Colour adds a channel axis, $(H, W, C)$ with $C = 3$. A batch adds $N$, the number of images, in front.
- **Data type.** Files decode to `uint8`; networks compute in `float32` (or lower precision on GPUs), so you convert at some point.
- **Indexing.** `img[row, col]` gives one pixel's channel values. Row 0 is the *top*, and the first index is the vertical one. `img[100, 200]` in the full pagoda photo is `[123, 47, 11]`: lots of red, little green, almost no blue, an orange roof tile.

The conversion matters for memory: the full photo is 819,840 bytes as `uint8` but 3,279,360 as `float32`. A 224 × 224 RGB input is 150,528 numbers, and a batch of 256 in `float32` is about 154 MB before the network does anything. So data loaders keep `uint8` as long as possible.

## Step 2: channels and colour spaces

Each channel alone is a grayscale image of one colour of light.

~~~dg-fig 04.01-channels.svg
A photo, its three channels shown as grey images, and the same numbers read in the wrong channel order. The red pillars are bright in the R channel and dark in B; the sky is bright everywhere, brightest in B. Swapping R and B turns the roof blue and the sky orange.
~~~

Channel order is a **convention**, not a law. PIL, torchvision, matplotlib and most deep-learning code use **RGB**; OpenCV decodes files to **BGR**. Hand BGR to a model trained on RGB and nothing crashes: shapes agree, values are in range. The model just sees orange skies and loses accuracy. The fix is one line, `img[..., ::-1]` (reverse the last axis), or `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

RGB mixes brightness and colour: a shadow lowers all three numbers at once. Other **colour spaces** separate them.

| Space | Channels | Typical use
| --- | --- | --- |
| RGB | red, green, blue | model input, display
| Grayscale | luma $Y = 0.299R + 0.587G + 0.114B$ | edge detection, cheap models
| HSV | hue, saturation, value | colour jitter: shift hue without changing brightness
| YCbCr | luma plus two colour-difference channels | JPEG compression, which stores colour at lower resolution than brightness

The weights are unequal because eyes are most sensitive to green, least to blue. That roof-tile pixel $(123, 47, 11)$ becomes $0.299 \times 123 + 0.587 \times 47 + 0.114 \times 11 = 36.8 + 27.6 + 1.3 = 65.6$, a dark grey, even though its red value alone is fairly high.

## Step 3: axis order and memory layout

The same three-channel image can be laid out in two ways.

- **HWC** (channels last): each pixel's R, G, B side by side, then the next pixel. NumPy, PIL and OpenCV give you this; TensorFlow and JAX code usually keeps it.
- **CHW** (channels first): the whole red plane, then green, then blue. PyTorch layers expect batches as **NCHW**.

~~~dg-fig 04.01-layout.svg
Top: the twelve numbers of a 2 × 2 colour image in the order they sit in memory, interleaved (HWC) or one plane at a time (CHW). Bottom: converting with transpose gives the true red channel; converting with reshape gives an image of the right shape made of scrambled stripes.
~~~

Where does element $(h, w, c)$ live? Counting from 0 in flat memory:

$$\text{HWC: } (h \cdot W + w)\cdot C + c, \qquad \text{CHW: } c \cdot H W + h \cdot W + w.$$

Here $h$ is the row, $w$ the column, $c$ the channel, and $H, W, C$ the sizes. Converting layouts means *moving* every number, which is what `transpose(2, 0, 1)` (NumPy) or `permute(2, 0, 1)` (PyTorch) does: axis 2 (channels) goes first, then rows, then columns.

The trap is `reshape(3, H, W)`. It gives the right shape but keeps the old order and just cuts the flat list into three pieces, so the first "channel" holds the interleaved R, G and B of the top third of the image (bottom right of the figure). Nothing errors, so the bug can survive into training and quietly ruin accuracy.

<div class="ex">
<div class="lab">Worked example</div>

**where does one number go?**

Take a 2 × 2 RGB image, so $H = W = 2$ and $C = 3$. We want the blue value of the bottom-left pixel: $h = 1$, $w = 0$, $c = 2$.

HWC position: $(1 \cdot 2 + 0) \cdot 3 + 2 = 2 \cdot 3 + 2 = 8$. In the top strip of the figure, position 8 (counting from 0) is "B2", the blue of pixel 2. Correct.

CHW position: $2 \cdot (2 \cdot 2) + 1 \cdot 2 + 0 = 8 + 2 = 10$. In the CHW strip, position 10 is again "B2". Same number, different address.

`reshape(3, 2, 2)` on the HWC array makes channel 0 out of positions 0 to 3: R0, G0, B0, R1. That is the scramble.

</div>

## Step 4: value range and normalisation

As floats, pixels are still 0 to 255, all positive, with a mean around 120 to 145. Fed straight in, they make first-layer outputs large and one-signed, defeating the module 01 initialisation, which assumes inputs with mean near 0 and spread near 1.

The standard recipe: divide by 255 to get $[0, 1]$, then for each channel $c$

$$x' = \frac{x - \mu_c}{\sigma_c},$$

where $x$ is the scaled pixel value, $\mu_c$ is the average of channel $c$ over the training images and $\sigma_c$ its standard deviation. This is **per-channel normalisation**. For ImageNet-pretrained models the statistics are $\mu = (0.485, 0.456, 0.406)$ and $\sigma = (0.229, 0.224, 0.225)$ for R, G, B; nearly every torchvision pretrained model expects exactly these.

~~~dg-fig 04.01-normalise.svg
The pagoda photo's three channels before and after ImageNet normalisation. The shapes of the histograms do not change at all; they are shifted and stretched so that values live roughly between -2.1 and 2.6, centred near zero instead of near 145.
~~~

This one photo does not end up with mean exactly 0 and std exactly 1 (its means are 0.36, 0.51, 0.65). That is expected: the statistics describe the whole training set, and a bright sunny photo sits above average.

<div class="ex">
<div class="lab">Worked example</div>

**normalising single pixels**

A red value of 200: divide by 255 to get $0.784$. Subtract the red mean and divide by the red std: $(0.784 - 0.485)/0.229 = 0.299/0.229 = 1.31$.

A blue value of 0 (no blue at all): $0/255 = 0$, then $(0 - 0.406)/0.225 = -1.80$.

The extremes: red 0 gives $-0.485/0.229 = -2.12$; blue 255 gives $(1 - 0.406)/0.225 = 2.64$. So normalised inputs live in about $[-2.1, 2.6]$, not in $[0, 1]$. Undo the normalisation before plotting them.

</div>

Training from scratch, compute $\mu_c$ and $\sigma_c$ on the training split only; including validation images leaks a little information, usually small but free to avoid. Fine-tuning a pretrained model, use *its* statistics, because its first layer was tuned to inputs scaled that way.

## Step 5: filters are fixed convolutions

Before deep learning, engineers hand-designed small weight grids and slid them over the image; each output pixel is the weighted sum under the grid. That is module 03's convolution with hand-chosen weights, and trained CNNs learn similar first-layer shapes.

- **Box blur**: every weight $1/k^2$, a plain neighbourhood average. Cheap, but near and far neighbours count equally, leaving blocky artefacts.
- **Gaussian blur**: weights proportional to $e^{-(x^2+y^2)/2\sigma^2}$, where $(x, y)$ is the offset from the centre and $\sigma$ sets the blur radius. Near neighbours count more, so the blur is smoother.
- **Sobel**: two 3 × 3 grids estimating horizontal ($G_x$) and vertical ($G_y$) brightness change. The **gradient magnitude** $\sqrt{G_x^2 + G_y^2}$ is large on edges, near zero on flat regions.

~~~dg-fig 04.01-filters.svg
Top: a grayscale crop, two blurs and the Sobel edge magnitude. Bottom: the values along the dashed row. The Gaussian smooths away the small wiggles of the roof decoration; the Sobel response is near zero over the flat sky and spikes exactly where brightness jumps from roof to sky.
~~~

The horizontal Sobel kernel is

$$K_x = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix}.$$

It subtracts the left column from the right one, with the middle row counted twice.

<div class="ex">
<div class="lab">Worked example</div>

**Sobel on a vertical edge**

Take a 3 × 3 patch where the left two columns are dark (10) and the right column is bright (200): every row is $(10, 10, 200)$.

Multiply element by element with $K_x$ and add. Row 1: $-1 \cdot 10 + 0 \cdot 10 + 1 \cdot 200 = 190$. Row 2: $-2 \cdot 10 + 0 + 2 \cdot 200 = 380$. Row 3: $190$. Total $G_x = 760$.

The vertical kernel $K_y = K_x^\top$ subtracts the top row from the bottom row; both rows are $(10, 10, 200)$, so $G_y = 0$.

Magnitude $\sqrt{760^2 + 0^2} = 760$: a strong vertical edge. On a flat patch (all 10s) both sums are exactly 0.

</div>

Two practical details. The Gaussian is **separable**: a $k \times k$ blur equals a $k$-tap horizontal pass then a $k$-tap vertical pass, so $k = 5$ costs 10 multiplies per pixel instead of 25. And at the border the grid hangs off the image, so the missing pixels are invented: zeros, a mirror of the inside (**reflect**) or copies of the edge (**replicate**). That only changes a thin border, but on a 32 × 32 image the border is a big share of the picture.

## Step 6: augmentation

**Augmentation** makes extra training examples by changing images in ways that should not change the answer: a flipped or slightly darker pagoda is still a pagoda. The model sees a new variant every epoch, which reduces overfitting and teaches it which changes to ignore.

~~~dg-fig 04.01-augment.svg
Top: flipping, cropping and brightening keep the label "pagoda". Bottom: an 8 × 8 handwritten 6 from scikit-learn's digits. Rotated 180 degrees it looks like the real 9 beside it, so that augmentation would teach wrong labels.
~~~

The common families:

- **Geometric**: random resized crop (a random region covering 8% to 100% of the area, resized to e.g. 224), horizontal flip, small rotations and shifts.
- **Photometric**: brightness, contrast, saturation and hue jitter; blur; noise.
- **Mixing**: **mixup** blends two images and their labels with weight $\lambda \sim \text{Beta}(\alpha, \alpha)$; **CutMix** pastes a patch of one image into another and mixes labels by pasted area.
- **Policies**: RandAugment applies $N$ random operations at a shared strength $M$: two knobs instead of dozens.

~~~dg-flow One image's path from disk to the network. Augmentation runs on the cheap uint8 form; conversion and normalisation come last.
JPEG bytes -(decode)-> uint8 HWC -(random crop, flip, jitter)-> augmented -(to float, /255)-> *normalise -(transpose, batch)-> NCHW float32
~~~

The rule that matters is **label preservation**. Horizontal flips are fine for animals and wrong for text, digits and left-versus-right medical scans; large rotations are wrong for 6 and 9. For detection, geometric transforms must also move the boxes; for segmentation, the mask too, resized with nearest-neighbour interpolation so class ids stay whole numbers.

Augmentation is random and happens only during training. Evaluation uses a fixed, deterministic pipeline; the usual ImageNet recipe resizes the shorter side to 256 and takes the centre 224 × 224 crop.

## Images as arrays in code

Both blocks were run on scikit-learn's sample photo; the printed values are real.

```python
import numpy as np
from sklearn.datasets import load_sample_image

img = load_sample_image("china.jpg")            # decoded JPEG, RGB order
print(img.shape, img.dtype, img.min(), img.max())  # (427, 640, 3) uint8 0 255
print(img[100, 200])                             # [123  47  11]  (row 100, col 200)
print(img.nbytes, img.astype(np.float32).nbytes) # 819840 3279360

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def to_model_input(img_hwc_uint8):               # (H, W, 3) RGB uint8
    x = img_hwc_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD                          # broadcasts over the last axis
    return x.transpose(2, 0, 1)                   # (3, H, W): move axes, never reshape

x = to_model_input(img)
print(x.shape, x.min().round(2), x.max().round(2))  # (3, 427, 640) -2.12 2.64
print(x.mean(axis=(1, 2)).round(2))              # [0.36 0.51 0.65]

```

To get your own $\mu_c, \sigma_c$, stream the training images, accumulate per-channel sums of $x$ and $x^2$ over `im.reshape(-1, 3) / 255`, and take $\mu = \overline{x}$, $\sigma = \sqrt{\overline{x^2} - \mu^2}$. That `reshape(-1, 3)` is safe: it keeps channels last and only merges rows and columns.

The same steps in PyTorch, including a Sobel filter as a convolution:

```python
import torch, torch.nn.functional as F

t = torch.from_numpy(img.copy()).permute(2, 0, 1)   # (3, H, W) view, no copy
print(t.shape, t.is_contiguous())                   # torch.Size([3, 427, 640]) False
batch = t.unsqueeze(0).float().div(255)             # (1, 3, H, W)
print(batch.stride())                               # (3, 1, 1920, 3): still HWC in memory

w = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
gray = (batch * w).sum(1, keepdim=True)             # (1, 1, H, W) luma
sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), sobel_x)
print(gx.shape)                                     # torch.Size([1, 1, 427, 640])
```

The stride printout shows that `permute` moves no data; it only changes how indices map to memory. The tensor has NCHW *shape* but HWC *memory order*, which PyTorch calls **channels_last**. Layers accept it, and on GPUs with tensor cores it is often the faster layout for convolutions.

In a real training job you would build the same pipeline from library transforms, for example torchvision's `v2.Compose([v2.RandomResizedCrop(224), v2.RandomHorizontalFlip(), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])` for training and `Resize(256)` plus `CenterCrop(224)` in place of the random steps for evaluation.

## Checking the pipeline

Preprocessing bugs are silent, so look for them on purpose.

1. **Look at what the model sees.** Take a loader batch, undo the normalisation ($x \sigma_c + \mu_c$, times 255), transpose to HWC and display it. Stripes mean a reshape bug, orange skies a channel swap, washed-out or black images a scaling bug.
2. **Print batch statistics.** Per-channel mean and std should be near 0 and 1. A mean near 120 means no scaling; near $-2$ means dividing by 255 twice.
3. **Compare against the reference.** For a pretrained model, run its official preprocessing and yours on the same file and compare the tensors with `allclose`. Differences usually come from the resize method.
4. **Check the eval pipeline separately.** A bug can hide in just one of the two pipelines. The same image twice through eval must give an identical tensor.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| `reshape(3, H, W)` instead of `transpose(2, 0, 1)` | Training runs, accuracy far below expectations; images look striped when viewed | Use transpose / permute; view a batch
| BGR fed to an RGB model | A few points of accuracy lost, worse on colour-dependent classes | Reverse channels after OpenCV decoding
| Wrong or missing normalisation | Pretrained model underperforms; batch mean far from 0 | Use the model's own mean and std, scale by 255 once
| Augmentation that changes the label | Model confuses specific classes (6/9, left/right, b/d) | Remove that transform for this task
| Boxes or masks not transformed with the image | Detections offset or mirrored; masks misaligned | Use transforms that take image and targets together
| Mask resized with bilinear interpolation | Invalid class ids at object borders | Nearest-neighbour for label maps
| Different eval resize than the reference | Small, persistent gap to published numbers | Match interpolation, short side and crop

## Going deeper

**Pixels are not light.** Stored values are **gamma-encoded** (sRGB): roughly light intensity to the power $1/2.2$, spending more of the 256 levels on dark tones where eyes are sensitive. So averaging stored values is not averaging light. Mixing black (0) and white (255) 50/50 in linear light gives a grey stored as about 188; averaging the stored numbers gives 128, visibly darker. For classification this rarely matters; for resizing and blending, linear light is more faithful.

**Downsampling needs a blur first.** Keeping every $k$-th pixel (as these figures do, to stay small) creates false patterns, **aliasing**, when detail is finer than the new pixel spacing. Proper resizers blur first (antialiasing). Libraries have differed in whether they antialias by default, a classic source of eval-time mismatch.

**Why Gaussian blur is separable.** The 2-D Gaussian factorises: $e^{-(x^2+y^2)/2\sigma^2} = e^{-x^2/2\sigma^2} \cdot e^{-y^2/2\sigma^2}$. So the kernel is a rank-one matrix $g g^\top$: convolve with $g$ along rows, then along columns. Sobel is separable too: $K_x = (1, 2, 1)^\top (-1, 0, 1)$, a vertical smoothing times a horizontal difference, which is why it is less noise-sensitive than a bare difference.

## Key terms

| Term | Meaning
| --- | --- |
| Pixel | One cell of the image grid, holding one number per channel
| Channel | One layer of the image, such as the red intensities of every pixel
| `uint8` | 8-bit unsigned integer, values 0 to 255; how decoded images are stored
| HWC / CHW | Channels-last versus channels-first axis order; PyTorch layers take NCHW
| Per-channel normalisation | Subtract each channel's training mean and divide by its standard deviation
| Filter (kernel) | A small weight grid slid over the image; each output is a weighted sum of neighbours
| Sobel operator | A pair of 3 × 3 filters estimating horizontal and vertical brightness change
| Augmentation | Random label-preserving changes to training images

<div class="take">
<div class="lab">Key takeaway</div>

an image is a `uint8` array with conventions, and the conventions are part of the model. Channel order, axis order (transpose, never reshape), value range, normalisation statistics and eval-time resizing must exactly match training, and every augmentation must keep the label true; look at a decoded batch before you trust any accuracy number.

</div>
