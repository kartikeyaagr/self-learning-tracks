---
id: "04.02"
course: "deep-learning"
module: "04"
slug: "object-detection-basics"
title: "Object detection basics"
keyIdeas: ["A detector outputs a variable-length list of (box, class, score). Dense detectors cope with the variable length by predicting at every feature-map location and then pruning.", "**IoU**, intersection area over union area, is the single yardstick of detection: it labels anchors in training, decides which boxes are duplicates in NMS, and decides hits at evaluation (COCO averages AP over IoU 0.5 to 0.95).", "**Anchors** are fixed reference boxes tiled over each feature map at several sizes and shapes. Each is labelled positive (IoU at least 0.5), negative (below 0.4) or ignored, which leaves a few positives among tens of thousands of negatives.", "Box targets are **encoded** relative to the anchor: centre shift divided by anchor size and the log of the size ratio. This makes targets scale-free, keeps decoded widths positive, and must round-trip exactly through decode.", "**Non-maximum suppression** keeps the highest-scoring box and deletes others overlapping it beyond a threshold. It is greedy and cannot tell a duplicate from a neighbour, so in crowds it deletes real objects; Soft-NMS or NMS-free detectors address this."]
question: "Your pedestrian detector works well on suburban streets but on footage of a busy railway platform it reports only about 80% of the people, even though, when you print the boxes before NMS, almost every person has a high-scoring candidate. AP50 on the platform footage is decent but COCO-style mAP is poor. Explain what is happening, propose two fixes for the missing people with their costs, and say what the AP50 versus mAP gap tells you."
answer: "The candidates exist before NMS, so the classifier and anchors are doing their job; the loss happens in post-processing. On a crowded platform, neighbouring people overlap heavily, and the IoU between two correct boxes for two different people often exceeds the NMS threshold (say 0.5). NMS keeps the higher-scoring box and deletes everything that overlaps it beyond the threshold, so the second person's correct box is removed as if it were a duplicate: NMS has no way to tell a duplicate from a neighbour. Fix 1: raise the NMS IoU threshold to about 0.6-0.7. Neighbours survive, but more genuine duplicates also survive on isolated people, which lowers precision; you can tune it on a crowd-heavy validation set. Fix 2: Soft-NMS, which multiplies the scores of overlapping boxes by a decay in IoU (linear or Gaussian) instead of zeroing them. The neighbour keeps a reduced but rankable score, so recall rises with a smaller precision cost; it adds a decay hyperparameter and a little compute. A structural fix is a set-prediction detector such as DETR, trained with one-to-one matching so it needs no NMS, at the price of much longer training. The AP50 versus mAP gap is a separate signal: AP50 counts a box as correct at IoU 0.5, while COCO mAP averages over thresholds up to 0.95, so a large gap means people are found but boxes are placed loosely, for example because of occlusion. That points at box regression (an IoU-based loss, more or better-matched positive anchors, anchor shapes that fit tall narrow people) rather than at NMS. Before any of this, check the boring causes: box format and resizing of labels, by testing that decode(encode(gt)) returns gt and by drawing labels on augmented images."
sources: [{"title": "Faster R-CNN (Ren et al., 2015)", "url": "https://arxiv.org/abs/1506.01497", "note": "Introduces anchors and the region proposal network; section 3.1 defines the box encoding used almost everywhere."}, {"title": "Dive into Deep Learning: Anchor Boxes", "url": "https://d2l.ai/chapter_computer-vision/anchor.html", "note": "Code for generating anchors, IoU matrices, labelling anchors and NMS."}, {"title": "FCOS: Fully Convolutional One-Stage Object Detection", "url": "https://arxiv.org/abs/1904.01355", "note": "The anchor-free alternative; useful contrast for what anchors actually buy you."}, {"title": "torchvision.ops.nms", "url": "https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html", "note": "The production NMS interface and its box-format expectations."}]
practice: ["iou-matrix", "non-maximum-suppression", "box-encode-and-decode", "anchor-box-generation", "soft-non-maximum-suppression"]
tags: ["object detection", "IoU"]
tools: ["NumPy"]
---

## What you will learn

- What a detector outputs (a list of boxes, each with a class and a score) and why that list is awkward for a neural network.
- How to measure overlap between two boxes with **IoU**, by hand and as a full matrix in NumPy.
- What **anchors** are, how they are tiled over an image, and how each one is labelled positive, negative or ignored.
- Why the network predicts *encoded offsets* instead of raw pixel coordinates, and how to encode and decode them.
- How **non-maximum suppression** turns a cloud of duplicate boxes into one box per object, and where it fails.
- How detectors are scored (precision, recall, AP, COCO mAP) so you can read a results table.

*Before you start:* lesson 04.01 (images as tensors, shapes like $(N, C, H, W)$) and module 03 (convolutional networks, feature maps and strides). You need logarithms and the idea of a classifier's score; nothing else.

Lesson 04.01 treated an image as one tensor with one answer. Real scenes hold many things, and applications need to know *where* each one is. This lesson builds the machinery every classic detector shares; lesson 04.03 covers the losses that train it.

## The idea in one picture

Classification answers "what is in this image". **Object detection** answers "what is where". The output is a list of detections, one row per object:

| Box (x1, y1, x2, y2) in pixels | Class | Score
| --- | --- | --- |
| (20, 30, 90, 140) | person | 0.89
| (120, 60, 230, 130) | dog | 0.89
| (250, 20, 300, 135) | person | 0.82

A **bounding box** is the smallest axis-aligned rectangle around an object. The **score** is the model's confidence that the box contains that class. The list's length changes per image (zero in an empty corridor, forty in a crowd), but a network has a fixed number of outputs. The trick: predict far too many candidate boxes at fixed positions, then throw most away.

~~~dg-fig 04.02-scene.svg
Left: the raw output of a simulated detector, 39 candidate boxes, many stacked on each object (darker means higher score). Right: after dropping low scores and removing duplicates, exactly one box per object is left. Dashed boxes are the ground truth.
~~~

Getting from left to right needs three tools: a measure of how much two boxes overlap (IoU), a rule for which candidates should fire during training (anchor matching), and a way to delete duplicates at test time (NMS).

Two families of detectors produce those candidates:

- **Two-stage** (Faster R-CNN): a **region proposal network** suggests about 1,000 to 2,000 boxes that probably contain *something*; a second head crops features for each (RoIAlign), classifies it and refines its box. Accurate, slower.
- **One-stage** (SSD, RetinaNet, YOLO): one dense head predicts class scores and box corrections at every feature-map cell. Faster and simpler; with a good loss (lesson 04.03) just as accurate.

~~~dg-flow A one-stage detector: dense predictions everywhere, then decode and prune.
Image -> Backbone + FPN -(per cell)-> *Dense head -(decode offsets)-> Candidate boxes -(score > 0.05)-> NMS -> Detections
~~~

The **backbone** is a CNN from module 03; the **FPN** (feature pyramid network) supplies feature maps at several strides, fine ones for small objects and coarse ones for large.

## Step 1: agree on a box format

Before any maths, decide how a box is written down. Three formats are common:

- **Corners** $(x_1, y_1, x_2, y_2)$: top-left and bottom-right. Used by torchvision and Pascal VOC.
- **Corner plus size** $(x, y, w, h)$: top-left plus width and height. Used by COCO annotation files.
- **Centre plus size** $(c_x, c_y, w, h)$: used by YOLO labels and by box encoding (step 4).

Any of them may be in pixels or **normalised** to $[0, 1]$, and $y$ usually grows *downwards* from the top of the image. Mixing formats is one of the most common detection bugs: the code runs and the boxes are nonsense. Convert to one format when loading labels and when exporting predictions, and never mix inside.

## Step 2: measure overlap with IoU

Every decision in detection compares two boxes: is this candidate on that object, is this box a duplicate, did the prediction hit? The yardstick is **intersection over union**:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

Here $|A|$ is the area of box $A$, $A \cap B$ the region both cover and $A \cup B$ the region either covers. IoU is 1 for identical boxes and 0 for boxes that do not touch. Because it is a ratio, it is scale-free: a 5-pixel error hurts a tiny box far more than a large one.

The intersection is itself a box: its top-left is the larger of the two top-left corners, its bottom-right the smaller of the two bottom-right corners, with width and height clamped at zero when the boxes do not overlap.

<div class="ex">
<div class="lab">Worked example</div>

**IoU of two offset squares**

$A = (0, 0, 10, 10)$ and $B = (5, 5, 15, 15)$ in corner format. Each has area $10 \times 10 = 100$.

Intersection top-left: $(\max(0, 5), \max(0, 5)) = (5, 5)$. Bottom-right: $(\min(10, 15), \min(10, 15)) = (10, 10)$. Width $10 - 5 = 5$, height 5, area 25.

Union: $100 + 100 - 25 = 175$ (subtract the intersection once, because both areas counted it).

IoU $= 25 / 175 = 0.143$. Each box covers half of the other along each axis, yet the IoU is far below 0.5. That is why 0.5 is already a demanding threshold.

</div>

~~~dg-fig 04.02-iou.svg
Left: IoU of two 10 by 10 squares as one slides away. Sliding sideways by only 3.3 pixels (a third of the side) brings IoU down to 0.5; sliding diagonally reaches 0.5 after 1.8 pixels on each axis. Right: the worked example.
~~~

With $N$ predictions and $M$ ground-truth boxes you need every pair, an $N \times M$ **IoU matrix**. Computing it with broadcasting (shown in the code section) is the workhorse of both training-time matching and evaluation.

## Step 3: tile anchors and label them

An **anchor** is a fixed reference box at a known position, chosen before the network sees any image. For each anchor the network predicts a score per class and a small correction that moves and resizes the anchor onto the object.

Anchors are tiled over a feature map. A feature map with **stride** $s$ has one cell per $s \times s$ block of input pixels, so cell $(i, j)$ has its centre at pixel $(s(j + 0.5), s(i + 0.5))$. At every cell you place $A$ anchors, typically 3 sizes times 3 aspect ratios (1:2, 1:1, 2:1), which gives $A = 9$.

~~~dg-fig 04.02-anchors.svg
Left: the nine anchors of one cell on a stride-8 grid; colour is size, line style is shape. Right: 768 anchors of size 40 (three shapes per cell) compared with one ground-truth box. Only 14 overlap it with IoU of at least 0.5; 22 are borderline and ignored; the other 732 are background.
~~~

The numbers grow quickly. RetinaNet uses 9 anchors per location on five pyramid levels with strides 8, 16, 32, 64 and 128. On an 800 by 800 image the stride-8 level alone has $100 \times 100 = 10{,}000$ cells and $90{,}000$ anchors; all five levels together hold $120{,}087$.

During training each anchor needs a **target**: a class (or "background") and, if on an object, the box to move to. RetinaNet assigns these by IoU:

- IoU $\ge 0.5$ with some ground-truth box: **positive**, assigned to the box it overlaps most.
- IoU $< 0.4$ with every box: **negative** (background).
- In between: **ignored**, so anchors that half-cover an object do not send confusing gradients either way.
- Faster R-CNN's proposal network uses stricter cut-offs (0.7 and 0.3) and adds one more rule that many detectors copy: each ground-truth box is also matched to its single best anchor, so a small or oddly shaped object is never left without any positive.

The result is a handful of positives in a sea of negatives, an imbalance that lesson 04.03 tackles with focal loss.

## Step 4: encode boxes as offsets

The head could predict raw pixel coordinates, but those are hard to learn: a 10-pixel error is huge for a 16-pixel object and negligible for a 400-pixel one, and nothing stops a width from going negative. Instead each target is written relative to its anchor. With the anchor in centre format $(x_a, y_a, w_a, h_a)$ and the ground truth $(x, y, w, h)$:

$$t_x = \frac{x - x_a}{w_a}, \quad t_y = \frac{y - y_a}{h_a}, \quad t_w = \log\frac{w}{w_a}, \quad t_h = \log\frac{h}{h_a}$$

- $t_x, t_y$ are the centre shift measured in *anchor widths and heights*. A 4-pixel shift on a 16-pixel anchor and a 32-pixel shift on a 128-pixel anchor are both $0.25$, so one regression head works at every scale.
- $t_w, t_h$ are log size ratios. "Twice as wide" is $\log 2 = +0.69$ and "half as wide" is $-0.69$, a symmetric pair.
- **Decoding** inverts this: $x = x_a + t_x w_a$ and $w = w_a e^{t_w}$, which is always positive.

~~~dg-fig 04.02-encoding.svg
Left: the log target treats growing and shrinking symmetrically, while a plain ratio target is lopsided. Right: decoding exponentiates, so a wild prediction gives an absurd width unless t_w is clamped (the solid line stops growing at 1,000 pixels for a 16-pixel anchor).
~~~

Two implementation details follow from the figure. Many libraries divide the targets by fixed "variances" such as $(0.1, 0.1, 0.2, 0.2)$ (SSD) or, equivalently, multiply them by weights $(10, 10, 5, 5)$ (the second stage of Faster R-CNN in torchvision), so the targets have roughly unit scale for the loss. And before calling `exp`, they clamp $t_w, t_h$ at about $\log(1000/16) \approx 4.14$, so one bad prediction early in training cannot overflow.

<div class="ex">
<div class="lab">Worked example</div>

**encode and decode one box**

Anchor in corners $(34, 34, 66, 66)$, so centre $(50, 50)$ and size $32 \times 32$. Ground truth $(34, 34, 82, 58)$, so centre $(58, 46)$ and size $48 \times 24$.

$t_x = (58 - 50)/32 = 0.25$. $t_y = (46 - 50)/32 = -0.125$.

$t_w = \log(48/32) = \log 1.5 = 0.405$. $t_h = \log(24/32) = \log 0.75 = -0.288$.

Decode: centre $x = 50 + 0.25 \times 32 = 58$, $y = 50 - 0.125 \times 32 = 46$; width $32 e^{0.405} = 48$, height $32 e^{-0.288} = 24$. Corners $(58 - 24, 46 - 12, 58 + 24, 46 + 12) = (34, 34, 82, 58)$, exactly the ground truth.

</div>

## Step 5: remove duplicates with NMS

A trained detector fires on many neighbouring anchors for the same object, as the left of the first figure showed. **Non-maximum suppression** (NMS) keeps one box per cluster:

1. Drop candidates below a low score threshold (often 0.05) and keep the top 1,000 or so per pyramid level.
2. Sort the rest by score, highest first.
3. Take the highest-scoring box and keep it.
4. Delete every remaining box whose IoU with it exceeds a threshold, commonly 0.5.
5. Repeat from step 3 until no candidates remain.

<div class="ex">
<div class="lab">Worked example</div>

**NMS on four boxes**

Boxes $A = (10, 10, 50, 50)$ score 0.9, $B = (12, 12, 52, 52)$ score 0.8, $C = (60, 10, 100, 50)$ score 0.7, $D = (30, 30, 70, 70)$ score 0.6. Threshold 0.5.

Keep $A$. IoU of $A$ and $B$: intersection $(12, 12)$ to $(50, 50)$ is $38 \times 38 = 1444$; union $1600 + 1600 - 1444 = 1756$; IoU $0.822 > 0.5$, so delete $B$.

$A$ and $C$ do not overlap (IoU 0), so $C$ stays. $A$ and $D$: intersection $20 \times 20 = 400$, union $3200 - 400 = 2800$, IoU $0.143$, so $D$ stays.

Next highest left is $C$: keep it. IoU of $C$ and $D$: intersection $10 \times 20 = 200$, union $3000$, IoU $0.067$, so $D$ stays. Finally keep $D$.

Result: $A$, $C$, $D$. Only the near-copy $B$ was removed.

</div>

NMS runs **per class**, so a person box never deletes an overlapping bicycle. A common trick shifts each class's boxes by a large class-dependent offset so one call handles all classes. The loop is $O(N^2)$ in the worst case, hence the top-$k$ cut in step 1.

## Detection in code

The NumPy below is the whole toolkit from steps 2 to 5. In production you would call `torchvision.ops.box_iou` and `torchvision.ops.nms` (or `batched_nms` for the per-class version), which expect corner format.

```python
import numpy as np

def box_area(b):                      # b: (N, 4) as x1, y1, x2, y2
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

def iou_matrix(a, b):                 # a: (N, 4), b: (M, 4) -> (N, M)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])   # top-left of each overlap
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])   # bottom-right of each overlap
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    return inter / (box_area(a)[:, None] + box_area(b)[None, :] - inter)

def nms(boxes, scores, thr=0.5):
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(int(i))
        order = rest[iou_matrix(boxes[[i]], boxes[rest])[0] <= thr]
    return keep

boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [60, 10, 100, 50], [30, 30, 70, 70.]])
scores = np.array([0.9, 0.8, 0.7, 0.6])
print(iou_matrix(boxes, boxes).round(3)[0])   # [1.    0.822 0.    0.143]
print(nms(boxes, scores))                     # [0, 2, 3]

def to_cxcywh(b):
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]

def encode(gt, anc):                  # corners in, (tx, ty, tw, th) out
    g, a = to_cxcywh(gt), to_cxcywh(anc)
    return np.c_[(g[:, :2] - a[:, :2]) / a[:, 2:], np.log(g[:, 2:] / a[:, 2:])]

def decode(t, anc, clip=np.log(1000 / 16)):
    a = to_cxcywh(anc)
    c = a[:, :2] + t[:, :2] * a[:, 2:]
    wh = a[:, 2:] * np.exp(np.minimum(t[:, 2:], clip))
    return np.c_[c - wh / 2, c + wh / 2]

anchor, gt = np.array([[34, 34, 66, 66.]]), np.array([[34, 34, 82, 58.]])
t = encode(gt, anchor)
print(t.round(3))                     # [[ 0.25  -0.125  0.405 -0.288]]
print(decode(t, anchor))              # [[34. 34. 82. 58.]]
```

The last line is a cheap test worth keeping in any detection codebase: decode(encode(gt)) must return gt exactly.

## Reading the result: precision, recall and mAP

A detection is a **true positive** if it overlaps a not-yet-matched ground-truth box of its class with IoU at or above a threshold; otherwise it is a **false positive** (wrong place, wrong class, or a second box on a found object). Unmatched ground-truth boxes are **false negatives**.

Sort all detections in the test set by score and walk down the list. After each one compute **precision** (the share of detections so far that were correct) and **recall** (the share of all true objects found so far). Plotting one against the other gives the precision-recall curve, and the area under it is the **average precision (AP)** for that class.

~~~dg-fig 04.02-pr.svg
Precision-recall curves for one simulated detector on 150 synthetic images, scored twice. Requiring IoU of at least 0.75 instead of 0.5 turns loosely placed boxes into false positives, and AP falls from 0.93 to 0.57.
~~~

Benchmarks differ in the IoU threshold they use:

- Pascal VOC reports AP at IoU 0.5 ("AP50"), averaged over classes into **mAP**.
- COCO's headline number averages AP over ten IoU thresholds, 0.50, 0.55, up to 0.95, and over all classes. It rewards tight boxes, not just finding the object. For the simulated detector in the figure that average is 0.55, even though its AP50 is 0.93.

Check which one a paper reports; they are not comparable. A big gap between AP50 and AP75 means objects are found but boxes are placed loosely, which points at box regression.

## What goes wrong

NMS has a blind spot built into its rules: it cannot tell a duplicate from a neighbour. Two people standing shoulder to shoulder can overlap with IoU above 0.5, and the lower-scoring one is deleted along with the true duplicates.

~~~dg-fig 04.02-nms-crowd.svg
Sixty simulated crowd scenes of nine overlapping people. With a threshold of 0.3, NMS deletes about a fifth of the real people; at 0.7 it leaves about 1.5 duplicate boxes per person. The default 0.5 is a compromise that works only when crowds are not too dense.
~~~

No single threshold is right for every scene. Two standard remedies: **Soft-NMS** multiplies the scores of overlapping boxes by a decay (linear or Gaussian in IoU) instead of setting them to zero, so a neighbour survives with a lower score; and set-prediction detectors such as DETR (lesson 04.03) learn one box per object and need no NMS at all.

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Boxes are shifted or stretched in a consistent way | box format mixed up, e.g. $(x, y, w, h)$ read as corners | convert once at load time; test that decode(encode(gt)) returns gt
| Neighbouring objects in crowds go missing | NMS threshold too low | raise it to 0.6-0.7, or use Soft-NMS
| Several boxes per object at test time | NMS threshold too high, or NMS run across classes the wrong way | lower the threshold; use per-class (batched) NMS
| Small objects never detected | no anchor reaches IoU 0.5 with them | add smaller anchor sizes, a finer pyramid level, or best-anchor matching
| Loss becomes NaN early in training | $e^{t_w}$ overflow from an unclamped prediction | clamp $t_w, t_h$ before exponentiating
| AP50 high but COCO mAP low | boxes found but loosely placed | better regression loss (IoU-based, lesson 04.03), more positive anchors

## Going deeper

**Anchor design is a prior.** Anchors encode a guess about object sizes and shapes. If your objects are long and thin (pens, power lines) and every anchor is roughly square, none reaches IoU 0.5 and only the best-anchor rule gives them a positive. YOLOv2 picked its anchor shapes by running k-means on the training boxes with $1 - \text{IoU}$ as the distance, which is a good recipe for any custom dataset.

**Anchor-free detectors.** FCOS predicts, at each location inside an object, the distances to the four edges plus a "centre-ness" score; CenterNet predicts a heatmap of object centres plus a size at each peak. Anchor hyperparameters disappear, but IoU, encoding (relative to the stride) and usually NMS remain.

**IoU's blind spot.** IoU is exactly 0 for any two disjoint boxes, however far apart, so it gives no gradient there. Generalised IoU (GIoU) and its relatives fix this and become losses in lesson 04.03.

**NMS speed.** Greedy NMS is sequential, which suits GPUs poorly. Fast NMS (from YOLACT) and Matrix NMS (from SOLOv2) relax the greedy rule into pure matrix operations, at a small cost in accuracy.

**Evaluation details.** COCO samples precision at 101 recall points after making the curve non-increasing, keeps at most 100 detections per image, and also reports AP by object size (small, medium, large), which is often where a model's weaknesses show.

## Key terms

| Term | Meaning
| --- | --- |
| Bounding box | The smallest axis-aligned rectangle around an object, written as corners or centre plus size
| IoU | Intersection area divided by union area; 1 for identical boxes, 0 for disjoint ones
| Anchor | A fixed reference box at a known position, size and shape that the network corrects
| Stride | How many input pixels one feature-map cell spans; sets the spacing of anchors
| Positive / negative / ignored | Anchor labels from IoU with the ground truth: on an object, background, or too ambiguous to train on
| Box encoding | Writing a target box as a centre shift in anchor units and a log size ratio
| NMS | Greedy removal of boxes that overlap a higher-scoring box beyond an IoU threshold
| AP / mAP | Area under the precision-recall curve for one class / its mean over classes (and, on COCO, over IoU thresholds)

<div class="take">
<div class="lab">Key takeaway</div>

a dense detector predicts a class score and an encoded correction for every anchor, learns from anchors labelled by IoU, and relies on NMS to keep one box per object; most detection bugs are box-format, encoding or threshold mistakes rather than architecture problems.

</div>
