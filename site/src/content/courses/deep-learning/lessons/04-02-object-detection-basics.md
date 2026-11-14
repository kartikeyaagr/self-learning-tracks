---
id: "04.02"
course: "deep-learning"
module: "04"
slug: "object-detection-basics"
title: "Object detection basics"
keyIdeas: ["A detector outputs a variable-size set of (box, class, score). Dense detectors predict at every feature-map location and then prune.", "IoU, intersection over union, is the yardstick for everything: matching anchors to ground truth, deciding duplicates in NMS, and scoring at evaluation (mAP at IoU 0.5 to 0.95).", "Anchors are reference boxes tiled over the feature map at several scales and aspect ratios. The network predicts a class and a small correction for each one.", "Boxes are regressed as encoded offsets: centre shift divided by anchor size, and log of the size ratio. This makes targets scale-invariant and keeps widths positive.", "Non-maximum suppression keeps the highest-scoring box and removes others that overlap it beyond a threshold. It is greedy and can delete real objects in crowds."]
question: "Your detector misses one of two people standing shoulder to shoulder, even though before NMS both had high-scoring boxes. Explain what happened and give two fixes with their costs."
answer: "The two people's boxes overlap with IoU above the NMS threshold (say 0.5). NMS keeps the higher-scoring box and deletes every candidate overlapping it beyond the threshold, including the correct box for the second person, because NMS cannot tell a duplicate from a neighbour. Fix 1: raise the threshold (0.6-0.7). This keeps neighbours but lets more true duplicates survive, lowering precision on isolated objects. Fix 2: Soft-NMS, which multiplies overlapping scores by a decay (linear or Gaussian in IoU) instead of zeroing them, so the second box survives with a reduced score and can still be ranked; costs a little more compute and a new hyperparameter. A structural fix is a set-prediction detector (DETR) trained with one-to-one matching, which needs no NMS but trains much more slowly."
sources: [{"title": "Faster R-CNN (Ren et al., 2015)", "url": "https://arxiv.org/abs/1506.01497", "note": "Introduces anchors and the region proposal network; section 3.1 defines the box encoding used almost everywhere."}, {"title": "Dive into Deep Learning: Anchor Boxes", "url": "https://d2l.ai/chapter_computer-vision/anchor.html", "note": "Code for generating anchors, IoU matrices, labelling anchors and NMS."}, {"title": "FCOS: Fully Convolutional One-Stage Object Detection", "url": "https://arxiv.org/abs/1904.01355", "note": "The anchor-free alternative; useful contrast for what anchors actually buy you."}, {"title": "torchvision.ops.nms", "url": "https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html", "note": "The production NMS interface and its box-format expectations."}]
practice: ["iou-matrix", "non-maximum-suppression", "box-encode-and-decode", "anchor-box-generation", "soft-non-maximum-suppression"]
tags: ["object detection", "IoU"]
tools: ["NumPy"]
---

## The detection problem

Classification answers "what is in this image". Detection answers "what is where": a list of boxes, each with a class and a confidence. The list has variable length, which is awkward for a network with fixed-size outputs. The standard trick is to predict far too many candidates at fixed positions and then prune.

Two families do this:

- **Two-stage** (Faster R-CNN): a region proposal network suggests about 1,000-2,000 boxes, then a second head crops features for each (RoIAlign) and classifies and refines it. Accurate, slower.
- **One-stage** (SSD, RetinaNet, YOLO): a single dense head predicts class scores and box offsets at every feature-map cell. Faster and simpler; with a good loss (next lesson) as accurate.

Anchor-free detectors such as FCOS skip anchors and predict, at each location, the distances to the four box edges. The rest of this lesson still applies: IoU, encoding and NMS.

~~~dg-flow A one-stage detector: dense predictions everywhere, then decode and prune.
Image -> Backbone + FPN -(per cell)-> *Dense head -(decode offsets)-> Candidate boxes -(score > 0.05)-> NMS -> Detections
~~~

## Box formats and IoU

Boxes appear as corners $(x_1, y_1, x_2, y_2)$, as corner plus size $(x, y, w, h)$, or as centre plus size $(c_x, c_y, w, h)$, in pixels or normalised to $[0, 1]$. Mixing formats is a top-three detection bug, so convert at the edges and keep one format inside.

IoU between boxes $A$ and $B$ is $|A \cap B| / |A \cup B|$. The intersection's corners are the max of the top-left corners and the min of the bottom-right corners, clamped at zero width and height when the boxes do not overlap. For $N$ predictions and $M$ ground-truth boxes, compute the full $N \times M$ matrix with broadcasting; it feeds both matching and evaluation.

<div class="ex">
<div class="lab">Worked example</div>

**IoU of two offset squares**

$A = (0, 0, 10, 10)$ and $B = (5, 5, 15, 15)$. Intersection corners: $(5, 5)$ to $(10, 10)$, area 25.

Union: $100 + 100 - 25 = 175$. IoU $= 25/175 = 0.143$. Half-overlapping on each axis gives a low IoU, which is why 0.5 is already a demanding threshold.

</div>

## Anchors and matching

At each cell of a feature map with stride $s$, place $A$ anchors centred on the cell, for example 3 scales × 3 aspect ratios (1:2, 1:1, 2:1). RetinaNet uses 9 anchors per location on pyramid levels with strides 8 to 128. On an 800×800 image the stride-8 level alone is 100×100 cells × 9 = 90,000 anchors; the whole pyramid has about 120,000.

During training each anchor gets a target by IoU with the ground truth:

- IoU ≥ 0.5 with some box: positive, assigned to the box it overlaps most.
- IoU < 0.4 with every box: negative (background).
- In between: ignored, so ambiguous anchors do not send noisy gradients.
- Each ground-truth box is also matched to its single best anchor, so small or oddly shaped objects are never left without a positive.

The result is extreme imbalance: a typical image has a handful of objects and tens of thousands of background anchors. The next lesson's focal loss exists because of this.

## Encoding box targets

Rather than predicting raw coordinates, the head predicts offsets relative to its anchor $(x_a, y_a, w_a, h_a)$:

$$t_x = \frac{x - x_a}{w_a}, \quad t_y = \frac{y - y_a}{h_a}, \quad t_w = \log\frac{w}{w_a}, \quad t_h = \log\frac{h}{h_a}$$

Dividing by anchor size makes a 4-pixel shift on a 16-pixel anchor and a 32-pixel shift on a 128-pixel anchor the same target, so one regression head works across scales. The log makes widths positive after decoding ($w = w_a e^{t_w}$) and turns "twice as wide" and "half as wide" into symmetric targets $\pm 0.69$. Implementations often divide by fixed variances such as $(0.1, 0.1, 0.2, 0.2)$ so targets have roughly unit scale, and clamp $t_w, t_h$ at about $\log(1000/16) \approx 4.1$ before `exp` so a bad prediction cannot overflow.

## Non-maximum suppression

A good detector fires on many neighbouring anchors for the same object. NMS turns that cluster into one box:

1. Sort candidates by score.
2. Take the highest, keep it.
3. Remove every remaining candidate whose IoU with it exceeds a threshold (commonly 0.5).
4. Repeat until no candidates remain.

```python
import numpy as np
def nms(boxes, scores, thr=0.5):           # boxes: (N, 4) x1, y1, x2, y2
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        tl = np.maximum(boxes[i, :2], boxes[rest, :2])
        br = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.prod(np.clip(br - tl, 0, None), axis=1)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (area(boxes[[i]]) + area(boxes[rest]) - inter)
        order = rest[iou <= thr]
    return keep
```

Run it per class; a common trick is to offset each class's boxes by a large class-dependent constant so one call handles all classes without cross-class suppression. Worst case is $O(N^2)$, so detectors keep only the top 1,000 candidates per level before NMS.

NMS is greedy and blind: two people standing shoulder to shoulder can have IoU above 0.5, and the lower-scoring one is deleted. Soft-NMS decays overlapping scores instead of zeroing them; set-prediction detectors (DETR, next lesson) avoid NMS entirely.

<div class="take">
<div class="lab">Key takeaway</div>

dense detectors predict a correction for every anchor, label anchors by IoU, regress scale-invariant offsets, and rely on NMS to remove duplicates. Most detection bugs are box-format, encoding or threshold mistakes, not architecture.

</div>
