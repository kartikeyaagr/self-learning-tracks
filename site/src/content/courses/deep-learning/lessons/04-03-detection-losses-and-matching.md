---
id: "04.03"
course: "deep-learning"
module: "04"
slug: "detection-losses-and-matching"
title: "Detection losses and matching"
keyIdeas: ["Dense detectors see about 1,000 background anchors per object. Plain cross-entropy is dominated by easy negatives; focal loss multiplies each term by $(1-p_t)^\\gamma$ (with $\\gamma = 2$, $\\alpha = 0.25$) so confident, correct predictions contribute almost nothing, and the total is divided by the number of positive anchors.", "Initialise the classification bias to a prior of about 0.01 foreground ($b \\approx -4.6$). Starting at 0 makes every anchor predict 0.5 and gives a first loss in the hundreds instead of about 1.", "L1 loss on box coordinates does not track IoU: the same 4-pixel error is IoU 0.67 on a small object and 0.96 on a large one. IoU-based losses fix that, but plain $1 - \\text{IoU}$ has zero gradient when boxes do not overlap.", "GIoU adds a penalty for empty space in the enclosing box, DIoU a penalty for centre distance, CIoU also an aspect-ratio term. GIoU equals IoU when one box contains the other, which DIoU fixes. In practice an IoU-family loss is combined with L1.", "DETR treats detection as set prediction: a fixed set of queries is matched one-to-one to ground truth by the Hungarian algorithm on a cost of class probability, L1 and GIoU; unmatched queries learn 'no object'. One-to-one matching is why it needs no NMS, and also why it trains slowly."]
question: "You are training a RetinaNet-style detector for small defects on circuit-board images: about 120,000 anchors per image and usually 0 to 3 defects. With plain binary cross-entropy and smooth-L1, the model predicts 'background' everywhere and recall is near zero. A teammate switches to focal loss but the very first iterations show a loss around 300 and soon NaN. After that is fixed, recall improves but many predicted boxes sit next to defects without overlapping them, and for tiny defects the box loss barely decreases. Explain each of the three problems and what you would change. Then: the team considers moving to a DETR-style model to remove NMS. What would they gain, what would it cost, and what must be true about the number of queries?"
answer: "Problem 1 is imbalance: with roughly 120,000 anchors and at most 3 positives, the cross-entropy sum is dominated by easy background anchors whose tiny per-anchor losses add up to far more than the positives' losses, so the cheapest solution is to call everything background. Focal loss multiplies each term by $\\alpha_t(1-p_t)^\\gamma$; with $\\gamma = 2$ an easy negative at $p_t = 0.99$ is down-weighted by $10^{-4}$ while a hard example keeps about half its weight, and the sum is divided by the number of positive anchors (clamped at 1 so images without defects do not divide by zero). Problem 2 is initialisation: with the final bias at 0 every anchor predicts 0.5, so all 120,000 negatives produce a sizable loss at once, around a few hundred after normalising by a handful of positives, which yields huge gradients and NaNs. Set the bias to $-\\log((1-\\pi)/\\pi)$ with $\\pi = 0.01$, about -4.6, and make sure a generic weight-init loop does not overwrite it. Problem 3 is the box loss: smooth-L1 on offsets is not aligned with IoU and is especially poor for tiny objects, where a pixel or two of error is a large IoU loss, and a plain 1 - IoU loss would give no gradient for boxes that do not overlap their target. Use GIoU or, better here, DIoU/CIoU (the centre-distance term pulls nearby but disjoint boxes straight onto the defect and also handles the containment case where GIoU reduces to IoU), combined with an L1 term. For DETR: one-to-one Hungarian matching means only one query is rewarded per defect and all others learn 'no object', so duplicates are penalised during training and NMS is unnecessary, which helps when two defects are close together and NMS would suppress one of them. The costs are much longer training (the original DETR needed about 500 COCO epochs; variants like Deformable DETR or denoising training reduce this), historically weaker small-object performance in the original design, and per-image matching overhead. The number of queries must exceed the maximum number of objects that can appear in an image, otherwise some defects can never be matched."
sources: [{"title": "Focal Loss for Dense Object Detection (Lin et al., 2017)", "url": "https://arxiv.org/abs/1708.02002", "note": "RetinaNet and focal loss; section 3 for the loss and section 4.1 for the prior-based bias initialisation."}, {"title": "Generalized Intersection over Union (Rezatofighi et al., 2019)", "url": "https://arxiv.org/abs/1902.09630", "note": "Defines GIoU and shows why IoU is a better regression target than L1."}, {"title": "Distance-IoU Loss (Zheng et al., 2019)", "url": "https://arxiv.org/abs/1911.08287", "note": "DIoU and CIoU, with the containment failure case of GIoU illustrated."}, {"title": "End-to-End Object Detection with Transformers (DETR)", "url": "https://arxiv.org/abs/2005.12872", "note": "Set prediction and the Hungarian matching cost; section 3.1."}]
practice: ["sigmoid-focal-loss", "hungarian-matching", "generalized-iou-loss", "complete-iou-loss"]
tags: ["focal loss", "bipartite matching"]
tools: ["NumPy", "SciPy"]
---

## What you will learn

- Why plain cross-entropy fails when 99,950 background anchors face 50 object anchors.
- What focal loss does, term by term, and why $\alpha$ and the bias initialisation matter as much as $\gamma$.
- Why L1 on box coordinates does not track IoU, and how IoU, GIoU, DIoU and CIoU losses fix it step by step.
- How DETR matches predictions to objects one-to-one with the Hungarian algorithm, and why that removes NMS.
- How to implement all three in PyTorch and SciPy, and which numbers to watch while training.

*Before you start:* you need lesson 04.02 (anchors, IoU, box encoding, NMS) and binary cross-entropy with a sigmoid output (module 01, where focal loss first appeared as a general-purpose loss). Some PyTorch autograd helps for the code.

## Two problems the previous lesson left open

Lesson 04.02 built a dense detector: about 100,000 anchors per image, labelled by IoU, regressing encoded offsets, with NMS at the end. Three training problems remain. **Imbalance**: perhaps 50 anchors are positive, and the loss is a sum, so background wins by numbers. **The wrong regression target**: offsets are trained with L1, but the detector is judged by IoU. **Duplicates by design**: every anchor above IoU 0.5 is trained to fire, so NMS must clean up. This lesson covers focal loss for the first, IoU-based losses for the second, and DETR's Hungarian matching for the third.

## The idea in one picture

Take one training image and score its anchors with a half-trained detector. Write $p_t$ for the probability the model gives to the *correct* answer for an anchor: $p_t = p$ for a positive anchor (the object probability) and $p_t = 1 - p$ for a negative one. An anchor with $p_t$ close to 1 is an **easy example**: the model already gets it right. One with small $p_t$ is a **hard example**.

| Group | Anchors | Typical $p_t$ | Summed cross-entropy | Share
| --- | --- | --- | --- | --- |
| Easy negatives | 99,000 | 0.95 to 0.999 | 2,570 | 82%
| Harder negatives | 950 | 0.3 to 0.95 | 499 | 16%
| Positives (objects) | 50 | 0.1 to 0.7 | 51 | 1.6%

The 50 object anchors contribute under 2% of the loss; most of the gradient goes into making confident background slightly more confident.

~~~dg-fig 04.03-focal.svg
Left: per-anchor loss against the probability of the correct class. Cross-entropy (γ = 0) still charges something in the easy zone on the right; with γ = 2 that zone is almost free. Right: on the same 100,000 simulated anchors, easy negatives make up 82% of the cross-entropy total but only 2% of the focal-loss total.
~~~

**Focal loss** reweights each anchor by how wrong it is. With standard settings the easy negatives' share drops from 82% to 2%, the confusing background patches get 92%, and the positives rise from 1.6% to 6%. (Shares come from the simulated $p_t$ values above; real images differ in detail, not direction.)

## Step 1: why cross-entropy is swamped

Binary cross-entropy for one anchor is

$$\text{CE}(p_t) = -\log p_t.$$

It is never zero. An easy negative at $p_t = 0.99$ costs $0.010$, but 100,000 of them sum to about 1,000, while 50 hard positives at $p_t = 0.3$ add only $50 \times 1.20 = 60$.

Two-stage detectors dodge this by sampling a fixed mix of positives and negatives; earlier one-stage detectors used **hard negative mining** (keep only the highest-loss negatives). Focal loss keeps every anchor and lets a weight do the work.

## Step 2: focal loss

Multiply cross-entropy by a factor that shrinks as the example gets easier:

$$\text{FL}(p_t) = -\alpha_t\,(1 - p_t)^{\gamma}\,\log p_t.$$

What each symbol means:

- $(1 - p_t)^\gamma$ is the **modulating factor**. If the model is confident and right ($p_t \to 1$) it goes to 0; if the model is wrong ($p_t$ small) it stays near 1, so the loss is nearly ordinary cross-entropy.
- $\gamma \ge 0$ is the **focusing parameter**. $\gamma = 0$ gives back cross-entropy; larger $\gamma$ silences easy examples more aggressively. The paper's default is $\gamma = 2$.
- $\alpha_t$ is a class weight: $\alpha$ for positives and $1 - \alpha$ for negatives. The paper's default is $\alpha = 0.25$.

Number check with $\gamma = 2$: an easy negative at $p_t = 0.99$ gets weight $(0.01)^2 = 10^{-4}$, so it now counts ten thousand times less. A hard example at $p_t = 0.3$ keeps weight $(0.7)^2 = 0.49$, about half.

$\alpha = 0.25$ looks backwards, since it gives *positives* the smaller weight; but $\gamma$ has already crushed the easy negatives, and in the paper's experiments the best $\alpha$ with $\gamma = 2$ was below 0.5. Tune the two together.

Finally, the summed loss is divided by the **number of positive anchors**, not the number of anchors: after focal weighting the negatives contribute little, so the positive count is a stable measure of how much signal the image holds.

<div class="ex">
<div class="lab">Worked example</div>

**how much each anchor counts**

Take $\gamma = 2$, $\alpha = 0.25$. An easy negative has $p_t = 0.99$ and $\alpha_t = 1 - 0.25 = 0.75$. A hard positive has $p_t = 0.2$ and $\alpha_t = 0.25$.

Easy negative, cross-entropy: $-\log 0.99 = 0.01005$.

Easy negative, focal: $0.75 \times (1 - 0.99)^2 \times 0.01005 = 0.75 \times 0.0001 \times 0.01005 = 7.5 \times 10^{-7}$.

Hard positive, cross-entropy: $-\log 0.2 = 1.609$.

Hard positive, focal: $0.25 \times (1 - 0.2)^2 \times 1.609 = 0.25 \times 0.64 \times 1.609 = 0.258$.

Ratio of hard to easy contribution: under cross-entropy $1.609 / 0.01005 \approx 160$; under focal loss $0.258 / (7.5 \times 10^{-7}) \approx 340{,}000$. One hard positive now outweighs about 340,000 easy negatives instead of 160.

</div>

## Step 3: start the classifier at the right prior

The other half of the focal-loss recipe is easy to miss. If the last classification layer starts with bias 0, every anchor predicts $p = 0.5$ at step 0. Then all 99,950 negatives are "medium-hard" and the first loss is huge.

The fix is to initialise the bias so every anchor starts by predicting a small object probability $\pi$:

$$b = -\log\frac{1 - \pi}{\pi}, \qquad \pi = 0.01 \;\Rightarrow\; b = -\log 99 \approx -4.6.$$

Here $b$ is the final layer's bias and $\pi$ is the **prior**, a rough guess of the foreground fraction. That layer's weights start near zero, so every logit is about $b$ and $\sigma(-4.6) \approx 0.01$.

Number check, from the code below on 99,950 negatives and 50 positives: with $b = 0$ the initial focal loss is **259.8**; with $b = -4.6$ it is **1.13**. The first produces enormous early gradients that can destabilise training; the second starts calmly.

## Step 4: why L1 on boxes is not enough

Lesson 04.02 trained box offsets with L1 (or smooth-L1). L1 treats the four coordinates as four unrelated numbers and adds their errors. IoU, the thing we evaluate, depends on how those errors interact with the box's size.

~~~dg-fig 04.03-l1-vs-iou.svg
Both predictions are shifted 2 pixels right, so $x_1$ and $x_2$ are each off by 2 and the L1 error is 4 pixels in both panels. On a 10-pixel object that is an IoU of 0.67; on a 100-pixel object it is 0.96.
~~~

Encoding offsets by anchor size (04.02) softens this, but the coordinates are still optimised independently, so lowering L1 does not always raise IoU. The natural fix is to train on IoU itself, with loss $1 - \text{IoU}$. That works while boxes overlap, and breaks when they do not.

## Step 5: the IoU family of losses

When boxes do not overlap, IoU is 0 however far apart they are, so $1 - \text{IoU}$ is flat and gives no direction. Disjoint boxes are common early in training and for small objects.

~~~dg-fig 04.03-iou-landscape.svg
Left: a 10×10 prediction slides away from its 10×10 target. Once the boxes stop overlapping, 1 − IoU is stuck at 1; GIoU and DIoU keep rising, so their gradient still points home. Right: a 6×6 box inside the target. IoU and GIoU are the same at every position; only DIoU prefers the centred box.
~~~

Each newer loss adds one penalty term to IoU. Write $A$ for the prediction, $B$ for the target, and $C$ for the **enclosing box**, the smallest box that covers both.

| Loss | Score being maximised | What the extra term fixes
| --- | --- | --- |
| IoU | $\text{IoU} = \lvert A \cap B\rvert / \lvert A \cup B\rvert$ | aligns training with the metric; zero gradient when disjoint
| GIoU | $\text{IoU} - \lvert C \setminus (A \cup B)\rvert / \lvert C\rvert$ | wasted space in $C$ shrinks as boxes approach; range $[-1, 1]$
| DIoU | $\text{IoU} - \rho^2(b, b^{gt}) / c^2$ | squared centre distance $\rho^2$ over the squared diagonal $c^2$ of $C$; pulls centres directly
| CIoU | DIoU score $- \alpha v$, $v = \frac{4}{\pi^2}\big(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\big)^2$ | also matches aspect ratio; $\alpha = v / ((1 - \text{IoU}) + v)$

In every case the loss is $1 - \text{score}$. In plain words: GIoU penalises "empty space between the boxes", DIoU penalises "distance between the centres", and CIoU adds "difference in shape".

<div class="ex">
<div class="lab">Worked example</div>

**GIoU and DIoU by hand**

Target $B = (0, 0, 10, 10)$, corners $(x_1, y_1, x_2, y_2)$. Disjoint prediction $A = (15, 0, 25, 10)$.

Intersection: the x-ranges $[0, 10]$ and $[15, 25]$ do not meet, so it is 0 and IoU $= 0$. Union $= 100 + 100 - 0 = 200$.

Enclosing box $C = (0, 0, 25, 10)$, area $250$. Empty part of $C$: $250 - 200 = 50$. GIoU $= 0 - 50/250 = -0.2$, so the GIoU loss is $1.2$, not stuck at 1.

DIoU: centres $(5, 5)$ and $(20, 5)$, so $\rho^2 = 15^2 = 225$. Diagonal of $C$: $c^2 = 25^2 + 10^2 = 725$. DIoU $= 0 - 225/725 = -0.310$, loss $1.310$.

Now the overlapping pair from 04.02, $A = (5, 5, 15, 15)$: IoU $= 25/175 = 0.143$. $C = (0, 0, 15, 15)$, area 225, empty part $225 - 175 = 50$, so GIoU $= 0.143 - 50/225 = 0.143 - 0.222 = -0.079$. Centres $(5,5)$ and $(10,10)$: $\rho^2 = 50$, $c^2 = 450$, DIoU $= 0.143 - 0.111 = 0.032$.

</div>

GIoU's blind spot: when one box contains the other, $C$ is the larger box, the empty-space term is 0, and GIoU equals IoU. In the figure's right panel a 6×6 box inside the target has IoU = GIoU = 0.36 anywhere; DIoU is 0.36 centred and 0.32 in a corner.

To see the effect on learning, we trained one box with each loss (Adam) and tracked its IoU.

~~~dg-fig 04.03-train.svg
One box trained by gradient descent with each loss. Left: from a disjoint start, plain IoU never moves; GIoU, DIoU and CIoU all reach IoU above 0.9 within about 220 steps. Right: from a small off-centre box inside the target, GIoU traces the IoU curve exactly, while DIoU and CIoU reach IoU 0.9 a little sooner (step 76 vs 80 to 86).
~~~

In this single toy run GIoU passes IoU 0.5 first (step 136 vs 171 for DIoU); the DIoU paper's faster-convergence claim is an average over many simulated box pairs, so treat one run as an illustration. The robust lesson: plain IoU cannot recover disjoint boxes, and the extra terms can. In practice detectors combine an IoU-family loss with an L1 term (DETR uses L1 plus GIoU).

## Step 6: from IoU thresholds to one-to-one matching

Every loss needs to know which prediction matches which object. Dense detectors use IoU thresholds, which is **many-to-one** matching: the source of duplicates and of NMS.

**DETR** (DEtection TRansformer) treats detection as **set prediction**. A transformer decoder turns $N = 100$ learned **object queries** into 100 (class, box) pairs, with an extra "no object" class. Training first finds the best **one-to-one assignment** of predictions to the $M$ objects:

$$\hat\sigma = \arg\min_{\sigma} \sum_{i=1}^{M} \Big[-\hat p_{\sigma(i)}(c_i) + \lambda_{L1}\lVert b_i - \hat b_{\sigma(i)}\rVert_1 + \lambda_{giou}\big(1 - \text{GIoU}(b_i, \hat b_{\sigma(i)})\big)\Big]$$

Here $i$ runs over objects with class $c_i$ and box $b_i$; $\sigma(i)$ is the query assigned to object $i$, no query used twice; $\hat p_{\sigma(i)}(c_i)$ is that query's probability for the correct class; and $\lambda_{L1} = 5$, $\lambda_{giou} = 2$ are the paper's weights.

This is the classic **assignment problem**, solved exactly by the **Hungarian algorithm** in $O(N^3)$, cheap for 100 queries. Matched queries then get classification and box losses (L1 plus GIoU); unmatched queries are trained to predict "no object", with that class weighted 0.1 because most queries are unmatched.

~~~dg-fig 04.03-matching.svg
Two objects and three queries. The cost matrix combines class probability, L1 distance and GIoU. The cheapest one-to-one assignment gives q1 to the person and q3 to the dog (total 0.47); q2, which straddles both, is matched to nothing and learns to say "no object".
~~~

<div class="ex">
<div class="lab">Worked example</div>

**matching three queries to two objects**

Costs from the figure (rows are queries, columns are person and dog): q1 $= (0.28, 7.28)$, q2 $= (4.04, 3.95)$, q3 $= (7.62, 0.19)$.

Check one entry. q1 gives the person class probability 0.70. Its L1 distance to the person box, on corner coordinates in $[0, 1]$, is $0.02 + 0.05 + 0.02 + 0.02 = 0.11$, and its GIoU with it is 0.787. Cost $= -0.70 + 5 \times 0.11 + 2 \times (1 - 0.787) = -0.70 + 0.55 + 0.43 = 0.28$. (DETR itself measures L1 on normalised centre-and-size boxes; corners keep the arithmetic short here.)

All six one-to-one options: (q1, q2) $= 0.28 + 3.95 = 4.23$; (q1, q3) $= 0.28 + 0.19 = 0.47$; (q2, q1) $= 4.04 + 7.28 = 11.32$; (q2, q3) $= 4.04 + 0.19 = 4.23$; (q3, q1) $= 7.62 + 7.28 = 14.90$; (q3, q2) $= 7.62 + 3.95 = 11.57$.

The minimum is 0.47: q1 takes the person, q3 the dog, q2 becomes "no object". Enumeration grows factorially; the Hungarian algorithm reaches the same answer in polynomial time.

</div>

~~~dg-flow Many-to-one matching needs NMS at test time; one-to-one matching does not.
[Dense detector] Anchors -(IoU > 0.5)-> Many anchors per object -> *NMS at test time
[DETR] 100 queries -(Hungarian, one-to-one)-> One query per object -> *No NMS
~~~

Why no NMS? Under IoU matching, duplicates are the intended output. Under one-to-one matching, a second query that also predicts the person is unmatched, so it is trained towards "no object": duplicates are penalised, and the queries learn to divide objects among themselves through decoder self-attention. The price: the original DETR needed about 500 epochs on COCO, against tens for Faster R-CNN, partly because the matching keeps changing early in training.

## Detection losses in code

All three pieces fit in a few lines. This uses PyTorch for the losses and SciPy for the assignment.

```python
import math, numpy as np, torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).sum() / targets.sum().clamp(min=1)

targets = torch.zeros(100_000); targets[:50] = 1          # 50 positives, 99,950 negatives
for b in [0.0, -math.log((1 - 0.01) / 0.01)]:
    logits = torch.full((100_000,), b)
    print(f"bias {b:.2f}: focal {sigmoid_focal_loss(logits, targets).item():.2f}")
# bias 0.00: focal 259.84
# bias -4.60: focal 1.13

def giou_loss(p, t):                                       # (N, 4) corner boxes
    lt, rb = torch.maximum(p[:, :2], t[:, :2]), torch.minimum(p[:, 2:], t[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area(p) + area(t) - inter
    hull = (torch.maximum(p[:, 2:], t[:, 2:]) - torch.minimum(p[:, :2], t[:, :2])).prod(1)
    return 1 - (inter / union - (hull - union) / hull)

gt = torch.tensor([[0., 0, 10, 10], [0., 0, 10, 10]])
pred = torch.tensor([[5., 5, 15, 15], [15., 0, 25, 10]], requires_grad=True)
loss = giou_loss(pred, gt); loss.sum().backward()
print(loss.detach())    # tensor([1.0794, 1.2000])
print(pred.grad[1])     # tensor([ 0.0400,  0.0000, -0.0080,  0.0000])  non-zero though disjoint

cost = np.array([[0.28, 7.28], [4.04, 3.95], [7.62, 0.19]])   # 3 queries x 2 objects
rows, cols = linear_sum_assignment(cost)
print(rows, cols, cost[rows, cols].sum())   # [0 2] [0 1] 0.47
```

The disjoint box's gradient is positive on $x_1$, so a descent step moves it towards the target, and negative on $x_2$, so it also grows (GIoU's habit of enlarging boxes). `linear_sum_assignment` accepts rectangular matrices and leaves extra queries unmatched. Production versions: `torchvision.ops.sigmoid_focal_loss` and `torchvision.ops.generalized_box_iou_loss`.

## Checking that it works

Loss curves alone hide most detection bugs. Log these from the first epoch:

- **Initial classification loss.** With the prior bias it should be around 1. Hundreds means the bias init was skipped or overwritten by a generic init loop.
- **Positives per image.** Zero for small objects means anchor sizes or IoU thresholds are wrong (04.02).
- **Mean IoU of matched pairs.** Should climb towards 0.7 and above. Flat while L1 falls usually means a box-format mix-up.
- **Duplicate rate (DETR).** Confident predictions overlapping another with IoU above 0.7 should fall towards zero.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Loss of several hundred at step 0, then NaN | classification bias left at 0 | set the bias to $-\log((1-\pi)/\pi)$ with $\pi = 0.01$
| Loss jumps between images | dividing by anchor count, or by positives with no clamp | divide by number of positives, clamped at 1 (often averaged across GPUs)
| Boxes stuck far from objects early in training | plain $1 - \text{IoU}$ loss on disjoint pairs | switch to GIoU/DIoU, keep an L1 term
| Small box inside a large target stops improving | GIoU containment blind spot | DIoU or CIoU, or add L1
| DETR outputs duplicates | too few epochs, or matching accidentally many-to-one | train longer, check the matcher returns one pair per object
| DETR misses many objects | fewer queries than objects in some images | raise $N$ above the maximum object count

## Going deeper

**Focal loss gradient.** For a positive with $p = \sigma(z)$, cross-entropy's gradient with respect to the logit is $p - 1$, so an example at $p = 0.99$ still pushes with magnitude 0.01. Focal loss's gradient carries the factor $(1-p)^\gamma$ and vanishes faster than linearly as $p \to 1$; in the figure, the 99,000 easy negatives total about 2.4 instead of 2,570. Quality Focal Loss and VarifocalNet keep the idea but replace the 0/1 target with the box's IoU, so scores rank boxes by localisation quality.

**Smarter assignment for dense detectors.** ATSS sets each object's IoU threshold from the statistics of its nearest anchors; YOLOX's SimOTA takes the cheapest few anchors under a DETR-like cost. Both remain many-to-one, so NMS stays.

**Making one-to-one matching train faster.** Each query gets a positive signal rarely, and ownership can flip between steps. Standard fixes: auxiliary losses after every decoder layer (already in DETR); deformable attention over a few sampled points (Deformable DETR reports better accuracy with about ten times fewer epochs); denoising queries built from noisy ground-truth boxes (DN-DETR, DINO); and hybrid schemes that add a one-to-many branch during training only.

**The matcher itself.** The Hungarian algorithm finds a minimum-cost perfect matching in a bipartite graph in $O(N^3)$. DETR runs it per image without gradients: it only decides which target each query is trained on. CIoU's weight $\alpha = v/((1-\text{IoU})+v)$ grows as IoU rises, so shape is fixed after position.

## Key terms

| Term | Meaning
| --- | --- |
| $p_t$ | Probability the model gives to the correct answer for one anchor
| Easy / hard example | An anchor the model already gets right with high confidence / one it gets wrong or is unsure about
| Focal loss | Cross-entropy times $\alpha_t (1-p_t)^\gamma$, which silences easy examples
| Prior bias initialisation | Setting the final bias to $-\log((1-\pi)/\pi)$ so every anchor starts predicting about 1% foreground
| GIoU / DIoU / CIoU | IoU minus a penalty for empty enclosing space / centre distance / centre distance and aspect ratio
| Enclosing box | Smallest box that covers both the prediction and the target
| Set prediction | Predicting a fixed-size set of (class, box) pairs and matching it one-to-one to the targets
| Hungarian matching | Exact minimum-cost one-to-one assignment, used by DETR to pair queries with objects

<div class="take">
<div class="lab">Key takeaway</div>

focal loss stops 100,000 easy background anchors from drowning 50 objects, IoU-family losses train boxes on the quantity that evaluation measures and still give a gradient when boxes do not overlap, and one-to-one Hungarian matching turns detection into set prediction, so the model learns to avoid duplicates instead of relying on NMS.

</div>
