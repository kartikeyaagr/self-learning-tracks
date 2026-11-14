---
id: "04.03"
course: "deep-learning"
module: "04"
slug: "detection-losses-and-matching"
title: "Detection losses and matching"
keyIdeas: ["Dense detectors see roughly 1,000 background anchors per object. Plain cross-entropy is dominated by easy negatives; focal loss multiplies each term by $(1-p_t)^\\gamma$ so confident, correct predictions contribute almost nothing.", "Initialise the classification bias to a prior of about 0.01 foreground, or the first iterations produce a huge loss from 100k anchors all predicting 0.5.", "L1 loss on box coordinates does not track IoU. IoU-based losses do, but plain $1 - \\text{IoU}$ has zero gradient when boxes do not overlap; GIoU, DIoU and CIoU add terms that still pull disjoint boxes together.", "DETR treats detection as set prediction: a fixed set of queries is matched one-to-one to ground truth with the Hungarian algorithm, and unmatched queries learn to predict 'no object'. One-to-one matching is why it needs no NMS."]
question: "Why does DETR not need NMS while RetinaNet does? What would you expect if you trained DETR with RetinaNet-style matching (every query whose box has IoU above 0.5 with a ground truth is a positive for it)?"
answer: "RetinaNet's matching is many-to-one: every anchor with IoU above 0.5 is a positive for that object, so the network is explicitly trained to produce several high-scoring boxes per object, and NMS is required to collapse them. DETR's Hungarian matching is one-to-one: exactly one query is positive for each object and every other query is trained toward 'no object', so a duplicate prediction is a penalised false positive and the decoder learns to suppress duplicates through query-to-query self-attention. If you switched DETR to IoU-threshold matching, several queries would be rewarded for the same object, duplicates would reappear at test time, and you would need NMS again. You would also lose the clean set-prediction loss, since the 'no object' targets would depend on arbitrary thresholds."
sources: [{"title": "Focal Loss for Dense Object Detection (Lin et al., 2017)", "url": "https://arxiv.org/abs/1708.02002", "note": "RetinaNet and focal loss; section 3 for the loss and section 4.1 for the prior-based bias initialisation."}, {"title": "Generalized Intersection over Union (Rezatofighi et al., 2019)", "url": "https://arxiv.org/abs/1902.09630", "note": "Defines GIoU and shows why IoU is a better regression target than L1."}, {"title": "Distance-IoU Loss (Zheng et al., 2019)", "url": "https://arxiv.org/abs/1911.08287", "note": "DIoU and CIoU, with the containment failure case of GIoU illustrated."}, {"title": "End-to-End Object Detection with Transformers (DETR)", "url": "https://arxiv.org/abs/2005.12872", "note": "Set prediction and the Hungarian matching cost; section 3.1."}]
practice: ["sigmoid-focal-loss", "hungarian-matching", "generalized-iou-loss", "complete-iou-loss"]
tags: ["focal loss", "bipartite matching"]
tools: ["NumPy", "SciPy"]
---

## Two problems the previous lesson left open

The anchor recipe from the last lesson produces two training problems. Classification is wildly imbalanced: about 100,000 anchors per image, of which perhaps 50 are positive. And box regression with L1 or smooth-L1 on encoded offsets optimises something only loosely related to what evaluation measures, which is IoU. This lesson covers the loss that fixed the first problem, the family that fixed the second, and the matching scheme that removed NMS altogether. Module 01 introduced focal loss as a general loss; here is why detection needed it.

## Focal loss

Write $p_t$ for the predicted probability of the true class (so $p_t = p$ for positives and $1-p$ for negatives). Cross-entropy is $-\log p_t$. An easy negative with $p_t = 0.99$ costs only 0.01, but 100,000 of them sum to 1,000, swamping 50 hard positives.

Focal loss adds a modulating factor:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log p_t$$

With $\gamma = 2$, the easy negative's weight is $(0.01)^2 = 10^{-4}$; a hard example with $p_t = 0.3$ keeps weight $0.49$. The paper's defaults are $\gamma = 2$, $\alpha = 0.25$ for positives (it looks backwards but compensates for the down-weighting of negatives), and the total is divided by the number of positive anchors, not the number of anchors.

The other half of the trick is initialisation. If the final classification bias starts at zero, every anchor predicts $p = 0.5$ and the first loss is enormous and unstable. Setting the bias to $b = -\log((1-\pi)/\pi)$ with $\pi = 0.01$ gives $b \approx -4.6$, so every anchor starts by predicting "background, probably".

<div class="ex">
<div class="lab">Worked example</div>

**how much each anchor counts**

Easy negative, $p_t = 0.99$: CE $= 0.010$, FL with $\gamma = 2$ and $\alpha_t = 0.75$ $\approx 0.75 \times 10^{-4} \times 0.010 = 7.5 \times 10^{-7}$.

Hard positive, $p_t = 0.2$: CE $= 1.61$, FL $= 0.25 \times 0.64 \times 1.61 = 0.26$.

The ratio of hard to easy contribution goes from about 160× under CE to about 340,000× under focal loss.

</div>

## IoU-based box losses

Two predictions with the same L1 error can have very different IoU, because L1 treats the four coordinates independently and ignores scale. Optimising IoU directly fixes that, but $1 - \text{IoU}$ is flat (gradient zero) whenever the boxes do not overlap, which is common early in training.

| Loss | Extra term | What it fixes |
| --- | --- | --- |
| IoU | none | aligns training with the metric; no gradient when disjoint |
| GIoU | $-\,\lvert C \setminus (A \cup B)\rvert / \lvert C\rvert$, $C$ the smallest enclosing box | gives gradient when disjoint; range $[-1, 1]$ |
| DIoU | $-\,\rho^2(b, b^{gt}) / c^2$, centre distance over enclosing diagonal | pulls centres together directly; converges faster than GIoU |
| CIoU | DIoU plus $\alpha v$, $v = \frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h})^2$ | also matches aspect ratio |

The loss is $1 - \text{(G/D/C)IoU}$. GIoU has a known weakness: when one box contains the other, the enclosing box equals the larger box, the extra term vanishes, and GIoU degenerates to IoU; DIoU's centre term still pushes. In practice detectors combine an IoU-family loss with an L1 term on the boxes.

## Set prediction with Hungarian matching

DETR reframes detection. A transformer decoder takes $N = 100$ learned query vectors and outputs 100 (class, box) predictions, where the class set includes "no object". Training needs to decide which prediction is responsible for which ground-truth box, and DETR uses a one-to-one assignment that minimises total cost:

$$\hat\sigma = \arg\min_{\sigma} \sum_i \Big[-\hat p_{\sigma(i)}(c_i) + \lambda_{L1}\lVert b_i - \hat b_{\sigma(i)}\rVert_1 + \lambda_{giou}\big(1 - \text{GIoU}(b_i, \hat b_{\sigma(i)})\big)\Big]$$

with $\lambda_{L1} = 5$ and $\lambda_{giou} = 2$. The Hungarian algorithm solves this in $O(N^3)$, cheap for 100 queries. Matched predictions get the classification and box losses; the remaining queries are trained to output "no object", with that class's weight reduced to 0.1 to offset the imbalance.

```python
from scipy.optimize import linear_sum_assignment
# cost: (num_queries, num_gt) built from class prob, L1 and GIoU terms
rows, cols = linear_sum_assignment(cost)   # query rows[k] <-> gt cols[k]
```

~~~dg-flow Many-to-one matching needs NMS at test time; one-to-one matching does not.
[Dense detector] Anchors -(IoU > 0.5)-> Many anchors per object -> *NMS at test time
[DETR] 100 queries -(Hungarian, one-to-one)-> One query per object -> *No NMS
~~~

Why no NMS? With IoU matching, several anchors are trained to fire on the same object, so duplicates are the intended behaviour and must be pruned. With one-to-one matching, a second query firing on an already-claimed object is penalised as a false positive, so the model learns (via self-attention between queries) to suppress its own duplicates. The cost: the original DETR needed about 500 epochs on COCO versus about 36 for Faster R-CNN, because matching is unstable early on. Deformable DETR and later variants cut this substantially.

<div class="take">
<div class="lab">Key takeaway</div>

focal loss handles foreground-background imbalance, IoU-family losses align box regression with the evaluation metric, and one-to-one Hungarian matching turns detection into set prediction so the model learns to avoid duplicates instead of relying on NMS.

</div>
