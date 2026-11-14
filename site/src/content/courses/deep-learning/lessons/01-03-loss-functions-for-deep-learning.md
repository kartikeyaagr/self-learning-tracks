---
id: "01.03"
course: "deep-learning"
module: "01"
slug: "loss-functions-for-deep-learning"
title: "Loss functions for deep learning"
keyIdeas: ["Most losses are negative log-likelihoods: cross-entropy assumes a categorical output, MSE a Gaussian with fixed variance, MAE a Laplace distribution, so the loss encodes what kind of noise you believe in.", "Softmax plus cross-entropy has the gradient $p - y$ with respect to the logits: bounded, summing to zero, and strongest for confidently wrong examples. Sigmoid plus squared error lacks this and stalls when the sigmoid saturates.", "Compute losses from logits with log-sum-exp (`CrossEntropyLoss`, `BCEWithLogitsLoss`), never by taking the log of probabilities; applying softmax before the loss is a silent bug that slows training badly.", "MSE's gradient grows linearly with the error, so outliers dominate; Huber loss is quadratic near zero and linear beyond a threshold $\\delta$, capping each example's gradient at $\\delta$.", "Focal loss multiplies cross-entropy by $(1-p_t)^\\gamma$ to down-weight easy examples in heavily imbalanced data; margin losses (hinge, triplet) only push until a gap is reached and then stop.", "Sanity-check a classifier's first loss against $\\ln C$ (2.30 for 10 classes) and watch the loss alongside accuracy: they can move independently."]
question: "You train a classifier on 1% positives and 99% negatives with plain binary cross-entropy. It reaches 99% accuracy but almost never predicts the positive class. Explain, using gradients, why plain BCE behaves this way and how focal loss and class weighting each change the gradient contributions."
answer: "Each example contributes a gradient of size $|p - y|$ on its logit (for BCE on logits, $\\partial\\mathcal{L}/\\partial z = \\sigma(z) - y$). Early on the network quickly learns to output a low probability for everything, because that fixes 99% of the examples at once. From then on 99% of the batch are easy negatives with small but non-zero gradients, and their sum still outweighs the gradient from the handful of positives, so the optimiser keeps polishing the negatives and accepts misclassifying positives. Accuracy looks great because predicting all-negative is already 99% correct; recall on the positive class is near zero, which is the number to watch.\n\nClass weighting (for example `pos_weight` in `BCEWithLogitsLoss`, or `weight=` in `CrossEntropyLoss`) multiplies every positive example's loss, and therefore its gradient, by a factor such as 99. That rebalances the total gradient mass between the two classes, regardless of how hard each individual example is.\n\nFocal loss instead scales each example by $(1-p_t)^\\gamma$. A well-classified negative with $p_t \\approx 0.95$ is scaled by $0.05^2 = 0.0025$ at $\\gamma = 2$, while a misclassified positive with $p_t = 0.1$ keeps $0.9^2 = 0.81$ of its loss. It reweights by difficulty, which handles imbalance indirectly because the easy examples are mostly the majority class. In practice both are combined (the $\\alpha_t$ term in focal loss), and because neither fixes the decision rule by itself, the probability threshold is then tuned on a validation set for the precision/recall trade-off you need rather than left at 0.5."
sources: [{"title": "Lin et al.: Focal Loss for Dense Object Detection", "url": "https://arxiv.org/abs/1708.02002", "note": "Sections 3 and 5: the focal term, why easy negatives dominate, and the $\\gamma$ ablation."}, {"title": "PyTorch docs: CrossEntropyLoss", "url": "https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html", "note": "Takes logits, supports class weights, ignore_index and label smoothing."}, {"title": "PyTorch docs: HuberLoss", "url": "https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html", "note": "Exact definition and its relation to SmoothL1Loss."}, {"title": "Deep Learning (Goodfellow et al.), Chapter 6.2", "url": "https://www.deeplearningbook.org/contents/mlp.html", "note": "Cost functions as maximum likelihood and why saturating outputs need log-likelihood losses."}]
practice: ["cross-entropy-logits", "softmax-cross-entropy-gradient", "log-loss-binary-cross-entropy", "loss-functions", "sigmoid-focal-loss"]
tags: ["cross-entropy", "Huber", "focal loss"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- What a loss function is for in a neural network, and why the standard losses are not arbitrary: most are negative log-likelihoods.
- How softmax cross-entropy turns logits into a single number, and why its gradient is simply "predicted probability minus target".
- How to compute cross-entropy without overflow (log-sum-exp), and why PyTorch's losses want raw logits.
- When to use MSE, MAE or Huber for regression, and how outliers change the picture.
- What focal loss and margin losses (hinge, triplet) add for class imbalance and embeddings, and how to spot a broken loss during training.

*Before you start:* you should know what a layer and a logit are (01.01) and what sigmoid and softmax do (01.02). It helps to have seen logistic regression and the idea of maximum likelihood from the ML course. Logarithms and derivatives are the only maths needed.

Lesson 01.02 ended at the output layer: sigmoid for yes/no, softmax for one of $C$ classes, nothing for a number. This lesson compares that output with the true answer to get one number, the **loss**, the only thing training reduces. Lesson 01.04 shows how its gradient flows back through the layers.

## The idea in one picture

A network sorts photos into three classes: cat, dog, bird. For each photo it outputs three **logits** (raw scores, any real numbers), and softmax turns them into probabilities. Here are four photos that are all, in truth, cats:

| Photo | Logits (cat, dog, bird) | Probabilities | p(cat) | Loss $-\ln p(\text{cat})$
| --- | --- | --- | --- | --- |
| A | 2.0, 1.0, 0.1 | 0.66, 0.24, 0.10 | 0.66 | 0.42
| B | 4.0, 0.5, 0.2 | 0.95, 0.03, 0.02 | 0.95 | 0.05
| C | 0.3, 0.2, 0.1 | 0.37, 0.33, 0.30 | 0.37 | 1.00
| D | 0.1, 3.0, 0.2 | 0.05, 0.90, 0.05 | 0.05 | 3.01

The loss only looks at one number per photo: the probability the network gave to the correct class. It takes minus the logarithm of it. The batch loss is the average, here $(0.42 + 0.05 + 1.00 + 3.01)/4 = 1.12$.

~~~dg-fig 01.03-idea.svg
Cross-entropy as a function of the probability given to the true class. Confident and right (B) costs almost nothing; unsure (C) costs 1; confident and wrong (D) costs 3 and keeps rising toward infinity as p goes to 0.
~~~

Read the curve from right to left: confident and right is almost free, unsure costs a moderate amount, and *confidently wrong* is punished hard, with no ceiling. Photo D alone outweighs the other three together, so most of the next training step goes into fixing D.

This loss is **cross-entropy**. The **target** is the true answer (here "cat", or the one-hot vector $[1, 0, 0]$); a **loss function** maps (output, target) to a number that is small when the output is good.

## Step 1: from logits to a loss

Write the logits for one example as a vector $z = (z_1, \dots, z_C)$, where $C$ is the number of classes. Softmax turns them into probabilities:

$$p_j = \frac{e^{z_j}}{\sum_{k=1}^{C} e^{z_k}}.$$

Each $p_j$ is positive and they sum to 1. If the true class is $y$, the cross-entropy loss is

$$\mathcal{L} = -\ln p_y = -z_y + \ln \sum_{k} e^{z_k}.$$

In the second form, raising the true logit lowers the loss and raising any other logit raises it.

Number check for photo A: $\ln(e^2 + e^1 + e^{0.1}) = \ln 11.212 = 2.417$, and $2.417 - 2.0 = 0.417$. Same answer as $-\ln 0.659$.

For a **binary** problem the network outputs one logit $z$, the probability of class 1 is $p = \sigma(z)$, and the target is $y \in \{0, 1\}$. **Binary cross-entropy (BCE)** is

$$\mathcal{L} = -\big[y \ln p + (1-y)\ln(1-p)\big],$$

which is again "minus log of the probability given to what actually happened".

## Step 2: where the standard losses come from

Why minus a logarithm? Most standard losses are the **negative log-likelihood (NLL)** of the data under a probability model of the output. Choose the kind of randomness; the loss follows:

| Assumed output distribution | Network predicts | NLL becomes | Typical task
| --- | --- | --- | --- |
| Categorical over $C$ classes | $p = \mathrm{softmax}(z)$ | cross-entropy | classification, next-token prediction
| Bernoulli (yes/no) | $p = \sigma(z)$ | binary cross-entropy | binary and multi-label classification
| Gaussian, fixed variance | the mean $\hat y$ | mean squared error, up to constants | regression with well-behaved noise
| Laplace, fixed scale | the centre $\hat y$ | mean absolute error, up to constants | regression with heavy-tailed noise

Maximising the likelihood of the labels is the same as minimising the average NLL; the log turns a product over examples into a sum, which is what makes mini-batch averaging work.

Check the Gaussian row. If $y \sim \mathcal{N}(\hat y, \sigma^2)$ then $-\ln p(y) = \frac{(y - \hat y)^2}{2\sigma^2} + \frac{1}{2}\ln(2\pi\sigma^2)$. With $\sigma$ fixed, the second term is a constant and the first is squared error times a constant. So choosing MSE is a statement: "I believe the errors are roughly Gaussian." If they are heavy-tailed (rare, huge errors), that belief is wrong and MSE will over-react to the rare cases, which Step 5 shows.

## Step 3: the gradient is "probability minus target"

Backpropagation (01.04) starts from the derivative of the loss with respect to the logits. Differentiate $\mathcal{L} = -z_y + \ln\sum_k e^{z_k}$ with respect to $z_j$: the first term gives $-1$ if $j = y$ and 0 otherwise; the second gives $e^{z_j}/\sum_k e^{z_k} = p_j$. So

$$\frac{\partial \mathcal{L}}{\partial z_j} = p_j - \mathbb{1}[j = y],$$

where $\mathbb{1}[j = y]$ is 1 for the true class and 0 elsewhere. In vector form: gradient $= p - \text{onehot}(y)$. Each entry lies between $-1$ and $1$, and the entries always sum to 0.

<div class="ex">
<div class="lab">Worked example</div>

**one cross-entropy step by hand (photo A)**

Logits $z = [2.0, 1.0, 0.1]$, true class 0 (cat). Exponentials: $e^{2.0} = 7.389$, $e^{1.0} = 2.718$, $e^{0.1} = 1.105$. Their sum is $11.212$.

Probabilities: $7.389/11.212 = 0.659$, $2.718/11.212 = 0.242$, $1.105/11.212 = 0.099$. So $p = [0.659, 0.242, 0.099]$.

Loss: $-\ln 0.659 = 0.417$.

Gradient on the logits: subtract the one-hot target $[1, 0, 0]$ from $p$: $[0.659 - 1,\ 0.242 - 0,\ 0.099 - 0] = [-0.341, 0.242, 0.099]$.

Reading it: gradient descent moves *against* the gradient, so the cat logit goes up and the dog and bird logits go down, dog more because it stole more probability. The numbers sum to $0$.

PyTorch agrees: `CrossEntropyLoss` returns 0.4170 and the logit gradient `[-0.3410, 0.2424, 0.0986]`.

</div>

The same simplification happens for sigmoid with BCE: $\partial\mathcal{L}/\partial z = \sigma(z) - y$. That is the real reason a sigmoid output is paired with BCE and not with squared error. Put squared error on a sigmoid and the chain rule adds a factor $\sigma(z)(1 - \sigma(z))$, the sigmoid's slope, which is nearly zero whenever the logit is large in either direction.

~~~dg-fig 01.03-saturation.svg
True label 1. Left: both losses fall as the logit grows. Right: the gradient that actually trains the network. Cross-entropy pushes at full strength (about −1) when the logit is very wrong; squared error on a sigmoid output almost stops pushing there.
~~~

At $z = -6$ (the network is 99.8% sure of the wrong answer) the BCE gradient is $-0.998$, while squared error on the sigmoid gives $-0.0025$, about 400 times smaller. The log cancels the sigmoid's flattening, so the most wrong examples get the strongest correction.

## Step 4: computing it without overflow

Computing softmax first and then taking the log breaks in floating point. A logit of 1000 makes $e^{1000}$ overflow to infinity, and a probability that rounds to 0 gives $-\ln 0 = \infty$. The fix is the **log-sum-exp trick**: subtract the largest logit $m$ before exponentiating, which changes nothing mathematically because

$$\ln\sum_k e^{z_k} = m + \ln\sum_k e^{z_k - m}.$$

Now the biggest exponent is $e^0 = 1$, so nothing overflows.

```python
import numpy as np

def cross_entropy(logits, y):          # logits (B, C), y (B,) int labels
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    loss = lse - logits[np.arange(len(y)), y]
    p = np.exp(logits - lse[:, None])
    grad = p.copy(); grad[np.arange(len(y)), y] -= 1
    return loss.mean(), grad / len(y)

L, g = cross_entropy(np.array([[2.0, 1.0, 0.1]]), np.array([0]))
print(round(L, 3), g.round(3))         # 0.417 [[-0.341  0.242  0.099]]
print(cross_entropy(np.array([[1000., 0., 0.]]), np.array([1]))[0])   # 1000.0
# the naive -log(softmax) on the same input returns inf
```

Binary cross-entropy has the same trick: $\mathcal{L} = \max(z, 0) - zy + \ln(1 + e^{-|z|})$, which never exponentiates a large positive number. This is why PyTorch's `CrossEntropyLoss` and `BCEWithLogitsLoss` take **raw logits** and do softmax or sigmoid internally. Feeding them probabilities is a common silent bug, shown on a real run below.

**Label smoothing** softens the target: the true class gets $1 - \varepsilon$ and the rest is spread over the other classes (one convention gives each $\varepsilon/(C-1)$; PyTorch's `label_smoothing=ε` mixes in a uniform vector, giving every class $\varepsilon/C$). A one-hot target is only met at an infinite logit gap, so the network keeps inflating its logits; smoothing gives a finite optimum and usually better **calibration** ("90% sure" being right about 90% of the time). For photo A with $\varepsilon = 0.1$ the PyTorch loss rises from 0.417 to 0.514.

## Step 5: regression losses: MSE, MAE and Huber

For a number-valued output, the **residual** is $r = \hat y - y$, prediction minus target. Three losses dominate:

- **MSE** (mean squared error): $\frac{1}{2}r^2$ per example, gradient $r$. The gradient grows linearly with the error.
- **MAE** (mean absolute error): $|r|$, gradient $\pm 1$. Robust, but not smooth at 0 and slow to fine-tune because the gradient never shrinks.
- **Huber**: quadratic for small residuals, linear for large ones, with a threshold $\delta$:

$$\mathcal{L}_\delta(r) = \begin{cases} \tfrac{1}{2} r^2 & |r| \le \delta \\ \delta\left(|r| - \tfrac{1}{2}\delta\right) & |r| > \delta \end{cases}$$

Its gradient is $r$ clipped to $[-\delta, \delta]$: it behaves like MSE near the answer and like MAE far from it. (The $\frac{1}{2}$ is a convention that makes the MSE gradient exactly $r$; PyTorch's `MSELoss` omits it, so its gradient is $2r$.)

~~~dg-fig 01.03-regression.svg
Left: MSE grows quadratically, MAE and Huber linearly for large residuals. Right: their gradients. MSE's has no limit, MAE's jumps from −1 to +1 at zero, and Huber's is MSE's line capped at ±δ.
~~~

Under MSE a target wrong by 100 pushes 100 times harder than one off by 1, so a few bad labels steer the batch. To see it, train a one-layer network (a straight line) on 40 points from $y = 1 + 0.8x$ plus noise, where 4 labels have been corrupted by subtracting 12.

~~~dg-fig 01.03-outliers.svg
Same data, same model, same optimiser. MSE tilts the line toward four bad labels (slope 0.26 instead of 0.8); Huber with δ = 1 limits each bad label's pull and recovers a slope of 0.71.
~~~

The outliers still pull, but with bounded force. DQN's TD-error clipping is equivalent to a Huber loss for exactly this reason (Module 08), and object detectors use **smooth L1** (Huber with $\delta = 1$) for bounding-box offsets.

## Step 6: focal and margin losses

**Focal loss** targets class imbalance in dense detection, where a detector scores on the order of 100k candidate locations per image and almost all are easy background. Their many small losses add up and drown out the few objects. Focal loss multiplies cross-entropy by a factor that shrinks for examples the model already gets right:

$$\mathrm{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \ln p_t,$$

where $p_t$ is the probability of the true class, $\gamma \ge 0$ sets how strongly easy examples are discounted (the paper's best was $\gamma = 2$), and $\alpha_t$ is an optional class weight. $\gamma = 0$, $\alpha_t = 1$ is plain cross-entropy.

~~~dg-fig 01.03-focal.svg
Focal loss for several γ. The curves agree for badly classified examples on the left; in the shaded easy region larger γ pushes the loss toward zero, so easy examples stop dominating the total.
~~~

Number check at $\gamma = 2$: an easy example at $p_t = 0.9$ has its loss scaled by $(0.1)^2 = 0.01$ (from 0.105 to 0.0011); a hard one at $p_t = 0.1$ by $(0.9)^2 = 0.81$ (from 2.30 to 1.87).

**Margin losses** do not model probabilities at all. They push until a required gap is reached and then contribute zero:

- **Hinge** (the SVM loss): $\max(0, 1 - y \cdot s)$ for labels $y = \pm 1$ and score $s$; zero once $y \cdot s \ge 1$.
- **Triplet**: $\max(0, d(a, p) - d(a, n) + m)$, where $d$ is a distance between embeddings. It pulls an anchor $a$ closer to a positive $p$ than to a negative $n$ by margin $m$; used for face embeddings.
- **Contrastive / InfoNCE**: cross-entropy over similarity scores, with one positive among many negatives (as in CLIP). It is really a softmax loss in disguise.

| Loss | Gradient size for a bad example | Typical use
| --- | --- | --- |
| Cross-entropy | bounded, at most 1 per logit | classification, language models
| MSE | grows with the error | clean regression
| Huber | capped at $\delta$ | noisy regression, box offsets, TD errors
| Focal | cross-entropy scaled by $(1-p_t)^\gamma$ | dense detection, extreme imbalance
| Hinge / triplet | constant until the margin is met, then 0 | SVMs, metric learning

## Losses in PyTorch

Every loss below takes raw outputs; `reduction='none'` returns one value per example.

```python
import torch, torch.nn as nn

z = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)       # logits, NOT softmax
loss = nn.CrossEntropyLoss()(z, torch.tensor([0]))            # target = class index
loss.backward()
print(loss.item(), z.grad)        # 0.4170  tensor([[-0.3410,  0.2424,  0.0986]])
print(nn.CrossEntropyLoss(label_smoothing=0.1)(z, torch.tensor([0])).item())   # 0.5137

zb = torch.tensor([-6.0, 0.0, 3.0]); yb = torch.tensor([1.0, 1.0, 0.0])
print(nn.BCEWithLogitsLoss(reduction='none')(zb, yb))   # tensor([6.0025, 0.6931, 3.0486])

pred = torch.tensor([2.5, 0.0, 10.0]); tgt = torch.tensor([2.0, 0.5, 0.0])
print(nn.MSELoss(reduction='none')(pred, tgt))          # tensor([  0.2500,   0.2500, 100.0000])
print(nn.L1Loss(reduction='none')(pred, tgt))           # tensor([ 0.5000,  0.5000, 10.0000])
print(nn.HuberLoss(delta=1.0, reduction='none')(pred, tgt))   # tensor([0.1250, 0.1250, 9.5000])
```

The third prediction is off by 10: MSE scores it 100 (gradient 20, since PyTorch has no ½), Huber 9.5 with gradient capped at 1. `CrossEntropyLoss` also takes `weight=` (class weights) and `ignore_index=` (skip padding); `torchvision.ops.sigmoid_focal_loss` gives a binary focal loss.

## Reading the result: is the loss working?

**Check the first loss value.** With random initialisation the network's probabilities are close to uniform, so the first cross-entropy should be about $\ln C$: 2.303 for 10 classes, 0.693 for a binary task. A first loss of 30 means huge logits (initialisation, 01.05) or the wrong input to the loss.

**Watch the loss and the metric together.** Cross-entropy can fall while accuracy is flat (more confidence on examples already right), and validation loss can rise while accuracy holds, which signals over-confidence rather than more mistakes.

**Recognise the double-softmax bug.** Below, the same small network is trained on scikit-learn's digits (10 classes, 8×8 images); the only difference is passing `softmax(logits)` instead of `logits` to `CrossEntropyLoss`.

~~~dg-fig 01.03-softmax-bug.svg
Correct: 80% test accuracy after 5 epochs, 96% after 60. Bug: 22% after 5 epochs and 82% after 60. Nothing crashes; the model just learns much worse.
~~~

The cause: the loss applies softmax again to numbers that are already squeezed into $[0, 1]$. The best it can then reach for the true class is $e/(e + 9) = 0.23$, so the loss cannot go below $1.46$, and the extra softmax shrinks every gradient. A training loss that plateaus well above zero on an easy problem is the symptom.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Loss is NaN or inf after a few steps | log of a probability that rounded to 0, or exp overflow | compute from logits with `CrossEntropyLoss` / `BCEWithLogitsLoss`, never `log(softmax(z))` by hand
| Loss plateaus high, learning very slow | softmax or sigmoid applied before a loss that expects logits | remove the extra activation
| Classifier with sigmoid + MSE barely learns | saturated sigmoid kills the gradient | switch to BCE on logits
| Regression fit dragged by a few points | squared error lets outliers dominate | Huber or MAE, and inspect those labels
| 99% accuracy, never predicts the rare class | easy majority examples dominate the gradient sum | class weights, focal loss, tune the decision threshold
| Over-confident predictions, poor calibration | one-hot targets push logits to infinity | label smoothing, temperature scaling after training

## Going deeper

**Cross-entropy is a distance between distributions.** For a target distribution $q$ (one-hot or smoothed) and prediction $p$, cross-entropy $H(q, p) = -\sum_j q_j \ln p_j = H(q) + \mathrm{KL}(q \,\|\, p)$. The entropy $H(q)$ does not depend on the network, so minimising cross-entropy is minimising the KL divergence from the target to the prediction. With one-hot targets $H(q) = 0$; with label smoothing it is positive, which is why a smoothed loss never reaches 0.

**Convex in the logits.** The Hessian of softmax cross-entropy with respect to $z$ is $\mathrm{diag}(p) - pp^\top$, a covariance matrix, hence positive semi-definite. The loss is convex in the logits; all the non-convexity of deep learning comes from the network that produces them.

**Proper scoring rules.** Log loss is *strictly proper*: its expected value is minimised only by the true probabilities, so cross-entropy models give meaningful probabilities while hinge only gives a ranking score.

**Huber as a mixed noise model.** Huber is the NLL of a distribution with a Gaussian centre and Laplace (exponential) tails. Choosing $\delta$ says where "normal noise" ends and "outlier" begins.

**Focal loss weights.** In the RetinaNet paper $\alpha = 0.25$ worked best with $\gamma = 2$, which down-weights the positives slightly because the focal term already removed most of the negatives' weight.

~~~dg-flow Where the loss sits: it is the root of the computational graph, and its gradient is the first signal backprop sends.
Logits -(softmax)-> Probabilities -(−log p_y)-> *Scalar loss -(∂L/∂z = p − y)-> Backward pass
~~~

## Key terms

| Term | Meaning
| --- | --- |
| Loss function | A rule that turns (output, target) into one number that is small when the output is good; training minimises its average
| Logit | A raw, unnormalised score from the last layer, before softmax or sigmoid
| Negative log-likelihood | Minus the log of the probability the model gives to the observed label; most standard losses are this
| Cross-entropy | $-\ln p_y$ for softmax outputs; its gradient on the logits is $p - \text{onehot}(y)$
| Log-sum-exp trick | Subtracting the largest logit before exponentiating so the loss never overflows
| Label smoothing | Moving a little target probability from the true class to the others to stop logits growing without bound
| Huber loss | Quadratic for small residuals, linear beyond $\delta$; gradient capped at $\delta$
| Focal loss | Cross-entropy scaled by $(1-p_t)^\gamma$ so easy examples contribute little
| Margin loss | A loss (hinge, triplet) that is zero once a required gap between scores or distances is reached

<div class="take">
<div class="lab">Key takeaway</div>

choose the loss as a statement about your output distribution and your data's noise, compute it from logits in its stable form (its gradient is simply probability minus target), and reach for Huber, focal or margin losses when outliers, imbalance or ranking make plain negative log-likelihood misbehave.

</div>
