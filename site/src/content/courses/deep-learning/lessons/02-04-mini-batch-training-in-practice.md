---
id: "02.04"
course: "deep-learning"
module: "02"
slug: "mini-batch-training-in-practice"
title: "Mini-batch training in practice"
keyIdeas: ["A mini-batch gradient is an unbiased estimate of the full gradient whose error shrinks like $1/\\sqrt{B}$: four times the batch only halves the noise.", "The **linear scaling rule** says: multiply the batch size by $k$, multiply the learning rate by $k$. It holds until the **critical batch size**, where the step size hits the stability limit and doubling $B$ no longer halves the number of steps; for Adam, square-root scaling is a safer start.", "**Warmup** ramps the learning rate up over the first 1-5% of training, because early gradients are large, Adam's statistics are unreliable and the network is fragile at initialisation.", "With a constant learning rate the loss stalls at a noise floor; **schedules** (step decay, cosine, warmup-stable-decay) lower the rate later so the loss can settle lower.", "**Global-norm clipping** caps the length of the whole gradient while keeping its direction, and **gradient accumulation** (each micro-batch loss divided by $k$) reproduces a batch that does not fit in memory."]
question: "You move a Transformer from batch 64 to batch 512 by multiplying the learning rate by 8 and keeping everything else, including a 100-step warmup. Loss spikes to NaN around step 50. Give two distinct reasons this happened and the changes you would make."
answer: "First, the warmup is too short for the new, much higher peak. The learning rate now climbs eight times faster in absolute terms during the first 100 steps, while gradients at initialisation are still large and Adam's second-moment estimates are based on only a few steps, so early updates overshoot and push the weights into a region where the loss explodes. Second, linear scaling is only approximately right, and for Adam it usually overestimates: the adaptive denominator already normalises gradient size, so the effective step grows nearly in proportion to the learning rate whatever the noise level, and 8x is likely beyond the stable range (closer to $\\sqrt 8 \\approx 2.8\\times$ is a safer start). The new batch may also be past the critical batch size, in which case no learning rate will give the hoped-for 8x fewer steps. Fixes: lengthen warmup (for example 1-2k steps or a fixed 1-5% of training rather than a copied step count), start from square-root scaling and sweep upward, add global-norm clipping at 1.0 so a single bad step cannot diverge, and log the learning rate, pre-clip gradient norm and clip fraction so you can see exactly when instability starts. If larger learning rates keep failing, compare loss against examples seen: if batch 512 needs nearly as many steps as batch 64, a smaller batch is the more efficient choice."
sources: [{"title": "Goyal et al.: Accurate, Large Minibatch SGD", "url": "https://arxiv.org/abs/1706.02677", "note": "The linear scaling rule, gradual warmup, and the batch-8192 ResNet-50 results."}, {"title": "Loshchilov & Hutter: SGDR", "url": "https://arxiv.org/abs/1608.03983", "note": "Cosine annealing and warm restarts."}, {"title": "Pascanu, Mikolov, Bengio: On the difficulty of training recurrent neural networks", "url": "https://arxiv.org/abs/1211.5063", "note": "Why gradients explode and the norm-clipping remedy."}, {"title": "PyTorch docs: clip_grad_norm_", "url": "https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html", "note": "Global-norm clipping as implemented, including the returned total norm."}]
practice: ["mini-batch-training-loop", "learning-rate-warmup-scheduler", "gradient-clipping-global-norm-rescaling", "gradient-accumulation", "generate-warmup-stable-decay-schedule"]
tags: ["batch size", "learning-rate schedule", "gradient clipping"]
tools: ["PyTorch"]
---

## What you will learn

- What the batch size controls, and how noisy a mini-batch gradient is (the $1/\sqrt{B}$ rule).
- How to change the learning rate when you change the batch size, and where that stops working (the **critical batch size**).
- Why modern recipes start with **warmup** and end with a **decaying schedule**.
- Two cheap kinds of insurance: **global-norm gradient clipping** and **gradient accumulation**.
- A complete PyTorch loop that uses all of them, and what to log.

*Before you start:* gradient steps and backpropagation (DL module 01), and SGD and Adam (covered in the ML Maths course). Batch normalisation (02.01) comes back in the accumulation section.

Lessons 02.01 to 02.03 changed the network itself. This one changes nothing in the network: it is about the knobs *around* the optimiser (batch size, learning rate over time, step size) that decide whether a good model trains smoothly, slowly, or blows up.

## The idea in one picture

The true gradient of the training loss is an average over every example. Computing it on the whole dataset for every step is too slow, so we estimate it from a random **mini-batch** of $B$ examples. How good is that estimate? We measured it on scikit-learn's 1,797 handwritten digits (a softmax classifier early in training), drawing 400 random batches per size and comparing each with the exact full-data gradient:

| Batch size B | 1 | 16 | 64 | 256 | 1024
| --- | --- | --- | --- | --- | --- |
| Relative error of the gradient | 7.49 | 1.81 | 0.88 | 0.42 | 0.15
| Cosine with the true gradient | 0.16 | 0.50 | 0.75 | 0.92 | 0.99

~~~dg-fig 02.04-noise.svg
Mini-batch gradient error on the digits data, measured (dots) and predicted (lines). Every 4× increase in batch size halves the error. Later in training (orange) the true gradient is smaller, so the same batch gives a much noisier estimate.
~~~

With one example the estimate is mostly noise (error 7.5 times the signal), yet on average it still points slightly downhill (cosine 0.16 is above zero). By $B = 64$ the error is smaller than the signal, and by $B = 1024$ the estimate is nearly exact. Going from 1 to 16 examples cut the error about 4 times, which is $\sqrt{16}$. That square root is the central fact of this lesson.

Vocabulary: the **batch size** $B$ is how many examples are averaged per estimate; a **step** is one parameter update; an **epoch** is one pass over the data (about $N/B$ steps); **gradient noise** is the mini-batch gradient minus the full gradient.

## Step 1: a mini-batch gradient is an unbiased, noisy estimate

Let $\ell_i(\theta)$ be the loss on example $i$ with parameters $\theta$. The full gradient is the average $g = \frac{1}{N}\sum_{i=1}^{N} \nabla \ell_i$. A mini-batch $\mathcal{B}$ of $B$ random indices gives

$$\hat g_B = \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla \ell_i.$$

Two facts make this work:

- **Unbiased**: averaged over all possible batches, $\mathbb{E}[\hat g_B] = g$. Each step is right on average.
- **Variance shrinks like $1/B$**: if one example's gradient has variance $\sigma^2$, the batch average has variance about $\sigma^2/B$, so its typical error is $\sigma/\sqrt{B}$. Doubling the batch cuts the error only by $\sqrt 2 \approx 1.41$.

Drawing without repeats from $N$ examples makes the variance slightly smaller, $\frac{\sigma^2}{B}\cdot\frac{N-B}{N-1}$, reaching zero at $B = N$. The theory lines in the figure use this and sit right on the measured dots.

<div class="ex">
<div class="lab">Worked example</div>

**every possible batch of two, by hand**

A tiny model has four training examples whose per-example gradients (for one weight) are $4, 0, 2, -2$. The true gradient is their mean, $(4 + 0 + 2 - 2)/4 = 1$.

Variance of a single-example estimate: deviations from 1 are $3, -1, 1, -3$, squares $9, 1, 1, 9$, mean $20/4 = 5$. So with $B = 1$ the estimate is off by about $\sqrt 5 \approx 2.2$, more than twice the signal.

Batches of two: there are 6 possible pairs. Their averages are $(4,0) \to 2$, $(4,2) \to 3$, $(4,-2) \to 1$, $(0,2) \to 1$, $(0,-2) \to -1$, $(2,-2) \to 0$.

Unbiased: the mean of the six estimates is $(2 + 3 + 1 + 1 - 1 + 0)/6 = 6/6 = 1$, exactly the true gradient.

Variance: deviations $1, 2, 0, 0, -2, -1$, squares sum to $1 + 4 + 0 + 0 + 4 + 1 = 10$, so variance $10/6 \approx 1.67$. The formula gives $\frac{5}{2}\cdot\frac{4-2}{4-1} = 2.5 \times \tfrac{2}{3} \approx 1.67$. It matches.

The pair $(0, -2)$ gives $-1$: it points the *wrong way*. Small batches occasionally push uphill; many steps average this out.

</div>

## Step 2: what the batch size buys and costs

Batch size sets three things at once:

- **Hardware efficiency.** A GPU works on thousands of numbers in parallel; a tiny batch leaves it idle, so up to some size a bigger batch costs almost no extra time per step.
- **Memory.** Saved activations grow linearly with $B$, so the largest batch that fits is often the real limit.
- **Noise.** Small batches give noisy steps. That slows convergence but also acts as a mild regulariser, which can help on small datasets.

So the useful question is: *if I double the batch, do I need half as many steps?* If yes, parallel hardware finishes sooner for the same compute. If no, you are paying for examples that do not help.

## Step 3: scale the learning rate with the batch

Halving the noise lets you take bigger steps without bouncing around more. Goyal et al. (2017) turned this into the **linear scaling rule**: when you multiply the batch size by $k$, multiply the learning rate by $k$:

$$\eta_{\text{new}} = \eta_{\text{base}} \times \frac{B_{\text{new}}}{B_{\text{base}}}.$$

Here $\eta$ (eta) is the learning rate and $B_{\text{base}}$ the batch size the recipe was tuned for. Intuition: with SGD, $k$ steps of size $\eta$ on $k$ batches move the weights about as far as one step of size $k\eta$ on the combined batch, *as long as the gradient barely changes over those $k$ steps*. With it they trained ResNet-50 on ImageNet at batch 8192 and learning rate $0.1 \times 8192/256 = 3.2$ in one hour, matching batch-256 accuracy, but only with 5 epochs of warmup (published values).

The proviso is where it fails: on a bowl with steepest curvature $\lambda_{\max}$, gradient descent diverges once $\eta > 2/\lambda_{\max}$, however clean the gradient. So the best learning rate grows with $B$, then hits a wall.

~~~dg-fig 02.04-critical.svg
A synthetic least-squares problem trained with SGD at every batch size, with the learning rate tuned for each. Left: steps to reach the target loss halve with each doubling of B up to roughly 64, then level off near 10. Right: the best learning rate follows the linear rule until it meets the stability limit.
~~~

Behind the figure (mean of 3 seeds): batch 1 needed 1,114 steps, batch 8 needed 92, batch 64 needed 25, and batches 512 to 2048 needed 9 to 11. Counted in examples, batches up to 32 used about 740 to 1,250, while batch 2048 used 18,432. The bend is the **critical batch size**: below it you trade examples for steps almost one-for-one; above it extra compute buys little speed.

Two refinements:

- **For Adam, scale more gently.** Adam already normalises gradient size, so its step grows with the learning rate whatever the noise. Square-root scaling ($\eta \propto \sqrt{B}$) is a safer start; then sweep.
- **The critical batch size grows during training.** In the noise figure's orange curve the true gradient has shrunk but the noise has not, so later training wants bigger batches.

## Step 4: warm up first

Starting at the full learning rate on step 0 is the most common way to blow up a large-batch or Adam run, because the first steps are special:

- At initialisation the loss is high and gradients can be large and poorly aligned.
- Adam's second-moment estimate $\hat v$ has seen only a few gradients, so its steps can be wildly sized.
- Normalisation layers, attention logits and deep stacks are most fragile before the weights settle.

**Warmup** ramps the learning rate linearly from near zero to its peak over the first few hundred to few thousand steps (often 1 to 5% of the run):

$$\eta_t = \eta_{\text{peak}} \cdot \frac{t + 1}{T_{\text{warm}}} \quad \text{for } t < T_{\text{warm}}.$$

~~~dg-fig 02.04-warmup.svg
The same 8-hidden-layer network, same Adam peak learning rate 0.02, same batches, one seed. Without warmup the gradient norm keeps spiking and around step 200 training collapses back above the loss of random guessing; with a 100-step warmup it keeps descending.
~~~

Without warmup the run hit loss 33.3 and gradient norm 1,425, finishing at 1.55; with warmup the norm peaked at 31 and the loss finished at 0.012. The damage was delayed: early large steps pushed the weights into a fragile region and the run collapsed later. One seed, so read the pattern, not the exact numbers.

## Step 5: decay the learning rate later

After warmup the learning rate should usually fall. Near a minimum every step is jolted by gradient noise, so the loss hovers at a **noise floor** that rises with $\eta$ (and falls with $B$). Lowering $\eta$ lowers the floor.

~~~dg-fig 02.04-schedules.svg
Four schedules with the same peak rate on the same noisy least-squares problem (batch 8, average of 8 seeds). The constant rate stalls at its noise floor; each decaying schedule drops well below it once its learning rate falls.
~~~

After 1,500 steps the constant schedule sat at excess loss 0.044; step decay reached 0.0035, warmup plus cosine 0.0032, and warmup-stable-decay 0.0089 (decaying only from step 1,200 and still falling). The common schedules:

- **Step decay**: divide by 10 at fixed points; the classic ImageNet ResNet recipe does so at epochs 30 and 60 of 90.
- **Cosine**: a smooth half-cosine from peak to a floor (formula below); no cut points to choose.
- **Warmup-stable-decay (WSD)**: hold the peak, then decay fast in the last 10 to 20%. You can branch a decay off any checkpoint, handy when the final step count is unknown.
- **One-cycle**: ramp up and back down within one run, peak chosen by a learning-rate range test.

The cosine schedule after warmup is

$$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi (t - T_{\text{warm}})}{T - T_{\text{warm}}}\right),$$

where $t$ is the step, $T$ the total steps and $\eta_{\min}$ the floor. Check: peak $3 \times 10^{-4}$, floor 0, warmup 500, total 10,000. Step 5,250 is halfway through the decay, so $\cos(\pi/2) = 0$ and $\eta = 0.5 \times 3\times 10^{-4} = 1.5 \times 10^{-4}$.

~~~dg-flow A typical modern schedule over one training run.
Step 0: lr ≈ 0 -(linear warmup, about 2% of steps)-> *Peak lr -(cosine or stable phase)-> Late training -(decay)-> lr ≈ 0.1 × peak or less
~~~

## Step 6: clip the gradient's global norm

One unlucky batch, or a suddenly steep direction (common in recurrent nets and Transformers), can produce a gradient hundreds of times larger than usual, and one such step can wreck a model. **Global-norm clipping** measures the length of the whole gradient across every parameter tensor and shrinks it if it exceeds a threshold $c$:

$$\|g\|_2 = \sqrt{\textstyle\sum_{\text{all params } p} \|g_p\|^2}, \qquad g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|_2}\right).$$

It keeps the direction and only limits the length. A typical threshold for Transformers is $c = 1.0$.

~~~dg-fig 02.04-clip.svg
Global-norm clipping shrinks the gradient onto the circle of radius c and keeps its direction. Value clipping cuts each entry separately, which squashes the vector into a box and bends its direction (here by 21°).
~~~

<div class="ex">
<div class="lab">Worked example</div>

**clipping a gradient by hand**

A model has two parameter tensors. Their gradients are $g_a = (2, -1)$ and $g_b = (2, 4)$. Threshold $c = 1$.

Global norm: $\sqrt{2^2 + (-1)^2 + 2^2 + 4^2} = \sqrt{4 + 1 + 4 + 16} = \sqrt{25} = 5$.

Scale factor: $\min(1, 1/5) = 0.2$. Clipped gradients: $g_a = (0.4, -0.2)$, $g_b = (0.4, 0.8)$. Every entry shrank by the same factor, so the direction is unchanged and the new norm is exactly 1.

Value clipping to $[-1, 1]$ instead gives $(1, -1)$ and $(1, 1)$. The cosine with the original is $\frac{2 + 1 + 2 + 4}{5 \times 2} = 0.9$, an angle of about $26°$: the update now points somewhere the loss never asked for.

</div>

## Step 7: gradient accumulation when the batch does not fit

If the batch you want does not fit in memory, split it into $k$ **micro-batches**, run forward and backward on each (PyTorch adds the gradients into `.grad`), then take one optimiser step. Divide each micro-batch loss by $k$ so the sum equals the full-batch mean. If micro-batches hold different numbers of valid tokens, weight by token count instead of averaging averages. BatchNorm (02.01) still sees only micro-batch statistics, so accumulation is not exact for BN networks; LayerNorm and RMSNorm (02.02) are unaffected.

On a small network, one batch of 64 and four accumulated micro-batches of 16 gave gradients matching to $1.1 \times 10^{-8}$; forgetting the `/ k` made the gradient exactly 4 times too big.

## A full training loop in code

Everything together on the digits data: shuffled mini-batches, warmup then cosine, and clipping with the pre-clip norm logged.

```python
import math, torch
from torch import nn
from sklearn.datasets import load_digits
torch.manual_seed(0)

X, y = load_digits(return_X_y=True)
X = torch.tensor((X - X.mean(0)) / (X.std(0) + 1e-8), dtype=torch.float32); y = torch.tensor(y)
perm = torch.randperm(len(X)); tr, va = perm[:1437], perm[1437:]
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X[tr], y[tr]),
                                     batch_size=64, shuffle=True, drop_last=True)

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
epochs = 10
total = epochs * len(loader)              # 10 * 22 = 220 optimiser steps
warmup = int(0.05 * total)                # 11 steps

def lr_at(step, peak=3e-3, floor=0.0):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))

step = 0
for epoch in range(epochs):
    clipped = 0
    for xb, yb in loader:
        for g in opt.param_groups: g["lr"] = lr_at(step)
        loss = nn.functional.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # pre-clip norm
        clipped += int(gnorm > 1.0)
        opt.step(); step += 1
    with torch.no_grad():
        acc = (model(X[va]).argmax(1) == y[va]).float().mean().item()
    if epoch in (0, 4, 9):
        print(f"epoch {epoch}: lr {g['lr']:.5f} loss {loss.item():.3f} clipped {clipped}/22 val acc {acc:.3f}")
# epoch 0: lr 0.00298 loss 0.903 clipped 2/22 val acc 0.844
# epoch 4: lr 0.00165 loss 0.160 clipped 0/22 val acc 0.953
# epoch 9: lr 0.00000 loss 0.059 clipped 0/22 val acc 0.956
```

The schedule runs on the *step* count, so it stays correct whatever the batch size; `drop_last=True` avoids a tiny, very noisy final batch; and clipping fired on 2 of the first 22 steps, then never, which is the healthy pattern. `torch.optim.lr_scheduler.LambdaLR` can wrap a function like `lr_at`; either way, update once per optimiser step.

<div class="ex">
<div class="lab">Worked example</div>

**scaling a recipe from 1 GPU to 8**

A recipe trains with batch 256 and SGD learning rate 0.1. On 8 GPUs at 256 each, the global batch is 2048, so linear scaling gives learning rate $0.1 \times 2048/256 = 0.8$.

Use 5 epochs of linear warmup from 0.1 to 0.8, then the original decay schedule with its epochs unchanged.

Each epoch now has 8 times fewer steps, so a warmup written in steps must be recomputed: 1,000 steps on 1 GPU covers the same epochs as 125 steps on 8.

If the learning rate that keeps the loss stable stops growing with batch size, you have passed the critical batch size: more GPUs finish epochs faster but need more epochs to reach the same loss.

</div>

## Reading the result

Log these every few steps:

- **Learning rate actually used.** Schedulers stepped per epoch instead of per step are a common bug.
- **Pre-clip gradient norm.** Repeated spikes of 10 to 100 times the median warn of divergence.
- **Clip fraction.** Clipping on most steps means the threshold is too low or the learning rate too high.
- **Loss versus examples seen**, not just steps, when comparing batch sizes.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Loss is NaN within the first hundred steps | No warmup, or warmup too short for the peak rate | Warm up over 1 to 5% of training; lower the peak
| Bigger batch, same epochs, worse final accuracy | Learning rate not scaled, or batch past the critical size | Scale lr (linear for SGD, about square-root for Adam), train longer, or use a smaller batch
| Loss plateaus and wobbles at a fixed level | Constant learning rate at its noise floor | Add a decay phase (cosine, step or WSD)
| Occasional huge loss spikes mid-training | Rare bad batches or steep directions | Global-norm clipping at about 1.0; check data for corrupt examples
| Accumulated run behaves like a much higher lr | Micro-batch losses not divided by k | Scale each micro-batch loss by 1/k (or by token share)

## Going deeper

**How big is "critical"?** McCandlish et al. (2018) estimate it with the *gradient noise scale* $B_{\text{noise}} = \operatorname{tr}(\Sigma)/\|g\|^2$: total per-example gradient variance over the squared true gradient, the batch at which noise and signal are about equal. On the digits model it was 52 early in training and 1,849 after 100 steps. Their empirical model (from memory) trades steps $S$ against examples $E$ as $(S/S_{\min} - 1)(E/E_{\min} - 1) = 1$, so at $B \approx B_{\text{noise}}$ you need about twice the minimum of each.

**The noise temperature.** Viewing SGD as a noisy continuous process, the random jiggle scales with $\eta / B$. Holding that ratio fixed gives the linear scaling rule, and explains why decaying $\eta$ and growing $B$ lower the noise floor in similar ways.

**Large batches and generalisation.** Keskar et al. (2017) reported that large batches find sharper minima that generalise worse. Much of that gap closes when learning rate, warmup and number of updates are retuned, but small-batch noise is a genuine regulariser on small datasets.

**Clipping and cliffs.** Pascanu et al. (2013) showed exploding gradients in recurrent nets appear as steep "cliffs" in the loss surface; norm clipping stops one step from jumping off. With Adam it also keeps one huge gradient from polluting the moment estimates.

## Key terms

| Term | Meaning
| --- | --- |
| Mini-batch | The $B$ random examples averaged to estimate the gradient for one step
| Gradient noise | Mini-batch gradient minus full gradient; shrinks like $1/\sqrt{B}$
| Linear scaling rule | Multiply the learning rate by $k$ when the batch is multiplied by $k$ (SGD, below the critical size)
| Critical batch size | The batch size beyond which doubling $B$ no longer roughly halves the steps needed
| Warmup | Ramping the learning rate up from near zero over the first part of training
| Learning-rate schedule | The planned learning rate at each step: step decay, cosine, WSD, one-cycle
| Global-norm clipping | Shrinking the whole gradient to length at most $c$, same direction
| Gradient accumulation | Summing micro-batch gradients (losses scaled by $1/k$) before one step

<div class="take">
<div class="lab">Key takeaway</div>

a mini-batch gradient is an unbiased estimate whose error shrinks only like $1/\sqrt{B}$, so pick the batch for hardware efficiency, scale the learning rate with it (linear for SGD, gentler for Adam) until you reach the critical batch size, always warm up, decay the learning rate later in training, and clip the global gradient norm as cheap insurance.

</div>
