---
id: "02.04"
course: "deep-learning"
module: "02"
slug: "mini-batch-training-in-practice"
title: "Mini-batch training in practice"
keyIdeas: ["A mini-batch gradient is an unbiased estimate of the full gradient whose variance shrinks like $1/B$; past a critical batch size, larger batches stop reducing the number of steps needed.", "The linear scaling rule says: multiply the batch size by $k$, multiply the learning rate by $k$; it holds up to a limit and needs warmup to survive the first steps.", "Warmup ramps the learning rate up over the first few hundred to few thousand steps, because early gradients are large, Adam's second-moment estimates are unreliable, and normalised networks are fragile at init.", "Schedules (step, cosine, warmup-stable-decay) lower the learning rate later to reduce gradient noise and settle into a minimum; global-norm gradient clipping caps the size of any single step."]
question: "You move a Transformer from batch 64 to batch 512 by multiplying the learning rate by 8 and keeping everything else, including a 100-step warmup. Loss spikes to NaN around step 50. Give two distinct reasons this happened and the changes you would make."
answer: "First, the warmup was defined in steps for the old setup and is now too short in proportion to the much larger peak learning rate: the learning rate climbs eight times faster in absolute terms while Adam's second-moment estimates are still based on few steps and gradients at init are large, so early updates overshoot. Second, linear scaling is only approximately right, and for Adam it usually overestimates: the adaptive denominator already normalises gradient size, so the effective step grows nearly linearly with the learning rate regardless of noise, and 8× is likely beyond the stable range (closer to $\\sqrt 8 \\approx 2.8\\times$ is a safer start). Fixes: lengthen warmup (e.g. 1-2k steps or a fixed fraction of training), use square-root scaling and sweep upward, add global-norm clipping at 1.0 so a single bad step can't diverge, and monitor gradient norm to see where instability starts."
sources: [{"title": "Goyal et al.: Accurate, Large Minibatch SGD", "url": "https://arxiv.org/abs/1706.02677", "note": "The linear scaling rule, gradual warmup, and the batch-8192 ResNet-50 results."}, {"title": "Loshchilov & Hutter: SGDR", "url": "https://arxiv.org/abs/1608.03983", "note": "Cosine annealing and warm restarts."}, {"title": "Pascanu, Mikolov, Bengio: On the difficulty of training recurrent neural networks", "url": "https://arxiv.org/abs/1211.5063", "note": "Why gradients explode and the norm-clipping remedy."}, {"title": "PyTorch docs: clip_grad_norm_", "url": "https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html", "note": "Global-norm clipping as implemented, including the returned total norm."}]
practice: ["mini-batch-training-loop", "learning-rate-warmup-scheduler", "gradient-clipping-global-norm-rescaling", "gradient-accumulation", "generate-warmup-stable-decay-schedule"]
tags: ["batch size", "learning-rate schedule", "gradient clipping"]
tools: ["PyTorch"]
---

## Why mini-batches

Full-batch gradient descent computes the exact gradient over all $N$ examples per step; stochastic gradient descent uses one. Mini-batches of $B$ examples sit between: the gradient estimate is unbiased, its variance falls like $1/B$, and a batch fills the GPU's parallel units. Batch size sets three things at once: **hardware efficiency** (too small leaves the GPU idle), **gradient noise** (which also acts as regularisation), and **memory** (activations scale linearly with $B$). The optimiser algorithms are covered in ML Maths; this lesson is about the knobs around them.

## Batch size and learning rate

If you double $B$, the gradient noise halves, so you can take larger steps. Goyal et al.'s **linear scaling rule**: when the batch is multiplied by $k$, multiply the learning rate by $k$. They trained ResNet-50 on ImageNet with batch 8192 (learning rate $0.1 \times 8192/256 = 3.2$) in one hour, matching batch-256 accuracy, but only with 5 epochs of warmup. The rule breaks down beyond a **critical batch size**, where noise is no longer the bottleneck and doubling $B$ no longer halves the number of steps; past it you pay twice the compute for the same progress. The critical size grows during training as the loss falls and can be estimated from the gradient noise scale (AI Research covers this for LLMs). For Adam, square-root scaling of the learning rate is often a better heuristic than linear.

Large batches have a reputation for generalising worse ("sharp minima", Keskar et al.); much of the gap disappears with retuned learning rate, warmup and longer schedules, but small-batch noise is a genuine regulariser for small datasets.

## Warmup

Starting at the full learning rate often diverges in the first hundred steps, especially with large batches, Adam, or Post-LN Transformers. Reasons:

- At initialisation the loss is high and gradients can be large and poorly aligned.
- Adam's $\hat v$ is estimated from very few samples, so early normalised steps can be huge.
- Normalisation layers and attention logits are most fragile before the network has settled.

Linear warmup from near zero to the peak learning rate over a few hundred to a few thousand steps (often 1-5% of training) fixes this cheaply.

## Schedules

After warmup the learning rate should usually decrease. Large steps make fast progress but bounce around the minimum at a level set by gradient noise; small steps reduce that noise floor.

- **Step decay**: divide by 10 at fixed epochs (classic ResNet: epochs 30, 60, 90). Loss drops sharply at each step.
- **Cosine**: $\eta_t = \eta_{min} + \tfrac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$. Smooth, one fewer hyperparameter; standard for vision and LLMs.
- **Warmup-stable-decay (WSD)**: hold the peak, then decay quickly in the last 10-20%. You can branch off and decay at any point, which is convenient when you don't know the final step count.
- **One-cycle**: ramp up then down within one run, found with a learning-rate range test.

~~~dg-flow A typical modern schedule over one training run.
Step 0: lr ≈ 0 -(linear warmup, about 2% of steps)-> *Peak lr -(cosine or stable phase)-> Late training -(decay)-> lr ≈ 0.1 × peak or less
~~~

## Gradient clipping

A single bad batch, or an exploding direction in an RNN or Transformer, can produce a gradient orders of magnitude larger than usual, and one such step can wreck a model. **Global-norm clipping** rescales the whole gradient when its norm exceeds a threshold $c$:

$$g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|_2}\right), \qquad \|g\|_2 = \sqrt{\textstyle\sum_{\text{all params}} \|g_p\|^2}$$

It preserves the direction and only limits the magnitude. Typical $c = 1.0$ for Transformers. Clipping per-element (value clipping) changes the direction and is rarely the right choice. Log the pre-clip norm: if clipping fires on most steps, the threshold is too low or the learning rate too high.

## Gradient accumulation

If the batch you want doesn't fit in memory, run $k$ micro-batches, accumulate their gradients (autograd sums into `.grad`), and step once. Scale each micro-batch loss by $1/k$ so the result equals the large-batch mean, and when micro-batches have different numbers of valid tokens, weight by token count rather than averaging averages. BatchNorm still only sees micro-batch statistics, so accumulation is not exactly equivalent for BN networks.

```python
import math, torch

def lr_at(step, peak, warmup, total, floor=0.0):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))

for step, (x, y) in enumerate(loader):
    for g in opt.param_groups: g["lr"] = lr_at(step, 3e-4, 500, total_steps)
    loss = loss_fn(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # returns pre-clip norm
    opt.step()
```

<div class="ex">
<div class="lab">Worked example</div>

**scaling a recipe from 1 GPU to 8**

A recipe trains with batch 256 and SGD learning rate 0.1. On 8 GPUs at 256 each, the global batch is 2048, so linear scaling gives learning rate 0.8.

Use 5 epochs of linear warmup from 0.1 to 0.8, then the original decay schedule with epochs unchanged. Each epoch now has 8× fewer steps, so the step-based warmup length must be recomputed from epochs, not copied.

If the learning rate needed to keep the loss stable stops growing with batch size, you have passed the critical batch size: more GPUs will finish epochs faster but need more epochs to reach the same loss.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

choose batch size for hardware efficiency, scale the learning rate with it (linear for SGD, closer to square-root for Adam), always warm up, decay the learning rate later in training, and clip the global gradient norm as cheap insurance.

</div>
