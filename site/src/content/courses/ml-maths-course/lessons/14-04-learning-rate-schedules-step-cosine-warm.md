---
id: "14.04"
course: "ml-maths-course"
module: "14"
slug: "learning-rate-schedules-step-cosine-warmup-one-cycle"
title: "Learning-rate schedules: step, cosine, warmup, one-cycle"
keyIdeas: ["A constant learning rate trades speed against a **noise floor**: SGD with a large $\\eta$ gets close fast but then bounces around the minimum, and the size of that wandering (and the extra loss it costs) is roughly proportional to $\\eta$. Decaying $\\eta$ shrinks it.", "Step decay cuts $\\eta$ by 10x at fixed milestones; cosine decay $\\eta_t = \\eta_{min} + \\tfrac12(\\eta_{max} - \\eta_{min})(1 + \\cos(\\pi t/T))$ needs only the total length $T$. With a good peak, the different decay shapes usually end up close.", "**Warmup** ramps $\\eta$ up over the first hundreds to thousands of steps, because early gradients, Adam's second-moment estimates, large-batch training and sharp early curvature are least stable at the start.", "One-cycle (warm up to a high peak, then anneal far below the start) and warmup-stable-decay (constant plateau, short final decay) are the two dominant modern shapes; WSD lets you stop at any plateau checkpoint and still finish with a short decay.", "Find the peak with an LR range test and a small 3x-step sweep; a schedule cannot rescue a peak learning rate that is 10x off."]
question: "Your cosine-scheduled run is configured for 100k steps, but you have to stop at 60k. A colleague suggests simply taking the 60k checkpoint. Why is that checkpoint likely worse than a run configured for 60k from the start, and how would a warmup-stable-decay schedule have avoided the problem?"
answer: "At 60% of a cosine schedule the learning rate is still $\\eta_{min} + 0.5(\\eta_{max}-\\eta_{min})(1+\\cos(0.6\\pi))$, and since $\\cos(0.6\\pi) \\approx -0.31$ that is about 35% of the peak (with a small floor). SGD's excess loss sits on a noise floor roughly proportional to the learning rate, so the model is still bouncing around in the large region that a high rate produces; the annealing phase, where the rate falls towards the floor and the loss drops substantially as the weights settle, never happened. A run configured for 60k would have spent its final steps at a low rate and reached a lower loss. In this lesson's regression experiment the same situation (a cosine planned for 5,000 steps stopped at 3,000) ended at an excess loss of 0.0038, against 0.0003 for a cosine planned for 3,000 steps: more than ten times worse. Under warmup-stable-decay the rate is constant on the plateau and the decay is a short separate phase, so you can take the 60k plateau checkpoint and run a decay over the next 10-20% of steps (or carve the decay out of the remaining budget), getting most of the benefit of a dedicated 60k run; in the toy experiment a 600-step WSD decay brought the loss to 0.0008. The general lesson: with cosine, the schedule commits you to $T$ in advance; WSD decouples the stopping decision from the schedule. If you are stuck with the cosine checkpoint, a short extra phase that anneals the rate from its current 35% down to the floor recovers much of the loss."
sources: [{"title": "Goyal et al.: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", "url": "https://arxiv.org/abs/1706.02677", "note": "The linear scaling rule and gradual warmup for large batches."}, {"title": "Loshchilov and Hutter: SGDR: Stochastic Gradient Descent with Warm Restarts", "url": "https://arxiv.org/abs/1608.03983", "note": "The cosine schedule and restarts."}, {"title": "Smith: A disciplined approach to neural network hyper-parameters, part 1", "url": "https://arxiv.org/abs/1803.09820", "note": "The LR range test, one-cycle policy and cyclical momentum."}, {"title": "Hu et al.: MiniCPM: Unveiling the Potential of Small Language Models", "url": "https://arxiv.org/abs/2404.06395", "note": "Warmup-stable-decay schedules and why they suit scaling experiments."}]
practice: ["linear-warmup", "cosine-annealing-with-warm-restarts", "lr-range-test-and-one-cycle-policy", "generate-warmup-stable-decay-schedule", "learning-rate-sweep"]
tags: ["learning rate", "warmup"]
tools: ["PyTorch"]
---

## What you will learn

- Why one fixed learning rate cannot be both fast and precise: the **noise floor** of SGD.
- The common schedule shapes and the one or two knobs each has.
- Why almost every modern recipe starts with a short **warmup**, and the four separate reasons behind it.
- How to choose the peak learning rate with an **LR range test** and a small sweep, and how to code a schedule in PyTorch.
- How to diagnose schedule problems, and why stopping a cosine run early hurts.

*Before you start:* lesson 14.01 (the step-size limit $\eta < 2/L$ and stochastic gradients), 14.02 (momentum) and 14.03 (Adam and AdamW). You need to be comfortable reading a loss-versus-step plot on a log scale.

## From optimiser to schedule

Lessons 14.01 to 14.03 chose *which direction* to step: plain gradient, momentum, or Adam. All still have one number in front, the learning rate $\eta$, so far held fixed. This lesson lets $\eta$ change during training. A rule that gives the learning rate at every step is called a **learning-rate schedule**, written $\eta_t$ for step $t$.

The motivation comes from the end of 14.01: with mini-batch gradients, a constant $\eta$ gets close to a minimum quickly and then stops improving, because gradient noise keeps kicking the weights around. Big steps help early, small steps help late; a schedule gives you both.

## The idea in one picture

Take a small, fully controlled problem: linear regression with 10 features and 4,000 examples, trained by SGD with batches of 8. Because it is a least-squares problem, we can compute the exact optimum and plot the **excess loss**, the training loss minus the best possible loss. Four runs of 3,000 steps each, each averaged over 10 random seeds:

| Schedule | Loss above the optimum at step 3,000
| --- | --- |
| constant $\eta = 0.1$ | 0.0084 (average over the last 300 steps)
| constant $\eta = 0.01$ | 0.0008 (average over the last 300 steps)
| step decay $0.1 \to 0.01 \to 0.001$ | 0.00022
| cosine from 0.1 down to 0 | 0.00027

~~~dg-fig 14.04-noiseball.svg
Left: the four schedules. Right: the constant rate of 0.1 drops fastest and then stalls on a plateau near 0.01; the rate of 0.01 is slow but reaches a lower plateau; both decaying schedules match the fast start and then sink below either plateau.
~~~

At step 100 the large rate is already near 0.01 while the small one is still near 0.3. Then the large rate goes flat. That plateau is the **noise floor**: the weights bounce around the optimum in a region whose size grows with $\eta$. A rate ten times smaller gives a floor about ten times lower (0.0008 against 0.0084) but takes hundreds of steps to get there. The step and cosine schedules start at 0.1, so they are fast, then shrink $\eta$, so the floor drops under them: they finish more than 30 times below the constant 0.1 run.

That is the whole lesson in one sentence: **move fast while you are far, then cut the step so the noise stops costing you.**

## Step 1: the noise floor, in symbols

Write the mini-batch gradient as the true gradient plus noise: $g_t = \nabla f(x_t) + \xi_t$, where $\xi_t$ has mean zero and variance $\sigma^2$ (bigger batches mean smaller $\sigma^2$). Take the simplest landscape, a one-dimensional bowl $f(x) = \tfrac12 h x^2$ with curvature $h$ and minimum at $x = 0$. One SGD step is

$$x_{t+1} = x_t - \eta (h x_t + \xi_t) = (1 - \eta h)\,x_t - \eta \xi_t.$$

The first term shrinks the distance to the minimum (14.01); the second adds a random kick of size $\eta$. Eventually they balance. Setting the variance of $x$ before and after a step equal gives the steady-state variance

$$V = \frac{\eta \sigma^2}{h(2 - \eta h)}, \qquad \text{excess loss} = \tfrac12 h V = \frac{\eta \sigma^2}{2(2 - \eta h)} \approx \frac{\eta \sigma^2}{4}.$$

$V$ is the mean squared distance the weight wanders from the minimum; the excess loss is what that costs. The key fact is the factor $\eta$ in the numerator: **the floor is proportional to the learning rate.**

<div class="ex">
<div class="lab">Worked example</div>

**how much does the floor drop when the rate drops?**

Take curvature $h = 1$ and gradient-noise variance $\sigma^2 = 4$.

With $\eta = 0.1$: $V = 0.1 \times 4 / (1 \times (2 - 0.1)) = 0.4 / 1.9 = 0.2105$, so the excess loss is $\tfrac12 \times 1 \times 0.2105 = 0.105$.

With $\eta = 0.01$: $V = 0.04 / 1.99 = 0.0201$, so the excess loss is $0.01005$.

A tenfold smaller rate gives an almost exactly tenfold lower floor (0.105 / 0.01005 = 10.5). The simulation above showed the same ratio, about 11.

The price: each step shrinks the gap by the factor $(1 - \eta h)$, so at $\eta = 0.01$ it removes 1% instead of 10% and reaching the floor takes about ten times as many steps.

</div>

## Step 2: the classic decaying shapes

The oldest schedules start at a peak $\eta_{max}$ and only go down.

~~~dg-fig 14.04-shapes.svg
Seven schedules over 1,000 steps, all with a peak of 0.1, computed with PyTorch's own scheduler classes where one exists. Left: decay-only shapes. Right: shapes that ramp up first.
~~~

- **Step decay** multiplies the rate by $\gamma$ (usually 0.1) at chosen **milestones**; the classic ResNet recipe drops at epochs 30, 60 and 90. Each drop shows as a sudden fall in the loss, because the noise floor falls tenfold at once (blue curve in the first figure, step 1,500).
- **Cosine decay** follows half a cosine wave from the peak to a floor over a planned length $T$:

$$\eta_t = \eta_{min} + \tfrac12(\eta_{max} - \eta_{min})\big(1 + \cos(\pi t / T)\big).$$

At $t = 0$ the cosine is 1 and $\eta = \eta_{max}$; at $t = T/2$ it is 0 (halfway); at $t = T$ it is $-1$ and $\eta = \eta_{min}$. The curve is flat at both ends, so it lingers near the peak early and near the floor late.

Exponential and linear decay complete the classic set; the table summarises all the shapes. Note how awkward exponential decay's end point is: $0.996^{1000} \approx 0.018$, so the figure's curve ends near 0.0018.

| Schedule | Formula or rule | Knobs | Notes
| --- | --- | --- | --- |
| Step decay | multiply by $\gamma$ (often 0.1) at fixed epochs | milestones, $\gamma$ | classic ResNet recipe: decay at epochs 30, 60, 90
| Exponential | $\eta_t = \eta_0\gamma^t$ | $\gamma$ | smooth, but hard to set the end point
| Cosine | $\eta_{min} + \tfrac12(\eta_{max}-\eta_{min})(1+\cos(\pi t/T))$ | $T$, $\eta_{min}$ | only needs the total length; spends long near the peak and near the end
| Linear decay | straight line from peak to (near) zero | $T$ | common for fine-tuning and BERT-style training
| Warmup-stable-decay | warmup, constant plateau, short decay (10-20% of steps) | plateau length | can extend training by continuing the plateau
| One-cycle | up to a high peak, then down far below the start | peak, final ratio | super-convergence on vision tasks
| Cosine with restarts | cosine, then jump back to the peak | cycle length | SGDR; explores several minima

In the first figure, step decay and cosine ended within about 25% of each other (0.00022 against 0.00027). That is typical: with a good peak and a real decay, the exact shape matters little.

## Step 3: warmup

**Warmup** raises $\eta$ from near zero to the peak over the first $W$ steps, usually linearly: $\eta_t = \eta_{max}(t+1)/W$ for $t < W$. Typical $W$ is a few hundred to a few thousand steps, or 1-5% of training. After warmup, a decay shape takes over. Four separate reasons point the same way:

1. **Large batches.** Goyal et al. trained ResNet-50 with batch 8,192 by scaling the learning rate linearly with batch size; the scaled rate was unstable at the start, and a 5-epoch linear warmup fixed it.
2. **Adam's early statistics.** Adam divides by $\sqrt{\hat v}$ (14.03), which in the first steps is estimated from a handful of gradients, so the step is noisy. RAdam's rectification behaves like an automatic warmup.
3. **Transformer architecture.** Post-LayerNorm transformers have large gradients near the output at initialisation and diverge without warmup. Moving LayerNorm inside the residual branch (pre-LN) makes training far less sensitive to it.
4. **Sharpness.** Early in training the loss surface is often sharp: its largest curvature $L$ is big, so the stability limit $2/L$ from 14.01 is low. A short warmup lets the curvature fall before the step size rises.

A small real test: an 8-layer MLP (128 units, no normalisation) on scikit-learn's digits, 600 steps of Adam at peak 0.01 with cosine decay. The only difference is a 60-step warmup.

~~~dg-fig 14.04-warmup.svg
Same network, same peak and same seeds. Without warmup the loss falls more slowly and stays about four times higher at the end; with a 60-step warmup every one of the five seeds ends with higher test accuracy.
~~~

Over five seeds, test accuracy was 93.1-94.8% (mean 93.7%) without warmup and 96.3-97.6% (mean 97.0%) with it. The run without warmup did not blow up; it was quietly damaged by large early steps and never fully recovered. That is more common in practice than a dramatic NaN.

## Step 4: one-cycle and warmup-stable-decay

Two shapes dominate modern practice, and both start with a ramp.

**One-cycle** (Leslie Smith) raises the rate to a high peak over roughly the first 30-45% of training, then brings it far below the starting value. PyTorch's `OneCycleLR` defaults start at peak/25, reach the peak at 30% of the steps, and end four orders of magnitude below the start (in the figure: 0.004, then 0.1, then $4 \times 10^{-7}$). Momentum moves the opposite way, from 0.95 down to 0.85 at the peak and back up, so the effective step $\eta/(1-\beta)$ from 14.02 does not swing as wildly. The high-rate middle acts as a regulariser and lets training finish in fewer epochs; Smith called this "super-convergence".

**Warmup-stable-decay (WSD)**, popular for LLM pre-training, holds the peak for most of training and decays only in a short final phase (often 10-20% of steps). Any plateau checkpoint can become a finished model by running a short decay from it, so one run serves several data budgets. Cosine cannot do this: its shape depends on a $T$ fixed in advance.

<div class="ex">
<div class="lab">Worked example</div>

**reading a cosine schedule with warmup**

Settings: peak $3\times10^{-4}$, floor $3\times10^{-5}$, 500 warmup steps, 10,000 total steps. Decay progress is $t = (\text{step} - 500)/9{,}500$.

Step 250 is in the warmup: $\eta = 3\times10^{-4} \times 251/500 = 1.506\times10^{-4}$, about half the peak.

Step 5,250: $t = 4{,}750 / 9{,}500 = 0.5$ and $\cos(0.5\pi) = 0$, so $\eta = 3\times10^{-5} + 0.5 \times 2.7\times10^{-4} \times 1 = 1.65\times10^{-4}$.

Step 9,000: $t = 8{,}500/9{,}500 = 0.895$ and $\cos(0.895\pi) = -0.946$, so $\eta = 3\times10^{-5} + 0.5 \times 2.7\times10^{-4} \times 0.054 = 3.73\times10^{-5}$.

So the last 10% of training runs within 25% of the floor ($3.73$ versus $3.0$). This low-rate tail is where much of the final loss drop happens, because the noise floor is now small.

</div>

~~~dg-fig 14.04-cosine-example.svg
The worked example drawn out: a steep linear warmup, a long cosine descent, and a flat tail that hugs the floor. The three dots are the hand-computed values (in units of 10⁻⁴).
~~~

## Step 5: choosing the peak

A schedule only scales a peak you choose, and no shape can rescue a peak that is ten times too high or too low. The quickest way to find a good range is the **LR range test**: run one short training where the rate grows exponentially every step, from far too small to far too large, and plot the (smoothed) loss against the rate.

~~~dg-fig 14.04-rangetest.svg
Left: one range-test run on a 4-layer MLP with SGD and momentum 0.9. Nothing happens below about 0.02, the loss falls steeply between 0.03 and 0.11, then rises and explodes. Right: full 600-step runs at four peaks confirm the shaded range.
~~~

Read the left panel in three zones: too small to move (below about 0.02), productive (0.03 to 0.11), and unstable (past the minimum at 0.11). Smith's rule is to take a peak somewhat below where the loss starts rising. Real runs (right, mean of 3 seeds) agree: a constant rate does best at 0.03 (96.7% test accuracy), one-cycle at 0.1 (97.6%), and both collapse at 0.3. One-cycle tolerates a higher peak because it spends only a short time there.

Then confirm with a sweep in 3x steps (for example 1e-4, 3e-4, 1e-3 for Adam); it matters more than the choice of shape.

## Schedules in code

In PyTorch a scheduler wraps the optimiser and changes `lr` in its parameter groups. The most flexible tool is `LambdaLR`, which multiplies the base rate by any function of the step. This one reproduces the worked example:

```python
import math, torch
model = torch.nn.Linear(10, 1)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

warmup, total, floor_ratio = 500, 10_000, 0.1
def factor(step):                      # multiplier applied to the base lr (3e-4)
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / (total - warmup)
    return floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * t))
sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)

seen = {}
for step in range(total):
    # ... forward, loss.backward() ...
    opt.step()                          # 1. update weights with the current lr
    sched.step()                        # 2. then move the schedule on by one step
    seen[step + 1] = opt.param_groups[0]["lr"]
for s in (250, 500, 5250, 9000, 10000):
    print(s, f"{seen[s]:.3g}")
# 250 0.000151
# 500 0.0003
# 5250 0.000165
# 9000 3.73e-05
# 10000 3e-05
```

Ready-made classes cover the standard shapes: `MultiStepLR` (step decay, usually stepped per epoch), `CosineAnnealingLR`, `OneCycleLR` (stepped per batch) and `SequentialLR` to glue a warmup to any decay. Call `opt.step()` before `sched.step()`, and know whether your scheduler counts batches or epochs.

## Reading the result

Always log the learning rate next to the loss. Then look for:

- **Loss drops sharply right after a decay.** Healthy: the lower rate removed a noise floor.
- **Spikes or NaNs in the first few hundred steps.** Lengthen warmup; if spikes return at the peak, lower the peak.
- **Loss still falling steeply when the rate hits the floor.** The decay came too early; train longer or decay later.
- **Validation loss rises during the low-rate tail.** That is overfitting, not a schedule bug: regularise or stop earlier.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Peak learning rate 10x too high | early divergence, or training "works" but ends worse | range test, then a 3x-step sweep
| No warmup with Adam or large batches | early spikes, or silently worse final loss (as in the warmup figure) | 1-5% linear warmup
| Cosine planned for $T$ but stopped early | final checkpoint worse than a shorter planned run | plan $T$ honestly, or use WSD
| Rate never decays (constant to the end) | loss plateaus at the noise floor | add any decay to 1-10% of peak or lower
| Scheduler stepped per epoch when written per step (or the reverse) | warmup takes 100x too long, or cosine finishes in the first epoch | plot the logged lr before trusting a run
| Batch size changed but peak kept | instability or slow training | rescale the peak, then re-check it

The early-stopping mistake deserves a picture.

~~~dg-fig 14.04-wsd.svg
All three runs stop at step 3,000 on the regression problem from the first figure (10 seeds each). A cosine planned for 5,000 steps is still at 35% of its peak when stopped and ends highest; WSD with a 600-step decay recovers most of the gap; a cosine planned for exactly 3,000 steps ends lowest on this small convex problem.
~~~

Excess losses: truncated cosine 0.0038, WSD 0.0008, planned cosine 0.0003. On this toy a well-planned cosine still wins; WSD's value is not needing the stopping point in advance. For LLM pre-training, published comparisons (MiniCPM among them) report WSD roughly matching cosine.

Two more interactions to keep in mind: when batch size changes, the peak usually needs to change too (roughly linearly for SGD at moderate batch sizes, closer to square-root scaling for Adam, and neither holds beyond the critical batch size); and schedules move momentum's effective step $\eta/(1-\beta)$ (14.02) and weight decay's per-step shrinkage $\eta\lambda$ (14.03) along with $\eta$.

## Going deeper

**Why decay is needed at all.** Classical stochastic approximation (Robbins and Monro) shows SGD converges to the exact minimum when $\sum_t \eta_t = \infty$ (enough total distance) and $\sum_t \eta_t^2 < \infty$ (accumulated noise stays finite). A constant rate fails the second condition: that is Step 1's noise floor. The textbook $\eta_t \propto 1/t$ satisfies both but decays too early for deep learning.

**Each eigen-direction has its own clock.** On a quadratic with curvatures $\lambda_i$ (14.01), the error along direction $i$ shrinks by $(1 - \eta\lambda_i)$ per step while its floor is proportional to $\eta$. Sharp directions converge quickly at any rate; flat ones need many large steps. A long high plateau makes progress along flat directions, and the decay then removes the noise along sharp ones. It also explains the WSD result: in 600 decay steps the flattest direction of the toy problem (curvature 0.09) has no time to settle.

**The edge of stability.** Curvature is not fixed during training. With full-batch gradient descent the sharpness $L$ often rises until it reaches about $2/\eta$ and hovers there. So a schedule also chooses *which* minima are reachable: a large rate rules out sharp ones, and a later decay lets the model settle into a narrower basin nearby.

**Restarts and averaging.** Cosine with restarts (SGDR) jumps back to the peak after each cycle, exploring a new basin and leaving a snapshot model at each low point. Stochastic weight averaging instead keeps a running mean of the weights at a moderate constant rate. Both treat the noise floor as a variance: shrink the step, or average over it.

~~~dg-flow A modern schedule has three phases, each solving a different problem.
Step 0 -(linear warmup)-> *Peak lr -(hold or decay slowly)-> Main phase -(anneal)-> Final lr ~ 1-10% of peak
~~~

## Key terms

| Term | Meaning
| --- | --- |
| Learning-rate schedule | A rule giving the learning rate $\eta_t$ at each training step
| Noise floor | The loss level where SGD stalls at a fixed rate; proportional to $\eta$ times the gradient noise
| Peak learning rate | The largest rate the schedule reaches; the knob that matters most
| Warmup | A ramp from near zero to the peak over the first few hundred to few thousand steps
| Step decay | Cutting the rate by a fixed factor (often 10x) at chosen milestones
| Cosine decay | A half-cosine curve from the peak to a floor over a planned length $T$
| One-cycle | Ramp up to a high peak, then anneal far below the starting rate, with momentum moving oppositely
| Warmup-stable-decay (WSD) | Warmup, a long constant plateau, then a short final decay
| LR range test | One short run with an exponentially growing rate, used to find where the loss falls fastest and where it blows up

<div class="take">
<div class="lab">Key takeaway</div>

warm up to protect the unstable first steps, spend most of training at a well-tuned peak, and anneal to a small fraction of it at the end; the shape matters less than getting the peak right and actually decaying.

</div>
