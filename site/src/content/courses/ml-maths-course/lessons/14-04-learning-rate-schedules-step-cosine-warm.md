---
id: "14.04"
course: "ml-maths-course"
module: "14"
slug: "learning-rate-schedules-step-cosine-warmup-one-cycle"
title: "Learning-rate schedules: step, cosine, warmup, one-cycle"
keyIdeas: ["A constant learning rate trades speed against a noise floor: large $\\eta$ moves fast but stalls in a ball whose size grows with $\\eta$; decaying $\\eta$ shrinks that ball.", "Step decay cuts $\\eta$ by 10x at fixed epochs; cosine decay $\\eta_t = \\eta_{min} + \\tfrac12(\\eta_{max} - \\eta_{min})(1 + \\cos(\\pi t/T))$ needs only the total length $T$.", "Warmup ramps $\\eta$ up over the first hundreds to thousands of steps, because early gradients, Adam's second-moment estimates and large-batch training are least stable at the start.", "One-cycle (warm up to a large peak, then anneal far below the start) and warmup-stable-decay (constant plateau, short final decay) are the two dominant modern shapes.", "Find the peak with an LR range test or a sweep; a schedule cannot rescue a peak learning rate that is 10x off."]
question: "Your cosine-scheduled run is configured for 100k steps, but you have to stop at 60k. A colleague suggests simply taking the 60k checkpoint. Why is that checkpoint likely worse than a run configured for 60k from the start, and how would a warmup-stable-decay schedule have avoided the problem?"
answer: "At 60% of a cosine schedule the learning rate is still about $\\eta_{min} + 0.5(\\eta_{max}-\\eta_{min})(1+\\cos(0.6\\pi)) \\approx 35\\%$ of the peak. The model is therefore still bouncing around in the large noise ball that a high learning rate produces; the annealing phase, where the rate falls towards the floor and the loss drops substantially as iterates settle, never happened. A run configured for 60k would have spent its final steps at a low rate and reached a lower loss. Under warmup-stable-decay the rate is constant on the plateau and decay is a short separate phase, so you can take the 60k plateau checkpoint and run a decay phase of 10-20% of the steps, getting nearly the result of a dedicated 60k run. The general lesson: with cosine, the schedule commits you to $T$ in advance; WSD decouples the stopping decision from the schedule."
sources: [{"title": "Goyal et al.: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", "url": "https://arxiv.org/abs/1706.02677", "note": "The linear scaling rule and gradual warmup for large batches."}, {"title": "Loshchilov and Hutter: SGDR: Stochastic Gradient Descent with Warm Restarts", "url": "https://arxiv.org/abs/1608.03983", "note": "The cosine schedule and restarts."}, {"title": "Smith: A disciplined approach to neural network hyper-parameters, part 1", "url": "https://arxiv.org/abs/1803.09820", "note": "The LR range test, one-cycle policy and cyclical momentum."}, {"title": "Hu et al.: MiniCPM: Unveiling the Potential of Small Language Models", "url": "https://arxiv.org/abs/2404.06395", "note": "Warmup-stable-decay schedules and why they suit scaling experiments."}]
practice: ["linear-warmup", "cosine-annealing-with-warm-restarts", "lr-range-test-and-one-cycle-policy", "generate-warmup-stable-decay-schedule", "learning-rate-sweep"]
tags: ["learning rate", "warmup"]
tools: ["PyTorch"]
---

## Why the learning rate should change

Lesson 14.01 ended with stochastic gradients: at a constant $\eta$, SGD reaches a neighbourhood of the minimum whose size grows with $\eta$ and the gradient noise, then bounces around inside it. Large steps make fast progress early, when the loss is far from optimal and the gradient signal dominates the noise; small steps are needed late to settle into the bottom. A **schedule** gets both by changing $\eta$ over training. It also interacts with stability: the very first steps, from a random initialisation with Adam's moment estimates still empty (14.03), are where a large step most often blows up.

This lesson covers the shapes and the reasons. The Deep Learning course's training module shows how schedules combine with batch size and gradient clipping in a full training loop, and the AI Research course covers learning-rate decay in its from-scratch network.

~~~dg-flow A modern schedule has three phases, each solving a different problem.
Step 0 -(linear warmup)-> *Peak lr -(hold or decay slowly)-> Main phase -(anneal)-> Final lr ~ 1-10% of peak
~~~

## The common shapes

| Schedule | Formula or rule | Knobs | Notes
| --- | --- | --- | --- |
| Step decay | multiply by $\gamma$ (often 0.1) at fixed epochs | milestones, $\gamma$ | classic ResNet recipe: decay at epochs 30, 60, 90
| Exponential | $\eta_t = \eta_0\gamma^t$ | $\gamma$ | smooth, but hard to set the end point
| Cosine | $\eta_{min} + \tfrac12(\eta_{max}-\eta_{min})(1+\cos(\pi t/T))$ | $T$, $\eta_{min}$ | only needs the total length; spends long near the peak and near the end
| Linear decay | straight line from peak to (near) zero | $T$ | common for fine-tuning and BERT-style training
| Warmup-stable-decay | warmup, constant plateau, short decay (10-20% of steps) | plateau length | can extend training by continuing the plateau
| One-cycle | up to a high peak, then down far below the start | peak, final ratio | super-convergence on vision tasks
| Cosine with restarts | cosine, then jump back to the peak | cycle length | SGDR; explores several minima

## Warmup

Warmup increases $\eta$ linearly from near zero to the peak over the first $W$ steps (commonly a few hundred to a few thousand, or 1-5% of training). Several distinct reasons point the same way:

- **Large batches.** Goyal et al. trained ResNet-50 with batch 8,192 by scaling the learning rate linearly with batch size; the scaled rate was unstable at the start, and a 5-epoch linear warmup fixed it.
- **Adam's early statistics.** $\hat v$ is estimated from very few gradients early on, so the adaptive step has high variance; RAdam derives a rectification that behaves like an automatic warmup.
- **Transformer architecture.** Post-LayerNorm transformers have large gradients near the output at initialisation and diverge without warmup; moving LayerNorm inside the residual branch (pre-LN) makes training far less sensitive to it.
- **Sharpness.** Early in training the loss surface is sharp (large $L$), so the $2/L$ limit of 14.01 is low; a short warmup lets curvature fall before the step size rises.

```
import math
def lr_at(step, peak=3e-4, warmup=500, total=10_000, floor=3e-5):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)          # 0 -> 1
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))
```

<div class="ex">
<div class="lab">Worked example</div>

**reading a cosine schedule**

Peak $3\times10^{-4}$, floor $3\times10^{-5}$, 500 warmup steps, 10,000 total steps.

At step 250 (mid-warmup) the rate is about $1.5\times10^{-4}$, half the peak.

At step 5,250, halfway through the decay, $\cos(\pi/2) = 0$, so $\eta = 3\times10^{-5} + 0.5\cdot2.7\times10^{-4} = 1.65\times10^{-4}$.

At step 9,000, $t = 0.895$ and $\cos(0.895\pi) = -0.946$, so $\eta \approx 3.7\times10^{-5}$: the last 10% of training runs within 25% of the floor, which is where much of the final loss drop happens.

</div>

## One-cycle and warmup-stable-decay

Leslie Smith's **one-cycle** policy raises the rate from a low value to a high peak over roughly the first 30-45% of training, then brings it down to far below the starting value (PyTorch's `OneCycleLR` defaults start at peak/25 and end four orders of magnitude below that), with momentum moving in the opposite direction. The high-rate middle acts as a regulariser and lets training finish in fewer epochs ("super-convergence"). The peak is chosen with an **LR range test**: run a short training while increasing $\eta$ exponentially, plot loss against $\eta$, and pick a value somewhat below where the loss starts to rise.

**Warmup-stable-decay (WSD)**, popular for LLM pre-training, keeps the peak rate constant for most of training and decays only in a short final phase. Because the decay is short, checkpoints from the plateau can be decayed at any point to produce a finished model, which makes it cheap to train several model sizes or data budgets from one run, something cosine cannot do since its shape depends on $T$ fixed in advance.

## Practical rules

- Tune the peak first. A sweep over 3x steps (1e-4, 3e-4, 1e-3) matters more than the schedule's shape; between cosine, linear and WSD with good peaks, the final losses are usually close.
- End low. Decaying to near zero, or to 1-10% of the peak, consistently beats stopping at a high rate.
- When batch size changes, the peak usually needs to change too: roughly linearly for SGD at moderate batch sizes, closer to square-root scaling for Adam, and neither holds beyond the critical batch size.
- Schedules interact with momentum's effective step $\eta/(1-\beta)$ (14.02) and with weight decay, whose per-step shrinkage $\eta\lambda$ (14.03) falls along with $\eta$.

<div class="take">
<div class="lab">Key takeaway</div>

warm up to protect the unstable first steps, spend most of training at a well-tuned peak, and anneal to a small fraction of it at the end; the shape matters less than getting the peak right and actually decaying.

</div>
