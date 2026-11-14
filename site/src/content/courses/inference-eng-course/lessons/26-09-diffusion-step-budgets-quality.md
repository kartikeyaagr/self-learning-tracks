---
id: "26.09"
course: "inference-eng-course"
module: "26"
slug: "diffusion-step-budgets-quality"
title: "Diffusion Step Budgets & Quality"
keyIdeas: ["Fewer steps cut latency linearly but can reduce quality.", "Quality curves flatten: going from 50 to 25 steps often costs little.", "Distilled models trade some quality or flexibility for 1 to 8 steps.", "Measure with automatic metrics plus human or judge preferences.", "Other levers: lower resolution then upscale, caching across steps, quantization."]
question: "How would you choose the number of denoising steps for a production image service?"
answer: "Generate images for a fixed prompt set at several step counts (for example 4, 8, 20, 30, 50) with your sampler and model, measure quality with automatic metrics (FID or CLIP-based scores) and human or model preference ratings, and measure latency and cost. Choose the smallest step count whose quality is within your acceptance threshold of the best, possibly offering different budgets per tier."
sources: [{"title": "DPM-Solver", "url": "https://arxiv.org/abs/2206.00927", "note": "Fast high-order sampler."}, {"title": "Progressive Distillation for Fast Sampling", "url": "https://arxiv.org/abs/2202.00512", "note": "Reducing step counts by distillation."}]
tags: ["diffusion"]
---

## The curve

| Steps | Latency | Quality (typical shape) |
| --- | --- | --- |
| 50 | 1.0× | best |
| 25 | 0.5× | nearly the same |
| 10 | 0.2× | small drop |
| 4 (distilled model) | 0.08× | acceptable for many uses |

~~~dg-raw
%%widget diff Lower the steps from 50 and watch latency fall linearly while the illustrative quality curve stays flat for a long time. Then switch to distilled guidance.
s = 25
%%
~~~

## Measuring

- A fixed prompt set that covers your product's real use.
- Automatic metrics such as FID and CLIP score, plus pairwise human preference tests. Automatic metrics often miss the artifacts users notice.
- Latency and GPU cost per image.

## Other levers

- **Step caching:** reuse features across adjacent steps, since they change little. Several research methods do this.
- **Quantization:** FP8 for the denoiser.
- **Resolution:** generate small, then upscale.
- **Guidance:** distilled guidance avoids the second pass per step.

<div class="ex">
<div class="lab">Worked example</div>

**choosing a budget**

A product generates thumbnails. In a preference test, users cannot tell 20-step images from 50-step images, and prefer 50 over 12 only 58% of the time.

20 steps cut latency and cost by 60%. 12 steps cut them by 76% with a small but measurable drop.

A reasonable choice is 20 steps for final images and 8 steps for live previews.

</div>

## Product tiers

Offer fast previews with few steps and final renders with more. Users accept a rough preview if the final image arrives soon after.

## Deep dive: sampling more to answer better

If one sample solves a problem with probability ⟦m1130⟧, and a perfect verifier can pick a correct one from ⟦m1131⟧ independent samples:

⟦M1129⟧

~~~dg-raw
%%chart Pass@k against samples for three per-sample success rates.
x = 1, 2, 4, 8, 16, 32
logx = 1
xlab = Samples k
ylab = Solved (%)
p = 0.05: 5, 9.8, 18.5, 33.7, 56, 80.6
p = 0.2: 20, 36, 59, 83.2, 97.2, 99.9
p = 0.5: 50, 75, 93.8, 99.6, 100, 100
%%
~~~

Cost grows linearly and the benefit saturates. Without a perfect verifier (majority vote, a reward model), real curves sit below these. The budget per request is a product decision: spend it on hard problems with low ⟦m1132⟧, where the curve is still steep.

<div class="take">
<div class="lab">Key takeaway</div>

step count is the main latency knob in diffusion. Find where the quality curve flattens and pick the smallest budget that meets your bar.

</div>
