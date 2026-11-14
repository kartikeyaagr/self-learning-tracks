---
id: "07.03"
course: "deep-learning"
module: "07"
slug: "diffusion-the-forward-process"
title: "Diffusion: the forward process"
keyIdeas: ["The forward process adds Gaussian noise in $T$ small steps: $q(x_t\\mid x_{t-1})=\\mathcal N(\\sqrt{1-\\beta_t}\\,x_{t-1},\\ \\beta_t I)$, with no learned parameters.", "Because Gaussians compose, you can jump straight to any step: $x_t=\\sqrt{\\bar\\alpha_t}\\,x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$, where $\\bar\\alpha_t=\\prod_{s\\le t}(1-\\beta_s)$.", "The $\\sqrt{1-\\beta_t}$ shrink keeps the variance at 1 ('variance preserving'), so the end point is close to $\\mathcal N(0,I)$ whatever the data.", "The schedule is best read as signal-to-noise ratio $\\mathrm{SNR}(t)=\\bar\\alpha_t/(1-\\bar\\alpha_t)$; linear and cosine schedules differ mainly in how quickly SNR falls.", "If $\\bar\\alpha_T$ is not essentially zero, training and sampling disagree at the last step; zero-terminal-SNR rescaling fixes it."]
question: "A colleague removes the $\\sqrt{1-\\beta_t}$ factor and uses $x_t=x_{t-1}+\\sqrt{\\beta_t}\\,\\epsilon_t$ with the same linear $\\beta$ schedule, still starting sampling from $\\mathcal N(0, I)$. Using the Gaussian addition rule, what is the distribution of $x_T$ given $x_0$, and why does sampling now go wrong?"
answer: "Without the shrink, each step adds independent noise with variance $\\beta_t$ and leaves the signal untouched, so $x_T = x_0 + \\sqrt{\\sum_t\\beta_t}\\,\\epsilon$. For the linear schedule $\\sum\\beta_t \\approx 1000\\cdot(10^{-4}+0.02)/2\\approx 10.05$, so $x_T\\sim\\mathcal N(x_0, 10.05\\,I)$. Two problems follow. The signal is never removed: the SNR at $T$ is about $\\|x_0\\|^2/10$ per pixel, not zero, so $x_T$ still depends on the image. And the total variance is about 11 (data variance plus 10), not 1. Starting the reverse chain from $\\mathcal N(0, I)$ therefore feeds the model inputs from a distribution it never saw at $t=T$: wrong scale and no image component. This is the 'variance exploding' family, which can work, but only if you sample from $\\mathcal N(0,\\sigma_\\text{max}^2 I)$ with $\\sigma_\\text{max}$ much larger than the data scale, as in score SDE and EDM models."
sources: [{"title": "Ho, Jain and Abbeel: Denoising Diffusion Probabilistic Models", "url": "https://arxiv.org/abs/2006.11239", "note": "Section 2 defines the forward process and the closed-form marginal used everywhere."}, {"title": "Nichol and Dhariwal: Improved Denoising Diffusion Probabilistic Models", "url": "https://arxiv.org/abs/2102.09672", "note": "Section 3.2 introduces the cosine schedule and explains why linear wastes steps."}, {"title": "Lin et al.: Common Diffusion Noise Schedules and Sample Steps are Flawed", "url": "https://arxiv.org/abs/2305.08891", "note": "The zero-terminal-SNR problem, the rescaling fix and why it needs v-prediction."}, {"title": "Lilian Weng: What are Diffusion Models?", "url": "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/", "note": "A careful derivation of every forward-process identity in one place."}]
practice: ["forward-diffusion-sample", "linear-beta-schedule", "cosine-noise-schedule", "rescale-to-zero-terminal-snr"]
tags: ["diffusion", "noise schedule", "SNR"]
tools: ["NumPy"]
---

## The idea in one paragraph

A diffusion model learns to generate data by learning to undo a fixed corruption. The **forward process** takes a clean image $x_0$ and adds a little Gaussian noise, then a little more, for $T$ steps (DDPM uses $T=1000$) until nothing of the image is left and $x_T$ is indistinguishable from pure noise. That process has no parameters and is trivial to run. The **reverse process**, a neural network, learns to remove the noise one step at a time (lesson 07.04). Generation then starts from fresh noise and runs the learned reverse chain (lesson 07.05). This lesson is about the forward half, because every formula in training and sampling is built from it.

~~~dg-flow The forward chain is fixed and cheap; only the reverse chain is learned.
[Forward, fixed] x0 data -(add noise)-> x1 -> ... -> *xT about N(0, I)
[Reverse, learned] xT noise -(denoise)-> ... -> x1 -> *x0 sample
~~~

## One step, and why the square root

Each forward step is

$$q(x_t\mid x_{t-1})=\mathcal N\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big),\qquad x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_t$$

with a small $\beta_t$ (DDPM: linear from $10^{-4}$ to $0.02$). The shrink factor $\sqrt{1-\beta_t}$ is not decoration. If $x_{t-1}$ has unit variance per pixel, then $x_t$ has variance $(1-\beta_t)\cdot 1+\beta_t=1$. Without the shrink, variance would grow by $\beta_t$ every step and the end point would depend on the data's scale. This is why images are scaled to $[-1, 1]$ first: the process assumes roughly unit-scale data.

## Gaussian facts that give the closed form

Two facts do all the work:

- A scaled Gaussian is Gaussian: $a\epsilon\sim\mathcal N(0, a^2)$.
- Independent Gaussians add by adding variances: $a\epsilon_1+b\epsilon_2 \overset{d}{=} \sqrt{a^2+b^2}\,\epsilon$.

Unroll two steps with $\alpha_t=1-\beta_t$: $x_t=\sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2}+\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-1}+\sqrt{1-\alpha_t}\,\epsilon_t$. The two noise terms combine into one Gaussian with variance $\alpha_t-\alpha_t\alpha_{t-1}+1-\alpha_t=1-\alpha_t\alpha_{t-1}$. Repeating to $x_0$ with $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$:

$$q(x_t\mid x_0)=\mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big),\qquad x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$$

This closed form is the reason diffusion training is cheap. To make a training example at step 700 you do not run 700 noising steps; you draw one $\epsilon$ and mix. It also means $x_t$ is just a weighted blend of signal and noise, with weights whose squares sum to 1.

```python
import numpy as np
T = 1000
betas = np.linspace(1e-4, 0.02, T)
abar = np.cumprod(1.0 - betas)                       # abar[t-1] is alpha-bar at step t
def q_sample(x0, t, rng):                            # t in 1..T
    eps = rng.standard_normal(x0.shape)
    return np.sqrt(abar[t-1]) * x0 + np.sqrt(1 - abar[t-1]) * eps, eps
```

## Reading a schedule as signal-to-noise ratio

Different papers parameterise schedules by $\beta_t$, $\bar\alpha_t$, $\sigma_t$ or log-SNR, but the quantity that matters is $\mathrm{SNR}(t)=\bar\alpha_t/(1-\bar\alpha_t)$, the ratio of signal power to noise power at step $t$.

| Step t | Linear: $\bar\alpha_t$ | Linear: SNR | Cosine: $\bar\alpha_t$
| --- | --- | --- | --- |
| 100 | 0.897 | 8.7 | 0.972
| 250 | 0.524 | 1.1 | 0.847
| 500 | 0.079 | 0.085 | 0.494
| 750 | 0.0034 | 0.0034 | 0.144
| 1000 | 0.00004 | 0.00004 | about 0

The linear schedule is already at SNR 0.085 by the halfway point: for most of the second half the input is almost pure noise, so many training steps teach little. Nichol and Dhariwal's **cosine schedule** sets $\bar\alpha_t=f(t)/f(0)$ with $f(t)=\cos^2\!\big(\tfrac{t/T+s}{1+s}\cdot\tfrac{\pi}{2}\big)$, $s=0.008$, and clips $\beta_t$ at 0.999. It destroys information more evenly, which matters most at low resolution where a little noise already hides a 32x32 image. Higher resolutions have more redundant pixels and want noisier schedules (shifted towards low SNR), which is why large-image models shift or rescale schedules.

<div class="ex">
<div class="lab">Worked example</div>

**what the model sees at step 250**

Linear schedule, $t=250$: $\sqrt{\bar\alpha_t}=\sqrt{0.524}=0.72$ and $\sqrt{1-\bar\alpha_t}=0.69$.

A pixel with clean value 0.8 and noise draw $\epsilon=-1.2$ becomes $0.72\cdot0.8+0.69\cdot(-1.2)=-0.25$.

SNR is 1.1, so signal and noise carry about equal power: shapes are visible, fine detail is gone.

</div>

## The last step must be pure noise

Sampling starts from $x_T\sim\mathcal N(0,I)$, which is only correct if $q(x_T\mid x_0)$ really is $\mathcal N(0,I)$, i.e. $\bar\alpha_T=0$. Stable Diffusion's schedule ends at $\bar\alpha_T\approx 0.0047$, so training inputs at $t=T$ still leak the image's mean brightness, while sampling inputs never contain that leak. Lin et al. (2023) showed this is why such models cannot produce very dark or very bright images, and fixed it by rescaling the schedule so $\sqrt{\bar\alpha_T}=0$ exactly (which in turn requires v-prediction, lesson 07.04, since predicting $\epsilon$ from pure noise is trivial and uninformative).

<div class="take">
<div class="lab">Key takeaway</div>

the forward process is a fixed variance-preserving noising chain whose marginal at any step has the closed form $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$; choose and diagnose schedules by their SNR curve, and make sure the final step really is pure noise.

</div>
