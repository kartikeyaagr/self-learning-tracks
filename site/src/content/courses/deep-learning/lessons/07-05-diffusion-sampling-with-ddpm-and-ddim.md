---
id: "07.05"
course: "deep-learning"
module: "07"
slug: "diffusion-sampling-with-ddpm-and-ddim"
title: "Diffusion: sampling with DDPM and DDIM"
keyIdeas: ["DDPM ancestral sampling runs all $T$ reverse steps: compute the posterior mean from $\\hat\\epsilon$, add fresh noise $\\sigma_t z$, repeat. 1000 network calls per sample.", "The target of each step is the forward posterior $q(x_{t-1}\\mid x_t,x_0)$, a Gaussian with closed-form mean and variance $\\tilde\\beta_t=\\frac{1-\\bar\\alpha_{t-1}}{1-\\bar\\alpha_t}\\beta_t$, evaluated at the predicted $\\hat x_0$.", "DDIM keeps the same marginals but uses a non-Markovian process, giving an update 'predict $\\hat x_0$, re-noise to the next level' whose stochasticity is a dial $\\eta$ (0 = deterministic).", "Because the network was only trained on marginals $q(x_t\\mid x_0)$, DDIM can visit any decreasing subsequence of steps: 50 steps instead of 1000 with the same trained model.", "Deterministic sampling makes noise-to-image a fixed map, enabling inversion, interpolation and reproducible seeds; quality at very low step counts needs better ODE solvers or distillation."]
question: "A DDIM sampler with 20 steps on a pixel-space model produces good images. Your teammate switches to DDPM ancestral sampling but keeps the same 20-step subsequence, using $\\sigma_t^2=\\beta_t$ from the original 1000-step schedule at each visited step. Images become noisy and grainy. Why?"
answer: "Ancestral DDPM is derived for consecutive steps with small $\\beta_t$: the reverse step's variance and the coefficient on $\\hat\\epsilon$ assume the chain removes only one step's worth of noise. On a 20-step subsequence each jump spans about 50 original steps, but the update still removes noise using one step's $\\beta_t$ and adds $\\sigma_t z$ sized for a single step. The mean therefore removes far too little noise per jump while fresh noise keeps being injected, so the sample arrives at $t=0$ with leftover noise. To use a subsequence with a stochastic sampler you must recompute the effective $\\beta$ between visited steps, $1-\\bar\\alpha_t/\\bar\\alpha_{t'}$, and the matching posterior variance, which is what DDIM's general formula with $\\eta=1$ does automatically."
sources: [{"title": "Song, Meng and Ermon: Denoising Diffusion Implicit Models", "url": "https://arxiv.org/abs/2010.02502", "note": "Sections 3 and 4: non-Markovian forward processes, the eta family and step skipping."}, {"title": "Ho, Jain and Abbeel: Denoising Diffusion Probabilistic Models", "url": "https://arxiv.org/abs/2006.11239", "note": "Algorithm 2 (ancestral sampling) and the posterior in equation 7."}, {"title": "Nichol and Dhariwal: Improved Denoising Diffusion Probabilistic Models", "url": "https://arxiv.org/abs/2102.09672", "note": "Learned reverse variances and why they matter for fewer sampling steps."}, {"title": "Hugging Face Diffusers: Schedulers", "url": "https://huggingface.co/docs/diffusers/api/schedulers/overview", "note": "How production code exposes DDPM, DDIM and ODE-solver samplers over the same model."}]
practice: ["ddim-step", "ddpm-ancestral-step", "posterior-mean-and-variance", "toy-ddpm-sampler"]
tags: ["diffusion", "DDIM", "sampling"]
tools: ["PyTorch"]
---

## Where we are

Lessons 07.03 and 07.04 gave us a schedule $\bar\alpha_t$ and a trained network $\epsilon_\theta(x_t, t)$. Sampling turns that network into images by running the reverse process from pure noise. The design question is how many network calls you spend: each one is a full forward pass of a large U-Net or DiT, so step count is roughly cost. (The Inference Engineering course, Specialized Inference Workloads, covers serving-side step budgets; this lesson covers the maths.)

## The forward posterior

The reverse step is trained to match $q(x_{t-1}\mid x_t, x_0)$, the forward process run backwards when you know the clean image. Bayes' rule on Gaussians gives it in closed form:

$$q(x_{t-1}\mid x_t,x_0)=\mathcal N\big(\tilde\mu_t,\ \tilde\beta_t I\big),\quad \tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\quad \tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$

At sampling time we do not know $x_0$, so we plug in the network's estimate $\hat x_0$. Substituting $\hat x_0$ from $\hat\epsilon$ gives exactly the mean from lesson 07.04.

## DDPM ancestral sampling

1. Draw $x_T\sim\mathcal N(0, I)$.
2. For $t=T,\dots,1$: predict $\hat\epsilon=\epsilon_\theta(x_t,t)$, compute $\mu=\frac{1}{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon\big)$, then $x_{t-1}=\mu+\sigma_t z$ with $z\sim\mathcal N(0, I)$, and no noise on the final step.
3. Return $x_0$, clipped to $[-1, 1]$.

The variance $\sigma_t^2$ can be $\beta_t$ or $\tilde\beta_t$; DDPM found both work similarly at $T=1000$ (they are the upper and lower bounds for the true reverse variance), while Improved DDPM learns an interpolation between them, which matters when you use fewer steps. Fresh noise is injected at every step, so the chain can correct earlier mistakes, but it needs all 1000 steps: skipping steps breaks the small-$\beta$ Gaussian assumption.

## DDIM: same model, different sampler

Song, Meng and Ermon (2020) observed that the training loss only depends on the marginals $q(x_t\mid x_0)$, never on the step-to-step process. Many forward processes share those marginals, including non-Markovian ones whose reverse step is

$$x_{t'}=\sqrt{\bar\alpha_{t'}}\,\hat x_0+\sqrt{1-\bar\alpha_{t'}-\sigma^2}\;\hat\epsilon+\sigma z,\qquad \hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}$$

for any earlier step $t'<t$. Read it as: estimate the clean image, then re-noise it to the next level, reusing the predicted noise direction for most of the noise. The free parameter is $\sigma=\eta\sqrt{\tfrac{1-\bar\alpha_{t'}}{1-\bar\alpha_t}}\sqrt{1-\tfrac{\bar\alpha_t}{\bar\alpha_{t'}}}$. With $\eta=1$ and consecutive steps it reproduces DDPM; with $\eta=0$ it is **deterministic**.

~~~dg-flow One DDIM step: denoise to an estimate of x0, then jump straight to the next noise level.
xt -(network)-> eps hat -(invert closed form)-> *x0 hat -(re-noise to t prime)-> x t prime
~~~

```python
def ddim_step(model, xt, t, t_prev, abar, eta=0.0):
    a, a_prev = abar[t], abar[t_prev]                 # t_prev < t; abar of "step 0" is 1
    eps = model(xt, t)
    x0 = ((xt - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
    sigma = eta * ((1 - a_prev) / (1 - a)).sqrt() * (1 - a / a_prev).sqrt()
    dir_xt = (1 - a_prev - sigma**2).sqrt() * eps
    return a_prev.sqrt() * x0 + dir_xt + sigma * torch.randn_like(xt)
```

Clipping $\hat x_0$ is a cheap guard against the network's early, very uncertain predictions pushing pixels out of range; it is standard for pixel-space models and must be removed for latent models, whose latents are not bounded.

## Step skipping

Since each DDIM update jumps between arbitrary levels, you can sample on a subsequence: 50 evenly spaced steps (every 20th) or 20 steps, with the same trained network. DDIM reported 10 to 50 times faster sampling at comparable quality, and deterministic DDIM degrades far more gracefully at low step counts than ancestral sampling. How you space the subsequence matters: spacing evenly in $t$ for a linear schedule wastes calls at high noise, and many samplers space steps in $\log$SNR or use the EDM $\sigma$ spacing (lesson 07.06), and always include the final step so the last update lands on clean data.

<div class="ex">
<div class="lab">Worked example</div>

**one deterministic DDIM jump**

Jump from a level with $\bar\alpha_t=0.5$ to $\bar\alpha_{t'}=0.8$. A pixel has $x_t=1.0$ and the network predicts $\hat\epsilon=0.6$; $\eta=0$.

$\hat x_0=(1.0-\sqrt{0.5}\cdot0.6)/\sqrt{0.5}=(1.0-0.424)/0.707=0.814$.

$x_{t'}=\sqrt{0.8}\cdot0.814+\sqrt{0.2}\cdot0.6=0.728+0.268=0.997$.

The pixel moves towards its predicted clean value while keeping the same noise direction at a lower magnitude.

</div>

## What determinism buys, and its limits

With $\eta=0$ the sampler is a fixed function from $x_T$ to $x_0$. The same seed always gives the same image; interpolating between two noise vectors interpolates images smoothly; and running the update backwards (**DDIM inversion**) maps a real image to a noise vector that regenerates it, the basis of many editing methods. DDIM is in fact a first-order (Euler) discretisation of the probability-flow ODE introduced in the next lesson, which is why higher-order ODE solvers (Heun, DPM-Solver) get good images in 10 to 25 steps, and why distillation methods target 1 to 4 steps.

| Sampler | Calls per image | Stochastic | Typical use
| --- | --- | --- | --- |
| DDPM ancestral | 1000 | yes | reference quality, slow
| DDIM, $\eta=0$ | 20 to 100 | no | fast, invertible, reproducible
| DDIM, $0<\eta\le1$ | 50 to 250 | partly | trades speed for robustness
| Higher-order ODE solvers | 10 to 25 | no | default in modern pipelines

<div class="take">
<div class="lab">Key takeaway</div>

ancestral DDPM sampling walks all $T$ posterior steps with fresh noise; DDIM reuses the same network to predict $\hat x_0$ and re-noise to any lower level, so it can skip steps and run deterministically, cutting sampling cost by 10 to 50 times.

</div>
