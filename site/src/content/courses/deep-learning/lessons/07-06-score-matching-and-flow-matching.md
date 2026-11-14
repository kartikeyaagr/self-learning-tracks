---
id: "07.06"
course: "deep-learning"
module: "07"
slug: "score-matching-and-flow-matching"
title: "Score matching and flow matching"
keyIdeas: ["The score $\\nabla_x\\log p(x)$ points towards higher density and does not need the normalising constant; Langevin dynamics turns a score into a sampler.", "Denoising score matching learns the score of noised data by regressing $-\\epsilon/\\sigma$, which makes a noise predictor a score model: $s_\\theta(x_t,t)=-\\epsilon_\\theta(x_t,t)/\\sqrt{1-\\bar\\alpha_t}$.", "In continuous time, diffusion is an SDE; its reverse needs only the score, and the probability-flow ODE $dx=[f-\\tfrac12 g^2\\nabla\\log p_t]\\,dt$ has the same marginals with no randomness.", "Flow matching skips the SDE: pick a path from noise to data, e.g. $x_t=(1-t)x_0+t x_1$, and regress the velocity $x_1-x_0$; sampling integrates the learned ODE.", "Straighter paths need fewer ODE steps, which is the main reason flow-matching models (SD3, Flux) sample well in 20 to 30 steps."]
question: "A flow-matching model trained with the straight path $x_t=(1-t)x_0+t x_1$ is sampled with 4 Euler steps and looks fine; a DDPM-trained model sampled with 4 DDIM (Euler) steps looks blurry and wrong. Both are well trained. Explain the difference in terms of the ODE each sampler integrates."
answer: "Both samplers integrate an ODE whose marginals match the data, but the trajectories differ. DDPM's probability-flow ODE follows curved paths: early on, the estimate $\\hat x_0$ is a blurry average of many plausible images, and the trajectory bends as the prediction sharpens. A 4-step Euler integration takes large straight steps along the tangent at each point, so it overshoots the curve and ends somewhere off the data manifold, often near the blurry average. The flow-matching velocity field is trained on straight noise-to-data pairs, so its marginal trajectories are much less curved, and a straight Euler step along the tangent stays close to the true path. Euler's error scales with curvature times step size squared, so straighter paths tolerate far larger steps. Remedies for the diffusion model are a higher-order solver, step spacing concentrated where curvature is high, or distillation."
sources: [{"title": "Song et al.: Score-Based Generative Modeling through Stochastic Differential Equations", "url": "https://arxiv.org/abs/2011.13456", "note": "The SDE view, the reverse SDE and the probability-flow ODE (sections 3 and 4)."}, {"title": "Yang Song: Generative Modeling by Estimating Gradients of the Data Distribution", "url": "https://yang-song.net/blog/2021/score/", "note": "The author's own long-form explanation of score matching, Langevin and the SDE picture."}, {"title": "Lipman et al.: Flow Matching for Generative Modeling", "url": "https://arxiv.org/abs/2210.02747", "note": "Conditional flow matching and why the per-sample target gives the right marginal field."}, {"title": "Karras et al.: Elucidating the Design Space of Diffusion-Based Generative Models", "url": "https://arxiv.org/abs/2206.00364", "note": "EDM: the sigma-space ODE, Heun sampling and the step-spacing recipe."}]
practice: ["denoising-score-matching-loss", "conditional-flow-matching-loss", "gaussian-mixture-score", "annealed-langevin-dynamics", "karras-euler-sampler"]
tags: ["score matching", "probability-flow ODE", "flow matching"]
tools: ["PyTorch"]
---

## Scores: densities without normalisation

A generative model of images is hard to write as a density $p(x)$ because $p$ must integrate to 1 over a million-dimensional space. The **score** sidesteps that. For $p(x)=e^{-E(x)}/Z$, the score is $s(x)=\nabla_x\log p(x)=-\nabla_x E(x)$: the constant $Z$ disappears. The score is a vector field pointing uphill in density. If you know it you can sample with **Langevin dynamics**:

$$x_{k+1}=x_k+\tfrac{\eta}{2}\,s(x_k)+\sqrt{\eta}\,z_k,\qquad z_k\sim\mathcal N(0,I)$$

which, for small step $\eta$ and many steps, produces samples from $p$. Gradient ascent alone would collapse onto the modes; the injected noise keeps the spread right.

## Learning a score by denoising

We cannot regress the true score directly because we do not know it. **Denoising score matching** (Vincent, 2011) gets around this. Perturb data with Gaussian noise, $\tilde x=x+\sigma\epsilon$. The score of the perturbation kernel is known, $\nabla_{\tilde x}\log q(\tilde x\mid x)=-(\tilde x-x)/\sigma^2=-\epsilon/\sigma$, and minimising $\mathbb E\,\|s_\theta(\tilde x,\sigma)+\epsilon/\sigma\|^2$ has the same minimiser as matching the score of the noised data distribution. That regression target is the noise, rescaled. So for the variance-preserving process:

$$s_\theta(x_t,t)=-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}$$

A DDPM noise predictor already is a score model. Song and Ermon (2019) arrived at diffusion from this side: one network conditioned on many noise levels $\sigma_1>\dots>\sigma_L$ (large noise fills low-density regions where the data score is unlearnable), sampled by **annealed Langevin dynamics**, a few Langevin steps at each level from largest to smallest.

<div class="ex">
<div class="lab">Worked example</div>

**a denoising score target in 1-D**

A data point $x=2.0$ is perturbed with $\sigma=0.5$ and $\epsilon=0.4$, so $\tilde x=2.2$.

Target score: $-\epsilon/\sigma=-0.8$, the same as $-(\tilde x-x)/\sigma^2=-0.2/0.25$.

It points from the noisy point back towards the data, with magnitude growing as $1/\sigma$: small noise means a steep pull.

</div>

## Continuous time: SDEs and the probability-flow ODE

Song et al. (2021) unified both views. Let the step count go to infinity and the forward process becomes a stochastic differential equation $dx=f(x,t)\,dt+g(t)\,dw$; the variance-preserving DDPM chain is $f=-\tfrac12\beta(t)x$, $g=\sqrt{\beta(t)}$. A classical result says this SDE has a reverse-time SDE

$$dx=\big[f(x,t)-g(t)^2\nabla_x\log p_t(x)\big]dt+g(t)\,d\bar w$$

which needs only the score of each noisy marginal $p_t$, exactly what the network learns. Ancestral sampling is a discretisation of it. The same paper showed there is also a deterministic **probability-flow ODE** with identical marginals $p_t$:

$$dx=\big[f(x,t)-\tfrac12 g(t)^2\nabla_x\log p_t(x)\big]dt$$

~~~dg-layers Three ways to use the same learned score, from most to least random.
Annealed Langevin | many noisy steps per noise level, corrects as it goes
Reverse SDE | one noisy step per level; ancestral DDPM is its discretisation
*Probability-flow ODE | deterministic; DDIM is its Euler step; any ODE solver works
~~~

The ODE view explains the previous lesson: DDIM is an Euler step of this ODE. It also allows exact likelihoods (via the change-of-variables formula) and off-the-shelf higher-order solvers. Karras et al.'s **EDM** writes the process as $x=x_0+\sigma\epsilon$, where the ODE is simply $dx/d\sigma=(x-D_\theta(x;\sigma))/\sigma$ for a denoiser $D_\theta$; with a Heun solver and a $\sigma$ grid dense near the data it reaches state-of-the-art quality in about 35 network calls.

## Flow matching

If sampling is integrating an ODE from noise to data, why not learn the ODE's velocity field directly? **Flow matching** (Lipman et al., 2022; rectified flow, Liu et al., 2022) does that. Pair a noise sample $x_0\sim\mathcal N(0,I)$ with a data sample $x_1$ and define a path between them, most simply the straight line

$$x_t=(1-t)\,x_0+t\,x_1,\qquad t\in[0,1]$$

whose velocity is $u=x_1-x_0$. Train $v_\theta(x_t,t)$ to match it:

$$L_\text{CFM}=\mathbb E_{t,\,x_0,\,x_1}\big\|v_\theta(x_t,t)-(x_1-x_0)\big\|^2$$

As with denoising score matching, regressing the per-pair (conditional) target gives the correct marginal velocity field in expectation, even though individual paths cross. Sampling draws $x_0$ and integrates $dx/dt=v_\theta(x,t)$ from $t=0$ to 1, for example with Euler steps $x\leftarrow x+\Delta t\,v_\theta(x,t)$.

```python
def cfm_loss(model, x1):                       # x1: batch of data
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, 1, 1)
    xt = (1 - t) * x0 + t * x1
    return ((model(xt, t) - (x1 - x0)) ** 2).mean()
```

Notice the convention: here $t=0$ is noise and $t=1$ is data, the opposite of DDPM's step index. Libraries disagree, so check before reusing code.

## Why straight paths matter

An ODE solver's error grows with path curvature. Diffusion's probability-flow paths curve noticeably, so few-step Euler sampling drifts off course. Linear interpolation paths are straight per pair; the learned marginal field is not perfectly straight, but it is much straighter, and "reflow" (retraining on the model's own noise-image pairs) straightens it further, towards one-step generation. Flow matching is also simpler: no $\beta$ schedule, no $\bar\alpha$, no variance bookkeeping. Gaussian diffusion is a special case with a particular curved path and weighting, so the two families differ in path choice and time weighting more than in kind. Stable Diffusion 3 and Flux use rectified-flow objectives with time sampled from a logit-normal distribution that emphasises the middle of the path.

<div class="take">
<div class="lab">Key takeaway</div>

a noise predictor is a score model, the score defines a reverse SDE and a deterministic probability-flow ODE with the same marginals, and flow matching learns an ODE velocity field directly by regressing $x_1-x_0$ along straight noise-to-data paths, which makes few-step sampling easier.

</div>
