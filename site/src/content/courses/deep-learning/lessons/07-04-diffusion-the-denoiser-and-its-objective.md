---
id: "07.04"
course: "deep-learning"
module: "07"
slug: "diffusion-the-denoiser-and-its-objective"
title: "Diffusion: the denoiser and its objective"
keyIdeas: ["The reverse step is a Gaussian whose mean is written in terms of a predicted noise $\\epsilon_\\theta(x_t, t)$; the ELBO then reduces, up to weights, to $\\|\\epsilon-\\epsilon_\\theta(x_t,t)\\|^2$.", "Training is one regression step: sample $x_0$, a step $t$, noise $\\epsilon$; build $x_t$ in closed form; regress the noise. No chain is ever unrolled.", "$\\epsilon$, $x_0$ and $v=\\sqrt{\\bar\\alpha_t}\\,\\epsilon-\\sqrt{1-\\bar\\alpha_t}\\,x_0$ are interchangeable targets; they differ in loss weighting across noise levels and in behaviour near the ends of the schedule.", "One network serves all $T$ steps, told which step it is through a sinusoidal timestep embedding injected into every residual block.", "The classic denoiser is a U-Net with residual blocks, GroupNorm, SiLU and self-attention at low resolutions; transformers on patches (DiT) are the main alternative."]
question: "You switch a working model from $\\epsilon$-prediction to zero-terminal-SNR, so at $t=T$ the input is pure noise. Training still uses $\\epsilon$-MSE, and samples come out as washed-out mush. Explain what the network learns at $t=T$ and what goes wrong when the sampler converts $\\hat\\epsilon$ into $\\hat x_0$."
answer: "At $t=T$ with $\\bar\\alpha_T=0$, the input is $x_T=\\epsilon$ exactly. The $\\epsilon$-MSE optimum is therefore the identity: the network copies its input and learns nothing about the data at that step, while the loss there is trivially zero. The sampler, however, needs $\\hat x_0$ (or a mean) and computes $\\hat x_0=(x_t-\\sqrt{1-\\bar\\alpha_t}\\hat\\epsilon)/\\sqrt{\\bar\\alpha_t}$, which is $0/0$ at $t=T$ and wildly amplified near it: any small error in $\\hat\\epsilon$ is divided by a tiny $\\sqrt{\\bar\\alpha_t}$. The first, most important steps, which set global layout and brightness, are thus driven by noise or by whatever clipping you apply. Switching the target to $v$ (or $x_0$) fixes it: at $t=T$, $v=-x_0$, so the network must predict the data mean given no information, a well-posed and well-scaled target, and conversion to $\\hat x_0$ involves no division by $\\sqrt{\\bar\\alpha_t}$."
sources: [{"title": "Ho, Jain and Abbeel: Denoising Diffusion Probabilistic Models", "url": "https://arxiv.org/abs/2006.11239", "note": "Section 3 derives the epsilon parameterisation and the simple loss; appendix B has the U-Net details."}, {"title": "Salimans and Ho: Progressive Distillation for Fast Sampling of Diffusion Models", "url": "https://arxiv.org/abs/2202.00512", "note": "Section 4 and appendix D introduce v-prediction and compare parameterisations."}, {"title": "Hang et al.: Efficient Diffusion Training via Min-SNR Weighting Strategy", "url": "https://arxiv.org/abs/2303.09556", "note": "Loss weighting across noise levels as a multi-task balance problem."}, {"title": "Peebles and Xie: Scalable Diffusion Models with Transformers", "url": "https://arxiv.org/abs/2212.09748", "note": "DiT and adaLN-Zero conditioning, the transformer alternative to the U-Net."}]
practice: ["noise-prediction-loss", "toy-diffusion-training-step", "prediction-parameterization", "sinusoidal-timestep-embedding", "time-conditioned-residual-block"]
tags: ["diffusion", "U-Net", "v-prediction"]
tools: ["PyTorch"]
---

## What the network has to predict

The reverse process models each step as a Gaussian, $p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\ \sigma_t^2 I)$. For small $\beta_t$ the true reverse step is close to Gaussian, so this is a sound choice. The DDPM ELBO decomposes into one KL per step between the tractable posterior $q(x_{t-1}\mid x_t, x_0)$ (lesson 07.05 gives its formula) and $p_\theta(x_{t-1}\mid x_t)$. Both are Gaussians with the same fixed variance, so each KL is just a scaled squared distance between means.

Ho et al. then made one substitution that defined the field. Since $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$, knowing $\epsilon$ is the same as knowing $x_0$. So write the mean in terms of a predicted noise:

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)$$

The per-step KL becomes a weighted $\|\epsilon-\epsilon_\theta\|^2$. Dropping the weights gives the **simple loss**, which trained better than the true ELBO in their experiments:

$$L_\text{simple}=\mathbb E_{x_0,\,t\sim U\{1..T\},\,\epsilon}\ \big\|\epsilon-\epsilon_\theta\big(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\ t\big)\big\|^2$$

~~~dg-flow One training step: no chain, just a random noise level and a regression target.
x0 from data -(pick t, draw eps)-> xt closed form -(network)-> *eps hat -(MSE vs eps)-> Loss
~~~

```python
def train_step(model, x0, abar, opt):              # x0: (B, C, H, W) in [-1, 1]
    t = torch.randint(1, T + 1, (x0.shape[0],))
    eps = torch.randn_like(x0)
    a = abar[t - 1].view(-1, 1, 1, 1)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    loss = ((model(xt, t) - eps) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
```

That is the entire training algorithm. Each example teaches the network one noise level, chosen uniformly; over many steps it learns all of them.

## Choice of target: epsilon, x0 or v

The network can output any of three quantities and the others follow algebraically:

- $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$
- $v=\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x_0$, and $\hat x_0=\sqrt{\bar\alpha_t}\,x_t-\sqrt{1-\bar\alpha_t}\,\hat v$

They are equivalent at the optimum but not in training. An MSE on $\epsilon$ equals an MSE on $x_0$ weighted by SNR$(t)$, so it pays little attention to very noisy steps and, at $\bar\alpha_t\to 0$, becomes useless: the input is pure noise, so predicting $\epsilon$ is just copying the input, and converting to $\hat x_0$ divides by $\sqrt{\bar\alpha_t}\approx 0$. Predicting $x_0$ has the mirror problem at low noise. **v-prediction** (Salimans and Ho, 2022) stays well scaled at both ends, which is why zero-terminal-SNR schedules and distilled samplers use it. A separate knob is per-step loss weighting; **Min-SNR-$\gamma$** clips the effective weight at $\gamma=5$ so the easy low-noise steps do not dominate the gradient, and reported roughly 3x faster convergence.

<div class="ex">
<div class="lab">Worked example</div>

**converting a prediction**

At a step with $\bar\alpha_t=0.25$, a pixel has $x_t=0.5$ and the network predicts $\hat\epsilon=0.4$.

$\hat x_0=(0.5-\sqrt{0.75}\cdot0.4)/\sqrt{0.25}=(0.5-0.346)/0.5=0.31$.

The equivalent v-target is $\hat v=\sqrt{0.25}\cdot0.4-\sqrt{0.75}\cdot0.307=0.2-0.266=-0.07$.

Any sampler can take whichever form it needs from the same network output.

</div>

## Telling the network which step it is on

A single network handles all 1000 noise levels, so it must know $t$; the right amount of denoising at $t=50$ and $t=900$ is very different. The step index is encoded with the transformer-style **sinusoidal embedding**: for frequencies $\omega_k=10000^{-k/(d/2)}$, the vector $[\sin(t\omega_k), \cos(t\omega_k)]$. An MLP (Linear, SiLU, Linear) maps it to a conditioning vector, and every residual block adds a projection of it to its feature maps, or uses it to produce a per-channel scale and shift after normalisation (FiLM / adaptive GroupNorm). Continuous-time models feed $\log$SNR or $\sigma$ instead of an integer step; the mechanism is the same.

## The U-Net denoiser

The denoiser must output a tensor the same shape as its input and must combine global structure with pixel-exact detail. The U-Net (covered in this course's Computer Vision module for segmentation) does exactly that: a downsampling path, a bottleneck and an upsampling path with skip connections at every resolution. Diffusion U-Nets add:

~~~dg-layers DDPM-style U-Net for 32x32 images; every block also receives the timestep embedding.
Input 32x32x3 | conv to 128 channels
Down path | 2 residual blocks per level, then downsample: 32, 16, 8, 4
*Middle | residual, self-attention, residual at 4x4
Up path | concatenate the skip from the same resolution, residual blocks, upsample
Output 32x32x3 | GroupNorm, SiLU, conv to the predicted noise
~~~

- Residual blocks of GroupNorm, SiLU and 3x3 convolutions, with the time embedding injected between the two convolutions.
- Self-attention at 16x16 and below, where it is affordable, so distant parts of the image agree.
- A final conv often initialised to zero, so the untrained model starts near a sensible output.

The DDPM CIFAR-10 model has about 36M parameters and reached FID 3.17. Latent diffusion runs the same U-Net on a 64x64x4 VAE latent and adds cross-attention to text (lesson 07.07). The **Diffusion Transformer** (DiT) replaces the U-Net with a ViT over latent patches and injects $t$ and the class through adaLN-Zero; it scales more predictably with compute and underlies most recent large models.

<div class="take">
<div class="lab">Key takeaway</div>

diffusion training is plain regression: noise a clean sample to a random step in closed form and train one timestep-conditioned network (usually a U-Net or DiT) to predict the noise, or equivalently $x_0$ or $v$, with a weighting that decides which noise levels it learns best.

</div>
