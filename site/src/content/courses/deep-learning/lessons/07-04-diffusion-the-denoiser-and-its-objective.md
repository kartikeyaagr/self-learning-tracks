---
id: "07.04"
course: "deep-learning"
module: "07"
slug: "diffusion-the-denoiser-and-its-objective"
title: "Diffusion: the denoiser and its objective"
keyIdeas: ["Each reverse step is a Gaussian whose mean is written through a predicted noise $\\epsilon_\\theta(x_t, t)$; the ELBO then reduces, up to per-step weights, to $\\|\\epsilon-\\epsilon_\\theta(x_t,t)\\|^2$, and dropping those weights gives the **simple loss**.", "Training is one regression step: sample $x_0$, a step $t$ and noise $\\epsilon$, build $x_t$ in closed form, and regress the noise. No chain is ever unrolled, so it trains like ordinary supervised learning.", "$\\epsilon$, $x_0$ and $v=\\sqrt{\\bar\\alpha_t}\\,\\epsilon-\\sqrt{1-\\bar\\alpha_t}\\,x_0$ are interchangeable targets. $\\epsilon$-MSE equals $x_0$-MSE weighted by SNR$(t)$, so it neglects very noisy steps and breaks at zero terminal SNR; $v$ stays well scaled at both ends.", "The per-step loss is very uneven (high at small $t$, near zero at large $t$ for $\\epsilon$), so the average training loss is a weak progress signal; track it per noise-level bucket and judge by samples.", "One network serves all $T$ steps, told which step it is through a sinusoidal timestep embedding injected into every residual block; without it, samples blur.", "The classic denoiser is a U-Net with residual blocks, GroupNorm, SiLU and self-attention at low resolutions; transformers on latent patches (DiT, with adaLN-Zero conditioning) are the main alternative."]
question: "You switch a working model from $\\epsilon$-prediction to zero-terminal-SNR, so at $t=T$ the input is pure noise. Training still uses $\\epsilon$-MSE, and samples come out as washed-out mush. Explain what the network learns at $t=T$ and what goes wrong when the sampler converts $\\hat\\epsilon$ into $\\hat x_0$. What would you change?"
answer: "At $t=T$ with $\\bar\\alpha_T=0$, the input is $x_T=\\epsilon$ exactly. The $\\epsilon$-MSE optimum is therefore the identity: the network copies its input, the loss there is trivially zero, and it learns nothing about the data at that step. In SNR terms, $\\epsilon$-MSE is $x_0$-MSE weighted by SNR$(t)$, which is exactly 0 at $t=T$ and tiny near it, so the steps that set global layout and brightness get almost no training signal. The sampler, however, needs $\\hat x_0$ (or a mean) and computes $\\hat x_0=(x_t-\\sqrt{1-\\bar\\alpha_t}\\hat\\epsilon)/\\sqrt{\\bar\\alpha_t}$, which is $0/0$ at $t=T$ and wildly amplified near it: any small error in $\\hat\\epsilon$ is divided by a tiny $\\sqrt{\\bar\\alpha_t}$ (on the standard linear schedule this factor is already about 60 at $t=900$). The first, most important steps are thus driven by noise or by whatever clipping you apply, which gives the washed-out result. The fix is to change the target to $v=\\sqrt{\\bar\\alpha_t}\\epsilon-\\sqrt{1-\\bar\\alpha_t}x_0$ (or to $x_0$). At $t=T$, $v=-x_0$, so the network must predict the data mean given no information, a well-posed and well-scaled target; its implied weight on $x_0$-error is $1+\\mathrm{SNR}\\ge 1$, and converting to $\\hat x_0=\\sqrt{\\bar\\alpha_t}x_t-\\sqrt{1-\\bar\\alpha_t}\\hat v$ involves no division. Retrain (or fine-tune) with v-MSE, keep the same schedule in training and sampling, and check per-step-bucket losses to confirm the high-noise buckets now carry signal."
sources: [{"title": "Ho, Jain and Abbeel: Denoising Diffusion Probabilistic Models", "url": "https://arxiv.org/abs/2006.11239", "note": "Section 3 derives the epsilon parameterisation and the simple loss; appendix B has the U-Net details."}, {"title": "Salimans and Ho: Progressive Distillation for Fast Sampling of Diffusion Models", "url": "https://arxiv.org/abs/2202.00512", "note": "Section 4 and appendix D introduce v-prediction and compare parameterisations."}, {"title": "Hang et al.: Efficient Diffusion Training via Min-SNR Weighting Strategy", "url": "https://arxiv.org/abs/2303.09556", "note": "Loss weighting across noise levels as a multi-task balance problem."}, {"title": "Peebles and Xie: Scalable Diffusion Models with Transformers", "url": "https://arxiv.org/abs/2212.09748", "note": "DiT and adaLN-Zero conditioning, the transformer alternative to the U-Net."}]
practice: ["noise-prediction-loss", "toy-diffusion-training-step", "prediction-parameterization", "sinusoidal-timestep-embedding", "time-conditioned-residual-block"]
tags: ["diffusion", "U-Net", "v-prediction"]
tools: ["PyTorch"]
---

## What you will learn

- What the denoising network has to output, and why predicting the *noise* is enough.
- Where the one-line loss $\|\epsilon-\epsilon_\theta(x_t,t)\|^2$ comes from, and what was dropped to get it.
- The whole training loop, trained for real on a toy dataset.
- The three interchangeable targets ($\epsilon$, $x_0$, $v$), how to convert between them, and which noise levels each one favours.
- How one network handles all 1000 noise levels (timestep embedding), and what U-Net and DiT denoisers look like.

*Before you start:* lesson 07.03 (the forward process, $\bar\alpha_t$, the closed form $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$, SNR), the ELBO from 07.01, residual blocks and normalisation (modules 03-04), the U-Net from the Computer Vision module, and basic PyTorch.

Lesson 07.03 built the fixed half of a diffusion model: a recipe that blends a clean sample with Gaussian noise, step by step, until only noise is left. This lesson builds the half that is learned. A single neural network, the **denoiser**, looks at a noisy input and says how to make it a little cleaner. Lesson 07.05 then runs that network repeatedly to turn fresh noise into new samples.

## The idea in one picture

Here is one complete training example, with a "data point" of just two numbers so you can follow every value. Take the linear DDPM schedule from 07.03 and step $t=207$, where $\bar\alpha_t=0.64$, so $\sqrt{\bar\alpha_t}=0.8$ and $\sqrt{1-\bar\alpha_t}=0.6$.

| Quantity | Coordinate 1 | Coordinate 2 | How it is made
| --- | --- | --- | --- |
| Clean sample $x_0$ | 0.80 | -0.30 | drawn from the training set
| Noise $\epsilon$ | 0.50 | -1.20 | drawn from $\mathcal N(0, I)$
| Noisy input $x_t$ | 0.94 | -0.96 | $0.8\,x_0 + 0.6\,\epsilon$
| Network output $\hat\epsilon$ | 0.40 | -1.00 | $\epsilon_\theta(x_t, t)$
| Squared error | 0.01 | 0.04 | $(\epsilon-\hat\epsilon)^2$, mean 0.025

That is the entire learning signal: the network sees $x_t$ and $t$, guesses which noise was mixed in, and is graded by mean squared error against the noise we actually used. It never sees $x_0$ directly, but because $x_t$ and $\epsilon$ together pin down $x_0$, guessing the noise is the same as guessing the clean sample.

~~~dg-fig 07.04-field.svg
A denoiser trained on the two-moons toy data. Each arrow runs from a noisy input to the clean point the network infers from it. With little noise (left) the arrows land on the moons; with heavy noise (right) the network can only guess the average, so the arrows land near the middle.
~~~

Three terms used throughout:

- The **denoiser** $\epsilon_\theta(x_t,t)$ is the neural network, with weights $\theta$. Its input is a noisy sample plus the step number; its output has the same shape as the sample.
- **Noise prediction** means the network outputs its estimate $\hat\epsilon$ of the noise that was added.
- The **predicted clean sample** $\hat x_0$ is what you get by subtracting that noise estimate back out. The figure shows it is a best *average* guess: at $t=600$ many clean points could have produced the same noisy input, so the network points at their mean. Sharp samples come from many small steps (07.05), not one big jump.

## Step 1: what the reverse step has to output

Generation runs the chain backwards: start from noise $x_T$ and repeatedly sample $x_{t-1}$ given $x_t$. We model each backward step as a Gaussian,

$$p_\theta(x_{t-1}\mid x_t)=\mathcal N\big(\mu_\theta(x_t,t),\ \sigma_t^2 I\big).$$

In words: the slightly cleaner sample is a bell curve centred on a mean $\mu_\theta$ computed by the network, with a fixed spread $\sigma_t$ (DDPM uses $\sigma_t^2=\beta_t$). This is sound because when each forward step adds only a little noise, the true backward step is very close to Gaussian.

We train $\mu_\theta$ as in a VAE, by maximising the **ELBO**, a lower bound on the data log-likelihood (07.01). For diffusion it splits into one KL divergence per step, between two Gaussians:

- the **forward posterior** $q(x_{t-1}\mid x_t, x_0)$: the backward step *when the clean sample is known*, an exact Gaussian by Bayes' rule (formula in 07.05);
- our model $p_\theta(x_{t-1}\mid x_t)$, which does not see $x_0$.

With equal fixed variances, each KL is just a scaled squared distance between the means. So training means: make the network's mean match the posterior mean.

## Step 2: write the mean through a noise prediction

Ho, Jain and Abbeel (2020) made the substitution that defined the field. Since $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$, knowing $\epsilon$ is the same as knowing $x_0$, so the posterior mean can be written in terms of noise, and the network predicts the noise:

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)$$

Symbol by symbol:

- $x_t$ is the current noisy input; $\epsilon_\theta(x_t,t)$ is the network's noise guess.
- $\beta_t$ is the amount of noise the forward process added at this one step, and $\alpha_t=1-\beta_t$.
- $\bar\alpha_t$ is the product of all $\alpha$ up to $t$, so $\sqrt{1-\bar\alpha_t}$ is the total noise level in $x_t$.
- Read it as: remove a small slice ($\beta_t/\sqrt{1-\bar\alpha_t}$) of the predicted total noise, then undo the slight shrink the forward step applied (divide by $\sqrt{\alpha_t}$).

Number check with the table above: at $t=207$ the schedule gives $\beta_t=0.0042$, so the slice is $0.0042/0.6=0.007$. One reverse step removes under 1% of the estimated noise. That is why sampling needs many steps.

With this form, the squared distance between means in each KL becomes a weighted squared distance between noises, $\lambda_t\|\epsilon-\epsilon_\theta\|^2$, with a weight $\lambda_t$ that depends only on the schedule.

## Step 3: drop the weights, get the simple loss

Ho et al. then dropped the weights $\lambda_t$ and found the result trained *better* than the true bound, as measured by sample quality. The **simple loss** is

$$L_\text{simple}=\mathbb E_{x_0,\ t\sim U\{1..T\},\ \epsilon\sim\mathcal N(0,I)}\ \big\|\epsilon-\epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\ t\big)\big\|^2.$$

The expectation says how to build each training example: pick a real sample, pick a step uniformly from 1 to $T$, draw fresh noise, mix them in closed form, and regress the noise.

~~~dg-flow One training step: no chain, just a random noise level and a regression target.
x0 from data -(pick t, draw eps)-> xt closed form -(network)-> *eps hat -(MSE vs eps)-> Loss
~~~

That is the entire training algorithm (the full code is below). Each example teaches one noise level; over many steps the network sees all of them. Nothing is ever run 1000 steps deep, so diffusion trains as cheaply and stably as ordinary regression (compare the balancing act of GANs in 07.02).

~~~dg-fig 07.04-loss.svg
Left: the noise-prediction loss of the toy denoiser while training, with and without the step number as input. Right: the trained model's loss measured separately at each step. Small t is hardest (there is barely any noise to find); at large t the input is almost pure noise, so "predict the input" is nearly perfect.
~~~

Read the right panel carefully. With the timestep, the loss is about 0.98 at $t=1$, 0.57 at $t=50$, 0.08 at $t=500$ and essentially 0 at $t=1000$. A high loss at small $t$ does not mean the model is bad there: when only a whisper of noise was added, *which* whisper is genuinely hard to tell apart from detail in the data. And the near-zero loss at $t=1000$ is not an achievement: $x_T$ is almost exactly $\epsilon$, so copying the input wins. The average loss you watch during training mixes these very different tasks, which is why it plateaus early (left panel) and is a poor progress metric on its own.

## Step 4: choose what the network predicts

The network can output any of three quantities; the other two follow by algebra from the closed form.

- **Noise**, $\hat\epsilon$. Convert with $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$.
- **Clean sample**, $\hat x_0$ directly.
- **Velocity**, $v=\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x_0$, with $\hat x_0=\sqrt{\bar\alpha_t}\,x_t-\sqrt{1-\bar\alpha_t}\,\hat v$.

At the optimum all three describe the same denoiser. In training they differ, because an MSE on one is a *weighted* MSE on another. Writing everything as error on the clean sample:

$$\|\epsilon-\hat\epsilon\|^2=\text{SNR}(t)\,\|x_0-\hat x_0\|^2,\qquad \|v-\hat v\|^2=\big(1+\text{SNR}(t)\big)\,\|x_0-\hat x_0\|^2,$$

where $\text{SNR}(t)=\bar\alpha_t/(1-\bar\alpha_t)$ is the signal-to-noise ratio from 07.03.

<div class="ex">
<div class="lab">Worked example</div>

**the same mistake, three ways of scoring it**

Use the table example: $\bar\alpha_t=0.64$, so $\text{SNR}=0.64/0.36=1.78$. The network guessed $\hat\epsilon=(0.40, -1.00)$.

Convert to a clean estimate: $\hat x_0=(x_t-0.6\,\hat\epsilon)/0.8=\big((0.94-0.24)/0.8,\ (-0.96+0.60)/0.8\big)=(0.875, -0.45)$. The true $x_0$ is $(0.80, -0.30)$.

$x_0$-MSE: $\big(0.075^2+0.15^2\big)/2=(0.005625+0.0225)/2=0.0141$.

$\epsilon$-MSE: 0.025 (from the table), and indeed $1.78\times0.0141=0.025$.

$v$-MSE: true $v=0.8\epsilon-0.6x_0=(-0.08, -0.78)$, guessed $\hat v=0.8\hat\epsilon-0.6\hat x_0=(-0.205, -0.53)$. MSE $=(0.125^2+0.25^2)/2=0.0391$, and $(1+1.78)\times0.0141=0.0391$.

Same prediction, three different loss values. Which one you minimise decides how hard the optimiser pushes at each noise level.

</div>

~~~dg-fig 07.04-weights.svg
How much each target cares about clean-sample error at each step, on a log scale. Epsilon-MSE weights the earliest steps about 10,000 and the last steps about 0.00004: a range of eight orders of magnitude. v-MSE never drops below 1; Min-SNR-5 caps the weight at 5.
~~~

So $\epsilon$-prediction pays almost no attention to very noisy steps, the ones that decide global layout. At the very end it breaks outright, as the next figure shows: converting $\hat\epsilon$ to $\hat x_0$ divides by $\sqrt{\bar\alpha_t}$, which goes to zero.

~~~dg-fig 07.04-blowup.svg
Error in the clean-sample estimate per unit of output error. From an epsilon prediction it grows to about 3x at t = 500, 60x at t = 900 and 157x at t = 1000 on the linear schedule; from a v prediction it never exceeds 1.
~~~

Predicting $x_0$ has the mirror problem: at low noise the network must reproduce fine detail exactly, and the conversion back to $\hat\epsilon$ amplifies its errors. **v-prediction** (Salimans and Ho, 2022) stays well scaled at both ends: at $t=T$, $v=-x_0$, a well-posed target, and converting back divides by nothing small. That is why zero-terminal-SNR schedules (07.03) and distilled few-step samplers use it.

A separate knob is per-step loss weighting. **Min-SNR-$\gamma$** (Hang et al., 2023) treats the steps as competing tasks and clips the effective weight at $\gamma=5$ (for $\epsilon$-prediction: multiply the loss by $\min(\text{SNR},\gamma)/\text{SNR}$), so the many easy low-noise steps stop dominating the gradient. The paper reported roughly 3x faster convergence.

<div class="ex">
<div class="lab">Worked example</div>

**converting a prediction at a noisier step**

At a step with $\bar\alpha_t=0.25$, a pixel has $x_t=0.5$ and the network predicts $\hat\epsilon=0.4$.

$\hat x_0=(0.5-\sqrt{0.75}\cdot0.4)/\sqrt{0.25}=(0.5-0.346)/0.5=0.307$.

The equivalent v-prediction is $\hat v=\sqrt{0.25}\cdot0.4-\sqrt{0.75}\cdot0.307=0.2-0.266=-0.066$.

Check the inverse: $\sqrt{0.25}\cdot0.5-\sqrt{0.75}\cdot(-0.066)=0.25+0.057=0.307$. Any sampler can take whichever form it needs from the same network output.

</div>

## Step 5: tell the network which step it is on

One network serves all 1000 noise levels, so it must know $t$: polishing a nearly clean image at $t=50$ and inventing a layout from static at $t=900$ are different jobs. A raw integer spanning 1 to 1000 is awkward for a network, so the step becomes a vector via the transformer-style **sinusoidal embedding**:

$$\text{emb}(t)=\big[\sin(t\,\omega_0),\dots,\sin(t\,\omega_{d/2-1}),\ \cos(t\,\omega_0),\dots,\cos(t\,\omega_{d/2-1})\big],\qquad \omega_k=10000^{-k/(d/2)}.$$

Here $d$ is the embedding size and $\omega_k$ are frequencies spread geometrically from 1 down to about $1/10000$. Fast coordinates distinguish neighbouring steps; slow ones say roughly where in the schedule you are.

~~~dg-fig 07.04-temb.svg
Left: three coordinates of a 64-dimensional step embedding, with periods of about 35, 200 and 2000 steps. Right: the cosine similarity between the code for one step and every other step peaks sharply at that step and drops to a low background, so each step gets a distinct code and close steps get similar ones.
~~~

The embedding then goes through a small MLP (Linear, SiLU, Linear) to make a conditioning vector, and **every residual block** uses it: either by adding a projection of it to the feature maps, or by producing a per-channel scale and shift after normalisation (FiLM, also called adaptive GroupNorm). Continuous-time models feed $\log\text{SNR}$ or the noise level $\sigma$ instead of an integer; the mechanism is identical.

How much does the step input matter? We trained the same small network twice on two moons, once with the step and once with it hidden.

~~~dg-fig 07.04-samples.svg
Left: the training data. Middle: 1000-step DDPM samples from the denoiser that is told the step: two crisp moons. Right: the identical network with the step hidden produces a shapeless cloud, because it has to apply one compromise correction at every noise level.
~~~

Measured: the mean distance from a sample to the nearest training point was 0.030 with the step and 0.161 without. The hidden-step model's loss stays at 0.056 even at $t=1000$, where copying the input gives 0, because it cannot tell it is at the last step.

## Step 6: the network itself

The denoiser must output a tensor the same shape as its input and combine global structure with pixel-exact detail. For images the classic choice is the **U-Net**: a downsampling path, a bottleneck and an upsampling path, with skip connections carrying detail across at every resolution.

~~~dg-layers DDPM-style U-Net for 32x32 images; every block also receives the timestep embedding.
Input 32x32x3 | conv to 128 channels
Down path | 2 residual blocks per level, then downsample: 32, 16, 8, 4
*Middle | residual, self-attention, residual at 4x4
Up path | concatenate the skip from the same resolution, residual blocks, upsample
Output 32x32x3 | GroupNorm, SiLU, conv to the predicted noise
~~~

Diffusion-specific additions:

- Residual blocks of GroupNorm, SiLU and 3x3 convolutions, with the time embedding injected between the two convolutions.
- Self-attention at low resolutions (the 16x16 level and the bottleneck in DDPM), where it is affordable, so distant parts of the image agree.
- A final convolution often initialised to zero, so the untrained model starts from a harmless constant output.

The DDPM CIFAR-10 model has about 36M parameters and reached FID 3.17 (published values). Latent diffusion runs the same U-Net on a 64x64x4 VAE latent instead of pixels and adds cross-attention to text (07.07). The **Diffusion Transformer** (DiT, Peebles and Xie) replaces the U-Net with a ViT over latent patches and injects $t$ and the class through **adaLN-Zero**: the conditioning vector sets each block's normalisation scale and shift plus a zero-initialised gate, so every block starts as the identity. DiT scales predictably with compute, and transformer denoisers underlie most recent large models.

## Training a denoiser in code

A complete, runnable version on two moons. The network is a small residual MLP rather than a U-Net (the data are 2-D points), but the loss, noising and step conditioning are exactly what an image model uses. It trains in a minute or two on one CPU thread.

```python
import math, torch, torch.nn as nn
from sklearn.datasets import make_moons
torch.manual_seed(0); torch.set_num_threads(1)

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
abar = torch.cumprod(1 - betas, 0)                  # abar[t-1] is alpha-bar at step t

X, _ = make_moons(4000, noise=0.05, random_state=0)
data = torch.tensor((X - X.mean(0)) / X.std(0), dtype=torch.float32)

def timestep_embedding(t, dim=32):                  # t: (B,) integer steps
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

class Denoiser(nn.Module):                          # tiny residual MLP; images would use a U-Net
    def __init__(self, h=128):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(32, h), nn.SiLU(), nn.Linear(h, h))
        self.inp = nn.Linear(2, h)
        self.blocks = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(h, h)) for _ in range(3)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(h, 2))
    def forward(self, x, t):
        h, c = self.inp(x), self.time(timestep_embedding(t))
        for blk in self.blocks:
            h = h + blk(h + c)                      # the step enters every block
        return self.out(h)

model = Denoiser()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
for step in range(1, 3001):
    x0 = data[torch.randint(0, len(data), (512,))]
    t = torch.randint(1, T + 1, (512,))
    eps = torch.randn_like(x0)
    a = abar[t - 1][:, None]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps      # closed-form noising (07.03)
    loss = ((model(xt, t) - eps) ** 2).mean()       # L_simple
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (1, 500, 3000):
        print(step, round(loss.item(), 3))          # 1 1.048 / 500 0.23 / 3000 0.227

with torch.no_grad():                               # loss at a few fixed steps
    for t_ in (10, 100, 500, 1000):
        a = abar[t_ - 1]; e = torch.randn_like(data)
        xt = a.sqrt() * data + (1 - a).sqrt() * e
        print(t_, round(((model(xt, torch.full((len(data),), t_)) - e) ** 2).mean().item(), 3))
# 10 0.87 / 100 0.679 / 500 0.079 / 1000 0.001
```

The pattern matches the figure: hardest at small $t$, trivial at $t=1000$. The first loss is about 1.0: an untrained network outputs roughly zero and the noise has unit variance. Sampling is the subject of 07.05.

## Checking that training worked

The training loss alone tells you little, because it averages easy and hard noise levels. Check these instead:

- **Loss per step bucket.** Log the loss separately for, say, ten ranges of $t$. Each bucket should fall over training. Expect high values at small $t$ and near zero at large $t$ for $\epsilon$-prediction, as in the loss figure.
- **The untrained baseline.** At initialisation the loss should be close to 1 (the variance of $\epsilon$) for data scaled to about unit size. Much larger means a scaling bug in $x_0$ or the schedule.
- **Reconstruction.** Noise a few validation samples to $t=200$ or $500$ and look at $\hat x_0$: a blurred version of the original, blurrier as $t$ grows.
- **Samples.** In the end only samples matter: FID for images, a distance to held-out data for toy problems. Sample from an EMA (exponential moving average) copy of the weights, which is standard practice because it is usually better than the raw weights.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Loss starts far above 1 | data not scaled to about [-1, 1], or wrong $\bar\alpha$ indexing | rescale data; check abar[t-1] is used for step t
| Samples smeared or average-looking | step not reaching the network, or embedding wrongly scaled | inject the step into every block; check the embedding varies with t
| Washed-out results with a zero-terminal-SNR schedule | $\epsilon$-prediction at $\bar\alpha_T=0$ is the identity, and $\hat x_0$ divides by zero | switch to v-prediction
| Fine detail poor, layout good | low-noise steps under-trained or under-weighted | check per-bucket loss; adjust weighting or sample more small $t$
| Global layout poor, detail good | high-noise steps get almost no weight under $\epsilon$-MSE | v-prediction, Min-SNR weighting, or a noisier schedule
| Good loss, bad samples | evaluating raw weights, or train and sample schedules differ | sample with EMA weights; share one schedule object

## Going deeper

**What weights did we drop?** With $\sigma_t^2=\beta_t$, the per-step ELBO weight on $\|\epsilon-\epsilon_\theta\|^2$ works out to $\lambda_t=\beta_t/\big(2\alpha_t(1-\bar\alpha_t)\big)$. On the linear schedule that is 0.5 at $t=1$, 0.074 at $t=10$ and about 0.005-0.01 from $t=100$ onwards. So the true bound spends most of its weight on the first few, nearly noise-free steps, which matter for likelihood (exact bits per pixel) but barely for how samples look. $L_\text{simple}$ flattens this to 1 everywhere, shifting effort towards harder, noisier steps. That is why it gives better samples but worse log-likelihood; Nichol and Dhariwal later recovered good likelihoods by learning $\sigma_t$ with a hybrid loss.

**What the optimal denoiser computes.** For MSE the best possible prediction is a conditional mean: $\epsilon^*(x_t,t)=\mathbb E[\epsilon\mid x_t]$, and correspondingly $\hat x_0^*=\mathbb E[x_0\mid x_t]$. That is exactly what the field figure shows: at high noise the posterior over clean points is broad, so its mean sits between the moons. Tweedie's formula links this mean to the gradient of the log-density of noisy data, $\epsilon^*(x_t,t)=-\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_t(x_t)$. A noise predictor is therefore a **score** estimator in disguise, the bridge to score matching in 07.06.

**Weighting and step sampling are the same knob.** Sampling $t$ with density $p(t)$ and weight $w(t)$ gives the same expected gradient as uniform $t$ with weight proportional to $p(t)\,w(t)$. Importance sampling of $t$, log-SNR shifts and Min-SNR are all choices of one function: how much each noise level counts. In continuous time, the common objectives are all weighted integrals over log-SNR, and the ELBO is one particular weighting (Kingma et al., 2023).

## Key terms

| Term | Meaning
| --- | --- |
| Denoiser | The network $\epsilon_\theta(x_t,t)$ that takes a noisy sample and step and predicts the noise (or $x_0$ or $v$)
| Forward posterior | $q(x_{t-1}\mid x_t,x_0)$, the exact backward step when the clean sample is known; the target the model's step must match
| $L_\text{simple}$ | Unweighted MSE between true and predicted noise, averaged over uniform steps
| $\epsilon$ / $x_0$ / $v$ prediction | Three interchangeable outputs; they differ in how training weights each noise level
| v (velocity) | $\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x_0$, a target that stays well scaled at both ends of the schedule
| SNR weighting | How much clean-sample error at step $t$ counts in the loss; $\epsilon$-MSE implies weight SNR$(t)$
| Min-SNR-$\gamma$ | Loss weighting that caps the effective weight at $\gamma$ (typically 5)
| Sinusoidal timestep embedding | Sines and cosines of $t$ at many frequencies, used to tell one network which step it is on
| adaLN-Zero / FiLM | Conditioning by predicting normalisation scale and shift from the step (and class) vector

<div class="take">
<div class="lab">Key takeaway</div>

diffusion training is plain regression: noise a clean sample to a random step in closed form and train one timestep-conditioned network (usually a U-Net or DiT) to predict the noise, or equivalently $x_0$ or $v$, with a weighting that decides which noise levels it learns best.

</div>
