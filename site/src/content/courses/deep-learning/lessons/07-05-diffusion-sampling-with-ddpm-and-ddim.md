---
id: "07.05"
course: "deep-learning"
module: "07"
slug: "diffusion-sampling-with-ddpm-and-ddim"
title: "Diffusion: sampling with DDPM and DDIM"
keyIdeas: ["A **sampler** turns the trained noise predictor into data by starting from $x_T\\sim\\mathcal N(0,I)$ and calling the network repeatedly. Each call is a full forward pass, so the step count is the cost.", "Each reverse step aims at the forward posterior $q(x_{t-1}\\mid x_t,x_0)$, a Gaussian with closed-form mean and variance $\\tilde\\beta_t=\\frac{1-\\bar\\alpha_{t-1}}{1-\\bar\\alpha_t}\\beta_t$, evaluated at the network's estimate $\\hat x_0$.", "DDPM ancestral sampling walks all $T$ steps: compute the mean from $\\hat\\epsilon$, add fresh noise $\\sigma_t z$ (none on the final step), repeat. That is 1000 network calls per sample, and its one-step formulas break if you skip steps.", "DDIM uses the same network as \"predict $\\hat x_0$, then re-noise to any lower level\", with a dial $\\eta$ for fresh noise. It keeps the training marginals, so it can visit 50 or 20 steps instead of 1000 without retraining.", "With $\\eta=0$ sampling is a fixed map from noise to data (an Euler discretisation of the probability-flow ODE), which gives reproducible seeds, smooth interpolation and inversion; very low step counts need better solvers or distillation."]
question: "A DDIM sampler with 20 steps on a pixel-space model produces good images. Your teammate switches to DDPM ancestral sampling but keeps the same 20-step subsequence, using $\\sigma_t^2=\\beta_t$ from the original 1000-step schedule at each visited step. Images become noisy and grainy. Why?"
answer: "Ancestral DDPM is derived for consecutive steps with small $\\beta_t$: the coefficient on $\\hat\\epsilon$, $\\beta_t/\\sqrt{1-\\bar\\alpha_t}$, the rescaling $1/\\sqrt{\\alpha_t}$ and the noise $\\sigma_t z$ assume that the chain removes only one step's worth of noise. On a 20-step subsequence each jump spans about 50 original steps, but the update still removes noise using one step's $\\beta_t$ (about 0.02 or less) and then adds fresh noise sized for a single step. The mean therefore removes far too little noise per jump while new noise keeps arriving, so the sample reaches $t=0$ with much of its starting noise still in it. On the lesson's toy data this exact bug gave a sample spread of about 0.95 instead of 0.51, with about 30% of samples outside $[-1,1]$. The fix is to compute the jump's effective quantities from the visited levels: the effective $\\beta$ between $t$ and $t'$ is $1-\\bar\\alpha_t/\\bar\\alpha_{t'}$, and the matching posterior variance follows from it. DDIM's general update with $\\eta=1$ does exactly this automatically, and $\\eta=0$ (deterministic) usually gives even better results at 20 steps."
sources: [{"title": "Song, Meng and Ermon: Denoising Diffusion Implicit Models", "url": "https://arxiv.org/abs/2010.02502", "note": "Sections 3 and 4: non-Markovian forward processes, the eta family and step skipping."}, {"title": "Ho, Jain and Abbeel: Denoising Diffusion Probabilistic Models", "url": "https://arxiv.org/abs/2006.11239", "note": "Algorithm 2 (ancestral sampling) and the posterior in equation 7."}, {"title": "Nichol and Dhariwal: Improved Denoising Diffusion Probabilistic Models", "url": "https://arxiv.org/abs/2102.09672", "note": "Learned reverse variances and why they matter for fewer sampling steps."}, {"title": "Hugging Face Diffusers: Schedulers", "url": "https://huggingface.co/docs/diffusers/api/schedulers/overview", "note": "How production code exposes DDPM, DDIM and ODE-solver samplers over the same model."}]
practice: ["ddim-step", "ddpm-ancestral-step", "posterior-mean-and-variance", "toy-ddpm-sampler"]
tags: ["diffusion", "DDIM", "sampling"]
tools: ["PyTorch"]
---

## What you will learn

- What a **sampler** is: the loop that turns a trained noise predictor into new data, and why its cost is counted in network calls.
- The forward posterior $q(x_{t-1}\mid x_t, x_0)$: the target every reverse step aims at, and how to evaluate it without knowing $x_0$.
- DDPM ancestral sampling, step by step, including the two standard choices for the noise it adds.
- DDIM: the same trained network used as "predict the clean image, then re-noise to a lower level", with a dial $\eta$ for randomness.
- Why DDIM can skip from 1000 steps to 50 or 20, what goes wrong if you skip with the wrong formula, and what a deterministic sampler lets you do (fixed seeds, interpolation, inversion).

*Before you start:* you need the forward process and the closed form $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ from lesson 07.03, and the noise-prediction network $\epsilon_\theta(x_t,t)$ and its training loss from lesson 07.04. Comfort with Gaussians (mean, variance, adding independent noise) is assumed.

## Where we are

Lesson 07.03 built a schedule $\bar\alpha_t$ that slowly destroys data, and lesson 07.04 trained a network $\epsilon_\theta(x_t, t)$ that guesses the noise inside a noisy input. Training never ran a chain. This lesson does: start from pure noise and call the network repeatedly until data comes out.

Every call is a full forward pass of a large U-Net or transformer, so the number of calls is essentially the cost of a sample. DDPM uses all 1000 steps; DDIM gets comparable results in 50 or fewer with the same network. (Serving-side step budgets are covered in the Inference Engineering course, Specialized Inference Workloads; here we do the maths.)

## The idea in one picture

To see samplers clearly we use a toy dataset with one "pixel". Its values come from two bumps, like an image region that is either dark or bright:

| Bump | Share of data | Centre | Spread (std)
| --- | --- | --- | --- |
| dark | 40% | -0.6 | 0.08
| bright | 60% | +0.5 | 0.12

For this data the best possible noise predictor can be written down exactly, so we use it in place of a trained network. So every difference below comes from the sampler, not from training.

~~~dg-fig 07.05-trajectories.svg
Eight runs that start from the same eight noise values. Left: DDPM adds fresh noise at each of 1000 steps, so the paths jitter. Middle: DDIM with η = 0 takes 50 smooth, noise-free jumps. Right: the 50-step DDIM samples (histogram) land on both bumps of the true data (line).
~~~

Time runs left to right, from step 1000 down to 0. Each path starts as a draw from $\mathcal N(0,1)$; the "dark or bright" decision happens late, in the last few hundred steps. Both samplers end on the two bumps, but one needed 1000 network calls and the other 50.

Terms used throughout:

- A **sampler** is the algorithm that maps starting noise $x_T$ to a sample $x_0$ using the trained network.
- A **network call** (often written NFE, number of function evaluations) is one evaluation of $\epsilon_\theta$.
- **Ancestral** sampling draws each $x_{t-1}$ from a distribution given $x_t$, injecting fresh randomness every step.
- A **deterministic** sampler adds no noise after $x_T$: the same start always gives the same output.

## Step 1: what one reverse step aims at

Going backwards one step means guessing $x_{t-1}$ from $x_t$. That is hard in general, but it becomes easy if someone tells you the clean data $x_0$. The forward process run backwards with $x_0$ known is called the **forward posterior**, and Bayes' rule on Gaussians gives it in closed form:

$$q(x_{t-1}\mid x_t,x_0)=\mathcal N\big(\tilde\mu_t,\ \tilde\beta_t I\big),\qquad \tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t,\qquad \tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t$$

The symbols: $\beta_t$ is the noise added at forward step $t$, $\alpha_t=1-\beta_t$, and $\bar\alpha_t$ is the product of all $\alpha$ up to $t$ (the fraction of signal variance left). The mean $\tilde\mu_t$ is a weighted average of "where the data is" ($x_0$) and "where we are now" ($x_t$). The variance $\tilde\beta_t$ is slightly smaller than $\beta_t$, because knowing $x_0$ removes some uncertainty.

At sampling time we do not know $x_0$. So we plug in the network's guess. From the closed form of the forward process,

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}},\qquad \hat\epsilon=\epsilon_\theta(x_t,t).$$

Substituting $\hat x_0$ into $\tilde\mu_t$ and simplifying gives exactly the mean from lesson 07.04:

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\hat\epsilon\Big).$$

So "predict the noise" and "predict the clean image, then use the posterior" are two ways of writing the same step. Keep the second reading in mind; DDIM is built on it.

## Step 2: DDPM ancestral sampling

The original sampler (Ho et al., Algorithm 2) walks every step:

1. Draw $x_T\sim\mathcal N(0, I)$.
2. For $t=T,\dots,1$: compute $\hat\epsilon=\epsilon_\theta(x_t,t)$ and the mean $\mu_\theta$ above, then set $x_{t-1}=\mu_\theta+\sigma_t z$ with fresh $z\sim\mathcal N(0, I)$. On the last step ($t=1$) add no noise.
3. Return $x_0$, clipped to $[-1, 1]$ for pixel data.

The only open choice is the noise size $\sigma_t^2$. DDPM tried $\sigma_t^2=\beta_t$ and $\sigma_t^2=\tilde\beta_t$ and found they work about equally well at $T=1000$. They are the two extremes of the true reverse variance: $\beta_t$ is right when the data is pure noise-like, $\tilde\beta_t$ is right when the data is a single point.

~~~dg-fig 07.05-variances.svg
The two standard noise sizes on the DDPM linear schedule. Left: on a log scale the curves lie on top of each other except at the very first steps. Right: their ratio is 0.45 at t = 2, 0.85 at t = 10 and above 0.96 from t = 50 on.
~~~

So with 1000 steps the choice hardly matters: the two differ only in the last few steps, where noise is tiny. With few steps it matters more, which is why Improved DDPM (Nichol and Dhariwal) learns an interpolation between them.

<div class="ex">
<div class="lab">Worked example</div>

**one DDPM step by hand**

Take a step with $\beta_t=0.02$ (so $\alpha_t=0.98$) at a level where $\bar\alpha_t=0.5$; then $\bar\alpha_{t-1}=0.5/0.98=0.5102$. A pixel has $x_t=1.0$ and the network predicts $\hat\epsilon=0.6$.

Mean from the noise form: the correction is $\frac{0.02}{\sqrt{0.5}}\times0.6=0.0283\times0.6=0.0170$. Then $\mu=(1.0-0.0170)/\sqrt{0.98}=0.9830\times1.0102=0.9930$.

Same mean from the posterior form: $\hat x_0=(1.0-\sqrt{0.5}\times0.6)/\sqrt{0.5}=(1.0-0.4243)/0.7071=0.8142$. The weights are $\sqrt{0.5102}\times0.02/0.5=0.0286$ on $\hat x_0$ and $\sqrt{0.98}\times0.4898/0.5=0.9697$ on $x_t$. So $\mu=0.0286\times0.8142+0.9697\times1.0=0.0233+0.9697=0.9930$. The two forms agree.

Noise: $\tilde\beta_t=\frac{0.4898}{0.5}\times0.02=0.01959$, so $\sigma_t=0.1400$ (with $\sigma_t^2=\beta_t$ it would be $0.1414$). The step is $x_{t-1}=0.9930+0.14z$.

Notice the scale: the mean moves the pixel by only 0.007, while the injected noise has a standard deviation of 0.14. One DDPM step is a tiny nudge plus a lot of fresh randomness, which is why it needs so many of them.

</div>

Fresh noise lets later steps wash out an early mistake, but each step must stay small: the Gaussian reverse step is only accurate for small $\beta_t$, so you cannot simply skip steps.

## Step 3: DDIM, the same model with a different sampler

Song, Meng and Ermon (2020) noticed that the training loss only uses the marginals $q(x_t\mid x_0)$, never the step-to-step chain. Many forward processes share those marginals, including **non-Markovian** ones, where $x_{t-1}$ depends on $x_0$ as well as $x_t$. A DDPM-trained network is valid for all of them, and each gives a different sampler.

The resulting update, for any earlier level $t'<t$, reads:

$$x_{t'}=\sqrt{\bar\alpha_{t'}}\,\hat x_0+\sqrt{1-\bar\alpha_{t'}-\sigma^2}\;\hat\epsilon+\sigma z,\qquad \sigma=\eta\sqrt{\frac{1-\bar\alpha_{t'}}{1-\bar\alpha_t}}\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t'}}}$$

In words: (1) estimate the clean image $\hat x_0$ from the current noise guess; (2) rebuild a noisy image at the lower level $t'$ by mixing $\hat x_0$ with noise, using the *predicted* noise direction $\hat\epsilon$ for most of it and fresh noise $z$ for the remainder $\sigma$. The parameter $\eta$ is a dial:

- $\eta=0$: no fresh noise at all. The sampler is **deterministic**.
- $\eta=1$: $\sigma^2$ equals the posterior variance $\tilde\beta$ computed for the jump from $t$ to $t'$. With consecutive steps this is exactly DDPM with $\sigma_t^2=\tilde\beta_t$.
- In between: partly random.

~~~dg-flow One DDIM step: denoise to an estimate of x0, then jump straight to the next noise level.
xt -(network)-> eps hat -(invert closed form)-> *x0 hat -(re-noise to t prime)-> x t prime
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one deterministic DDIM jump**

Jump from a level with $\bar\alpha_t=0.5$ to $\bar\alpha_{t'}=0.8$, a large jump. A pixel has $x_t=1.0$, the network predicts $\hat\epsilon=0.6$, and $\eta=0$ so $\sigma=0$.

Clean estimate: $\hat x_0=(1.0-\sqrt{0.5}\times0.6)/\sqrt{0.5}=(1.0-0.4243)/0.7071=0.8142$.

Re-noise: $x_{t'}=\sqrt{0.8}\times0.8142+\sqrt{0.2}\times0.6=0.7283+0.2683=0.9966$.

The pixel keeps its noise direction (+0.6) but at a smaller size ($\sqrt{0.2}=0.447$ instead of $\sqrt{0.5}=0.707$), and its signal part grows from $0.707\times0.814$ to $0.894\times0.814$.

With $\eta=1$ instead: $\sigma=\sqrt{0.2/0.5}\times\sqrt{1-0.5/0.8}=0.632\times0.612=0.387$, and the coefficient on $\hat\epsilon$ shrinks to $\sqrt{0.2-0.15}=0.224$. Most of the noise at the new level is then fresh rather than reused.

</div>

## Step 4: skipping steps

Because the DDIM update can jump between any two levels, you can run it on a short decreasing list of steps, such as every 20th step (50 calls) or every 100th (10 calls). Nothing is retrained. The original paper reported 10 to 50 times faster sampling at comparable quality, and found that deterministic DDIM degrades far more gracefully than stochastic sampling as the step count drops.

The toy data shows the same pattern. We measure the distance between sampled and true values with the 1-D Wasserstein distance (W1, the average gap between matching quantiles; lower is better).

~~~dg-fig 07.05-steps.svg
Distance to the true data against the number of network calls (100,000 samples each). Deterministic DDIM (η = 0) is best at low step counts; at 1000 steps every η reaches the sampling noise floor. Reusing the one-step DDPM formula on skipped steps (dashed) stays bad until all 1000 steps are used.
~~~

Three things to notice. With 20 calls, $\eta=0$ reaches W1 = 0.038 against 0.048 for $\eta=1$; with 50 calls it is 0.019 against 0.031, so the deterministic sampler buys roughly a factor of two in calls at the same quality. At 1000 calls all three settings reach the floor set by using a finite number of samples (about 0.002), so randomness costs nothing once steps are tiny. And the buggy curve shows why "just run DDPM on fewer steps" fails, which we return to under What goes wrong.

How you space the visited steps also matters. Even spacing in $t$ on a linear schedule spends many calls at very high noise, where little changes. Production samplers often space steps evenly in log signal-to-noise ratio or use the EDM spacing (lesson 07.06), and they always end on the clean level so the final update lands on data.

## Sampling in code

Both samplers in PyTorch. The `model` here is the exact noise predictor for the two-bump toy data, so the code runs in seconds on a CPU; replace it with your trained network.

```python
import torch
torch.manual_seed(0)
T = 1000
beta = torch.cat([torch.zeros(1), torch.linspace(1e-4, 0.02, T, dtype=torch.float64)])
alpha = 1 - beta
abar = torch.cumprod(alpha, 0)            # abar[0] = 1: "step 0" is clean data

PI, MU, SD = torch.tensor([.4, .6]), torch.tensor([-.6, .5]), torch.tensor([.08, .12])
def model(x, t):                          # exact eps-prediction for the toy mixture
    a = abar[t]; v = a * SD**2 + 1 - a; m = a.sqrt() * MU
    logw = PI.log() - 0.5 * v.log() - 0.5 * (x[:, None] - m) ** 2 / v
    x0 = (logw.softmax(1) * (MU + a.sqrt() * SD**2 / v * (x[:, None] - m))).sum(1)
    return (x - a.sqrt() * x0) / (1 - a).sqrt()

@torch.no_grad()
def ddpm_sample(n):
    x = torch.randn(n, dtype=torch.float64)
    for t in range(T, 0, -1):
        eps = model(x, t)
        mean = (x - beta[t] / (1 - abar[t]).sqrt() * eps) / alpha[t].sqrt()
        var = (1 - abar[t - 1]) / (1 - abar[t]) * beta[t]          # beta-tilde
        x = mean + var.sqrt() * torch.randn_like(x) if t > 1 else mean
    return x.clamp(-1, 1)

@torch.no_grad()
def ddim_sample(n, steps=50, eta=0.0, x=None):
    ts = torch.linspace(T, 0, steps + 1).round().long()          # T, ..., 0
    x = torch.randn(n, dtype=torch.float64) if x is None else x
    for t, tp in zip(ts[:-1], ts[1:]):
        a, ap = abar[t], abar[tp]
        eps = model(x, t)
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        sigma = eta * ((1 - ap) / (1 - a)).sqrt() * (1 - a / ap).sqrt()
        x = ap.sqrt() * x0 + (1 - ap - sigma**2).sqrt() * eps + sigma * torch.randn_like(x)
    return x

for name, xs in [('DDPM, 1000 calls', ddpm_sample(50_000)),
                 ('DDIM eta=0, 50 calls', ddim_sample(50_000, 50)),
                 ('DDIM eta=0, 10 calls', ddim_sample(50_000, 10))]:
    left = xs < 0
    print(f'{name:22s} left bump {left.float().mean():.3f}  '
          f'means {xs[left].mean():+.3f} {xs[~left].mean():+.3f}  stds {xs[left].std():.3f} {xs[~left].std():.3f}')
# DDPM, 1000 calls       left bump 0.401  means -0.600 +0.499  stds 0.076 0.117
# DDIM eta=0, 50 calls   left bump 0.396  means -0.592 +0.491  stds 0.066 0.105
# DDIM eta=0, 10 calls   left bump 0.401  means -0.469 +0.431  stds 0.176 0.135

z = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
print(ddim_sample(3, 50, x=z.clone()), ddim_sample(3, 50, x=z.clone()))
# tensor([-0.6109,  0.3974,  0.5568]) printed twice: same noise, same output
```

The true values are a 0.40 share, centres $-0.6$ and $+0.5$, spreads 0.08 and 0.12. DDPM with 1000 calls matches all of them. DDIM with 50 calls is close but slightly too narrow (0.066 instead of 0.08). With 10 calls the bumps are pulled towards each other and smeared: the jumps are too coarse to follow the sharp final stretch.

Two details matter: `abar[0] = 1` makes the last jump land on clean data, and clipping $\hat x_0$ to $[-1, 1]$ guards against wild early predictions in pixel space (turn it off for latent models, whose latents are unbounded).

## Checking the result

For images you would compute FID or look at grids; for any model, a few cheap checks catch most sampler bugs.

- **Compare against your slowest sampler.** Run the full 1000-step DDPM once as a reference. A fast sampler should approach it as you raise the step count. If it never does, the bug is in the update, not in the step budget.
- **Check determinism.** With $\eta=0$, two runs from the same $x_T$ must give identical outputs, as in the code above.
- **Look at the range and spread.** Samples well outside the data range, or a spread much wider than the data, usually mean noise was left in.

Deterministic sampling also turns generation into a fixed function from noise to data, which you can plot for the toy problem:

~~~dg-fig 07.05-map.svg
DDIM with η = 0 sends each starting noise value to one fixed output. The map is an increasing staircase: noise below about -0.25 goes to the dark bump, noise above it to the bright bump. Fewer steps give a softer, less accurate staircase.
~~~

The switch point is no accident. A standard normal falls below $-0.245$ with probability 0.403, almost exactly the dark bump's 40% share. In one dimension the deterministic sampler is (approximately) the map that sends the noise's quantiles to the data's quantiles. In high dimensions the same idea explains why nearby noise vectors give similar images and why interpolating between two noise vectors gives a smooth morph.

Running the DDIM update in the other direction, from clean to noisy, is called **DDIM inversion**. It finds a noise vector that regenerates a given input. On the toy data, inverting $x_0=0.3$ with 50 steps and sampling back gave 0.330, and with 1000 steps 0.301. Real-image editing methods build on this, and the small round-trip error is why they often add extra correction steps.

## What goes wrong

The most common bug is the one from the practice question: running the DDPM update on a skipped schedule.

~~~dg-fig 07.05-naive.svg
20-step samples. Left: the DDIM formula with η = 1 (which computes the correct jump sizes) lands on both bumps, a little too narrow. Right: reusing the one-step DDPM coefficients removes too little noise per jump; the spread is about 0.95 instead of 0.51 and about 30% of samples fall outside [-1, 1].
~~~

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Grainy, noisy samples with few steps | DDPM one-step $\beta_t$ used on a skipping schedule | Use the DDIM update (any $\eta$) or recompute the jump's $\beta$ as $1-\bar\alpha_t/\bar\alpha_{t'}$
| Faint noise left in every sample | Noise added on the final step, or last step does not reach level 0 | Skip $\sigma z$ at the end; include step 0 (`abar[0] = 1`)
| Washed-out or over-saturated colours in a latent model | Clipping $\hat x_0$ to $[-1, 1]$ on unbounded latents | Disable clipping (or use thresholding designed for that model)
| Results shifted by one level, slight blur | Off-by-one between `abar[t]` and `abar[t-1]` indexing | Fix one convention: training and sampling must use the same $t$
| NaN at the first step | $\sqrt{1-\bar\alpha_{t'}-\sigma^2}$ of a negative number with $\eta>1$, or division by $\sqrt{\bar\alpha_T}$ near 0 | Keep $0\le\eta\le1$; compute in float32 or higher
| Mushy samples at 5 to 10 steps | First-order sampler too coarse for the curved paths | Better step spacing, a higher-order ODE solver, or distillation

## Going deeper

**Why DDIM keeps the right marginals.** DDIM defines the non-Markovian step

$$q_\sigma(x_{t-1}\mid x_t,x_0)=\mathcal N\Big(\sqrt{\bar\alpha_{t-1}}\,x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}},\ \sigma_t^2 I\Big).$$

Suppose $x_t\mid x_0\sim\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$. Then the fraction in the mean is a standard normal, so $x_{t-1}$ has mean $\sqrt{\bar\alpha_{t-1}}x_0$ and variance $(1-\bar\alpha_{t-1}-\sigma_t^2)+\sigma_t^2=1-\bar\alpha_{t-1}$. That is the correct marginal one level down. By induction from $T$, every marginal matches, whatever $\sigma_t$ you choose, so the $\epsilon$-trained network is valid for the whole family. Nothing in the argument needs $t-1$ to be the next integer, which is why skipping is allowed.

**DDIM is Euler's method on an ODE.** Rescale to $\bar x=x/\sqrt{\bar\alpha}$ and $s=\sqrt{(1-\bar\alpha)/\bar\alpha}$ (the noise-to-signal ratio). The $\eta=0$ update becomes

$$\bar x_{t'}=\bar x_t+(s_{t'}-s_t)\,\hat\epsilon,$$

one Euler step of $d\bar x/ds=\epsilon_\theta$, the probability-flow ODE of lesson 07.06. Check with the worked example: $\bar x_t=1/0.7071=1.414$, $s$ goes from 1 to 0.5, so $\bar x_{t'}=1.414-0.5\times0.6=1.114$ and $x_{t'}=1.114\times\sqrt{0.8}=0.997$, the same answer. This view explains the error pattern in the steps figure: Euler's method has error that shrinks roughly in proportion to step size. Higher-order solvers such as Heun or DPM-Solver reach good images in about 10 to 25 calls, and distillation methods target 1 to 4.

**Why randomness helps at many steps.** A deterministic sampler carries every small error forward to the end. A stochastic one keeps re-sampling, which pulls the distribution back towards the right marginals, so it is more forgiving of an imperfect network; but each injected noise must then be removed again, which costs steps. With few steps the second effect dominates, which is why $\eta=0$ wins on the left of the steps figure. With an imperfect trained network (unlike our exact one), larger-scale studies such as EDM (lesson 07.06) found that some added noise can help at moderate step counts.

**Learned variances.** Improved DDPM sets $\sigma_t^2=\exp(v\log\beta_t+(1-v)\log\tilde\beta_t)$ per dimension, with $v$ output by the network, which helps most when steps are skipped.

| Sampler | Calls per image | Stochastic | Typical use
| --- | --- | --- | --- |
| DDPM ancestral | 1000 | yes | reference quality, slow
| DDIM, $\eta=0$ | 20 to 100 | no | fast, invertible, reproducible
| DDIM, $0<\eta\le1$ | 50 to 250 | partly | trades speed for robustness
| Higher-order ODE solvers | 10 to 25 | no | default in modern pipelines

## Key terms

| Term | Meaning
| --- | --- |
| Sampler | Algorithm that turns starting noise into a sample using the trained network
| Network call (NFE) | One forward pass of $\epsilon_\theta$; the main cost of sampling
| Forward posterior | $q(x_{t-1}\mid x_t,x_0)$, the Gaussian each reverse step aims at once $x_0$ is known
| $\hat x_0$ | The network's current estimate of the clean data, computed from $\hat\epsilon$
| Ancestral sampling | Drawing each $x_{t-1}$ with fresh noise, as DDPM does
| $\tilde\beta_t$ | Posterior variance, the smaller of the two standard noise sizes
| DDIM | Sampler that predicts $\hat x_0$ and re-noises to any lower level; same trained model
| $\eta$ | DDIM's randomness dial: 0 deterministic, 1 matches DDPM
| DDIM inversion | Running the deterministic update from data to noise to find a latent that regenerates the input

<div class="take">
<div class="lab">Key takeaway</div>

DDPM ancestral sampling walks all $T$ posterior steps with fresh noise; DDIM reuses the same network to predict $\hat x_0$ and re-noise to any lower level, so it can skip steps and run deterministically, cutting sampling cost by 10 to 50 times.

</div>
