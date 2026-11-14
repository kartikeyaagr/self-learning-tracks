---
id: "07.03"
course: "deep-learning"
module: "07"
slug: "diffusion-the-forward-process"
title: "Diffusion: the forward process"
keyIdeas: ["The **forward process** adds Gaussian noise in $T$ small steps, $x_t=\\sqrt{1-\\beta_t}\\,x_{t-1}+\\sqrt{\\beta_t}\\,\\epsilon_t$. It has no learned parameters; only the reverse process is a neural network.", "The $\\sqrt{1-\\beta_t}$ shrink keeps unit-variance data at variance 1 ('variance preserving'), so the end point is close to $\\mathcal N(0, I)$ whatever the image. Without it the variance grows to about 11 on the DDPM schedule.", "Because independent Gaussians merge into one, you can jump to any step directly: $x_t=\\sqrt{\\bar\\alpha_t}\\,x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$ with $\\bar\\alpha_t=\\prod_{s\\le t}(1-\\beta_s)$. This makes training examples cheap.", "Read any schedule as a signal-to-noise ratio $\\mathrm{SNR}(t)=\\bar\\alpha_t/(1-\\bar\\alpha_t)$. The linear schedule reaches SNR 1 near step 259; the cosine schedule holds out until about step 496 and wastes fewer steps on pure noise.", "If $\\bar\\alpha_T$ is not essentially zero (Stable Diffusion's ends near 0.0047), training at $t=T$ leaks the image's brightness while sampling never does; zero-terminal-SNR rescaling, paired with v-prediction, fixes it."]
question: "A colleague removes the $\\sqrt{1-\\beta_t}$ factor and uses $x_t=x_{t-1}+\\sqrt{\\beta_t}\\,\\epsilon_t$ with the same linear $\\beta$ schedule ($10^{-4}$ to $0.02$, $T=1000$), still starting sampling from $\\mathcal N(0, I)$. Using the Gaussian addition rule, what is the distribution of $x_T$ given $x_0$, and why does sampling now go wrong? What would you change to make a no-shrink process work?"
answer: "Without the shrink, each step leaves the signal untouched and adds independent noise of variance $\\beta_t$. By the adding rule all those noise terms merge into one Gaussian whose variance is the sum, so $x_T = x_0 + \\sqrt{\\sum_t\\beta_t}\\,\\epsilon$. For the linear schedule $\\sum_t\\beta_t = 1000\\cdot(10^{-4}+0.02)/2 = 10.05$, so $x_T\\sim\\mathcal N(x_0,\\ 10.05\\,I)$. Two things are now wrong. First, the signal is never removed: $x_T$ still contains $x_0$ at full strength, and the SNR at $T$ is about (data variance)/10 per pixel, around 0.1 for unit-scale data, not zero. Second, the total variance is about 11 (1 from unit-variance data plus 10.05 from noise; a simulation gives 11.1), not 1. Starting the reverse chain from $\\mathcal N(0, I)$ therefore feeds the network inputs it never saw at $t=T$: the wrong scale and no image component, so the first denoising steps are extrapolations and the samples are garbage. The fix is either to restore the shrink (the variance-preserving process, which ends near $\\mathcal N(0, I)$ with $\\bar\\alpha_T\\approx 4\\times10^{-5}$), or to embrace the 'variance exploding' family deliberately: let the noise level $\\sigma_t$ grow to a value much larger than the data scale (tens of times larger) so the image becomes negligible, and start sampling from $\\mathcal N(0, \\sigma_\\text{max}^2 I)$, as score-SDE and EDM models do. Either way, check the endpoint by noising real data to step $T$ and confirming mean 0, the intended variance, and near-zero correlation with $x_0$."
sources: [{"title": "Ho, Jain and Abbeel: Denoising Diffusion Probabilistic Models", "url": "https://arxiv.org/abs/2006.11239", "note": "Section 2 defines the forward process and the closed-form marginal used everywhere."}, {"title": "Nichol and Dhariwal: Improved Denoising Diffusion Probabilistic Models", "url": "https://arxiv.org/abs/2102.09672", "note": "Section 3.2 introduces the cosine schedule and explains why linear wastes steps."}, {"title": "Lin et al.: Common Diffusion Noise Schedules and Sample Steps are Flawed", "url": "https://arxiv.org/abs/2305.08891", "note": "The zero-terminal-SNR problem, the rescaling fix and why it needs v-prediction."}, {"title": "Lilian Weng: What are Diffusion Models?", "url": "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/", "note": "A careful derivation of every forward-process identity in one place."}]
practice: ["forward-diffusion-sample", "linear-beta-schedule", "cosine-noise-schedule", "rescale-to-zero-terminal-snr"]
tags: ["diffusion", "noise schedule", "SNR"]
tools: ["NumPy"]
---

## What you will learn

- What the **forward process** of a diffusion model is: a fixed recipe that turns any image into pure noise in many small steps.
- Why each step shrinks the image by $\sqrt{1-\beta_t}$ before adding noise, and what goes wrong without the shrink.
- How two Gaussian facts give a one-line formula for jumping from the clean image to any step $t$.
- How to read and compare noise schedules (linear, cosine, Stable Diffusion's) through one number, the signal-to-noise ratio.
- Why the last step must be pure noise, and how the zero-terminal-SNR rescaling fixes schedules that are not.

*Before you start:* you need the normal distribution, variance, and the rule that variances of independent variables add. Lesson 07.01 (VAEs) introduced the idea of mapping data to a simple Gaussian; lesson 07.02 (GANs) showed a generator trained by an adversary. No neural networks appear in this lesson: the forward process has no parameters at all.

## From GANs to diffusion

Lesson 07.02 ended with the main weakness of GANs: two networks chasing each other, with training that can oscillate or collapse. Diffusion models drop the adversary and turn generation into many small regression problems ("here is a noisy image, predict the noise"), which train as calmly as any supervised model.

The trick is to split the job in two halves:

- The **forward process** destroys data. It takes a clean image $x_0$ and adds a little Gaussian noise, then a bit more, for $T$ steps (the original DDPM paper uses $T = 1000$), until $x_T$ is indistinguishable from random noise. It is a fixed formula, with nothing to learn.
- The **reverse process** is a neural network that learns to undo one step at a time (lesson 07.04). To generate, you draw fresh noise and run the learned reverse chain back to an image (lesson 07.05).

~~~dg-flow The forward chain is fixed and cheap; only the reverse chain is learned.
[Forward, fixed] x0 data -(add noise)-> x1 -> ... -> *xT about N(0, I)
[Reverse, learned] xT noise -(denoise)-> ... -> x1 -> *x0 sample
~~~

This lesson is about the forward half. It is the easy part, but every formula in training and sampling is built from it.

## The idea in one picture

Take 170 points shaped like two interlocking moons (a toy 2-D dataset, standardised to mean 0 and variance 1) and push every point through the forward process.

~~~dg-fig 07.03-moons.svg
Two moons dissolving under the linear schedule. By step 150 the shape is a fuzzy outline; by step 1000 it is a round standard Gaussian blob that no longer depends on the data.
~~~

The same thing happens to a real image. Here is an 8x8 handwritten zero from scikit-learn's digits set, scaled to $[-1, 1]$ and noised with one fixed noise draw at increasing steps:

~~~dg-fig 07.03-digit.svg
The digit fades as the signal weight under each panel falls: clearly a zero at step 100, a faint ring at 250, and pure noise by 500 and beyond.
~~~

Read the two figures together with this table (linear schedule, $T = 1000$):

| Step $t$ | Signal weight $\sqrt{\bar\alpha_t}$ | Noise weight $\sqrt{1-\bar\alpha_t}$ | What you can still see
| --- | --- | --- | --- |
| 0 | 1.00 | 0.00 | the clean data
| 100 | 0.95 | 0.32 | everything, slightly grainy
| 250 | 0.72 | 0.69 | rough shape only
| 500 | 0.28 | 0.96 | almost nothing
| 1000 | 0.006 | 1.00 | nothing: pure noise

Every noised sample is a **blend** of the clean image and standard Gaussian noise. The two weights come from the **noise schedule**, the list of small numbers $\beta_1, \dots, \beta_T$ saying how much noise each step adds. Their squares always sum to 1 ($0.72^2 + 0.69^2 \approx 1$); the rest of the lesson explains why.

## Step 1: one small noising step

Each forward step does two things to the current image $x_{t-1}$: it shrinks it slightly towards zero, then adds a little fresh Gaussian noise.

$$x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_t, \qquad \epsilon_t\sim\mathcal N(0, I).$$

The same thing written as a probability distribution, the form you will see in papers, is

$$q(x_t\mid x_{t-1})=\mathcal N\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big).$$

What each symbol means:

- $x_t$ is the image after $t$ steps, same shape as the image.
- $\beta_t$ is the **noise level of step $t$**, a small number between 0 and 1. DDPM uses a **linear schedule**: $\beta_t$ rises evenly from $10^{-4}$ at $t = 1$ to $0.02$ at $t = 1000$.
- $\epsilon_t$ is fresh standard Gaussian noise, drawn independently for every pixel at every step ($I$, the identity matrix, says the pixels' noises are independent).
- $q$ is the usual letter for the forward process (the learned reverse process gets $p_\theta$).

**Why the square-root shrink?** Suppose a pixel currently has variance 1. After the step its variance is $(1-\beta_t)\cdot 1 + \beta_t = 1$: the signal loses exactly as much variance as the noise adds. Number check with $\beta_t = 0.02$: the signal keeps $0.98$ of its variance and the noise brings $0.02$, total $1.00$. This property is why the process is called **variance preserving** (VP).

Without the shrink, the image would never fade and the variance would grow by $\beta_t$ at every step. Over the linear schedule the $\beta$'s add up to $\sum_t \beta_t = 10.05$, so a unit-variance input would end with variance about 11. The left panel below shows exactly this in a simulation of 20,000 values.

~~~dg-fig 07.03-variance.svg
Left: without the shrink the variance climbs to about 11; with it, the variance settles at 1, even when the data start with variance 9. Right: 300 small steps and one closed-form jump (Step 3) give the same distribution.
~~~

The dashed green curve shows that the end point is about $\mathcal N(0, 1)$ *whatever* the data's scale, but large-scale data take longer to forget it. That is why images are rescaled to $[-1, 1]$ before diffusion: schedules are designed for roughly unit-scale data.

## Step 2: two facts about Gaussians

To get from one step to many, you need only two facts:

- **Scaling.** If $\epsilon\sim\mathcal N(0, 1)$, then $a\epsilon\sim\mathcal N(0, a^2)$: multiplying by $a$ multiplies the variance by $a^2$.
- **Adding.** If $\epsilon_1$ and $\epsilon_2$ are independent standard Gaussians, then $a\epsilon_1+b\epsilon_2$ has exactly the same distribution as $\sqrt{a^2+b^2}\,\epsilon$ for a single standard Gaussian $\epsilon$. Independent Gaussians merge into one Gaussian whose variance is the sum of the variances.

So any pile of independent noise terms can be replaced by *one* with the right variance.

## Step 3: jump to any step in one go

Write $\alpha_t = 1 - \beta_t$ (how much signal variance step $t$ keeps). Unroll two steps:

$$x_t=\sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2}+\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-1}+\sqrt{1-\alpha_t}\,\epsilon_t.$$

The two noise terms are independent, so they merge into one with variance $\alpha_t-\alpha_t\alpha_{t-1}+1-\alpha_t=1-\alpha_t\alpha_{t-1}$. After any number of steps, the signal is scaled by the square root of the product of the $\alpha$'s and the noise variance is one minus that product. Define the **cumulative product**

$$\bar\alpha_t=\prod_{s=1}^{t}\alpha_s=(1-\beta_1)(1-\beta_2)\cdots(1-\beta_t).$$

Then for every $t$:

$$q(x_t\mid x_0)=\mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big),\qquad x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon.$$

Here $\bar\alpha_t$ ("alpha-bar") is the fraction of the original signal's variance that survives to step $t$; it starts near 1 and falls towards 0. The right panel of the variance figure checks this: running 300 tiny steps and making one closed-form jump to $t = 300$ give matching histograms (both have mean 0.01 and variance close to 1 in the simulation).

This closed form makes diffusion training cheap: for a training example at step 700 you do not run 700 steps, you draw one $\epsilon$ and mix. It also explains the table in the picture section: its weights are $\sqrt{\bar\alpha_t}$ and $\sqrt{1-\bar\alpha_t}$.

<div class="ex">
<div class="lab">Worked example</div>

**two steps by hand, then one jump**

Take a tiny schedule $\beta_1 = 0.1$, $\beta_2 = 0.2$, a pixel with clean value $x_0 = 0.8$, and noise draws $\epsilon_1 = 0.5$, $\epsilon_2 = -1.0$.

Step 1: $x_1 = \sqrt{0.9}\cdot 0.8 + \sqrt{0.1}\cdot 0.5 = 0.9487\cdot 0.8 + 0.3162\cdot 0.5 = 0.7589 + 0.1581 = 0.9170$.

Step 2: $x_2 = \sqrt{0.8}\cdot 0.9170 + \sqrt{0.2}\cdot(-1.0) = 0.8944\cdot 0.9170 - 0.4472 = 0.8202 - 0.4472 = 0.3730$.

Closed form: $\bar\alpha_2 = 0.9\cdot 0.8 = 0.72$, so the signal part is $\sqrt{0.72}\cdot 0.8 = 0.8485\cdot 0.8 = 0.6788$, and the noise part must be $0.3730 - 0.6788 = -0.3058$.

Where does $-0.3058$ come from? The step-1 noise survives step 2's shrink: $\sqrt{0.8}\cdot\sqrt{0.1}\cdot 0.5 = 0.2828\cdot 0.5 = 0.1414$; add step 2's noise $-0.4472$ to get $-0.3058$. Its variance is $0.8\cdot 0.1 + 0.2 = 0.28 = 1 - 0.72$, exactly as the closed form says.

So the same $x_2$ is $\sqrt{0.72}\,x_0 + \sqrt{0.28}\,\epsilon$ with a single draw $\epsilon = -0.3058/0.5292 = -0.578$. Two steps with two noises, or one jump with one noise: same distribution.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**what the network sees at step 250**

Linear schedule, $t = 250$: $\bar\alpha_t = 0.524$, so $\sqrt{\bar\alpha_t} = 0.72$ and $\sqrt{1-\bar\alpha_t} = 0.69$.

A pixel with clean value 0.8 and noise draw $\epsilon = -1.2$ becomes $0.72\cdot 0.8 + 0.69\cdot(-1.2) = 0.576 - 0.828 = -0.25$.

Signal and noise carry about equal power here, which matches the digit figure: the shape is visible, fine detail is gone.

</div>

## Step 4: read a schedule through its signal-to-noise ratio

Papers describe schedules by $\beta_t$, $\bar\alpha_t$, a noise level $\sigma_t$ or "log-SNR". All are handles on one quantity, the **signal-to-noise ratio**

$$\mathrm{SNR}(t)=\frac{\bar\alpha_t}{1-\bar\alpha_t},$$

the power of the surviving signal divided by the power of the noise at step $t$. SNR above 1 means the image dominates; below 1, the noise dominates. Two schedules with the same SNR curve train essentially the same model, however they are written down.

| Step t | Linear: $\bar\alpha_t$ | Linear: SNR | Cosine: $\bar\alpha_t$ | Cosine: SNR
| --- | --- | --- | --- | --- |
| 100 | 0.897 | 8.7 | 0.972 | 35
| 250 | 0.524 | 1.1 | 0.847 | 5.5
| 500 | 0.079 | 0.085 | 0.494 | 0.98
| 750 | 0.0034 | 0.0034 | 0.144 | 0.17
| 1000 | 0.00004 | 0.00004 | about 0 | about 0

The linear schedule reaches SNR 1 around step 259 and is down to 0.085 by step 500. For most of the second half the input is almost pure noise, so many training steps teach the network very little. Nichol and Dhariwal's **cosine schedule** fixes this by defining $\bar\alpha_t$ directly:

$$\bar\alpha_t=\frac{f(t)}{f(0)},\qquad f(t)=\cos^2\!\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big),\quad s=0.008,$$

and then clipping the implied $\beta_t = 1 - \bar\alpha_t/\bar\alpha_{t-1}$ at 0.999 so the last steps stay well behaved. The small offset $s$ keeps $\beta_t$ from being vanishingly small near $t = 0$.

~~~dg-fig 07.03-weights.svg
Signal and noise weights for both schedules. Linear (solid) crosses equal power at t of about 259; cosine (dashed) holds on to the signal until about step 496 and fades almost linearly.
~~~

On a log scale the difference is clearest:

~~~dg-fig 07.03-snr.svg
Log SNR per step. Linear drops fast early and then slowly; cosine falls evenly across the middle and plunges to zero at the end; Stable Diffusion's scaled-linear schedule never reaches pure noise.
~~~

The right schedule depends on resolution. At 32x32 a little noise already hides the image, so the gentle cosine curve helps. At high resolution, averaging a $k\times k$ block of neighbouring pixels cuts the noise variance by $k^2$ but leaves a smooth signal untouched, so the effective SNR is far higher than the per-pixel number; large-image models shift their schedules towards more noise.

## Step 5: the last step must really be pure noise

Generation starts by drawing $x_T\sim\mathcal N(0, I)$. That is only consistent with training if the forward process really ends there, which needs $\bar\alpha_T = 0$ (SNR exactly zero). Look at the end of the SNR plot. Stable Diffusion's **scaled-linear** schedule ($\beta$ spaced evenly in square root from 0.00085 to 0.012) ends at $\bar\alpha_T\approx 0.0047$, so $\sqrt{\bar\alpha_T}\approx 0.068$: about 7% of the image is still mixed into the "pure noise" at the last training step.

Negligible per pixel, but the network sees all pixels at once. Averaging over $64\times 64 = 4096$ pixels shrinks the noise's mean to a standard deviation of $1/64\approx 0.016$, while the image's mean brightness survives at weight 0.068. The simulation below noises 300 synthetic 64x64 images of different brightness to step $T$.

~~~dg-fig 07.03-leak.svg
Mean of x_T against the clean image's brightness. Under the scaled-linear schedule brightness still shows through (slope 0.069, correlation 0.92); after zero-terminal-SNR rescaling the correlation is about zero.
~~~

In training the network learns to read overall brightness from $x_T$; at sampling time $x_T$ is real noise with mean near 0, so it always "reads" medium brightness. Lin et al. (2023) showed that this is why such models struggle to produce very dark or very bright images. Their fix rescales $\sqrt{\bar\alpha_t}$ linearly so the last value is exactly 0 while the first stays unchanged. With SNR zero at $t = T$, predicting the noise $\epsilon$ becomes trivial ($x_T = \epsilon$) and useless, so the rescaling has to be paired with v-prediction (lesson 07.04).

## The forward process in code

The full forward process is a dozen lines of NumPy. In PyTorch it is the same with `torch` in place of `np`, and `t` becomes a tensor of per-example steps.

```python
import numpy as np

T = 1000
betas = np.linspace(1e-4, 0.02, T)          # DDPM linear schedule, betas[t-1] is beta_t
abar = np.cumprod(1.0 - betas)              # abar[t-1] is alpha-bar at step t

def q_sample(x0, t, rng):                   # t in 1..T, returns x_t and the noise used
    eps = rng.standard_normal(x0.shape)
    return np.sqrt(abar[t-1]) * x0 + np.sqrt(1 - abar[t-1]) * eps, eps

rng = np.random.default_rng(0)
x0 = rng.uniform(-1, 1, size=(10_000, 64))   # stand-in "images" already scaled to [-1, 1]
for t in [1, 250, 500, 1000]:
    xt, eps = q_sample(x0, t, rng)
    corr = np.corrcoef(x0.ravel(), xt.ravel())[0, 1]
    print(t, round(abar[t-1], 4), round(xt.var(), 3), round(corr, 3))
# 1    0.9999  0.334  1.0
# 250  0.5241  0.651  0.519
# 500  0.0786  0.947  0.167
# 1000 0.0     1.0    0.002

def rescale_zero_terminal_snr(abar):               # Lin et al. 2023, Algorithm 1
    s = np.sqrt(abar)
    s0, sT = s[0], s[-1]
    s = (s - sT) * s0 / (s0 - sT)                   # shift so last = 0, rescale so first unchanged
    return s ** 2

sd = np.cumprod(1 - np.linspace(0.00085**0.5, 0.012**0.5, T) ** 2)   # Stable Diffusion
print(sd[-1].round(4), rescale_zero_terminal_snr(sd)[[0, -1]])   # 0.0047 [0.99915 0.]
```

Two details are worth reading. Uniform data on $[-1, 1]$ has variance $1/3$, so the variance of $x_t$ climbs from 0.334 to 1: "variance preserving" is exact only for unit-variance data. And the correlation between $x_0$ and $x_t$ (0.52 at step 250, 0.002 at step 1000) measures how much image is left.

## Checking your forward process

Forward-process bugs are silent: the model still trains, just on the wrong task. Run these checks on any new implementation or schedule.

- **Endpoints.** Print $\bar\alpha_1$ and $\bar\alpha_T$. The first should be close to 1 (0.9999 for linear), the last essentially 0. If $\bar\alpha_T$ is above about $10^{-3}$, decide consciously whether to rescale.
- **Monotone.** $\bar\alpha_t$ must strictly decrease; any $\beta_t \le 0$ or $\ge 1$ is a bug.
- **Statistics at $T$.** Noise a batch of real images to step $T$: the per-pixel mean should be about 0, the variance about 1, and the correlation with $x_0$ about 0.
- **Look at it.** Plot one image at several steps. Recognisable at the last step, or gone after a few percent of steps, means the schedule does not fit your data.
- **Off-by-one.** Decide once whether $t$ runs 0 to $T-1$ or 1 to $T$, and index `abar` the same way in training and sampling.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Data left in [0, 255] or [0, 1] | images still visible at $t = T$; samples washed out | scale to $[-1, 1]$ before noising
| Dropping the $\sqrt{1-\beta_t}$ shrink | variance grows to about 11; sampling from $\mathcal N(0, I)$ gives garbage | keep the shrink, or treat it as a variance-exploding model and sample from a wide Gaussian
| Using $\bar\alpha_t$ where $\sqrt{\bar\alpha_t}$ is needed | signal fades far too fast; model barely learns | weights are square roots; their squares sum to 1
| Same noise $\epsilon$ for every pixel or every example | network can cheat; poor samples | draw independent noise with the full image shape
| Off-by-one between $t$ and array index | small but systematic quality loss | test $\bar\alpha$ lookups at $t = 1$ and $t = T$
| Non-zero terminal SNR | cannot generate very dark or very bright images | zero-terminal-SNR rescaling plus v-prediction
| Linear schedule at low resolution | half the training steps see near-pure noise | cosine schedule

## Going deeper

**The reverse of one step is also Gaussian, if you know $x_0$.** By Bayes' rule, combining $q(x_t\mid x_{t-1})$ with $q(x_{t-1}\mid x_0)$ gives

$$q(x_{t-1}\mid x_t, x_0)=\mathcal N\big(\tilde\mu_t,\ \tilde\beta_t I\big),\quad \tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t,\quad \tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t.$$

The reverse network imitates this posterior: not knowing $x_0$, it predicts it (or the noise, or $v$) from $x_t$. Lesson 07.04 builds the loss from it.

**Why many tiny steps.** The true reverse $q(x_{t-1}\mid x_t)$, without $x_0$, is in general complicated, but when $\beta_t$ is small it is very nearly Gaussian, so each reverse step can be modelled as a Gaussian.

**Continuous time.** As $T\to\infty$ the chain becomes the stochastic differential equation $dx = -\tfrac12\beta(t)\,x\,dt+\sqrt{\beta(t)}\,dW$ (the VP SDE of Song et al., 2021), the view lesson 07.06 uses.

**Variance exploding.** The other family only adds noise, $x_t = x_0 + \sigma_t\epsilon$, with $\sigma_t$ growing far beyond the data scale (score-SDE "VE" and EDM models). Dividing by $\sqrt{1+\sigma_t^2}$ turns it into a VP process with $\mathrm{SNR}=1/\sigma_t^2$, another reason SNR is the common language.

**Only the endpoints and the weighting really matter.** Kingma et al. (Variational Diffusion Models) showed that in continuous time the variational bound depends on the schedule only through its SNR at the two ends; the shape in between changes the variance of the loss estimate, not its value. In practice the weighting of noise levels still matters a lot for sample quality.

**Latent diffusion.** Stable Diffusion runs this exact process on an autoencoder's latent code (lesson 07.01) instead of pixels; the latents are first multiplied by a fixed factor so their variance is close to 1.

## Key terms

| Term | Meaning
| --- | --- |
| Forward process | The fixed chain that adds Gaussian noise to data step by step until only noise remains
| Reverse process | The learned network chain that removes noise one step at a time to generate data
| $\beta_t$ (noise schedule) | The amount of noise variance added at step $t$; the list of all of them is the schedule
| $\bar\alpha_t$ | Product of $(1-\beta_s)$ up to $t$: the fraction of signal variance left at step $t$
| Closed-form marginal | $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$: jump from clean data to any step in one draw
| Variance preserving | The $\sqrt{1-\beta_t}$ shrink that keeps unit-variance data at variance 1 at every step
| SNR | $\bar\alpha_t/(1-\bar\alpha_t)$, signal power over noise power; the common language for schedules
| Zero terminal SNR | Requiring $\bar\alpha_T = 0$ so training's last step matches sampling's pure-noise start

<div class="take">
<div class="lab">Key takeaway</div>

the forward process is a fixed, parameter-free chain that blends data with Gaussian noise, $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$, keeping the variance at 1; compare schedules by their SNR curves, scale data to unit size first, and make sure the last step is truly pure noise.

</div>
