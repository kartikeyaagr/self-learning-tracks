---
id: "07.06"
course: "deep-learning"
module: "07"
slug: "score-matching-and-flow-matching"
title: "Score matching and flow matching"
keyIdeas: ["The **score** $\\nabla_x\\log p(x)$ is a field of arrows pointing towards higher density. The normalising constant drops out of it, and **Langevin dynamics** (score step plus Gaussian noise) turns it into a sampler.", "**Denoising score matching** learns the score of noised data by regressing $-\\epsilon/\\sigma$, so a DDPM noise predictor already is a score model: $s_\\theta(x_t,t)=-\\epsilon_\\theta(x_t,t)/\\sqrt{1-\\bar\\alpha_t}$.", "Many noise levels are essential: at small noise the score is learned only where data lives, and plain Langevin cannot move mass between far-apart modes. **Annealed Langevin** goes from large noise to small.", "In continuous time diffusion is an SDE. Its reverse needs only the score, and the deterministic **probability-flow ODE** $dx=[f-\\tfrac12 g^2\\nabla\\log p_t]\\,dt$ has the same marginals; DDIM is its Euler step.", "**Flow matching** picks a path from noise to data, such as $x_t=(1-t)x_0+t x_1$, and regresses the velocity $x_1-x_0$. The network learns the average velocity through each point, and sampling integrates that ODE.", "Straighter trajectories need fewer Euler steps. This is the main sampling advantage of flow-matching models such as SD3 and Flux; reflow straightens the paths further."]
question: "A flow-matching model trained with the straight path $x_t=(1-t)x_0+t x_1$ is sampled with 4 Euler steps and looks fine. A DDPM-trained model sampled with 4 DDIM (Euler) steps looks blurry and wrong. Both are well trained. Explain the difference in terms of the ODE each sampler integrates. What would you expect each to produce with just 1 step, and what could you change to fix the diffusion model without retraining?"
answer: "Both samplers integrate an ODE whose marginals match the data, so with many small steps both are fine; the difference is how curved the trajectories are. DDPM's probability-flow ODE, stepped uniformly in the DDPM time index, follows paths that change little for a long stretch and then bend sharply near the data end. Early on, the implied clean-image estimate $\\hat x_0$ is a blurry average of many plausible images, and the trajectory turns as that estimate sharpens. A 4-step Euler integration moves along the tangent at each point, so it overshoots the bend and ends off the data manifold, often near the blurry average. The flow-matching field is trained on straight noise-to-data pairs, so its marginal trajectories bend only gently and evenly, and a straight Euler step stays close to the true path. Euler's local error scales with curvature times step size squared, so straighter paths tolerate much larger steps. In the 1-D experiment of this lesson, with exact fields, the error at 2 steps was 0.60 for flow matching versus 1.19 for DDIM, and at 4 steps 0.22 versus 0.31; from 8 steps on they were equal. With 1 step both fail in the same way: the first velocity is $\\mathbb E[x_1]-x_0$ (or the DDIM equivalent), so every sample lands on the mean of the data, which for images is a grey blur. Fixes for the diffusion model that need no retraining: use a second-order solver (Heun, DPM-Solver), and space the steps where curvature is high (EDM's $\\sigma$ grid dense near the data). With retraining you could distil it into a few-step model, or switch to a flow-matching objective and apply reflow to straighten the trajectories further."
sources: [{"title": "Song et al.: Score-Based Generative Modeling through Stochastic Differential Equations", "url": "https://arxiv.org/abs/2011.13456", "note": "The SDE view, the reverse SDE and the probability-flow ODE (sections 3 and 4)."}, {"title": "Yang Song: Generative Modeling by Estimating Gradients of the Data Distribution", "url": "https://yang-song.net/blog/2021/score/", "note": "The author's own long-form explanation of score matching, Langevin and the SDE picture."}, {"title": "Lipman et al.: Flow Matching for Generative Modeling", "url": "https://arxiv.org/abs/2210.02747", "note": "Conditional flow matching and why the per-sample target gives the right marginal field."}, {"title": "Karras et al.: Elucidating the Design Space of Diffusion-Based Generative Models", "url": "https://arxiv.org/abs/2206.00364", "note": "EDM: the sigma-space ODE, Heun sampling and the step-spacing recipe."}]
practice: ["denoising-score-matching-loss", "conditional-flow-matching-loss", "gaussian-mixture-score", "annealed-langevin-dynamics", "karras-euler-sampler"]
tags: ["score matching", "probability-flow ODE", "flow matching"]
tools: ["PyTorch"]
---

## What you will learn

- What a **score** is, why it avoids the normalising constant, and how Langevin dynamics samples with it.
- How denoising score matching learns a score from data, and why a DDPM noise predictor already is one.
- Why many noise levels are needed (annealed Langevin).
- The reverse SDE and the deterministic probability-flow ODE, and where DDPM and DDIM sit.
- Flow matching: regressing the velocity $x_1 - x_0$ along straight paths, and why straight paths need fewer steps.

*Before you start:* the forward process and $\bar\alpha_t$ (07.03), the noise-prediction loss (07.04) and DDPM/DDIM sampling (07.05). An ODE is just "a rule for how a point moves".

Lessons 07.03 to 07.05 built diffusion as a discrete chain: add noise in 1000 steps, learn to predict it, walk back. Here we look at the same machine from two angles. **Score matching** explains *what* the noise predictor has learned. **Flow matching** drops the noise chain for a simpler recipe used by Stable Diffusion 3 and Flux.

## The idea in one picture

Take a 1-D "dataset": two bumps, 30% of the mass around $-2$ and 70% around $+2$, each with spread 0.8. For images we can never write the density $p(x)$ down, because it must integrate to 1 over millions of dimensions. What we can work with is the **score**: the derivative of the log-density, $s(x) = \frac{d}{dx}\log p(x)$. It answers one local question: *if I nudge $x$, which way does the density go up, and how steeply?* Here are its values for our mixture (computed exactly):

| x | -3 | -2 | -1 | 0 | 1 | 2 | 3
| --- | --- | --- | --- | --- | --- | --- | --- |
| density p(x) | 0.068 | 0.150 | 0.069 | 0.022 | 0.160 | 0.349 | 0.160
| score s(x) | +1.56 | 0.00 | -1.53 | +1.25 | +1.56 | 0.00 | -1.56

~~~dg-fig 07.06-score.svg
Left: the two-bump density. Middle: its score is positive left of each peak and negative right of it, so the arrows always point uphill; it is zero exactly on the peaks. Right: Langevin sampling with this score reproduces the density, while plain gradient ascent (no noise) collapses every point onto a peak.
~~~

At $x = -3$ the score is positive (move right, towards the left peak), at $-1$ negative, at the peaks zero, and in the valley at $x = 0$ it is $+1.25$, pointing towards the heavier bump.

Now start 20,000 points uniformly on $[-6, 6]$ and repeatedly move each a little along its arrow *plus a little random noise*. After 6000 small steps the histogram matches $p(x)$: 31.7% of points sit on the left bump (true 30%), mean 0.73 and standard deviation 2.03 (true 0.80 and 2.00). Without the noise every point just climbs to the nearest peak: optimisation, not sampling. That is the whole idea of score-based generation: **learn the arrows, then follow them with noise.**

## Step 1: the score ignores the normalising constant

Write any density as $p(x) = e^{-E(x)}/Z$, where $E(x)$ is an "energy" that is low where data is common and $Z$ is the number that makes everything integrate to 1. For images, $Z$ is a sum over every possible image, hopeless to compute. Take the log and differentiate:

$$s(x) = \nabla_x \log p(x) = -\nabla_x E(x) - \nabla_x \log Z = -\nabla_x E(x).$$

Here $\nabla_x$ is the gradient with respect to the input (one entry per pixel). $Z$ does not depend on $x$, so it vanishes, and a network $s_\theta(x)$ with output shaped like $x$ can represent the score with no constraint: no softmax, no integral.

Number check: for a standard Gaussian $p(x) \propto e^{-x^2/2}$, the energy is $x^2/2$ and the score is $-x$. At $x = 2$ the score is $-2$: "go left, towards the centre", and the farther out you are, the stronger the pull.

## Step 2: Langevin dynamics turns a score into samples

**Langevin dynamics** is the noisy hill-climb from the picture:

$$x_{k+1} = x_k + \frac{\eta}{2}\, s(x_k) + \sqrt{\eta}\, z_k, \qquad z_k \sim \mathcal N(0, I).$$

- $x_k$ is the current sample after $k$ steps; $\eta$ is a small step size.
- $\frac{\eta}{2}s(x_k)$ pushes uphill in density.
- $\sqrt{\eta}\,z_k$ is fresh Gaussian noise that keeps the points spread out.

For small $\eta$ and many steps, the distribution of $x_k$ converges to $p$: drift alone collapses onto modes, noise alone spreads forever, together they settle on $p$. (Larger $\eta$ adds bias; a Metropolis accept/reject test, MALA, removes it.)

<div class="ex">
<div class="lab">Worked example</div>

**one Langevin step by hand**

Target: the standard Gaussian, so $s(x) = -x$. Current point $x_k = 2.0$, step size $\eta = 0.1$, and the noise draw happens to be $z_k = 0.5$.

Drift: $\frac{\eta}{2}s(x_k) = 0.05 \times (-2.0) = -0.10$.

Noise: $\sqrt{0.1} \times 0.5 = 0.3162 \times 0.5 = 0.158$.

New point: $x_{k+1} = 2.0 - 0.10 + 0.158 = 2.058$. This draw of noise outweighed the pull, so the point moved slightly outward; on average (when $z_k = 0$) it would move to 1.90, towards the centre.

</div>

## Step 3: learning the score by denoising

We cannot regress on the true score because we do not know it. **Denoising score matching** (Vincent, 2011) gets around this. Corrupt a data point: $\tilde x = x + \sigma \epsilon$ with $\epsilon \sim \mathcal N(0, I)$. For this *one* corruption we know which way is "back to the data": the score of the Gaussian kernel $q(\tilde x \mid x)$ is

$$\nabla_{\tilde x} \log q(\tilde x \mid x) = -\frac{\tilde x - x}{\sigma^2} = -\frac{\epsilon}{\sigma}.$$

Regress the network on that target:

$$L_\text{DSM} = \mathbb E_{x,\epsilon}\Big\|\, s_\theta(\tilde x, \sigma) + \frac{\epsilon}{\sigma} \Big\|^2.$$

Each target is noisy, but the minimiser is the *average* target at each $\tilde x$, which is exactly the score of the **noised data distribution** $p_\sigma$ (the data blurred by noise $\sigma$).

<div class="ex">
<div class="lab">Worked example</div>

**a denoising score target in 1-D**

A data point $x = 2.0$ is perturbed with $\sigma = 0.5$ and $\epsilon = 0.4$, so $\tilde x = 2.0 + 0.5 \times 0.4 = 2.2$.

Target score: $-\epsilon/\sigma = -0.4/0.5 = -0.8$. The same number from the other formula: $-(\tilde x - x)/\sigma^2 = -0.2/0.25 = -0.8$.

It points back towards the data, with size growing like $1/\sigma$: with $\sigma = 0.1$ it would be $-4.0$.

</div>

**A noise predictor is a score model.** In the variance-preserving process of 07.03, $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$, the noise has size $\sqrt{1-\bar\alpha_t}$, and the same argument gives

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}.$$

The DDPM loss $\|\epsilon_\theta - \epsilon\|^2$ is denoising score matching with a particular weighting. If $\bar\alpha_t = 0.84$ and the network predicts $\hat\epsilon = 0.6$, then $\sqrt{1 - 0.84} = 0.4$ and the score is $-0.6/0.4 = -1.5$.

~~~dg-fig 07.06-dsm.svg
Left: blurring the two-bump data with more noise fills the empty valley and finally merges the bumps. Right: true scores of the blurred densities (solid) and the scores one small network learned with denoising score matching (dotted). The fit is close for σ = 0.8 and 2.5; at σ = 0.1 it is visibly off in the valley and in the tails, where almost no noisy training points land.
~~~

The network is the small MLP from the code section; its largest error over $|x| < 4$ is 0.03 at $\sigma = 2.5$, 0.12 at $\sigma = 0.8$ and 1.19 at $\sigma = 0.1$: **at small noise, the score is learned only where data lives.**

## Step 4: many noise levels, and annealed Langevin

Why train on a range of $\sigma$ rather than one tiny value?

1. **Low-density regions are unlearnable at small noise**: no training points land there. Large noise fills them.
2. **Plain Langevin cannot move mass between far-apart modes.** Even with the *exact* score, points rarely cross a wide empty valley, so the split between modes is whatever the starting points had.

~~~dg-fig 07.06-anneal.svg
Two bumps at -4 and +4 holding 20% and 80% of the mass. Plain Langevin with the exact score keeps the starting split (49% left). Annealed Langevin, which runs a few steps at each of 20 decreasing noise levels, recovers 21%.
~~~

Song and Ermon (2019) proposed **annealed Langevin dynamics**: train one network $s_\theta(x, \sigma)$ on levels $\sigma_1 > \dots > \sigma_L$, run a few Langevin steps at the largest, then hand the result to the next level down. At large $\sigma$ the bumps merge into one blob whose score "sees" that the right side is four times heavier, so proportions are fixed early and later levels only sharpen. Step sizes scale as $\eta_i \propto \sigma_i^2$. This is diffusion seen from the score side.

## Step 5: continuous time, one SDE and one ODE

Song et al. (2021) unified the views. With infinitely many noise steps the forward process becomes a **stochastic differential equation (SDE)**, a rule for how $x$ moves in a tiny time $dt$:

$$dx = f(x, t)\, dt + g(t)\, dw.$$

Here $f$ is the deterministic **drift**, $g(t)$ sets how much noise enters and $dw$ is a tiny Gaussian kick. The DDPM chain is $f = -\tfrac12\beta(t)\,x$, $g = \sqrt{\beta(t)}$: shrink a little, add a little noise.

A classical result (Anderson, 1982) says this has a **reverse-time SDE** that runs from noise back to data:

$$dx = \big[f(x,t) - g(t)^2\,\nabla_x \log p_t(x)\big]\, dt + g(t)\, d\bar w.$$

The only unknown is $\nabla_x \log p_t$, the score of the noisy marginal, exactly what the network learns; DDPM ancestral sampling discretises this SDE. A deterministic process has the *same* marginals $p_t$, the **probability-flow ODE**:

$$dx = \big[f(x,t) - \tfrac12\, g(t)^2\,\nabla_x \log p_t(x)\big]\, dt.$$

Same score, half the coefficient, no noise: each noise vector maps to exactly one image.

~~~dg-layers Three ways to use the same learned score, from most to least random.
Annealed Langevin | many noisy steps per noise level, corrects as it goes
Reverse SDE | one noisy step per level; ancestral DDPM is its discretisation
*Probability-flow ODE | deterministic; DDIM is its Euler step; any ODE solver works
~~~

**DDIM is an Euler step of this ODE.** The ODE view also gives exact log-likelihoods, a fixed noise-to-image map (for interpolation and editing by inversion) and any higher-order solver. Karras et al.'s **EDM** writes $x = x_0 + \sigma\epsilon$, where the ODE is $dx/d\sigma = (x - D_\theta(x;\sigma))/\sigma$ for a denoiser $D_\theta$; with Heun's solver and a $\sigma$ grid dense near the data it reached state-of-the-art quality in about 35 network evaluations (published result).

## Step 6: flow matching, learn the velocity directly

If sampling is "integrate an ODE from noise to data", why go through scores and $\beta$ schedules? **Flow matching** (Lipman et al., 2022; rectified flow, Liu et al., 2022) learns the ODE's velocity directly:

1. Draw noise $x_0 \sim \mathcal N(0, I)$ and a data sample $x_1$, independently.
2. Join them with the straight line $x_t = (1-t)\,x_0 + t\,x_1$, $t \in [0, 1]$, which moves at constant **velocity** $u = x_1 - x_0$.
3. Train a network $v_\theta(x_t, t)$ to predict that velocity:

$$L_\text{CFM} = \mathbb E_{t,\,x_0,\,x_1}\big\|\, v_\theta(x_t, t) - (x_1 - x_0) \big\|^2.$$

To sample, draw $x_0$ and integrate $dx/dt = v_\theta(x, t)$ from 0 to 1, e.g. Euler steps $x \leftarrow x + \Delta t\, v_\theta(x, t)$. Here $t = 0$ is noise and $t = 1$ is data, the opposite of DDPM's index; libraries disagree, so check.

<div class="ex">
<div class="lab">Worked example</div>

**one flow-matching training target, and why paths do not cross**

Pair noise $x_0 = -1.0$ with data $x_1 = 3.0$ and draw $t = 0.25$. The input is $x_t = 0.75 \times (-1.0) + 0.25 \times 3.0 = -0.75 + 0.75 = 0.0$, and the target velocity is $3.0 - (-1.0) = 4.0$. If the network outputs 3.5 there, this example contributes $(3.5 - 4.0)^2 = 0.25$ to the loss.

Following it for the remaining time $0.75$ lands at $0.0 + 0.75 \times 4.0 = 3.0$, the data point: on one straight pair, one Euler step is exact.

Now let the data be two equally likely points, $-2$ and $+2$. At $t = 0.5$ the point $x_t = 0$ is reached by the pair $(2, -2)$ with velocity $-4$ and by $(-2, 2)$ with velocity $+4$, equally likely, so the squared-error minimiser there is their average, $0$.

The network learns the **average** velocity of all pairs through each point. Training lines cross; the learned flow cannot, since it has one velocity per point.

</div>

As with denoising score matching, the per-pair (conditional) targets are noisy but their average is the correct **marginal** velocity field, whose ODE carries noise exactly onto data (Lipman et al. prove the two losses have equal gradients). The loss never reaches zero; its floor is the variance of the targets.

~~~dg-fig 07.06-paths.svg
1-D data with two equal bumps at ±2, exact fields. Left: training pairs as straight lines, which cross. Middle: flow-matching ODE trajectories never cross and bend gently. Right: diffusion probability-flow trajectories (DDIM, linear β schedule) barely move for half the sampler's steps, then bend sharply.
~~~

## Step 7: why straight paths need fewer steps

An Euler step moves along the current tangent: exact on a straight trajectory, overshooting on a curved one, with error per step growing like curvature times step size squared. In the paths figure, the flow-matching field bends gently and evenly (averaging over pairs bends it a little); the diffusion trajectories do nothing for a long time, then turn quickly, which is where few-step sampling goes wrong.

~~~dg-fig 07.06-steps.svg
Left: error of the final samples against the number of Euler steps, using exact fields on the 1-D example (lower is better; the dashed line is the error of a perfect sampler with 4000 samples). Right: a small flow-matching network trained on the two-moons data, sampled with 1, 4 and 32 Euler steps.
~~~

The left panel's numbers (1-D Wasserstein distance to the true data; finite-sample floor 0.049):

| Euler steps | 1 | 2 | 4 | 8 | 16 | 32 | 64
| --- | --- | --- | --- | --- | --- | --- | --- |
| Flow matching | 2.00 | 0.60 | 0.22 | 0.13 | 0.08 | 0.06 | 0.05
| Diffusion (DDIM) | 1.97 | 1.19 | 0.31 | 0.12 | 0.09 | 0.07 | 0.06

At 2 and 4 steps the straight-path model is clearly ahead (0.60 vs 1.19, 0.22 vs 0.31); from 8 steps on they are equal. With one step *both* fail: from $t = 0$ the velocity is $\mathbb E[x_1] - x_0$, so every sample lands on the **data mean** (here 0, in the empty valley, error about 2). That is the 1-D version of the blurry average image, and the moons panel shows it as one clump at the centre.

**Reflow** retrains on (noise, sample) pairs generated by the model itself; those pairs lie on non-crossing trajectories, so the new field is straighter, moving towards one-step generation. Flow matching also needs no $\beta$ schedule or $\bar\alpha$ bookkeeping. SD3 and Flux use rectified-flow objectives with $t$ drawn from a logit-normal distribution that emphasises the middle of the path (published choice).

## Score matching and flow matching in code

Denoising score matching on the two-bump data, then annealed Langevin:

```python
import torch, math
torch.set_num_threads(1); torch.manual_seed(0)
def sample_data(n):                                  # 0.3*N(-2, 0.8^2) + 0.7*N(2, 0.8^2)
    return torch.where(torch.rand(n, 1) < 0.3, -2.0, 2.0) + 0.8 * torch.randn(n, 1)
net = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.SiLU(),
                          torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
score = lambda x, sigma: net(torch.cat([x, sigma.log()], 1)) / sigma
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
for step in range(3000):
    x = sample_data(512)
    sigma = torch.exp(torch.empty(512, 1).uniform_(math.log(0.05), math.log(5.0)))
    eps = torch.randn_like(x)
    loss = ((sigma * score(x + sigma * eps, sigma) + eps) ** 2).mean()   # sigma^2-weighted DSM
    opt.zero_grad(); loss.backward(); opt.step()
print(round(loss.item(), 3))                         # 0.725  (never 0: targets are noisy)

with torch.no_grad():                                # annealed Langevin, largest noise first
    x = 8 * torch.rand(5000, 1) - 4
    sigmas = torch.logspace(math.log10(5.0), math.log10(0.05), 15)
    for s in sigmas:
        step = 2e-4 * (s / sigmas[-1]) ** 2          # step size proportional to sigma^2
        for _ in range(100):
            x = x + step / 2 * score(x, s.expand(5000, 1)) + step.sqrt() * torch.randn_like(x)
print(round((x < 0).float().mean().item(), 3), round(x.mean().item(), 2), round(x.std().item(), 2))
# 0.334 0.58 1.91      (true: 0.30 on the left, mean 0.80, std 2.00)
```

The $\|\sigma s + \epsilon\|^2$ form (DSM times $\sigma^2$) gives every noise level a similar scale.

Conditional flow matching on two moons (same script), then Euler sampling:

```python
from sklearn.datasets import make_moons
torch.manual_seed(0)
X, _ = make_moons(20000, noise=0.05, random_state=0)
data = torch.tensor((X - X.mean(0)) / X.std(0), dtype=torch.float32)
H = 256                                              # input (x, y, t) -> velocity (vx, vy)
net = torch.nn.Sequential(torch.nn.Linear(3, H), torch.nn.SiLU(), torch.nn.Linear(H, H), torch.nn.SiLU(),
                          torch.nn.Linear(H, H), torch.nn.SiLU(), torch.nn.Linear(H, 2))
v = lambda x, t: net(torch.cat([x, t], 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 8000)
for step in range(8000):
    x1 = data[torch.randint(len(data), (512,))]      # data  (t = 1)
    x0 = torch.randn_like(x1)                        # noise (t = 0)
    t = torch.rand(512, 1)
    loss = ((v((1 - t) * x0 + t * x1, t) - (x1 - x0)) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
print(round(loss.item(), 2))                         # 1.52  (a floor, not a failure)

@torch.no_grad()
def sample(n, steps):                                # Euler: x <- x + dt * v(x, t)
    x = torch.randn(n, 2, generator=torch.Generator().manual_seed(1))
    for i in range(steps):
        x = x + v(x, torch.full((n, 1), i / steps)) / steps
    return x
for steps in [1, 4, 32]:                             # mean distance to the nearest real point
    print(steps, round(torch.cdist(sample(2000, steps), data[:5000]).min(1).values.mean().item(), 3))
# 1 0.351    4 0.064    32 0.023     (fresh real points score 0.013)
```

## Checking the result

- **Do not judge by the loss value.** Both losses have a floor above zero (0.725 and 1.52), set by target ambiguity. Judge quality by samples.
- **Measure the sample distribution**: histograms or nearest-neighbour distances in low dimensions, FID for images (07.02). Check mode proportions: samples can be sharp but in the wrong ratio.
- **Separate model error from sampler error.** With the exact score, the same annealed sampler puts 31.7% of points on the left (mean 0.73, std 2.04), so most of the code's gap (33.4%, mean 0.58) is model error.
- **Plot error against step count.** If it flattens far above a fine-step reference, the model, not the solver, is the bottleneck.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Samples pile onto peaks, too little spread | Langevin noise term missing or scaled wrong | use $\sqrt{\eta}\,z$ with the drift $\frac{\eta}{2}s$, exactly as written
| Correct shapes, wrong proportions between modes | plain Langevin at one small noise level | anneal over many noise levels, largest first
| Loss dominated by the smallest σ, large-noise levels barely learned | unweighted DSM: targets scale as $1/\sigma$ | weight by $\sigma^2$, or predict $\epsilon$ and convert
| Samples blurry or all near the data mean | too few ODE steps on a curved field | more steps, a second-order solver, or a straighter (flow-matching / reflowed) model
| Flow-matching samples come out as noise | time convention flipped (t = 1 treated as noise) | check which end is data in both training and sampling

## Going deeper

**Explicit score matching.** Hyvärinen (2005) showed $\mathbb E_p\|s_\theta - \nabla\log p\|^2$ equals, up to a constant, $\mathbb E_p\big[\tfrac12\|s_\theta\|^2 + \operatorname{tr}\nabla_x s_\theta\big]$, which needs no true score but costs one backward pass per dimension for the trace. Denoising score matching avoids the trace, at the price of learning the score of noised data.

**Tweedie's formula.** For $\tilde x = x + \sigma\epsilon$, the best denoiser is $\mathbb E[x \mid \tilde x] = \tilde x + \sigma^2\,\nabla \log p_\sigma(\tilde x)$. So denoiser, noise predictor and score model are one object: $D = \tilde x + \sigma^2 s$, $\hat\epsilon = -\sigma s$.

**Velocity and score are interchangeable.** For the straight path with Gaussian noise, $x_t = (1-t)x_0 + t x_1$, the marginal velocity and the score of $p_t$ satisfy $v(x,t) = \big(x + (1-t)\,s(x,t)\big)/t$ (checked numerically on our two-bump example). Any Gaussian path $x_t = \alpha_t x_1 + \sigma_t x_0$ gives a velocity that is a linear combination of $x$ and the score, so diffusion and flow matching differ in the *path* and the loss's *time weighting*, not in kind.

**Couplings.** Independent pairing makes training paths cross and bends the field; minibatch optimal-transport couplings and reflow reduce crossing. And since the ODE is a continuous normalising flow, $\frac{d}{dt}\log p_t(x_t) = -\nabla\cdot v$ along a trajectory gives exact likelihoods.

## Key terms

| Term | Meaning
| --- | --- |
| Score | $\nabla_x\log p(x)$: which way, and how steeply, log-density increases
| Langevin dynamics | Follow the score plus Gaussian noise: $x \leftarrow x + \frac{\eta}{2}s + \sqrt\eta\,z$
| Denoising score matching | Learns the score of noised data by regressing $-\epsilon/\sigma$
| Annealed Langevin | Langevin over decreasing noise levels, largest first
| Reverse SDE | Noisy process from noise to data that needs only the score of each noisy marginal
| Probability-flow ODE | Deterministic process with the same marginals; DDIM is its Euler discretisation
| Flow matching | Regress a velocity on per-pair targets such as $x_1 - x_0$; sample by integrating the ODE
| Marginal velocity | Average of all per-pair velocities through a point; what the network learns

<div class="take">
<div class="lab">Key takeaway</div>

a noise predictor is a score model; the score defines both a reverse SDE and a deterministic probability-flow ODE with the same marginals; and flow matching learns an ODE velocity directly by regressing $x_1 - x_0$ along straight noise-to-data paths, whose gentler curvature lets an Euler sampler use fewer steps.

</div>
