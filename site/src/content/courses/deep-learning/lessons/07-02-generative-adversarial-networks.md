---
id: "07.02"
course: "deep-learning"
module: "07"
slug: "generative-adversarial-networks"
title: "Generative adversarial networks"
keyIdeas: ["A GAN pits a **generator** $G(z)$, which turns noise into samples, against a **discriminator** $D(x)$, a real-versus-fake classifier, in the game $\\min_G\\max_D\\ \\mathbb E[\\log D(x)]+\\mathbb E[\\log(1-D(G(z)))]$. No likelihood is ever computed; $G$ learns only from $D$'s gradient.", "For a fixed generator the best discriminator is $D^*=p_\\text{data}/(p_\\text{data}+p_g)$, and against it the generator minimises $2\\,\\mathrm{JSD}(p_\\text{data}\\|p_g)-\\log 4$, which is smallest exactly when $p_g=p_\\text{data}$ and $D^*=\\tfrac12$ everywhere.", "Use the non-saturating generator loss $-\\log D(G(z))$: the minimax loss $\\log(1-D)$ has logit gradient $D$, nearly zero just when the generator is bad (0.01 versus 0.99 at $D=0.01$).", "GAN losses hover instead of falling, so judge training by samples from fixed noise, by $D$'s outputs on real and fake data, and by sample statistics; mode collapse, oscillation and an all-winning discriminator are the classic failures.", "Wasserstein critics with a gradient penalty, spectral normalisation and R1 penalties keep the discriminator smooth so it still gives useful gradient; FID, the Fréchet distance between Inception-feature Gaussians, scores both quality and diversity."]
question: "Halfway through training, your GAN's discriminator loss drops to almost zero and stays there, and generated samples stop improving. Explain what the generator's gradient looks like in this state under the minimax loss and under the non-saturating loss, and name two changes that would restore a useful signal. How would you confirm from monitoring, rather than from the losses alone, that the fix worked?"
answer: "A discriminator loss near zero means $D$ outputs close to 1 on real data and close to 0 on fakes, so the fakes' logits are very negative. Under the minimax generator loss $\\log(1-D)$ the gradient with respect to the logit is $\\sigma(l)=D\\approx 0$, so the generator receives essentially nothing: the loss has saturated. The non-saturating loss $-\\log D$ gives a logit gradient of $1-D\\approx 1$, which fixes that factor, but the signal the generator actually uses is the chain rule through $\\partial D/\\partial x$. When real and fake samples lie on separate thin manifolds and $D$ is a sharp classifier, $D$ is flat almost everywhere except at a narrow boundary, so $\\partial D/\\partial x$ near the fakes can be tiny or point in unhelpful directions; the Jensen-Shannon divergence is stuck near $\\log 2$ and gives no direction. Restoring signal means making $D$ smooth or weaker: (1) add a gradient penalty (WGAN-GP on interpolates between real and fake with $\\lambda=10$, or R1 on real data), or spectral normalisation, so $D$ stays Lipschitz and its gradient points toward the data; switching to a Wasserstein critic also keeps the loss informative when supports do not overlap; (2) weaken or slow $D$: lower its learning rate or update it less often, augment its inputs (differentiable augmentation of both real and fake), or add instance noise. To confirm the fix, do not rely on loss values, which only measure each player against a moving opponent. Instead track mean $D(\\text{real})$ and $D(\\text{fake})$, which should move back inside (0, 1) rather than pinned at 1 and 0; save sample grids from a fixed set of noise vectors to see visual improvement; and compute FID on a fixed number of samples (for example 50k, with the same code each time), which should start falling again and also reveals whether diversity (mode coverage) recovers."
sources: [{"title": "Goodfellow et al.: Generative Adversarial Nets", "url": "https://arxiv.org/abs/1406.2661", "note": "The original paper: the minimax value, the optimal discriminator and the JSD result (section 4)."}, {"title": "Goodfellow: NIPS 2016 Tutorial on GANs", "url": "https://arxiv.org/abs/1701.00160", "note": "Why the non-saturating loss, mode collapse, and practical tips from the inventor."}, {"title": "Gulrajani et al.: Improved Training of Wasserstein GANs", "url": "https://arxiv.org/abs/1704.00028", "note": "WGAN-GP: why weight clipping fails and how the gradient penalty works."}, {"title": "Heusel et al.: GANs Trained by a Two Time-Scale Update Rule", "url": "https://arxiv.org/abs/1706.08500", "note": "Introduces both TTUR and the FID metric."}]
practice: ["gan-training-step", "frechet-distance-between-feature-sets"]
tags: ["GAN", "minimax", "FID"]
tools: ["PyTorch"]
---

## What you will learn

- How a generator and a discriminator play a game that teaches one network to make realistic samples, with no likelihood.
- What the ideal game optimises: a closed-form best discriminator and the Jensen-Shannon divergence.
- Why the textbook generator loss stalls, and why everyone uses the **non-saturating** loss.
- How to train a small GAN in PyTorch and read its (non-falling) losses.
- The classic failures and the fixes that stuck: Wasserstein critics, gradient penalties, spectral normalisation.
- How GANs are scored without a likelihood, using FID.

*Before you start:* you need binary cross-entropy and the sigmoid (lesson 02.03 of the ML course), backpropagation and Adam (modules 01-02 of this course), and the VAE from lesson 07.01, which is the model GANs were designed to improve on.

## From VAEs to a game

Lesson 07.01 trained a VAE through an explicit decoder distribution $p_\theta(x \mid z)$, and the usual Gaussian choice is what makes its samples blurry: when unsure, the average of plausible images scores best under squared error.

Goodfellow et al. (2014) took another route: train a second network to *judge* samples, and let the first learn whatever fools the judge. That is a **generative adversarial network (GAN)**. It never computes $p(x)$, which makes it an **implicit** generative model: you can sample from it, but not ask how likely an image is.

## The idea in one picture

Strip the problem down to one number per sample. Real data are values near 3 (a bell curve with mean 3 and standard deviation 0.6). A half-trained generator produces values near 1 (mean 1, standard deviation 0.8). A discriminator looks at a single value $x$ and outputs the probability that it came from the real data.

Where should it say "real"? Where real samples are common and fakes are rare. The best possible answer at each point is simply the share of real density among all density there:

| $x$ | 1 | 2 | 2.5 | 3 | 4
| --- | --- | --- | --- | --- | --- |
| real density $p_\text{data}(x)$ | 0.003 | 0.166 | 0.470 | 0.665 | 0.166
| generator density $p_g(x)$ | 0.499 | 0.228 | 0.086 | 0.022 | 0.000
| best discriminator $D^*(x)$ | 0.005 | 0.42 | 0.85 | 0.97 | 1.00

~~~dg-fig 07.02-dstar.svg
The optimal discriminator (dashed) is near 0 where only fakes live, near 1 where only real data lives, and crosses 0.5 where the two densities are equal. Its slope tells the generator which way to move its samples.
~~~

From the generator's side: its samples sit around $x = 1$, where $D^*$ is almost 0. But $D^*$ rises to the right, so nudging samples rightwards makes them look more real. That slope, $\partial D / \partial x$, is the only thing the generator learns from. Repeat (the discriminator re-fits, the generator moves) and the generator's bump slides onto the real one; when they match, the best discriminator outputs 0.5 everywhere, a coin flip.

- The **generator** $G$ turns random **noise** $z$ (say 100 standard-normal numbers) into a sample $G(z)$.
- The **discriminator** $D$ is a classifier that outputs $D(x) \in (0, 1)$, its belief that $x$ is real.

## Step 1: write the game down

The discriminator is an ordinary binary classifier trained with cross-entropy: label 1 for real data, label 0 for generated samples. Its objective, which it wants to *maximise*, is the log-probability it assigns to the correct labels:

$$V(D, G) = \mathbb{E}_{x \sim p_\text{data}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p(z)}\big[\log\big(1 - D(G(z))\big)\big].$$

The first term rewards saying "real" on real data. The second rewards saying "fake" on generated data. The generator controls only the second term and wants it as low as possible. Together this is a **minimax game**:

$$\min_G \max_D V(D, G).$$

Here $p_\text{data}$ is the (unknown) distribution of real data, which we only see through samples; $p(z)$ is the noise distribution we chose; and $p_g$ is the distribution of $G(z)$, which no one writes down but which the generator's weights determine.

~~~dg-flow G only sees the data through D's gradient; D sees both real and generated samples.
Noise z -(generator)-> Fake image -> *Discriminator -(real or fake?)-> Loss
Real image -> *Discriminator
~~~

In practice the players alternate on each mini-batch: one step for $D$, then one for $G$, whose gradient flows *through* $D$ via $\partial D / \partial x$ at the fake sample.

<div class="ex">
<div class="lab">Worked example</div>

**the two losses on a tiny batch**

Take two real samples on which $D$ outputs $0.8$ and $0.6$, and two fakes on which it outputs $0.3$ and $0.1$.

Discriminator loss on reals (cross-entropy with label 1): $-\tfrac12(\ln 0.8 + \ln 0.6) = \tfrac12(0.223 + 0.511) = 0.367$.

Discriminator loss on fakes (label 0): $-\tfrac12(\ln 0.7 + \ln 0.9) = \tfrac12(0.357 + 0.105) = 0.231$. Total discriminator loss: $0.367 + 0.231 = 0.598$, which is $-V$ estimated on this batch.

If $D$ were reduced to coin flips ($D = 0.5$ on everything) the total would be $\ln 2 + \ln 2 = 1.386$. That is the value to expect when the generator has won completely, so a discriminator loss below it means $D$ still separates the two.

Generator loss in the non-saturating form (Step 3): $-\tfrac12(\ln 0.3 + \ln 0.1) = \tfrac12(1.204 + 2.303) = 1.753$. The fake that $D$ rejects hardest (0.1) contributes the most.

</div>

## Step 2: what the ideal game optimises

Suppose the discriminator could be anything and we fixed the generator. At a single point $x$ the discriminator's objective is $p_\text{data}(x)\log D + p_g(x)\log(1 - D)$. Differentiating in $D$ and setting the result to zero gives

$$D^*(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_g(x)},$$

which is exactly the ratio in the table above. Check one entry: at $x = 2$, $0.166 / (0.166 + 0.228) = 0.42$.

Put $D^*$ back into $V$. After rearranging, the generator is left minimising

$$V(D^*, G) = 2\,\mathrm{JSD}(p_\text{data} \,\|\, p_g) - \log 4,$$

where the **Jensen-Shannon divergence** is $\mathrm{JSD}(p\|q) = \tfrac12\mathrm{KL}(p\|m) + \tfrac12\mathrm{KL}(q\|m)$ with $m = \tfrac12(p + q)$. It is a symmetric, bounded relative of the KL divergence: 0 when $p = q$ and at most $\log 2 \approx 0.693$ (in nats) when the two have no overlap at all. For the 1-D example the JSD works out (numerically) to 0.504, so $V(D^*, G) = 2(0.504) - 1.386 = -0.377$. The minimum, $-\log 4 = -1.386$, is reached only when $p_g = p_\text{data}$, at which point $D^* = \tfrac12$ everywhere.

So in the idealised limit (unlimited capacity, an optimal discriminator at every step) a GAN minimises JSD and its only fixed point is the data distribution. The rest of the lesson is about the gap between that promise and real training.

## Step 3: the non-saturating generator loss

Write the discriminator's output as $D = \sigma(l)$, where $l$ is the **logit** (the raw score before the sigmoid). Early in training fakes are easy to spot, so $D(G(z))$ is close to 0 and $l$ is very negative.

The minimax generator minimises $\log(1 - \sigma(l))$. Its derivative with respect to the logit is $-\sigma(l) = -D$. When $D \approx 0$, the gradient is about 0: the generator gets almost no signal at exactly the moment it is worst. This is called **saturation**.

The fix, already in the original paper, is to flip the question. Instead of minimising the chance that $D$ is right, the generator *maximises the chance that $D$ is wrong*: it minimises $-\log D(G(z))$. The derivative with respect to the logit is now $-(1 - \sigma(l)) = -(1 - D)$, close to 1 when $D \approx 0$.

~~~dg-fig 07.02-saturate.svg
Left: the minimax loss is nearly flat for confidently rejected fakes (D near 0), while the non-saturating loss is steep there. Right: the size of the gradient each loss passes back; at D = 0.01 it is 0.01 versus 0.99.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**how much gradient each loss gives**

Suppose $D(G(z)) = 0.01$, so the logit is $l = \ln(0.01/0.99) \approx -4.6$.

Minimax loss $\log(1 - D)$: $|\partial / \partial l| = \sigma(l) = 0.01$.

Non-saturating loss $-\log D$: $|\partial / \partial l| = 1 - \sigma(l) = 0.99$.

The non-saturating version pushes the generator $0.99 / 0.01 = 99$ times harder at this point.

</div>

Both losses are smallest when $D(G(z))$ is large, so they agree on where the generator should go; only the size of the push differs. In code the non-saturating loss is just "cross-entropy with the *real* label applied to fakes", which is why the training step below calls `bce(D(fake), ones)`. (Strictly, with the non-saturating loss the game no longer has a single value function, and in the ideal analysis it corresponds to a different divergence; in practice it is the default.)

## A GAN in code

A training step has two halves: update $D$ on a real and a fake batch, then update $G$ through the just-updated $D$. The detail beginners miss is `.detach()`: in $D$'s update the fake batch must not send gradient into $G$. The toy data are 2-D points from eight tight clusters on a circle, small enough to train on a CPU in under a minute.

```python
import math, torch, torch.nn as nn
torch.set_num_threads(1); torch.manual_seed(0)

def real_batch(n):                        # 8 tight clusters on a circle of radius 2
    k = torch.randint(0, 8, (n,))
    ang = k * (2 * math.pi / 8)
    centres = torch.stack([2 * torch.cos(ang), 2 * torch.sin(ang)], 1)
    return centres + 0.05 * torch.randn(n, 2)

def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, o))

G, D = mlp(8, 2), mlp(2, 1)               # G: 8-d noise -> 2-d point;  D: point -> logit
opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()              # sigmoid + log loss in one stable op
B = 256; ones, zeros = torch.ones(B, 1), torch.zeros(B, 1)

for step in range(1, 5001):
    real = real_batch(B)
    fake = G(torch.randn(B, 8))
    # discriminator step: real -> 1, fake -> 0; detach so G gets no gradient here
    d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)
    opt_d.zero_grad(); d_loss.backward(); opt_d.step()
    # generator step (non-saturating): ask D to call the fakes real
    g_loss = bce(D(fake), ones)
    opt_g.zero_grad(); g_loss.backward(); opt_g.step()
    if step in (1, 500, 1000, 2000, 5000):
        with torch.no_grad():
            dr = torch.sigmoid(D(real)).mean().item()
            df = torch.sigmoid(D(fake)).mean().item()
        print(f"step {step:4d}  d_loss {d_loss.item():.3f}  g_loss {g_loss.item():.3f}"
              f"  D(real) {dr:.2f}  D(fake) {df:.2f}")
# step    1  d_loss 1.441  g_loss 0.655  D(real) 0.52  D(fake) 0.52
# step  500  d_loss 1.073  g_loss 0.915  D(real) 0.65  D(fake) 0.44
# step 1000  d_loss 1.096  g_loss 1.106  D(real) 0.67  D(fake) 0.43
# step 2000  d_loss 1.191  g_loss 1.104  D(real) 0.56  D(fake) 0.40
# step 5000  d_loss 1.312  g_loss 0.754  D(real) 0.58  D(fake) 0.49
```

Two choices are standard practice: $D$ returns a logit and the loss is `BCEWithLogitsLoss`, which avoids `log(0)` when $D$ is confident; and Adam uses $\beta_1 = 0.5$ instead of 0.9 (popularised by DCGAN), because heavy momentum makes the game overshoot.

## Reading the result

This is where GANs differ most from everything earlier in the course: **the losses do not tell you whether training is working.** Each loss is measured against an opponent that keeps getting better, so a flat or rising loss can mean progress.

~~~dg-fig 07.02-curves.svg
Left: both losses fluctuate around a level instead of falling; the discriminator loss drifts up toward 2 log 2 = 1.386 as fakes improve. Right: the discriminator's average outputs on real and fake data stay apart but well inside (0, 1), which means it is still giving useful gradient.
~~~

What to read instead:

- **Samples from a fixed set of noise vectors**, saved every few hundred steps, so you compare like with like.
- **$D$'s outputs.** $D(\text{real}) \approx 1$ and $D(\text{fake}) \approx 0$ for long means $D$ has won and $G$ is starving; both at 0.5 from the start can mean $D$ is too weak. Above they sit near 0.6 and 0.4: a healthy contest.
- **A sample statistic** such as FID (below).

~~~dg-fig 07.02-ring.svg
Generated samples (dots) against the eight real clusters (circles) during the run above. At step 1 the generator outputs a blob; by step 500 it spreads around the ring; by step 5000 samples are tight and on target, but one cluster has gone empty.
~~~

The snapshots tell the real story. Counting a cluster as covered when at least 1% of 2,000 samples land within 0.15 of its centre, the run went from 0/8 clusters at step 1 to 8/8 at step 500 (but only 23% of samples on target), then 7/8 with 93% on target at step 5000. The generator got sharper and *lost* a cluster on the way. That is a mild case of the most famous GAN failure.

## What goes wrong

**Mode collapse.** The generator only has to fool the current $D$. If one region fools $D$ well, nothing rewards spreading out, so many $z$ map to the same few outputs; $D$ eventually penalises that region and $G$ hops to another.

**Oscillation.** Even for the game $\min_x \max_y xy$, simultaneous gradient steps spiral *around* the equilibrium at the origin and slowly outward. GANs inherit this rotation: losses swing and samples cycle.

**The discriminator wins.** Real images lie on a thin surface in pixel space and early fakes on another; thin surfaces rarely overlap, so $D$ can separate them perfectly. The JSD is then stuck at $\log 2$ however close they are, and $G$ learns nothing about which way to move.

~~~dg-fig 07.02-jsd.svg
Two very thin distributions slid past each other. The Jensen-Shannon divergence stays at log 2 until they overlap, so it gives no direction; the Wasserstein distance (drawn rescaled) shrinks steadily as the gap closes.
~~~

| Failure | What you see | Why | Common remedy
| --- | --- | --- | --- |
| Mode collapse | Many $z$ give the same few outputs; empty clusters | Nothing in the loss rewards diversity | Minibatch statistics in $D$, WGAN-style losses, larger batches
| Oscillation | Losses swing, samples cycle between styles | Simultaneous gradient play rotates around the equilibrium | Lower learning rates, Adam with $\beta_1 = 0$ to $0.5$, two time-scale updates (TTUR)
| Discriminator wins | $D$ loss near 0, $G$ stops improving | Disjoint supports, $D$ too strong | Gradient penalties, spectral normalisation, augment $D$'s inputs
| Artefacts at scale | Checkerboard patterns, broken textures | Transposed-convolution upsampling, unstable normalisation | Resize-then-convolve, careful architectures (DCGAN, StyleGAN)
| Forgetting to detach | $G$ gets updated in the wrong direction during $D$'s step | Fake batch carries gradient into $G$ | `fake.detach()` in the discriminator loss

A note from our own runs, to calibrate expectations: on this easy 2-D toy the minimax generator loss also trained fine (8 of 8 clusters, 90% of samples on target after 5,000 steps), because the discriminator never became confident: its loss stayed between 1.05 and 1.37, close to the coin-flip value. Saturation bites hardest on high-dimensional data, where the discriminator separates fakes almost perfectly early on. And adding spectral normalisation to every layer of this small discriminator, with the same learning rate, *broke* training (never more than 2 of 8 clusters and 10% of samples on target): stabilisers interact with learning rates and architecture, and you tune them together.

## Evaluating a GAN

With no likelihood, GANs are judged by sample statistics. The standard is the **Fréchet Inception Distance (FID)** (Heusel et al., 2017): embed real and generated images with a pretrained Inception-v3 (2,048-dimensional pooled features), fit a Gaussian (mean $\mu$, covariance $\Sigma$) to each set, and compute

$$\mathrm{FID} = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\big).$$

The first term catches a shift in average content, the trace term a mismatch in spread (missing variety). Lower is better.

```python
import numpy as np
from scipy.linalg import sqrtm

def frechet(a, b):                     # a, b: (n, d) feature matrices
    mu1, mu2 = a.mean(0), b.mean(0)
    S1, S2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = sqrtm(S1 @ S2).real      # drop tiny imaginary parts from rounding
    return ((mu1 - mu2) ** 2).sum() + np.trace(S1 + S2 - 2 * covmean)
```

To see what it measures, we used the 2-D points as "features" and scored perfect fakes covering only some clusters.

~~~dg-fig 07.02-fid.svg
Fréchet distance to the real ring when the fake samples cover k of the 8 clusters (5,000 samples each, 20 random choices of clusters). Every fake sample is flawless, yet missing clusters raise the score; our trained GAN (7 of 8) scores 0.12.
~~~

Covering one cluster scores 7.7, four clusters 0.88, seven 0.11 and all eight 0.001. FID punishes missing modes that a glance at a few samples misses. Caveats: it depends on sample count (50,000 is the convention), feature network and preprocessing, so only compare FIDs computed by the same code.

## Going deeper

**Wasserstein GAN.** Arjovsky et al. (2017) replaced the JSD with the **earth mover's (Wasserstein-1) distance**: the least total "mass times distance" needed to move one distribution onto the other, which shrinks smoothly even with no overlap. By the Kantorovich-Rubinstein duality, $W_1(p, q) = \sup_{\|f\|_L \le 1} \mathbb{E}_p[f] - \mathbb{E}_q[f]$, so the discriminator becomes a **critic** $f$ that outputs an unbounded score and must be **1-Lipschitz** (its output may change by at most 1 per unit change of input). The critic maximises $\mathbb{E}[f(\text{real})] - \mathbb{E}[f(\text{fake})]$ and the generator minimises $-\mathbb{E}[f(\text{fake})]$.

**Enforcing the Lipschitz constraint.** The original WGAN clipped every weight to $[-0.01, 0.01]$, which works poorly (it pushes the critic toward very simple functions). **WGAN-GP** (Gulrajani et al., 2017) instead adds a penalty $\lambda(\|\nabla_{\hat x} f(\hat x)\|_2 - 1)^2$ with $\lambda = 10$, evaluated at random points $\hat x$ on lines between real and fake samples:

```python
eps = torch.rand(B, 1)
x_hat = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
grad, = torch.autograd.grad(D(x_hat).sum(), x_hat, create_graph=True)
gp = ((grad.norm(dim=1) - 1) ** 2).mean()
d_loss = D(fake.detach()).mean() - D(real).mean() + 10 * gp
```

`create_graph=True` lets the penalty be differentiated with respect to $D$'s weights. WGAN-GP takes several critic steps per generator step (5 in the paper) and Adam with $\beta_1 = 0$, $\beta_2 = 0.9$.

**Cheaper smoothness.** **Spectral normalisation** (Miyato et al., 2018) divides each weight matrix by an estimate of its largest singular value, which bounds every layer's Lipschitz constant; it costs one power-iteration step per update. The **R1 penalty** penalises $\|\nabla_x D(x)\|^2$ on *real* data only, which keeps $D$ flat near the data so the game does not spiral; StyleGAN uses it by default.

**Why games are hard.** The joint gradient field of a game has a rotational part with no counterpart in single-loss optimisation; smaller steps, two time-scale updates, extragradient methods and averaging the generator's weights all damp it.

**Where GANs stand now.** Diffusion models (next lessons) have largely replaced GANs for images: their training is plain regression with no game, and they cover modes well. Adversarial losses survive in the decoders of latent autoencoders and in distilling diffusion samplers into few-step generators.

## Key terms

| Term | Meaning
| --- | --- |
| Generator $G$ | Network that maps random noise $z$ to a sample; trained only through the discriminator's gradient
| Discriminator $D$ | Classifier that outputs the probability a sample is real
| Minimax game | $G$ minimises and $D$ maximises the same value $V(D, G)$
| Optimal discriminator | $D^* = p_\text{data}/(p_\text{data} + p_g)$ for a fixed generator
| Jensen-Shannon divergence | Symmetric, bounded divergence (at most $\log 2$) that the ideal GAN minimises
| Non-saturating loss | Generator minimises $-\log D(G(z))$, keeping gradient strong when fakes are rejected
| Mode collapse | Generator covers only part of the data distribution
| WGAN critic | Unbounded, 1-Lipschitz scorer whose gap estimates the Wasserstein distance
| FID | Fréchet distance between Gaussians fitted to Inception features of real and generated samples

<div class="take">
<div class="lab">Key takeaway</div>

a GAN trains a generator purely through a discriminator's gradient; the ideal game minimises the Jensen-Shannon divergence, but in practice you need the non-saturating loss, careful optimiser settings and Lipschitz control (WGAN-GP, spectral norm, R1) to keep the game from collapsing or cycling, and you judge the result by samples and FID, never by the losses.

</div>
