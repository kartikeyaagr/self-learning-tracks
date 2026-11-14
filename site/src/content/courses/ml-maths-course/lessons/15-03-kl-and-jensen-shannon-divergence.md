---
id: "15.03"
course: "ml-maths-course"
module: "15"
slug: "kl-and-jensen-shannon-divergence"
title: "KL and Jensen-Shannon divergence"
keyIdeas: ["$D_{\\mathrm{KL}}(p\\,\\|\\,q) = \\sum_x p(x)\\log\\frac{p(x)}{q(x)}$ is the extra code length from using $q$ when data comes from $p$; it is never negative and is zero only when $p = q$.", "KL is not symmetric: forward KL $D(p\\,\\|\\,q)$ forces $q$ to cover all of $p$, while reverse KL $D(q\\,\\|\\,p)$ lets $q$ settle on one mode.", "Jensen-Shannon divergence compares both distributions with their average, so it is symmetric, always finite and at most $\\log 2$.", "KL terms appear as regularisers in ML: the VAE prior term, the penalty that keeps a fine-tuned policy near its reference, and distillation."]
question: "Let $p = \\mathcal{N}(0, 1)$ and $q = \\mathcal{N}(1, 4)$ (variance 4). (a) Compute $D_{\\mathrm{KL}}(p\\,\\|\\,q)$ in nats. (b) Without computing, is $D_{\\mathrm{KL}}(q\\,\\|\\,p)$ larger or smaller, and why? (c) If you fit a single Gaussian $q$ to a two-peaked $p$, which direction of KL gives a broad $q$ that covers both peaks?"
answer: "(a) For Gaussians, $D(\\mathcal{N}(\\mu_1,\\sigma_1^2)\\,\\|\\,\\mathcal{N}(\\mu_2,\\sigma_2^2)) = \\ln\\frac{\\sigma_2}{\\sigma_1} + \\frac{\\sigma_1^2 + (\\mu_1-\\mu_2)^2}{2\\sigma_2^2} - \\frac12 = \\ln 2 + \\frac{1 + 1}{8} - 0.5 \\approx 0.443$ nats. (b) Larger. Now the expectation is over the wide $q$, which puts substantial mass where the narrow $p$ has tiny density, and $\\log(q/p)$ is very large there; numerically it is about 1.31 nats. (c) Forward KL, $D(p\\,\\|\\,q)$: it penalises $q$ heavily wherever $p$ has mass and $q$ does not, so the best $q$ spreads to cover both peaks. Reverse KL would let $q$ sit on one peak."
sources: [{"title": "Wikipedia, Kullback-Leibler divergence", "url": "https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence", "note": "Properties, closed forms and interpretations."}, {"title": "Wikipedia, Jensen-Shannon divergence", "url": "https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence", "note": "Definition, bounds and the metric property of its square root."}, {"title": "Kingma and Welling, Auto-Encoding Variational Bayes (2013)", "url": "https://arxiv.org/abs/1312.6114", "note": "The KL term in the VAE objective, with the Gaussian closed form."}, {"title": "MacKay, Information Theory, Inference, and Learning Algorithms (free book)", "url": "https://www.inference.org.uk/itprnn/book.pdf", "note": "Chapter 2 on relative entropy and Gibbs' inequality."}]
practice: ["kl-divergence-discrete-and-continuous", "diagonal-gaussian-kl-divergence", "population-stability-index", "ppo-with-reference-policy-kl"]
tags: ["KL divergence", "forward and reverse KL", "Jensen-Shannon divergence"]
tools: ["NumPy"]
---

## Where this lesson sits

The AI Research course (lesson 01.12) introduces KL divergence with examples. This lesson keeps the basics short and focuses on the properties you use in derivations and on the two directions.

## Definition and basic facts

$D_{\mathrm{KL}}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p, q) - H(p)$.

- It is the extra bits per symbol paid for coding $p$ with a code built for $q$ (lesson 15.02).
- Gibbs' inequality: $D_{\mathrm{KL}}(p\,\|\,q) \ge 0$, with equality only if $p = q$. It follows from Jensen's inequality applied to the concave logarithm.
- It is infinite if $q(x) = 0$ somewhere that $p(x) > 0$: a code with no codeword for a possible symbol cannot work.
- It is not symmetric and does not satisfy the triangle inequality, so it is not a distance.

For two Gaussians there is a closed form:
$D(\mathcal{N}(\mu_1,\sigma_1^2)\,\|\,\mathcal{N}(\mu_2,\sigma_2^2)) = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12$.
For diagonal multivariate Gaussians, sum this over dimensions. The VAE uses it with $q = \mathcal{N}(\mu, \sigma^2)$ and a standard normal prior, giving $\tfrac12\sum_j(\mu_j^2 + \sigma_j^2 - \ln\sigma_j^2 - 1)$.

## Forward and reverse KL

When you fit a simple $q$ to a complicated $p$, the direction matters.

- Forward KL, $D(p\,\|\,q)$, averages over $p$. Wherever $p$ has mass and $q$ is small, $\log(p/q)$ is large, so $q$ must cover all of $p$. The fit is broad and mass-covering. Maximum likelihood minimises this direction.
- Reverse KL, $D(q\,\|\,p)$, averages over $q$. It does not care about regions where $q$ is zero, but punishes $q$ for putting mass where $p$ is small. The fit is narrow and mode-seeking. Variational inference minimises this direction, which is why variational posteriors often underestimate uncertainty.

~~~dg-layers Which direction to expect in common methods.
Maximum likelihood, distillation | forward KL: data or teacher distribution on the left; broad fits
Variational inference, VAE posterior | reverse KL: approximation on the left; narrow, mode-seeking fits
*KL penalty to a reference policy | KL of the trained policy from the reference; keeps outputs near the reference
~~~

<div class="ex">
<div class="lab">Worked example</div>

**two peaks, one Gaussian**

Let $p$ be an equal mixture of Gaussians centred at $-3$ and $+3$, each with variance 1. Fitting one Gaussian by forward KL gives mean 0 and variance 10 (the mixture's own mean and variance), which puts most of its mass in the valley between the peaks. Fitting by reverse KL gives a Gaussian centred on one of the peaks with variance close to 1, ignoring the other. Neither is right; they fail in opposite ways.

</div>

## Jensen-Shannon divergence

Let $m = \tfrac12(p + q)$. Then
$\mathrm{JS}(p, q) = \tfrac12 D_{\mathrm{KL}}(p\,\|\,m) + \tfrac12 D_{\mathrm{KL}}(q\,\|\,m)$.
Because $m$ is positive wherever $p$ or $q$ is, JS is always finite. It is symmetric, lies between 0 and $\ln 2$ nats (1 bit), and its square root is a true metric.

JS appears in the original GAN analysis: with the optimal discriminator, the generator's objective equals $2\,\mathrm{JS}(p_{\text{data}}, p_g) - \ln 4$. When the two distributions barely overlap, JS sits near its maximum and changes little as the generator improves, so gradients are weak. This observation motivated other divergences and distances for GAN training.

## KL in practice

- VAE: the KL term pulls each posterior toward the prior; too strong a weight makes the model ignore the latent.
- Fine-tuning language models with rewards: a penalty $\beta\,D_{\mathrm{KL}}(\pi\,\|\,\pi_{\text{ref}})$ stops the policy from drifting into degenerate text that exploits the reward model.
- Distillation: the student minimises cross-entropy (equivalently forward KL) to the teacher's softened distribution.
- Monitoring: the population stability index used for drift is a symmetrised KL between binned feature distributions.

## Estimating KL from samples

With samples from $p$ and densities you can evaluate, $D(p\,\|\,q) \approx \frac{1}{N}\sum_i \log\frac{p(x_i)}{q(x_i)}$. For a language-model policy, the per-token log-ratio of sampled tokens gives an unbiased but noisy estimate; individual samples can be negative even though the expectation is not.

## Common mistakes

- Swapping the arguments: library functions differ in which argument is the data distribution; check the documentation against a two-point example.
- Adding a tiny epsilon to zero probabilities without thinking: it makes KL finite but the value then depends mostly on the epsilon.
- Reading a small KL as "similar everywhere": KL averages over $p$, so large differences where $p$ is rare barely count.

<div class="take">
<div class="lab">Key takeaway</div>

KL is the extra code length from using the wrong distribution; it is non-negative and asymmetric, forward KL covers and reverse KL seeks modes, and JS divergence is its bounded, symmetric relative.

</div>
