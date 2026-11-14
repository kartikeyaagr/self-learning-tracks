---
id: "07.01"
course: "deep-learning"
module: "07"
slug: "autoencoders-and-vaes"
title: "Autoencoders and VAEs"
keyIdeas: ["A plain autoencoder compresses $x$ to a code $z$ and reconstructs it, but its code space has holes, so decoding a random $z$ gives garbage: it is a compressor, not a generator.", "A VAE fixes the prior $p(z)=\\mathcal N(0,I)$, learns a decoder $p_\\theta(x\\mid z)$ and an approximate posterior $q_\\phi(z\\mid x)$, and maximises the ELBO $\\mathbb E_q[\\log p_\\theta(x\\mid z)] - \\mathrm{KL}(q_\\phi(z\\mid x)\\,\\|\\,p(z))$.", "The reparameterisation $z=\\mu+\\sigma\\odot\\epsilon$, $\\epsilon\\sim\\mathcal N(0,I)$ moves the randomness outside the network so gradients flow through the sample.", "For a diagonal Gaussian encoder the KL has a closed form: $\\tfrac12\\sum_j(\\mu_j^2+\\sigma_j^2-\\log\\sigma_j^2-1)$. Predict $\\log\\sigma^2$, not $\\sigma$.", "Typical failure modes are blurry samples and posterior collapse (the KL goes to zero and the decoder ignores $z$)."]
question: "You train a VAE on MNIST with the loss `F.mse_loss(x_hat, x) + kl.mean()`, where `mse_loss` averages over all 784 pixels and the KL is summed over 20 latent dimensions. Samples from the prior all look like the same grey blob. What happened, and how do you fix it?"
answer: "Averaging the reconstruction error over 784 pixels makes it about 784 times smaller than the per-image negative log-likelihood, while the KL is still summed over latent dimensions. The KL therefore dominates: the cheapest solution is to set $q(z\\mid x)$ equal to the prior for every input (KL = 0), so $z$ carries no information and the decoder learns to output the mean image. This is posterior collapse caused by mis-weighting, not by the model. Fix it by summing the reconstruction loss over pixels per image (equivalently, choosing a sensible decoder variance) and averaging over the batch, so both terms are per-example log-densities. KL warm-up or a free-bits floor helps if collapse persists. Check by watching per-dimension KL: active dimensions should sit well above zero."
sources: [{"title": "Kingma and Welling: Auto-Encoding Variational Bayes", "url": "https://arxiv.org/abs/1312.6114", "note": "The original VAE paper: the ELBO, the reparameterisation trick and the closed-form Gaussian KL (appendix B)."}, {"title": "Kingma and Welling: An Introduction to Variational Autoencoders", "url": "https://arxiv.org/abs/1906.02691", "note": "A longer, gentler monograph by the same authors; chapter 2 derives the ELBO gap and gradient estimators."}, {"title": "Doersch: Tutorial on Variational Autoencoders", "url": "https://arxiv.org/abs/1606.05908", "note": "Intuition-first tutorial with conditional VAEs; good for the 'why does the code space become dense' picture."}]
practice: ["variational-autoencoder-forward-pass", "reparameterized-gaussian-sample", "diagonal-gaussian-kl-divergence", "autoencoder-forward-pass", "kl-divergence-discrete-and-continuous"]
tags: ["VAE", "ELBO", "reparameterisation"]
tools: ["PyTorch"]
---

## From compression to generation

An **autoencoder** is two networks trained end to end. The encoder maps an input $x$ (say a 28x28 MNIST digit flattened to 784 numbers) to a short code $z$ (say 20 numbers); the decoder maps $z$ back to $\hat x$; the loss is reconstruction error, usually $\|x-\hat x\|^2$. The bottleneck forces the code to keep only what matters. That is useful for compression, denoising and feature learning, but it does not give you a generator. Nothing tells the encoder where in code space to put things, so the codes of real images form a few scattered islands. Pick a random $z$ between them and the decoder produces something that looks like no digit at all.

A **variational autoencoder** (Kingma and Welling, 2013) makes the code space usable by turning the autoencoder into a latent-variable model:

- **Prior**: $p(z)=\mathcal N(0, I)$. Generation is "sample $z$ from the prior, decode it".
- **Decoder / likelihood**: $p_\theta(x\mid z)$, for example a Gaussian centred on the decoder output, or a Bernoulli per pixel.
- **Encoder / approximate posterior**: $q_\phi(z\mid x)=\mathcal N(\mu_\phi(x), \mathrm{diag}\,\sigma^2_\phi(x))$. The true posterior $p_\theta(z\mid x)$ is intractable, so we learn an approximation.

~~~dg-flow The encoder outputs a distribution, not a point; the sample is drawn with external noise so gradients reach mu and sigma.
x -(encoder)-> mu, log var -(z = mu + sigma * eps)-> *z -(decoder)-> x hat
~~~

## The ELBO

We would like to maximise $\log p_\theta(x)=\log\int p_\theta(x\mid z)p(z)\,dz$, but the integral over $z$ is intractable. For any $q$, Jensen's inequality gives a lower bound, the **evidence lower bound**:

$$\log p_\theta(x) \ge \mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] - \mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)$$

The two terms read naturally. The first is a reconstruction term: codes drawn from the encoder should decode back to $x$. The second is a regulariser: each input's code distribution should stay close to the prior, which packs the islands together into one smooth ball that the prior actually covers. The gap between the two sides is exactly $\mathrm{KL}(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x))$, so maximising the ELBO over $\phi$ also makes the encoder a better posterior approximation. (KL divergence itself is covered in AI Research, Math Fundamentals.)

With a Gaussian decoder of fixed variance $\sigma_x^2$, $-\log p_\theta(x\mid z)$ is $\|x-\hat x\|^2/(2\sigma_x^2)$ plus a constant, so the reconstruction term is a scaled MSE. The choice of $\sigma_x$ silently sets the balance between the two terms; using plain MSE with weight 1 is a specific, often too-weak, choice of that balance.

## Reparameterisation

The reconstruction term is an expectation over $z\sim q_\phi$, estimated with one sample per input. You cannot backpropagate through "draw a random number". The trick is to write the sample as a deterministic function of the parameters and independent noise:

$$z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon,\qquad \epsilon\sim\mathcal N(0, I)$$

Now $\partial z/\partial\mu = 1$ and $\partial z/\partial\sigma=\epsilon$, and the gradient is an ordinary pathwise derivative with low variance. This works for any distribution you can write as a differentiable transform of fixed noise; discrete latents need other tools (Gumbel-softmax, straight-through, or vector quantisation as in VQ-VAE).

## KL to a standard Gaussian

For a diagonal Gaussian encoder and a standard normal prior the KL is a sum over latent dimensions with no sampling needed:

$$\mathrm{KL}=\tfrac12\sum_{j=1}^{d}\left(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\right)$$

Each dimension pays for moving its mean away from 0 and for a variance different from 1. Networks predict $\log\sigma^2$ so the variance is always positive and the gradient is well behaved near zero.

```python
import torch
def vae_loss(x, enc, dec):
    mu, logvar = enc(x).chunk(2, dim=-1)          # (B, d) each
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    x_hat = dec(z)
    rec = ((x - x_hat) ** 2).sum(-1)               # Gaussian decoder, up to scale
    kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(-1)
    return (rec + kl).mean()                       # negative ELBO per example
```

Sum over dimensions inside each example and average over the batch. Averaging over pixels instead shrinks the reconstruction term by a factor of 784 and the model happily collapses onto the prior.

<div class="ex">
<div class="lab">Worked example</div>

**KL for a 2-D code**

The encoder outputs $\mu=(1, 0)$ and $\log\sigma^2=(0, -2)$, so $\sigma^2=(1, 0.135)$.

Dimension 1: $\tfrac12(1+1-0-1)=0.5$ nats, paid only for the shifted mean.

Dimension 2: $\tfrac12(0+0.135+2-1)=0.568$ nats, paid only for the narrow variance.

Total KL is 1.07 nats. A dimension the decoder does not use drifts to $\mu=0,\sigma=1$ and costs nothing, which is how a VAE switches off unneeded latents.

</div>

## What goes wrong, and where VAEs are used

| Symptom | Cause | Usual fix
| --- | --- | --- |
| Blurry samples | Gaussian likelihood averages plausible outputs; per-pixel independence | Better likelihoods, perceptual or adversarial losses, hierarchical latents
| Posterior collapse | A powerful autoregressive decoder models $x$ alone, KL pushes $q$ to the prior | KL warm-up, "free bits" (a floor on per-dimension KL), weaker decoder
| Ignored dimensions | KL cost exceeds reconstruction benefit | Usually fine; it is automatic dimension selection
| Entangled factors | Nothing asks for disentanglement | $\beta$-VAE ($\beta>1$ on the KL), at a cost in reconstruction

The most important modern use of a VAE is as a **compressor for diffusion**. Latent diffusion (Stable Diffusion) trains a VAE with a small KL weight plus perceptual and adversarial losses that maps a 512x512x3 image to a 64x64x4 latent, and runs the diffusion process (lessons 07.03 to 07.07) in that latent space, which is 48 times smaller.

<div class="take">
<div class="lab">Key takeaway</div>

a VAE is an autoencoder whose encoder outputs a Gaussian, trained on the ELBO = reconstruction minus KL to the prior; reparameterisation makes the sample differentiable, and the closed-form KL keeps the code space dense enough that decoding prior samples produces real-looking data.

</div>
