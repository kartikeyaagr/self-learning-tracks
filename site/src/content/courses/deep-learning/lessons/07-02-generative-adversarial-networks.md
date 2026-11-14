---
id: "07.02"
course: "deep-learning"
module: "07"
slug: "generative-adversarial-networks"
title: "Generative adversarial networks"
keyIdeas: ["A GAN trains a generator $G(z)$ against a discriminator $D(x)$ in the minimax game $\\min_G\\max_D\\ \\mathbb E[\\log D(x)]+\\mathbb E[\\log(1-D(G(z)))]$; no likelihood is ever computed.", "At the optimal discriminator $D^*=p_\\text{data}/(p_\\text{data}+p_g)$ the generator minimises $2\\,\\mathrm{JSD}(p_\\text{data}\\|p_g)-\\log 4$.", "Use the non-saturating generator loss $-\\log D(G(z))$: the original loss gives almost no gradient exactly when the generator is bad.", "Instabilities (mode collapse, oscillation, a discriminator that wins outright) come from simultaneous gradient play and disjoint supports; WGAN-GP, spectral norm and R1 penalties tame them.", "GANs are judged with sample statistics such as FID, the Fréchet distance between Inception-feature Gaussians of real and generated images."]
question: "Halfway through training, your GAN's discriminator loss drops to almost zero and stays there, and generated samples stop improving. Explain what the generator's gradient looks like in this state under the minimax loss and under the non-saturating loss, and name two changes that would restore a useful signal."
answer: "A near-zero discriminator loss means $D(G(z))\\approx 0$ for fakes, so the logit is very negative. Under the minimax loss $\\log(1-D)$ the gradient with respect to the logit is $\\sigma(l)\\approx 0$, so the generator receives essentially nothing. The non-saturating loss $-\\log D$ gives a logit gradient near 1, but that only helps if $D$'s gradient with respect to the image is informative; when real and fake supports are disjoint and $D$ is a sharp classifier, $\\partial D/\\partial x$ can be tiny or point in unhelpful directions. Restoring signal means constraining $D$: add a gradient penalty (WGAN-GP on interpolates, or R1 on reals) or spectral normalisation so $D$ stays smooth, and/or weaken it (fewer $D$ steps, lower $D$ learning rate, augmentation of $D$'s inputs). Switching to a Wasserstein-style critic also keeps the loss informative when supports do not overlap."
sources: [{"title": "Goodfellow et al.: Generative Adversarial Nets", "url": "https://arxiv.org/abs/1406.2661", "note": "The original paper: the minimax value, the optimal discriminator and the JSD result (section 4)."}, {"title": "Goodfellow: NIPS 2016 Tutorial on GANs", "url": "https://arxiv.org/abs/1701.00160", "note": "Why the non-saturating loss, mode collapse, and practical tips from the inventor."}, {"title": "Gulrajani et al.: Improved Training of Wasserstein GANs", "url": "https://arxiv.org/abs/1704.00028", "note": "WGAN-GP: why weight clipping fails and how the gradient penalty works."}, {"title": "Heusel et al.: GANs Trained by a Two Time-Scale Update Rule", "url": "https://arxiv.org/abs/1706.08500", "note": "Introduces both TTUR and the FID metric."}]
practice: ["gan-training-step", "frechet-distance-between-feature-sets"]
tags: ["GAN", "minimax", "FID"]
tools: ["PyTorch"]
---

## Two networks, one game

A VAE needs an explicit likelihood $p_\theta(x\mid z)$, and the Gaussian choice is what makes its samples blurry. Goodfellow et al. (2014) avoided likelihoods altogether. A **generator** $G$ maps noise $z\sim\mathcal N(0,I)$ (often 100 or 512 dimensions) to a sample $G(z)$. A **discriminator** $D$ maps any sample to the probability that it is real. $D$ is trained as a binary classifier, real versus generated; $G$ is trained to fool it. The only learning signal $G$ ever receives is the gradient of $D$'s output with respect to the image.

$$\min_G\max_D\ V(D,G)=\mathbb E_{x\sim p_\text{data}}[\log D(x)]+\mathbb E_{z\sim p(z)}[\log(1-D(G(z)))]$$

~~~dg-flow G only sees the data through D's gradient; D sees both real and generated samples.
Noise z -(generator)-> Fake image -> *Discriminator -(real or fake?)-> Loss
Real image -> *Discriminator
~~~

## What the game optimises

Fix $G$ and solve for the best discriminator pointwise: maximising $p_\text{data}(x)\log D + p_g(x)\log(1-D)$ gives $D^*(x)=p_\text{data}(x)/(p_\text{data}(x)+p_g(x))$. Plug it back in and the generator's objective becomes $2\,\mathrm{JSD}(p_\text{data}\,\|\,p_g)-\log 4$, minimised exactly when $p_g=p_\text{data}$, where $D^*=\tfrac12$ everywhere. So in the idealised infinite-capacity limit, a GAN is a Jensen-Shannon divergence minimiser.

Two facts in that derivation explain most practical trouble. First, it assumes $D$ is optimal at every step, which never holds; training is a simultaneous game, not minimisation of a fixed loss, and gradient descent on games can circle instead of converging. Second, real images live on thin low-dimensional manifolds. Early in training $p_g$ and $p_\text{data}$ barely overlap, the JSD sits at its maximum $\log 2$ no matter how close the manifolds are, and a near-perfect discriminator gives the generator a flat landscape.

## The non-saturating loss

Write the discriminator output as $D=\sigma(l)$ for a logit $l$. Early on, $D$ rejects fakes confidently, $l$ is very negative and $D(G(z))\approx 0$. The minimax generator loss $\log(1-\sigma(l))$ has gradient $-\sigma(l)$ with respect to $l$: nearly zero, exactly when the generator most needs to learn. The standard fix is to have $G$ maximise $\log D(G(z))$ instead, whose gradient is $1-\sigma(l)$, close to 1 in the same regime. Both losses share the same fixed point; only the dynamics differ.

<div class="ex">
<div class="lab">Worked example</div>

**how much gradient each loss gives**

Suppose $D(G(z))=0.01$, i.e. logit $l\approx-4.6$.

Minimax loss $\log(1-D)$: $|\partial/\partial l| = \sigma(l) = 0.01$.

Non-saturating loss $-\log D$: $|\partial/\partial l| = 1-\sigma(l) = 0.99$.

The non-saturating version pushes the generator about 99 times harder at the start of training.

</div>

```python
# one training step, non-saturating GAN (BCE on logits)
bce = torch.nn.BCEWithLogitsLoss()
fake = G(torch.randn(B, z_dim))
d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)   # detach: no G grads here
opt_d.zero_grad(); d_loss.backward(); opt_d.step()
g_loss = bce(D(fake), ones)                                    # "label fakes as real"
opt_g.zero_grad(); g_loss.backward(); opt_g.step()
```

## Training instabilities and the fixes that stuck

| Failure | What you see | Why | Common remedy
| --- | --- | --- | --- |
| Mode collapse | Many $z$ give the same few images | $G$ finds one output $D$ currently accepts; nothing penalises low diversity directly | Minibatch statistics, WGAN-style losses, larger batches
| Oscillation | Losses swing, samples cycle between styles | Simultaneous gradient play rotates around the equilibrium | Lower learning rates, Adam with $\beta_1=0$ to $0.5$, two time-scale updates (TTUR)
| Discriminator wins | $D$ loss to 0, $G$ stops improving | Disjoint supports, $D$ too strong | Gradient penalties, spectral normalisation, data augmentation for $D$
| Artefacts at scale | Checkerboards, broken textures | Transposed-conv upsampling, unstable normalisation | Resize-then-conv, careful architecture (DCGAN, StyleGAN)

The **Wasserstein GAN** (Arjovsky et al., 2017) replaces JSD with the earth mover's distance, which still varies smoothly when supports do not overlap. Its "critic" outputs an unbounded score and must be 1-Lipschitz; the original paper clipped weights to $\pm 0.01$, and WGAN-GP replaced clipping with a penalty $\lambda(\|\nabla_{\hat x}D(\hat x)\|_2-1)^2$, $\lambda=10$, on random interpolates between real and fake samples. **Spectral normalisation** divides each weight matrix by its largest singular value to bound $D$'s Lipschitz constant cheaply, and the **R1 penalty** (gradient norm on real data only) is the default in StyleGAN.

## Evaluating samples

With no likelihood, you judge a GAN by statistics of its samples. The **Fréchet Inception Distance** embeds real and generated images with an Inception-v3 network, fits a Gaussian to each feature set, and computes $\|\mu_r-\mu_g\|^2+\mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$. Lower is better; it captures both quality and diversity, but it depends on sample count (use 50k), on the feature network and on preprocessing, so only compare FIDs computed the same way. Diffusion models (next lessons) have largely replaced GANs for image generation because their training is a plain regression with no game, but adversarial losses survive inside VAE tokenisers and in few-step distilled diffusion samplers.

<div class="take">
<div class="lab">Key takeaway</div>

a GAN trains a generator purely through a discriminator's gradient; the ideal game minimises Jensen-Shannon divergence, but in practice you need the non-saturating loss and Lipschitz control (WGAN-GP, spectral norm, R1) to keep the game from collapsing or cycling.

</div>
