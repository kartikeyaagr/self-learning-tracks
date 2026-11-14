---
id: "07.01"
course: "deep-learning"
module: "07"
slug: "autoencoders-and-vaes"
title: "Autoencoders and VAEs"
keyIdeas: ["A plain autoencoder compresses $x$ to a short code $z$ and rebuilds it. Nothing controls where codes go, so its code space has holes and decoding a random $z$ often gives garbage: it is a compressor, not a generator.", "A VAE fixes the prior $p(z)=\\mathcal N(0,I)$, learns a decoder $p_\\theta(x\\mid z)$ and an encoder $q_\\phi(z\\mid x)$ that outputs a Gaussian per input, and maximises the ELBO $\\mathbb E_q[\\log p_\\theta(x\\mid z)] - \\mathrm{KL}(q_\\phi(z\\mid x)\\,\\|\\,p(z))$: reconstruct well, but keep every code cloud near the prior.", "The reparameterisation $z=\\mu+\\sigma\\odot\\epsilon$, $\\epsilon\\sim\\mathcal N(0,I)$ moves the randomness outside the network, so gradients flow to $\\mu$ and $\\sigma$ through the sample.", "For a diagonal Gaussian encoder the KL has a closed form, $\\tfrac12\\sum_j(\\mu_j^2+\\sigma_j^2-\\log\\sigma_j^2-1)$; for example $\\mu=(1,0)$, $\\log\\sigma^2=(0,-2)$ costs 1.07 nats. Predict $\\log\\sigma^2$, not $\\sigma$.", "Sum the reconstruction over pixels and the KL over latents, then average over the batch. Averaging over pixels makes the KL dominate and every latent collapses to the prior; watch per-dimension KL to catch it.", "VAE samples tend to be blurry, and powerful decoders can cause posterior collapse. Today the VAE's main job is as the compressor in latent diffusion."]
question: "You train a VAE on MNIST with the loss `F.mse_loss(x_hat, x) + kl.mean()`, where `mse_loss` averages over all 784 pixels and the KL is summed over 20 latent dimensions. Reconstructions look like a grey smudge, samples from the prior all look like the same blob, and the logged KL is about 0.001 nats. What happened, how would you confirm it, and how do you fix it?"
answer: "Averaging the reconstruction error over 784 pixels makes it about 784 times smaller than a per-image negative log-likelihood, while the KL is still summed over the 20 latent dimensions per image. The KL therefore dominates the loss. The cheapest solution is to set $q(z\\mid x)$ equal to the prior $\\mathcal N(0, I)$ for every input (KL = 0), so $z$ carries no information about the image and the decoder learns to output the average digit, which is the grey blob. This is posterior collapse caused by mis-weighting, not by the model architecture. (Using MSE at all also implies a Gaussian decoder with variance 1/2, which is already a weak reconstruction term for pixels in [0, 1].) To confirm: log the KL per latent dimension; all 20 will be near zero, and the encoder's $\\mu$ will be near 0 and $\\log\\sigma^2$ near 0 for every input. A quick ablation that changes only the reduction makes the difference obvious (on the 8x8 sklearn digits, averaging instead of summing drove all eight per-dimension KLs to about 0 and raised reconstruction loss from 18.3 to 27.3 nats). Fix: compute the reconstruction loss per image by summing over pixels (binary cross-entropy with `reduction='none'` then `.sum(-1)`, or a Gaussian log-likelihood with a sensible or learned decoder variance), sum the KL over latents, add them and average over the batch, so both terms are per-example log-densities. If the KL still collapses with correct scaling, for example with a powerful decoder, use KL warm-up (anneal its weight from 0 to 1) or a free-bits floor on per-dimension KL. Afterwards, several latent dimensions should sit well above zero KL and prior samples should look like varied digits."
sources: [{"title": "Kingma and Welling: Auto-Encoding Variational Bayes", "url": "https://arxiv.org/abs/1312.6114", "note": "The original VAE paper: the ELBO, the reparameterisation trick and the closed-form Gaussian KL (appendix B)."}, {"title": "Kingma and Welling: An Introduction to Variational Autoencoders", "url": "https://arxiv.org/abs/1906.02691", "note": "A longer, gentler monograph by the same authors; chapter 2 derives the ELBO gap and gradient estimators."}, {"title": "Doersch: Tutorial on Variational Autoencoders", "url": "https://arxiv.org/abs/1606.05908", "note": "Intuition-first tutorial with conditional VAEs; good for the 'why does the code space become dense' picture."}]
practice: ["variational-autoencoder-forward-pass", "reparameterized-gaussian-sample", "diagonal-gaussian-kl-divergence", "autoencoder-forward-pass", "kl-divergence-discrete-and-continuous"]
tags: ["VAE", "ELBO", "reparameterisation"]
tools: ["PyTorch"]
---

## What you will learn

- What an autoencoder is (an encoder, a short code, a decoder) and why a plain one compresses well but cannot generate new data.
- How a variational autoencoder (VAE) turns the code into a probability distribution, so that "pick a random code and decode it" produces something realistic.
- The VAE loss, the ELBO, read as "reconstruct well" minus "keep codes close to a standard Gaussian", with every symbol explained.
- The reparameterisation trick that lets gradients pass through a random sample, and the closed-form KL term, computed by hand.
- How to train a small VAE in PyTorch, how to read its two loss terms, and how to spot blurry samples and posterior collapse.

*Before you start:* you need a multilayer perceptron trained with backpropagation and Adam (modules 01 and 02 of this course), the idea of a probability density and of a Gaussian $\mathcal N(\mu, \sigma^2)$, and log-likelihood losses such as cross-entropy. KL divergence is defined below in one line; AI Research, Math Fundamentals covers it in depth.

## Where this module starts

Every model so far in this course was **discriminative**: given an input, predict a label. A **generative model** learns the distribution of the data itself, so it can produce new examples that look like the training set. That is harder, because it must capture everything about the data, not just the part that decides the label.

This module covers the three families that matter in practice. This lesson builds the **variational autoencoder**, the simplest one with a clean probabilistic story, and the one whose encoder later becomes the compressor inside latent diffusion. Lesson 07.02 covers GANs, which trade the probabilistic story for sharp samples. Lessons 07.03 to 07.07 build diffusion and flow matching, today's default for images, audio and video.

## The idea in one picture

Take the 8x8 handwritten digits that ship with scikit-learn (`load_digits`). Each image is 64 numbers between 0 and 1. We keep the digits 0 to 7 (1,443 images) so that each class gets its own colour, and train two small networks that squeeze each image through a **code** of just 2 numbers and then rebuild it. One is a plain autoencoder, the other a VAE. Plotting each image's code as a point gives the next figure.

~~~dg-fig 07.01-latent.svg
The same digits, encoded to 2 numbers by two models. The plain autoencoder places codes anywhere, on its own arbitrary scale, with empty gaps between classes. The VAE packs them into a round blob that sits inside the standard Gaussian (dashed circles), so a random draw from that Gaussian lands among real codes.
~~~

Some measurements from those runs:

| Quantity | Plain autoencoder | VAE
| --- | --- | --- |
| Range of code values | about −6.6 to 14.2 | centred on 0, spread about 1.0 (by design)
| Reconstruction loss, nats per image (2-D code) | 19.7 | 21.3
| Random code decoded (8-D code): median distance to nearest real digit | 1.72 | 1.18

For reference, a real digit's nearest *other* real digit is at median distance 0.95. So the plain autoencoder rebuilds slightly better, but its random samples sit much further from anything real. The VAE gives up a little reconstruction to make its code space usable.

New words, in plain language:

- The **encoder** is a network that maps an input $x$ to a short code $z$. The **decoder** maps a code back to an image $\hat x$.
- The code space is called the **latent space**, and $z$ is a **latent variable**: a hidden description of the image (which digit, how slanted, how thick) that is never observed directly.
- **Nats** are units of information when you use natural logarithms. A loss of 21 nats per image means $-\log p(x\mid z) \approx 21$.

## Step 1: the plain autoencoder

An **autoencoder** is an encoder and a decoder trained together on one objective: the output should equal the input.

$$\min_{\text{enc},\,\text{dec}} \; \frac1n\sum_{i=1}^n \ell\big(x_i,\ \text{dec}(\text{enc}(x_i))\big)$$

Here $x_i$ is one training image, $\text{enc}(x_i) = z_i$ is its code, $\text{dec}(z_i) = \hat x_i$ is the reconstruction, and $\ell$ is a reconstruction loss such as squared error $\|x - \hat x\|^2$ or per-pixel cross-entropy. The **bottleneck** (a code much shorter than the input, here 2 or 8 numbers instead of 64) is what makes this non-trivial: the network cannot copy, so it must keep only what matters.

Autoencoders are useful for compression, denoising, anomaly detection (unusual inputs reconstruct badly) and feature learning. But they are not generators. Nothing in the loss says *where* in latent space the codes should go. The encoder puts each class wherever is convenient, at any scale, and the space between and around those islands is never trained. Decode a code from such a hole and you get something that looks like no digit at all.

~~~dg-fig 07.01-samples.svg
Left: real digits. Middle: an 8-D plain autoencoder decoding random codes drawn uniformly from the box its training codes occupy; many are smeared, half-formed shapes. Right: an 8-D VAE decoding codes drawn from N(0, I); blurry, but mostly recognisable digits.
~~~

The holes grow with dimension: in 8-D the box around the training codes is almost all empty space.

## Step 2: make the code a distribution

The VAE (Kingma and Welling, 2013; Rezende and colleagues proposed the same idea independently in 2014) fixes this by writing down a probabilistic model and training it properly. Three pieces:

- **Prior** $p(z) = \mathcal N(0, I)$: before seeing any image, a code is a vector of independent standard normal numbers. Generation is then simple: sample $z$ from the prior and decode it.
- **Decoder, or likelihood,** $p_\theta(x\mid z)$: a network with weights $\theta$ that turns a code into a distribution over images. For pixels in $[0,1]$ a common choice is one Bernoulli per pixel, whose probability is the decoder's sigmoid output; then $-\log p_\theta(x\mid z)$ is the binary cross-entropy *summed* over pixels. A Gaussian with fixed variance gives a scaled squared error instead.
- **Encoder, or approximate posterior,** $q_\phi(z\mid x) = \mathcal N\big(\mu_\phi(x),\ \mathrm{diag}\,\sigma_\phi^2(x)\big)$: a network with weights $\phi$ that, for each image, outputs a mean vector $\mu$ and a variance vector $\sigma^2$. Instead of one code per image, it gives a small cloud of plausible codes.

Why "approximate posterior"? The true posterior $p_\theta(z\mid x)$, "which codes could have produced this image", follows from Bayes' rule but needs the integral $p_\theta(x) = \int p_\theta(x\mid z)\,p(z)\,dz$, which no one can compute for a neural decoder. So we learn an encoder that approximates it, one forward pass per image. This shared-encoder shortcut is called **amortised** inference.

~~~dg-flow The encoder outputs a distribution, not a point; the sample is drawn with external noise so gradients reach mu and sigma.
x -(encoder)-> mu, log var -(z = mu + sigma * eps)-> *z -(decoder)-> x hat
~~~

## Step 3: the loss, called the ELBO

Ideally we would maximise the **log-likelihood** $\log p_\theta(x)$ of each training image: how probable the model finds the data. That needs the impossible integral above. Instead we maximise a lower bound on it, the **evidence lower bound (ELBO)**:

$$\log p_\theta(x) \;\ge\; \underbrace{\mathbb E_{z\sim q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction}} \;-\; \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{stay near the prior}}$$

Symbol by symbol:

- $\mathbb E_{z\sim q_\phi(z\mid x)}[\cdot]$ means "average over codes sampled from the encoder's cloud for this image". In practice one sample per image per step is enough.
- $\log p_\theta(x\mid z)$ is how well the decoder rebuilds $x$ from that code: the negative of the reconstruction loss.
- $\mathrm{KL}(q\,\|\,p) = \mathbb E_q[\log q(z) - \log p(z)]$ is the **Kullback-Leibler divergence**: a non-negative number that is zero only when the two distributions are identical. Here it measures how far this image's code cloud is from the standard Gaussian.

The training loss is the negative ELBO, summed over the dimensions of each example and averaged over the batch:

$$\mathcal L(x) = \underbrace{-\log p_\theta(x\mid z)}_{\text{reconstruction loss}} + \mathrm{KL}\big(q_\phi(z\mid x)\,\|\,\mathcal N(0, I)\big), \qquad z \sim q_\phi(z\mid x).$$

The two terms pull in opposite directions, and that tension is the design. Reconstruction wants each image's cloud tiny and far from the others; the KL wants every cloud to be the standard Gaussian. The compromise is the right panel of the first figure: classes still separate, but they fill the prior with no big holes, and neighbouring clouds overlap enough that the decoder learns sensible outputs everywhere in between.

## Step 4: reparameterisation, or how to backpropagate through a sample

The reconstruction term needs a sample $z$ from the encoder's Gaussian. "Draw a random number" has no gradient, so backpropagation would stop there and the encoder would never learn. The fix is to move the randomness outside the network:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal N(0, I).$$

Here $\odot$ means element-wise multiplication and $\epsilon$ is fresh noise that does not depend on any weight. The sample has exactly the same distribution as before, but now it is an ordinary differentiable function of $\mu$ and $\sigma$: $\partial z_j/\partial \mu_j = 1$ and $\partial z_j/\partial \sigma_j = \epsilon_j$. Autograd handles the rest.

<div class="ex">
<div class="lab">Worked example</div>

**one reparameterised sample**

The encoder outputs $\mu = (1, 0)$ and $\log\sigma^2 = (0, -2)$. Then $\sigma = \exp(\tfrac12\log\sigma^2) = (e^0, e^{-1}) = (1, 0.368)$.

Draw noise $\epsilon = (0.5, -1.2)$. The code is $z = (1 + 1 \times 0.5,\ 0 + 0.368 \times (-1.2)) = (1.5,\ -0.441)$.

Gradients: $\partial z/\partial\mu = (1, 1)$ and $\partial z/\partial\sigma = \epsilon = (0.5, -1.2)$. If the decoder's loss wants $z_2$ to move up, it can do that by raising $\mu_2$, and (for this $\epsilon$) by shrinking $\sigma_2$.

</div>

This works for any distribution you can write as a differentiable transform of fixed noise. Discrete codes cannot be written that way, so they need other tools: the Gumbel-softmax relaxation, straight-through estimators, or vector quantisation as in VQ-VAE.

## Step 5: the KL term in closed form

For a diagonal Gaussian encoder and a standard normal prior, the KL needs no sampling at all. It splits into one term per latent dimension $j$, with $d$ the number of latent dimensions:

$$\mathrm{KL} = \tfrac12\sum_{j=1}^{d}\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right).$$

Each dimension pays $\tfrac12\mu_j^2$ for moving its mean away from 0, and $\tfrac12(\sigma_j^2 - \log\sigma_j^2 - 1)$ for having a width other than 1. That second part is zero at $\sigma_j^2 = 1$ and grows without bound as $\sigma_j^2 \to 0$, so the encoder can never collapse its cloud to a point for free.

~~~dg-fig 07.01-kl.svg
The per-dimension KL cost. Left: shifting the mean costs a parabola, mu squared over two. Right: as a function of the variance the cost is zero at 1 and climbs steeply as the variance shrinks; the marked points are the worked example below.
~~~

Networks output $\log\sigma^2$ rather than $\sigma$: any real number is a valid log-variance, so no constraint is needed, and $\exp$ keeps the variance positive.

<div class="ex">
<div class="lab">Worked example</div>

**KL for a 2-D code, by hand**

Same encoder output as before: $\mu = (1, 0)$, $\log\sigma^2 = (0, -2)$, so $\sigma^2 = (e^0, e^{-2}) = (1, 0.135)$.

Dimension 1: $\tfrac12(1^2 + 1 - 0 - 1) = \tfrac12 \times 1 = 0.5$ nats. It pays only for its shifted mean.

Dimension 2: $\tfrac12(0^2 + 0.135 - (-2) - 1) = \tfrac12 \times 1.135 = 0.568$ nats. It pays only for its narrow variance.

Total: $0.5 + 0.568 = 1.068$ nats. In words: this image's code carries about one nat of information beyond what the prior already says.

A dimension the decoder does not use drifts to $\mu = 0,\ \sigma = 1$ and costs exactly 0, which is how a VAE switches off latents it does not need.

</div>

## A VAE in code

This trains an 8-dimensional VAE on all 1,797 sklearn digits in about half a minute on one CPU thread. Note where each sum and mean sits: sum over pixels, sum over latents, mean over the batch.

```python
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
torch.set_num_threads(1); torch.manual_seed(0)

X = torch.tensor(load_digits().data / 16.0, dtype=torch.float32)   # (1797, 64), pixels in [0, 1]

class VAE(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 2 * d))
        self.dec = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64))  # pixel logits
    def forward(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=-1)                 # (B, d) each
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterisation
        return self.dec(z), mu, logvar

def neg_elbo(x, logits, mu, logvar):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction='none').sum(-1)  # sum over pixels
    kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(-1)                         # sum over latents
    return (rec + kl).mean(), rec.mean(), kl.mean()                                # mean over batch

model = VAE(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    for xb in X[torch.randperm(len(X))].split(64):
        loss, rec, kl = neg_elbo(xb, *model(xb))
        opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    loss, rec, kl = neg_elbo(X, *model(X))
    print(f"-ELBO {loss:.1f} = rec {rec:.1f} + KL {kl:.1f} nats")
    # -ELBO 24.3 = rec 21.1 + KL 3.2 nats
    mu, logvar = model.enc(X).chunk(2, dim=-1)
    print((0.5 * (mu**2 + logvar.exp() - logvar - 1)).mean(0).numpy().round(2))
    # [0.06 0.8  0.02 0.83 0.58 0.02 0.11 0.8 ]   KL per latent dimension
    samples = torch.sigmoid(model.dec(torch.randn(16, 8)))       # generate: z ~ N(0, I), decode
```

Binary cross-entropy on grey (non-binary) pixels is a common shortcut that is not strictly a normalised density, but trains well. To generate, skip the encoder: sample the prior and decode.

## Reading the result

**Watch the two terms separately, not just their sum.** A healthy run looks like the next figure: the reconstruction term drops quickly, and the KL starts near zero (a freshly initialised encoder outputs clouds close to the prior) and then rises as the codes begin to carry information, before levelling off.

~~~dg-fig 07.01-curves.svg
Training a 2-D VAE on the 0-7 digits for 150 epochs. Reconstruction falls from about 40 to 21 nats per image; the KL climbs from about 0.1 to 2.6 nats as the encoder learns to use its code. The negative ELBO is their sum.
~~~

**Check the KL per dimension.** In the code output above, four of the eight dimensions carry 0.58 to 0.83 nats each, while the other four sit at 0.02 to 0.11: the model decided it only needs about four dimensions for these tiny images and switched the rest off. That is normal. Every dimension at zero is not.

**Look at samples from the prior and at interpolations.** Decode $z \sim \mathcal N(0, I)$; decode points on a straight line between two images' means. Both should look like data and change smoothly. Reconstructions alone are not enough: a plain autoencoder reconstructs well too.

**Compare held-out ELBOs across models.** The negative ELBO upper-bounds $-\log p_\theta(x)$, so lower is better, provided both models use the same likelihood and pixel scaling.

## What goes wrong

The most common bug is a scaling mistake that silently turns the KL into the dominant term. Below, two 8-D VAEs differ only in one line: one sums the reconstruction loss over the 64 pixels, the other averages it.

~~~dg-fig 07.01-collapse.svg
Average KL per latent dimension after training. Summing over pixels (blue), six dimensions carry information. Averaging over pixels shrinks the reconstruction term 64-fold, so the cheapest solution is to copy the prior: every KL bar is about zero and reconstruction gets much worse (27.3 against 18.3 nats).
~~~

With the mean, a nat of reconstruction is worth 1/64 of a nat of KL, so the encoder outputs the prior for every image, $z$ carries nothing, and the decoder learns to output the average digit. For 784-pixel MNIST the factor is 784.

| Symptom | Likely cause | Fix
| --- | --- | --- |
| All samples look like the same grey blob; KL near 0 | Reconstruction averaged over pixels, or a weight on the KL that is too large | Sum over pixels and latents, mean over batch
| KL near 0 even with correct scaling | Posterior collapse: a powerful (e.g. autoregressive) decoder models $x$ without $z$ | KL warm-up from weight 0 to 1, "free bits" (a floor on per-dimension KL), a weaker decoder
| Blurry samples | Gaussian or per-pixel likelihood averages over all plausible images; samples land between modes | Hierarchical latents, perceptual or adversarial losses, or a diffusion decoder
| Loss becomes NaN | Network outputs $\sigma$ directly and it goes negative, or $\exp(\log\sigma^2)$ overflows | Predict $\log\sigma^2$, clamp it to a sane range
| A few dimensions have KL 0 | Model does not need them | Usually fine: automatic dimension selection
| Factors mixed across dimensions | Nothing asks for disentanglement | $\beta$-VAE ($\beta > 1$ on the KL), at a cost in reconstruction

Plain MSE with weight 1 is also a hidden choice: a Gaussian decoder with variance $\sigma_x^2$ gives $-\log p_\theta(x\mid z) = \|x - \hat x\|^2/(2\sigma_x^2) + \text{const}$, so "MSE plus KL" means $\sigma_x^2 = 1/2$. On pixels in $[0,1]$ that is a very noisy decoder, so the KL wins and samples get blurry. Choosing $\sigma_x$ (or learning it) sets the balance between the terms.

## Going deeper

**Where the bound comes from.** For any distribution $q(z\mid x)$, write $\log p_\theta(x) = \log \mathbb E_q\big[p_\theta(x, z)/q(z\mid x)\big]$ and apply Jensen's inequality (the log of an average is at least the average of the log) to get the ELBO. Doing the algebra exactly instead gives an identity:

$$\log p_\theta(x) = \mathrm{ELBO}(x) + \mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big).$$

The left side does not depend on $\phi$, so raising the ELBO through the encoder pulls $q$ toward the true posterior; the bound is tight exactly when they match. Two separate gaps remain in practice: the Gaussian family may be unable to express the true posterior (the *approximation gap*), and one shared encoder network may not find the best Gaussian for each image (the *amortisation gap*).

**Tighter bounds and variants.** The importance-weighted autoencoder (IWAE) averages $K$ samples inside the logarithm, giving a bound that tightens as $K$ grows; it is also the usual way to estimate $\log p_\theta(x)$. A conditional VAE feeds a label to both networks; VQ-VAE swaps the Gaussian code for a learned codebook; hierarchical VAEs (NVAE, VDVAE) stack latent layers and close much of the sample-quality gap.

**Rate and distortion.** The average KL is the *rate* (nats each code carries about its image); reconstruction loss is the *distortion*. A weight $\beta$ on the KL picks a point on that trade-off: $\beta < 1$ sharpens reconstructions, $\beta > 1$ compresses harder.

**Why the KL "fills" the prior.** The average of all the encoder clouds, $q(z) = \frac1n\sum_i q_\phi(z\mid x_i)$, is called the aggregate posterior. Samples look realistic only where $q(z)$ and $p(z)$ agree. In our 2-D run the encoder means have standard deviation about 1.0 per dimension while each cloud has $\sigma \approx 0.30$, so the aggregate roughly matches the prior. Where it does not (holes in the aggregate), prior samples decode badly, which is one source of poor VAE samples.

**The most important modern use.** Latent diffusion (Stable Diffusion) trains an autoencoder with a small KL weight plus perceptual and adversarial losses. It maps a 512x512x3 image to a 64x64x4 latent, 48 times fewer numbers, and runs the whole diffusion process of lessons 07.03 to 07.07 in that latent space. There the VAE is a compressor whose job is sharp reconstructions, and the diffusion model takes over the job of generating codes.

## Key terms

| Term | Meaning
| --- | --- |
| Autoencoder | Encoder plus decoder trained to reproduce the input through a short code
| Latent variable $z$ | A hidden code that describes an example; in a VAE it has a prior $\mathcal N(0, I)$
| Encoder $q_\phi(z\mid x)$ | Network giving a Gaussian (mean and log-variance) over codes for each input
| Decoder $p_\theta(x\mid z)$ | Network giving a distribution over inputs for each code
| ELBO | Reconstruction log-likelihood minus KL to the prior; a lower bound on $\log p_\theta(x)$
| KL divergence | Non-negative measure of how different two distributions are; zero only if identical
| Reparameterisation | Writing $z = \mu + \sigma\odot\epsilon$ so gradients flow through the sample
| Posterior collapse | KL goes to zero and the decoder ignores $z$; codes carry no information

<div class="take">
<div class="lab">Key takeaway</div>

a VAE is an autoencoder whose encoder outputs a Gaussian, trained on the ELBO (reconstruction minus KL to a standard normal prior); reparameterisation makes the sample differentiable, the closed-form KL keeps the code space filled so that decoding prior samples gives realistic data, and correct scaling (sum over pixels and latents, mean over batch) is what keeps the two terms in balance.

</div>
