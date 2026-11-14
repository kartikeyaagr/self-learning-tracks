---
id: "15.05"
course: "ml-maths-course"
module: "15"
slug: "information-in-ml-losses"
title: "Information in ML losses"
keyIdeas: ["Minimising average cross-entropy on data is maximum likelihood, which is the same as minimising forward KL from the data distribution to the model.", "Label smoothing and distillation replace one-hot targets with soft ones, changing the loss's minimum and how confident the model becomes.", "The VAE objective is a reconstruction term minus a KL term, which can be read as distortion plus the rate (bits) used to describe the latent.", "Contrastive losses such as InfoNCE give a lower bound on mutual information that can never exceed $\\log N$ for $N$ candidates."]
question: "You train an image classifier with 1,000 classes using label smoothing $\\varepsilon = 0.1$ (the target puts $0.9 + 0.1/1000$ on the true class and $0.1/1000$ on each other class). (a) What is the lowest possible training loss per example, in nats? (b) Why does the loss not reach zero even for a perfect model? (c) A contrastive model is trained with InfoNCE using batches of 256. Its loss reaches 1.0 nats. What lower bound on mutual information does this give, and what is the largest bound this setup could ever certify?"
answer: "(a) The minimum of cross-entropy over $q$ is the entropy of the target, reached at $q$ equal to the target. The target has one entry $0.9001$ and 999 entries $0.0001$, so $H = -0.9001\\ln 0.9001 - 999 \\times 0.0001 \\ln 0.0001 \\approx 0.0948 + 0.9201 \\approx 1.01$ nats. (b) With soft targets, the model's best prediction is the target itself, and cross-entropy then equals the target's entropy, not zero; a loss near 1.0 is the floor, not a failure. (c) The bound is $I \\ge \\ln N - L = \\ln 256 - 1.0 \\approx 5.55 - 1.0 = 4.55$ nats. The largest value it can ever certify is $\\ln 256 \\approx 5.55$ nats, reached at zero loss, however much information the two views truly share."
sources: [{"title": "Mueller, Kornblith and Hinton, When Does Label Smoothing Help? (2019)", "url": "https://arxiv.org/abs/1906.02629", "note": "Effects of label smoothing on calibration and distillation."}, {"title": "Hinton, Vinyals and Dean, Distilling the Knowledge in a Neural Network (2015)", "url": "https://arxiv.org/abs/1503.02531", "note": "Soft targets with temperature."}, {"title": "van den Oord, Li and Vinyals, Representation Learning with Contrastive Predictive Coding (2018)", "url": "https://arxiv.org/abs/1807.03748", "note": "InfoNCE and its mutual-information bound."}, {"title": "Kingma and Welling, Auto-Encoding Variational Bayes (2013)", "url": "https://arxiv.org/abs/1312.6114", "note": "The evidence lower bound used by VAEs."}]
practice: ["cross-entropy-loss", "symmetric-clip-contrastive-loss", "vae-training-step", "dpo-closed-form-loss"]
tags: ["maximum likelihood", "label smoothing", "distillation", "ELBO", "InfoNCE"]
tools: ["PyTorch"]
---

## Maximum likelihood is cross-entropy is KL

Let $p_{\text{data}}$ be the data distribution and $q_\theta$ the model. The expected negative log-likelihood is
$\mathbb{E}_{x\sim p_{\text{data}}}[-\log q_\theta(x)] = H(p_{\text{data}}, q_\theta) = H(p_{\text{data}}) + D_{\mathrm{KL}}(p_{\text{data}}\,\|\,q_\theta)$.
The first term is fixed, so maximum likelihood, minimum cross-entropy and minimum forward KL are the same optimisation. This explains two common behaviours: models trained by likelihood are mass-covering (lesson 15.03), and the loss floor is the entropy of the data, not zero. A language model cannot reach zero loss on text that is genuinely uncertain.

## Soft targets: label smoothing and distillation

Label smoothing mixes the one-hot target with a uniform distribution: $p_\varepsilon = (1-\varepsilon)\,\text{one-hot} + \varepsilon/K$. The loss becomes
$H(p_\varepsilon, q) = (1-\varepsilon)\,H(\text{one-hot}, q) + \varepsilon\,H(u, q)$,
where $u$ is uniform. The second term penalises any class probability approaching zero, so logits cannot grow without bound. This usually improves calibration and sometimes accuracy. It also squeezes examples of a class into tighter clusters in the penultimate layer, which removes information about how classes relate; Mueller and colleagues found that teachers trained with label smoothing are worse for distillation.

Distillation uses the teacher's probabilities as targets, softened with a temperature $T$: $p_T = \mathrm{softmax}(z/T)$. A higher $T$ reveals more about the teacher's view of similar classes (a husky looks somewhat like a wolf). Gradients of the softened loss scale as $1/T^2$, so the loss is multiplied by $T^2$ to keep its weight comparable to the hard-label loss.

<div class="ex">
<div class="lab">Worked example</div>

**the loss that would not fall**

An engineer added label smoothing with $\varepsilon = 0.1$ to a 1,000-class model and saw the training loss stall near 1.0 nats instead of falling toward 0.1 as before. Nothing was wrong: with smoothed targets, the minimum loss is the target entropy, about 1.01 nats here. Tracking accuracy and the unsmoothed validation loss alongside the training loss avoided a pointless search for a bug.

</div>

## The VAE objective as rate and distortion

A VAE maximises the evidence lower bound
$\mathcal{L} = \mathbb{E}_{q(z\mid x)}[\log p(x\mid z)] - D_{\mathrm{KL}}(q(z\mid x)\,\|\,p(z))$.
In information terms, the first term is (minus) the distortion: how well $x$ is reconstructed from the code $z$. The KL term is the rate: on average, the number of nats needed to communicate $z$ beyond what the prior already predicts. Weighting the KL term by $\beta$ trades rate against distortion. With $\beta > 1$, latents carry less information and tend to be smoother; with too strong a KL weight, or a very powerful decoder, the rate falls to zero and the latent is ignored (posterior collapse).

~~~dg-flow The VAE as a communication system.
x -(encoder q(z given x))-> Code z -(rate: KL to prior)-> *Channel -(decoder p(x given z))-> Reconstruction -(distortion: -log p(x given z))-> Loss
~~~

## Contrastive losses and mutual information

InfoNCE scores one positive pair against $N - 1$ negatives with a softmax:
$L = -\mathbb{E}\Big[\log \frac{e^{s(x, y^+)}}{\sum_{j=1}^{N} e^{s(x, y_j)}}\Big]$.
It is a cross-entropy over $N$ candidates. Its minimum gives a lower bound $I(X;Y) \ge \log N - L$. Two practical consequences follow. The bound can never exceed $\log N$, so certifying large mutual information requires large batches or many negatives. And a lower loss does not always mean better representations; the loss is a training signal, not a precise measurement of information. CLIP's symmetric loss is InfoNCE applied in both directions.

## Preference losses

Direct preference optimisation and similar methods can be derived from a reward objective with a KL penalty to a reference model. The optimal policy under that objective is the reference reweighted by the exponentiated reward, and the loss compares log-probability ratios between preferred and rejected responses. The KL appears implicitly in those ratios.

## Proper scoring

Log loss is a strictly proper scoring rule: its expected value is minimised only by reporting the true probabilities. That is why it encourages calibrated predictions, and why models trained on it can be calibrated with small post-hoc corrections, while accuracy alone cannot.

## Checking your understanding

Three quick checks tie this lesson together. A binary classifier that always predicts the base rate of 10% has log loss equal to the label entropy, about 0.325 nats. Label smoothing with $\varepsilon = 0.1$ on two classes raises the loss floor from 0 to the entropy of $(0.95, 0.05)$, about 0.199 nats. An InfoNCE loss of $\ln N$ means the model is at chance among $N$ candidates and certifies no information at all.

<div class="take">
<div class="lab">Key takeaway</div>

most ML losses are information quantities: likelihood is cross-entropy and forward KL, soft targets change the loss floor, the VAE trades rate against distortion, and InfoNCE bounds mutual information by at most $\log N$.

</div>
