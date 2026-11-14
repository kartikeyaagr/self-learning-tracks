---
id: "07.07"
course: "deep-learning"
module: "07"
slug: "guidance-and-conditioning"
title: "Guidance and conditioning"
keyIdeas: ["A conditional diffusion model learns $\\epsilon_\\theta(x_t,t,c)$; the condition enters through embedding addition (classes), cross-attention (text tokens), adaptive normalisation (DiT) or input concatenation (images, masks).", "Bayes gives $\\nabla\\log p(x\\mid y)=\\nabla\\log p(x)+\\nabla\\log p(y\\mid x)$, so classifier guidance adds a scaled gradient from a classifier trained on noisy inputs.", "Classifier-free guidance trains one model with the condition dropped 10 to 20 percent of the time and samples with $\\hat\\epsilon=\\epsilon_u+w(\\epsilon_c-\\epsilon_u)$; $w>1$ sharpens adherence to $c$.", "Guidance trades diversity for fidelity: FID usually bottoms out at a small $w$, and high $w$ oversaturates; rescaling the guided prediction's standard deviation reduces the artefacts.", "CFG doubles the network calls per step (batched as one pass of size $2B$), and negative prompts are just a different unconditional branch."]
question: "You fine-tune a text-to-image model on a new dataset but forget to keep the 10 percent caption dropout. At sampling time with $w=7.5$, images are noisy and incoherent, while $w=1$ looks reasonable. Explain why, and what you would do short of retraining from scratch."
answer: "CFG needs $\\epsilon_\\theta(x_t,\\varnothing)$ to be a good unconditional prediction. Without caption dropout during fine-tuning, the network never sees the null condition on the new data, and its null-branch behaviour drifts as the shared weights change: the unconditional prediction becomes arbitrary and no longer approximates the unconditional score. At $w=1$ the null branch cancels out ($\\hat\\epsilon=\\epsilon_c$), so samples look fine. At $w=7.5$ the sampler extrapolates 6.5 times the difference between a sensible and a meaningless prediction, which injects large, structured error at every step. The fix is to continue fine-tuning with caption dropout restored (typically 10 to 20 percent), which re-teaches the null branch on the new data. As a stopgap, use a low $w$, or take the unconditional branch from the original pre-fine-tuning model, accepting some mismatch."
sources: [{"title": "Ho and Salimans: Classifier-Free Diffusion Guidance", "url": "https://arxiv.org/abs/2207.12598", "note": "Joint conditional and unconditional training, the guidance formula and the FID vs IS trade-off."}, {"title": "Dhariwal and Nichol: Diffusion Models Beat GANs on Image Synthesis", "url": "https://arxiv.org/abs/2105.05233", "note": "Classifier guidance and the noise-aware classifier (section 4)."}, {"title": "Rombach et al.: High-Resolution Image Synthesis with Latent Diffusion Models", "url": "https://arxiv.org/abs/2112.10752", "note": "VAE latent space plus cross-attention conditioning on text."}, {"title": "Sander Dieleman: Guidance, a cheat code for diffusion models", "url": "https://sander.ai/2022/05/26/guidance.html", "note": "Why guidance works, the Bayes-rule view and the temperature analogy."}]
practice: ["classifier-free-guidance", "dit-patchify-and-adaln-zero-block", "transformer-decoder-block"]
tags: ["classifier-free guidance", "cross-attention", "conditioning"]
tools: ["PyTorch"]
---

## Conditioning: getting the prompt into the network

An unconditional model generates any image from the training distribution. Useful models generate an image *of something*: a class label, a caption, a sketch, the unmasked part of a photo. A conditional model simply learns $\epsilon_\theta(x_t, t, c)$ with the same loss as before (lesson 07.04); the work is in how $c$ reaches the network.

| Condition | Mechanism | Example
| --- | --- | --- |
| Class label | Learned embedding added to the timestep embedding | ImageNet class-conditional U-Nets
| Text | Cross-attention: image features are queries, text-encoder tokens are keys and values | Latent (Stable) Diffusion
| Class or pooled text in a transformer | adaLN: the condition sets per-channel scale, shift and gate | DiT
| Image-shaped input | Concatenate to $x_t$ along channels | Inpainting (image plus mask), super-resolution
| Spatial control on a frozen model | Trainable copy of the encoder feeding residuals into the U-Net | ControlNet

Cross-attention is the workhorse for text. At each U-Net resolution, flattened image features of shape (HW, C) attend over the prompt's token embeddings (77 tokens for CLIP-based models), so each image location can pick up the words relevant to it. Latent diffusion runs all of this on the 64x64x4 VAE latent from lesson 07.01, which makes 512x512 text-to-image affordable.

## Classifier guidance

Conditioning alone often gives images that only loosely match $c$. **Guidance** strengthens the condition at sampling time. Bayes' rule on scores gives

$$\nabla_{x_t}\log p(x_t\mid y)=\nabla_{x_t}\log p(x_t)+\nabla_{x_t}\log p(y\mid x_t)$$

The first term is the unconditional score (lesson 07.06). Dhariwal and Nichol (2021) trained a separate classifier $p_\phi(y\mid x_t, t)$ on noisy images and added its gradient with a scale $s$: in $\epsilon$ form, $\hat\epsilon=\epsilon_\theta(x_t)-s\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_\phi(y\mid x_t)$. With $s>1$ this samples from a sharpened distribution, roughly $p(x)\,p(y\mid x)^s$, which pushes samples towards images the classifier is very sure about. It let diffusion beat GANs on ImageNet, but needs a classifier trained on every noise level, and its gradients can be adversarial rather than meaningful.

## Classifier-free guidance

Ho and Salimans (2022) removed the classifier. Train a single network on both tasks by replacing $c$ with a null token $\varnothing$ for a random 10 to 20 percent of training examples. At sampling time, run it twice per step and extrapolate:

$$\hat\epsilon=\epsilon_\theta(x_t,\varnothing)+w\,\big(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big)$$

The difference $\epsilon_c-\epsilon_u$ is, up to scale, the score of an implicit classifier $\nabla\log p(c\mid x_t)$, so this is classifier guidance where the classifier comes free from the generative model. $w=0$ is unconditional, $w=1$ is the plain conditional model, and $w>1$ moves past the conditional prediction, away from the unconditional one. (The original paper writes $(1+w)\epsilon_c-w\epsilon_u$, so its $w$ is this $w$ minus 1; check the convention in any code you read.) Stable Diffusion's default is $w=7.5$.

~~~dg-flow Each sampling step runs the same network twice and extrapolates from the unconditional prediction.
xt -(with prompt)-> eps cond -(w times difference)-> *Guided eps -> Sampler step
xt -(null prompt)-> eps uncond -> *Guided eps
~~~

```python
def cfg_eps(model, xt, t, cond, null, w):
    x2 = torch.cat([xt, xt]); c2 = torch.cat([cond, null])
    e_c, e_u = model(x2, t.repeat(2), c2).chunk(2)   # one batched pass of size 2B
    return e_u + w * (e_c - e_u)
```

<div class="ex">
<div class="lab">Worked example</div>

**guidance on one pixel**

At some step the conditional prediction is $\epsilon_c=0.50$ and the unconditional one is $\epsilon_u=0.30$.

With $w=1$: $\hat\epsilon=0.50$. With $w=7.5$: $\hat\epsilon=0.30+7.5\cdot0.20=1.80$.

The guided noise estimate lies far outside the range either branch predicts. Applied over every pixel and step, that extrapolation is what makes guided images vivid, and at high $w$, oversaturated.

</div>

## Costs and failure modes

- **Diversity loss.** Guidance concentrates mass on the most "typical" images for the prompt. FID typically improves from $w=1$ to a modest value, often well below the scales used in practice, and then worsens, while precision and prompt alignment keep rising. Choose $w$ for the use case, not by FID alone.
- **Oversaturation.** Large $w$ inflates the magnitude of the predicted $\hat x_0$, giving burnt colours and harsh contrast. **CFG rescale** (Lin et al., 2023) rescales the guided prediction so its standard deviation matches the conditional one, then blends with factor $\phi\approx0.7$; dynamic thresholding (Imagen) clips $\hat x_0$ by a high percentile instead.
- **Compute.** Two network evaluations per step. Batching them as one pass of size $2B$ keeps latency close to one call on an under-utilised GPU, but the FLOPs double. Distilled models fold guidance into the weights to remove the second pass.
- **Negative prompts.** Replace $\varnothing$ by the embedding of an unwanted concept; the extrapolation then moves away from it instead of away from the generic image.
- **Dropout matters.** A model never trained with null conditions has a meaningless $\epsilon_u$ and CFG produces garbage; this is a training-time decision.

<div class="take">
<div class="lab">Key takeaway</div>

condition the denoiser through embeddings, cross-attention, adaLN or concatenation, then strengthen the condition at sampling time with classifier-free guidance, which extrapolates from an unconditional to a conditional prediction of the same network; the scale $w$ trades diversity and naturalness for prompt fidelity.

</div>
