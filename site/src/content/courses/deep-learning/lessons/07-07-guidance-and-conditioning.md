---
id: "07.07"
course: "deep-learning"
module: "07"
slug: "guidance-and-conditioning"
title: "Guidance and conditioning"
keyIdeas: ["A conditional diffusion model learns $\\epsilon_\\theta(x_t,t,c)$ with the usual noise-prediction loss; the condition enters through embedding addition (classes), cross-attention (text tokens), adaLN (DiT) or channel concatenation (images, masks).", "Bayes' rule on scores gives $\\nabla\\log p(x\\mid y)=\\nabla\\log p(x)+\\nabla\\log p(y\\mid x)$. Scaling the classifier term by $w$ targets $p(x\\mid y)\\,p(y\\mid x)^{w-1}$, which moves samples away from other classes and narrows them; classifier guidance does this with a separate noise-aware classifier.", "Classifier-free guidance trains one model with the condition replaced by a null token 10 to 20 percent of the time and samples with $\\hat\\epsilon=\\epsilon_u+w(\\epsilon_c-\\epsilon_u)$: $w=0$ is unconditional, $w=1$ conditional, $w>1$ extrapolates. The branch difference is an implicit classifier gradient.", "Guidance trades diversity for fidelity: in the toy, class fidelity hits 100% at $w=1$ while the share of one blob climbs from 50% to 100% by $w=6$. FID is best at a modest $w$ for imperfect networks; high $w$ overshoots $\\hat x_0$ and oversaturates, which CFG rescale or dynamic thresholding reduce.", "CFG doubles the network calls per step (batched as one pass of size $2B$), negative prompts are just a different unconditional branch, and without condition dropout the null branch is meaningless, so high $w$ breaks."]
question: "You fine-tune a text-to-image model on a new dataset but forget to keep the 10 percent caption dropout. At sampling time with $w=7.5$, images are noisy and incoherent, while $w=1$ looks reasonable. Explain why, and what you would do short of retraining from scratch."
answer: "CFG computes $\\hat\\epsilon=\\epsilon_u+w(\\epsilon_c-\\epsilon_u)$ and needs $\\epsilon_\\theta(x_t,\\varnothing)$ to be a good unconditional prediction, so that the difference $\\epsilon_c-\\epsilon_u$ approximates $-\\sqrt{1-\\bar\\alpha_t}\\,\\nabla\\log p(c\\mid x_t)$, the implicit classifier gradient. Without caption dropout during fine-tuning, the network never sees the null condition on the new data, and its null-branch behaviour drifts as the shared weights change: the unconditional prediction becomes arbitrary and no longer approximates the unconditional score. At $w=1$ the null branch cancels out exactly ($\\hat\\epsilon=\\epsilon_c$), so samples look fine. At $w=7.5$ the sampler adds 6.5 times the difference between a sensible and a meaningless prediction, which injects large, structured error at every step, mostly at high noise where guidance has the most effect; the result is off-manifold, incoherent images (the same failure the toy model without label dropout shows). The fix is to continue fine-tuning with caption dropout restored (typically 10 to 20 percent of examples get the empty-prompt embedding), which re-teaches the null branch on the new data; a short run is usually enough because the conditional branch is already good. Verify by sweeping $w$ in {1, 3, 5, 7.5} on fixed seeds and checking that quality improves rather than collapses as $w$ rises. As stopgaps: use a low $w$ (1.5 to 3), apply guidance only on a middle interval of noise levels, or take the unconditional branch from the original pre-fine-tuning model, accepting some mismatch because the two networks no longer agree where the caption carries no information."
sources: [{"title": "Ho and Salimans: Classifier-Free Diffusion Guidance", "url": "https://arxiv.org/abs/2207.12598", "note": "Joint conditional and unconditional training, the guidance formula and the FID vs IS trade-off."}, {"title": "Dhariwal and Nichol: Diffusion Models Beat GANs on Image Synthesis", "url": "https://arxiv.org/abs/2105.05233", "note": "Classifier guidance and the noise-aware classifier (section 4)."}, {"title": "Rombach et al.: High-Resolution Image Synthesis with Latent Diffusion Models", "url": "https://arxiv.org/abs/2112.10752", "note": "VAE latent space plus cross-attention conditioning on text."}, {"title": "Sander Dieleman: Guidance, a cheat code for diffusion models", "url": "https://sander.ai/2022/05/26/guidance.html", "note": "Why guidance works, the Bayes-rule view and the temperature analogy."}]
practice: ["classifier-free-guidance", "dit-patchify-and-adaln-zero-block", "transformer-decoder-block"]
tags: ["classifier-free guidance", "cross-attention", "conditioning"]
tools: ["PyTorch"]
---

## What you will learn

- How a diffusion model is told *what* to generate: the condition $c$ (a class, a caption, a mask) and the four standard ways it enters the network.
- Why conditioning alone is often too weak, and how **guidance** strengthens it at sampling time without retraining.
- Classifier guidance from Bayes' rule, and why classifier-free guidance (CFG) replaced it.
- The CFG formula, computed by hand on one pixel and run in PyTorch on a model that trains in about a minute on a CPU.
- What the guidance scale $w$ buys (fidelity) and costs (diversity, oversaturation, twice the compute).

*Before you start:* you need the noise-prediction network and its loss (07.04), the DDIM sampling step (07.05) and the link between noise prediction and the score $\nabla_x\log p$ (07.06). Cross-attention reuses the transformer module's attention.

## From "an image" to "an image of something"

Lesson 07.06 showed that a noise predictor is a score model and that sampling follows the score from noise to data. So far the model generated *any* image from the training distribution; useful models generate an image *of something*. This lesson adds that control in two layers: **conditioning** feeds the request into the network, and **guidance** turns up how strongly the sampler obeys it.

## The idea in one picture

Take a tiny 2-D data set with three classes. Class A is made of two blobs: one sits far from everything else (top left) and one sits in the middle, close to classes B and C. We train one small denoiser that receives the class label as input, then ask it for class A three times, changing only one number, the **guidance scale** $w$.

| Guidance scale $w$ | Samples in class A | A-samples in the far blob | What it means
| --- | --- | --- | --- |
| 0 | 55% | 27% | Label ignored: samples from all classes
| 1 | 100% | 52% | Plain conditional model: both blobs, as in the data
| 4 | 100% | 94% | Guided: the "most unmistakably A" blob wins

~~~dg-fig 07.07-idea.svg
One trained network, one label, three guidance scales. Faint dots are the real data; black dots are generated. Raising w pushes samples away from the regions where A could be confused with B or C.
~~~

At $w=1$ the model reproduces class A faithfully, including the half that lives near the other classes. At $w=4$ nearly every sample is in the far blob: more clearly "A", but the variety inside the class has collapsed. That is the whole lesson in miniature: guidance trades **diversity** (covering everything the class contains) for **fidelity** (matching the request unmistakably).

Two words you will meet throughout:

- The **condition** $c$ is whatever the user supplies: a class id, a caption's token embeddings, a mask.
- The **null condition** $\varnothing$ is a special "no request" input. A model that understands $\varnothing$ can also act as an unconditional model, which is the key to classifier-free guidance.

## Step 1: get the condition into the network

A conditional denoiser learns $\epsilon_\theta(x_t,t,c)$ with exactly the same loss as before (lesson 07.04): predict the noise that was added. Nothing about the forward process or the sampler changes. The only new design decision is how $c$ reaches the network, and that depends on what shape $c$ has.

| Condition | Mechanism | Example
| --- | --- | --- |
| Class label | Learned embedding added to the timestep embedding | ImageNet class-conditional U-Nets
| Text | Cross-attention: image features are queries, text-encoder tokens are keys and values | Latent (Stable) Diffusion
| Class or pooled text in a transformer | adaLN: the condition sets per-channel scale, shift and gate | DiT
| Image-shaped input | Concatenate to $x_t$ along channels | Inpainting (image plus mask), super-resolution
| Spatial control on a frozen model | Trainable copy of the encoder feeding residuals into the U-Net | ControlNet

**Embedding addition** is what our toy uses: the label picks a learned vector that is added to a hidden layer, like the timestep embedding.

**Cross-attention** is the workhorse for text. At each U-Net resolution, flattened image features of shape (HW, C) are queries over the prompt's token embeddings (77 tokens for CLIP-based models), so each image location picks up the words relevant to it: the dog's patch attends to "dog", the background to "beach". Latent diffusion runs all of this on the 64x64x4 VAE latent from lesson 07.01, which makes 512x512 text-to-image affordable.

**adaLN** lets the condition set the scale, shift and gate after each normalisation (DiT). **Concatenation** suits image-shaped conditions: an inpainting model sees the noisy image, the masked photo and the mask stacked as channels.

~~~dg-flow How the condition reaches a text-to-image denoiser at every sampling step.
Prompt text -(text encoder)-> Token embeddings -(keys and values)-> *Cross-attention in each block
Noisy latent x_t -(U-Net features as queries)-> *Cross-attention in each block -> Predicted noise
~~~

## Step 2: why conditioning is not enough

A perfect conditional model samples exactly from $p(x\mid c)$, which is what $w=1$ gave us in the picture. In practice networks under-use the condition (the caption is a weak signal next to the noisy image), so they hedge towards generic images. And users usually want images that are *unmistakably* what they asked for, even at the cost of variety.

**Guidance** fixes this at sampling time. It does not change the training loss of the main model; it changes the direction the sampler steps in.

## Step 3: Bayes' rule on scores

Start from Bayes' rule for the noisy image $x_t$ and a label $y$: $p(x_t\mid y)=p(y\mid x_t)\,p(x_t)/p(y)$. Take the log and the gradient with respect to $x_t$. The term $p(y)$ does not depend on $x_t$, so it disappears:

$$\nabla_{x_t}\log p(x_t\mid y)=\nabla_{x_t}\log p(x_t)+\nabla_{x_t}\log p(y\mid x_t)$$

In words: the conditional score is the unconditional score (lesson 07.06) plus the gradient of a classifier. The classifier term points towards images that look more like class $y$.

Now put a knob on the classifier term. Multiplying it by $w$ is the same as sampling from a **tilted** distribution

$$\tilde p_w(x\mid y)\ \propto\ p(x)\,p(y\mid x)^w\ =\ p(x\mid y)\,p(y\mid x)^{w-1}.$$

At $w=1$ this is the true conditional. For $w>1$, the extra factor $p(y\mid x)^{w-1}$ down-weights every region where the classifier is unsure, which is exactly where the classes overlap.

~~~dg-fig 07.07-bayes.svg
Left: two overlapping classes and the classifier probability they imply. Right: the tilted target for the "dog" class. As w grows it slides away from the cat class and narrows.
~~~

The numbers from the right panel: at $w=1$ the target peaks at $x=0.80$ (the true dog mean) with standard deviation 0.70; at $w=3$ it peaks at 1.00 with standard deviation 0.56; at $w=8$, at 1.20 with 0.49. Guidance moves samples *away from the other class* and makes them *less varied*, like lowering a temperature on the condition.

## Step 4: classifier guidance

Dhariwal and Nichol (2021) used the formula literally. They trained a separate classifier $p_\phi(y\mid x_t,t)$ on *noisy* images at every noise level, and at each sampling step added its gradient with a scale $s$. Because $\epsilon=-\sqrt{1-\bar\alpha_t}\,\nabla\log p$, adding to the score means subtracting from the noise prediction:

$$\hat\epsilon=\epsilon_\theta(x_t)-s\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_\phi(y\mid x_t).$$

Here $s$ plays the role of $w$, and with $s>1$ samples are pushed towards images the classifier is very sure about. It let diffusion beat GANs on ImageNet, but needs an extra classifier trained on every noise level, and a classifier's input gradient is often adversarial: it can raise $p_\phi(y\mid x_t)$ through imperceptible pixel patterns rather than by drawing more of a dog.

## Step 5: classifier-free guidance

Ho and Salimans (2022) removed the classifier. Train *one* network on both tasks: for a random 10 to 20 percent of training examples, replace $c$ by the null token $\varnothing$. This **condition dropout** (label or caption dropout) teaches the same weights to produce both $\epsilon_\theta(x_t,c)$ and $\epsilon_\theta(x_t,\varnothing)$. At sampling time, run the network twice per step and extrapolate:

$$\hat\epsilon=\epsilon_\theta(x_t,\varnothing)+w\,\big(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big)$$

Write $\epsilon_c$ and $\epsilon_u$ for the conditional and unconditional predictions. Then:

- $w=0$ gives $\epsilon_u$, the unconditional model.
- $w=1$ gives $\epsilon_c$, the plain conditional model.
- $w>1$ goes *past* $\epsilon_c$, further away from $\epsilon_u$.

Why does this work? Bayes' rule gives $\log p(c\mid x_t)=\log p(x_t\mid c)-\log p(x_t)+\text{const}$; converting scores to noise predictions, $\epsilon_c-\epsilon_u=-\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p(c\mid x_t)$. The branch difference *is* the gradient of an implicit classifier that comes free with the generative model, so CFG is classifier guidance with $s=w-1$ on top of the conditional model, without a separate network whose gradients can be adversarial.

Watch the convention. The original paper writes $(1+w)\epsilon_c-w\epsilon_u$, so its $w$ equals this lesson's $w$ minus 1. Stable Diffusion's default of $w=7.5$ uses this lesson's convention, as do most libraries; check before copying a number.

~~~dg-flow Each sampling step runs the same network twice and extrapolates from the unconditional prediction.
xt -(with prompt)-> eps cond -(w times difference)-> *Guided eps -> Sampler step
xt -(null prompt)-> eps uncond -> *Guided eps
~~~

<div class="ex">
<div class="lab">Worked example</div>

**guidance on one pixel, with exact numbers**

Data: one-pixel images that are either $-1$ (class "minus") or $+1$ (class "plus"), equally common. At the current step $\bar\alpha_t=0.64$, so $\sqrt{\bar\alpha_t}=0.8$ and $\sqrt{1-\bar\alpha_t}=0.6$. The noisy pixel is $x_t=0.2$ and we ask for "plus".

Conditional branch. If the clean pixel is $+1$, then $x_t=0.8\cdot1+0.6\,\epsilon$, so $\epsilon_c=(0.2-0.8)/0.6=-1.000$. If it were $-1$: $\epsilon_-=(0.2+0.8)/0.6=1.667$.

Unconditional branch. Without the label, the model averages the two cases weighted by how likely each is given $x_t$. The log-odds of plus are $\big((x_t+0.8)^2-(x_t-0.8)^2\big)/(2\cdot0.36)=(1.00-0.36)/0.72=0.889$, so $p(\text{plus}\mid x_t)=1/(1+e^{-0.889})=0.709$. Then $\epsilon_u=0.709\cdot(-1.000)+0.291\cdot1.667=-0.223$.

Implicit classifier check. $\epsilon_c-\epsilon_u=-0.777$. The classifier gradient is $\frac{d}{dx}\log p(\text{plus}\mid x)=(1-0.709)\cdot\frac{2\cdot0.8}{0.36}=0.291\cdot4.444=1.295$, and $-0.6\cdot1.295=-0.777$. The two numbers match, as Step 5 claims.

Guided predictions and the clean-image estimate $\hat x_0=(x_t-0.6\,\hat\epsilon)/0.8$. With $w=0$: $\hat\epsilon=-0.223$, $\hat x_0=0.417$ (a hedge between the classes). With $w=1$: $\hat\epsilon=-1.000$, $\hat x_0=1.000$ (exactly right). With $w=3$: $\hat\epsilon=-0.223+3(-0.777)=-2.554$, $\hat x_0=(0.2+1.532)/0.8=2.165$.

Reading: at $w=3$ the model predicts a pixel of 2.17 although real pixels are only ever $\pm1$. Guidance extrapolates outside the data range. Later steps pull it back when the branches agree, but this overshoot is the seed of the oversaturated colours seen at high $w$.

</div>

## Classifier-free guidance in code

Here is the whole method on the 2-D toy from the first figure: a conditional MLP, 15 percent label dropout, and a DDIM sampler (lesson 07.05) with CFG. Training takes about 80 seconds on one CPU thread.

```python
import math, torch, torch.nn as nn
torch.set_num_threads(1); torch.manual_seed(0)
MU = torch.tensor([[-1.6, 1.2], [0.3, 0.1], [1.6, 1.2], [0.0, -1.6]])  # 4 blobs
LAB = torch.tensor([0, 0, 1, 2])                                     # class 0 owns two blobs
def sample_data(n):
    k = torch.randint(0, 4, (n,)); return MU[k] + 0.28 * torch.randn(n, 2), LAB[k]
def alpha_bar(t):
    return torch.cos(t * math.pi / 2).pow(2).clamp(1e-4, 0.9999)
NULL = 3                                                             # extra id = "no condition"

class Denoiser(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.emb = nn.Embedding(4, h)                                # 3 classes + null token
        self.inp = nn.Linear(2 + 16, h)
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h),
                                 nn.SiLU(), nn.Linear(h, 2))
    def forward(self, x, t, c):
        f = torch.arange(8) + 1.0
        temb = torch.cat([torch.sin(t[:, None] * f * 3), torch.cos(t[:, None] * f * 3)], 1)
        return self.net(self.inp(torch.cat([x, temb], 1)) + self.emb(c))  # label embedding added

model = Denoiser(); opt = torch.optim.Adam(model.parameters(), 2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 6000)
for step in range(6000):
    x0, c = sample_data(1024)
    c = torch.where(torch.rand(1024) < 0.15, torch.full_like(c, NULL), c)  # label dropout
    t = torch.rand(1024); a = alpha_bar(t)[:, None]
    eps = torch.randn_like(x0); xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    loss = ((model(xt, t, c) - eps) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
print(round(loss.item(), 3))                                         # 0.296

@torch.no_grad()
def sample_cfg(model, n, cls, w, steps=50, seed=0):
    g = torch.Generator().manual_seed(seed); x = torch.randn(n, 2, generator=g)
    ts = torch.linspace(1, 0, steps + 1)
    cond, null = torch.full((n,), cls), torch.full((n,), NULL)
    for t_now, t_next in zip(ts[:-1], ts[1:]):
        a, a_next = alpha_bar(t_now), alpha_bar(t_next)
        e_c, e_u = model(torch.cat([x, x]), t_now.repeat(2 * n),
                         torch.cat([cond, null])).chunk(2)           # one batched pass of size 2n
        e = e_u + w * (e_c - e_u)                                    # classifier-free guidance
        x0_hat = (x - (1 - a).sqrt() * e) / a.sqrt()
        x = a_next.sqrt() * x0_hat + (1 - a_next).sqrt() * e         # DDIM step
    return x

for w in [0, 1, 3, 7.5]:
    x = sample_cfg(model, 4000, cls=0, w=w)
    d = torch.cdist(x, MU); near = d.argmin(1)
    print(w, (LAB[near] == 0).float().mean().item(),                 # fraction in class 0
          (near == 0).float().mean().item(),                         # fraction in the far blob
          (d.min(1).values < 0.84).float().mean().item())            # fraction on the data
# 0    0.535  0.251  0.975
# 1    0.999  0.506  0.976
# 3    1.000  0.871  0.969
# 7.5  1.000  1.000  0.805
```

The null condition is just one more embedding id (for text models, usually the empty prompt's embedding), the two branches share one batch of size $2n$, and guidance itself is one line inside an ordinary sampler.

## Reading the result: fidelity, diversity and the best $w$

There is no single correct $w$, so measure several things at once. The next figure sweeps $w$ for the trained toy and for an exact-score model of the same data.

~~~dg-fig 07.07-tradeoff.svg
Left: the trained net reaches 100% class fidelity by w = 1; beyond that the blob split drifts from the true 50/50 and, past w = 5, samples leave the data. Right: distance to the real class is smallest at w = 1 for an exact model, and at w = 2 for a simulated model that uses the label only half as strongly.
~~~

What to notice:

- **Fidelity saturates early:** 100 percent of samples are in class A from $w=1$ on.
- **Diversity falls steadily:** the far-blob share goes 50% at $w=1$, 77% at $w=2$, 88% at $w=3$ and 100% by $w=6$. Half the class disappears.
- **Samples eventually leave the data:** the fraction within three standard deviations of a real blob drops from 98% at $w=1$ to 81% at $w=7.5$ and 67% at $w=10$.
- **The best $w$ depends on the model.** The Fréchet distance (the 2-D analogue of FID) to real class-A samples is about zero at $w=1$ for the exact and trained models. For a simulated weak model whose conditional branch is $0.5\,\epsilon_c+0.5\,\epsilon_u$, the minimum moves to $w=2$: guidance compensates for a network that under-uses its condition.

Real image networks are imperfect in this way, so FID typically improves from $w=1$ to a modest value and then worsens, while precision and prompt alignment keep rising. Practitioners often run well above the FID-optimal scale because people prefer vivid, on-prompt images: choose $w$ for the use case, and report it with any sample or score.

For a real model: sample a fixed grid of prompts and seeds at several $w$, measure FID (realism and coverage) and CLIP score (prompt alignment), then inspect the images for saturation and repeated compositions.

## What goes wrong

**Oversaturation.** Large $w$ inflates the predicted clean image $\hat x_0$, as the hand example showed. In a 64-"pixel" toy with exact scores, the largest predicted pixel during sampling peaks at 0.87 for $w=1$, 1.59 for $w=3$ and 3.96 for $w=7.5$, against a real range of $[-1,1]$.

~~~dg-fig 07.07-oversat.svg
Largest pixel in the predicted clean image over 50 DDIM steps (64-pixel toy, exact scores). At w = 7.5 early predictions reach about four times the real range; CFG rescale with φ = 0.7 cuts the peak to 1.83.
~~~

Exact scores pull the final image back into range here; real networks recover only partly, which shows up as burnt colours and harsh contrast. **CFG rescale** (Lin et al., 2023) rescales the guided prediction so its standard deviation matches the conditional one, then blends the result with the unrescaled prediction using a factor $\phi\approx0.7$. **Dynamic thresholding** (Imagen) instead clips $\hat x_0$ at a high percentile of its absolute values and rescales.

**No condition dropout.** A model that never saw $\varnothing$ during training has a meaningless $\epsilon_u$. At $w=1$ the null branch cancels out, so samples look fine, but at high $w$ the sampler extrapolates along a random direction.

~~~dg-fig 07.07-nodrop.svg
Class-A samples from the model trained with 15% label dropout (w = 7.5: 81% on class A) and from an identical model trained with none (w = 1: 98%; w = 7.5: 0%, every sample thrown off the plot). Points beyond the axes are drawn at the edge.
~~~

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Images ignore parts of the prompt | $w$ too low, or weak text conditioning | Raise $w$; better text encoder; more cross-attention capacity
| Every sample looks the same | $w$ too high: diversity collapse | Lower $w$; guidance only on a middle interval of noise levels
| Burnt colours, harsh contrast | Large $w$ inflates $\hat x_0$ | CFG rescale ($\phi\approx0.7$), dynamic thresholding, lower $w$
| Fine at $w=1$, garbage at high $w$ | Null branch never trained (no dropout) | Fine-tune with 10 to 20% condition dropout
| Numbers from a paper do not reproduce | $w$ convention differs by 1 | Check whether code uses $(1+w)\epsilon_c-w\epsilon_u$
| Sampling twice as slow as expected | Two network passes per step | Batch both branches as one pass of size $2B$; guidance-distilled models
| An unwanted concept keeps appearing | The null branch is generic | Negative prompt: use the unwanted concept's embedding in place of $\varnothing$

## Going deeper

**CFG does not sample the tilted distribution exactly.** The guided score $\nabla\log p(x_t)+w\nabla\log p(c\mid x_t)$ is the score of $p_t(x_t)\,p_t(c\mid x_t)^w$ at each noise level separately, but those noisy tilted marginals are not the result of noising the clean tilted distribution $p_0(x)p_0(c\mid x)^w$. So guided sampling reverses no single forward process: CFG is a heuristic whose *direction* is right, and its output depends somewhat on the sampler and step count.

**Guidance acts mostly at high noise.** At low noise the noisy image already determines its class, so $p(c\mid x_t)$ is near 0 or 1, its gradient is near zero, and $\epsilon_c\approx\epsilon_u$. The oversaturation figure shows this: all curves meet in the last 20 steps. Applying guidance only on a limited interval of noise levels keeps most of the benefit with less diversity loss. Interval guidance (Kynkäänniemi et al., 2024) builds on this observation.

**Cross-attention in numbers.** For a 64x64 latent at the top U-Net level, flattened features have HW = 4096 query positions; the CLIP prompt gives 77 keys and values. The attention matrix is 4096 x 77, cheap compared with self-attention's 4096 x 4096, which is why text conditioning adds little cost per layer.

**Why the null branch is trained jointly.** The two branches must agree wherever the condition carries no information; joint training with dropout gives that, at the cost of a little conditional training signal, which is why 10 to 20 percent rather than 50 is standard.

## Key terms

| Term | Meaning
| --- | --- |
| Condition $c$ | The user's request (class, caption, mask) given to the denoiser as an input
| Null condition $\varnothing$ | A special "no request" input that turns the same network into an unconditional model
| Cross-attention | Image features as queries attending over text tokens as keys and values
| adaLN | Condition-dependent scale, shift and gate after each normalisation layer (DiT)
| Classifier guidance | Adding a scaled gradient of a noise-aware classifier to the score
| Classifier-free guidance | $\hat\epsilon=\epsilon_u+w(\epsilon_c-\epsilon_u)$ using one jointly trained network
| Guidance scale $w$ | How far to extrapolate past the conditional prediction; 1 means no extra guidance
| Condition dropout | Replacing $c$ by $\varnothing$ for 10 to 20% of training examples
| CFG rescale | Matching the guided prediction's standard deviation to the conditional one to curb oversaturation

<div class="take">
<div class="lab">Key takeaway</div>

condition the denoiser through embeddings, cross-attention, adaLN or concatenation, then strengthen the condition at sampling time with classifier-free guidance, which extrapolates from an unconditional to a conditional prediction of the same network; the scale $w$ trades diversity and naturalness for prompt fidelity, and it only works if the null branch was trained with condition dropout.

</div>
