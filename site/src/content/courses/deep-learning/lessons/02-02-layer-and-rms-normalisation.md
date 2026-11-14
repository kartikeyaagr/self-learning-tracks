---
id: "02.02"
course: "deep-learning"
module: "02"
slug: "layer-and-rms-normalisation"
title: "Layer and RMS normalisation"
keyIdeas: ["**LayerNorm** standardises each example (each token in a Transformer) over its own $D$ features, then applies a learned gain $\\gamma$ and bias $\\beta$. It never looks at other examples, so it behaves identically in training and inference and does not depend on batch size or padding.", "**RMSNorm** drops the mean subtraction and the bias: $y = g \\odot x / \\sqrt{\\mathrm{mean}(x^2) + \\epsilon}$. It is cheaper, loses only invariance to a constant shift, and works as well in large language models, which is why most open LLMs use it.", "The backward pass of a normalisation layer is not elementwise: the gradient is projected to remove the components along the mean and along the normalised vector itself, then divided by the input's scale. Multiplying the input by $c$ therefore divides its gradient by $c$.", "GroupNorm is the image-model relative: statistics over groups of channels and the spatial positions of one image, the default when batches are small (diffusion U-Nets, detection).", "Pre-LN Transformers (norm inside the residual branch, plus one final norm) keep a clean identity path and train stably without long warmup; Post-LN puts large gradients near the output at initialisation and needs warmup."]
question: "You switch a Transformer from LayerNorm to BatchNorm over the batch and sequence dimensions to \"normalise more data at once\". Training loss looks fine, but generation quality collapses and padded batches behave differently from unpadded ones. Explain both problems and why LayerNorm or RMSNorm avoids them."
answer: "BatchNorm computes each feature's statistics over all tokens in the batch, including padding positions and tokens from other sequences. With padding, the pad embeddings enter the mean and variance, so the same real sequence gets different normalisation depending on how much padding its batch contains: outputs depend on batch composition, which is exactly the scatter you see when one example is placed in different batches. In a causal language model it is worse: statistics pooled over the sequence let information from later tokens leak into earlier positions during training, so the model can lean on a signal that does not exist at generation time. At generation time you decode one token at a time, often with batch size one, so the model either uses training-mode statistics of a single token (degenerate: with one value per feature the normalised output is 0) or running averages accumulated over full teacher-forced batches, which do not match the distribution of incrementally generated states. Training loss looks fine because training always runs with full, similar batches. LayerNorm and RMSNorm compute statistics over the $D$ features of each token separately. A token's output depends only on its own hidden state, so padding, batch size and decoding mode cannot change it, nothing leaks between positions, and training and inference run the identical computation. You can verify this by running one sequence alone and inside a padded batch and checking that its outputs match to floating-point tolerance."
sources: [{"title": "Ba, Kiros, Hinton: Layer Normalization", "url": "https://arxiv.org/abs/1607.06450", "note": "Motivation for per-example statistics and the comparison with batch norm in RNNs."}, {"title": "Zhang & Sennrich: Root Mean Square Layer Normalization", "url": "https://arxiv.org/abs/1910.07467", "note": "RMSNorm's definition, the re-centring ablation and speed measurements."}, {"title": "Xiong et al.: On Layer Normalization in the Transformer Architecture", "url": "https://arxiv.org/abs/2002.04745", "note": "Why Post-LN needs warmup and Pre-LN does not, via gradient norms at init."}, {"title": "PyTorch docs: LayerNorm", "url": "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html", "note": "normalized_shape semantics and elementwise affine parameters."}]
practice: ["layer-normalization", "layer-norm", "rmsnorm-forward-pass", "norm-gated-linear-transform", "layernorm-backward"]
tags: ["layer norm", "RMSNorm", "Pre-LN"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- Why BatchNorm is awkward for sequence models, and how LayerNorm fixes it by normalising each example over its own features.
- How to compute LayerNorm and RMSNorm by hand on a four-number token, and what the learned scale and shift do.
- Why RMSNorm (no mean, no shift) is the default in most open large language models.
- How gradients flow back through a normalisation layer, and the scale invariance that falls out of it.
- Where to put the norm in a Transformer block (Pre-LN versus Post-LN) and why it decides whether deep stacks train without warmup.

*Before you start:* BatchNorm and its train/eval difference (02.01), backpropagation (module 01) and residual connections. You only need to know that a Transformer processes a sequence of token vectors of width $D$.

## Picking up from BatchNorm

Lesson 02.01 showed that rescaling activations to zero mean and unit variance makes deep networks much easier to train. BatchNorm gets those statistics by pooling each feature over the batch, which has three side effects: an example's output depends on its batch-mates, training and inference compute different things, and small batches give noisy statistics. Sequence models hit all three: sentences are padded to equal length, generation runs one token at a time, and large models fit only a few sequences per device. This lesson's layers keep the normalising and drop the batch.

## The idea in one picture

Take one token's hidden vector with $D = 4$ features: $x = [3, -1, 1, 1]$. **Layer normalisation** (LayerNorm) uses only these four numbers: it subtracts their mean and divides by their standard deviation. **RMS normalisation** (RMSNorm) skips the mean and only divides by the root mean square.

| | feature 1 | feature 2 | feature 3 | feature 4
| --- | --- | --- | --- | --- |
| raw token $x$ | 3 | -1 | 1 | 1
| LayerNorm output | 1.41 | -1.41 | 0 | 0
| RMSNorm output | 1.73 | -0.58 | 0.58 | 0.58

~~~dg-fig 02.02-token.svg
One token before and after normalisation. LayerNorm centres first, so the two features that sat exactly at the mean (value 1) become 0. RMSNorm only rescales, so every feature keeps its sign and the ratios between features are unchanged.
~~~

In plain words: both layers bring the token to a standard size (LayerNorm output has mean 0 and variance 1; RMSNorm output has root mean square 1), so the next layer sees inputs of a predictable scale however large or small the previous layer made them. Neither looked at any other token, which is what makes them independent of batch size, padding and decoding mode.

A **token** is one position in a sequence; its **hidden state** is a vector of $D$ numbers, the **features** (or model width). The learned **gain** $\gamma$ and **bias** $\beta$, one per feature, let the network rescale and shift the result if that helps.

## Step 1: LayerNorm, the formula

For a single vector $x \in \mathbb{R}^D$, LayerNorm computes the mean and the variance over its $D$ entries, standardises, then applies a per-feature affine map:

$$\mu = \frac{1}{D}\sum_{i=1}^{D} x_i, \qquad \sigma^2 = \frac{1}{D}\sum_{i=1}^{D} (x_i - \mu)^2, \qquad y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta.$$

What each symbol is:

- $\mu$ and $\sigma^2$ are scalars, one pair per token, recomputed every call; nothing is stored.
- $\epsilon$ is a small constant (PyTorch default $10^{-5}$) that prevents division by zero for a nearly constant token.
- $\gamma, \beta \in \mathbb{R}^D$ are learned and shared by every token, starting at $\gamma = 1$, $\beta = 0$; $\odot$ is elementwise multiplication.
- The variance divides by $D$, not $D - 1$ (the "biased" variance), which matters when you check your code against a library.

Number check with $x = [3, -1, 1, 1]$: $\mu = 4/4 = 1$; the deviations are $[2, -2, 0, 0]$; $\sigma^2 = (4 + 4 + 0 + 0)/4 = 2$; dividing the deviations by $\sqrt 2 \approx 1.414$ gives $[1.414, -1.414, 0, 0]$, the table row above.

Because the statistics belong to the token, there are no running averages and no `train()` versus `eval()` difference, and a batch of one gives the same result as a batch of a thousand.

~~~dg-fig 02.02-batchdep.svg
One fixed 16-feature example placed in 120 random batches of each size. Its BatchNorm output for feature 1 depends on its batch-mates (with batch size 2 it is always plus or minus 1); its LayerNorm output is the same value, -0.42, every time.
~~~

With batch size 2, BatchNorm can only output $\pm 1$ per feature (two numbers standardised against each other). Larger batches shrink the scatter (standard deviation 0.69 at size 4, 0.19 at 32, 0.09 at 128 in this run) but never remove it.

## Step 2: which axes get averaged

In a Transformer the hidden states have shape (batch $N$, sequence $T$, features $D$). LayerNorm computes one mean and variance for each of the $N \times T$ tokens, over its $D$ features. In PyTorch, `nn.LayerNorm(D)` normalises over the last dimension only (the `normalized_shape` argument).

~~~dg-layers What gets averaged together, for an activation tensor of shape (batch N, sequence T, features D).
BatchNorm | one statistic per feature, pooled over all N × T positions
*LayerNorm | one statistic per (n, t) token, pooled over its D features
RMSNorm | same as LayerNorm but only the root mean square, no centring
~~~

For images, shaped $(N, C, H, W)$, **GroupNorm** splits the channels into groups (often 32) and normalises each group over its channels and all spatial positions of one image. It is batch-independent, which is why diffusion U-Nets and detection models, with small per-device batches, use it. With one group it equals LayerNorm over $(C, H, W)$; with one channel per group it is **InstanceNorm**.

| Norm | Statistics over | Batch-dependent | Typical home
| --- | --- | --- | --- |
| BatchNorm | N, H, W per channel | yes | CNN classifiers
| LayerNorm | features of one example | no | Transformers, RNNs
| RMSNorm | features, no mean | no | LLMs (Llama, Qwen, DeepSeek)
| GroupNorm | channel groups and H, W of one image | no | U-Nets, small-batch vision

## Step 3: RMSNorm, LayerNorm without the centring

Zhang and Sennrich (2019) found that LayerNorm's rescaling matters much more than its re-centring. RMSNorm keeps only the rescaling:

$$r = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \epsilon}, \qquad y = g \odot \frac{x}{r}.$$

Here $r$ is the **root mean square** of the token (the square root of the average squared entry) and $g \in \mathbb{R}^D$ is the learned gain. There is no $\beta$ and no mean.

Number check: for $x = [3, -1, 1, 1]$ the squares are $[9, 1, 1, 1]$, their mean is 3, so $r = \sqrt 3 \approx 1.732$ and $y = [1.732, -0.577, 0.577, 0.577]$. The root mean square of the output is exactly 1.

What you gain: one fewer reduction per call, no bias parameter, less memory traffic. The RMSNorm paper reported speed-ups of roughly 7% to 64% over LayerNorm depending on model and implementation (stated from memory; fused GPU kernels narrow the gap). Quality is in practice the same, so Llama, Mistral, Qwen, DeepSeek and most other open LLMs use it. PyTorch has `nn.RMSNorm` since version 2.4.

What you lose: invariance to a constant shift. Add 2 to every feature and LayerNorm's output is unchanged, but RMSNorm's output moves. In practice the surrounding learned layers cope with this.

~~~dg-fig 02.02-invariance.svg
Left: multiplying the input by c leaves both outputs unchanged, and the input gradient shrinks exactly as 1/c (a straight line of slope -1 on log axes). Right: adding the same constant to every feature is removed by LayerNorm but leaks through RMSNorm (a shift of 2 moved its output by about 0.8 per feature).
~~~

## Step 4: the backward pass

Normalisation is not elementwise: changing one input $x_j$ moves the mean and spread, and so every output $y_i$. The gradient still has a compact form.

Write $\hat x = (x - \mu)/\sigma$ with $\sigma = \sqrt{\sigma^2 + \epsilon}$ (the normalised vector before the gain), let $\bar y$ be the gradient of the loss with respect to the output, and let $\bar{\hat x} = \gamma \odot \bar y$ be that gradient pushed back through the gain. Then

$$\bar x = \frac{1}{\sigma}\left(\bar{\hat x} - \mathrm{mean}(\bar{\hat x}) - \hat x \cdot \mathrm{mean}(\bar{\hat x} \odot \hat x)\right).$$

Read it as a **projection**. The incoming gradient has a part that would shift all features equally (useless: the layer subtracts the mean), a part along $\hat x$ that would only resize the token (useless: the layer divides that out), and the rest. The formula deletes the first two and divides by $\sigma$. For RMSNorm, with $\hat x = x / r$, only the scale direction is removed:

$$\bar x = \frac{1}{r}\left(\bar{\hat x} - \hat x \cdot \mathrm{mean}(\bar{\hat x} \odot \hat x)\right).$$

The parameter gradients are plain sums over every token in the batch: $\bar\gamma = \sum \bar y \odot \hat x$ and $\bar\beta = \sum \bar y$.

<div class="ex">
<div class="lab">Worked example</div>

**backpropagate through LayerNorm by hand**

Use the same token $x = [3, -1, 1, 1]$ with $\gamma = 1$, $\beta = 0$, $\epsilon = 0$. From step 1, $\sigma = \sqrt 2 = 1.414$ and $\hat x = [1.414, -1.414, 0, 0]$.

Suppose the loss only cares about the first output and $\partial L/\partial y_1 = 1$, so $\bar y = \bar{\hat x} = [1, 0, 0, 0]$.

Mean term: $\mathrm{mean}(\bar{\hat x}) = 1/4 = 0.25$. Subtracting it gives $[0.75, -0.25, -0.25, -0.25]$.

Scale term: $\mathrm{mean}(\bar{\hat x} \odot \hat x) = 1.414/4 = 0.354$. Multiply by $\hat x$: $[0.5, -0.5, 0, 0]$. Subtracting it gives $[0.25, 0.25, -0.25, -0.25]$.

Divide by $\sigma = 1.414$: $\bar x = [0.177, 0.177, -0.177, -0.177]$. PyTorch autograd returns the same four numbers.

Reading it: the entries sum to 0 (no shift component) and their dot product with $\hat x$ is $0.177 \times 1.414 - 0.177 \times 1.414 = 0$ (no scale component). The surprise is $\bar x_2 = +0.177$: raising the *low* feature $x_2 = -1$ raises $y_1$, because it shrinks the spread that $y_1$ is divided by.

</div>

A NumPy forward and backward pass, checked against autograd:

```python
import numpy as np

def layernorm_fwd_bwd(x, gamma, beta, dy, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    sigma = np.sqrt(x.var(-1, keepdims=True) + eps)   # biased variance, like PyTorch
    xhat = (x - mu) / sigma
    y = gamma * xhat + beta
    dxhat = dy * gamma
    dx = (dxhat - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / sigma
    dgamma = (dy * xhat).reshape(-1, x.shape[-1]).sum(0)
    dbeta = dy.reshape(-1, x.shape[-1]).sum(0)
    return y, dx, dgamma, dbeta

# against torch.nn.functional.layer_norm on random (3, 6) inputs:
# max |dx - autograd| = 4.4e-16, max |dgamma - autograd| = 1.3e-15
# dx.sum(-1) = [4.7e-16, 1.8e-16, -3.3e-16]   (no shift component)
```

Sign and broadcasting slips in hand-written backward passes do not crash; they just train badly, so always check against autograd.

A consequence of the projection: since the output is unchanged when the input is multiplied by $c$, the input gradient shrinks by exactly $1/c$. In the invariance figure a 64-feature token had input-gradient size 6.35 at $c = 1$ and 0.635 at $c = 10$. The same holds for weights feeding a norm: if they grow, their gradients shrink in proportion, so a normalised network partly self-stabilises.

## Step 5: where the norm goes in a Transformer block

A Transformer block has two sublayers (attention and a feed-forward network $F$), each wrapped in a residual connection. The norm can go in two places.

- **Post-LN** (the original 2017 Transformer): $x_{l+1} = \mathrm{LN}(x_l + F(x_l))$. The norm sits on the main path, after the add.
- **Pre-LN**: $x_{l+1} = x_l + F(\mathrm{LN}(x_l))$. The norm sits inside the branch, so the main path is a clean sum of residuals, the identity shortcut that makes ResNets trainable (03.04). One final norm goes before the output head.

~~~dg-flow Pre-LN block: the norm only touches the branch, so the residual stream x passes through every layer unchanged except for additions.
x -> LayerNorm -> Attention -(add back to x)-> *x + attn -> LayerNorm -> Feed-forward -(add back)-> *Next block
~~~

Xiong et al. (2020) showed that at initialisation Post-LN puts large gradients on parameters near the output, while Pre-LN gradients are well behaved. Large late-layer gradients at a normal learning rate make early updates unstable, which is why Post-LN models needed a long **learning-rate warmup** (start tiny, ramp up over thousands of steps).

~~~dg-fig 02.02-preln.svg
Gradient sizes at initialisation in a small Transformer (width 64, averaged over random seeds). Left: in a 12-layer model, Post-LN gradients grow towards the output while Pre-LN gradients shrink gently. Right: as the model gets deeper, the last layer's Post-LN gradient grows while the Pre-LN one falls.
~~~

The right panel is the effect Xiong et al. proved: the Pre-LN last-layer gradient falls like $1/\sqrt{L}$, so each layer automatically takes smaller steps as the stack grows, while the Post-LN one does not shrink. In our runs Post-LN went from 0.19 at 2 layers to 0.32 at 32 layers; Pre-LN went from 0.13 to 0.10. So a learning rate that suits a shallow Post-LN model is too large for a deep one. Our model is tiny (width 64), so the gaps are smaller than in the paper, but the direction matches. Here is what it does to training on a toy task (reverse a 10-token sequence):

~~~dg-fig 02.02-train.svg
Training loss of a 12-layer Transformer on the reverse-a-sequence task, Adam with peak learning rate 0.003. Post-LN without warmup never leaves chance level; the same model with a 100-step warmup learns; Pre-LN learns fastest, and RMSNorm tracks LayerNorm almost exactly.
~~~

In this run (one seed, an illustration only) Post-LN without warmup was still at loss 2.95 after 100 steps, barely below chance ($\ln 20 = 3.0$). A 100-step linear warmup rescued it (0.015 at step 100). Both Pre-LN models reached 0.07 by step 50 with no warmup. Pre-LN is now standard in GPT-style models, usually with RMSNorm; carefully tuned Post-LN can still match or beat it, which is why some designs add extra norms instead.

## Normalisation in code

```python
import torch, torch.nn as nn

x = torch.tensor([[3., -1., 1., 1.]])          # one token, D = 4
ln = nn.LayerNorm(4, eps=0.0)                    # gamma = 1, beta = 0 at init
rms = nn.RMSNorm(4, eps=0.0)                     # weight g = 1, no bias
print(ln(x))    # tensor([[ 1.4142, -1.4142,  0.0000,  0.0000]])
print(rms(x))   # tensor([[ 1.7321, -0.5774,  0.5774,  0.5774]])

# Hidden states of a Transformer: (batch N, sequence T, features D)
h = torch.randn(2, 5, 8) * 3 + 1
y = nn.LayerNorm(8)(h)                           # normalised over the last dim only
print(y.mean(-1)[0, :3], y.std(-1, unbiased=False)[0, :3])
# tensor([0., 0., 0.]) up to ~2e-8      tensor([1.0000, 1.0000, 1.0000])

alone = nn.LayerNorm(8)(h[:1])                   # same token, batch of one
print(torch.allclose(alone, y[:1]))              # True

class MyRMSNorm(nn.Module):                      # Llama-style RMSNorm
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.g = nn.Parameter(torch.ones(d))
    def forward(self, x):
        xf = x.float()                           # do the reduction in fp32
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * r).type_as(x) * self.g
print(torch.allclose(MyRMSNorm(8)(h), nn.RMSNorm(8, eps=1e-6)(h), atol=1e-6))  # True
```

Note the upcast in `MyRMSNorm`: with bf16 or fp16 activations, the sum of thousands of squares is computed in fp32. bf16 keeps only about three significant digits, so long sums in it lose precision; fp16 also overflows above 65504, which squared activations can exceed.

## Checking that it works

- **Output statistics.** With $\gamma = 1, \beta = 0$, each token after LayerNorm should have mean about 0 and (biased) standard deviation about 1 over the last axis; after RMSNorm, root mean square about 1.
- **Batch independence.** Run one example alone and inside a batch; outputs should match to floating-point tolerance. If not, some layer is mixing examples (a stray BatchNorm, or a norm over the wrong axis).
- **Gradient check.** Compare custom kernels with autograd in float64: differences near $10^{-15}$ are fine, $10^{-3}$ is a bug.
- **Per-layer gradient norms.** Log them by layer early in training, as in the Pre-LN figure. A strong ramp towards the output says norm placement or warmup needs attention.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Normalising over the wrong axis, e.g. `nn.LayerNorm(T)` on a (N, D, T) tensor | Training works but tokens leak information into each other; causal models "see the future" | Put features last, or pass the correct `normalized_shape`
| Using BatchNorm in a sequence model | Results change with padding and batch size; generation quality drops | LayerNorm or RMSNorm per token
| Post-LN deep stack without warmup | Loss stalls near chance, spikes or diverges in the first few hundred steps | Switch to Pre-LN, or add a warmup of a few thousand steps
| Pre-LN without a final norm before the head | Output logits grow with depth; unstable loss | Add one last LayerNorm or RMSNorm after the final block
| Norm reduction in bf16 or fp16 | Slight quality loss, or NaNs with fp16 | Accumulate in fp32 as most library kernels do
| Applying weight decay to $\gamma$ / $g$ and $\beta$ | Gains are pushed towards 0, slowing training | Exclude norm parameters (and biases) from weight decay

## Going deeper

**Geometry.** Ignoring $\epsilon$ and the affine part, RMSNorm maps every nonzero vector onto the sphere of radius $\sqrt D$: it keeps the direction and discards the length. LayerNorm first projects onto the zero-mean hyperplane, then onto the radius-$\sqrt D$ sphere inside it. The backward formulas are the tangent-space projections of these maps, which is why the gradient is orthogonal to $\hat x$ (and, for LayerNorm, to the all-ones vector).

**Why Pre-LN behaves.** In Pre-LN, $x_L = x_0 + \sum_l F_l(\mathrm{LN}(x_l))$, so the derivative of the output with respect to any earlier state contains an identity term independent of depth. In Post-LN every gradient passes back through a norm per layer, with no such bypass. The cost of Pre-LN is that the residual stream grows with depth, so late blocks add relatively small updates. Variants such as sandwich norm, DeepNorm (a rescaled Post-LN) and QK-norm (normalising queries and keys inside attention) aim for stability without that dilution.

**Scale invariance and the effective learning rate.** For a weight matrix $W$ followed by a norm, the loss satisfies $L(cW) = L(W)$, so $\nabla L(W) \perp W$ and $\|\nabla L(cW)\| = \|\nabla L(W)\|/c$. Plain gradient steps, being orthogonal to $W$, always increase $\|W\|$, which lowers the effective step size $\eta / \|W\|^2$. Weight decay counteracts the growth, so it still matters in normalised networks even though it barely changes the function they compute.

**Cost and alternatives.** Both layers are memory-bound (one read and write per activation, few operations each), so production kernels fuse them with the residual add; that is where RMSNorm's one-fewer-reduction saving shows. Some models drop norms entirely, using careful initialisation (Fixup), adaptive gradient clipping (NFNets) or an elementwise $\tanh$ (Dynamic Tanh): the norm's real job is keeping signal scale under control at every depth.

## Key terms

| Term | Meaning
| --- | --- |
| LayerNorm | Standardises each example (token) over its own features, then applies learned gain and bias
| RMSNorm | Divides each token by its root mean square; no centring and no bias
| GroupNorm | Normalises groups of channels over the spatial positions of one image
| Gain $\gamma$ / $g$ | Learned per-feature scale applied after normalising
| Batch independence | A token's output depends only on its own values, never on other examples
| Scale invariance | Multiplying the input by $c$ leaves the output unchanged and divides the input gradient by $c$
| Pre-LN | Norm inside the residual branch: $x + F(\mathrm{LN}(x))$
| Post-LN | Norm after the residual add: $\mathrm{LN}(x + F(x))$; needs warmup in deep stacks

<div class="take">
<div class="lab">Key takeaway</div>

LayerNorm and RMSNorm normalise each token on its own, so they are batch-independent and identical at train and test time; their backward pass projects out the mean and scale directions, and placing them inside the residual branch (Pre-LN) keeps deep stacks stable.

</div>
