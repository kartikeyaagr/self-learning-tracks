---
id: "02.02"
course: "deep-learning"
module: "02"
slug: "layer-and-rms-normalisation"
title: "Layer and RMS normalisation"
keyIdeas: ["LayerNorm normalises each example (each token in a Transformer) over its own feature dimension, so it behaves identically in training and inference and has no dependence on batch size.", "RMSNorm drops the mean subtraction and the bias: $y = g \\odot x / \\sqrt{\\mathrm{mean}(x^2) + \\epsilon}$; it is cheaper and works as well in large language models.", "The backward pass of a normalisation layer is not elementwise: the gradient is projected to remove components along the mean and along the normalised vector itself.", "Pre-LN Transformers (normalise inside the residual branch) train stably without warmup tricks; Post-LN can have exploding gradients near the output at initialisation."]
question: "You switch a Transformer from LayerNorm to BatchNorm over the batch and sequence dimensions to \"normalise more data at once\". Training loss looks fine, but generation quality collapses and padded batches behave differently from unpadded ones. Explain both problems and why LayerNorm or RMSNorm avoids them."
answer: "BatchNorm computes each feature's statistics over all tokens in the batch, including padding positions and tokens from other sequences. With padding, the pad embeddings enter the mean and variance, so the same real sequence gets different normalisation depending on how much padding its batch contains: outputs depend on batch composition. At generation time you decode one token at a time, often with batch size one, so the model either uses training-mode statistics of a single token (degenerate) or running averages accumulated over full teacher-forced batches, which do not match the distribution of incrementally generated states. LayerNorm and RMSNorm compute statistics over the $D$ features of each token separately. A token's output depends only on its own hidden state, so padding, batch size and decoding mode cannot change it, and training and inference run the identical computation."
sources: [{"title": "Ba, Kiros, Hinton: Layer Normalization", "url": "https://arxiv.org/abs/1607.06450", "note": "Motivation for per-example statistics and the comparison with batch norm in RNNs."}, {"title": "Zhang & Sennrich: Root Mean Square Layer Normalization", "url": "https://arxiv.org/abs/1910.07467", "note": "RMSNorm's definition, the re-centring ablation and speed measurements."}, {"title": "Xiong et al.: On Layer Normalization in the Transformer Architecture", "url": "https://arxiv.org/abs/2002.04745", "note": "Why Post-LN needs warmup and Pre-LN does not, via gradient norms at init."}, {"title": "PyTorch docs: LayerNorm", "url": "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html", "note": "normalized_shape semantics and elementwise affine parameters."}]
practice: ["layer-normalization", "layer-norm", "rmsnorm-forward-pass", "norm-gated-linear-transform", "layernorm-backward"]
tags: ["layer norm", "RMSNorm", "Pre-LN"]
tools: ["NumPy", "PyTorch"]
---

## Normalise across features, not across the batch

BatchNorm (02.01) ties each example's output to the rest of the batch. Layer normalisation removes that coupling by computing statistics over the **feature** dimension of one example. For a vector $x \in \mathbb{R}^D$ (one token's hidden state, for instance):

$$\mu = \frac{1}{D}\sum_i x_i, \quad \sigma^2 = \frac{1}{D}\sum_i (x_i - \mu)^2, \quad y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

$\gamma, \beta \in \mathbb{R}^D$ are learned. There are no running statistics, no train/eval difference, and a batch of one works exactly like a batch of a thousand. That is why every Transformer, RNN and most generative models use LayerNorm or a relative. For CNNs, **GroupNorm** sits in between: it normalises over groups of channels plus spatial positions within one image, and is the default in diffusion U-Nets where batches are small.

| Norm | Statistics over | Batch-dependent | Typical home |
| --- | --- | --- | --- |
| BatchNorm | N, H, W per channel | yes | CNN classifiers |
| LayerNorm | features of one example | no | Transformers, RNNs |
| RMSNorm | features, no mean | no | LLMs (Llama, Qwen, DeepSeek) |
| GroupNorm | channel groups and H, W of one image | no | U-Nets, small-batch vision |

~~~dg-layers What gets averaged together, for an activation tensor of shape (batch N, sequence T, features D).
BatchNorm | one statistic per feature, pooled over all N × T positions
*LayerNorm | one statistic per (n, t) token, pooled over its D features
RMSNorm | same as LayerNorm but only the root mean square, no centring
~~~

## RMSNorm

Zhang and Sennrich observed that LayerNorm's benefit comes mostly from rescaling, not recentring. RMSNorm keeps only the scale:

$$r = \sqrt{\frac{1}{D}\sum_i x_i^2 + \epsilon}, \qquad y = g \odot \frac{x}{r}$$

There is no $\beta$ and no mean. It saves one reduction per call and some memory traffic, and in practice matches LayerNorm quality, so most open LLMs use it. It is usually computed in fp32 even when activations are bf16, because the sum of squares over $D = 4096$ elements can lose precision or overflow in half precision. AI Research's RMSNorm lesson and Inference Engineering's "MLP, RMSNorm & Residual Connections" go deeper on its role inside a decoder block.

## The backward pass

Normalisation couples all $D$ outputs to all $D$ inputs: changing one $x_j$ moves the mean and the variance, which moves every $y_i$. Let $\hat x = (x - \mu)/\sigma$ with $\sigma = \sqrt{\sigma^2 + \epsilon}$, and let $\bar{\hat x} = \gamma \odot \bar y$ be the upstream gradient after the scale. Then

$$\bar x = \frac{1}{\sigma}\left(\bar{\hat x} - \mathrm{mean}(\bar{\hat x}) - \hat x \cdot \mathrm{mean}(\bar{\hat x} \odot \hat x)\right)$$

Read it as a projection: remove the component of the gradient that would only shift the mean (which normalisation undoes anyway) and the component along $\hat x$ (which would only change the scale), then divide by $\sigma$. For RMSNorm, with $\hat x = x / r$, there is no mean term:

$$\bar x = \frac{1}{r}\left(\bar{\hat x} - \hat x \cdot \mathrm{mean}(\bar{\hat x} \odot \hat x)\right)$$

The parameter gradients are simple sums over the batch: $\bar\gamma = \sum \bar y \odot \hat x$, $\bar\beta = \sum \bar y$. A useful consequence: because the input gradient is orthogonal to $\hat x$, scaling the input up by a constant scales the gradient down by the same constant, which is the scale invariance that keeps normalised networks stable.

```python
import numpy as np

def layernorm_fwd_bwd(x, gamma, beta, dy, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    sigma = np.sqrt(x.var(-1, keepdims=True) + eps)
    xhat = (x - mu) / sigma
    y = gamma * xhat + beta
    dxhat = dy * gamma
    dx = (dxhat - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / sigma
    dgamma = (dy * xhat).reshape(-1, x.shape[-1]).sum(0)
    dbeta = dy.reshape(-1, x.shape[-1]).sum(0)
    return y, dx, dgamma, dbeta
```

Check any hand-written version against finite differences or autograd before trusting it; sign and broadcasting slips here are common.

## Pre-LN versus Post-LN

The original Transformer put LayerNorm **after** the residual add: $x_{l+1} = \mathrm{LN}(x_l + F(x_l))$. Xiong et al. showed that at initialisation this places large gradients on the last layers, which is why Post-LN models needed long learning-rate warmup. Pre-LN puts it inside the branch, $x_{l+1} = x_l + F(\mathrm{LN}(x_l))$, leaving a clean identity path from input to output (the same idea as ResNets in 03.04). Pre-LN trains stably with little warmup and is standard in GPT-style models, usually with one final norm before the output head.

<div class="ex">
<div class="lab">Worked example</div>

**RMSNorm on one token**

$x = [3, -1, 1, 1]$, $D = 4$. Mean of squares $= (9 + 1 + 1 + 1)/4 = 3$, so $r = \sqrt{3} \approx 1.732$ (ignoring $\epsilon$).

$\hat x = [1.732, -0.577, 0.577, 0.577]$; the root mean square of $\hat x$ is exactly 1. With $g = 1$ this is the output.

LayerNorm on the same $x$: $\mu = 1$, $\sigma^2 = (4 + 4 + 0 + 0)/4 = 2$, so $\hat x = [1.414, -1.414, 0, 0]$. Centring changed which features look large: the two 1s are now zero.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

LayerNorm and RMSNorm normalise each token on its own, so they are batch-independent and identical at train and test time; their backward pass projects out the mean and scale directions, and placing them inside the residual branch (Pre-LN) keeps deep stacks stable.

</div>
