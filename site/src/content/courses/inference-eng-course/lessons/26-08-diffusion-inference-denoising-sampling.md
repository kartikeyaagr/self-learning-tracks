---
id: "26.08"
course: "inference-eng-course"
module: "26"
slug: "diffusion-inference-denoising-sampling"
title: "Diffusion Inference: Denoising & Sampling"
keyIdeas: ["Diffusion generates by starting from noise and denoising over many steps.", "Each step runs the full network on the whole latent: compute bound.", "Latency is steps x step time; there is no token-by-token streaming.", "Samplers (DDIM, DPM-Solver and others) set how many steps are needed.", "Classifier-free guidance doubles work per step unless batched or distilled."]
question: "Why does diffusion inference behave so differently from LLM decode in terms of batching and bottlenecks?"
answer: "A diffusion model runs a fixed number of denoising steps over the whole image or latent at once, each step a full forward pass of a large network on many spatial tokens. Work per step is large and compute bound, with no growing KV cache. Latency is roughly steps times step time, and batching helps less because each step is already compute heavy."
sources: [{"title": "Denoising Diffusion Probabilistic Models", "url": "https://arxiv.org/abs/2006.11239", "note": "DDPM."}, {"title": "Denoising Diffusion Implicit Models", "url": "https://arxiv.org/abs/2010.02502", "note": "DDIM, fewer-step sampling."}, {"title": "Hugging Face Diffusers", "url": "https://huggingface.co/docs/diffusers/index", "note": "Pipelines and schedulers."}]
tags: ["diffusion"]
---

## The process

~~~dg-flow Diffusion sampling.
Noise latent -> *Denoise steps 1 to N (full network each) -> Clean latent -> VAE decode -> Image
~~~

An image diffusion model starts from random noise in a compressed latent space and removes a little noise per step. Each step runs the whole denoising network. A VAE decoder then turns the clean latent into pixels.

## Cost structure

| Factor | Effect |
| --- | --- |
| Steps N | latency grows linearly |
| Resolution | more latent tokens; attention cost grows quadratically |
| Classifier-free guidance | two passes per step: with and without the prompt |
| VAE decode | a one-time cost at the end |

## Compared with LLM decode

| | LLM decode | Diffusion |
| --- | --- | --- |
| Work per step | small, memory bound | large, compute bound |
| State | KV cache grows | a fixed-size latent |
| Number of steps | output length, variable | fixed by the sampler |
| Benefit from batching | large | moderate, since steps are already compute bound |

<div class="ex">
<div class="lab">Worked example</div>

**why batching helps less**

An LLM decode step for one sequence reads all weights for very little compute. Batch 64 costs about the same time as batch 1, so throughput rises about 60 times.

A diffusion step at 1024 × 1024 already processes 4,096 latent tokens through every layer. It is compute bound at batch 1. Batch 4 takes close to 4 times as long, so throughput rises only modestly.

</div>

## Samplers

Better ODE and SDE solvers reach similar quality in 20 to 50 steps where early methods needed 1,000. Distilled models go down to 1 to 4 steps.

## Deep dive: why diffusion models batch differently

A diffusion model runs the same network ⟦m1128⟧ times per image, on all tokens of the image at once:

⟦M1127⟧

With thousands of latent tokens per image, each step is already a large, compute-bound matmul. Arithmetic intensity is high at batch 1, so batching adds throughput only until the GPU's compute is full, often at small batch sizes. The levers are fewer steps (distilled samplers) and cheaper steps (quantization, caching features across steps), not larger batches.

<div class="take">
<div class="lab">Key takeaway</div>

diffusion is a fixed number of heavy, compute-bound steps. Speed comes from fewer steps, cheaper steps and efficient guidance.

</div>
