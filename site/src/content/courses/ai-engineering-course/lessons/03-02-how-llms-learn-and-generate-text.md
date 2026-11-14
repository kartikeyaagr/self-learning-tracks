---
id: "03.02"
course: "ai-engineering-course"
module: "03"
slug: "how-llms-learn-and-generate-text"
title: "How LLMs learn and generate text"
keyIdeas: ["Pretraining: show the model text, ask it to predict each next token, measure cross-entropy loss, and update weights by gradient descent.", "Training is parallel: in one pass the model predicts every position in a sequence at once (with causal masking).", "Scaling laws (Kaplan 2020): loss falls as a smooth power law with more parameters, data and compute.", "Chinchilla (2022): for a fixed compute budget, models were too big and undertrained. About 20 training tokens per parameter is compute-optimal.", "Modern models train far beyond 20 tokens per parameter because inference cost matters more than training cost.", "After pretraining come supervised fine-tuning and preference training, which turn a text predictor into an assistant."]
question: "What loss does pretraining minimise, and what did the Chinchilla paper change about how big models are trained?"
answer: "Pretraining minimises cross-entropy loss on next-token prediction: the negative log-probability of the actual next token at each position, averaged over the corpus. Chinchilla showed that for a fixed compute budget, earlier models were too large and trained on too little data; compute-optimal training scales data with parameters, roughly 20 tokens per parameter (70B params on 1.4T tokens beat a 280B model)."
sources: [{"title": "Andrej Karpathy: Let's build GPT from scratch (video)", "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY", "note": "Code a small GPT and its training loop."}, {"title": "Kaplan et al.: Scaling Laws for Neural Language Models (2020)", "url": "https://arxiv.org/abs/2001.08361", "note": "Loss falls predictably with parameters, data and compute."}, {"title": "Hoffmann et al.: Training Compute-Optimal LLMs (Chinchilla, 2022)", "url": "https://arxiv.org/abs/2203.15556", "note": "About 20 tokens per parameter for compute-optimal training."}]
---

## The training objective

Pretraining is simple to state. Take a huge corpus, cut it into sequences, and for every position ask the model to predict the next token. The loss is **cross-entropy**: the negative log probability the model gave to the token that actually came next. Average it over all positions and minimise with gradient descent.

~~~dg-flow One training step.
Batch of text -(forward pass)-> Predicted distributions -(compare with real next tokens)-> *Cross-entropy loss -(backprop)-> Weight update
~~~

A key efficiency: the transformer predicts every position in a sequence in one forward pass. A causal mask stops each position from seeing future tokens, so one 4,000-token sequence gives 4,000 training examples at once. Karpathy's GPT-from-scratch video builds exactly this loop.

## Training vs generation

| Aspect | Training | Generation |
| --- | --- | --- |
| Input | Real text | Prompt plus generated tokens |
| Positions processed | All in parallel | One new token per step |
| Output used | Loss on every position | Only the last position's distribution |
| Weights | Updated | Frozen |

At inference the model only ever uses the distribution at the final position. Errors can compound: the model conditions on its own earlier tokens, which may be wrong.

## Scaling laws

Kaplan et al. (2020) found that loss falls as a smooth power law as you increase parameters, data or compute, over many orders of magnitude. That made it possible to predict a big model's loss from small runs and plan budgets in advance.

Hoffmann et al. (Chinchilla, 2022) revisited the balance. For a fixed compute budget, they found earlier large models were undertrained: you should scale data as fast as parameters. Their 70B Chinchilla trained on 1.4T tokens beat the 280B Gopher. The rule of thumb that came out of it is **about 20 tokens per parameter**.

## Why modern models overtrain

Chinchilla optimises training compute only. Once a model is deployed, every user request costs inference compute, which grows with parameter count. So labs now train smaller models on far more tokens than Chinchilla suggests (Llama 3 8B saw about 15T tokens, close to 2,000 per parameter). It costs more to train but less to serve.

## After pretraining

Pretraining gives a base model that continues text. Two more stages make it useful:

1. **Supervised fine-tuning (SFT)** on instruction and response examples.
2. **Preference training** (RLHF, DPO and related methods) to prefer helpful, honest, harmless answers.

## The compute-optimal trade-off

Training compute is roughly 6 × N × D floating point operations, where N is parameters and D is training tokens. For a fixed budget you can train a big model on few tokens or a small model on many. Hoffmann et al. (2022, "Chinchilla") found that loss is lowest when tokens grow in proportion to parameters, about 20 tokens per parameter.

~~~dg-raw
%%widget chinchilla For a fixed compute budget, how loss changes as you trade model size against training tokens.
~~~

Their 70B model trained on 1.4 trillion tokens beat the 280B Gopher, which used the same compute on fewer tokens. The lesson for the field was that earlier large models were undertrained.

## Why everyone now trains past "optimal"

Chinchilla-optimal minimizes training cost for a given quality. It ignores inference. A model you serve to millions of users runs inference far more than it ever trained, and inference cost grows with N (about 2 × N operations per generated token). So it pays to train a smaller model for much longer:

| Model | Parameters | Training tokens | Tokens per parameter |
| --- | --- | --- | --- |
| Chinchilla (2022) | 70B | 1.4T | 20 |
| Llama 2 7B (2023) | 7B | 2T | about 290 |
| Llama 3 8B (2024) | 8B | over 15T | about 1,900 |

Loss keeps falling slowly past the 20:1 point, so these models are better than a Chinchilla-optimal model of the same size, and far cheaper to serve than a Chinchilla-optimal model of the same quality.

## What pretraining does and doesn't give you

After pretraining, the model is a strong next-token predictor over internet-like text. Ask it "What is the capital of France?" and it may continue with three more quiz questions, because that is a likely continuation of a list of quiz questions. Turning it into an assistant is the job of post-training, covered in 03.04.

<div class="ex">
<div class="lab">Worked example</div>

**estimating training cost**

An 8B model on 15T tokens needs 6 × 8×10^9 × 15×10^12 = 7.2×10^23 FLOPs.

An H100 sustains maybe 400 TFLOP/s in practice on bf16 training, so that is 7.2×10^23 / 4×10^14 = 1.8×10^9 GPU-seconds, or about 500,000 GPU-hours.

At a few dollars per GPU-hour, that is a seven-figure training run for a "small" model, before any failed experiments.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

LLMs learn by minimising next-token cross-entropy over huge corpora, in parallel over every position. Scaling laws make loss predictable, and inference cost pushes labs to train smaller models on more data.

</div>
