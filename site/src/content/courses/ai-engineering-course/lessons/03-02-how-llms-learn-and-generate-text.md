---
id: "03.02"
course: "ai-engineering-course"
module: "03"
slug: "how-llms-learn-and-generate-text"
title: "How LLMs learn and generate text"
keyIdeas: ["Pretraining: show the model text, ask it to predict every next token in parallel (with causal masking), minimise cross-entropy by gradient descent.", "Scaling laws: loss falls as a smooth power law with parameters, data and compute; training compute is about 6ND.", "Chinchilla (2022): for a fixed training budget, about 20 tokens per parameter is optimal. Modern models train far past that because serving cost dominates.", "Post-training turns a text predictor into a product: supervised fine-tuning, preference tuning, and reinforcement learning on verifiable tasks.", "Synthetic data, generated and filtered by models, now makes up much of post-training data, which brings collapse and contamination risks."]
question: "What loss does pretraining minimise, and what did the Chinchilla paper change about how big models are trained?"
answer: "Pretraining minimises cross-entropy loss on next-token prediction: the negative log-probability the model assigned to the actual next token at each position, averaged over the corpus. A causal mask lets one forward pass score every position in a sequence at once. Chinchilla showed that for a fixed compute budget, earlier large models were too big and trained on too little data; compute-optimal training scales data in proportion to parameters, roughly 20 tokens per parameter, and their 70B model on 1.4T tokens beat the 280B Gopher trained with the same compute. Labs now deliberately train smaller models far past that ratio, because a smaller model is cheaper to serve for its whole deployed life."
sources: [{"title": "Kaplan et al.: Scaling Laws for Neural Language Models (2020)", "url": "https://arxiv.org/abs/2001.08361", "note": "Loss falls predictably with parameters, data and compute."}, {"title": "Hoffmann et al.: Training Compute-Optimal LLMs (Chinchilla, 2022)", "url": "https://arxiv.org/abs/2203.15556", "note": "About 20 tokens per parameter for compute-optimal training."}, {"title": "Llama Team: The Llama 3 Herd of Models (2024)", "url": "https://arxiv.org/abs/2407.21783", "note": "A full modern recipe: 15T-token pretraining, SFT, rejection sampling and DPO with heavy synthetic data."}, {"title": "Andrej Karpathy: Let's build GPT from scratch (video)", "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY", "note": "Code a small GPT and its training loop."}]
practice: ["cross-entropy-logits", "chinchilla-compute-optimal", "transformer-training-flop-estimator", "build-assistant-only-sft-loss-mask", "dpo-closed-form-loss"]
---

## The training objective

Pretraining is simple to state. Take a huge corpus, cut it into sequences, and at every position ask the model to predict the next token. The loss is **cross-entropy**: the negative log probability the model gave to the token that actually came next, averaged over all $T$ positions,

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{1:t-1})
$$

and you minimise it with gradient descent. If the model gave the real next token probability 0.5, that position costs $-\log 0.5 \approx 0.69$; at probability 0.01 it costs 4.6.

~~~dg-flow One training step.
Batch of text -(forward pass)-> Predicted distributions -(compare with real next tokens)-> *Cross-entropy loss -(backprop)-> Weight update
~~~

The key efficiency is that a transformer predicts every position in one forward pass. A causal mask ([Masked attention](lesson:aieng:05.01)) stops each position from seeing later tokens, so one 4,000-token sequence yields 4,000 training examples at once. Karpathy's GPT-from-scratch video builds exactly this loop.

## Training vs generation

| Aspect | Training | Generation |
| --- | --- | --- |
| Input | Real text | Prompt plus generated tokens |
| Positions processed | All in parallel | One new token per step |
| Output used | Loss on every position | Only the last position's distribution |
| Weights | Updated | Frozen |

At inference the model only uses the distribution at the final position, and it conditions on its own earlier output. In training it always saw real text, so an early mistake at generation time can compound.

## Scaling laws and the compute-optimal trade-off

Kaplan et al. (2020) found that loss falls as a smooth power law as you increase parameters, data or compute, over many orders of magnitude. That made it possible to predict a big model's loss from small runs and plan budgets in advance.

Training compute is roughly $C \approx 6ND$ floating point operations, for $N$ parameters and $D$ training tokens (2 for the forward pass, 4 for the backward). For a fixed $C$ you can train a big model on few tokens or a small one on many. Hoffmann et al. (2022, "Chinchilla") found loss is lowest when tokens grow in proportion to parameters, at **about 20 tokens per parameter**. Their 70B Chinchilla trained on 1.4T tokens beat the 280B Gopher, which used the same compute on fewer tokens: the earlier large models had been undertrained.

~~~dg-raw
%%widget chinchilla For a fixed compute budget, how loss changes as you trade model size against training tokens.
~~~

## Why everyone now trains past "optimal"

Chinchilla minimises training compute for a given quality and ignores inference. A model served to millions of users spends far more compute on inference than it ever did on training, and inference costs about $2N$ operations per generated token. So it pays to train a smaller model for much longer:

| Model | Parameters | Training tokens | Tokens per parameter |
| --- | --- | --- | --- |
| Chinchilla (2022) | 70B | 1.4T | 20 |
| Llama 2 7B (2023) | 7B | 2T | about 290 |
| Llama 3 8B (2024) | 8B | over 15T | about 1,900 |
| Qwen3 8B (2025) | 8B | about 36T | about 4,500 |

Loss keeps falling slowly past 20:1, so these models beat a Chinchilla-optimal model of the same size and are far cheaper to serve than a Chinchilla-optimal model of the same quality. The limit is now data: high-quality web text is finite, which is one reason synthetic data matters (below).

<div class="ex">
<div class="lab">Worked example</div>

**estimating a training run**

An 8B model on 15T tokens needs $6 \times 8{\times}10^{9} \times 15{\times}10^{12} = 7.2{\times}10^{23}$ FLOPs.

An H100 might sustain 400 TFLOP/s on bf16 training (about 40% of its dense peak), so that is $7.2{\times}10^{23} / 4{\times}10^{14} = 1.8{\times}10^{9}$ GPU-seconds, or about 500,000 GPU-hours. At an illustrative \$2–3 per GPU-hour, roughly \$1–1.5 million for one clean run of a "small" model. Meta's model card lists about 1.5M H100-hours for Llama 3.1 8B: real runs land well above the clean estimate once utilisation, restarts and long-context stages are counted, and that is before any failed experiments.

</div>

## Post-training: from predictor to product

A base model continues internet-like text. Post-training is a sequence of smaller, targeted stages:

~~~dg-flow A typical 2024–26 post-training pipeline.
Base model -(SFT on prompts + ideal answers)-> Instruct model -(preference tuning: RLHF / DPO)-> *Aligned model -(RL on verifiable tasks)-> Reasoning model
~~~

1. **Supervised fine-tuning (SFT).** Train on (prompt, ideal response) pairs in the chat template, with loss only on the assistant tokens. Tens of thousands to a few million examples teach format, instruction following and refusals.
2. **Preference tuning.** Show pairs of answers and train the model to prefer the better one: RLHF trains a reward model and optimises against it with RL; DPO trains on the pairs directly. This shapes helpfulness, tone and safety.
3. **Reinforcement learning with verifiable rewards.** For maths and code, a checker (unit tests, an exact answer) gives the reward, and the model learns to produce long reasoning before answering. DeepSeek-R1 and the o-series made this a standard stage; [Reasoning models and human feedback](lesson:aieng:08.01) covers it.

Labs also run a **mid-training** phase between pretraining and SFT: higher-quality and domain data (code, maths), and long-context extension where the sequence length is raised and the position encoding rescaled ([Positional embeddings](lesson:aieng:04.01)).

## Synthetic data

Human-written examples are slow and expensive, so most modern post-training data is generated by models and then filtered. The Llama 3 paper describes the common recipe: sample several responses per prompt from the current best model, score them with a reward model or a checker (rejection sampling), keep the best, and repeat over several rounds. Code answers are kept only if they compile and pass tests; maths answers only if the final answer matches.

This works because checking is easier than generating. It fails in predictable ways:

- **Collapse and narrowing.** Training repeatedly on a model's own outputs loses the rare and diverse cases; mixing in real data and filtering for diversity counters it.
- **Inherited errors and style.** A teacher's mistakes and verbal tics carry over unless the filter catches them.
- **Contamination.** Generated questions can reproduce benchmark items, inflating eval scores; decontaminate against your evals.

When you generate your own fine-tuning data, the same rules apply; [Synthetic data and distillation](lesson:aieng:07.08) covers the app-engineer version, and [Preference tuning for products](lesson:aieng:07.09) covers DPO and graders.

## Common mistakes

- Treating Chinchilla's 20:1 as the right ratio for a model you will serve; it optimises training cost only.
- Reading a lower pretraining loss as a better assistant: post-training quality often decides more than base loss.
- Assuming a fine-tuned model "learned facts" from SFT; post-training mostly shapes behaviour, and facts it adds go stale ([Making a model useful](lesson:aieng:03.04)).

<div class="take">
<div class="lab">Key takeaway</div>

LLMs learn by minimising next-token cross-entropy over huge corpora, in parallel over every position. Scaling laws make loss predictable, serving cost pushes labs to train small models on far more data, and a stack of post-training stages, increasingly fed by filtered synthetic data, turns the predictor into an assistant.

</div>
