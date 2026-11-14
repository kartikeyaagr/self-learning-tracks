---
id: "03.04"
course: "ai-engineering-course"
module: "03"
slug: "making-a-model-useful"
title: "Making a model useful"
keyIdeas: ["A base model only continues text. Several techniques make it useful, at very different costs.", "Prompting and few-shot examples: cheapest, change behaviour per call with no training.", "Retrieval (RAG): add fresh or private knowledge at query time. Best for facts that change.", "Supervised fine-tuning (SFT): train on example inputs and outputs. Best for format, tone, style and narrow skills.", "Preference training (RLHF, DPO): train on which of two answers is better, to make outputs helpful and safe.", "Parameter-efficient fine-tuning (LoRA) trains small adapter matrices instead of all weights, cutting memory and cost."]
question: "A support bot answers in the wrong tone and also doesn't know your product's latest pricing. Which technique fixes each problem, and why not the other way round?"
answer: "Stale pricing is a knowledge problem: use retrieval (RAG) to put current pricing into the prompt at query time; fine-tuning would freeze facts that change and can't be updated without retraining. Wrong tone is a behaviour problem: fix it with a better system prompt and examples, and if that's not enough, supervised fine-tuning (e.g. LoRA) on example replies in the right tone. RAG doesn't reliably change style."
sources: [{"title": "Ouyang et al.: InstructGPT (2022)", "url": "https://arxiv.org/abs/2203.02155", "note": "SFT plus RLHF: a 1.3B instruct model preferred over 175B GPT-3."}, {"title": "Hu et al.: LoRA (2021)", "url": "https://arxiv.org/abs/2106.09685", "note": "Low-rank adapters for cheap fine-tuning."}, {"title": "OpenAI: Optimizing LLM accuracy", "url": "https://platform.openai.com/docs/guides/optimizing-llm-accuracy", "note": "When to use prompting, RAG or fine-tuning."}]
---

## From text predictor to assistant

A pretrained model has broad knowledge but no idea it is supposed to help you. InstructGPT (Ouyang et al., 2022) showed how much the next steps matter: their 1.3B model after fine-tuning and RLHF was preferred by human raters over the raw 175B GPT-3.

## The ladder of techniques

~~~dg-layers Try the cheaper rungs first. Each one up costs more and changes the model more.
Prompting | instructions, few-shot examples, system prompt
*Retrieval (RAG) | inject relevant, current documents at query time
Fine-tuning (SFT / LoRA) | train on input/output examples for style, format, narrow skills
Preference training | RLHF / DPO on pairs of better and worse answers
Pretraining | new base model: only for labs
~~~

OpenAI's accuracy guide frames the choice as two axes: **context optimisation** (what the model needs to know: RAG) and **LLM optimisation** (how the model needs to act: fine-tuning). Many systems need both.

## Supervised fine-tuning

Collect examples of the input and the ideal output, and continue training on them. It's good for:

- consistent tone and format;
- a narrow task the model does poorly with prompting alone;
- shortening prompts (behaviour learned instead of described every call).
It's bad at adding facts that change: the knowledge goes stale and the model can blend it wrongly.

## LoRA

Full fine-tuning updates every weight, which needs as much GPU memory as training. LoRA (Hu et al., 2021) freezes the original weights and learns two small matrices per layer whose product is the weight update. Trainable parameters drop by orders of magnitude, adapters are megabytes, and you can swap adapters per customer on one base model. QLoRA goes further by keeping the frozen base in 4-bit.

## Preference training

SFT teaches what a good answer looks like. Preference training teaches which of two answers is better. RLHF trains a reward model on human comparisons and optimises the LLM against it with reinforcement learning. DPO skips the reward model and trains directly on the preference pairs. Module 8 returns to this for reasoning models.

<div class="ex">
<div class="lab">Worked example</div>

**wrong tone, stale pricing**

Pricing changes every month, so it belongs in retrieval: fetch the current price page into the prompt. Fine-tuning would bake in prices that go stale.

Tone is a stable behaviour. Start with a system prompt and examples; if it is still inconsistent, a small SFT or LoRA run on a few hundred good replies fixes it.

</div>

## How cheap is LoRA, really?

LoRA freezes the original weight matrix W and learns a low-rank update B × A next to it. Only A and B train. Set a model shape and a rank below and compare trainable parameters and memory with full fine-tuning.

~~~dg-raw
%%widget lora Trainable parameters and training memory: full fine-tuning versus LoRA.
~~~

The reason it works: the change that fine-tuning makes to a weight matrix tends to have low "intrinsic rank". A 4,096 × 4,096 matrix has 16.8 M entries, but the useful adjustment for a narrow task can be captured by B (4,096 × 16) times A (16 × 4,096), which is 131k entries, about 0.8% of the size. Hu et al. (2021) reported LoRA on GPT-3 175B with 10,000× fewer trainable parameters and about 3× less GPU memory than full fine-tuning, at similar quality.

## When to fine-tune at all

Fine-tuning changes how the model behaves. It is a poor way to add facts.

| You want | Better tool |
| --- | --- |
| The model to know your product docs | RAG (03.05): facts stay current and citable |
| A consistent output format or tone | Prompting with examples first; fine-tune if that's not enough |
| A small model to match a big one on one task | Fine-tune or distil (07.03) on the big model's outputs |
| Lower latency and cost on a high-volume task | Fine-tune a small model |
| New behavior the base model can't do from prompts | Fine-tune, with a few thousand good examples |

A useful rule: get the best result you can with prompting and retrieval, build an eval set, and only fine-tune when the eval shows a gap that more prompting doesn't close. The eval set is also what tells you whether the fine-tune helped.

<div class="ex">
<div class="lab">Worked example</div>

**a classification fine-tune that paid off**

A ticket router uses a large model with a long prompt: 2,500 input tokens per ticket, 400 ms, 92% accuracy on the eval set.

3,000 labelled tickets train a LoRA on a 3B open model. The prompt shrinks to 150 tokens because the instructions are now in the weights. Accuracy on the same eval: 93%, latency 60 ms, cost per ticket down by over 90%.

The fine-tune didn't make the model smarter. It made a narrow, well-defined job cheap.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

prompt first, retrieve for knowledge, fine-tune for behaviour, and use preference training to shape what "good" means. Each rung costs more than the one before.

</div>
