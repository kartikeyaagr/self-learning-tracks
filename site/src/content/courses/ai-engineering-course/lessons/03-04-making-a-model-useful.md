---
id: "03.04"
course: "ai-engineering-course"
module: "03"
slug: "making-a-model-useful"
title: "Making a model useful"
keyIdeas: ["A base model only continues text. Several techniques make it useful, at very different costs: climb the ladder from the cheapest rung.", "Prompting and few-shot examples change behaviour per call with no training; retrieval adds fresh or private knowledge at query time.", "Supervised fine-tuning teaches format, tone and narrow skills; it is a poor way to add facts that change.", "Preference training (RLHF, DPO) teaches which of two answers is better, shaping helpfulness and safety.", "LoRA trains small low-rank adapters instead of all weights, so a fine-tune costs megabytes and a fraction of the memory."]
question: "A support bot answers in the wrong tone and also doesn't know your product's latest pricing. Which technique fixes each problem, and why not the other way round?"
answer: "Stale pricing is a knowledge problem: use retrieval to put the current price page into the prompt at query time, so the answer changes the moment the page does and can cite it. Fine-tuning would bake in prices that go stale, cannot be updated without retraining, and can blend old and new figures. Wrong tone is a behaviour problem: fix it first with a clearer system prompt and a few example replies; if an eval still shows inconsistency, a small supervised fine-tune (for example LoRA on a few hundred good replies) makes the tone stick and shortens the prompt. Retrieval does not reliably change style, because the model treats retrieved text as evidence, not as instructions on how to sound."
rubric: ["Classifies pricing as a knowledge problem and tone as a behaviour problem", "Uses retrieval for pricing, with the reason that facts change and should be current and citable", "Starts tone fixes with prompting and examples before fine-tuning", "Explains why fine-tuning facts goes stale or blends", "Explains why retrieval does not reliably change style", "Mentions measuring with an eval before and after"]
sources: [{"title": "Ouyang et al.: InstructGPT (2022)", "url": "https://arxiv.org/abs/2203.02155", "note": "SFT plus RLHF: a 1.3B instruct model preferred over 175B GPT-3."}, {"title": "Hu et al.: LoRA (2021)", "url": "https://arxiv.org/abs/2106.09685", "note": "Low-rank adapters for cheap fine-tuning."}, {"title": "Dettmers et al.: QLoRA (2023)", "url": "https://arxiv.org/abs/2305.14314", "note": "Fine-tune adapters on top of a 4-bit frozen base model."}, {"title": "OpenAI: Optimizing LLM accuracy", "url": "https://platform.openai.com/docs/guides/optimizing-llm-accuracy", "note": "Context optimisation versus LLM optimisation."}]
practice: ["lora-finetune-memory", "lora-finetune-and-merge", "build-assistant-only-sft-loss-mask", "dpo-closed-form-loss", "instructgpt-rm-and-ppo-ptx"]
---

## From text predictor to assistant

A pretrained model has broad knowledge but no idea it is supposed to help you. InstructGPT (Ouyang et al., 2022) showed how much the next steps matter: their 1.3B model after supervised fine-tuning and RLHF was preferred by human raters over the raw 175B GPT-3. Labs do this post-training for you ([How LLMs learn](lesson:aieng:03.02)); this lesson is about the same tools in your hands.

## The ladder of techniques

~~~dg-layers Try the cheaper rungs first. Each one up costs more and changes the model more.
Prompting | instructions, few-shot examples, system prompt
*Retrieval (RAG) | inject relevant, current documents at query time
Fine-tuning (SFT / LoRA) | train on input/output examples for style, format, narrow skills
Preference training | RLHF / DPO on pairs of better and worse answers
Pretraining | new base model: only for labs
~~~

OpenAI's accuracy guide frames the choice as two axes: **context optimisation** (what the model needs to know, solved with prompts and retrieval) and **LLM optimisation** (how the model needs to act, solved with fine-tuning). Many systems need both.

| Rung | Changes | Time to first result | Updating it |
| --- | --- | --- | --- |
| Prompting | Behaviour per call | Minutes | Edit text |
| Retrieval | What the model knows right now | Days | Re-index documents |
| SFT / LoRA | Default behaviour, format, narrow skill | Days to weeks, needs labelled examples | Retrain |
| Preference training | Which answers it prefers | Weeks, needs comparison data | Retrain |

## Prompting and retrieval

The first two rungs need no training. A clear system prompt with a few well-chosen examples fixes most format and tone problems, and it is the baseline any fine-tune has to beat ([Prompting that holds up](lesson:aieng:03.06)). Retrieval puts documents into the prompt at query time, so the model can answer about private or fresh data and cite it ([Retrieval-augmented generation](lesson:aieng:03.05)).

## Supervised fine-tuning

Collect examples of the input and the ideal output, and continue training on them, with the loss only on the response tokens. It is good for:

- consistent tone and format;
- a narrow task the model does poorly with prompting alone;
- shortening prompts, because behaviour learned in the weights no longer has to be described on every call;
- making a small, cheap model match a big one on one job (distillation, [Synthetic data and distillation](lesson:aieng:07.08)).

It is bad at adding facts that change: the knowledge goes stale, and the model can blend old and new versions wrongly.

## LoRA: cheap fine-tuning

Full fine-tuning updates every weight and needs optimiser state for all of them, so it takes several times the model's size in GPU memory. LoRA (Hu et al., 2021) freezes the original matrix $W$ and learns a low-rank update next to it:

$$
W' = W + \frac{\alpha}{r} BA, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k}
$$

Only $A$ and $B$ train. It works because the change fine-tuning makes to a weight matrix tends to have low intrinsic rank. A 4,096 × 4,096 matrix has 16.8M entries; with $r = 16$, $B$ and $A$ together hold $2 \times 4{,}096 \times 16 = 131{,}072$, about 0.8% of the size. Hu et al. reported LoRA on GPT-3 175B with 10,000× fewer trainable parameters and about 3× less GPU memory than full fine-tuning, at similar quality. Adapters are megabytes, so one base model can serve many per-customer adapters, and QLoRA (Dettmers et al., 2023) goes further by keeping the frozen base in 4-bit.

~~~dg-raw
%%widget lora Trainable parameters and training memory: full fine-tuning versus LoRA.
~~~

Rank, alpha, target modules, data size and the memory arithmetic for real models are in [LoRA and QLoRA in practice](lesson:aieng:07.07).

## Preference training

SFT teaches what a good answer looks like. Preference training teaches which of two answers is better. RLHF trains a reward model on human comparisons and optimises the LLM against it with reinforcement learning, with a penalty for drifting from the original model. DPO skips the reward model and trains directly on the preference pairs, which is simpler and now common. Reasoning models add RL on tasks with checkable answers ([Reasoning models and human feedback](lesson:aieng:08.01)); the product view of DPO and graders is in [Preference tuning for products](lesson:aieng:07.09).

## When to climb

Get the best result you can with prompting and retrieval, build an eval set, and fine-tune only when the eval shows a gap that more prompting does not close; the same eval then tells you whether the fine-tune helped. The full decision, including when long context or a combination wins and what each costs at different volumes, is in [Prompt, RAG, fine-tune or long context?](lesson:aieng:07.06).

<div class="ex">
<div class="lab">Worked example</div>

**a classification fine-tune that paid off (illustrative numbers)**

A ticket router uses a large model with a long prompt: 2,500 input tokens per ticket, 400 ms, 92% accuracy on the eval set.

3,000 labelled tickets train a LoRA on a 3B open model. The prompt shrinks to 150 tokens because the instructions are now in the weights, a 94% cut in input tokens before any price difference between the models. Accuracy on the same eval: 93%, latency 60 ms, cost per ticket down by over 90%.

The fine-tune did not make the model smarter. It made a narrow, well-defined job cheap. The same move would have failed for a question-answering bot over changing docs, where the facts, not the behaviour, are the problem.

</div>

## Common mistakes

- Fine-tuning to teach product facts, then retraining every time they change.
- Fine-tuning before writing an eval, so nobody can say whether it helped or what it broke.
- Training on a few hundred inconsistent examples: the model learns the inconsistency.
- Skipping the prompt baseline, then crediting the fine-tune with gains a better prompt would have given.

<div class="take">
<div class="lab">Key takeaway</div>

Prompt first, retrieve for knowledge, fine-tune for behaviour, and use preference training to shape what "good" means. Each rung costs more than the one before, so climb only when an eval shows the rung below has run out.

</div>
