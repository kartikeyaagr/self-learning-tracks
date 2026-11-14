---
id: "01.01"
course: "ai-engineering-course"
module: "01"
slug: "what-ai-engineering-is"
title: "What AI engineering is"
keyIdeas: ["AI engineering means building products on top of pretrained foundation models instead of training models from scratch.", "The model is a dependency you call over an API or host yourself. Most of the work is around it: prompts, context, retrieval, tools, evaluation, cost and latency.", "ML engineering starts from data and a training loop. AI engineering starts from a product need and a model that already mostly works.", "Because outputs are open-ended text, evaluation is the hard part. You can't check a paragraph with accuracy on a test set.", "The core loop: prototype with a strong model, build an eval set, then optimise prompt, context, model choice and serving.", "You still need to understand internals (attention, KV cache, quantization) to reason about cost, latency and failure modes."]
question: "How is an AI engineer's job different from an ML engineer's? Give two concrete differences."
answer: "ML engineers mostly train and tune models from data; AI engineers start from a pretrained foundation model and adapt it with prompts, retrieval, tools and sometimes fine-tuning. Evaluation is also different: ML work measures accuracy on labelled data, while AI engineering has to judge open-ended outputs with rubrics, LLM judges or human review. A third difference is cost: inference per token dominates instead of training compute."
sources: [{"title": "Chip Huyen: AI Engineering (O'Reilly, 2025)", "url": "https://www.oreilly.com/library/view/ai-engineering/9781098166298/", "note": "The book that defined the field: building apps on top of foundation models."}, {"title": "Latent Space: The Rise of the AI Engineer", "url": "https://www.latent.space/p/ai-engineer", "note": "swyx's essay that named the role and its split from ML engineering."}, {"title": "Anthropic: Building effective agents", "url": "https://www.anthropic.com/engineering/building-effective-agents", "note": "What production LLM systems actually look like."}]
---

## A new kind of engineering

Until around 2022, using machine learning in a product meant collecting data, training a model and serving it. Foundation models changed the order. A single pretrained model can now summarise, classify, write code and answer questions with no task-specific training. Chip Huyen defines AI engineering as the process of building applications on top of these models. swyx's essay "The Rise of the AI Engineer" argued the role would become as common as the backend engineer, because the skills needed are closer to software engineering than to research.

## What changes compared with ML engineering

| Aspect | ML engineering | AI engineering |
| --- | --- | --- |
| Starting point | A dataset and a training objective | A product need and a pretrained model |
| Main lever | Model architecture and training data | Prompts, context, retrieval, tools, model choice |
| Output | Usually a label or a number | Open-ended text, code, images, actions |
| Evaluation | Accuracy, F1 on a held-out set | Rubrics, LLM judges, human review, task success |
| Cost driver | Training compute | Inference: tokens in and out, per request |

The model is now a dependency. You can call it through an API or host an open-weights model, but in both cases most of your engineering effort goes into everything around it.

## The work around the model

- **Prompt and context**: what instructions and information the model sees on each call.
- **Retrieval**: finding the right documents to put in the context (embeddings, vector search).
- **Tools and agents**: letting the model call functions, read results and decide what to do next.
- **Evaluation**: deciding whether a change made the system better or worse.
- **Serving and optimisation**: latency, throughput and cost, which is where internals like the KV cache and quantization matter.

~~~dg-flow The usual order of work: start with the best model, measure, then optimise.
Product need -(prototype)-> Strong model + prompt -(measure)-> *Eval set -(iterate)-> Context, retrieval, tools -(optimise)-> Cheaper / faster model
~~~

## Why internals still matter

You can ship a demo without knowing what attention is. You can't explain why a 100k-token prompt is slow and expensive, why a quantized model gets worse at maths, or why your agent forgets an instruction from early in the conversation. This course goes from the application layer (embeddings, retrieval) into the model (transformers, attention, KV cache), out to optimisation (quantization, distillation, speculative decoding) and back up to reasoning and agents.

## What the job looks like day to day

A useful way to see AI engineering is to follow one feature from idea to production. Say a support team wants a "suggest a reply" button.

1. **Frame the task.** Write down the input (the ticket thread plus customer record), the output (a draft reply under 150 words, in the company tone) and what "good" means (correct facts, no promises the policy forbids, a human edits it less than half the time).
2. **Build a baseline in an afternoon.** One prompt, one model call, ten real tickets pasted in by hand. Read every output. Most of what you learn about the problem comes from this step.
3. **Collect an eval set.** 50 to 200 real tickets with notes on what a good reply contains. This is the test suite you will run after every change.
4. **Add context.** The baseline invents refund rules, so you retrieve the relevant policy sections (RAG, module 3) and the last three orders.
5. **Harden.** Structured output so the UI can parse it, a timeout and fallback model, logging of every prompt and response, and a check that blocks replies containing a refund amount.
6. **Measure in production.** Track the edit distance between the draft and what the agent actually sent. That number tells you whether the next change helped.

None of these steps trains a model. Every one of them decides whether the feature works.

## The failure modes are different

Traditional software fails loudly: an exception, a 500, a failing test. An LLM feature fails quietly. It returns fluent, confident text that is wrong 8% of the time, and nobody notices until a customer complains. That changes how you build:

- **Tests become evals.** You can't assert an exact string. You score outputs with rules, reference answers, or a second model acting as a judge, and you look at the pass rate across many cases.
- **Changes are statistical.** A prompt edit that fixes three examples may break five others. You only know by re-running the whole set.
- **Behavior drifts without a deploy.** A provider updates a model alias, or users start asking a new kind of question, and quality moves while your code is unchanged. Pin model versions and keep sampling production traffic into your evals.

<div class="ex">
<div class="lab">Worked example</div>

**a prompt change that looked like a win**

You add "Always apologise for the inconvenience" to the system prompt. On the five tickets you tried, replies feel warmer.

The 120-case eval shows the pass rate fell from 81% to 74%: replies to billing questions now open with an apology and run past the 150-word limit, and three replies apologised for an outage that never happened.

Without the eval set you would have shipped it.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

AI engineering is building products around a pretrained model. The hard parts are context, evaluation and cost, and understanding the model's internals is what lets you reason about all three.

</div>
