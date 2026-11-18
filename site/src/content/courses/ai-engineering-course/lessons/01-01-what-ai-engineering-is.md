---
id: "01.01"
course: "ai-engineering-course"
module: "01"
slug: "what-ai-engineering-is"
title: "What AI engineering is"
keyIdeas: ["AI engineering means building products on top of pretrained foundation models instead of training models from scratch.", "The model is a dependency you call over an API or host yourself. Most of the work is around it: prompts, context, retrieval, tools, evaluation, cost and latency.", "Climb the build ladder one rung at a time: prompt a strong model, add context (RAG, tools), and fine-tune only when an eval says the lower rungs are not enough.", "LLM features fail quietly with fluent wrong answers, so tests become evals scored over many cases, and every change is a statistical question.", "AI engineer interviews usually add an LLM-application design round and an evals discussion to the coding and system design rounds."]
question: "How is an AI engineer's job different from an ML engineer's? Give two concrete differences, and say where fine-tuning fits."
answer: "An ML engineer usually starts from a dataset and a training objective, and the main lever is the model and its training data. An AI engineer starts from a product need and a pretrained foundation model, and the main levers are prompts, context (retrieval, tools) and model choice. Evaluation differs too: ML work measures accuracy or F1 on labelled data, while AI engineering judges open-ended text with rules, reference answers, LLM judges and human review, across a whole eval set, because a single output can be fluent and wrong. A third difference is cost: inference tokens per request dominate rather than training compute. Fine-tuning is the top rung of the ladder, not the starting point: you reach for it after prompting and retrieval have been measured and fall short, usually to fix behaviour or format, or to make a smaller model good enough."
rubric: ["Names the different starting point: dataset and training loop vs product need and a pretrained model.", "Names the evaluation difference: labelled accuracy vs scoring open-ended outputs over an eval set.", "Mentions cost shifting to per-request inference tokens, or another concrete difference such as the main lever.", "Places fine-tuning after prompting and retrieval, justified by an eval, not as the default first step."]
sources: [{"title": "Chip Huyen: AI Engineering (O'Reilly, 2025)", "url": "https://www.oreilly.com/library/view/ai-engineering/9781098166298/", "note": "The book that defined the field: building apps on top of foundation models."}, {"title": "Latent Space: The Rise of the AI Engineer", "url": "https://www.latent.space/p/ai-engineer", "note": "swyx's essay that named the role and its split from ML engineering."}, {"title": "Anthropic: Building effective agents", "url": "https://www.anthropic.com/engineering/building-effective-agents", "note": "What production LLM systems actually look like."}]
practice: ["rag-pipeline-capstone", "rag-context-packing", "react-agent-capstone"]
---

## A new kind of engineering

Until around 2022, using machine learning in a product meant collecting data, training a model and serving it. Foundation models changed the order. A single pretrained model can now summarise, classify, write code and answer questions with no task-specific training. Chip Huyen defines AI engineering as the process of building applications on top of these models. swyx's essay "The Rise of the AI Engineer" argued the role would become as common as the backend engineer, because the skills needed are closer to software engineering than to research.

| Aspect | ML engineering | AI engineering |
| --- | --- | --- |
| Starting point | A dataset and a training objective | A product need and a pretrained model |
| Main lever | Model architecture and training data | Prompts, context, retrieval, tools, model choice |
| Output | Usually a label or a number | Open-ended text, code, images, actions |
| Evaluation | Accuracy, F1 on a held-out set | Rubrics, LLM judges, human review, task success |
| Cost driver | Training compute | Inference: tokens in and out, per request |

The model is now a dependency. You can call it through an API or host an open-weights model, but in both cases most of your engineering effort goes into everything around it: what the model sees on each call, which documents you retrieve, which tools it may call, how you decide a change helped, and what each request costs and how long it takes.

## The build ladder

Most features climb the same ladder, and the course follows it. Each rung costs more to build and maintain than the one below, so you climb only when an eval shows the current rung falls short.

~~~dg-flow The usual order of work: start with the best model, measure, then climb only as far as you must.
Product need -(prototype)-> Strong model + prompt -(measure)-> *Eval set -(add context)-> RAG, tools, agents -(if still short)-> Fine-tune -(optimise)-> Cheaper / faster model
~~~

| Rung | What it fixes | Where the course covers it |
| --- | --- | --- |
| Prompt a strong model | Task framing, format, tone | [Prompting that holds up](lesson:aieng:03.06), [Choosing a model](lesson:aieng:01.05) |
| Add context (RAG) | Missing or private knowledge, freshness | Module 2, [RAG](lesson:aieng:03.05), module 11 |
| Add tools and agents | Actions, multi-step work, live data | Modules 8 and 9 |
| Fine-tune | Behaviour, style, format at scale, a small model that must act like a big one | [Prompt, RAG, fine-tune or long context?](lesson:aieng:07.06) |
| Optimise serving | Cost and latency once quality is fixed | [What your provider optimises](lesson:aieng:06.05), module 12 |

Evals ([module 10](lesson:aieng:10.01)) sit beside every rung: they are how you know whether to climb.

You can ship a demo without knowing what attention is. You can't explain why a 100k-token prompt is slow and expensive, why a quantized model gets worse at maths, or why your agent forgets an instruction from early in the conversation. That is why the course also goes inside the model (modules 4 and 5, from [attention](lesson:aieng:04.02) to the [KV cache](lesson:aieng:04.04)) before looking at what your provider does to serve it cheaply.

## One feature, idea to production

Follow one feature through the ladder. A support team wants a "suggest a reply" button.

1. **Frame the task.** Input: the ticket thread plus customer record. Output: a draft reply under 150 words, in the company tone. Good means correct facts, no promises the policy forbids, and a human edits it less than half the time.
2. **Build a baseline in an afternoon.** One prompt, one model call, ten real tickets pasted in by hand. Read every output. Most of what you learn about the problem comes from this step.
3. **Collect an eval set.** 50 to 200 real tickets with notes on what a good reply contains. This is the test suite you run after every change.
4. **Add context.** The baseline invents refund rules, so you retrieve the relevant policy sections and the last three orders.
5. **Harden.** Structured output so the UI can parse it, a timeout and fallback model, logging of every prompt and response, and a check that blocks replies containing a refund amount.
6. **Measure in production.** Track the edit distance between the draft and what the agent actually sent. That number tells you whether the next change helped.

None of these steps trains a model. Every one of them decides whether the feature works.

## Where this breaks: quiet failures

Traditional software fails loudly: an exception, a 500, a failing test. An LLM feature fails quietly. It returns fluent, confident text that is wrong 8% of the time, and nobody notices until a customer complains. That changes how you build:

- **Tests become evals.** You can't assert an exact string. You score outputs with rules, reference answers, or a second model acting as a judge, and you look at the pass rate across many cases.
- **Changes are statistical.** A prompt edit that fixes three examples may break five others. You only know by re-running the whole set.
- **Behaviour drifts without a deploy.** A provider updates a model alias, or users start asking a new kind of question, and quality moves while your code is unchanged. Pin model versions and keep sampling production traffic into your evals.

<div class="ex">
<div class="lab">Worked example</div>

**A prompt change that looked like a win**

You add "Always apologise for the inconvenience" to the system prompt. On the five tickets you tried, replies feel warmer.

The 120-case eval shows the pass rate fell from 81% to 74% (97 to 89 passing cases): replies to billing questions now open with an apology and run past the 150-word limit, and three replies apologised for an outage that never happened.

Without the eval set you would have shipped it. With only five cases, a drop of this size is invisible; [Building an eval set](lesson:aieng:10.02) shows how many cases you need to see it.

</div>

## The interview loop

AI engineer interviews test the same skills. The exact loop varies by company, but most combine these rounds:

| Round | What you are asked | What they look for |
| --- | --- | --- |
| Coding | Practical code: call a model API, parse and validate output, write retrieval or scoring code; sometimes classic algorithm questions | Clean, tested code that handles bad model output |
| System design | A general backend design (queues, caches, storage) | Standard scaling trade-offs |
| LLM application design | "Design a support assistant over our docs" or "an agent that triages tickets" | The ladder above, retrieval choices, cost and latency budgets, failure handling |
| Evals discussion | "How would you know it works? How would you catch a regression?" | Eval sets, graders and judges, online metrics, rollout |
| Project deep dive | A system you built | Measured decisions, not a tour of tools |

The design and evals rounds are where AI engineering differs most from backend work. The rest of this course is built to prepare you for both, and [ML System Design](lesson:mlsd:11.03) covers the interview framing of the same RAG problem.

<div class="take">
<div class="lab">Key takeaway</div>

AI engineering is building products around a pretrained model. Climb the ladder (prompt, context, tools, fine-tune) only as far as an eval says you must, and treat evaluation, cost and latency as the real work.

</div>
