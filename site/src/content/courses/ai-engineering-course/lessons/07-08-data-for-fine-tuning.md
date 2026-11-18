---
id: "07.08"
course: "ai-engineering-course"
module: "07"
slug: "data-for-fine-tuning"
title: "Data for fine-tuning: synthetic data and distillation"
keyIdeas: ["Distillation trains a small student on a large teacher: its soft probabilities (logit distillation), its generated answers (sequence-level), or its reasoning traces.", "Soft targets carry how-wrong-is-wrong information; temperature T above 1 makes it visible, and the KL term is multiplied by T squared so its gradients keep their scale.", "Most app teams distil through text: a frontier model labels or writes outputs for real inputs, and the student is fine-tuned on them. Check the teacher's terms and licence first.", "Synthetic data is only as good as its filters: verify or judge every example, deduplicate exactly and approximately, and decontaminate against your eval set, including paraphrases.", "Keep real inputs as the anchor; training on generated data alone narrows the distribution and inherits the teacher's mistakes."]
question: "You want a 4B open model to replace a frontier model for classifying and summarizing support tickets. You have 200,000 unlabelled tickets and a 600-ticket human-labelled eval set. Design the data pipeline, explain which kind of distillation you can use and why, and list the checks that keep the student's eval scores honest."
answer: "Use sequence-level distillation: the frontier model's API returns text, not full token distributions, so have it label and summarize a sample of real tickets with a carefully tested prompt (logit distillation would need the teacher's probabilities and the same tokenizer). Sample by category and length so rare classes are covered; generate 20,000 to 30,000 examples. Filter: schema and label validity checks, a judge or second-teacher agreement check, and exact plus MinHash near-duplicate removal so repeated templates do not dominate. Decontaminate: remove any training ticket that matches an eval ticket by n-gram overlap or embedding similarity, since paraphrases slip past n-gram checks. Then fine-tune with assistant-only loss, compare student and teacher on the 600 human labels per category, and inspect where they disagree. Before starting, confirm the teacher's terms allow training a model on its outputs, or use an open-weights teacher whose licence permits distillation."
rubric: ["Picks sequence-level (output) distillation and explains why logit distillation needs teacher probabilities and a shared tokenizer", "Uses real inputs, sampled for coverage, rather than fully invented prompts", "Includes verification or judge filtering plus exact and near-duplicate removal", "Decontaminates against the eval set, including paraphrases", "Evaluates the student against human labels per slice and checks provider terms or licence"]
sources: [{"title": "Hinton, Vinyals and Dean: Distilling the Knowledge in a Neural Network (2015)", "url": "https://arxiv.org/abs/1503.02531", "note": "Soft targets, temperature and the T-squared scaling."}, {"title": "DeepSeek-AI: DeepSeek-R1 (2025)", "url": "https://arxiv.org/abs/2501.12948", "note": "Reasoning traces from a large model distilled into 1.5B to 70B Qwen and Llama students."}, {"title": "Wang et al.: Self-Instruct (2022)", "url": "https://arxiv.org/abs/2212.10560", "note": "Bootstrapping instruction data from a model's own generations, with filtering."}, {"title": "Yang et al.: Rethinking Benchmark and Contamination for Language Models with Rephrased Samples (2023)", "url": "https://arxiv.org/abs/2311.04850", "note": "Paraphrased test items evade n-gram decontamination; contamination found in GPT-generated synthetic data."}]
practice: ["kl-divergence-discrete-and-continuous", "text-deduplication-minhash", "perform-stable-exact-deduplication", "temperature-based-data-mixtures", "build-assistant-only-sft-loss-mask"]
tools: ["DeepSeek-R1", "Qwen3", "gpt-oss"]
tags: ["Fine-tuning", "Distillation", "Data"]
---

## Why the data decides the fine-tune

[LoRA and QLoRA in practice](lesson:aieng:07.07) makes training cheap; what is left is the data. For app teams the data almost always comes from a bigger model, either as labels and answers for real inputs (distillation) or as invented inputs and outputs (synthetic data). The target is usually a small model of 1 to 8B parameters, which is cheaper per token, faster, runs on one GPU or a device, and is easy to keep updated. On a narrow task, trained well, it can match a frontier model. This lesson covers how to get that data, how to clean it, and the loss behind classic distillation.

## Soft targets and temperature

Hinton, Vinyals and Dean noticed that a trained model's output distribution says more than its top answer. A ticket the teacher calls "billing" with some weight on "refund" and almost none on "spam" tells the student that billing and refunds are related. A hard label throws that away. They called this information dark knowledge, and trained the student to match the teacher's whole distribution, softened by a temperature $T$:

$$
\mathcal{L} = \alpha\, T^2\, \mathrm{KL}\!\left(\mathrm{softmax}(z_t / T)\,\|\,\mathrm{softmax}(z_s / T)\right) + (1 - \alpha)\, \mathrm{CE}(y, z_s)
$$

$z_t$ and $z_s$ are teacher and student logits and $y$ is the true label. $T > 1$ flattens both distributions so small probabilities become large enough to learn from. The gradient of the soft term shrinks like $T^{-2}$, so multiplying by $T^2$ keeps its weight steady as you change $T$. Common settings are $T$ between 2 and 5 and $\alpha$ around 0.5. [KL divergence](lesson:air-fanout-v1:01.12) has the background.

<div class="ex">
<div class="lab">Worked example</div>

**Four ticket classes: billing, refund, account, spam.** Teacher logits $z_t = (6, 3, 2, -1)$; an early student has $z_s = (4, 3, 1, 0)$.

| $T$ | Teacher probabilities | Student probabilities | KL | $T^2 \times$ KL |
| --- | --- | --- | --- | --- |
| 1 | 0.935, 0.047, 0.017, 0.001 | 0.696, 0.256, 0.035, 0.013 | 0.182 | 0.182 |
| 2 | 0.720, 0.161, 0.097, 0.022 | 0.509, 0.309, 0.114, 0.069 | 0.105 | 0.421 |
| 4 | 0.497, 0.235, 0.183, 0.086 | 0.382, 0.297, 0.180, 0.140 | 0.035 | 0.561 |

At $T = 1$ the teacher's view that refund is about three times as plausible as account is buried under 0.935. At $T = 4$ the ranking of wrong answers is plain, and the student is pushed to put less on refund and spam relative to the teacher's shape. Without the $T^2$ factor the soft term would shrink from 0.18 to 0.035 and be swamped by the hard-label loss.

</div>

DistilBERT used this recipe on language and kept about 97% of BERT's GLUE score with 40% fewer parameters, running 60% faster.

## Three ways to distil an LLM

| Flavour | Student learns from | Needs | Example |
| --- | --- | --- | --- |
| Logit distillation | Teacher's next-token distribution at every position | Teacher logits and the same tokenizer | DistilBERT; open-weights teacher and student pairs |
| Sequence-level | Teacher's generated answers, by ordinary SFT | Only text, so any API works | Labelling tickets with a frontier model, then fine-tuning a small one |
| Reasoning traces | Teacher's chains of thought plus answers | A reasoning teacher that shows its reasoning | DeepSeek-R1 distilled into six dense students from 1.5B to 70B on about 800k curated samples |

Hsieh et al.'s "Distilling Step-by-Step" trained small models on the teacher's rationales as an extra target. A 770M T5 beat few-shot 540B PaLM on a benchmark while using less of the training data. DeepSeek reported that its distilled students beat small models trained with RL directly. Reasoning is easier to copy than to discover at small scale.

Sequence-level distillation has one structural weakness. The student only sees states the teacher visits. Once it makes an early mistake the teacher never made, it is in unfamiliar territory, and errors compound over long outputs. **On-policy distillation** (Agarwal et al.'s GKD; Thinking Machines' 2025 write-up) fixes this. The student generates, the teacher scores every token the student produced, and training minimizes per-token reverse KL. It needs teacher log-probabilities, so in practice an open-weights teacher, and it is far cheaper than RL for the same target.

**Terms and licences.** Several API providers' terms forbid using outputs to develop competing models, and the wording changes; read the current terms of the exact service. Open-weights teachers vary too: DeepSeek-R1's licence explicitly permits distillation, and gpt-oss and Qwen3 ship under Apache 2.0.

## Building a synthetic dataset

~~~dg-flow A synthetic-data pipeline; most examples die in the filters.
Real inputs + seed tasks -(teacher generates)-> Candidates -(verify or judge)-> Kept -(exact + near-duplicate removal)-> Unique -(decontaminate vs eval)-> *Training set -(SFT)-> Student
~~~

**Inputs.** Prefer real inputs from logs (anonymised) and have the teacher produce outputs; invented inputs drift toward what the teacher finds easy. When you must invent them, generate with structure. Self-Instruct bootstraps new tasks from a small seed set and filters near-duplicates. Evol-Instruct rewrites prompts to be harder. Magpie samples user turns from an aligned model's chat template. Persona or attribute prompts spread topics, lengths and difficulty. Sample by slice so rare categories get enough examples.

**Outputs.** Generate with your best prompt, at moderate temperature, sometimes several candidates per input. **Rejection sampling** keeps only candidates that pass a check: code that runs its tests, maths that matches a reference, JSON that validates, or a judge score above a threshold ([graders and LLM-as-judge](lesson:aieng:10.03)).

**Deduplicate.** Exact hashing removes repeats. MinHash with locality-sensitive hashing removes near-duplicates such as templated tickets that differ by a name. Without it, a few frequent templates dominate the loss.

**Decontaminate.** Remove training examples that overlap your eval and benchmark items. N-gram overlap (13-grams is a common choice) catches copies. Yang et al. showed that paraphrased or translated test items slip past it and can lift a 13B model to GPT-4-level benchmark scores. They also found that synthetic data generated by GPT-3.5 and GPT-4 can be contaminated. Add an embedding-similarity or LLM-based check for anything near the eval set.

<div class="ex">
<div class="lab">Worked example</div>

**A ticket-summary dataset, illustrative funnel and cost.** 30,000 real tickets are sampled across 12 categories. A teacher at \$3 per million input and \$15 per million output tokens writes a label and summary for each (600 tokens in, 700 out): \$54 + \$315 = \$369. A cheaper judge at \$1 and \$5 checks each (1,500 tokens in, 100 out): \$45 + \$15 = \$60.

| Stage | Examples left | Removed because |
| --- | --- | --- |
| Generated | 30,000 | |
| Schema and label valid | 29,400 | malformed output, label outside the set |
| Judge score at least 4 of 5 | 24,700 | unfaithful or vague summaries |
| Exact and near-duplicate removal | 21,300 | templated tickets |
| Decontaminated against the 600-ticket eval | 21,180 | 120 near matches by embedding similarity |

About \$430 buys 21,000 clean examples. The 600 human-labelled tickets stay out of training and decide whether the student ships.

</div>

## Where this breaks

- **The student inherits the teacher's errors and biases.** A teacher that is 92% accurate on a category caps the student near that level. Spot-check teacher labels against humans before generating at scale.
- **Collapse from recursion.** Shumailov et al. (Nature, 2024) showed that models trained repeatedly on generated data lose the tails of the original distribution. Keep real inputs and some human-written outputs in every round.
- **Breadth loss.** A student distilled on tickets is good at tickets and worse at everything else. Keep the general model for traffic outside the task.
- **Judge-pleasing data.** If the same model generates and judges, the filter rewards its own style. Use a different judge, calibrated on human grades.
- **Leaky evals.** Generating the eval set with the same prompt and teacher as the training set makes the student look better than it is.

## What interviewers probe

- Why soft targets carry more information than hard labels, what temperature does, and why the loss has $T^2$.
- Logit versus sequence-level versus on-policy distillation, and what each needs from the teacher.
- How you filter, deduplicate and decontaminate synthetic data, and why n-gram matching is not enough.
- Legal and licence checks before distilling from an API model.

<div class="take">
<div class="lab">Key takeaway</div>

Distillation turns a large model's outputs into a small model's training data. Use soft targets when you have the teacher's probabilities and generated text when you only have an API. The quality comes from the pipeline: real inputs, verified outputs, deduplication, decontamination against your eval, and a human-labelled test set that never enters training.

</div>
