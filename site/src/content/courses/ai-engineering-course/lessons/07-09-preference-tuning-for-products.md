---
id: "07.09"
course: "ai-engineering-course"
module: "07"
slug: "preference-tuning-for-products"
title: "Preference tuning for products"
keyIdeas: ["SFT needs the ideal output; DPO needs only which of two outputs is better; reinforcement fine-tuning (RFT) needs only a grader that scores an output. Pick by what your team can actually produce.", "DPO fits taste and policy (concision, tone, when to refuse, which candidate to show), trained from pairs collected from edits, side-by-side ratings or judge-ranked samples, starting from an SFT model.", "RFT fits checkable tasks where the model already succeeds sometimes: it samples several answers per prompt, scores them with your grader, and pushes toward the better ones (GRPO).", "Prompts the model always or never solves carry no signal in group-relative RL; filter by base pass rate first.", "Every grader and judge can be gamed. Keep a held-out eval with a different grader, read samples at each checkpoint, and watch length, format and refusal rates."]
question: "Your SQL-generation feature produces correct queries 55% of the time on a held-out set of 400 analyst questions, and analysts also complain the explanations are long-winded. You have a sandbox database with reference answers and a log of 3,000 explanation edits analysts made. Which tuning methods would you use for each problem, what data and grader would each need, and what could go wrong?"
answer: "Correctness is checkable, so use reinforcement fine-tuning: prompts are analyst questions, the grader runs the generated SQL in the sandbox and compares the result set to the reference (1 for a match, 0 otherwise, with a small penalty for invalid SQL). A 55% base pass rate is a good regime, since groups of samples contain both successes and failures; drop questions the model always or never solves. Start with a few hundred prompts and hold out the 400 for evaluation. Explanation length is a preference: build DPO pairs from the edit log, the analyst's edited explanation as chosen and the original as rejected, train from the current SFT model with a beta around 0.1, and judge on held-out win rate with position-swapped comparisons plus a human sample. Risks: the SQL grader can be gamed by queries that match the reference by accident or by selecting everything, so add row and column checks and read samples; DPO can overshoot into terse or unhelpful answers, so track length and a helpfulness rubric."
rubric: ["Matches RFT to the checkable problem and DPO (or SFT on edits) to the taste problem", "Defines a concrete grader for SQL (execute and compare results) and notes how it could be gamed", "Uses the edit log as chosen versus rejected pairs and starts DPO from the SFT model", "Considers base pass rate and filters prompts with no learning signal", "Evaluates on held-out data with a different grader or human check, tracking side effects such as length"]
sources: [{"title": "Rafailov et al.: Direct Preference Optimization (2023)", "url": "https://arxiv.org/abs/2305.18290", "note": "Preference tuning as a classification loss on pairs, no reward model or RL loop."}, {"title": "Lambert et al.: Tulu 3 (2024)", "url": "https://arxiv.org/abs/2411.15124", "note": "An open SFT, then DPO, then RL-with-verifiable-rewards recipe, with data and code."}, {"title": "AWS: Reinforcement fine-tuning in Amazon Bedrock", "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/reinforcement-fine-tuning.html", "note": "A hosted GRPO service with Lambda or model-judge graders and start-small guidance."}, {"title": "Lilian Weng: Reward Hacking in Reinforcement Learning (2024)", "url": "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/", "note": "How policies exploit reward models and graders, with LLM examples."}]
practice: ["dpo-closed-form-loss", "dpo-reward-accuracy-margin", "canonicalize-pairwise-preference-data", "grpo-group-relative-advantage", "remove-response-length-bias-from-judge-scores"]
tools: ["Amazon Bedrock", "Tinker", "Hugging Face TRL"]
tags: ["Fine-tuning", "Preference tuning", "RL"]
---

## Three methods, three kinds of supervision

[Reasoning models and human feedback](lesson:aieng:08.01) explains how RLHF, DPO and GRPO work. The product question is different: which signal can your team produce, and which method turns it into a better model?

| Method | You provide | Best for | Not for |
| --- | --- | --- | --- |
| SFT | Ideal outputs | Formats, policies, skills you can demonstrate | Taste you cannot write down; tasks where good answers vary widely |
| DPO | Pairs: a better and a worse output for the same prompt | Tone, concision, refusal calibration, choosing among plausible answers | Correctness you could check automatically; missing knowledge |
| RFT | Prompts plus a grader that scores any output | Checkable tasks: extraction, code, SQL, maths, classification with a policy | Tasks the model never gets right; vague goals with no reliable grader |

The usual order is SFT first, then DPO or RFT on top. Tulu 3, a fully open recipe, runs exactly that sequence: SFT, then DPO, then RL with verifiable rewards. SFT gets the model into the right format, which preference and RL methods then sharpen. None of them adds knowledge ([prompt, RAG, fine-tune or long context](lesson:aieng:07.06)).

## DPO from the product side

DPO raises the probability of the chosen response relative to the rejected one, measured against a frozen reference model (usually your SFT model):

$$
\mathcal{L} = -\log \sigma\!\left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]\right)
$$

$\beta$ sets how far the model may move from the reference. OpenAI's DPO API exposed it as a value between 0 and 2; 0.1 is a common start. Lower values let the model move further.

**Where pairs come from in a product:**

- **Edits.** A human rewrote the model's draft: the edit is chosen and the original is rejected. This is the cleanest source because the pair differs only in what the human cared about.
- **Side-by-side ratings.** Show two candidates and log the pick. Randomize which side each appears on.
- **Regenerations and thumbs.** A thumbs-down followed by a regeneration the user kept gives a weak pair. These signals are noisy and skew toward fast, short answers.
- **Judge-ranked samples.** Sample several responses from your own SFT model and rank them with a rubric judge calibrated on human labels ([graders and LLM-as-judge](lesson:aieng:10.03)). Samples from the model you are tuning match what it actually produces, which pairs written by other models do not.

The format is one prompt, a preferred output and a non-preferred output. Hosted APIs typically accept single-turn pairs only. Clean the pairs: drop ties and identical pairs, put every pair in the same orientation, and balance lengths so that "longer" is not the hidden label.

## RFT: sampling against a grader

~~~dg-seq One reinforcement fine-tuning step for one prompt.
Trainer > Model: prompt, sample k answers
Model -> Trainer: k candidate answers
Trainer > Grader: score each answer
Grader -> Trainer: rewards between 0 and 1
Trainer > Trainer: advantage = reward minus group mean, divided by group std
Trainer > Model: update toward above-average answers, with a KL leash to the reference
~~~

You supply prompts, reference answers where you have them, and a grader. Graders come in a few types: exact or normalized string match, numeric tolerance, schema validation, executing code or SQL against tests, a model-graded rubric, or a weighted mix of these. Amazon Bedrock's managed RFT runs GRPO with a reward function in Lambda or a model judge. As of November 2026 it supports Nova 2 Lite, gpt-oss-20B and Qwen3 32B. Its guidance is to start with 100 to 200 examples and to fall back to SFT if rewards stay at zero. Tinker and open libraries (TRL, verl) run the same loop on open weights. OpenAI's RFT on o4-mini was retired in October 2026 along with the rest of its fine-tuning platform, a reminder to keep graders and data portable.

<div class="ex">
<div class="lab">Worked example</div>

**Group-relative advantages.** Eight samples for one SQL question get rewards $(1, 0, 0, 1, 0, 0, 0, 0)$. The mean is 0.25 and the standard deviation 0.433. Each correct sample gets advantage $(1 - 0.25)/0.433 = 1.73$, each wrong one $-0.58$. The update pushes hard toward the two successes.

If all eight are correct, or all eight wrong, the standard deviation is zero and the prompt contributes nothing while still costing eight samples and eight grader calls. At a 55% base pass rate, the chance that a group of 8 is all right or all wrong is $0.55^8 + 0.45^8 \approx 1\%$, so nearly every prompt teaches something. At a 97% pass rate that chance rises to 78%. Filter prompts by base pass rate before training.

**One DPO pair.** The policy gives the chosen answer log-probability $-42$ (reference $-45$) and the rejected answer $-50$ (reference $-48$). With $\beta = 0.1$ the margin is $0.1 \times [(-42 + 45) - (-50 + 48)] = 0.5$ and the loss is $-\log \sigma(0.5) = 0.474$. The implicit rewards are $+0.3$ for chosen and $-0.2$ for rejected. Reward accuracy (the share of pairs with a positive margin) and the mean margin are the training curves to watch. A held-out win rate decides whether to ship.

</div>

## Reward hacking is the default, not the exception

The policy optimizes the grader, not your intent. Lilian Weng's survey collects many cases. Product versions you will meet:

- **Judge biases.** A model judge that prefers longer, more confident or more flattering answers trains a verbose, sycophantic model. Correct for length and calibrate the judge on human labels.
- **Format gaming.** A string grader that searches for the answer anywhere rewards outputs that list every option. Parse a single final answer.
- **Test gaming.** A code grader that only runs visible tests rewards special-casing them. Use hidden tests and check that test files were not changed.
- **SQL that matches by accident.** A query that returns the right count with wrong rows passes a weak comparison. Compare full result sets, columns included.
- **Text aimed at the judge.** Outputs that address the judge ("this answer fully satisfies the rubric") can fool rubric graders. Strip or ignore such text and test the grader against adversarial outputs.

Defences: a KL penalty or a moderate $\beta$ keeps the model near its reference; a held-out eval uses a different grader from training; samples are read at every checkpoint; length, refusal rate and format errors are tracked as guardrail metrics; training stops when the held-out score peaks, even if training reward is still rising.

## Where this breaks

- **DPO lowers the chosen answer's probability too.** Both log-probabilities often fall while the margin grows. Watch generations, not just the loss.
- **Biased preference data.** Thumbs come from a few vocal users and favour speed; weight or filter by segment.
- **Refusal drift.** Pairs that reward helpfulness can erode refusals, and the reverse makes the model preachy. Keep a safety and over-refusal set in the eval.
- **Using RL where SFT would do.** If you can write the right answer, SFT on it is cheaper and more stable.

## What interviewers probe

- When DPO or RFT beats SFT, and why neither adds knowledge.
- Where preference pairs come from in a live product, and how you clean them.
- How you would design a grader for a given task, and how a model would game it.
- Why prompts with zero reward variance waste compute in GRPO.
- Which metrics decide shipping (held-out win rate or pass rate under a different grader), as opposed to training reward.

<div class="take">
<div class="lab">Key takeaway</div>

Choose the method by the signal you can produce: demonstrations for SFT, better-versus-worse pairs for DPO, a grader for RFT. Start from an SFT model, train on prompts where the model sometimes succeeds, and expect the grader to be gamed. A held-out eval with a different grader, plus reading samples, is what makes the result trustworthy.

</div>
