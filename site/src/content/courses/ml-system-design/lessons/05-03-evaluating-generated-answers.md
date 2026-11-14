---
id: "05.03"
course: "ml-system-design"
module: "05"
slug: "evaluating-generated-answers"
title: "Evaluating generated answers"
keyIdeas: ["Generated outputs rarely have one right answer, so evaluation mixes **reference-based** metrics, **rubric-based** human or model grading, **pairwise preference**, and **task success** checks.", "Build a versioned **evaluation set** from real traffic, with hard and adversarial cases, and keep it separate from anything used for prompts or training.", "**LLM-as-judge** scales grading but has biases: position, length, self-preference and sensitivity to the rubric wording. Calibrate it against human labels.", "In production, measure task outcomes (resolution rate, edits, thumbs, escalations), safety and cost per request, not just offline scores."]
question: "You are launching an LLM assistant that answers questions about a company's HR policies using retrieved documents. Design the evaluation: datasets, metrics, how you use model judges, and what you monitor after launch."
answer: "Datasets: 500 to 1,000 real employee questions sampled from search logs and HR tickets, stratified by topic, plus hand-written hard cases (ambiguous questions, policies that differ by country, questions with no answer in the documents, attempts to extract other employees' data). For each, HR experts write a reference answer and list the documents that support it. Metrics: retrieval recall of supporting documents; answer correctness against the reference graded on a rubric (correct, partially correct, wrong); faithfulness (every claim supported by retrieved text); appropriate refusal when the answer is absent; citation accuracy; and latency and cost. Use an LLM judge with a specific rubric and examples for correctness and faithfulness, validated against expert labels on a 200-item subset (target agreement above 85%), randomising answer order for pairwise comparisons and checking length bias. After launch, monitor thumbs-down and follow-up rates, escalations to HR, a weekly expert review of a random sample, refusal rate, and alerts for responses citing no document."
sources: [{"title": "Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (2023)", "url": "https://arxiv.org/abs/2306.05685", "note": "Agreement and biases of model judges."}, {"title": "Liang et al., Holistic Evaluation of Language Models (HELM, 2022)", "url": "https://arxiv.org/abs/2211.09110", "note": "Evaluating models across many scenarios and metrics."}, {"title": "Es et al., RAGAS: Automated Evaluation of Retrieval Augmented Generation (2023)", "url": "https://arxiv.org/abs/2309.15217", "note": "Faithfulness and relevance metrics for RAG."}]
practice: ["remove-response-length-bias-from-judge-scores", "fit-elo-ratings-from-pairwise-votes", "evaluation-metrics-bleu-rouge", "score-multiple-choice-continuations"]
tags: ["LLM evaluation", "LLM-as-judge"]
---

## Why it is harder

For a classifier, the label is right or wrong. For a generated summary, answer or piece of code, many outputs are acceptable, and small changes in wording can be fine while a single wrong number is a serious error. Evaluation therefore combines several methods, each catching different failures. The AI Engineering course covers LLM foundations and RAG (aieng 03); this lesson focuses on evaluation design.

## Methods

| Method | How | Strengths | Weaknesses
| --- | --- | --- | --- |
| Reference overlap (BLEU, ROUGE) | compare n-grams with a reference | cheap, reproducible | punishes valid paraphrases
| Exact or programmatic checks | parse output, run tests, compare a number | objective | only for structured tasks
| Human rubric grading | experts score against criteria | most trustworthy | slow, expensive
| LLM-as-judge | a model grades against a rubric | scalable | biased, needs calibration
| Pairwise preference | which of two answers is better? | easier for raters | relative, needs many comparisons
| Task outcome | did the user's problem get solved? | closest to value | delayed, noisy

For code, run tests. For extraction, check fields exactly. For open-ended answers, use rubrics and pairwise comparisons, validated by humans.

## The evaluation set

Build it from real usage, not invented examples alone:

- sample real queries across topics and difficulty;
- add hard cases: ambiguous questions, missing information, adversarial prompts, long inputs;
- write references or rubrics with domain experts;
- version it, and keep it out of prompt examples and training data.

A few hundred well-chosen examples with careful labels are worth more than thousands of noisy ones. Grow the set from production failures: every serious bug becomes a test case.

~~~dg-flow An evaluation loop for generated outputs. Production failures become new test cases.
Real traffic -(sample + hard cases)-> Eval set -(model outputs)-> Graders: checks, judge, humans -(scores)-> *Release decision -(launch)-> Production monitoring -(failures)-> Eval set
~~~

## Using model judges well

An LLM judge grades faster and cheaper than people, but:

- **position bias**: it may prefer the first answer shown; randomise order and average both orders;
- **length bias**: it may prefer longer answers; compare at similar lengths or regress scores on length (the `remove-response-length-bias-from-judge-scores` problem shows how);
- **self-preference**: it may favour outputs from its own model family;
- **rubric sensitivity**: vague criteria give unstable scores; use specific rubrics with examples.

Validate the judge on a human-labelled subset and report agreement before trusting it. Keep a slice of human grading running permanently.

## Pairwise comparisons and ratings

When comparing several systems, pairwise votes fitted with a Bradley-Terry model give a rating per system with confidence intervals. This is more reliable than averaging 1-to-10 scores, which people and models use inconsistently.

<div class="ex">
<div class="lab">Worked example</div>

**a judge that rewarded padding**

A team optimised a summariser against an LLM judge asked "how helpful is this summary, 1-10?". Scores rose steadily while users complained summaries had become longer and less useful. The judge rewarded length and hedging. Rewriting the rubric to check specific facts against the source, plus a length penalty, made judge scores track user ratings again.

</div>

## RAG-specific checks

For retrieval-augmented answers, measure the retrieval and the generation separately: did retrieval find the supporting passages (recall), is every claim in the answer supported by them (faithfulness), and does the answer address the question (relevance)? A wrong answer with good retrieval is a generation problem; a wrong answer with poor retrieval is a search problem.

## In production

Offline scores are not the end. Track task outcomes (resolution rate, user edits to drafts, follow-up questions, escalations), explicit feedback, safety incidents, latency and cost per request. Sample production outputs for regular human review.

## Cost and latency are part of quality

Two systems with the same answer quality are not equal if one costs five times as much per request or takes eight seconds to respond. Report tokens per request, cost per thousand requests and p95 latency in every evaluation table, next to the quality scores. Many launch decisions come down to whether a smaller model, a shorter prompt or fewer retrieved passages keep quality within a tolerance while cutting cost.

<div class="take">
<div class="lab">Key takeaway</div>

evaluate generated outputs with a mix of programmatic checks, rubric grading, pairwise preference and task outcomes on a versioned set built from real traffic, and calibrate any model judge against human labels for position, length and rubric bias.

</div>
