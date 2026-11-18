---
id: "10.01"
course: "ai-engineering-course"
module: "10"
slug: "why-llm-evals-are-different"
title: "Why LLM evals are different"
keyIdeas: ["LLM outputs are open-ended, so most cases have no single correct label: you grade against criteria, references or executable checks, and the criteria themselves come from reading outputs.", "The same input can give different outputs, even at temperature 0, so one run of an eval is a sample. Separate run-to-run noise from sampling error before believing a change.", "You evaluate a system (prompt, retrieval, tools, model snapshot, decoding settings), not a model, and any of those can change under you.", "Eval-driven development is a loop: look at traces, name the failure modes, write an eval for each, change one thing, re-run, and feed production failures back into the set.", "Offline evals gate changes before release; online evals (sampled grading of live traffic, user signals, A/B tests) tell you whether the offline set still matches reality."]
question: "Your team shipped an LLM support assistant with a 'quality score' from a public benchmark and a generic 'helpfulness 1-10' judge. Users complain anyway. Explain why this evaluation failed and lay out the evaluation process you would put in place instead."
answer: "Both measures were chosen before anyone looked at the product's outputs. A public benchmark measures a model on someone else's tasks, and a generic helpfulness score has no stated criteria, so neither is tied to the failures users actually see. I would start by reading 100 or so real conversations with a domain expert, writing a note on each failure, then grouping the notes into named failure modes (wrong refund policy, missed escalation, invented order status) and counting them. Each frequent mode gets its own check: code assertions where possible (did it call the order lookup tool, does the reply contain a ticket link), a binary LLM judge validated against expert labels where not. Every case is run several times because outputs vary, and results are reported with confidence intervals and per-slice. Changes are made one at a time and re-run against the set. After release, a sample of live traces is graded with the same judges, user signals are tracked, and new failures are added to the set."
rubric: ["Diagnoses that the metrics were generic and chosen before error analysis, so they did not reflect real failure modes.", "Starts from reading real traces and grouping failures into named, counted categories.", "Matches each failure mode to the cheapest valid grader (code check before LLM judge) and validates any judge against expert labels.", "Accounts for non-determinism (repeats, intervals) and slices rather than one aggregate score.", "Closes the loop online: sampled grading of production traffic, user signals, and new failures added to the eval set."]
sources: [{"title": "Hamel Husain: Your AI Product Needs Evals", "url": "https://hamel.dev/blog/posts/evals/", "note": "The three levels of evaluation (assertions, model and human grading, A/B tests) and the habit of reading traces."}, {"title": "Shankar et al.: Who Validates the Validators? (2024)", "url": "https://arxiv.org/abs/2404.12272", "note": "Introduces criteria drift: people only learn what to grade for by grading outputs."}, {"title": "Thinking Machines: Defeating Nondeterminism in LLM Inference (2025)", "url": "https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/", "note": "Why greedy decoding still varies between requests on a shared server: kernels that are not batch-invariant."}, {"title": "Anthropic docs: Define success criteria and build evaluations", "url": "https://docs.claude.com/en/docs/test-and-evaluate/develop-tests", "note": "Turning success criteria into graded test cases; code, human and model grading."}]
practice: ["judge-human-agreement-kappa", "pass-at-k-estimator", "bootstrap-confidence-interval", "binomial-distribution"]
tools: ["Claude", "GPT-5"]
tags: ["LLM evaluation"]
---

## Three properties that break the classic recipe

In classic supervised learning, evaluation is settled before you start: a held-out test set, one label per example, one metric like accuracy or AUC. [Evaluating generated answers](lesson:mlsd:05.03) lists the methods that replace that recipe for generated text. This module is about running them as an engineer shipping an LLM product, and that starts with three properties that change how every eval is built.

**1. Outputs are open-ended.** A good answer to "Can I return a gift without a receipt?" can be phrased a thousand ways, while one wrong number of days makes it a bad answer. There is rarely a single label to compare against. You grade against criteria ("states the 30-day window", "does not promise a cash refund"), against reference facts, or by executing the output (run the SQL, run the tests). Worse, you usually do not know the right criteria up front. Shankar et al. call this **criteria drift**: people define what "good" means partly by grading outputs, and the criteria shift as they see more. That is why evals start with reading outputs, not with choosing a metric.

**2. Outputs are non-deterministic.** Sampling at temperature 0.7 gives different answers on each call. Temperature 0 is not a guarantee either: hosted inference batches your request with other people's, and many GPU kernels give slightly different floating-point results at different batch sizes, so a near-tie between two tokens can flip and the rest of the answer diverges from there (the Thinking Machines post traces this to kernels that are not batch-invariant). Reasoning models often fix or ignore temperature entirely ([Text generation](lesson:aieng:03.03)). Practically, one run of an eval is a sample, and you need to know how big the run-to-run noise is before you believe a two-point improvement.

**3. You are evaluating a system, not a model.** The thing that produced an output is a prompt template, a retrieval index, tool definitions and their live backends, a model snapshot, decoding settings and any post-processing. A change to any of them can move quality. So can changes you did not make: a provider updating the model behind an alias, a document re-indexed, a tool API returning a new field. An eval result means nothing without the versions of everything that produced it.

| | Classic ML eval | LLM product eval |
| --- | --- | --- |
| Ground truth | One label per example | Criteria, references or executable checks; often several per case |
| Metric | Chosen up front (accuracy, AUC) | Discovered by reading outputs; one metric per failure mode |
| Determinism | Same input, same prediction | Same input, a distribution of outputs |
| Unit under test | A trained model artifact | Prompt + retrieval + tools + model snapshot + settings |
| Input space | Fixed feature schema | Free text, multi-turn, shifts with users and features |
| Cost of grading | Free (compare labels) | Can be a model call per case per criterion |
| Quality dimensions | Usually one | Correctness, faithfulness, format, tone, safety, cost, latency |

## The eval-driven development loop

The working method that falls out of these properties is a loop, run every time you change something and continuously after launch.

~~~dg-flow Eval-driven development. Production failures become new cases, so the set keeps tracking reality.
Traces (logs or synthetic) -(read with a domain expert)-> Failure notes -(group and count)-> *Named failure modes -(one check each)-> Eval set + graders -(change one thing, re-run)-> Release decision -(ship, sample live traffic)-> Traces (logs or synthetic)
~~~

1. **Look at the data.** Read 50 to 100 traces end to end (input, retrieved context, tool calls, output). For each bad one, write a short note on the first thing that went wrong. Hamel Husain and Shreya Shankar call this open coding.
2. **Name and count failure modes.** Group the notes into categories (axial coding) and count them. "Invented an order status: 14. Missed an escalation request: 9. Wrong tone: 3." This is the same error analysis as [Error analysis and evaluation slices](lesson:mlsd:05.04), applied to traces.
3. **Write a check per failure mode.** Use the cheapest grader that is valid: a code assertion if the failure is checkable in code, an LLM judge with a binary verdict if it is not ([Graders and LLM-as-judge](lesson:aieng:10.03)). Each check runs on a set of cases chosen to exercise it ([Building an eval set](lesson:aieng:10.02)).
4. **Change one thing and re-run.** Compare against the previous run on the same cases. Two changes at once and you cannot attribute the delta.
5. **Ship and sample.** Grade a sample of live traces with the same checks, and add new failures to the set.

The order matters. Teams that start at step 3 with off-the-shelf metrics ("helpfulness", ROUGE, a generic faithfulness score) measure things that are easy to compute rather than things that are going wrong, and the numbers stay flat while users complain.

## Measuring the noise

<div class="ex">
<div class="lab">Worked example</div>

**How much does a re-run move the score?**

A support assistant has 200 eval cases, graded pass/fail. You run every case 5 times at the production temperature.

- 150 cases pass all 5 runs, 20 fail all 5, and 30 are flaky: they pass some runs and fail others, about half the time on average (75 passes out of their 150 runs).
- Mean pass rate over all 1,000 runs: (150 × 5 + 75) / 1,000 = **82.5%**.
- A single run's pass count is 150 + (passes among the 30 flaky cases). If each flaky case is a coin flip, that second term has standard deviation $\sqrt{30 \times 0.5 \times 0.5} = 2.74$ cases, or **1.4 points** of pass rate. Two runs of the *same prompt* will often differ by 2 to 3 points.
- Separately, the 200 cases are a sample of possible inputs. The 95% margin from that is $1.96\sqrt{0.825 \times 0.175 / 200} = \pm 5.3$ points. More repeats shrink the first number but never the second; only more cases do.
- The strict "passes every time" rate is 150/200 = **75%**, not 82.5%. For a user who asks the same thing twice, consistency is part of quality. [Evaluating agents](lesson:aieng:10.05) formalises this as pass^k.

So a single run showing a new prompt at 84% versus 82% tells you nothing. And the 30 flaky cases are the most useful thing in the run: they mark where the prompt is ambiguous enough that sampling decides the outcome. Read them first.

</div>

Repeats are cheap insurance. For a pass/fail eval, 3 to 5 runs per case usually expose the flaky cases; how many cases you need, and how to compare two variants on the same cases, are covered in [Building an eval set](lesson:aieng:10.02) and [Regression testing in CI](lesson:aieng:10.06).

## Offline and online evaluation

~~~dg-layers Where each kind of evaluation sits. Offline evals gate a change; online evals check the offline set still matches reality.
Offline: assertions | code checks on every case, run on every commit; cheap and fast
Offline: graded eval set | LLM judges and expert review on a versioned set; gates releases
*Online: sampled live grading | the same judges on a sample of production traces; finds new failure modes
Online: user signals and experiments | thumbs, edits, retries, escalations, A/B tests on business metrics
~~~

| | Offline | Online |
| --- | --- | --- |
| Inputs | A fixed, versioned eval set | Real traffic, which shifts |
| When | Before a change ships | After, continuously |
| Ground truth | Labels and references you wrote | Usually none; judges and user behaviour as proxies |
| Answers | "Did this change make things worse on what we know?" | "What are we missing?" and "Did it help users?" |

Online signals are noisy and biased (most users never click thumbs-down; a retry can mean a bad answer or a curious user), but they are the only way to find failure modes your set does not contain. A/B tests on business metrics are the final word and follow the rules in [A/B testing and safe rollout](lesson:mlsd:05.05). Live traces flow back into the set through observability tooling ([Observability for LLM apps](lesson:aieng:12.07)).

An eval is not a guardrail. A guardrail runs on every request, inside the request path, and blocks or rewrites output; it must be fast and its false positives hurt users ([Guardrails](lesson:aieng:12.02)). An eval runs on samples, outside the request path, to measure. The same classifier can serve both roles, but its thresholds and latency budgets differ.

## Common mistakes

- **Trusting a public benchmark as a product eval.** Leaderboard scores say how a model does on someone else's tasks, possibly with test data in its training set. They help shortlist models ([Choosing a model](lesson:aieng:01.05)), not decide whether your prompt works.
- **Generic metrics before error analysis.** A 1-to-10 "quality" score from a judge with no criteria drifts with wording and rewards length. Start from named failure modes and binary checks.
- **One run, one number.** Without repeats and an interval, a two-point move is indistinguishable from noise.
- **Not pinning versions.** "gpt-5" or "claude-sonnet-latest" today is not the same model next quarter. Record the dated snapshot, prompt hash, index version and settings with every result.
- **An eval set that always passes.** A set at 100% cannot detect improvement and probably no longer contains the hard cases. Keep adding the failures you find.
- **Ignoring cost and latency.** Two variants with equal quality are not equal if one costs four times more per request. Report both next to every quality number.

## What interviewers probe

Expect "how would you evaluate X?" in almost every AI engineering loop. Strong answers start from reading traces and naming failure modes, separate code checks from judged criteria, mention validating the judge against humans, quantify noise (repeats, intervals, slices), and close the loop with online sampling. Weak answers name a metric first.

<div class="take">
<div class="lab">Key takeaway</div>

LLM outputs are open-ended, vary between runs and come from a whole system that keeps changing, so evals cannot be a fixed test set and one metric. Read traces, name the failure modes, give each a check, measure the noise, and keep feeding production failures back into the set.

</div>
