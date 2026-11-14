---
id: "11.07"
course: "inference-eng-course"
module: "11"
slug: "measure-quality-regression"
title: "Measure Quality Regression"
keyIdeas: ["Speed means nothing if quality falls below what the product needs.", "Use several signals: perplexity, standard benchmarks, product-specific evals, and side-by-side comparisons.", "Test the hard cases: math, code, long context, structured output, other languages.", "Run evals with the same sampling settings and engine as production.", "Set an acceptance threshold before testing."]
question: "Why is perplexity on generic text not enough to approve a quantized model for production?"
answer: "Perplexity measures average next-token prediction on some text, which can stay nearly unchanged while specific capabilities (math, code, long context, instruction following, rare languages) degrade. You should also run task evals that match your product, compare outputs on real prompts, and check long-context and structured-output behavior."
sources: [{"title": "lm-evaluation-harness", "url": "https://github.com/EleutherAI/lm-evaluation-harness", "note": "Standard benchmark runner."}, {"title": "Give Me BF16 or Give Me Death? Accuracy-Performance Trade-Offs in LLM Quantization", "url": "https://arxiv.org/abs/2411.02355", "note": "Large study of quantized model quality."}]
tags: ["quality", "evals"]
---

## Why perplexity is not enough

Perplexity on held-out text is cheap and catches broken quantization quickly. But it averages over every token, and users do not experience averages. A model can lose very little perplexity while getting noticeably worse at multi-step math, at producing valid JSON, or at finding a fact in a 30,000-token document. You need checks that match how the model is used.

## Layers of quality checks

~~~dg-layers Quality checks, from cheap to convincing.
Perplexity | fast sanity check on held-out text
Standard benchmarks | knowledge, math and code tasks through an evaluation harness
*Product evals | your own prompts with graded expected answers
Side by side | human or model-judged comparison of BF16 and quantized outputs
Online | a canary with real users and product quality metrics
~~~

Run them in order. Each layer is more expensive and more convincing. A change that fails an early layer does not need the later ones.

## What to test

Quantization errors show up first in some kinds of tasks:

- **Reasoning and math.** Long chains of steps amplify small errors.
- **Code generation.** One wrong token can break a program.
- **Long-context retrieval and QA.** Especially for KV quantization.
- **Structured output.** JSON validity and tool-call argument correctness.
- **Languages other than English**, if your users use them. Less common languages often degrade first.

## Running evaluations correctly

- **Use the production setup.** The same engine, the same chat template and the same sampling settings. An eval run with a different template measures the template, not the quantization.
- **Use greedy or fixed-seed sampling** for comparisons, or several samples per prompt to average out randomness.
- **Use enough samples.** On a 500-example benchmark with accuracy around 80%, the standard error is about √(0.8 × 0.2 ÷ 500) ≈ 1.8 points. A 1-point difference is within noise. Report confidence intervals, or use paired tests on the same examples.
- **Compare against the same baseline run.** Rerun BF16 with the same harness rather than quoting a published number.

## Set the acceptance rule in advance

Decide what counts as acceptable before you see results, for example: "No product eval drops by more than 1 point. No standard benchmark drops by more than 2 points. JSON validity stays above 99%." Then the decision is mechanical and not argued after the fact.

| Eval | BF16 | FP8 | INT4 AWQ |
| --- | --- | --- | --- |
| Product QA (500 examples) | 82.4 | 82.1 | 79.8 |
| Math benchmark | 71.0 | 70.4 | 66.2 |
| JSON validity | 99.6% | 99.5% | 98.9% |

With the rule above, FP8 passes on every row. INT4 AWQ fails product QA (−2.6) and math (−4.8), and falls below the JSON threshold.

<div class="ex">
<div class="lab">Worked example</div>

**a regression perplexity missed**

A team quantized a 7B model to INT4. Perplexity rose by 2%, which looked harmless.

Their product extracts fields from invoices into JSON. JSON validity fell from 99.7% to 97.1%, and field accuracy fell 4 points, mostly on long invoices.

At their volume, that meant about 2,500 failed extractions a day. They moved to FP8 weights, where both numbers stayed within 0.2 points of BF16.

</div>

## Side by side and online checks

For changes that pass the automated evals, a side-by-side comparison catches subtler differences in tone, completeness or formatting. A model judge can score thousands of pairs cheaply; spot-check its verdicts by hand. For high-traffic products, finish with a canary: route a few percent of real traffic to the quantized model and compare product metrics such as thumbs-down rate, retries or task completion.

## Deep dive: how many eval items you need

Accuracy ⟦m668⟧ measured on ⟦m669⟧ items has standard error ⟦m670⟧. For a 95% interval of ⟦m671⟧:

⟦M667⟧

| Accuracy | ±2 points | ±1 point | ±0.5 points |
| --- | --- | --- | --- |
| 80% | 1,537 | 6,147 | 24,586 |
| 95% | 456 | 1,825 | 7,299 |

Comparing two models on the same items is more sensitive, because most items get the same answer from both. Only the items where they disagree matter (McNemar's test). With 500 items and 20 disagreements, a 15 to 5 split is significant at 5% (exact test, p = 0.04). A 13 to 7 split is not (p = 0.26).

<div class="ex">
<div class="lab">Worked example</div>

**a 1-point acceptance rule on 500 items**

The rule "quantized model within 1 point of BF16" on a 500-item benchmark at 80% accuracy: SE ⟦m672⟧ points per model.

For two independent runs, noise alone moves the measured difference by about ⟦m673⟧ points at 95%. The rule cannot be checked with this set. Use paired comparison, more items, or both.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

approve quantization with evaluations that match your product, run in the production setup, with enough samples and a threshold set beforehand. Perplexity alone hides the regressions users notice.

</div>
