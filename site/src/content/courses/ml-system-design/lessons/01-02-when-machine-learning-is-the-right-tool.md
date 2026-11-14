---
id: "01.02"
course: "ml-system-design"
module: "01"
slug: "when-machine-learning-is-the-right-tool"
title: "When machine learning is the right tool"
keyIdeas: ["ML pays off when the pattern is too complex to write as rules, there is enough data with labels, the pattern repeats, and mistakes are tolerable and measurable.", "A **heuristic baseline** is the first thing to build. It sets the bar ML must clear and often ships first.", "ML adds cost: data pipelines, labels, retraining and monitoring. The quality gain has to pay for them.", "Prefer ML when the world changes faster than people can update rules, and when scale makes manual decisions impossible."]
question: "A support team wants ML to route 2,000 tickets a day into 6 queues. Today a keyword rule routes 80% correctly. How would you decide whether ML is worth it, and what would you build first?"
answer: "First measure the value of better routing: how much time or customer satisfaction is lost per misrouted ticket, and how many of the 400 daily errors could a model fix. Then check prerequisites: are historical tickets labelled with their final queue (reassignments give labels for free), is the label distribution stable, and can errors be corrected cheaply by an agent? If labels exist, build a quick text classifier (TF-IDF plus logistic regression) and compare it with the rule on a time-based holdout. If it moves accuracy from 80% to, say, 92% and the error cost justifies a retraining and monitoring pipeline, ship it with the rule as a fallback. If not, improve the rules. Either way, the keyword rule is the baseline and a safety net."
sources: [{"title": "Google, Rules of Machine Learning, Rule 1 and 3", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "\"Don't be afraid to launch a product without machine learning.\""}, {"title": "Google, Problem framing course", "url": "https://developers.google.com/machine-learning/problem-framing", "note": "Deciding whether ML fits and how to frame it."}]
practice: ["popularity-baseline-leave-last-out", "logistic-regression", "k-fold-cross-validation"]
tags: ["problem framing", "heuristics vs ML"]
---

## A tool with running costs

Machine learning replaces hand-written rules with a function fitted to data. That is powerful when the rules would be too many, too subtle or too fast-changing to maintain. But a learned function needs things a rule does not: a steady supply of labelled examples, a training pipeline, monitoring for drift, and someone on call when it degrades. The question is not "can ML do this?" but "does the quality gain pay for those costs?"

## Signs that ML fits

| Question | Good sign for ML | Warning sign
| --- | --- | --- |
| Is the pattern complex? | many interacting signals, fuzzy boundaries | a few clear thresholds
| Is there data? | thousands or more labelled examples, or labels you can collect | only a handful, or labels nobody can define
| Does it repeat? | the same decision many times a day | a one-off decision
| Can you tolerate errors? | mistakes are cheap or reversible | one mistake is catastrophic and unexplainable
| Does it change? | patterns drift, adversaries adapt | stable rules that rarely need edits
| Is the scale large? | too many decisions for people | a team can decide each case

Fraud scoring, search ranking, spam filtering and demand forecasting tick most boxes. Calculating tax, enforcing a legal age limit or routing by an exact product code do not: rules are cheaper, exact and auditable.

## Start with a heuristic

Before any model, write the simplest rule that works: "sort by popularity", "flag transactions over £5,000 from a new device", "route tickets containing 'refund' to billing". It does three jobs. It ships value now. It gives a baseline number that any model must beat. And it becomes the fallback when the model is unavailable.

~~~dg-flow The usual path from rules to ML. Each step must beat the last on the business metric.
Manual process -(codify)-> Heuristic rules -(log decisions + outcomes)-> Labelled data -(fit)-> *Simple model -(if it pays)-> Complex model
~~~

Logging is the bridge. Once the heuristic runs, record its inputs and the eventual outcomes. That log becomes your first training set, and it is often the only realistic source of labels.

## Estimating whether it is worth it

Put rough numbers on the value. If a better ranking increases conversion by 0.5% on £20M of annual sales, that is £100k a year. If maintaining the ML pipeline costs one engineer half-time plus £20k of compute, it is worth it. If the gain is 0.05%, it is not, at least not yet.

A useful habit: estimate the ceiling. If even a perfect model would only fix 10% of current errors, because the rest come from missing data or ambiguous labels, ML cannot deliver much.

<div class="ex">
<div class="lab">Worked example</div>

**ML that was not needed**

A retailer planned a model to predict which products to restock. Analysis showed 90% of stock-outs came from a supplier delay that no model could see. A rule that reordered when stock fell below two weeks of average sales fixed most of the problem. ML came later, for the remaining 10%, where seasonal patterns were genuinely hard to hand-code.

</div>

## When ML is the only option

Some tasks have no workable rules at all: recognising objects in images, transcribing speech, ranking billions of documents for free-text queries, or generating text. Here the question is not whether to use ML but which model and how to get data. Pretrained models (covered in 04.03) often make these feasible with little labelled data.

## Questions to ask before committing

Who will own the model after launch, and will they retrain it? How will you know if it gets worse? What is the fallback if the model or its features are unavailable? Can you explain a decision to a customer or a regulator if asked? If nobody can answer these, the project is not ready, however good the offline numbers look. Writing the answers down is a cheap way to find gaps in the plan before any data work starts.

<div class="take">
<div class="lab">Key takeaway</div>

use ML when the pattern is complex, repeated at scale, learnable from available labels and tolerant of errors; build a heuristic first as baseline, fallback and label source, and check the quality gain pays for the pipeline.

</div>
