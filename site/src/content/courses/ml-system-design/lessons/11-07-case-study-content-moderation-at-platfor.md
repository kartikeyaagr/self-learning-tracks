---
id: "11.07"
course: "ml-system-design"
module: "11"
slug: "case-study-content-moderation-at-platform-scale"
title: "Case study: content moderation at platform scale"
keyIdeas: ["Platform moderation enforces many written policies across text, images, video and live content; each policy is a separate decision with its own costs and legal context.", "The system layers hash matching for known content, fast classifiers per policy, LLM-based policy reasoning for hard cases, and human review sized to capacity.", "Prevalence (the share of views that contain violating content), measured by random sampling, is the honest top-level metric, because actions taken depend on detection itself.", "Appeals, reviewer agreement and policy versioning keep labels consistent as policies and adversaries change."]
question: "Design the moderation system for a social platform with 1 billion pieces of content a day (posts, comments, images, short videos) across 30 policies (hate speech, harassment, spam, nudity, violence, self-harm and others) in 50 languages. There are 20,000 human reviewers. Explain the layers, labels, thresholds, review routing, appeals and how you measure success."
answer: "Layer the system. Hash matching (perceptual hashes) blocks known violating media at upload. Fast multimodal classifiers score every item on every policy, with text models covering the main languages and multilingual embeddings for the rest. For each policy, a high-precision threshold triggers automatic action, a middle band goes to review, and borderline content may be demoted rather than removed. A large language model given the written policy assists on ambiguous text cases and new policies, and drafts rationales for reviewers. Review queues are prioritised by predicted severity times reach (views expected), and routed by language and expertise. Labels come from reviewers under versioned guidelines, with agreement audits and appeals. Measure prevalence of violating views per policy by random sampling, plus proactive detection rate, appeal overturn rate and reviewer accuracy. Monitor new attack patterns and per-language performance. Trade-offs: over- versus under-enforcement per policy, and speed versus accuracy."
rubric: ["Layers hash matching, per-policy classifiers and human review", "Treats each policy separately with its own thresholds and actions, including demotion", "Covers multimodal content and many languages, with a plan for low-resource languages", "Prioritises review by severity and expected reach, within reviewer capacity", "Uses versioned policy labels, reviewer agreement audits and appeals", "Measures prevalence by random sampling as the main metric", "Uses LLM-based policy reasoning or zero-shot methods for new or ambiguous cases", "Monitors adversarial adaptation and per-language performance"]
sources: [{"title": "Kiela et al., The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes (2020)", "url": "https://arxiv.org/abs/2005.04790", "note": "Why image and text must be judged together."}, {"title": "Wikipedia, Content moderation", "url": "https://en.wikipedia.org/wiki/Content_moderation", "note": "Background, approaches and debates."}, {"title": "Wikipedia, PhotoDNA", "url": "https://en.wikipedia.org/wiki/PhotoDNA", "note": "Hash matching for known illegal images."}, {"title": "Wikipedia, Perceptual hashing", "url": "https://en.wikipedia.org/wiki/Perceptual_hashing", "note": "Matching near-duplicates of known content."}]
practice: ["precision-recall-curve-and-average-precision", "text-deduplication-minhash", "clip-zero-shot-classifier", "sigmoid-focal-loss"]
tags: ["content moderation", "policy classifiers", "human review", "prevalence"]
---

## Requirements

Clarify scope: 1 billion items a day (about 11,600 per second), 30 policies, 50 languages, text, images, short video and comments, and 20,000 reviewers who can each review a few hundred items a day, so several million reviews a day in total. Actions include remove, demote, age-restrict, label, warn the author and suspend accounts. Some categories (child safety, terrorism) carry legal reporting duties and zero tolerance.

## Metrics

- Prevalence per policy: the share of content views that contain violating content, estimated by reviewing a random sample of views. This counts what the system missed and is not biased by its own detections.
- Proactive rate: the share of actioned content found before any user report.
- Precision of automatic actions, measured by sampling and by appeal overturn rates.
- Reviewer accuracy against expert audits, and time to action for severe content.

## Layers

1. Hash matching: perceptual hashes of known violating images and videos, shared industry lists for the most serious categories. Fast and precise.
2. Per-policy classifiers on every item: a multimodal model (image or frame encoder plus text encoder over caption, overlaid text and transcript) with a head per policy. Judge image and text together, since a benign image with hateful text, or the reverse, is common.
3. LLM policy reasoning: for ambiguous text and newly written policies, a large model given the policy text classifies and explains. It is too costly for every item, so it runs on the uncertain band.
4. Human review for the remaining uncertain and high-impact items.

## Thresholds and actions

Per policy, set an automatic-action threshold at a precision target (higher for removals with account penalties), a review band sized to the reviewer capacity for that policy and language, and a demotion band for borderline content that is allowed but should not be amplified.

<div class="ex">
<div class="lab">Worked example</div>

**queues ordered by the wrong thing**

A review queue was processed oldest first. Severe items from accounts with large audiences waited hours while reviewers cleared low-reach spam. Ordering the queue by predicted severity times expected views over the next 24 hours cut the views of violating content before action by more than half, with the same number of reviewers.

</div>

## Labels

Reviewers label under written guidelines; labels are versioned by policy version, since a policy change can make old labels wrong. Measure inter-reviewer agreement; low agreement shows a guideline is unclear. Experts audit samples. Appeals provide corrections and a measure of precision. Reviewer wellbeing (exposure limits, blurred previews) is part of the design.

## Languages

Train multilingual models, collect labels for the largest languages first, and measure performance per language. For low-resource languages, use multilingual embeddings, translation for review, and lower automatic-action rates until evaluation shows enough precision.

## Serving

~~~dg-flow Moderation pipeline. Cheap, precise layers run first; scarce human review is spent where it matters most.
Upload -(perceptual hash)-> Known-content match -(block)-> Enforcement
Upload -(no match)-> *Per-policy multimodal classifiers -(high score)-> Enforcement
Per-policy multimodal classifiers -(uncertain)-> LLM policy check -(still uncertain)-> Review queue by severity x reach -(decision)-> Enforcement
Enforcement -(appeals)-> Review queue by severity x reach
~~~

## Monitoring

Track prevalence per policy and language weekly, proactive rate, appeal overturns, queue backlogs and time to action. Adversaries adapt (misspellings, coded words, altered images); watch for new clusters of reported content and update hashes and models quickly.

## Likely follow-up questions

- "How do you launch a new policy?" Write guidelines, label a seed set, start with LLM-based classification and review, then train a fast head when labels suffice.
- "How do you handle live video?" Sample frames and audio continuously with a fast model, and allow reviewers to interrupt a stream.

## Trade-offs

- Over-enforcement (silencing legitimate speech) versus under-enforcement (harm), set per policy.
- Automation versus reviewer capacity and accuracy.
- Global models versus local context and law.

## A numeric sanity check

At 1 billion items a day and 30 policies, the classifiers make 30 billion policy scores a day, but they share one encoder pass per item, so the cost is about 11,600 encoder passes per second plus small heads. If 2% of items fall into some review band, that is 20 million candidates a day, several times what 20,000 reviewers can handle at a few hundred each. This is why the review band must be sized to capacity per policy and language, why prioritisation by severity and reach matters, and why the LLM check runs only on the uncertain band.

<div class="take">
<div class="lab">Key takeaway</div>

moderate in layers (hashes, per-policy multimodal classifiers, LLM policy checks, prioritised human review), set thresholds per policy and language, keep labels versioned with appeals and audits, and measure success by prevalence from random sampling.

</div>
