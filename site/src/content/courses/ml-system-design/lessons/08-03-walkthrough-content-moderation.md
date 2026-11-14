---
id: "08.03"
course: "ml-system-design"
module: "08"
slug: "walkthrough-content-moderation"
title: "Walkthrough: content moderation"
keyIdeas: ["Content moderation combines rare positives, many policy categories, several content types (text, images, video), adversaries, human review capacity and high costs for both kinds of error.", "A typical design has fast hashing for known bad content, multimodal classifiers per policy, thresholds for automatic action and human review, and feedback from appeals and reviewers.", "Labels come from reviewers under changing policies. Version labels by policy, measure reviewer agreement and handle delays.", "Evaluation is by policy and by language or region, at operating points tied to reviewer capacity, with monitoring for new attack patterns."]
question: "Walk through the design of a system that detects and acts on violent and graphic content in images and videos uploaded to a social platform with 50 million uploads a day. Cover the main stages and two key trade-offs."
answer: "Clarify: about 580 uploads per second on average, perhaps 2,000 at peak; actions are block at upload, age-gate or blur, or send to review; review capacity about 100,000 items a day; latency target a few seconds for images, minutes for long videos. Stages: perceptual-hash matching against a database of known violating media blocks exact and near copies instantly; a multimodal classifier (image encoder on frames sampled from videos plus text from captions and audio transcripts) outputs per-policy probabilities; video uses sampled keyframes first and a deeper pass if uncertain. Thresholds per policy: auto-action above a high-precision threshold, review between thresholds sized to capacity, allow below; graphic but newsworthy content gets blur plus a warning rather than removal. Labels from reviewers with versioned guidelines and appeals; active learning picks uncertain items for labelling. Evaluation by policy and region at the operating points; monitor prevalence estimates from a random sample, appeal overturn rates, and new hashes. Trade-offs: speed versus accuracy for long videos (quick keyframe pass then deep pass), and precision versus recall per policy given the different harms of over- and under-enforcement."
rubric: ["Clarifies volume, content types, actions, review capacity and latency", "Uses hash matching for known content before ML", "Proposes multimodal classifiers with per-policy outputs and explains video sampling", "Sets two thresholds per policy tied to precision targets and reviewer capacity", "Treats reviewer labels as versioned by policy, with agreement checks and appeals", "Evaluates per policy and per language or region, with prevalence from random sampling", "Handles adversaries and new content with monitoring, active learning and fast hash updates", "Discusses trade-offs between speed, cost and accuracy, and between over- and under-enforcement"]
sources: [{"title": "Kiela et al., The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes (2020)", "url": "https://arxiv.org/abs/2005.04790", "note": "Why image and text must be judged together."}, {"title": "Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP, 2021)", "url": "https://arxiv.org/abs/2103.00020", "note": "Shared image-text embeddings, useful for multimodal classifiers."}, {"title": "Google, Rules of Machine Learning", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Heuristics plus models; monitoring."}]
practice: ["precision-recall-curve-and-average-precision", "sigmoid-focal-loss", "text-deduplication-minhash", "clip-zero-shot-classifier"]
tags: ["content moderation", "interview walkthrough"]
---

## The prompt

"Design a system that detects violent and graphic content in uploaded images and videos." This walkthrough shows how the eight-step framework (08.02) applies, with the emphasis on labels, thresholds and adversaries. Module 11 includes a full case study on content moderation; this lesson focuses on how to present it in 45 minutes.

## 1. Clarify (5 minutes)

Questions and assumed answers:

- Volume: 50M uploads a day, about 580 per second, peaking near 2,000.
- Content: images, short videos (under 60 s) and some long videos; captions and audio.
- Actions: block at upload, blur with warning, restrict to adults, send to human review, allow.
- Review capacity: 100,000 items a day across languages.
- Latency: images scored before they appear; videos within a few minutes.
- Costs: missing graphic violence harms users and advertisers; removing newsworthy content harms expression and trust.

## 2. Frame (5 minutes)

Multi-label classification over policies (graphic violence, gore, self-harm, and so on), producing a calibrated probability per policy. The decision layer maps probabilities to actions with two thresholds per policy. Online metrics: prevalence of violating views (estimated by random sampling), reviewer workload, appeal overturn rate. Guardrails: wrongful removal rate, latency.

## 3. Data and labels (5 minutes)

Labels come from human reviewers following written policies. Policies change, so labels are versioned by policy version. Measure agreement between reviewers; low agreement points to unclear guidelines. Sample for labelling in three ways: random (to estimate prevalence), near the thresholds (active learning), and from user reports.

## 4. Features and models (10 minutes)

~~~dg-flow Moderation pipeline. Cheap exact matching first, models next, people for the uncertain middle.
Upload -(perceptual hash)-> Known-content match -(no match)-> *Multimodal classifier -(score per policy)-> Decision thresholds -(uncertain)-> Human review -(labels + appeals)-> Training data
~~~

- **Hash matching** against known violating media catches re-uploads instantly and cheaply.
- **Image model**: a pretrained vision encoder fine-tuned per policy.
- **Video**: sample keyframes (for example one per second), score them, aggregate with max or attention pooling; run a deeper pass on uncertain videos.
- **Text and audio**: captions, comments and transcripts through a text encoder; fuse with visual embeddings.
- **Context**: uploader history, prior strikes, account age.

Baseline: hash matching plus a frozen image-embedding classifier. Main model: fine-tuned multimodal encoder with per-policy heads and focal or weighted loss for rare positives.

## 5. Evaluation and serving (10 minutes)

Evaluate per policy at the operating points: precision for automatic action (high bar, for example 97%), recall including review, review volume per day, and slices by language, region and content type. Prevalence is estimated from a random sample labelled by reviewers.

Serving: images scored synchronously on GPU with batching in about 100 ms; short videos scored asynchronously, with content held from distribution until scored; long videos get a fast keyframe pass, then a deeper pass.

## 6. Monitoring and adversaries (5 minutes)

Monitor score distributions, action rates per policy and region, review queue length, appeal overturn rate and prevalence. Adversaries crop, filter, overlay text or splice content; add augmentations to training, update hashes quickly and retrain frequently. Keep a fast path for policy teams to add rules during crises.

<div class="ex">
<div class="lab">Worked example</div>

**a trade-off to state out loud**

"For long videos, scoring every frame with the full model would cost about 30 GPU-seconds per video. I'd sample one frame per second with a lighter model and only run the full model on segments scoring above a low threshold, which keeps cost around 3 GPU-seconds while catching nearly all violating segments in our tests."

</div>

## What interviewers want to hear

Rare positives and thresholds tied to reviewer capacity, versioned labels, multimodal models, hash matching for known content, per-policy and per-region evaluation, and adversarial monitoring. The model architecture matters less than these system decisions.

## Likely follow-up questions

Be ready for three probes. First, "how do you handle a new policy with no labels?": start with rules and keyword or hash lists, route matches to reviewers, use their decisions as the first labels, and train once a few thousand exist. Second, "how do you stop reviewers burning out?": blur by default, rotate queues, cap exposure per shift and let the model pre-sort so humans see fewer clear-cut cases. Third, "what if a classifier is biased against a dialect?": measure false-positive rates by language and region on a stratified sample, and fix with targeted labelling and per-slice thresholds rather than a global change.

<div class="take">
<div class="lab">Key takeaway</div>

for moderation, lead with volume, actions and reviewer capacity, then present hash matching, multimodal per-policy classifiers, two thresholds per policy, versioned reviewer labels, per-policy and per-region evaluation, and monitoring for adversaries.

</div>
