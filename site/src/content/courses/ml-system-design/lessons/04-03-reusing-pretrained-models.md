---
id: "04.03"
course: "ml-system-design"
module: "04"
slug: "reusing-pretrained-models"
title: "Reusing pretrained models"
keyIdeas: ["Pretrained models for text, images, audio and code turn many problems with little labelled data into small fine-tuning or feature-extraction tasks.", "Options run from **frozen features** (cheapest) to **fine-tuning the top layers**, **parameter-efficient fine-tuning** such as LoRA, and **full fine-tuning** (most data and compute).", "**Zero-shot** and **few-shot** use of large models can serve as a baseline or a labeller before any training.", "Check licences, domain mismatch, serving cost and whether a large model can be **distilled** into a smaller one for production."]
question: "An insurance company wants to classify claim photos into 12 damage types. It has 3,000 labelled photos and needs 50 ms inference on CPU. Design the modelling approach from zero-shot to production."
answer: "Start with a zero-shot baseline: a CLIP-style model scoring each photo against text prompts for the 12 types, measured on a labelled holdout. Then train a linear classifier (or small MLP) on frozen image embeddings from a strong pretrained encoder using the 3,000 photos; this is cheap and often close to fine-tuning with little data. If accuracy is still short, fine-tune the encoder's top layers or use LoRA adapters with augmentation, using a time-split or claim-level holdout. For production on CPU at 50 ms, distil the fine-tuned model into a smaller network (for example a mobile-sized CNN) trained on the large model's soft labels over a bigger pool of unlabelled claim photos, then quantise to int8. Track per-class recall, since some damage types are rare, and route low-confidence predictions to human adjusters, whose labels feed the next training round."
sources: [{"title": "Hu et al., LoRA: Low-Rank Adaptation of Large Language Models (2021)", "url": "https://arxiv.org/abs/2106.09685", "note": "Parameter-efficient fine-tuning."}, {"title": "Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP, 2021)", "url": "https://arxiv.org/abs/2103.00020", "note": "Zero-shot image classification from text prompts."}, {"title": "Hinton, Vinyals and Dean, Distilling the Knowledge in a Neural Network (2015)", "url": "https://arxiv.org/abs/1503.02531", "note": "Training a small model to imitate a large one."}]
practice: ["clip-zero-shot-classifier", "lora-finetune-and-merge", "clip-cosine-retrieval", "sentence-embeddings"]
tags: ["transfer learning", "fine-tuning"]
tools: ["PyTorch"]
---

## Why reuse

Training a strong image or language model from scratch needs huge datasets and compute. Pretrained models have learned general representations from that scale: edges and textures, word meanings and syntax, speech sounds. For most product tasks the question is not whether to use one but how much to adapt it.

## The adaptation ladder

| Approach | What changes | Data needed | Cost
| --- | --- | --- | --- |
| Zero-shot prompting | nothing; describe the task in text | none | serving cost of a large model
| Frozen features + small head | only a linear or MLP head | hundreds to thousands | very low
| Fine-tune top layers | last few layers and head | thousands | low
| Parameter-efficient (LoRA, adapters) | small added matrices | thousands | low, many tasks share one base
| Full fine-tuning | all weights | tens of thousands or more | high, one copy per task

Climb only as far as the quality requires. Frozen embeddings with logistic regression are surprisingly strong, cheap to retrain and easy to serve.

~~~dg-flow Climb the ladder only as far as quality requires; distil at the end if serving cost matters.
Zero-shot baseline -(labelled data)-> Frozen features + head -(not enough)-> LoRA or partial fine-tune -(not enough)-> *Full fine-tune -(too slow to serve)-> Distilled small model
~~~

## Zero-shot and few-shot as tools

Large language and vision-language models can do many classification and extraction tasks from a description. Even if they are too slow or costly for production, they are useful as:

- a **baseline** to measure trained models against;
- a **labeller** that pre-labels data for humans to correct, which is faster than labelling from scratch;
- a **teacher** for distillation into a small production model.

Validate their labels on a human-labelled sample before trusting them; they are often confidently wrong on domain-specific cases.

## Domain mismatch

A model pretrained on web photos may struggle with medical scans, satellite images or factory parts; a general language model may not know internal product names or legal jargon. Signs of mismatch: frozen features barely beat simple baselines, or errors cluster on domain-specific terms. Remedies are continued pretraining on unlabelled in-domain data, then fine-tuning, or choosing a domain-specific base model. Measure the gap on a small labelled in-domain set before deciding which.

## Serving costs

A fine-tuned large model can be too expensive for high-volume or low-latency paths. Options:

- **distillation**: train a small student on the large model's outputs over plenty of unlabelled data;
- **quantisation**: 8-bit or 4-bit weights to cut memory and latency;
- **caching**: precompute embeddings for items, so only queries need online inference;
- **cascades**: run the large model only on uncertain cases.

<div class="ex">
<div class="lab">Worked example</div>

**from API calls to a small model**

A support team classified incoming emails into 40 intents using a large hosted language model, at a cost of several cents per email. They used it to label 200,000 emails, had agents correct a sample of 5,000, and fine-tuned a small text encoder on the result. The small model matched the large one's accuracy within a point, ran on CPU in 15 ms and cut cost by over 95%.

</div>

## Practical checks

- **Licence**: can the weights be used commercially, and does fine-tuning change that?
- **Data rights**: may your data be sent to a hosted API?
- **Reproducibility**: pin the exact model version; hosted models change underneath you.
- **Evaluation**: compare against a frozen-features baseline so you know what fine-tuning bought.
- **Updates**: plan how to move to a new base model without retraining everything from scratch.

## Choosing a base model

Pick the smallest base that clears the quality bar, because size drives serving cost for the life of the system. Compare two or three candidates with frozen features first; that experiment takes hours and usually ranks them the same way full fine-tuning would. Prefer models with a clear licence and a stable release, and keep the evaluation set fixed so a later swap to a newer base can be judged on the same terms.

<div class="take">
<div class="lab">Key takeaway</div>

start from pretrained models, climb the ladder from zero-shot and frozen features to LoRA and full fine-tuning only as quality demands, and plan for serving cost with distillation, quantisation and caching.

</div>
