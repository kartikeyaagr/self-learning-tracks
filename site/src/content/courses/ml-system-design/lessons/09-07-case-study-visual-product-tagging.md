---
id: "09.07"
course: "ml-system-design"
module: "09"
slug: "case-study-visual-product-tagging"
title: "Case study: visual product tagging"
keyIdeas: ["Product tagging predicts many attributes from images and text (category, colour, pattern, material, style), so it is multi-label and hierarchical rather than a single class.", "Labels come from sellers (noisy and incomplete), annotators (expensive) and user behaviour, and the taxonomy keeps changing.", "A shared image-text encoder with one small head per attribute, plus zero-shot scoring for new attributes, keeps the system maintainable.", "Per-attribute confidence thresholds decide what is applied automatically, what goes to review, and what is left blank."]
question: "Design a system that automatically tags products in a marketplace with category and attributes (colour, pattern, material, style, sleeve length and so on) from seller images and text. There are 2 million new listings a day and the taxonomy changes monthly. Cover labels, model design, thresholds and how tags are used."
answer: "Treat it as hierarchical, multi-label prediction. Use a shared image-text encoder (CLIP-style, fine-tuned on the catalogue) that embeds all images and the title and description; attach small heads per attribute group, with sigmoid outputs for multi-valued attributes and softmax for exclusive ones. Category is predicted first and gates which attributes apply. Labels combine seller-provided fields (noisy, used with confidence weighting), annotator labels on a stratified sample, and implicit signals such as search queries that led to purchases. New attributes start with zero-shot prompts, then get heads once labels exist. Calibrate per attribute and set thresholds for auto-apply, review and leave blank. Batch inference on new and edited listings within minutes. Evaluate per attribute with precision at the auto-apply threshold and coverage; online, measure search and filter use. Monitor drift by category and seller corrections."
rubric: ["Frames the problem as hierarchical and multi-label, with category gating attributes", "Uses both image and text inputs with a shared encoder and per-attribute heads", "Handles noisy seller labels and gets clean labels on a stratified sample", "Handles taxonomy changes, for example with zero-shot prompts for new attributes", "Calibrates per attribute and sets auto-apply, review and blank thresholds", "Evaluates per attribute (precision at threshold and coverage), not with one overall accuracy", "Chooses batch or near-real-time inference appropriate to listing volume", "Connects tags to downstream uses such as search filters and measures their impact"]
sources: [{"title": "Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP, 2021)", "url": "https://arxiv.org/abs/2103.00020", "note": "Zero-shot classification from text prompts."}, {"title": "Shankar et al., Deep Learning based Large Scale Visual Recommendation and Search for E-Commerce (2017)", "url": "https://arxiv.org/abs/1703.02344", "note": "Visual features for a large retail catalogue."}, {"title": "Google, Rules of Machine Learning", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Practical advice on labels, features and launches."}]
practice: ["clip-zero-shot-classifier", "sigmoid-focal-loss", "expected-calibration-error", "top-k-classification-accuracy"]
tags: ["product tagging", "multi-label classification", "zero-shot", "taxonomy"]
---

## Requirements

Clarify what tags are for. Usually they feed search filters ("blue", "linen", "long sleeve"), recommendations, and catalogue quality checks. A wrong tag in a filter shows users the wrong products, which is worse than a missing tag. Assume 2 million new or edited listings a day, tags needed within an hour of listing, around 5,000 categories and 200 attribute types, and monthly taxonomy changes.

## Metrics

Per attribute: precision at the auto-apply threshold and coverage (the share of listings that receive a tag). Category accuracy at each level of the hierarchy. Online: use of filters, click-through after filtering, and the rate at which sellers correct tags.

## Data and labels

- Seller fields: abundant but incomplete and sometimes wrong (sellers pick broad categories, or misstate materials). Use them with a confidence weight, and estimate their error rate per attribute on a reviewed sample.
- Annotators: accurate but expensive. Spend them on a stratified sample across categories and on uncertain predictions (active learning).
- Behaviour: a listing bought after a search for "red dress" is weak evidence that it is red and a dress.

## Model

Use a shared encoder over images and text. A CLIP-style model fine-tuned on the catalogue gives aligned image and text embeddings; combine all listing images (average or attention pooling) with the title and description.

- Category: a hierarchical classifier, predicting top level then finer levels, so errors stay near the right branch.
- Attributes: a small head per attribute group. Use softmax for exclusive attributes (sleeve length) and sigmoid for attributes that can take several values (colours, materials). Focal loss helps with rare values.
- Gating: the predicted category determines which attributes apply; a sofa has no sleeve length.
- New attributes: score zero-shot by comparing the image embedding with text prompts such as "a photo of a velvet sofa"; once enough reviewed labels exist, train a head.

<div class="ex">
<div class="lab">Worked example</div>

**text and image disagree**

A listing titled "red dress" showed a blue dress in every photo; the seller had copied an old title. Image-only and text-only models each tagged it confidently and differently. Training the fused model with examples where the two sources disagree, and routing large disagreements to review, cut colour errors in filters by about a third and found thousands of mislabelled listings for sellers to fix.

</div>

## Thresholds

Calibrate each attribute head (temperature scaling on a held-out set), then choose two thresholds per attribute: above the high one, apply automatically; between them, send to review or ask the seller to confirm; below, leave blank. Set thresholds from the precision the downstream use needs: filters need high precision; recommendation features can accept lower.

## Serving

~~~dg-flow Tagging pipeline. Listings are tagged in micro-batches shortly after they are created or edited.
New or edited listing -(images, title, description)-> Encoder -(fused embedding)-> *Category and attribute heads -(calibrated scores)-> Threshold policy -(auto-apply, review, blank)-> Catalogue and search index
~~~

2 million listings a day is about 23 per second on average; micro-batches on a few GPUs handle it easily. Re-tag the full catalogue in a background job when the model or taxonomy changes.

## Monitoring

Track coverage and score distributions by category, seller corrections, review queue size, and filter engagement. A rise in corrections for one category often means a taxonomy or seller-behaviour change.

## Trade-offs

- Precision versus coverage, set per attribute by its downstream use.
- One large shared model is easy to maintain but harder to update for one attribute; small heads on a frozen encoder make targeted changes cheap.
- Zero-shot tags arrive immediately but are less accurate than trained heads.

## Likely follow-up questions

- "How do you handle a new category added to the taxonomy?" Add its name as a text prompt for zero-shot scoring, route its predictions to review, and train it into the hierarchical head once a few hundred reviewed examples exist.
- "What if sellers game tags to appear in more filters?" Predicted tags are computed independently of seller claims; large disagreements between the two are flagged, and repeated offenders lose the ability to set certain fields.
- "How do you know a tag improves the business?" Run an experiment where filtered search uses predicted tags in one arm and seller tags only in the other, and compare purchases per search and zero-result rates.

<div class="take">
<div class="lab">Key takeaway</div>

tag with a shared image-text encoder and small per-attribute heads gated by category, learn from noisy seller labels plus targeted reviews, bootstrap new attributes zero-shot, and apply tags only above per-attribute, calibrated thresholds set by their downstream use.

</div>
