---
id: "11.02"
course: "ml-system-design"
module: "11"
slug: "case-study-blurring-faces-and-licence-plates-in-street-image"
title: "Case study: blurring faces and licence plates in street imagery"
keyIdeas: ["Privacy blurring is a detection problem where recall matters most: a missed face is a privacy failure, while an extra blur on a sign is a small cost.", "Street imagery has very small, distant, partly hidden and angled faces and plates, so the detector, input resolution and tiling must target small objects.", "The system runs as a large batch pipeline over billions of images, so cost per image and throughput drive model and hardware choices.", "Evaluation needs a carefully labelled test set with object size and conditions recorded, plus a process for user reports of missed blurs."]
question: "Design a system that blurs faces and licence plates in panoramic street imagery before publication. The pipeline processes 50 million new panoramas a month, each about 8,000 by 4,000 pixels. Explain the detection approach, operating point, evaluation, cost and handling of misses."
answer: "Split each panorama into overlapping tiles (or project to cube faces) at full resolution, because many faces and plates are only 10 to 20 pixels tall. Run a detector trained for small objects (single-stage with feature pyramids, or a two-stage model) with separate face and plate classes, then merge detections across tiles with non-maximum suppression. Choose a low threshold for very high recall (for example 99% on faces taller than 10 pixels) and accept extra blurs on signs and statues; optionally a second, more accurate classifier removes obvious false positives only when very confident. Blur generously (expand boxes, irreversible blur). Labels: annotators mark faces and plates in diverse sampled tiles, with size, occlusion and region recorded. Evaluate recall by size bucket, region and condition. Run as a batch pipeline on GPUs with tiling and batching; at about 50 tiles per panorama this is 2.5 billion tiles a month. Users can report missed blurs; fixes are applied quickly and become training data."
rubric: ["States that recall is the priority and explains the asymmetric costs", "Handles small objects with tiling or full-resolution processing and a suitable detector", "Merges detections across overlapping tiles with non-maximum suppression", "Sets the threshold for a recall target by object size and accepts false positives", "Builds a labelled test set stratified by size, occlusion, region and plate formats", "Estimates batch throughput and cost, and designs the pipeline accordingly", "Applies irreversible, generously sized blurs before publication", "Provides a user report path and feeds misses back into training"]
sources: [{"title": "Frome et al., Large-scale Privacy Protection in Google Street View (2009)", "url": "https://research.google/pubs/large-scale-privacy-protection-in-google-street-view/", "note": "A reference design for face and plate blurring at scale."}, {"title": "Deng et al., RetinaFace: Single-stage Dense Face Localisation in the Wild (2019)", "url": "https://arxiv.org/abs/1905.00641", "note": "A face detector that handles small and varied faces."}, {"title": "Ren et al., Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (2015)", "url": "https://arxiv.org/abs/1506.01497", "note": "Two-stage detection, a strong baseline."}, {"title": "Wikipedia, Face detection", "url": "https://en.wikipedia.org/wiki/Face_detection", "note": "Background on approaches and difficulties."}]
practice: ["non-maximum-suppression", "iou-matrix", "gaussian-blur-2d", "anchor-box-generation"]
tags: ["object detection", "privacy", "recall-first", "large-scale batch inference"]
---

## Requirements

Clarify the constraint: nothing is published until it is blurred. Faces and licence plates must be blurred irreversibly; other personal items (house numbers) may be added later. Assume 50 million panoramas a month (about 19 per second on average), each 8,000 by 4,000 pixels, processed in batch within a day or two of capture.

## Metrics

Recall is primary, measured on a labelled test set and broken down by object size, occlusion, angle, lighting and region (plate formats differ by country). A target might be 99% recall on faces at least 10 pixels tall and 98% on plates. Precision matters less, but too many false blurs damage the product, so track false blur rate per image as a guardrail.

## Data and labels

Sample tiles across countries, seasons, times of day and road types. Annotators draw boxes on faces (including side and back-of-head views, people in vehicles and on posters, which policy may also require blurring) and plates (including partial ones). Record size and occlusion so that evaluation can be sliced. Hard negatives (faces on billboards, signs that resemble plates) help precision.

## Handling small objects

In an 8,000-pixel-wide panorama, a pedestrian across the road may have a face 12 pixels tall. Downscaling the panorama would erase it. Instead:

- Project the panorama into cube faces or split into overlapping tiles (for example 1,024 by 1,024 with 128 pixels of overlap), keeping full resolution.
- Use a detector with feature pyramids and small anchor sizes, or a two-stage model, trained on tiles.
- Merge detections across tile overlaps with non-maximum suppression.

<div class="ex">
<div class="lab">Worked example</div>

**the faces at the edge of the tile**

An early version used non-overlapping tiles. Faces cut by a tile boundary were split into two halves that each looked unlike a face, and recall on those was under 70%. Adding 128-pixel overlaps and merging detections across tiles raised recall at boundaries to match the rest of the image, at the cost of about 25% more tiles.

</div>

## Model and operating point

Train face and plate classes together or as two heads. Set thresholds per class and per size bucket to meet recall targets; this usually means accepting false positives such as statues, mannequins and printed faces. A second-stage classifier may remove false positives only when very confident, and only for large, clear detections. Expand each box by a margin before blurring, and use a strong blur or pixelation that cannot be reversed.

## Serving as a batch pipeline

~~~dg-flow Blurring pipeline. Nothing reaches publication without passing through the detector and blur step.
New panoramas -(projection and tiling)-> Tiles -(batched GPU inference)-> *Face and plate detector -(NMS across tiles)-> Boxes -(expand, blur)-> Blurred panoramas -(publish)-> Public imagery
User reports -(manual fix)-> Blurred panoramas
User reports -(new labels)-> Training data
~~~

With about 50 tiles per panorama, the system processes 2.5 billion tiles a month, about 1,000 per second on average. A mid-sized detector at a few milliseconds per tile per GPU needs a few dozen GPUs running constantly, with headroom for re-processing when the model improves.

## Monitoring

Track detections per panorama by region (a sudden drop suggests a camera or pipeline issue), user reports of missed blurs per million panoramas, and recall on a fixed test set after each model update. Re-process older imagery when a much better model ships.

## Likely follow-up questions

- "Can you un-blur false positives like shop signs?" Only with strong evidence and preferably a second model plus review, because the risk is asymmetric.
- "Should original images be kept?" Keep unblurred originals only under strict access controls and retention limits, or not at all, depending on law and policy.
- "How do you handle new countries with different plates?" Collect and label local samples before launch, and test recall per format.

## Trade-offs

- Recall versus false blurs: set heavily towards recall.
- Tile size and overlap versus cost.
- A larger model improves recall on small objects but raises the cost of each re-processing pass.

<div class="take">
<div class="lab">Key takeaway</div>

treat privacy blurring as recall-first detection of small objects, tile at full resolution with overlaps, set per-size thresholds for recall targets, run it as a cost-aware batch pipeline, and close the loop with user reports.

</div>
