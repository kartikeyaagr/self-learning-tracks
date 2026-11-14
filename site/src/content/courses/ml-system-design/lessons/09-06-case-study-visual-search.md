---
id: "09.06"
course: "ml-system-design"
module: "09"
slug: "case-study-visual-search"
title: "Case study: visual search"
keyIdeas: ["Visual search maps a query photo to an embedding and finds the nearest items in a large index; quality depends on the embedding and on cropping to the object of interest.", "Embeddings are trained with contrastive objectives on pairs that should match (same product, user-confirmed matches) and pairs that should not.", "An approximate nearest-neighbour index makes retrieval over hundreds of millions of items fast, at a small cost in recall.", "A re-ranking stage adds business signals (availability, price, quality) and diversity to the visually similar candidates."]
question: "Design a visual search feature for a shopping app: a user photographs an item (a lamp, a jacket) and sees similar products from a catalogue of 200 million items, within 500 ms. Cover embeddings, indexing, the request path and evaluation."
answer: "Detect objects in the query photo and let the user pick one, so the embedding describes the object rather than the scene. Embed the crop with an image encoder fine-tuned contrastively on matching pairs (product photos versus user photos of the same product, plus clicks and purchases from search). Precompute embeddings for all catalogue images and store them in an approximate nearest-neighbour index (HNSW or IVF with product quantisation), sharded by category. At query time, retrieve the top 500 by cosine similarity, filter by availability and region, and re-rank with a model using visual similarity, price, quality and popularity. Evaluate with recall at k on held-out matching pairs and with human relevance ratings; online, measure clicks and purchases per search. Monitor index freshness, embedding drift after model updates and latency. Trade-offs: index recall versus memory and speed, and exact-match versus style-similar results."
rubric: ["Clarifies query type, catalogue size, latency and what counts as a good match", "Crops to the object with detection or user selection before embedding", "Trains embeddings contrastively on matching pairs, including user photos versus catalogue photos", "Uses an approximate nearest-neighbour index and explains its recall-speed-memory trade-off", "Adds filtering and a re-ranking stage with business signals", "Evaluates with recall at k on matching pairs and human relevance judgements", "Plans for re-embedding the catalogue when the model changes, and for new items", "Monitors latency, index freshness and online engagement"]
sources: [{"title": "Jing et al., Visual Search at Pinterest (2015)", "url": "https://arxiv.org/abs/1505.07647", "note": "Detection plus embeddings for search over a large image corpus."}, {"title": "Zhai et al., Visual Discovery at Pinterest (2017)", "url": "https://arxiv.org/abs/1702.04680", "note": "Product features built on visual search."}, {"title": "Johnson, Douze and Jegou, Billion-scale similarity search with GPUs (2017)", "url": "https://arxiv.org/abs/1702.08734", "note": "Approximate nearest-neighbour search at very large scale."}, {"title": "Malkov and Yashunin, Efficient and robust approximate nearest neighbor search using HNSW graphs (2016)", "url": "https://arxiv.org/abs/1603.09320", "note": "A widely used graph-based index."}]
practice: ["clip-cosine-retrieval", "symmetric-clip-contrastive-loss", "embedding-similarity", "non-maximum-suppression"]
tags: ["visual search", "image embeddings", "approximate nearest neighbour", "object detection"]
---

## Requirements

Clarify the goal: find the exact product, or similar-looking products? Users photographing a lamp in a friend's home want the exact one if it is sold, otherwise close alternatives. Assume a catalogue of 200 million images across 50 million products, new items daily, a 500 ms budget end to end, and results limited to items in stock in the user's region.

## Metrics

Offline: recall at 10 and 50 on held-out pairs of a user photo and its correct product, plus human relevance ratings on a sample of queries (exact, similar, unrelated). Online: click-through, add-to-cart and purchases per visual search, and the share of searches with no click.

## Data

- Catalogue images: clean, studio-style photos.
- User photos: cluttered scenes, varied lighting and angles. The domain gap between these two is the main difficulty.
- Matching pairs: products that users found and bought after a visual search, reviews with photos, and labelled pairs from annotators.

## Cropping

A whole-scene embedding of a living room matches other living rooms. Run an object detector on the query, show boxes for the main objects, and let the user tap one (default to the largest or most central). Catalogue images can be processed the same way, so that the index contains object-level embeddings.

## Embeddings

Start from a pretrained image encoder (a CLIP-style model trained on image-text pairs is a good start, as it already groups images by meaning). Fine-tune with a contrastive loss: pull a user photo and its matching catalogue photo together, push other products apart. Hard negatives, such as similar items of a different brand, sharpen fine distinctions. Normalise embeddings so cosine similarity is a dot product.

<div class="ex">
<div class="lab">Worked example</div>

**closing the domain gap**

A furniture retailer's first model, trained only on catalogue images, reached 70% recall at 10 when tested on catalogue photos but 22% on user photos. Adding 300,000 user-photo and catalogue-photo pairs (from purchases after search and from reviews) and training contrastively across the two domains raised recall on user photos to 58%. The change in data mattered more than any change in architecture.

</div>

## Indexing and serving

Embedding 200 million images at 256 dimensions in 8-bit form takes about 51 GB; product quantisation can shrink this several times. Use an HNSW or IVF index, sharded by top-level category, with replicas for throughput.

~~~dg-flow Visual search request. Catalogue embeddings are computed offline; the request embeds one crop and searches the index.
Catalogue images -(batch embedding)-> ANN index by category
Query photo -(object detection, user picks a box)-> Crop -(image encoder)-> *ANN search -(top 500)-> Filter and re-rank -(top 20)-> Results
~~~

Re-rank with a light model using visual similarity, price, ratings, availability and popularity, and add diversity so the top results are not twenty copies of one item from different sellers.

## Monitoring and updates

New products are embedded and inserted daily. A new embedding model requires re-embedding the whole catalogue and rebuilding the index; run both in parallel and switch atomically, because embeddings from different models are not comparable. Monitor latency by stage, zero-click rate, and recall on a fixed test set after each change.

## Trade-offs

- Index recall versus speed and memory: tune the search parameters on a recall-latency curve.
- Exact match versus style similarity: show exact matches first when found, then similar items.
- Larger encoders improve quality but cost more per query and per catalogue rebuild.

## Likely follow-up questions

- "How do you handle a query for an item you do not sell?" Show the closest style matches and say clearly that they are similar rather than exact; the re-ranker can use a similarity threshold to decide.
- "How do you add text to the query, such as 'like this but in red'?" With a shared image-text space, add the text embedding to the image embedding or filter candidates by the colour attribute.
- "How large an index can one machine hold?" At 64 bytes per item after product quantisation, 200 million items fit in about 13 GB, so a few replicas per shard suffice; the limit is usually query throughput, not memory.

<div class="take">
<div class="lab">Key takeaway</div>

crop to the object, embed with a model trained contrastively across user and catalogue photos, search an approximate nearest-neighbour index, then filter and re-rank with business signals; plan full re-embedding whenever the model changes.

</div>
