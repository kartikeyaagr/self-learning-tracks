---
id: "01.01"
course: "ml-system-design"
module: "01"
slug: "what-ml-system-design-is"
title: "What ML system design is"
keyIdeas: ["An ML system is a data pipeline, a training pipeline, a serving path and a feedback loop. The model is one component, and rarely the one that fails first.", "**Design** means choosing how each part works and how the parts connect: what is predicted, from which data, how fast, at what cost and how you will know it still works.", "Every design decision trades off accuracy, latency, cost, freshness, and how easy the system is to operate.", "A good design starts from the product decision the model supports and works backwards to data and models, not forwards from a favourite algorithm."]
question: "A team says their \"ML system\" is a gradient-boosted model with 0.91 AUC in a notebook. List the other components that must exist before it can affect users, and for each name one way it could fail silently."
answer: "Beyond the model you need: (1) a data pipeline that produces training examples, which can silently drop a segment or change a join; (2) a labelling process, where labels can be delayed or biased by the current system; (3) a training pipeline with versioned data and code, where an unpinned dependency can change preprocessing; (4) a serving path that computes features for live requests, which can diverge from the offline computation; (5) logging of inputs, predictions and outcomes, which can stop capturing a field; (6) monitoring and alerting, where thresholds that are too loose miss drift; and (7) a rollout and rollback process, where a bad model can reach everyone at once without a canary. None of these raise an exception when they go wrong; the model just keeps producing numbers."
sources: [{"title": "Sculley et al., Hidden Technical Debt in Machine Learning Systems (2015)", "url": "https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html", "note": "Why the code around the model dominates cost and risk."}, {"title": "Google, Rules of Machine Learning", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Practical rules for the first ML system in a product."}, {"title": "Chip Huyen, Designing Machine Learning Systems (book page)", "url": "https://huyenchip.com/books/", "note": "A book-length treatment of the same lifecycle."}]
practice: ["training-serving-skew", "hashed-dataset-split", "population-stability-index"]
tags: ["ML system design", "system boundaries"]
---

## More than a model

A model in a notebook answers one question: how well can these features predict this label on this dataset? An ML system has to answer many more. Where do the examples come from tomorrow? Who labels them, and how late? How are features computed when a request arrives in 30 ms? What happens when the model is down? How will anyone notice that quality has dropped?

ML system design is the practice of answering those questions on purpose. It overlaps with ordinary system design, covered in System Design 101 (caching, queues, storage and scaling), and with ML in production, covered in the Machine Learning course (skew, reproducible data, monitoring, experiments). This course sits between them: it is about the decisions that are specific to systems whose behaviour is learned from data.

~~~dg-flow The loop every ML system runs. Each arrow is a place where things can go wrong without an error.
Product decision -(defines)-> Prediction task -(needs)-> Data and labels -(train)-> *Model -(serve)-> Predictions -(change)-> User behaviour -(logged as)-> Data and labels
~~~

## The parts and their contracts

| Part | What it does | Typical silent failure
| --- | --- | --- |
| Data pipeline | turns raw events into examples | a join drops new users
| Labels | attaches outcomes to examples | labels arrive weeks late, so recent data looks negative
| Training | fits and versions models | preprocessing differs between runs
| Serving | computes features and predictions live | a feature is computed differently than in training
| Logging | records inputs, outputs and outcomes | a field stops being logged
| Monitoring | detects drift and quality loss | alert thresholds are too loose
| Rollout | releases models gradually | a bad model reaches 100% of traffic

Each part has an implicit contract with the next. The serving path promises to compute `days_since_signup` exactly as training did; the logging pipeline promises to record what was served. Design makes these contracts explicit so they can be checked.

## The trade-offs you are balancing

Almost every decision moves one of five dials:

- **Quality**: how good the predictions are on the metric that matters.
- **Latency**: how long a prediction takes, usually measured at p99.
- **Cost**: compute for training and serving, storage, labelling.
- **Freshness**: how quickly the system reflects new data.
- **Operability**: how easy it is to debug, roll back and hand to another team.

A larger model raises quality and cost. Precomputing predictions cuts latency but hurts freshness. A feature from a slow service may add 1% quality and 40 ms of latency. Stating these trade-offs out loud, with rough numbers, is most of what interviewers mean by "design".

<div class="ex">
<div class="lab">Worked example</div>

**the same task, two designs**

A food-delivery app wants to show "estimated delivery time". Design A computes it nightly per restaurant and stores it: cheap, fast, but wrong at lunch rush. Design B predicts per order at checkout using live courier positions: accurate, but needs a streaming feature pipeline and a 50 ms model. A common answer is a hybrid: a nightly baseline per restaurant plus a light online correction from current load. The right choice depends on how much a wrong estimate costs the business.

</div>

## Working backwards

Start from the decision the prediction drives. If a fraud score blocks payments, false positives lose revenue and anger customers, so precision at the operating point matters more than AUC. If a ranking model orders a feed, what matters is engagement over weeks, not click-through on one session. Writing down the decision first tells you the metric, the latency budget and the cost of errors. Only then pick data and models.

The rest of this course follows that order: framing (this module), data and labels, features, models, evaluation, serving, monitoring, then interview practice and 21 case studies that put it together.

## How this course uses case studies

The last three modules are case studies: ETA prediction, pricing, fraud, spam, visual search, feeds, ads, recommendations, autocomplete, retrieval-augmented generation and more. Each follows the same template: requirements, metrics, data and labels, features, model, serving, monitoring and trade-offs, ending with a rubric of what a strong design covers. Reading them after the concept modules shows how the same handful of decisions reappear in very different products. Trying each prompt yourself before reading the model design is the fastest way to build the habit.

<div class="take">
<div class="lab">Key takeaway</div>

an ML system is data, labels, training, serving, logging, monitoring and rollout around a model; design means choosing each part from the product decision backwards and stating the quality, latency, cost, freshness and operability trade-offs.

</div>
