---
id: "01.03"
course: "ml-system-design"
module: "01"
slug: "training-and-prediction"
title: "Training and prediction"
keyIdeas: ["Training and prediction are two different programs that must agree: same features, same preprocessing, same meaning of every input.", "Training runs offline on historical data, optimises for throughput and can take hours; prediction runs on live data and is bounded by a latency budget.", "The **prediction time** of each training example matters: features must be computed as they would have been known at that moment, or the model learns from the future.", "Predictions can be computed in batch ahead of time, online per request, or from a stream; the choice sets freshness, latency and cost."]
question: "Your churn model's training set joins each customer's label (churned in the next 30 days) to their current account features. Offline AUC is 0.93; in production it is 0.71. What went wrong, and how would you rebuild the training set?"
answer: "The features were taken as of today rather than as of the moment each prediction would have been made, so they contain information from after the prediction time. For churned customers, today's features already reflect churn (zero recent logins, a cancelled plan, a support ticket), which makes the problem look easy offline. At serving time those signals are not yet present. Fix it by choosing a prediction timestamp for each example (for example, the first of each month), computing every feature from data available strictly before that timestamp (a point-in-time join against feature history or event logs), and computing the label from the 30 days after it. Validate on later months than you train on. Offline AUC will drop toward the production value, which is the honest number."
sources: [{"title": "Google, Rules of Machine Learning, Rules 29-37 (training-serving skew)", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "How skew arises and how to prevent it."}, {"title": "Google Cloud, MLOps: continuous delivery and automation pipelines", "url": "https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning", "note": "How training and serving pipelines fit together."}]
practice: ["training-serving-skew", "point-in-time-feature-join", "hashed-dataset-split"]
tags: ["training vs inference", "batch vs online"]
---

## Two programs, one contract

Training reads a large table of historical examples, fits parameters and writes a model artifact. Prediction loads that artifact and applies it to new inputs. They often run on different machines, in different languages, written by different people. The model only works if both programs agree on what every input means.

~~~dg-flow Training and serving share feature definitions and the model artifact. Everything else differs.
Historical events -(point-in-time features + labels)-> Training job -(model artifact)-> *Model registry -(deploy)-> Serving -(live features)-> Prediction
~~~

| Aspect | Training | Prediction
| --- | --- | --- |
| Data | months of history | one request, or today's batch
| Goal | best fit on a metric | right answer within the latency budget
| Time budget | minutes to days | milliseconds to hours
| Hardware | many GPUs or a big CPU job | CPU servers, GPUs, or a phone
| Failure cost | a wasted run | wrong decisions for real users

## Prediction time

Every training example corresponds to a moment when the model would have been asked to predict. Call it $t_0$. Features must use only data available before $t_0$; the label describes what happened after $t_0$. Breaking this rule is called **leakage**, and it produces models that look excellent offline and fail live.

The fix is a **point-in-time join**: for each example, look up each feature's value as of $t_0$, not as of today. This needs either a history of feature values (a feature store with time travel) or the raw event log from which features can be recomputed. Choosing $t_0$ is a design decision. For churn, it might be the first of each month; for fraud, the authorisation time of each transaction; for ranking, the time each impression was served.

<div class="ex">
<div class="lab">Worked example</div>

**a leak through an "updated" field**

A loan default model used `account_status` from the customer table. For defaulted loans the status had since been set to "collections". The model learned that "collections" predicts default, reaching 0.99 AUC. Rebuilding the table with status as of the application date brought AUC to 0.74, which matched production.

</div>

## Three ways to serve predictions

**Batch** prediction scores every entity on a schedule and stores the results for lookup. It suits churn scores, daily recommendations or weekly forecasts. It is cheap per prediction and needs no low-latency model, but predictions are only as fresh as the last run and new entities have none.

**Online** prediction computes the answer per request. It is needed when the input only exists at request time: a search query, a transaction, the current basket. It needs a fast model and fast features.

**Streaming** prediction sits between them: a stream processor updates features (clicks in the last 5 minutes) and possibly predictions as events arrive. It gives freshness without computing everything at request time, at the cost of operating a streaming system.

Many real systems combine them. Lesson 06.01 goes deeper into the trade-offs.

## Keeping the two programs in step

Three habits remove most training/serving mismatch:

- define each feature once and compute it from the same code in both paths (a feature store, 03.02);
- log the exact features used at serving time, and train future models from those logs;
- version the model, the feature definitions and the preprocessing together, so a deploy can never mix them.

The Machine Learning course's "From notebook to service" lesson (mlcore 09.01) covers skew detection in detail.

## Where predictions run

Prediction does not always run on a server. Models on phones or browsers avoid network latency and keep data on the device, but must be small and are updated only when the app updates. Models inside a database or a stream processor avoid moving data. The location changes which features are available: an on-device keyboard model sees what is typed but not the server-side history, and a server model sees history but adds a network round trip. Decide where prediction runs early, because it constrains the model size, the feature set and how quickly you can ship fixes.

<div class="take">
<div class="lab">Key takeaway</div>

training and prediction are separate programs bound by one contract; build every example as of its prediction time, choose batch, online or streaming serving by freshness and latency, and share feature code so the contract holds.

</div>
