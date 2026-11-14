---
id: "02.05"
course: "ml-system-design"
module: "02"
slug: "people-who-adapt-to-the-model"
title: "People who adapt to the model"
keyIdeas: ["In fraud, spam, abuse and ranking manipulation, people change their behaviour in response to the model. Past data describes attackers who have already moved on.", "Defences combine fast-updating models, features that are costly to fake, human review, rules for known attacks and anomaly detection for unknown ones.", "The model's own decisions shape its future labels: blocked transactions never get chargebacks, so they need a separate labelling path such as a small random holdout.", "Treat the system as a contest. Monitor attack patterns, keep some defences undisclosed, and plan for frequent retraining."]
question: "Your account-takeover model blocks risky logins. Six months after launch, recall on newly labelled attacks has dropped from 85% to 55%, even though the model is retrained monthly. Give three reasons this happens and three changes to the system."
answer: "Reasons: (1) attackers adapted, for example by using residential proxies or warming up accounts, so the features the model relied on no longer separate attacks; (2) feedback loop on labels: blocked logins never become confirmed takeovers, so the only attacks in new training data are the ones the model missed, and the model under-weights patterns it used to catch; (3) monthly retraining is too slow compared with how quickly campaigns change. Changes: keep a small random holdout of borderline logins that are challenged with a step-up check instead of blocked, so you still observe outcomes; add features that are costly for attackers to fake (device history, behavioural signals, graph features linking accounts and IPs) and an unsupervised anomaly detector for new patterns; and update faster, with rules analysts can deploy within hours for active campaigns plus more frequent or online model updates."
sources: [{"title": "Biggio and Roli, Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning (2018)", "url": "https://arxiv.org/abs/1712.03141", "note": "Evasion, poisoning and the arms race with attackers."}, {"title": "Wikipedia, Adversarial machine learning", "url": "https://en.wikipedia.org/wiki/Adversarial_machine_learning", "note": "Overview of evasion and poisoning attacks."}, {"title": "Sculley et al., Hidden Technical Debt in Machine Learning Systems (2015)", "url": "https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html", "note": "Feedback loops and the model affecting its own data."}]
practice: ["isolation-forest", "population-stability-index", "two-sample-ks-drift", "text-deduplication-minhash"]
tags: ["adversarial ML", "feedback"]
---

## When the data fights back

Most ML assumes the world stays roughly the same between training and serving. That fails when people benefit from fooling the model. Spammers rewrite messages, fraudsters change merchants and devices, sellers stuff keywords, and bot operators mimic human timing. Each time the model improves, the cheapest attacks stop working and new ones appear. Past data then describes attackers who have already changed tactics.

~~~dg-flow The adversarial loop. Every defence changes the data that the next model is trained on.
Attackers -(try tactics)-> Requests -(score)-> *Model -(block / allow)-> Outcomes -(only allowed ones get labels)-> Training data -(retrain)-> Model
Model -(blocked tactics stop working)-> Attackers
~~~

## How adaptation shows up

- **Recall decays** on new attacks while precision on old ones stays high.
- **Feature importance shifts** as signals the model relied on stop separating classes.
- **Probing**: bursts of slightly varied attempts, looking for the decision boundary.
- **Campaigns**: sudden spikes from coordinated accounts that share infrastructure.

## Defences in layers

| Layer | Role | Update speed
| --- | --- | --- |
| Rules and blocklists | stop known attacks immediately | minutes
| Supervised model | generalise from labelled attacks | days to weeks
| Anomaly detection | flag behaviour unlike normal users | continuous
| Graph and link analysis | connect accounts sharing devices, IPs or payment methods | hours
| Human review | judge ambiguous cases, produce labels | hours to days
| Friction | step-up checks (CAPTCHA, one-time codes) for medium risk | instant

No single layer is enough. Rules are fast but brittle; models generalise but lag; anomaly detectors catch new patterns but produce false alarms; humans are accurate but expensive.

## Features that are hard to fake

Prefer signals that cost an attacker real money or time: account age and history, device and payment reputation built over months, graph connections to known bad actors, behaviour across many sessions. Signals that are free to change, such as user-agent strings or message wording, are useful but decay fast.

## The model labels its own data

If a model blocks a transaction, you never learn whether it was fraud. Over time, training data contains only attacks the model missed, so it slowly forgets what it used to catch. Fixes:

- **holdout traffic**: let a small random fraction of would-be-blocked cases through a softer check (step-up authentication, delayed fulfilment) to keep observing outcomes;
- **analyst review** of a sample of blocked cases to label them;
- **keep old positives** in training even if the pattern has become rare, with sensible weighting.

<div class="ex">
<div class="lab">Worked example</div>

**a spam filter that was taught to fail**

A forum's spam classifier learned that messages containing links to a new domain were spam. Spammers switched to image text and URL shorteners within a week. Because the filter deleted spam before anyone saw it, there were few new labelled examples. Adding a "report spam" button and a small review queue of borderline messages restored a labelled stream and recall recovered over two retraining cycles.

</div>

## Operating the contest

- **Retrain often** and monitor recall on the newest labelled attacks, not just overall metrics.
- **Keep some defences quiet**: do not reveal exact thresholds or which signal triggered a block.
- **Measure attack cost**: how many attempts an attacker needs to succeed. Raising that cost is the real goal.
- **Plan for incidents**: a playbook for active campaigns, with rules analysts can ship in hours, a way to roll them back, and a review a week later to fold the rule into the model or retire it so that rules do not pile up unexamined.

## Measuring success

Overall precision and recall are not enough in an adversarial setting. Track recall on attacks first seen in the last week, the time from a new campaign starting to it being blocked, and losses per thousand transactions or messages. These measure how quickly the system adapts, which is what decides who wins. A model with slightly lower overall AUC but a faster update loop is often the better choice.

<div class="take">
<div class="lab">Key takeaway</div>

when people adapt to the model, past data goes stale and the model's decisions hide its own labels; combine fast rules, regularly retrained models, anomaly detection, costly-to-fake features and a holdout or review stream that keeps outcomes observable.

</div>
