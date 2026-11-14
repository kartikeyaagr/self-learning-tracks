---
id: "09.05"
course: "ml-system-design"
module: "09"
slug: "case-study-email-spam-filtering"
title: "Case study: email spam filtering"
keyIdeas: ["Spam filtering is high-volume binary classification where a false positive (a lost real email) is far more costly than a false negative.", "Sender reputation, authentication and volume signals catch most spam before content models run; content models handle the rest.", "User actions (\"report spam\", \"not spam\") provide labels quickly, but they are noisy and biased towards what users see.", "Spammers adapt continually, so the system needs fast updates, near-duplicate clustering of campaigns, and per-user personalisation."]
question: "Design a spam filter for an email provider receiving 10 billion messages a day. Decide inbox, spam folder or reject for each message. Explain the layers, labels, the operating point and how you handle spammers who adapt."
answer: "Use layers. At connection time, reject based on IP and domain reputation, authentication (SPF, DKIM, DMARC) and rate limits; this removes most volume cheaply. For accepted mail, a classifier uses sender reputation, authentication results, header features, links and their domain reputation, and content features from a compact text model, plus campaign features from near-duplicate clustering (many recipients getting near-identical messages). Labels come from user reports, \"not spam\" rescues and honeypot accounts, de-noised by weighting reliable reporters. Set the threshold for a very low false-positive rate on legitimate mail, with per-user adjustments learned from that user's actions. Evaluate on a stratified human-labelled sample. Monitor spam-folder rescue rate, report rate and campaign bursts; push updated reputation lists within minutes. Trade-offs: a global versus personalised notion of spam, and catch rate versus lost legitimate mail."
rubric: ["Clarifies volume, actions and the asymmetric cost of false positives", "Puts cheap reputation, authentication and rate checks before content models", "Describes labels from user reports, rescues and honeypots, and handles their noise", "Uses campaign or near-duplicate features across recipients", "Sets the operating point for a very low false-positive rate on legitimate mail", "Includes personalisation for borderline categories such as newsletters", "Handles adversarial adaptation with fast reputation updates and frequent retraining", "Evaluates on a stratified human-labelled sample and monitors rescue rates"]
sources: [{"title": "Wikipedia, Naive Bayes spam filtering", "url": "https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering", "note": "The classic content-based approach and its limits."}, {"title": "Wikipedia, Email spam", "url": "https://en.wikipedia.org/wiki/Email_spam", "note": "Background on spam campaigns and countermeasures."}, {"title": "Wikipedia, Locality-sensitive hashing", "url": "https://en.wikipedia.org/wiki/Locality-sensitive_hashing", "note": "Grouping near-duplicate messages into campaigns."}, {"title": "Google, Rules of Machine Learning", "url": "https://developers.google.com/machine-learning/guides/rules-of-ml", "note": "Combining heuristics with learned models."}]
practice: ["tf-idf", "text-classification", "text-deduplication-minhash", "precision-recall-at-k"]
tags: ["spam filtering", "text classification", "adversarial", "feedback loops"]
---

## Requirements

Clarify the actions: reject at the server, deliver to the spam folder, or deliver to the inbox. Also ask about scope: spam only, or phishing and malware too (these need stricter handling). Assume 10 billion messages a day (about 115,000 per second on average), a budget of tens of milliseconds per message, and a very low tolerance for losing legitimate mail.

## Metrics

The key metric is the false-positive rate on legitimate mail, ideally well under 0.1%, because a missed job offer or invoice costs far more than an extra spam message. Second is spam catch rate. Online signals: the rate at which users move mail out of spam ("rescues"), report spam from the inbox, and complaint volume.

## Layers

1. Connection level: IP reputation, sending rate, and whether the domain is newly registered. Rejecting here saves all further cost.
2. Authentication: SPF, DKIM and DMARC results. Failing authentication for a domain that claims to be a bank is a strong signal.
3. Classifier on accepted mail: headers, links, attachments and content.
4. Personalisation: adjust for each user's own behaviour.

## Data and labels

- User reports and rescues: plentiful and fast but noisy; some users report newsletters they signed up for. Weight users by their historical agreement with consensus.
- Honeypot addresses that no real person uses: any mail they receive is unsolicited.
- Human review on a stratified sample, used for evaluation and to calibrate the noisy labels.

Labels are biased by the current filter: users see and report only what reached the inbox. Sample some spam-folder mail for review to learn about errors in that direction.

## Features

- Sender: domain and IP reputation, age, history of complaints and rescues.
- Authentication results and header anomalies.
- Links: domain reputation, URL shorteners, mismatch between visible text and target.
- Content: a compact text model over subject and body (bag of words with TF-IDF, or a small transformer distilled for speed).
- Campaign: how many recipients received a near-duplicate of this message recently, using MinHash signatures.

<div class="ex">
<div class="lab">Worked example</div>

**catching a campaign by its shape**

A spam campaign varied every message with random words and images, defeating content features. But MinHash over the message structure grouped 2 million messages sent within an hour from 3,000 new domains. A campaign feature (cluster size, number of sending domains, share of new domains) let the classifier catch the whole wave after the first few thousand reports, and the reputation of the 3,000 domains dropped within minutes.

</div>

## Model

A gradient-boosted model over reputation, header, link and campaign features, with a text model's score as one input, is a strong design. Calibrate it, then set the threshold to meet the false-positive target on a recent labelled sample. Personalisation can be a small per-user bias, learned from that user's reports and rescues, applied mainly to borderline categories such as promotions.

## Serving

~~~dg-flow Spam filtering layers. Each layer is cheaper than the next and removes volume before it.
Incoming connection -(IP and domain reputation, rate limits)-> Accepted message -(SPF, DKIM, DMARC)-> Feature extraction -(campaign lookup)-> *Spam classifier -(per-user adjustment)-> Inbox or spam folder
~~~

## Monitoring and adaptation

Track rescue and report rates per hour, the share of new sending domains, and cluster sizes. Reputation lists should update within minutes from fresh reports; models retrain daily. Watch for sudden drops in catch rate, which usually mean a new technique.

## Trade-offs

- Global versus personalised: personalisation fits newsletters well but can hide phishing; never personalise away security verdicts.
- Catch rate versus false positives: tilted strongly towards protecting legitimate mail.
- Explaining why mail was filtered helps users but can help spammers probe the system.

## Likely follow-up questions

- "A bank complains that its statements go to spam. What do you look at?" Authentication results for its sending domain, recent changes to its sending IPs, and the content features that pushed the score; often a misconfigured DKIM record is the cause.
- "How do you handle a new sender with no reputation?" Treat it as neutral, apply stricter content scoring, and let reputation build from recipients' actions over the first days.
- "Would a large language model help?" As a slow second pass on borderline mail or for phishing intent, yes; it is too expensive to run on every message at this volume.
- "How fast must the system react to a new campaign?" Minutes; reputation and campaign features carry the response while models retrain.

<div class="take">
<div class="lab">Key takeaway</div>

filter in layers from cheap reputation and authentication to content models, use campaign features across recipients, learn from noisy user feedback with care, and set the threshold to protect legitimate mail first.

</div>
