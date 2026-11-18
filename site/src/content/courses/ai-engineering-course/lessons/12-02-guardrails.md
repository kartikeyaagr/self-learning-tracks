---
id: "12.02"
course: "ai-engineering-course"
module: "12"
slug: "guardrails"
title: "Guardrails"
keyIdeas: ["A guardrail is a check outside the main model call that can block, rewrite or reroute an input or an output; it complements alignment and permissions, it does not replace them.", "Three kinds of checker: rules and regexes (microseconds, exact, brittle), small classifiers (milliseconds, cheap, fixed taxonomy) and LLM checks (hundreds of milliseconds, flexible, costly and attackable).", "PII redaction replaces detected entities with typed placeholders before the model and the logs see them, and restores them only where policy allows.", "Layer guards by cost: cheap ones first, slow ones in parallel with the main call or streamed, and budget both the latency each adds and the false positives they compound.", "A refusal is a product response: say what the user can do instead, log a reason code, and measure the false-refusal rate on benign traffic as carefully as the catch rate."]
question: "Design the guardrails for a customer-support chatbot at a bank: which checks run on input and output, in what order, and how do you keep the p95 time to first visible token under 1.2 seconds?"
answer: "Input: regex and checksum detection of card and account numbers plus an NER pass for names and addresses, replacing them with typed placeholders before the model and the logs see them; a small injection and jailbreak classifier; a scope check (banking support only) as a classifier or a small-model policy check. Output: a fast safety and PII-leak classifier on each streamed chunk, a rule that no account data appears unless it belongs to the authenticated user, and a groundedness check against retrieved policy documents for answers that state fees or rates. Ordering: redaction runs first because it changes what the model sees; the classifiers and the scope check run in parallel with the main call, and visible tokens are held until they pass, so the added latency is the slowest guard beyond TTFT rather than the sum. The groundedness judge is too slow to gate streaming, so run it on fee and rate answers before sending, or sample it offline. Measure catch rate and false-refusal rate per guard, and the combined block rate on benign traffic."
rubric: ["Separates input checks from output checks and names concrete ones relevant to a bank (PII, scope, injection, leakage, groundedness).", "Puts redaction first and explains why order matters.", "Runs slow checks in parallel with the main call or on streamed chunks, and computes the latency they add.", "Considers false positives: per-guard and compounded refusal rates on benign traffic.", "Treats refusals as a designed user experience with logging, not an error string.", "Does not rely on guardrails alone for security-critical actions (permissions and confirmations for anything that moves money)."]
sources: [{"title": "Inan et al.: Llama Guard (2023)", "url": "https://arxiv.org/abs/2312.06674", "note": "An LLM fine-tuned as an input and output safety classifier with a customizable taxonomy."}, {"title": "Rebedea et al.: NeMo Guardrails (2023)", "url": "https://arxiv.org/abs/2310.10501", "note": "Programmable rails for topic control, dialogue paths and moderation around any LLM."}, {"title": "Sharma et al.: Constitutional Classifiers (2025)", "url": "https://arxiv.org/abs/2501.18837", "note": "Input and output classifiers trained from a written constitution; reports the refusal and compute overhead."}, {"title": "Microsoft Presidio", "url": "https://microsoft.github.io/presidio/", "note": "Open-source PII detection (regex, checksums, NER) and anonymization."}]
practice: ["moderation-review-queue", "regex-patterns-for-nlp", "fraud-cost-based-decision", "untrusted-input-taint-check"]
tools: ["Llama Guard", "Prompt Guard", "ShieldGemma", "NeMo Guardrails", "Presidio", "OpenAI Moderation"]
---

## What a guardrail is

A guardrail is code outside the main model call that inspects what goes in or comes out and decides to pass it, block it, rewrite it or send it somewhere else. The model's own training (refusals, safety tuning) is one layer. The guardrail is the layer you control, version and measure. It is not a security boundary against a determined attacker. Stopping an injected instruction from doing damage is a permissions problem, covered in [Prompt injection](lesson:aieng:12.03). The design-interview view of guardrails is one paragraph in [LLM serving and RAG](lesson:mlsd:06.05). This lesson is about building them.

| Where | Check | Catches | Typical action |
| --- | --- | --- | --- |
| Input | PII detection | Card numbers, emails, names the model and logs should not see | Redact to placeholders |
| Input | Injection and jailbreak detection | "Ignore your instructions", role-play exploits | Block or flag |
| Input | Scope and policy | Off-topic, disallowed requests (legal advice from a support bot) | Polite refusal or handoff |
| Input | Abuse and cost | Huge inputs, rapid repeats, scripted traffic | Rate limit, truncate |
| Output | Safety classifier | Harmful content the model produced anyway | Block, regenerate |
| Output | Leakage | PII of other users, system prompt text, secrets | Block, redact |
| Output | Groundedness | Claims not supported by retrieved sources ([Evaluating RAG](lesson:aieng:10.04)) | Regenerate, add caveat, cite |
| Output | Format and business rules | Bad JSON, a refund above the limit ([Structured outputs](lesson:aieng:12.01)) | Repair, retry, reject |

## Rules, classifiers and LLM checks

Every guard is built from one of three kinds of checker, and the trade-off is latency and cost against flexibility.

| Checker | Latency (illustrative) | Good at | Weak at |
| --- | --- | --- | --- |
| Rules: regex, checksums, blocklists, length limits | under 1 ms | Exact formats: card numbers with a Luhn check, IBANs, API-key patterns | Anything paraphrased or misspelled |
| Small classifier: fine-tuned encoder or small guard model | 10–50 ms on a GPU | Fixed categories at scale: toxicity, injection attempts, topic | New categories need new training data |
| LLM check: a guard model or general model with a written policy | 200–1,000 ms | Nuanced, changeable policies ("no investment advice, but explaining what an ETF is is fine") | Cost, latency, and it can be prompted against |

Concrete options: hosted moderation endpoints (OpenAI's moderation API is free), open guard models such as Llama Guard (classifies prompts and responses against a taxonomy you can edit), Prompt Guard (a small injection and jailbreak classifier), ShieldGemma (Gemma-based safety classifiers), and policy-following models such as gpt-oss-safeguard that read your written policy at inference time. NeMo Guardrails wraps these into programmable "rails" around a dialogue. Training your own classifier is worth it when your policy is specific and your traffic is large. Anthropic's Constitutional Classifiers paper shows the price at the high end: an early version survived over 3,000 hours of red teaming without a universal jailbreak, and the improved classifiers added 23.7% inference overhead and 0.38 percentage points of extra refusals on production traffic.

## PII redaction

Redaction keeps personal data away from the model provider and, just as important, out of your logs and traces ([Observability](lesson:aieng:12.07)). The usual pipeline, which Presidio implements:

1. **Detect.** Regexes with checksums for structured identifiers (cards, IBANs, national IDs, phone numbers, emails), and NER for names, addresses and organizations.
2. **Replace with typed, consistent placeholders.** "Refund Maria Lopez, card 4111 1111 1111 1111" becomes "Refund `<PERSON_1>`, card `<CARD_1>`". Use the same placeholder for the same value across the conversation so the model can still reason about who is who.
3. **Keep the mapping server-side** and restore values only in what the authenticated user sees, never in tool calls to third parties unless policy allows it.

NER misses unusual names and flags harmless words (a customer called "Paris", a product called "Jordan"), so measure recall and precision on your own text. Over-redaction quietly breaks features: an address-change flow cannot work if every address becomes `<LOCATION_1>` before the tool that needs it.

## Topic and policy rules

Scope rules ("only answer questions about our products") can be an embedding-similarity check against allowed topics, a small classifier, or an LLM check given the written policy. Write the policy as a document with examples of allowed and disallowed requests. It becomes the guard's prompt, the labelling guide for its eval set, and the thing legal reviews. Rules that must always hold, such as "never quote a fee not in the fee table" or "refunds over a limit need an agent", belong in code or in tool permissions, not in a classifier that is right 98% of the time.

## Layering and the latency budget

Guards are cheap one at a time and expensive in a stack, in two ways: latency and false positives.

~~~dg-flow Guards layered by cost around a streaming chat call.
User message -(regex + NER, 3 ms)-> *Redacted input -(in parallel)-> Classifiers + scope check + main model -(hold until input guards pass)-> Streamed chunks -(chunk classifier, 20 ms)-> User -(async)-> Groundedness sample
~~~

<div class="ex">
<div class="lab">Worked example</div>

**Fitting guards into a 1,200 ms first-token budget**

A bank's support bot targets a p95 time to first visible token of 1,200 ms. Illustrative p95 latencies: redaction 3 ms, injection classifier 20 ms, moderation API 120 ms, scope check with a small LLM 450 ms, main model time to first token 900 ms, output chunk classifier 20 ms, groundedness judge 700 ms.

**All in series:** 3 + 20 + 120 + 450 + 900 + 20 = 1,513 ms, which is 313 ms over budget, before the judge.

**Layered:** redaction must run first, because it changes what the model sees. The three input guards then start at the same moment as the main call, and the first chunk is shown only when all of them have passed: 3 + max(20, 120, 450, 900) + 20 = 923 ms. The p95 of a maximum is a little worse than the maximum of the p95s, so plan for about 950 ms. The price is wasted model calls: if 2% of requests are blocked, 2% of prefill spend buys nothing.

**The judge** cannot gate a stream: holding the answer for it gives 923 + 700 = 1,623 ms. Run it before sending only on the answers that state fees or rates, and on a sample of the rest offline.

**False positives compound.** Four blocking guards with benign false-positive rates of 0.5%, 1%, 2% and 1.5%, assumed independent, pass a benign request with probability $0.995 \times 0.99 \times 0.98 \times 0.985 = 0.951$. About 4.9% of good requests are refused by something, ten times the best single guard.

</div>

Set a target for the combined false-refusal rate, measure it on a sample of real benign traffic, and give each guard a share of it. A guard that cannot meet its share flags for review instead of blocking.

## Refusal handling

A block needs a reply, and the reply is product design. Say what the bot can do ("I can't advise on investments, but I can explain the fees on this account"), offer the human path when one exists, and keep the wording consistent so users learn the boundary. Log a reason code per guard, so you can see which guard drives refusals and replay the cases. Model refusals arrive differently: some APIs return a refusal stop reason with output that does not match your schema, so handle it in the same place as guard blocks. Over-refusal is a real failure. XSTest is a public set of safe prompts that sound unsafe ("how do I kill a Python process?"). Build your own version from benign traffic and track the false-refusal rate next to the catch rate.

## Where this breaks

- **Guards are bypassable.** Paraphrases, other languages, base64, text in images and multi-turn escalation all lower catch rates. Treat a guard as a filter that reduces volume, and red-team it with adaptive attacks, not a fixed list.
- **Checking the wrong unit.** A per-message check misses a request split across five turns. Run conversation-level checks on a window, not just the last message.
- **Streaming leaks.** An output check that runs after the stream has finished means the user already read the text. Gate chunks or accept the risk deliberately.
- **Drift.** Guard accuracy changes when the product, the users or the main model changes. Guards need their own eval sets and regression tests, like any model.
- **Guard as permission.** Using a classifier to decide whether a payment tool may run is the wrong tool. Permissions and confirmations decide that.

## What interviewers probe

Expect questions on which guards you would add for a given product and in what order, how much latency each adds and how to hide it, how you would measure false refusals, where redaction happens relative to logging, why a classifier alone cannot stop prompt injection, and what you would do when a guard and the main model disagree.

<div class="take">
<div class="lab">Key takeaway</div>

Layer guards by cost: exact rules first, small classifiers in parallel with the main call, LLM checks only where nuance is worth hundreds of milliseconds. Budget the latency they add and the false refusals they compound as carefully as what they catch.

</div>
