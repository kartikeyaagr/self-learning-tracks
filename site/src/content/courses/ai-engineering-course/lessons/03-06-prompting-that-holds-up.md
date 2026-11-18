---
id: "03.06"
course: "ai-engineering-course"
module: "03"
slug: "prompting-that-holds-up"
title: "Prompting that holds up"
keyIdeas: ["A prompt holds up when it passes an eval set, survives messy and hostile input, and keeps working when the model version changes.", "Put stable policy in the system prompt, data and the request in the user turn; long documents first, the question last; each kind of content in its own tagged section.", "Few-shot examples teach format and boundaries more than instructions do. Choose them to cover hard cases, balance the labels and vary the order, because models favour recent and frequent labels.", "Split work into a chain of prompts with checkable intermediate outputs when one prompt does too many things.", "Treat prompts as code: templates in version control, a version id on every trace, and an eval run before every change ships."]
question: "A support-ticket triage prompt passes 71% of a 200-ticket eval. Describe how you would find out why and rewrite it, and how you would know the rewrite is better."
answer: "Read the failures and group them first, rather than editing on a hunch: for example wrong category on billing-versus-account tickets, invented refund promises, malformed JSON, and following instructions pasted inside the ticket. Then fix each group with the matching tool: explicit category definitions with the boundary cases, a rule with its reason (refunds are decided by a human) plus an allowed-actions list, schema-constrained output for the JSON, and the ticket wrapped in a tagged block that the system prompt declares to be data, never instructions. Add three to five examples chosen from the confused pairs, balanced across labels. Keep the prompt in version control, re-run the same 200 tickets, and compare per failure group and per ticket (which flipped), not just the total. A move from 71% to 86% on the same tickets is well outside the noise; I would also check no slice got worse, then ship behind the eval in CI."
rubric: ["Starts from error analysis on the failing cases, grouped by type", "Matches each failure type to a specific technique (definitions, examples, structure, constrained output, data delimiting)", "Designs few-shot examples deliberately (coverage, label balance, order)", "Compares on the same eval cases and per slice, with some notion of noise", "Treats the prompt as versioned code tested in CI", "Notes prompting alone is not a security boundary for injection"]
sources: [{"title": "Anthropic: Prompting best practices", "url": "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices", "note": "Clarity, examples, XML structure, long-context ordering, chaining."}, {"title": "OpenAI: Prompt engineering guide", "url": "https://platform.openai.com/docs/guides/prompt-engineering", "note": "Message roles, instruction hierarchy, few-shot and reusable prompts."}, {"title": "Zhao et al.: Calibrate Before Use (2021)", "url": "https://arxiv.org/abs/2102.09690", "note": "Majority-label and recency bias in few-shot prompts, and a calibration fix."}, {"title": "Lu et al.: Fantastically Ordered Prompts (2021)", "url": "https://arxiv.org/abs/2104.08786", "note": "The order of the same few-shot examples swings accuracy from near chance to near state of the art."}]
practice: ["paired-bootstrap-regression-gate", "chat-template-format", "record-schema-validation", "untrusted-input-taint-check", "rag-context-packing"]
tools: ["Claude API", "OpenAI API"]
---

## What "holds up" means

A prompt that works on the five examples you tried is a draft. A prompt that holds up passes an eval set of real inputs, keeps passing when users paste in messy or hostile text, and survives a model upgrade with at most a re-run of the eval. Everything below is aimed at those three tests. [Context engineering](lesson:aieng:09.01) covers what to put in the window for agents; [Chain of thought](lesson:aieng:08.02) covers reasoning prompts. This lesson is about writing the instructions themselves.

## Roles and order

APIs separate a **system** (or developer) message from **user** turns. Models are trained to give system content higher priority, so it is the place for the stable part: role, policy, output format, tool guidance. The user turn carries the request and the data for this call. Keeping that split also keeps the system prompt identical across calls, which is what makes prompt caching work.

Order inside the context matters. Anthropic's guidance is to put long documents near the top and the question and instructions after them, and reports up to 30% better responses in its tests with the query at the end on multi-document inputs. The cause is positional: models use the start and end of a long context more reliably than the middle ("Lost in the Middle", cited in [Context engineering](lesson:aieng:09.01)). For very long inputs, restating the one critical constraint just before the question is cheap insurance.

## Structure beats prose

A prompt that mixes rules, background, examples and input in one paragraph forces the model to guess where each starts. Give each its own section, with XML-style tags or clear headings, and refer to them by name.

Before:

```text
You are a helpful support bot for Acme. Read the ticket and categorise it and
say how urgent it is and draft a reply, be polite and don't promise refunds.
Ticket: {ticket}
```

After:

```text
<role>You triage support tickets for Acme, a broadband provider.</role>

<categories>
billing: charges, invoices, payment methods. Includes "charged twice".
account: login, password, ownership changes. A locked account is account, not billing.
outage: no service or degraded speed at the customer's address.
other: anything else.
</categories>

<rules>
- Never promise or estimate a refund; refunds are decided by the billing team,
  and a promise we break costs more than the refund.
- Urgency is high only for outages affecting business lines or medical devices.
</rules>

<output>JSON matching the TriageResult schema. Draft reply under 120 words.</output>

<ticket>{ticket}</ticket>
```

The rewrite does four things that each fix a class of failure: it defines labels with their boundary cases, states rules positively with the reason behind them (models generalise from a reason, not from a bare "don't"), pins the output to a schema, and puts the variable input in a tagged block. Schema-constrained decoding is better than a format request where the API supports it ([Structured outputs](lesson:aieng:12.01)).

## Few-shot examples

Examples show format, tone and where category boundaries fall more precisely than instructions can. They also carry risks, because models copy their surface features.

- **Selection.** Spend examples on the confusable cases, not the easy ones. For varied inputs, retrieve the most similar examples per request from a labelled pool instead of fixing them.
- **Label balance.** Zhao et al. showed few-shot predictions lean toward labels that appear most often in the prompt and toward the label of the last example. Three billing examples and one outage example teach "billing" as the default.
- **Order.** Lu et al. found that the order of the same examples moved accuracy from near chance to near state of the art on some tasks, and that a good order for one model did not transfer to another. Shuffle the order on your eval set and check the spread before trusting one arrangement.
- **Consistency.** Every example should match the output format exactly; one sloppy example licenses sloppy outputs.
- **Quantity.** A handful (Anthropic suggests 3 to 5) is usually enough. Reasoning models often do better with fewer examples and a clearer statement of the goal.

## Decomposition and prompt chaining

When one prompt classifies, extracts, decides and writes, a failure anywhere looks the same from outside. Split it into a chain where each step has an output you can check in code or with a small eval.

~~~dg-flow A triage chain: each arrow carries something you can test on its own.
Ticket -(prompt 1)-> Extracted fields -(validate schema)-> *Category + urgency -(prompt 2, with policy chunks)-> Draft reply -(prompt 3 or rules)-> Policy check -> Agent review
~~~

Chaining costs latency and more calls, and errors still propagate, so use it when steps are genuinely different jobs or need different context. Steps that do not depend on each other can run in parallel.

## Prompts are code

- Keep prompts in version control as templates with named variables, not as strings built inline in application code.
- Log the prompt id, version and model version on every trace, so a quality change can be tied to a commit ([Observability for LLM apps](lesson:aieng:12.07)).
- Pin the model version together with the prompt. A prompt tuned for one model is a hypothesis for the next.
- Run the eval set in CI on every prompt change, with a regression gate ([Regression testing prompts and agents in CI](lesson:aieng:10.06)).
- Unit-test the rendering: missing variables, very long inputs, inputs containing your delimiter tags.

## Writing for injection resistance

Any text that came from a user, a web page, an email or a tool result can contain instructions. Prompting reduces the damage:

- Wrap untrusted content in tagged blocks and state in the system prompt that content inside them is data to analyse, never instructions to follow.
- Escape or strip your delimiter tags from untrusted text so it cannot close the block early.
- Restate what the task is after the untrusted block.

None of this is a security boundary. A determined injection will sometimes win against any wording, so anything with real consequences (sending email, issuing refunds, reading private data) must be limited by permissions and confirmations outside the model. [Prompt injection](lesson:aieng:12.03) covers those structural defences.

<div class="ex">
<div class="lab">Worked example</div>

**rewriting the triage prompt (illustrative eval numbers)**

The one-paragraph prompt above passes 142 of 200 labelled tickets (71%). Reading the 58 failures gives four groups:

| Failure group | Before | Fix | After |
| --- | --- | --- | --- |
| Billing vs account confusion | 24 | Category definitions with boundary cases, 4 balanced examples | 9 |
| Promised or estimated a refund | 13 | Rule with its reason, allowed-actions list | 2 |
| Output not valid JSON | 11 | Schema-constrained output | 0 |
| Followed instructions inside the ticket | 10 | Tagged ticket block declared as data, task restated after it | 4 |
| Other (new failures included) | 0 | | 13 |
| **Total failing** | **58** | | **28** |

Passing goes from 142 to 172 (86%). On 200 cases the standard error of each pass rate is about $\sqrt{0.71 \times 0.29 / 200} \approx 0.032$, roughly ±6 points at 95%, so a 15-point gain measured on the same tickets is real. The "other" row matters: 13 tickets that used to pass now fail, mostly long tickets where the reply ran past 120 words. That is the next fix, and it would have been invisible in the total alone.

</div>

## Common mistakes

- **Editing on anecdotes.** Fixing the one ticket a manager forwarded often breaks five others. Change the prompt only against the eval set.
- **Shouting.** Capitals and "CRITICAL" make current models over-apply a rule. State it once, with its reason.
- **Negative-only instructions.** "Don't be verbose" is weaker than "Reply in at most three sentences".
- **Unbalanced or unrealistic examples.** They shift the label distribution and teach the wrong length and style.
- **Changing the model without re-running the eval.** Prompts written for older models are often over-specified for newer ones and can make them worse.
- **Relying on the prompt for security.** See above.

## What interviewers probe

- Take a weak prompt and improve it live, explaining which failure each change targets.
- How you choose and order few-shot examples, and what biases they introduce.
- When to split one prompt into a chain, and what it costs.
- How you version, test and roll out prompt changes.
- How you would make a prompt that processes emails harder to hijack, and why that is not enough.

<div class="take">
<div class="lab">Key takeaway</div>

a prompt holds up when every change is driven by grouped failures on an eval set: structure the prompt into tagged sections with the question last, choose few-shot examples for coverage and balance, chain steps you can check, version prompts like code, and never rely on wording alone to stop injection.

</div>
