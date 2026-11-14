---
id: "09.04"
course: "ai-engineering-course"
module: "09"
slug: "build-an-agent-plan-you-can-check"
title: "Build an agent plan you can check"
keyIdeas: ["A checkable plan states the goal, inputs, tools, steps, stop conditions, approvals and success criteria before any code.", "Tools: list each with its permission level (read vs write); keep writes behind confirmation.", "Stop conditions: success, max steps, budget, or a specific error that needs a human.", "Checkpoints: where a human reviews, especially before irreversible or external actions.", "Evaluation: a set of real past cases with expected outcomes; measure task success, steps, cost, and failure types from traces.", "Iterate from traces: read failed runs, classify the failure (context, tool, reasoning, budget), fix one thing, re-run the eval set."]
question: "Write a checkable plan for an agent that triages new GitHub issues: list its tools, stopping conditions, human checkpoints and how you'd evaluate it."
answer: "Goal: label type and priority, suggest an owner and flag duplicates for each new issue. Tools: read (get_issue, search_issues, list_labels, read CODEOWNERS) and low-risk writes (add_labels, post_comment, assign); no close permission. Stop: labels and comment done, or a max of ~15 tool calls/budget, or low confidence -> label needs-triage. Checkpoints: closing duplicates or anything external needs maintainer approval. Evaluation: replay ~100 past triaged issues, measure label accuracy, priority accuracy, duplicate precision, steps and cost per issue; read failed traces and classify as context, tool, reasoning or budget failures before changing anything."
sources: [{"title": "Anthropic: Building effective agents", "url": "https://www.anthropic.com/engineering/building-effective-agents", "note": "Simple, composable patterns and when to add autonomy."}, {"title": "Anthropic: Writing effective tools for agents", "url": "https://www.anthropic.com/engineering/writing-tools-for-agents", "note": "Designing and evaluating tools with the agent."}, {"title": "Hamel Husain: Your AI Product Needs Evals", "url": "https://hamel.dev/blog/posts/evals/", "note": "Building an evaluation loop for LLM products."}]
tags: ["Exercise"]
---

## Why plan first

Agents fail in ways that are hard to see: a wrong tool call 12 steps in, a loop that burns budget, an action that should have needed approval. Writing a plan you can check, before building, forces each of these decisions into the open. It also gives you the spec for your evaluation.

## The plan template

| Section | Question it answers |
| --- | --- |
| Goal | What does done look like, in one sentence? |
| Inputs | What does the agent receive, and from where? |
| Tools | Which tools, read or write, and with what permissions? |
| Steps | Expected path for a typical case (the agent may deviate) |
| Stop conditions | When does it finish, give up, or escalate? |
| Checkpoints | Which actions need a human first? |
| Context plan | What's in the system prompt, what's fetched just in time? |
| Evaluation | Which cases, which metrics, what pass rate? |

## Worked example: GitHub issue triage

<div class="ex">
<div class="lab">Worked example</div>

**triage new issues**

Goal: every new issue gets a type label, a priority, a likely owner and, if it's a duplicate, a link to the original, within 10 minutes.

Inputs: the new issue's title, body and author, via an MCP GitHub server.

Tools (read): get_issue, search_issues, list_labels, read_file (for CODEOWNERS). Tools (write): add_labels, post_comment, assign.

Stop: labels applied and comment posted; or 15 tool calls; or confidence low, in which case label needs-triage and stop.

Checkpoints: closing as duplicate requires a maintainer's approval; the agent only suggests it.

</div>

~~~dg-flow The triage agent's expected path.
New issue -(read)-> Issue text -(search similar)-> Possible duplicates -(classify)-> *Labels + priority -(CODEOWNERS)-> Owner -(write, low risk)-> Labels + comment
~~~

**Context plan**
System prompt: label definitions, priority rules, examples of three well-triaged issues. Fetched just in time: similar issues, CODEOWNERS. Tool results trimmed to titles and URLs.

**Evaluation**

- Take 100 past issues that maintainers already triaged; their labels are the expected output.
- Metrics: label accuracy, priority within one level, duplicate precision (false duplicates are expensive), median tool calls and cost per issue.
- Pass bar before launch, for example 90% type-label accuracy and zero wrong auto-closures (the agent can't close, by design).

## Iterating from traces

Hamel Husain's advice for any LLM product applies: look at your data. Read failed traces and sort each failure:

1. **Context**: it lacked information (missing label definitions).
2. **Tool**: wrong tool, bad arguments, or an unhelpful tool output.
3. **Reasoning**: right information, wrong conclusion.
4. **Budget**: ran out of steps.
Fix the most common category, re-run the eval set, and compare. Anthropic's tool-writing guide recommends the same loop for tools: let the agent use them on realistic tasks, read where it struggles, and rewrite descriptions and outputs.

## How many test cases is enough?

A plan you can check needs an eval, and an eval needs enough cases to tell real changes from noise. Set your pass rate and number of cases below.

~~~dg-raw
%%widget evalcalc Margin of error for an eval pass rate, and cases needed to detect a change.
~~~

Rules of thumb that fall out of the maths:

- 50 cases at an 80% pass rate gives roughly ±11 points of uncertainty. Good for catching big regressions, useless for comparing two similar prompts.
- 400 cases gives about ±4 points.
- To reliably detect a 5-point change between two variants at an 80% pass rate, you need around 1,000 cases per variant. Most teams don't have that, so they compare variants on the same cases (paired comparison) and read the cases that flipped.

## What goes in the eval set

1. **Real inputs**, sampled from logs or written by people who know the users. Synthetic cases are fine for coverage but under-represent how messy real input is.
2. **The hard cases first**: every production failure becomes a test case.
3. **Grading that matches the task**: exact match for labels, unit tests for code, a rubric scored by a judge model for open text. Check the judge against 50 or so human grades before trusting it.
4. **Trajectory checks for agents**: not just "was the final answer right" but "did it call the refund tool at most once" and "did it ask before deleting".

<div class="ex">
<div class="lab">Worked example</div>

**a triage agent's plan with checks attached**

Goal: label new GitHub issues with type and area, and ask for missing reproduction steps.

Steps and checks: (1) read the issue, check that it has a title and body; (2) classify type, checked against 300 human-labelled issues, target 90%; (3) classify area, same set, target 85%; (4) if bug and no reproduction steps, draft a comment, checked by a judge rubric on 50 cases; (5) apply labels, allowed only from a fixed list, enforced in code.

Each step has a number, so when the agent gets worse you know which step to fix.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

before building an agent, write down its goal, tools with permissions, stop conditions, checkpoints, context plan and evaluation set. Then improve it by reading traces and fixing the most common failure type.

</div>
