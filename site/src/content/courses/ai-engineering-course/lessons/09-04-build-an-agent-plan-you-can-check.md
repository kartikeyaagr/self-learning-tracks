---
id: "09.04"
course: "ai-engineering-course"
module: "09"
slug: "build-an-agent-plan-you-can-check"
title: "Build an agent plan you can check"
keyIdeas: ["A checkable plan states the goal, inputs, tools, steps, stop conditions, approvals, context plan and success criteria before any code.", "Tools: list each with its permission level (read vs write); keep writes behind confirmation, and leave out permissions the agent does not need.", "Stop conditions: success, max steps, budget, or a specific condition that needs a human.", "Every step gets a check with a number attached, so when the agent gets worse you know which step to fix.", "Iterate from traces: read failed runs, classify the failure (context, tool, reasoning, budget), fix the most common one, re-run the eval set. Size the set so a change is bigger than the noise."]
question: "Write a checkable plan for an agent that triages new GitHub issues: list its tools, stopping conditions, human checkpoints and how you'd evaluate it."
answer: "Goal: label type and priority, suggest an owner and flag duplicates for each new issue. Tools: read (get_issue, search_issues, list_labels, read CODEOWNERS) and low-risk writes (add_labels, post_comment, assign); no close permission. Stop: labels and comment done, or a cap of about 15 tool calls or a token budget, or low confidence, in which case label needs-triage and stop. Checkpoints: closing duplicates or anything external needs maintainer approval. Evaluation: replay about 100 past triaged issues, measure label accuracy, priority accuracy, duplicate precision, steps and cost per issue, and size the set from the margin of error you need. Read failed traces and classify them as context, tool, reasoning or budget failures before changing anything."
rubric: ["States a one-sentence goal with a concrete done condition.", "Lists tools split into read and write, with write permissions limited and destructive actions excluded or gated.", "Gives stop conditions: success, step or budget cap, and a low-confidence escalation.", "Names human checkpoints for irreversible or external actions.", "Evaluates on real past cases with per-step metrics, enough cases to beat noise, and a trace-reading loop."]
sources: [{"title": "Anthropic: Building effective agents", "url": "https://www.anthropic.com/engineering/building-effective-agents", "note": "Simple, composable patterns and when to add autonomy."}, {"title": "Anthropic: Writing effective tools for agents", "url": "https://www.anthropic.com/engineering/writing-tools-for-agents", "note": "Designing and evaluating tools with the agent."}, {"title": "Hamel Husain: Your AI Product Needs Evals", "url": "https://hamel.dev/blog/posts/evals/", "note": "Building an evaluation loop for LLM products."}]
practice: ["evaluate-agent-trajectory", "authorize-tool-call", "bootstrap-confidence-interval", "react-agent-capstone"]
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

The loop mechanics behind stop conditions and checkpoints (step caps, persisted approvals) are in [The agent loop](lesson:aieng:09.02); the context plan follows [Context engineering](lesson:aieng:09.01).

## Worked example: GitHub issue triage

<div class="ex">
<div class="lab">Worked example</div>

**triage new issues, with a check on every step**

Goal: every new issue gets a type label, an area label, a priority, a likely owner and, if it's a duplicate, a link to the original, within 10 minutes. Bugs without reproduction steps get a comment asking for them.

Inputs: the new issue's title, body and author, via an MCP GitHub server.

Tools (read): get_issue, search_issues, list_labels, read_file (for CODEOWNERS). Tools (write): add_labels, post_comment, assign. No close or delete permission, enforced by the token's scopes, not the prompt.

Stop: labels applied and comment posted; or 15 tool calls; or confidence low, in which case label needs-triage and stop.

Checkpoints: closing as duplicate requires a maintainer's approval; the agent only suggests it.

Steps and their checks:

1. Read the issue. Check: title and body are present, else label needs-info.
2. Classify type. Checked against 300 human-labelled issues, target 90%.
3. Classify area. Same set, target 85%.
4. Search for duplicates. Duplicate precision on the same set, target 95%, because a false duplicate annoys a real user.
5. If a bug has no reproduction steps, draft a comment. Scored by a judge rubric on 50 cases, judge checked against human grades first.
6. Apply labels. Allowed only from a fixed list, enforced in code.

Each step has a number, so when the agent gets worse you know which step to fix.

</div>

~~~dg-flow The triage agent's expected path.
New issue -(read)-> Issue text -(search similar)-> Possible duplicates -(classify)-> *Labels + priority -(CODEOWNERS)-> Owner -(write, low risk)-> Labels + comment
~~~

**Context plan.** System prompt: label definitions, priority rules, three well-triaged example issues. Fetched just in time: similar issues, CODEOWNERS. Tool results trimmed to titles and URLs.

**Evaluation.** Replay past issues that maintainers already triaged; their labels are the expected output. Measure the per-step numbers above plus median tool calls and cost per issue. Pass bar before launch: for example 90% type-label accuracy and zero wrong auto-closures (the agent can't close, by design). Grade the trajectory as well as the outcome: "searched before labelling as duplicate", "never called assign twice". How to grade agent runs is in [Evaluating agents](lesson:aieng:10.05).

## Iterating from traces

Hamel Husain's advice for any LLM product applies: look at your data. Read failed traces and sort each failure:

1. **Context**: it lacked information (missing label definitions).
2. **Tool**: wrong tool, bad arguments, or an unhelpful tool output.
3. **Reasoning**: right information, wrong conclusion.
4. **Budget**: ran out of steps.

Fix the most common category, re-run the eval set, and compare. Anthropic's tool-writing guide recommends the same loop for tools: let the agent use them on realistic tasks, read where it struggles, and rewrite descriptions and outputs.

## How many test cases is enough?

A check is only useful if it can tell a real change from noise. This is a preview; building the set, slicing it and labelling it are in [Building an eval set](lesson:aieng:10.02).

~~~dg-raw
%%widget evalcalc Margin of error for an eval pass rate, and cases needed to detect a change.
~~~

For a pass rate $\hat{p}$ on $n$ cases, the 95% margin of error is about $1.96\sqrt{\hat{p}(1-\hat{p})/n}$:

- 50 cases at 80%: $1.96\sqrt{0.16/50} = 0.11$, so roughly ±11 points. Good for catching big regressions, useless for comparing two similar prompts.
- 400 cases at 80%: about ±4 points.
- To reliably detect a 5-point change between two variants near 80%, you need around 900 to 1,000 cases per variant. Most teams don't have that, so they compare variants on the same cases (a paired comparison) and read the cases that flipped.

## Common mistakes

- **No stop condition except "done".** The agent loops until the budget, or the bill, notices.
- **Permissions in the prompt.** "Never close issues" in the system prompt is a request; a token without the scope is a guarantee.
- **One end-to-end score.** "Triage is 78% right" doesn't say which step broke. Keep the per-step numbers.
- **Synthetic cases only.** They are fine for coverage but miss how messy real issues are. Every production failure becomes a test case.

<div class="take">
<div class="lab">Key takeaway</div>

Before building an agent, write down its goal, tools with permissions, stop conditions, checkpoints, context plan and a check for every step. Then improve it by reading traces and fixing the most common failure type, on an eval set big enough to see the change.

</div>
