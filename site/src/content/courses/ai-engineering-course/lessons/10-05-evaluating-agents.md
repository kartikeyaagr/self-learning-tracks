---
id: "10.05"
course: "ai-engineering-course"
module: "10"
slug: "evaluating-agents"
title: "Evaluating agents"
keyIdeas: ["Grade the outcome (the state of the environment after the run), and use trajectory checks as assertions for policy, ordering and budget, not as an exact path to match.", "Tool-call accuracy breaks into selection, argument validity, argument correctness and redundant calls; compare calls structurally, not as strings.", "Run several trials per task. pass@k (at least one of k succeeds) suits tasks with a verifier you can retry against; pass^k (all k succeed) suits user-facing tasks where every run must work.", "Separate harness failures (sandbox crash, rate limit) from agent failures, rerun the former, and treat flaky tasks as the reliability backlog.", "Compare agents on cost per solved task, including what a failure costs downstream, not on success rate or price per run alone."]
question: "A customer-support agent passes 70% of trials on your 120-task eval. Product wants to know whether it is ready to handle refunds unsupervised. What would you measure beyond that 70%, and how would you set up the eval so the numbers can be trusted?"
answer: "Run several trials per task (five, say) in a sandbox reset to a fresh database snapshot each time, with external APIs mocked and the simulated user's model pinned. Grade each trial on the end state (was the right refund issued, once, for the right amount), plus trajectory assertions: identity verified before any refund, no forbidden tools, within a step and cost budget. Report pass^k as well as the per-trial rate, because a customer gets one run: 70% per trial can mean pass^5 well under 30% if successes are spread across tasks. Bucket tasks into always-pass, flaky and always-fail, and read transcripts of the flaky ones. Exclude and rerun harness errors rather than counting them as failures. Audit a sample of failures for simulator mistakes. Finally, compute cost per resolved ticket including the cost of a wrong refund or a human handoff, and gate on zero policy violations, not only on the average."
rubric: ["Grades end state in the environment rather than the agent's final message", "Adds trajectory assertions for policy and ordering (verify before refund) and budget", "Runs multiple trials and reports a consistency metric such as pass^k", "Isolates trials with fresh sandbox state and mocked external services", "Distinguishes harness or simulator failures from agent failures", "Reports cost per solved task including failure cost, or gates on zero critical violations"]
sources: [{"title": "Yao et al.: tau-bench, Tool-Agent-User Interaction in Real-World Domains (2024)", "url": "https://arxiv.org/abs/2406.12045", "note": "Simulated users, end-state database grading and the pass^k metric."}, {"title": "Anthropic: Demystifying evals for AI agents (2026)", "url": "https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents", "note": "Tasks, trials, graders, outcomes versus transcripts, isolation and saturation."}, {"title": "Jimenez et al.: SWE-bench (2023)", "url": "https://arxiv.org/abs/2310.06770", "note": "Coding-agent tasks graded by running the repository's tests."}, {"title": "Chen et al.: Evaluating Large Language Models Trained on Code (2021)", "url": "https://arxiv.org/abs/2107.03374", "note": "The unbiased pass@k estimator from n samples."}]
practice: ["pass-at-k-and-pass-hat-k", "evaluate-agent-trajectory", "validate-tool-arguments", "pass-at-k-estimator", "react-agent-capstone"]
tools: ["tau-bench", "SWE-bench"]
tags: ["agent evaluation", "pass^k"]
---

## What changes when the system acts

A single model call produces text, and [graders](lesson:aieng:10.03) score text. An agent ([the agent loop](lesson:aieng:09.02)) takes a dozen steps, calls tools that change state, and may talk to a user along the way. Three things follow. Many different paths can be correct. Errors compound, so a run that is 95% reliable per step fails often over 20 steps. And the same task gives different results on different runs, so one trial per task tells you little.

The vocabulary Anthropic uses in its agent-eval guide is worth adopting: a **task** has inputs and success criteria, each attempt is a **trial**, a **grader** scores one aspect of a trial, the **transcript** is everything the agent did, and the **outcome** is the state of the environment at the end.

## Grade the outcome, assert on the trajectory

The primary grade should read the environment, not the agent's last message. An agent that says "your refund is on its way" has passed only if the refunds table holds one refund, for the right order and amount. τ-bench grades this way: it compares the database at the end of the conversation with an annotated goal state. SWE-bench grades a coding agent by running the repository's tests, including tests the agent never saw.

The transcript is still graded, but as a set of assertions, not as a reference path to match. Exact-path matching fails good agents that took a valid detour.

| Check | Example | Grader |
| --- | --- | --- |
| Outcome | refund row exists, amount correct, exactly one | code, on environment state |
| Policy | identity verified before any refund call | code, on the ordered tool calls |
| Forbidden action | never calls `delete_account` | code |
| Budget | at most 25 tool calls and 200k tokens | code |
| Communication | told the user the refund timeline | rubric judge on the final turns |

Hard policy checks are gates: one violation fails the trial whatever the outcome was.

## Tool-call accuracy

When a step has a known right call (single-turn function-calling sets, or the "required" calls of a multi-step task), score calls on four separate axes, because each has a different fix:

- **selection**: right tool chosen (fix tool descriptions and tool count);
- **validity**: arguments parse and match the schema (fix with strict schema mode; see [tool calling](lesson:aieng:08.03));
- **correctness**: argument values match after normalisation, such as dates, case and units (fix context or examples);
- **redundancy**: repeated identical calls and calls whose result is never used (fix loop control).

Compare calls structurally: parse name and arguments and compare fields, as the Berkeley Function Calling Leaderboard does, rather than comparing JSON strings, where key order and whitespace create false failures.

## Trials: pass@k and pass^k

Run $n$ trials per task and count $c$ successes. Two different questions can be asked of the same counts.

**pass@k**: if you sample $k$ runs, is at least one correct? Use it when you can check and retry: a coding agent whose patch must pass tests can try $k$ times and keep the one that passes. The unbiased estimator from Chen et al. is

$$
\text{pass@}k = 1 - \binom{n-c}{k} \Big/ \binom{n}{k}
$$

**pass^k**: are all $k$ independent runs correct? Use it when a user gets one run and there is no verifier to retry against, as in customer support. τ-bench introduced it; the unbiased estimator is

$$
\text{pass}^k = \binom{c}{k} \Big/ \binom{n}{k}
$$

Both equal the plain success rate $c/n$ at $k = 1$. Plugging the rate into $1-(1-p)^k$ or $p^k$ is biased when $n$ is small; the binomial forms are not. Average either over tasks.

<div class="ex">
<div class="lab">Worked example</div>

**three tasks, five trials each**

| Task | Successes $c$ of 5 | pass@1 | pass@2 | pass^2 | pass^3 |
| --- | --- | --- | --- | --- | --- |
| look up order | 5 | 1.0 | 1.0 | 1.0 | 1.0 |
| change address | 3 | 0.6 | $1 - \tfrac{1}{10} = 0.9$ | $\tfrac{3}{10} = 0.3$ | $\tfrac{1}{10} = 0.1$ |
| partial refund | 1 | 0.2 | $1 - \tfrac{6}{10} = 0.4$ | 0 | 0 |
| **mean** | | **0.60** | **0.77** | **0.43** | **0.37** |

The same agent is 77% at pass@2 and 43% at pass^2. If the address task were a coding task with tests, two tries would almost always get there. As a customer conversation, it fails seven pairs of runs out of ten. τ-bench reported this gap on real agents: gpt-4o solved under half the retail tasks and its pass^8 fell below 25%.

</div>

## Handling flaky runs

Not every failed trial is the agent's fault. Before computing any of the numbers above, sort failures:

1. **Harness failures**: the sandbox did not start, a mocked service crashed, the provider returned a rate-limit error, a tool timed out because the eval machine was overloaded. Mark the trial as errored, exclude it from $n$, rerun it, and track the error rate. A task that errors often is an environment bug to fix, not a hard task.
2. **Simulator failures** (next section): the simulated user contradicted its own scenario.
3. **Agent failures**: everything else. Only these count.

Then bucket tasks by their trial results: always pass, flaky ($0 < c < n$), always fail. Always-fail tasks are capability gaps or broken tasks (read one transcript and check the task is solvable). Flaky tasks are where reliability work pays off, and reading a passing and a failing transcript of the same task side by side is usually the fastest way to find the cause. Trials of one task are correlated, so error bars must resample tasks, not trials; [Regression testing in CI](lesson:aieng:10.06) shows how.

## Simulated users

Conversational agents need someone to talk to. A second LLM plays the user from a scenario: a persona, a hidden goal ("change the address on order 4417 but only if it has not shipped"), facts to reveal only when asked, and a rule to end the conversation. τ-bench uses exactly this setup.

Simulators introduce their own errors. They forget their goal, reveal everything in the first message (making the task too easy), or get stuck in politeness loops. Pin the simulator model and its prompt as part of the eval version, sample a share of failed transcripts and label how many were the simulator's fault, and script the turns that matter (the exact wording of a tricky request) instead of leaving them to the simulator.

## Sandboxed environments

~~~dg-seq One isolated trial.
Harness > Sandbox: restore DB snapshot, frozen clock, mocks on
Harness > Agent: task + tools
Agent > Sandbox: tool calls and results, many rounds
Agent > Simulated user: questions and replies
Harness > Grader: final sandbox state + transcript
Grader > Harness: outcome pass or fail, assertion results, cost
~~~

Each trial starts from a clean state: a fresh container or database snapshot, external APIs replaced by mocks that return recorded responses, a fixed clock, and no network beyond what the task allows. Shared state between trials is a quiet source of both false failures and false passes. Anthropic describes an agent scoring better on some tasks by reading the git history that earlier trials had left behind. Coding and computer-use agents need the strongest isolation ([coding and computer-use agents](lesson:aieng:09.08)).

## Cost per solved task

A cheaper agent that fails more is not automatically cheaper. With per-trial success $p$ and cost $r$, if failures can be detected and retried, the cost per solved task is $r/p$. If failures go to a human, add the expected cost of that handoff.

<div class="ex">
<div class="lab">Worked example</div>

**two configurations, illustrative prices**

| | Success per trial | Cost per trial | Cost per success $r/p$ | With a \$4 human handoff per failure |
| --- | --- | --- | --- | --- |
| A, small model | 0.62 | \$0.12 | \$0.19 | \$0.12 + 0.38 × \$4 = \$1.64 |
| B, large model | 0.78 | \$0.45 | \$0.58 | \$0.45 + 0.22 × \$4 = \$1.33 |

If retries are free and failures are detectable, A is three times cheaper per success. If each failure costs a person's time, B is cheaper per ticket. Put both numbers in the eval report, along with p50 and p95 run time.

</div>

## Where this breaks

- **Grader bugs look like agent failures.** A checker that expects `"2026-03-01"` fails an agent that wrote `"March 1, 2026"` correctly. Read transcripts of failures and of passes; both kinds of grader error exist.
- **Saturation.** A suite at 98% catches regressions but no longer shows improvement. Anthropic's guide notes frontier models going from about 30% to over 80% on SWE-bench Verified. Keep adding harder tasks from production failures.
- **The sandbox is not production.** Real tools are slower, return messier errors and rate-limit. Replay a sample of real traces against the mocks to check they behave alike.
- **Public benchmarks leak.** Tasks built from public repositories may be in training data. Use your own tasks for decisions and public ones for orientation.

## What interviewers probe

- "pass@k or pass^k for this product?" Depends on whether a failure can be detected and retried before a user sees it.
- "Your agent's score dropped 5 points overnight with no code change. What do you check?" Harness error rate, provider-side model change, simulator drift, then flaky tasks.
- "How do you grade an agent whose path varies?" End-state checks plus trajectory assertions, never exact-path match.

<div class="take">
<div class="lab">Key takeaway</div>

Grade agents on the state they leave behind, with trajectory assertions for policy and budget. Run several isolated trials per task, report pass^k when every run must work and pass@k when you can verify and retry, discard harness errors before scoring, and decide between configurations on cost per solved task.

</div>
