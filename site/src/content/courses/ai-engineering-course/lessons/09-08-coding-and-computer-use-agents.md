---
id: "09.08"
course: "ai-engineering-course"
module: "09"
slug: "coding-and-computer-use-agents"
title: "Coding and computer-use agents"
keyIdeas: ["Coding agents work because the environment gives real feedback: compilers, linters and tests tell the loop when a step failed.", "Run them in a sandbox with both filesystem and network isolation, no production credentials, resource and time limits, and a branch or snapshot to roll back.", "The verify loop is reproduce, edit, run targeted tests, read failures, repeat, then the full suite; guard against the agent editing tests to pass.", "SWE-bench-style evals apply hidden fail-to-pass and pass-to-pass tests to the agent's patch; report cost per resolved issue alongside the resolve rate.", "Computer-use agents act on screenshots, so they fail on grounding, timing, pop-ups, long horizons and injected page content; bound them with VMs, allowlists, step caps and confirmations."]
question: "You are asked to let a coding agent fix failing CI jobs on a company monorepo without a human watching each step. Describe the environment, the loop and the guards you would put in place, and how you would know it works."
answer: "Run each job in a fresh container or microVM with only the repository checked out on a new branch, no production secrets, CPU, memory and wall-clock limits, and network egress limited to the package mirror. Give the agent read, search, edit-by-diff and run-command tools. The loop: reproduce the failure locally, find the cause, make a minimal edit, run the failing test and its neighbours, read the output, iterate, then run the affected suite and linters. Guards: test and CI config files are read-only or any change to them blocks the run; a step and token budget; a final diff check (size, files touched); the agent opens a pull request and never merges. Evaluate on a replay set of past CI failures with the real fix's tests held out: resolve rate, pass rate across repeated runs, regressions introduced, human review acceptance and cost per resolved job."
rubric: ["Specifies an isolated sandbox with filesystem and network limits and no production credentials", "Describes a verify loop grounded in running tests, starting from reproduction", "Guards against reward hacking such as editing or deleting tests", "Keeps irreversible steps (merge, deploy) behind human review", "Proposes an offline eval on historical failures with hidden tests, repeated runs and cost per resolved task"]
sources: [{"title": "Jimenez et al.: SWE-bench (2023)", "url": "https://arxiv.org/abs/2310.06770", "note": "2,294 real GitHub issues from 12 Python repositories, graded by the repository's tests."}, {"title": "Yang et al.: SWE-agent (2024)", "url": "https://arxiv.org/abs/2405.15793", "note": "The agent-computer interface: tool design changes coding-agent results."}, {"title": "Xie et al.: OSWorld (2024)", "url": "https://arxiv.org/abs/2404.07972", "note": "369 real desktop tasks with execution-based checks; GUI grounding as the main failure."}, {"title": "Anthropic: Claude Code sandboxing", "url": "https://www.anthropic.com/engineering/claude-code-sandboxing", "note": "Filesystem plus network isolation; 84% fewer permission prompts in internal use."}]
practice: ["react-agent-capstone", "evaluate-agent-trajectory", "pass-at-k-estimator", "authorize-tool-call"]
tools: ["SWE-bench", "Claude Code", "OSWorld"]
tags: ["Agents"]
---

## Why coding is where agents work best

The agent loop of [09.02](lesson:aieng:09.02) compounds errors unless each step is checked. Software gives you checks for free: the code compiles or it does not, the linter passes or not, the test goes from red to green. That turns a long chain of guesses into a loop with ground truth at every step, and it is the same property that makes coding a good target for reinforcement learning with verifiable rewards ([08.01](lesson:aieng:08.01)). Computer-use agents, which drive a screen like a person, lack most of that feedback, which is why they are much less reliable today.

## The coding agent's interface

SWE-agent (Yang et al., 2024) made the point that the **agent-computer interface** matters as much as the model: purpose-built commands for viewing a file window, searching and editing, with concise output and errors the model can act on, beat a raw shell. Modern coding agents converge on a small tool set:

- **Read and search**: open a file range, grep, list files. Keep outputs short; return paths and line numbers.
- **Edit by diff**: replace an exact string or apply a patch, rather than rewriting whole files. Diffs are cheaper, reviewable and fail loudly when the target text is missing.
- **Run commands**: tests, builds, linters, with output truncated to the useful tail.
- **Project instructions**: a checked-in file (often `AGENTS.md` or `CLAUDE.md`) with build commands, conventions and no-go areas, loaded into context every run.

## The sandbox

A coding agent executes code it wrote, in a repository full of text it did not write. Treat every run as untrusted code execution.

~~~dg-layers A coding agent's sandbox, outermost layer first.
Isolation boundary | fresh container, gVisor or microVM per task; destroyed afterwards
Network | deny by default; allowlist the package mirror and the git host
*Filesystem | only the checked-out repo on a new branch; no home directory, keys or cloud credentials
Resources | CPU, memory, disk and wall-clock limits; kill runaway processes
Actions | open a pull request at most; merge and deploy stay with humans
~~~

Anthropic's write-up of Claude Code's sandbox explains why you need both of the middle layers: without network isolation a compromised agent can exfiltrate SSH keys, and without filesystem isolation it can escape and gain network access. Inside a well-bounded sandbox the agent can run without asking permission at each step; Anthropic reports sandboxing cut permission prompts by 84% in internal use. That is the practical trade: tighter walls buy more autonomy.

## The verify loop

~~~dg-flow The loop a reliable coding agent runs.
Issue -(failing test)-> Reproduced -(minimal edit)-> *Patch -(targeted tests)-> Test output -(fail: read, fix)-> Patch
Test output -(pass)-> Full suite + lint -(green)-> Diff review -(open PR)-> Human review
~~~

1. **Reproduce first.** A failing test that captures the bug is the agent's reward signal; without it, "fixed" means "looks fixed".
2. **Run the narrowest test after each edit**, then widen. Full suites are slow and their output floods context.
3. **Feed back real output**, truncated to the failure and the stack trace.
4. **Finish on evidence**: the final message cites the test run that passed.

**Reward hacking** is the main failure of test-driven agents: editing the test instead of the code, special-casing the test input, deleting assertions, catching and ignoring the exception, or marking the test as skipped. Guards that work: make test and CI configuration files read-only or fail the run if they change; keep some acceptance tests hidden from the agent; check the diff for skips and deleted assertions; and have a reviewer read the change, since a passing suite is necessary but not sufficient.

<div class="ex">
<div class="lab">Worked example</div>

**Cost per resolved issue.** An agent run averages 40 steps, each resending about 50k tokens of context and generating 800 output tokens. Illustrative prices: \$3 per million input, \$0.30 per million cached input, \$15 per million output.

- Input: 40 × 50k = 2M tokens; output 40 × 800 = 32k tokens.
- With 90% of input read from cache: 0.2M × \$3 + 1.8M × \$0.30 + 0.032M × \$15 = \$0.60 + \$0.54 + \$0.48 = **\$1.62 per attempt**.
- Without caching: 2M × \$3 + \$0.48 = **\$6.48 per attempt**.
- At a 60% resolve rate, the cost per resolved issue is \$1.62 / 0.6 = **\$2.70** with caching and \$10.80 without.

Prompt caching ([12.04](lesson:aieng:12.04)) is the biggest single lever, and an append-only history is what keeps the cache hitting. The next lever is fewer, smaller steps: shorter tool outputs and earlier stopping on hopeless runs.

</div>

## Evaluating coding agents

SWE-bench (Jimenez et al., 2023) set the template: 2,294 real issues from 12 Python repositories; the agent gets the repo and the issue text and must produce a patch. Grading runs two hidden test sets from the real fix: **fail-to-pass** tests that must now pass, and **pass-to-pass** tests that must keep passing. At release the best model resolved 1.96%. SWE-bench Verified is a 500-task subset that humans checked for solvable, well-specified issues and fair tests; it became the standard headline number.

Read benchmark numbers with care. Public issues can leak into training data; scaffolds (tools, retries, test-time sampling) differ between submissions; and a single-run resolve rate hides variance. For your own agent, build the eval from your history: past bugs with their fix's tests held out, run each task several times, and report resolve rate, consistency across runs (pass^k from [10.05](lesson:aieng:10.05)), regressions introduced, review acceptance and cost per resolved task.

## Computer-use agents

A computer-use agent sees screenshots and emits actions: click at (x, y), type, scroll, press keys, wait. It can operate any application, which is its appeal and its problem: there is no compiler, only pixels.

OSWorld (Xie et al., 2024) measured this on 369 real desktop tasks with scripted checks: humans completed 72.36% and the best model of the time 12.24%, failing mainly on **GUI grounding** (finding the right element) and operational knowledge of the apps. Models have improved a lot since, but the failure modes are stable:

- **Grounding errors**: clicking a pixel next to the button, or the right label in the wrong row.
- **Timing**: acting on a screenshot taken before the page finished loading.
- **Interruptions**: cookie banners, modals, login expiry, layout changes.
- **Long horizons**: 30–100 actions, each with a small error rate, compound as in [09.02](lesson:aieng:09.02).
- **Injected content**: a web page or image containing instructions. Anthropic runs classifiers over screenshots for this, and still recommends a VM, no sensitive data, a domain allowlist and human confirmation for consequential actions.
- **Things it must not do alone**: purchases, sending messages, accepting terms, solving CAPTCHAs.

Screenshots are also expensive. On Anthropic's current models an image costs about $\lceil w/28 \rceil \times \lceil h/28 \rceil$ visual tokens, so a 1280×800 screen is 46 × 29 = 1,334 tokens. If every screenshot stays in context, step $t$ carries $t$ of them: over 30 steps that is $1{,}334 \times (1 + 2 + \dots + 30) \approx 620$k input tokens of screenshots alone. Keeping only the last three cuts it to about 116k. Clear old screenshots ([09.01](lesson:aieng:09.01)) and prefer structured access (an API, the DOM, the accessibility tree) whenever the target exposes one; use pixels only where nothing else exists.

<div class="take">
<div class="lab">Key takeaway</div>

Coding agents are reliable to the degree their loop is grounded in tests and bounded by a sandbox. Reproduce, edit, test, repeat; lock the tests; isolate files and network; keep merges human; and measure cost per resolved task. Computer use lacks that feedback, so bound it harder: VMs, allowlists, step caps, confirmations and structured access where possible.

</div>

## Where this breaks

- **Green tests, wrong fix**: weak tests reward a patch that silences the symptom.
- **Flaky tests** teach the loop that failures are random, and it stops reading them.
- **Context floods**: a 5,000-line test log or a whole-file read crowds out the issue.
- **Secrets in the sandbox**: a `.env` file in the checkout is reachable by injected instructions in a dependency's README.
- **Unbounded retries**: a hopeless run burns budget; cap steps and stop when the same error repeats.

Interviewers ask how you would sandbox an agent, how you stop it gaming tests, how you would evaluate it on your own codebase, what a resolved issue costs, and why computer use is harder than coding.
