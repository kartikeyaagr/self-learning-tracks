---
id: "08.04"
course: "ai-engineering-course"
module: "08"
slug: "tree-of-thought"
title: "Tree of thought"
keyIdeas: ["Chain of thought commits to one path; an early mistake cannot be undone.", "Tree of Thoughts (ToT) treats reasoning as search: propose several next steps, have an evaluator rate each partial state, keep the best, and backtrack from dead ends.", "On Game of 24, GPT-4 with ToT solved 74% vs 4% with chain-of-thought prompting, at many times the model calls.", "The practical form is best-of-n with a verifier: sample n complete answers, check them (tests, a solver, a trained verifier) and return one that passes.", "The verifier sets the ceiling. With a noisy verifier, more samples stop helping; accuracy approaches the verifier's precision on accepted answers."]
question: "Tree of Thoughts solved 74% of Game of 24 puzzles vs 4% for chain of thought with GPT-4. What does ToT do differently that explains the gap, and when would you use plain best-of-n with a verifier instead?"
answer: "Chain of thought follows one path and cannot recover when an early arithmetic choice leaves an unsolvable state. ToT generates several candidate next steps at each stage, has the model rate each partial state (sure, maybe or impossible to reach 24), keeps the best few (breadth 5 in the paper) and prunes dead ends. Exploring and judging alternatives lifts success from 4% to 74%, at the cost of many more model calls. Best-of-n is simpler: sample n complete answers independently and return one the verifier accepts. Use it when you can check a finished answer cheaply and reliably (unit tests, a solver, a schema plus business rules) but partial states are hard to judge, which covers most code generation and structured extraction. Use step-level search when partial states can be evaluated and early mistakes are fatal."
rubric: ["Explains that chain of thought cannot backtrack from an early bad step.", "Describes ToT's three parts: proposing several next steps, evaluating partial states, and search that keeps the best and prunes.", "Notes the cost: many more model calls per problem.", "Describes best-of-n with a verifier and when it is the better choice (finished answers are cheap to check, partial states are not)."]
sources: [{"title": "Yao et al.: Tree of Thoughts (2023)", "url": "https://arxiv.org/abs/2305.10601", "note": "Search over reasoning steps with self-evaluation."}, {"title": "Cobbe et al.: Training Verifiers to Solve Math Word Problems (2021)", "url": "https://arxiv.org/abs/2110.14168", "note": "GSM8K and best-of-n selection with a trained verifier."}, {"title": "Snell et al.: Scaling LLM Test-Time Compute Optimally (2024)", "url": "https://arxiv.org/abs/2408.03314", "note": "When more inference compute beats a bigger model."}]
practice: ["pass-at-k-estimator", "beam-search-step", "beam-search-decoding"]
---

## Beyond one chain

A chain of thought is a single path. If step 2 is wrong, steps 3 onward build on the error, and the model rarely goes back. People solving puzzles try an approach, notice it is going nowhere, and back up. Tree of Thoughts (Yao et al., 2023) gives the model the same ability by wrapping it in ordinary search code.

## How ToT works

The problem is broken into steps, and each partial solution is a node in a tree.

~~~dg-flow Tree of Thoughts: expand, evaluate, keep the best, repeat.
Problem -(generate k thoughts)-> Candidate states -(LLM rates: sure / maybe / impossible)-> *Keep best b -(expand again or backtrack)-> Solution
~~~

| Part | Paper's version | Stronger version |
| --- | --- | --- |
| Proposer | Ask the model for k possible next steps | Sample at higher temperature for diversity |
| Evaluator | Ask the model "can 10, 13, 13 still reach 24? sure / maybe / impossible", or vote between states | A trained verifier, unit tests, a solver, a rule |
| Search | Breadth-first keeping b states per level, or depth-first with backtracking | Beam search, best-first, Monte Carlo tree search |

The model plays both proposer and critic; the search loop is plain code.

| Task (GPT-4) | Chain of thought | Tree of Thoughts |
| --- | --- | --- |
| Game of 24 (success) | 4% | 74% (b = 5) |
| Mini crosswords (words correct) | about 16% | about 60% |
| Creative writing (human preference) | Less preferred | Preferred |

Game of 24 (combine four numbers with + − × ÷ to make 24) has clear intermediate states and dead ends that are easy to judge, which is exactly where search shines. The price is calls: the first level alone can generate up to 36 candidate states from four numbers (6 pairs × up to 6 operations, before removing duplicates), each proposed and evaluated by the model. Success went from 4% to 74%, and cost rose by an order of magnitude or more.

## Run the search yourself

The figure is a working breadth-first ToT on Game of 24. An evaluator rates every new state as possible or impossible and the best b are kept. The evaluator is an exact solver with an adjustable error rate, standing in for an LLM's judgement.

~~~dg-raw
%%widget tot Breadth-first Tree of Thoughts on Game of 24 with a noisy evaluator.
~~~

- Numbers 4, 9, 10, 13, breadth 5, error 15%: usually solved, for example (13 − 9) × (10 − 4) = 24.
- Breadth 1 with 20% error: close to a single chain. One misjudged state and the search fails.
- Error 0%: breadth 1 is enough. A good evaluator is worth more than a wide search.

## Best-of-n with a verifier

The last point is the practical lesson. Most production systems skip step-level search and use its simplest form: sample n complete answers independently, check each with a verifier, and return one that passes. Cobbe et al. introduced this on GSM8K with a trained verifier that ranks candidate solutions. In code generation the verifier is the test suite: generate several patches, run the tests, keep one that passes. That is search with a near-perfect evaluator, and it is one of the most reliable patterns there is.

How much it helps depends on two numbers: how often one sample is right ($p$), and how often the verifier accepts wrong answers (its false-accept rate).

<div class="ex">
<div class="lab">Worked example</div>

**eight samples, two verifiers**

A generator is right on 30% of samples. Draw $n = 8$ and return the first one the verifier accepts (or the first sample if none is accepted).

With a perfect verifier you get pass@8: $1 - 0.7^8 = 1 - 0.058 = 0.942$.

A real verifier accepts 90% of right answers and 10% of wrong ones. Per sample: right and accepted $0.3 \times 0.9 = 0.27$; wrong and accepted $0.7 \times 0.1 = 0.07$; rejected $0.66$.

- At least one accepted: $1 - 0.66^8 = 1 - 0.036 = 0.964$.
- An accepted answer is right with probability $\frac{0.27}{0.27 + 0.07} = 0.794$.
- Accuracy is about $0.964 \times 0.794 = 0.766$, plus 0.002 from the fallback: about 77%, up from 30%.

Raise the false-accept rate to 30%: wrong and accepted becomes 0.21, an accepted answer is right $\frac{0.27}{0.48} = 56\%$ of the time, and accuracy is about 56%. Going from 8 to 80 samples barely changes either figure, because the ratio $\frac{0.27}{0.27 + b}$ does not depend on $n$.

</div>

So once a few samples are drawn, **accuracy is capped by the verifier's precision, not by n**. Spend effort on the checker: tighter tests, a solver, rules in code, or a process reward model that scores each step ([Diagnose four reasoning failures](lesson:aieng:08.05) has a figure for this trade-off). Ranking candidates by a graded verifier score, rather than taking the first one accepted, helps further when the scores are calibrated.

## Test-time compute

ToT, voting ([Chain of thought](lesson:aieng:08.02)) and best-of-n are all ways to spend more compute at inference for a better answer. Reasoning models fold much of this into one long chain, learned through RL ([Reasoning models and human feedback](lesson:aieng:08.01)). Snell et al. (2024) found that for some problems, compute-optimal test-time search let a small model beat one about 14 times larger at matched compute, with the best strategy depending on how hard the question is: revision of a single answer for easier questions, wider search for harder ones. Choosing between a reasoning model's own budget and your own sampling is covered in [Using reasoning models in apps](lesson:aieng:08.06).

## Where this breaks

- **No reliable evaluator.** If the model judges its own partial states and is wrong 30% of the time, a wide search mostly explores wrong branches. Open-ended writing has no cheap verifier at all.
- **Correlated samples.** Best-of-n assumes the samples differ. At low temperature, or when the model has one fixed misconception, eight samples are one sample eight times.
- **Cost and latency.** Each candidate is a full generation, and each check may be another call. Use these patterns where a wrong answer is expensive and can be detected, not on every request.

<div class="take">
<div class="lab">Key takeaway</div>

Tree of Thoughts turns reasoning into search: propose steps, judge them, keep the best and backtrack. In practice the workhorse is best-of-n with a verifier, and the verifier's precision, not the number of samples, sets how good the answer gets.

</div>
