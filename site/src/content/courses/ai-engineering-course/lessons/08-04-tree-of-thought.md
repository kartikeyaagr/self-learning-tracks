---
id: "08.04"
course: "ai-engineering-course"
module: "08"
slug: "tree-of-thought"
title: "Tree of thought"
keyIdeas: ["Chain of thought commits to one path; an early mistake can't be undone.", "Tree of Thoughts (ToT) treats reasoning as search: generate several candidate next steps ('thoughts'), evaluate them, expand the promising ones, and backtrack from dead ends.", "Three pieces: a thought generator, a state evaluator (the LLM rates states as sure / maybe / impossible), and a search algorithm (BFS or DFS).", "On Game of 24, GPT-4 with ToT solved 74% vs 4% with chain-of-thought prompting.", "Costs many more model calls; best for problems with clear intermediate states and checkable progress.", "Same theme as test-time compute scaling: search, voting and verification at inference can substitute for a bigger model on hard problems."]
question: "Tree of Thoughts solved 74% of Game of 24 puzzles vs 4% for chain of thought with GPT-4. What does ToT do differently that explains the gap?"
answer: "Chain of thought follows one path and can't recover if an early arithmetic choice leaves an unsolvable state. ToT generates several candidate next steps at each stage, has the model evaluate each partial state (sure/maybe/impossible to reach 24), keeps the best few (BFS with breadth 5) and prunes or backtracks from dead ends. Exploring and evaluating alternatives is what lifts success from 4% to 74%, at the cost of many more model calls."
sources: [{"title": "Yao et al.: Tree of Thoughts (2023)", "url": "https://arxiv.org/abs/2305.10601", "note": "Search over reasoning steps with self-evaluation."}, {"title": "Snell et al.: Scaling LLM Test-Time Compute Optimally (2024)", "url": "https://arxiv.org/abs/2408.03314", "note": "When more inference compute beats a bigger model."}, {"title": "Besta et al.: Graph of Thoughts (2023)", "url": "https://arxiv.org/abs/2308.09687", "note": "Generalising chains and trees to graphs of thoughts."}]
---

## Beyond one chain

A chain of thought is a single path. If step 2 is wrong, steps 3 onward build on the error, and the model rarely goes back. Humans solving puzzles try an approach, notice it's going nowhere, and back up. Tree of Thoughts (Yao et al., 2023) gives the model the same ability.

## How ToT works

The problem is broken into steps. Each partial solution is a node in a tree.

~~~dg-flow Tree of Thoughts: expand, evaluate, keep the best, repeat.
Problem -(generate k thoughts)-> Candidate states -(LLM rates: sure / maybe / impossible)-> *Keep best b -(expand again or backtrack)-> Solution
~~~

1. **Thought generator**: ask the model for several possible next steps from the current state.
2. **State evaluator**: ask the model to judge each state, for example "can 10, 13, 13 still reach 24? sure / maybe / impossible", or to vote between states.
3. **Search**: breadth-first search keeping the best b states per level, or depth-first with backtracking when a state is judged impossible.

The model plays both roles, proposer and critic, orchestrated by ordinary code.

## The results

| Task (GPT-4) | Chain of thought | Tree of Thoughts |
| --- | --- | --- |
| Game of 24 | 4% | 74% (b = 5) |
| Creative writing coherence (human preference) | Lower | Preferred |
| Mini crosswords (words correct) | ~16% | ~60% |

Game of 24 (combine four numbers with + - x / to make 24) has clear intermediate states and dead ends that are easy to judge, which is exactly where search shines. Chain of thought fails because a bad first operation leaves an unsolvable state with no way back.

## The price

ToT uses many more model calls per problem: every generated and evaluated thought is a call. It's worth it for hard problems where progress can be checked, not for everyday questions.

## Test-time compute

ToT is one instance of a broader idea: spend more compute at inference to get better answers. Options include self-consistency voting (08.02), best-of-n with a verifier, tree or beam search with a process reward model, and the long internal reasoning of o1/R1-style models. Snell et al. (2024) found that, for some problems, optimally allocated test-time compute let a small model beat one 14x larger. Graph of Thoughts extends trees to graphs where partial solutions can be merged.

## Run the search yourself

The figure is a working breadth-first Tree of Thoughts on the Game of 24. Each level combines two of the remaining numbers with + − × ÷. An evaluator rates every new state as possible or impossible, and the best b states are kept. The evaluator here is an exact solver with an adjustable error rate, standing in for an LLM's "sure / maybe / impossible" judgement.

~~~dg-raw
%%widget tot Breadth-first Tree of Thoughts on Game of 24 with a noisy evaluator.
~~~

Things to try:

- Default numbers 4, 9, 10, 13, breadth 5, error 15%: usually solved, for example (13 − 9) × (10 − 4) = 24.
- Breadth 1 with 20% error: close to a single chain of thought. One misjudged state and the search fails.
- Raise breadth to 10: almost always solved even with a noisy evaluator, at roughly twice the calls.
- Error 0%: the evaluator is perfect, so breadth 1 is enough. A good evaluator is worth more than a wide search.

## What the counts mean

The first level alone generates up to 36 candidate states from four numbers (6 pairs × up to 6 operations, minus duplicates). Every candidate needs a model call to propose and another to evaluate in the original method. Yao et al. reported ToT on Game of 24 using far more tokens per problem than chain of thought: success went from 4% to 74%, and cost went up by an order of magnitude or more. That trade only makes sense when a wrong answer is expensive and states can be judged.

## The general pattern

Tree of Thoughts has three swappable parts, and you can build your own versions:

| Part | Simple version | Stronger version |
| --- | --- | --- |
| Proposer | Ask the model for k next steps | Sample at higher temperature for diversity |
| Evaluator | Ask the model to rate a state | A trained verifier, unit tests, a solver, or a rule |
| Search | BFS keeping b states | Beam search, best-first, or Monte Carlo tree search |

In code generation, the evaluator can be the test suite: generate several patches, run the tests, keep the ones that pass, and refine. That is ToT with a perfect evaluator, and it is one of the most reliable patterns in practice.

<div class="take">
<div class="lab">Key takeaway</div>

Tree of Thoughts turns reasoning into search: propose several next steps, have the model judge them, keep the best and backtrack from dead ends. It costs more calls and pays off on problems with checkable intermediate states.

</div>
