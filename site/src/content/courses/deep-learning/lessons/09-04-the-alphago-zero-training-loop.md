---
id: "09.04"
course: "deep-learning"
module: "09"
slug: "the-alphago-zero-training-loop"
title: "The AlphaGo Zero training loop"
keyIdeas: ["AlphaGo Zero is a loop: self-play with MCTS generates $(s,\\pi,z)$ examples, the network trains on recent examples, and an evaluator decides which network generates the next games.", "MCTS acts as a policy improvement operator and training as policy evaluation plus distillation: a sample-based, neural version of policy iteration.", "Moves are sampled with $\\pi\\propto N^{1/\\tau}$: $\\tau=1$ for the first 30 moves for opening diversity, then $\\tau\\to0$ (most-visited move); Dirichlet root noise adds exploration everywhere.", "Training samples uniformly from the last 500,000 games with random symmetries; a new checkpoint replaces the self-play network only if it wins at least 55 percent of 400 evaluation games.", "Resignation saves compute but needs a false-positive check (10 percent of games played to the end); AlphaZero later dropped evaluation gating altogether."]
question: "To save compute, you cut self-play to 50 MCTS simulations per move while keeping everything else the same. Training loss goes down nicely, but playing strength plateaus early. Use the policy-iteration view of the loop to explain why, and say what you would change if compute really is limited."
answer: "The loop improves only as far as search improves on the network. With 50 simulations on a 362-move board, the search barely explores beyond the network's top few priors, so the visit distribution $\\pi$ is close to $p$ itself. Training $p$ towards $\\pi$ then teaches the network what it already believes: the policy improvement step has almost no effect, and the loss falls because the target is easy, not because play improves. Value targets also come from weak, near-raw-network games, so $v$ learns to evaluate a weak policy. With limited compute, keep a strong improvement signal where it matters: KataGo's playout-cap randomisation runs a full search (say 600 or more simulations) on a random fraction of moves and records only those as policy targets, while the remaining moves use a cheap search just to keep the game going. Also reduce other costs first: a smaller network, resignation with false-positive checks, and a larger replay window to reuse data."
sources: [{"title": "Silver et al.: Mastering the game of Go without human knowledge", "url": "https://www.nature.com/articles/nature24270", "note": "AlphaGo Zero; Methods describe self-play, the replay buffer, evaluation gating, temperature and resignation."}, {"title": "Silver et al.: Mastering Chess and Shogi by Self-Play (AlphaZero)", "url": "https://arxiv.org/abs/1712.01815", "note": "The same loop without gating, applied to three games."}, {"title": "Anthony, Tian and Barber: Thinking Fast and Slow with Deep Learning and Tree Search", "url": "https://arxiv.org/abs/1705.08439", "note": "Expert iteration: search as the expert, the network as the apprentice."}, {"title": "Wu: Accelerating Self-Play Learning in Go (KataGo)", "url": "https://arxiv.org/abs/1902.10565", "note": "Playout-cap randomisation and other changes that cut self-play compute about 50x."}]
practice: ["turn-root-visits-into-search-policy", "sample-recent-self-play-replay-batch", "apply-dihedral-go-symmetry", "mix-dirichlet-noise-into-root-priors", "alphago-policy-value-loss"]
tags: ["self-play", "expert iteration", "AlphaGo Zero"]
tools: ["NumPy", "PyTorch"]
---

## The loop

The previous three lessons supplied the pieces: rules and encoding (09.01), PUCT search (09.02) and a policy-value network (09.03). AlphaGo Zero connects them in a loop that starts from random weights and no human games, and after three days of training the 20-block version beat the version of AlphaGo that defeated Lee Sedol, 100 games to 0.

~~~dg-flow Three processes run continuously and asynchronously; the evaluator gates which weights generate data.
Best network -(MCTS self-play)-> Replay buffer of games -(sample minibatches)-> *Training -(checkpoint)-> Evaluator -(wins 55 percent?)-> Best network
~~~

1. **Self-play.** The current best network plays games against itself. Each move runs 1,600 MCTS simulations (about 0.4 seconds), records the root visit distribution $\pi_t$, and samples a move. When the game ends, every stored position is labelled with the result $z_t=\pm1$ from its player's perspective.
2. **Training.** Positions are sampled uniformly from the most recent 500,000 games, transformed by a random board symmetry, and used in minibatches of 2,048 for SGD with momentum 0.9 and a stepped learning-rate schedule. Every 1,000 steps a checkpoint is saved.
3. **Evaluation.** Each checkpoint plays 400 games against the current best network, with 1,600 simulations per move and no exploration noise. If it wins at least 55 percent, it becomes the new best network and starts generating self-play games.

The 20-block run used 4.9 million self-play games and 700,000 minibatches; the 40-block run, 29 million games over 40 days.

## Why it works: search as policy improvement

Lesson 08.02's policy iteration alternated evaluation and greedy improvement. Here:

- MCTS guided by $(p,v)$ produces a move distribution $\pi$ that is stronger than $p$. Search is a **policy improvement operator**: it looks ahead and corrects the network's mistakes locally.
- Training $p$ towards $\pi$ **distils** the improved policy into the network, and training $v$ towards $z$ **evaluates** the improved policy, since $z$ comes from games played with search.
- The next round of search uses the better network, producing a still better $\pi$.

This pattern, called expert iteration (Anthony et al., 2017, developed independently at the same time), avoids the high variance of policy-gradient self-play: the network is never trained on rewards directly, only by supervised targets produced by search. It is closer to generalised policy iteration than to the actor-critic methods of lesson 08.06, though the two-headed network is structurally the same.

## Exploration: temperature and noise

A deterministic player facing itself would replay the same game. Two mechanisms keep self-play diverse:

- **Temperature.** Moves are sampled from $\pi(a)\propto N(s_0,a)^{1/\tau}$. For the first 30 moves $\tau=1$, sampling in proportion to visits, so openings vary. After that $\tau\to0$, i.e. play the most-visited move, so the rest of the game reflects the search's best judgement and $z$ is a meaningful label.
- **Dirichlet noise** at the root ($\varepsilon=0.25$, $\alpha=0.03$, lesson 09.02) ensures every move is occasionally searched even if the network has learned to ignore it, so blind spots can be discovered and corrected.

<div class="ex">
<div class="lab">Worked example</div>

**the effect of temperature**

At the root after 1,600 simulations, three candidate moves have visit counts $(800, 600, 200)$.

$\tau=1$: probabilities $(0.50, 0.375, 0.125)$; the third move is still played one time in eight.

$\tau=0.5$ (square the counts): $(0.62, 0.35, 0.04)$. As $\tau\to0$, the first move is played every time.

Evaluation games and matches use $\tau\to0$ throughout; self-play uses $\tau=1$ only in the opening.

</div>

## Saving compute: resignation

Many self-play games are decided long before the end, and playing them out wastes simulations. AlphaGo Zero resigns when both the root value and the best child's value fall below a threshold $v_\text{resign}$. To choose that threshold safely, 10 percent of games disable resignation and are played to the end; $v_\text{resign}$ is set so that fewer than 5 percent of those games would have been resigned by a player who could still have won. Without this check, a value head that is wrong about some position type causes premature resignations, so those positions never get played out and the error is never corrected.

## Variations and practical lessons

| Choice | AlphaGo Zero | AlphaZero and later engines
| --- | --- | --- |
| Which network plays self-play | best checkpoint, gated at 55 percent | always the latest weights (AlphaZero)
| Simulations per move | 1,600 | 800 (AlphaZero); randomised caps (KataGo)
| Dirichlet $\alpha$ | 0.03 for Go | scaled to the typical number of legal moves (0.3 for chess)
| Targets | $\pi$ and $z$ | plus ownership, score and other auxiliary targets (KataGo)

Small-scale reproductions (Leela Zero, Minigo, KataGo) found the loop robust but sensitive to a few things: a replay window too short overfits to the latest games and causes forgetting and cycles in strength; too few simulations make $\pi$ barely better than $p$, so there is little to distil; value targets from games with heavy exploration noise are biased; and buggy rules (lesson 09.01) are learned and exploited faithfully. KataGo reported roughly a 50x reduction in compute versus AlphaGo Zero's approach by combining auxiliary targets, playout-cap randomisation and architectural changes. The same loop, with a learned model in place of the rules, became MuZero.

<div class="take">
<div class="lab">Key takeaway</div>

AlphaGo Zero alternates MCTS self-play, which turns the network's policy into a stronger search policy, with supervised training that distils that policy and learns game outcomes; temperature, root noise, a large replay window, gated evaluation and checked resignation keep the loop exploring and improving.

</div>
