---
id: "09.02"
course: "deep-learning"
module: "09"
slug: "monte-carlo-tree-search"
title: "Monte Carlo tree search"
keyIdeas: ["MCTS grows a search tree selectively: each simulation **selects** a path with a bandit rule, **expands** one new leaf, **evaluates** it, and **backs up** the value along the path, so strong lines get searched deeply and weak ones are dropped early.", "Each edge stores a visit count $N$, a value sum $W$, a mean $Q=W/N$ and a prior $P$. Values are from the view of the player who chose the edge, so they flip sign at every level on the way up.", "AlphaGo Zero selects with PUCT, $Q(s,a)+c_\\text{puct}P(s,a)\\sqrt{\\sum_bN(s,b)}/(1+N(s,a))$: the network's prior decides where to look, the search's mean values decide what is actually good, and the bonus fades as a move is visited.", "There are no random rollouts in AlphaGo Zero: a new leaf is evaluated once by the value head, and the policy head supplies the priors for its children; finished games are scored by the rules.", "The move played is chosen from root visit counts, not $Q$, because many visits mean the search kept confirming the move; in self-play, Dirichlet noise ($\\varepsilon=0.25$, $\\alpha=0.03$ for Go) on the root priors makes sure every move occasionally gets a real look."]
question: "A teammate's MCTS converts each leaf value to the root player's perspective once and adds that same number to every edge on the path; selection at every node then maximises $Q+U$. Search looks fine when it only expands the root's children, but deeper searches approve moves that lose to a simple reply, and adding more simulations does not fix it. What game is the search actually solving, why does the depth-1 case hide it, why do extra simulations not help, and how do you fix it?"
answer: "At nodes where the opponent is to move, the edges now store the root player's value, and selection picks the edge that maximises it. The search therefore models an opponent who chooses the reply that is best for the root player: it is solving a cooperative game, not minimax. A move whose only refutation is one strong reply looks excellent, because the search assumes the opponent will never play that reply. With only the root's children expanded, every selection happens at a root-player node, where maximising the root player's value is correct, so the bug is invisible; it appears as soon as the tree contains opponent decisions. More simulations do not help because the search converges to the wrong target: once it finds a line where the 'cooperative' opponent lets it win, that line's mean value approaches +1, PUCT's bonus for the alternatives fades like $1/(1+N)$, and the search keeps confirming the illusion instead of looking for the refutation (in the lesson's tic-tac-toe test, 30% of runs still chose a losing move at 1,600 simulations). The fix is to store each edge's value from the perspective of the player who chose it: back up $v$ from the leaf player's view and negate it at every level, so that each node's maximisation is that node's player looking after their own interests. A good regression test is a position with a single only-move defence, checked against an exact solver or a known answer."
sources: [{"title": "Silver et al.: Mastering the game of Go without human knowledge", "url": "https://www.nature.com/articles/nature24270", "note": "AlphaGo Zero; the Methods section gives the exact select, expand, backup and play rules, PUCT and the root noise."}, {"title": "Silver et al.: Mastering Chess and Shogi by Self-Play (AlphaZero)", "url": "https://arxiv.org/abs/1712.01815", "note": "The same search for chess and shogi, with how the Dirichlet alpha scales with the number of legal moves."}, {"title": "Chess Programming Wiki: UCT", "url": "https://www.chessprogramming.org/UCT", "note": "UCT as bandits in a tree, with history and variants such as first-play urgency."}]
practice: ["select-action-with-puct", "run-deterministic-neural-guided-mcts", "back-up-mcts-simulation-path", "initialize-legal-mcts-edge-statistics", "mix-dirichlet-noise-into-root-priors"]
tags: ["MCTS", "PUCT"]
tools: ["NumPy"]
---

## What you will learn

- Why growing only the promising part of the game tree beats full-width minimax in Go.
- The four phases of a simulation (select, expand, evaluate, back up) and what each edge stores.
- How the PUCT rule balances "this move has scored well" against "the network likes it and we have barely tried it".
- Why values flip sign at every level, and what breaks when they do not.
- How AlphaGo Zero picks its move from visit counts and keeps self-play exploring with Dirichlet noise.

*Before you start:* lesson 09.01 (game trees, minimax, why Go's tree is too big) and lesson 08.05 (multi-armed bandits and the UCB rule). Lesson 08.06 (a policy plus a value estimate) helps.

Lesson 09.01 showed that full-width minimax is hopeless for Go: around $10^{360}$ positions and no reliable hand-written evaluation. **Monte Carlo tree search (MCTS)** sidesteps both problems by running many short **simulations** from the current position. Each one walks down the tree built so far, adds one new position, estimates who is winning there, and reports back up the path. Moves that keep looking good get more simulations, so the tree grows deep along strong lines only.

## The idea in one picture

Take a tic-tac-toe position, small enough that we can check every answer exactly. Squares are numbered 0 to 8, row by row. X has stones on 1 and 5, O has stones on 6 and 7, and X is to move.

O threatens to complete the bottom row on square 8. So X has one sensible move: block on 8. That block also threatens the right column (5 and 8, needing 2), so O must reply on 2; that reply threatens O's diagonal 2-4-6, so X takes 4, and the stone on 4 makes two X threats at once (squares 0 and 3). O can block only one, so X wins. Every other X move loses immediately to O playing 8. An exact solver confirms this: move 8 is a win, moves 0, 2, 3 and 4 are losses.

Now run MCTS with no knowledge at all: every legal move gets the same prior, and positions are judged by playing random moves to the end of the game. After 200 simulations the root looks like this:

| Move for X | 0 | 2 | 3 | 4 | 8
| --- | --- | --- | --- | --- | --- |
| Visits $N$ | 3 | 6 | 4 | 6 | 180
| Mean value $Q$ | -0.33 | 0.00 | 0.00 | 0.00 | +0.88
| True result | loss | loss | loss | loss | win

~~~dg-fig 09.02-idea.svg
Left: the test position, X to move. Right: root visit counts after 200 MCTS simulations with uniform priors and random playouts. The search spent 180 of its 199 root visits on square 8, the only winning move.
~~~

Read it in plain words. The search tried each move a few times, noticed that square 8 kept producing good results, and spent nearly all of its effort there, checking that the result held up against O's best replies. (The counts add to 199, not 200, because the very first simulation is spent creating the root itself.) The rejected moves still show $Q\approx0$ rather than $-1$: rough estimates, but enough to stop wasting time on them.

A **node** is a position in the search tree and an **edge** $(s,a)$ is a move $a$ played from position $s$. Values run from $-1$ (loss) to $+1$ (win).

## Step 1: what each edge stores

For every edge $(s,a)$ in the tree, AlphaGo Zero keeps four numbers:

- The **visit count** $N(s,a)$: how many simulations took this edge.
- $W(s,a)$: the sum of the values backed up through it.
- The **mean value** $Q(s,a)=W/N$: the search's current estimate of the move, from the view of the player who makes it.
- The **prior** $P(s,a)$: how promising the move looked before search (the policy head's output, fixed when $s$ was expanded; uniform in the example above).

A new edge starts at $N=W=Q=0$. The tree begins as a single root node and gains at most one new node per simulation, so one search of 1,600 simulations adds at most 1,600 positions, a vanishing fraction of the full game tree.

## Step 2: select a path with PUCT

At every node it reaches, a simulation must choose one child. This is the **exploration-exploitation dilemma** of lesson 08.05, once per node: the best average so far (exploit), or a move we know little about (explore)?

**UCT** (Kocsis and Szepesvari, 2006) answered this by running the UCB bandit rule at every node. AlphaGo Zero uses a variant called **PUCT** ("predictor plus UCB"), which also uses the network's prior:

$$a^*=\arg\max_a\Big[Q(s,a)+c_\text{puct}\,P(s,a)\,\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}\Big]$$

Symbol by symbol:

- $Q(s,a)$ is the exploitation term: what the search has found so far.
- The second term is the **exploration bonus**, often written $U(s,a)$.
- $\sum_b N(s,b)$ is the parent's total visits; its square root grows slowly.
- $1+N(s,a)$ shrinks the bonus as this move gets visited.
- $P(s,a)$ scales the bonus: moves the network likes are explored first and longest.
- $c_\text{puct}$ sets how much exploration matters relative to $Q$.

The effect is a division of labour: the network decides *where to look*, the search decides *what is actually good*.

<div class="ex">
<div class="lab">Worked example</div>

**one PUCT decision**

The root has been visited 100 times, so $\sqrt{100}=10$. Take $c_\text{puct}=1.5$.

Move A has $Q=0.50$, $P=0.20$, $N=40$. Bonus $=1.5\times0.20\times10/41=3/41=0.073$. Score $=0.50+0.073=0.573$.

Move B has $Q=0.30$, $P=0.50$, $N=10$. Bonus $=1.5\times0.50\times10/11=7.5/11=0.682$. Score $=0.30+0.682=0.982$.

B is searched next, even though its average is lower: the network rates it highly and it has had few visits.

Suppose B keeps returning 0.30. After 18 more visits to B (root total 118, $\sqrt{118}=10.86$), B scores $0.30+1.5\times0.5\times10.86/29=0.581$ and A scores $0.50+1.5\times0.2\times10.86/41=0.579$, so B is still just ahead. After 19 more (total 119), B scores $0.573$ and A scores $0.580$: A takes the next visit.

</div>

~~~dg-fig 09.02-bonus.svg
Left: the PUCT bonus for three priors, against the UCB1 bonus. PUCT starts higher for liked moves and decays like 1/(1+N), much faster than UCB1's 1/sqrt(N). Right: the example continued; B's score falls with each visit until A overtakes it after 19 extra visits.
~~~

The left panel shows the main difference from the UCB1 rule of lesson 08.05, whose bonus $\sqrt{2\ln t/N}$ has no prior and fades only like $1/\sqrt N$. PUCT's bonus fades like $1/N$, so once a move has a reliable $Q$, the search trusts it. Values of $c_\text{puct}$ in public reimplementations range from about 1 to 5; the AlphaGo Zero paper does not state the value it used.

## Step 3: expand and evaluate the leaf

Selection stops at a position not yet in the tree, a **leaf**.

**Expand.** Run the network once on the leaf: $f_\theta(s)=(p,v)$. The policy $p$ has 362 entries for Go (361 points plus pass). Set illegal moves to zero, renormalise, and create one edge per legal move with $P=p_a$ and $N=W=0$.

**Evaluate.** The same call gives a value $v\in[-1,1]$: the network's estimate of the final result *for the player to move at the leaf*. That number is the simulation's result. If the leaf is a finished game, use the real result from the rules instead.

This is where AlphaGo Zero departs from classic MCTS. Programs like Crazy Stone and MoGo evaluated a leaf by a **rollout**: playing fast, random-ish moves to the end and recording who won, which reached strong amateur level in Go. The original AlphaGo mixed rollouts with a value network. AlphaGo Zero dropped rollouts entirely: one network call per simulation, no playout.

## Step 4: back up the value, flipping its sign

Now walk back up the path from the leaf to the root, updating each edge: add 1 to $N$, add the value to $W$, recompute $Q=W/N$.

The value must be **negated at every level**. Tic-tac-toe and Go are zero-sum: a position worth $+0.6$ to the player about to move there is worth $-0.6$ to the player who just moved into it. Each edge must store the value from the point of view of the player who *chose* that edge, because that player is the one maximising over it during selection.

<div class="ex">
<div class="lab">Worked example</div>

**backing up one simulation**

Path: at the root X plays 8 (edge e1); then O plays 2 (edge e2); the new leaf has X to move. The network says $v=+0.6$ for X at the leaf.

Edge e2 was chosen by O. From O's side the leaf is worth $-0.6$. Before: $N=4$, $W=-3.0$ ($Q=-0.75$). After: $N=5$, $W=-3.6$, $Q=-3.6/5=-0.72$.

Edge e1 was chosen by X. Negate again: $+0.6$. Before: $N=10$, $W=7.0$ ($Q=0.70$). After: $N=11$, $W=7.6$, $Q=7.6/11=0.691$.

Both means moved towards this simulation's result, each seen from its own chooser's side.

</div>

~~~dg-seq One AlphaGo Zero simulation: no rollouts; the network is called once per simulation, at the new leaf.
Search > Tree : select by PUCT from the root
Tree -> Search : path to an unexpanded leaf
Search > Network : evaluate leaf position
Network -> Search : priors p, value v
Search > Tree : expand leaf with legal priors
Search > Tree : back up v, flipping sign per level
~~~

AlphaGo Zero repeats the four steps 1,600 times per move in self-play.

## Step 5: choose the move, and keep self-play exploring

**Which move to play.** After the simulations, AlphaGo Zero plays from the root *visit counts*, not from $Q$:

$$\pi(a)\propto N(s_0,a)^{1/\tau}$$

Here $s_0$ is the root and $\tau$ a **temperature**: $\tau=1$ samples in proportion to visits, $\tau\to0$ plays the most visited move (lesson 09.04). Visit counts are the robust choice: a move with a high $Q$ from three visits may be a lucky estimate, while a high $N$ means the search kept coming back and kept confirming it.

**Dirichlet noise at the root.** In self-play the network trains on the search's output, so a move it wrongly gives near-zero prior might never be searched or corrected. So the root priors are mixed with noise:

$$P(s_0,a)=(1-\varepsilon)\,p_a+\varepsilon\,\eta_a,\qquad \eta\sim\mathrm{Dir}(\alpha)$$

$\varepsilon=0.25$ is the share of noise and $\eta$ is a random probability vector drawn from a **Dirichlet distribution** with concentration $\alpha$. AlphaGo Zero used $\alpha=0.03$ for Go. A small $\alpha$ makes $\eta$ spiky: in our own draws over 362 moves, the median number of moves holding 90% of the noise mass was 19. So each search gives a different small, random set of moves a serious boost, rather than nudging all of them a little.

~~~dg-fig 09.02-dirichlet.svg
Root priors over 30 legal moves after mixing in 25% Dirichlet noise. With a small alpha (left) one or two unlikely moves get a large boost and will really be searched; with a large alpha (right) the noise is a thin, nearly useless layer over everything.
~~~

Noise is added at the root only, and only in self-play, not in competitive games.

## MCTS in code

A complete MCTS for tic-tac-toe. The "network" is a stand-in returning uniform priors and one random playout's result; swap `evaluate` for a policy-value network and the rest is the AlphaGo Zero search.

```python
import math, numpy as np

LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def to_move(b): return 'X' if b.count('X') == b.count('O') else 'O'
def legal(b):   return [i for i, c in enumerate(b) if c == '.']
def play(b, a): return b[:a] + to_move(b) + b[a+1:]
def terminal_value(b):                      # from the view of the player to move, or None
    for i, j, k in LINES:
        if b[i] != '.' and b[i] == b[j] == b[k]:
            return -1.0                     # the player who just moved completed a line
    return 0.0 if '.' not in b else None

class Node:
    def __init__(self, board):
        self.board, self.children, self.expanded = board, {}, False

def evaluate(board, rng):                   # stand-in for the network: uniform priors + one random playout
    acts = legal(board)
    b, sign = board, 1.0
    while (t := terminal_value(b)) is None:
        b, sign = play(b, rng.choice(legal(b))), -sign
    return {a: 1 / len(acts) for a in acts}, sign * t

def simulate(root, c_puct, rng):
    node, path = root, []
    while node.expanded:                                              # 1. select
        sqrt_total = math.sqrt(sum(node.N.values()))
        a = max(node.N, key=lambda a: node.Q[a] + c_puct * node.P[a] * sqrt_total / (1 + node.N[a]))
        path.append((node, a))
        node = node.children.setdefault(a, Node(play(node.board, a)))
    v = terminal_value(node.board)
    if v is None:                                                     # 2. expand + 3. evaluate
        node.P, v = evaluate(node.board, rng)
        node.N = {a: 0 for a in node.P}; node.W = {a: 0.0 for a in node.P}
        node.Q = {a: 0.0 for a in node.P}; node.expanded = True
    for parent, a in reversed(path):                                  # 4. back up
        v = -v                       # value now from the view of the player who chose edge a
        parent.N[a] += 1; parent.W[a] += v; parent.Q[a] = parent.W[a] / parent.N[a]

rng = np.random.default_rng(0)
root = Node('.X...XOO.')            # X to move; O threatens 6-7-8
for _ in range(200):
    simulate(root, c_puct=1.5, rng=rng)
print({a: root.N[a] for a in root.N})            # {0: 3, 2: 6, 3: 4, 4: 6, 8: 180}
print({a: round(root.Q[a], 2) for a in root.Q})  # {0: -0.33, 2: 0.0, 3: 0.0, 4: 0.0, 8: 0.88}
print('play', max(root.N, key=root.N.get))       # play 8
```

Details worth copying exactly: the leaf value is from the leaf player's view and is negated *before* updating each edge; unvisited edges have $Q=0$, as in AlphaGo Zero; terminal positions are scored by the rules.

## Reading the result

On a game small enough to solve exactly, compare against the truth. The next figure runs three versions of the search 40 times each on the opening position.

~~~dg-fig 09.02-converge.svg
Share of root visits on the winning square 8, averaged over 40 runs. A good prior and value (here simulated by adding noise to the exact answer) converge fastest; uniform priors with random rollouts get there more slowly; a prior that puts 83% on a losing move costs early simulations but is overruled by the search.
~~~

Measured numbers from those runs:

- With uniform priors and random rollouts, square 8 gets 59% of root visits at 50 simulations and 88% at 200; from 100 simulations on, all 40 runs choose it.
- With a good simulated network, the share is already 79% at 10 simulations.
- With a misleading prior, *no* run picks square 8 at 10 simulations (the search barely has time to look past the prior), but from 20 simulations on, 98% or more do. The search corrects a bad network, which is why training the network towards visit counts (lesson 09.03) improves it.

Diagnostics for a real engine, where the truth is unknown:

- **Visit concentration.** The top move's share should grow with more simulations; a flat spread suggests $c_\text{puct}$ too high or values with no signal.
- **Search versus prior.** If visits always equal the raw policy $p$, the search adds nothing; large disagreement is normal early in training.
- **Tactical test positions.** Keep positions with a known only move (like the one above) and check the engine finds it within a fixed budget.

## What goes wrong

The most damaging bug is in backup. Suppose you convert each leaf value once to the *root player's* view and add that number to every edge on the path. At the root's children this is correct, but at opponent nodes selection now picks the reply best *for the root player*: the search imagines a cooperative opponent.

~~~dg-fig 09.02-signbug.svg
A deeper test position (O to move, six empty squares, only square 8 wins), 40 runs per point. With correct sign flipping, the share of runs choosing the winning move climbs to 0.93 and losing choices vanish by 200 simulations. With the root-perspective bug, choices freeze early: 30% of runs pick a losing move even at 1,600 simulations.
~~~

More simulations do not help: once the buggy search finds a line where the "cooperative" opponent lets it win, that line's $Q$ is $+1$ and it stops looking elsewhere.

| Mistake | Symptom | Fix
| --- | --- | --- |
| Same-sign backup (root perspective everywhere) | plays moves that lose to one obvious reply; more simulations do not help | negate $v$ at every level, store values from the chooser's view
| Leaf value from the wrong player | engine plays the worst moves confidently | the value head predicts for the player to move at the leaf; check one known position
| Illegal moves keep prior mass | search wastes visits, may select an illegal move | mask before renormalising the priors
| Choosing the move by highest $Q$ | occasional blunders on moves with few visits | choose by visit count
| $c_\text{puct}$ too high or too low | visits nearly equal to the prior, or tunnel vision on one line | tune on test positions or by self-play matches
| No noise in self-play | the network never learns about moves it initially ignores | mix $\varepsilon=0.25$ of $\mathrm{Dir}(\alpha)$ into root priors

## Going deeper

**Where UCT's guarantee comes from.** Kocsis and Szepesvari proved that UCT's choice at the root converges to the minimax-optimal move as the number of simulations goes to infinity, with the failure probability shrinking polynomially. The name "Monte Carlo tree search" comes from Remi Coulom's 2006 Crazy Stone paper, and the predictor-UCB idea behind PUCT from Rosin (2011). The exact AlphaGo Zero rule has no comparable published guarantee.

**Search as policy improvement.** The visit distribution $\pi$ is $p$ corrected by look-ahead. Grill et al. (2020) showed that, approximately, PUCT's visit counts track the solution of a regularised policy-improvement problem: maximise expected $Q$ while staying close (in a KL sense) to the prior. That is why distilling $\pi$ back into $p$ (lesson 09.03) behaves like policy iteration (lesson 08.02).

**First-play urgency.** $Q=0$ for unvisited edges is optimistic in bad positions and pessimistic in good ones. Engines such as Leela Zero start unvisited children somewhat below the parent's value (**first play urgency**), so the search does not visit every low-prior move.

**A growing constant.** AlphaZero's published pseudocode (stated from memory) lets the exploration constant grow slowly with the parent's visits, $c(s)=\log\frac{1+N(s)+c_\text{base}}{c_\text{base}}+c_\text{init}$ with $c_\text{base}=19652$ and $c_\text{init}=1.25$. It also scaled the Dirichlet $\alpha$ inversely with the typical number of legal moves: 0.3 for chess, 0.15 for shogi, 0.03 for Go.

**Tree reuse.** After a move is played, its subtree becomes the new root, keeping its statistics.

**Parallel search.** Many simulations run at once and leaf evaluations are batched on the accelerator (AlphaGo Zero batched 8). Without care, every thread would descend the same most promising path. A **virtual loss** temporarily counts an in-flight path as if it had lost, so other threads spread out; the real value replaces it at backup.

**Cost.** One simulation is one network call plus an argmax over legal moves at each level; memory grows by at most one node per simulation. The network call dominates, so AlphaGo Zero spends its compute on 1,600 well-chosen evaluations per move rather than millions of shallow ones.

## Key terms

| Term | Meaning
| --- | --- |
| MCTS | Search that grows a tree selectively by repeated simulations of select, expand, evaluate, back up
| Simulation | One pass from the root to a new leaf and back, adding at most one node
| Visit count $N(s,a)$ | Number of simulations that took edge $(s,a)$; used to choose the final move
| Mean value $Q(s,a)$ | Average backed-up value of an edge, from the view of the player who chose it
| Prior $P(s,a)$ | The policy network's probability for the move, fixed when the node is expanded
| PUCT | Selection rule $Q+c_\text{puct}P\sqrt{\sum_bN}/(1+N)$: mean value plus a prior-weighted bonus that fades with visits
| Rollout | Classic leaf evaluation by playing the game out with fast moves; not used by AlphaGo Zero
| Dirichlet noise | Random spiky probabilities mixed into the root priors in self-play so every move sometimes gets searched
| Virtual loss | Temporary pessimistic update that keeps parallel simulations on different paths

<div class="take">
<div class="lab">Key takeaway</div>

MCTS repeats select, expand, evaluate and back up; AlphaGo Zero selects with PUCT, which adds a prior-weighted, visit-decaying bonus to each edge's mean value, evaluates new leaves with the value head instead of rollouts, flips value signs on the way up, and plays from root visit counts.

</div>
