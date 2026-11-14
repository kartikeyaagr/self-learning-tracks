---
id: "09.02"
course: "deep-learning"
module: "09"
slug: "monte-carlo-tree-search"
title: "Monte Carlo tree search"
keyIdeas: ["MCTS grows a search tree selectively: each simulation selects a path with a bandit rule, expands one leaf, evaluates it, and backs the value up the path.", "Each edge stores a visit count $N$, total value $W$, mean value $Q=W/N$ and prior $P$; values are always from the perspective of the player to move, so they flip sign at each level.", "AlphaGo Zero selects with PUCT: $a=\\arg\\max_a Q(s,a)+c_\\text{puct}P(s,a)\\frac{\\sqrt{\\sum_bN(s,b)}}{1+N(s,a)}$, an optimism bonus weighted by the network's prior.", "There are no random rollouts in AlphaGo Zero: a leaf is evaluated once by the value head, and the policy head supplies the priors for its children.", "Dirichlet noise on the root priors guarantees exploration in self-play; the final move is chosen from root visit counts, not from $Q$."]
question: "A teammate's MCTS converts each leaf value to the root player's perspective once and adds that same number to every edge on the path; selection at every node then maximises $Q+U$. Search looks fine when it only expands the root's children, but deeper searches approve moves that lose to a simple reply. What game is the search actually solving, and why does the depth-1 case hide it?"
answer: "At nodes where the opponent is to move, the edges now store the root player's value, and selection picks the edge that maximises it. The search therefore models an opponent who chooses the reply that is best for the root player: it is solving a cooperative game, not minimax. A move whose only refutation is one strong reply looks excellent, because the search assumes the opponent will never play that reply. With only the root's children expanded, every selection happens at a root-player node, where maximising the root player's value is correct, so the bug is invisible; it appears as soon as the tree contains opponent decisions. The fix is to store each edge's value from the perspective of the player who chose it: back up $v$ from the leaf player's view and negate it at every level, so that each node's maximisation is that node's player looking after their own interests."
sources: [{"title": "Silver et al.: Mastering the game of Go without human knowledge", "url": "https://www.nature.com/articles/nature24270", "note": "AlphaGo Zero; the Methods section gives the exact select, expand, backup and play rules, PUCT and the root noise."}, {"title": "Silver et al.: Mastering Chess and Shogi by Self-Play (AlphaZero)", "url": "https://arxiv.org/abs/1712.01815", "note": "The same search for chess and shogi, with how the Dirichlet alpha scales with the number of legal moves."}, {"title": "Chess Programming Wiki: UCT", "url": "https://www.chessprogramming.org/UCT", "note": "UCT as bandits in a tree, with history and variants such as first-play urgency."}]
practice: ["select-action-with-puct", "run-deterministic-neural-guided-mcts", "back-up-mcts-simulation-path", "initialize-legal-mcts-edge-statistics", "mix-dirichlet-noise-into-root-priors"]
tags: ["MCTS", "PUCT"]
tools: ["NumPy"]
---

## Search without a full tree

Lesson 09.01 showed that full-width minimax is hopeless for Go. **Monte Carlo tree search** builds only the part of the tree that looks relevant, spending more simulations on promising lines and fewer on bad ones, and converges towards minimax values where it searches deeply. The idea that made it work, **UCT** (Kocsis and Szepesvari, 2006), is to treat move choice at every node as a multi-armed bandit (lesson 08.05): pick the child with the best upper confidence bound. Early Go programs evaluated leaves with random playouts to the end of the game, which reached strong amateur level. AlphaGo replaced playouts with neural networks; AlphaGo Zero dropped playouts entirely.

## What each edge stores

For every edge $(s,a)$ in the tree:

- $N(s,a)$: how many simulations took this edge.
- $W(s,a)$: sum of the values backed up through it; $Q(s,a)=W/N$ is the mean.
- $P(s,a)$: the prior probability from the policy head, computed once when $s$ was expanded.

## One simulation, four phases

1. **Select.** Start at the root and repeatedly pick the child edge maximising the selection score until reaching an unexpanded node.
2. **Expand.** Run the network on that leaf position to get a policy $p$ (362 move probabilities) and a value $v\in[-1,1]$. Mask illegal moves, renormalise, and create child edges with $P=p$, $N=W=0$.
3. **Evaluate.** The leaf's value is $v$, from the perspective of the player to move at the leaf. A terminal position is scored by the rules instead.
4. **Backup.** Walk back up the path, adding the value to each edge's $W$ and 1 to its $N$, **negating the value at each level**, since a position good for the player to move is bad for the player who moved into it.

~~~dg-seq One AlphaGo Zero simulation: no rollouts; the network is called once per simulation, at the new leaf.
Search > Tree : select by PUCT from the root
Tree -> Search : path to an unexpanded leaf
Search > Network : evaluate leaf position
Network -> Search : priors p, value v
Search > Tree : expand leaf with legal priors
Search > Tree : back up v, flipping sign per level
~~~

## The PUCT rule

AlphaGo Zero selects with a variant of the predictor-UCB rule:

$$a^*=\arg\max_a\Big[Q(s,a)+c_\text{puct}\,P(s,a)\,\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}\Big]$$

The first term exploits what the search has found. The second is an exploration bonus that is large for moves the network likes (high $P$) and has not tried much (small $N$), and that shrinks as the move is visited. Compare UCB1's $\sqrt{\ln t/N}$: PUCT decays faster in $N$ and is shaped by the prior, so the network decides where to look and the search decides what is actually good. $c_\text{puct}$ is a tuned constant; public reimplementations use values around 1 to 5, sometimes growing slowly with visit count. Unvisited edges start with $Q=0$ in AlphaGo Zero; later engines use a "first play urgency" value somewhat below the parent's $Q$ so that search does not waste visits on every low-prior move.

<div class="ex">
<div class="lab">Worked example</div>

**one PUCT decision**

Root with 100 total visits ($\sqrt{100}=10$), $c_\text{puct}=1.5$.

Move A: $Q=0.50$, $P=0.20$, $N=40$. Bonus $=1.5\cdot0.20\cdot10/41=0.073$; score $0.573$.

Move B: $Q=0.30$, $P=0.50$, $N=10$. Bonus $=1.5\cdot0.50\cdot10/11=0.682$; score $0.982$.

B is searched next despite its lower mean: the network rates it highly and it has few visits. If B's $Q$ stays at 0.3, its bonus shrinks with every visit until A takes over.

</div>

```python
def select(node, c_puct):
    sqrt_total = math.sqrt(sum(e.N for e in node.edges))
    return max(node.edges, key=lambda e: e.Q + c_puct * e.P * sqrt_total / (1 + e.N))

def backup(path, v):                  # path: edges from root to leaf; v from the leaf player's view
    for edge in reversed(path):
        v = -v                         # the edge's mover is the opponent of whoever moves next
        edge.N += 1; edge.W += v; edge.Q = edge.W / edge.N
```

## Root noise, move choice and practical details

- **Dirichlet noise.** In self-play, the root priors are mixed with noise: $P(s_0,a)=(1-\varepsilon)p_a+\varepsilon\eta_a$, $\eta\sim\mathrm{Dir}(\alpha)$, with $\varepsilon=0.25$ and $\alpha=0.03$ for Go. A small $\alpha$ concentrates the noise on a few random moves, so every move occasionally gets a real look even if the network gives it near-zero prior. Noise is applied at the root only and not in competitive play.
- **Choosing the move.** After all simulations (1,600 per move in AlphaGo Zero), the move is chosen from root visit counts, $\pi(a)\propto N(s_0,a)^{1/\tau}$. Visit counts are robust: a move with a high $Q$ from only a few visits might be a lucky estimate, while a high $N$ means the search kept confirming it. Lesson 09.04 covers the temperature $\tau$.
- **Tree reuse.** After a move is played, the subtree below it becomes the new root, keeping its statistics.
- **Parallelism.** Many simulations run concurrently, and leaf evaluations are batched on the GPU (AlphaGo Zero batched 8). A **virtual loss**, temporarily counting an in-flight path as a loss, stops parallel threads from all descending the same line.

<div class="take">
<div class="lab">Key takeaway</div>

MCTS repeats select, expand, evaluate and back up; AlphaGo Zero selects with PUCT, which adds a prior-weighted, visit-decaying bonus to each edge's mean value, evaluates new leaves with the value head instead of rollouts, flips value signs on the way up, and plays from root visit counts.

</div>
