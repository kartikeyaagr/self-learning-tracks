---
id: "09.04"
course: "deep-learning"
module: "09"
slug: "the-alphago-zero-training-loop"
title: "The AlphaGo Zero training loop"
keyIdeas: ["AlphaGo Zero is a loop: the best network plays itself with MCTS, every position becomes a training example $(s, \\pi, z)$, the network trains on a window of recent games, and an evaluator decides which network plays next.", "$\\pi$ is the share of root visits each move got in the search (the policy target) and $z$ is the final result from the view of the player to move (the value target). Both are ordinary supervised labels.", "Search is a policy improvement step and training copies the improvement into the network: a sampled, neural version of policy iteration (also called expert iteration). If the search is barely better than the network, there is nothing to learn.", "Variety comes from temperature ($\\tau = 1$ for the first 30 moves, then play the most-visited move) and Dirichlet noise at the root; stability comes from a 500,000-game replay window, random board symmetries and a 55%-of-400-games promotion gate.", "Judge the loop by playing strength, not training loss: on tic-tac-toe a 4-simulation run reached a lower loss than a 64-simulation run yet still lost 70% of games to a perfect player, while the 64-simulation run lost 1.5%."]
question: "To save compute, you cut self-play to 50 MCTS simulations per move while keeping everything else the same. Training loss goes down nicely, but playing strength plateaus early. Use the policy-iteration view of the loop to explain why, say which diagnostics would have shown the problem before you looked at strength, and say what you would change if compute really is limited."
answer: "The loop improves only as far as search improves on the network. With 50 simulations on a 362-move board, the search barely explores beyond the network's top few priors, so the visit distribution $\\pi$ is close to $p$ itself. Training $p$ towards $\\pi$ then teaches the network what it already believes: the policy improvement step has almost no effect, and the loss falls because the target is easy, not because play improves. Value targets also come from weak, near-raw-network games, so $v$ learns to evaluate a weak policy, and with little diversity it can even memorise the results of repetitive games (in the lesson's tic-tac-toe run with 4 simulations the value loss went to almost zero while the network still lost 70% of games to a perfect player). Diagnostics that would have flagged it: agreement between the raw network's top move and the search's top move near 100% (search has nothing left to teach); flat or falling Elo against fixed earlier checkpoints; shrinking opening diversity and repetitive games; and a policy cross-entropy that is low simply because $\\pi$ is nearly as flat or as peaked as $p$. With limited compute, keep a strong improvement signal where it matters: KataGo's playout-cap randomisation runs a full search (say 600 or more simulations) on a random fraction of moves and records only those as policy targets, while the remaining moves use a cheap search just to keep the game going. Also reduce other costs first: a smaller network, resignation with false-positive checks (10 percent of games played out), symmetry augmentation and a larger replay window to get more value out of every game played."
sources: [{"title": "Silver et al.: Mastering the game of Go without human knowledge", "url": "https://www.nature.com/articles/nature24270", "note": "AlphaGo Zero; Methods describe self-play, the replay buffer, evaluation gating, temperature and resignation."}, {"title": "Silver et al.: Mastering Chess and Shogi by Self-Play (AlphaZero)", "url": "https://arxiv.org/abs/1712.01815", "note": "The same loop without gating, applied to three games."}, {"title": "Anthony, Tian and Barber: Thinking Fast and Slow with Deep Learning and Tree Search", "url": "https://arxiv.org/abs/1705.08439", "note": "Expert iteration: search as the expert, the network as the apprentice."}, {"title": "Wu: Accelerating Self-Play Learning in Go (KataGo)", "url": "https://arxiv.org/abs/1902.10565", "note": "Playout-cap randomisation and other changes that cut self-play compute about 50x."}]
practice: ["turn-root-visits-into-search-policy", "sample-recent-self-play-replay-batch", "apply-dihedral-go-symmetry", "mix-dirichlet-noise-into-root-priors", "alphago-policy-value-loss"]
tags: ["self-play", "expert iteration", "AlphaGo Zero"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- How AlphaGo Zero joins game rules, MCTS and a policy-value network into one loop that teaches itself from random weights.
- What a self-play training example $(s, \pi, z)$ is and where each part comes from.
- How temperature, root noise, a replay window and symmetries keep the loop exploring.
- Why the loop improves: search is a policy improvement step, and training copies the improvement into the network.
- How to run the whole loop on tic-tac-toe in a minute and read its diagnostics.
- What breaks in practice: too little search, a short replay window, unsafe resignation.

*Before you start:* you need the rules and board encoding from 09.01, PUCT search with root noise from 09.02, and the two-headed network and its loss from 09.03. The "why it works" section compares the loop with policy iteration from lesson 08.02; a rough memory of "evaluate the policy, then make it greedier, repeat" is enough.

The last three lessons built the parts: exact rules (09.01), a guided search (09.02) and a network that outputs move probabilities and a win estimate (09.03). With random weights they play badly. This lesson connects them into the loop DeepMind called AlphaGo Zero. Starting from random weights and the rules alone, with no human games, the 20-block version trained for three days and then beat AlphaGo Lee (the version that defeated Lee Sedol) 100 games to 0.

## The idea in one picture

Go is far too expensive for a laptop, so we run the same loop on tic-tac-toe, where perfect play can be computed exactly to measure progress. One **round** of the loop is: play 60 games of the network against itself with 64 MCTS simulations per move, then take 200 training steps on the most recent games. After each round we pit the raw network (no search, just its top move) against a perfect player and count how often it loses.

| Round | Games lost to a perfect player | Picks a best move in critical positions | Self-play games drawn
| --- | --- | --- | --- |
| 0 (random weights) | 78% | 47% | (no games yet)
| 1 | 34% | 70% | 38%
| 2 | 5% | 76% | 72%
| 16 | 1.5% | 86% | 87%

A **critical position** here is one where at least one legal move throws away the result, so picking a best move actually matters. Perfect tic-tac-toe is a draw, so a rising draw rate in self-play is exactly what you expect as both sides improve.

~~~dg-fig 09.04-strength.svg
One run per setting on tic-tac-toe. With 64 or 16 simulations per self-play move the network goes from losing most games to almost never losing within a few rounds. With only 4 simulations it stays weak, even though it trains on just as much data.
~~~

Nobody told the network a single good move: everything it knows came from games against itself, labelled by a search slightly better than the network. The orange curve shows the catch: if the search is no better than the network, there is nothing to learn. Terms used from here on: **self-play** (the network plays both sides, MCTS choosing every move), the **search policy** $\pi$ (each move's share of root visits, the policy target), the **outcome** $z$ ($+1$ win, $-1$ loss, 0 draw, from the view of the player to move) and the **replay buffer** (recent self-play positions that training samples from).

## Step 1: self-play turns search into training examples

The best network so far plays a full game against itself. At every position $s_t$:

1. Run MCTS from $s_t$, guided by the network's priors $p$ and values $v$. AlphaGo Zero used 1,600 simulations, about 0.4 seconds of thinking per move.
2. Read off the root visit counts $N(s_t, a)$ and normalise them into $\pi_t(a) = N(s_t,a) / \sum_b N(s_t,b)$. Store the pair $(s_t, \pi_t)$.
3. Choose the move to play from the visit counts (step 2 explains exactly how), play it, and continue.

When the game ends, every stored position gets the outcome label $z_t$, from the point of view of the player to move at $s_t$. Because players alternate, the labels alternate in sign along the game.

<div class="ex">
<div class="lab">Worked example</div>

**labelling one game**

X and O play seven moves of tic-tac-toe and X completes a line with the seventh move. The stored positions are $s_1, \dots, s_7$, the positions just before each move.

X was to move at $s_1, s_3, s_5, s_7$, and X won, so those four get $z = +1$.

O was to move at $s_2, s_4, s_6$, and O lost, so those three get $z = -1$.

In code, start from the end: the final position (O to move, O has lost) has value $-1$; walking backwards, flip the sign at every ply: $s_7 \to +1$, $s_6 \to -1$, $s_5 \to +1$, and so on. A sign error here makes the network learn to lose.

</div>

No policy gradient is computed: both targets are plain supervised labels.

## Step 2: choosing the move to play

Always playing the most-visited move would make a deterministic network replay the same game. Two mechanisms create variety.

**Temperature.** The move is sampled from

$$\pi_\tau(a) = \frac{N(s_0,a)^{1/\tau}}{\sum_b N(s_0,b)^{1/\tau}},$$

where $s_0$ is the root and $\tau > 0$ the **temperature**. With $\tau = 1$ moves are sampled in proportion to visits; as $\tau \to 0$ the rule becomes "play the most-visited move". AlphaGo Zero used $\tau = 1$ for the first 30 moves, so openings vary, and $\tau \to 0$ afterwards, so the rest of the game reflects the search's real judgement and $z$ is a meaningful label.

~~~dg-fig 09.04-temperature.svg
The same visit counts (800, 600, 200) turned into move probabilities at four temperatures. Lowering the temperature moves probability from the less-visited moves onto the most-visited one.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the effect of temperature**

After 1,600 simulations three candidate moves have visit counts $(800, 600, 200)$.

$\tau = 1$: divide by the total 1,600: $(0.50,\ 0.375,\ 0.125)$. The third move is still played one time in eight.

$\tau = 0.5$: raise the counts to the power $1/0.5 = 2$: $(640{,}000,\ 360{,}000,\ 40{,}000)$, total $1{,}040{,}000$. Dividing gives $(0.615,\ 0.346,\ 0.038)$.

$\tau = 0.25$: fourth powers $(4.10,\ 1.30,\ 0.016) \times 10^{11}$, total $5.41 \times 10^{11}$, so $(0.757,\ 0.240,\ 0.003)$.

$\tau \to 0$: the first move is played every time. Evaluation games and real matches use this throughout; self-play uses $\tau = 1$ only in the opening.

</div>

The training target stays the $\tau = 1$ distribution $\pi_t$: temperature decides which game gets played, not what is taught.

**Root noise** (09.02): $P(s_0,a) = 0.75\,p_a + 0.25\,\eta_a$ with $\eta \sim \text{Dir}(0.03)$ for Go. Temperature varies which move is *played*; noise varies which moves are *searched*, so a move with a near-zero prior still gets explored and a wrong prior can be corrected.

## Step 3: train on a window of recent games

Training runs in parallel with self-play, reading from the replay buffer.

- **What to sample.** Positions uniformly from the most recent 500,000 games. A long window mixes many past versions of the network, keeping data diverse and preventing forgetting.
- **Symmetry.** Go's rules ignore rotation and mirroring, so each sampled position (with its $\pi$) gets one of the 8 symmetries at random: up to eight times the data for free.
- **Loss** (09.03): $\ell = (z - v)^2 - \pi^\top \log p + c\,\|\theta\|^2$ with $c = 10^{-4}$, where $p, v$ are the network's outputs on $s$.
- **Optimiser.** Minibatches of 2,048 positions, SGD with momentum 0.9 and a learning rate stepped down twice during training. A checkpoint is saved every 1,000 steps.

## Step 4: decide which network plays next

Each checkpoint is evaluated against the current best network over 400 games, with 1,600 simulations per move, $\tau \to 0$ and no root noise. If it wins at least 55% (220 games), it becomes the new best network and from then on generates the self-play games. Otherwise it is discarded as a self-play player (training simply continues from it).

~~~dg-flow Three processes run continuously and asynchronously; the evaluator gates which weights generate data.
Best network -(MCTS self-play)-> Replay buffer of recent games -(sample minibatches)-> *Training -(checkpoint every 1,000 steps)-> Evaluator -(wins 55 percent of 400?)-> Best network
~~~

The gate is insurance: a worse network never generates data. The price is that improvements arrive in jumps. AlphaZero (which also learned chess and shogi) dropped the gate, always used the latest weights and still trained well.

For scale: the 20-block AlphaGo Zero run used 4.9 million self-play games and 700,000 minibatches (about 1.4 billion sampled positions); the 40-block run used 29 million games over about 40 days.

## Why it works: search improves, training copies

Policy iteration (08.02) alternates evaluating a policy and making it greedier. AlphaGo Zero is a sampled, neural version.

- **Improvement.** MCTS guided by $(p, v)$ looks ahead and corrects the network's local mistakes, so its visit distribution $\pi$ is a stronger policy than $p$. Search is a **policy improvement operator**.
- **Distillation.** Training $p$ towards $\pi$ copies that improvement into the network, so next round the network starts where the search ended.
- **Evaluation.** Training $v$ towards $z$ evaluates the improved player, because $z$ comes from games played with search.

~~~dg-fig 09.04-improve.svg
One tic-tac-toe position where the player to move must block on square 7. The untrained network spreads its prior almost evenly over the five empty squares; the search's visit share on square 7 grows from 50% at 32 simulations to 93% at 512. That gap between p and π is what training teaches.
~~~

Even a random network improves here: the search sees that any move but 7 lets the opponent win two plies later, knowledge that comes from exact terminal values, not from the network. Early signal comes from such short tactics near game ends; as $v$ improves, the improvement spreads backwards through the game.

The same idea was published independently as **expert iteration** (Anthony, Tian and Barber, 2017): search is the expert, the network the apprentice. Supervised targets avoid the high variance of policy gradients on win/loss rewards; despite its actor-critic-like network, the method is closer to generalised policy iteration than to lesson 08.06.

## Saving compute: resignation

Many self-play games are decided long before the end, and playing them out wastes simulations. AlphaGo Zero resigns when both the root value and the best child's value fall below a threshold $v_\text{resign}$. To choose that threshold safely, resignation is switched off in 10% of games, which are played to the end. $v_\text{resign}$ is set so that fewer than 5% of those games would have been resigned by a player who could still have won (a **false positive**).

Without the check, a value head wrong about some kind of position resigns it every time, no game produces the $z$ that would correct it, and the error becomes permanent.

## The loop in code

The complete loop on tic-tac-toe in PyTorch: rules, PUCT with root noise, self-play with temperature, a replay window, symmetries and the policy-value loss. It skips the gate (like AlphaZero) and measures progress against a perfect minimax player. About a minute on one CPU thread.

```python
import math, functools, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); torch.manual_seed(0); rng = np.random.default_rng(0)
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
# A board is a tuple of 9 cells from the mover's view: +1 = mine, -1 = opponent's, 0 = empty.
def result(s):        # value for the player to move, or None if the game goes on
    if any(s[a] + s[b] + s[c] == -3 for a, b, c in LINES): return -1.0   # opponent just won
    return 0.0 if 0 not in s else None
def play(s, a):       # place my stone, then flip the board so the next mover is +1
    return tuple(-x for x in s[:a] + (1,) + s[a+1:])

net = nn.Sequential(nn.Linear(18, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
def enc(S): S = np.asarray(S); return torch.tensor(np.c_[S == 1, S == -1], dtype=torch.float32)

def search(s, sims, f, c=1.5, eps=0.25, alpha=0.3):     # PUCT (09.02); f(s) -> (priors, value)
    N, W, P = {}, {}, {}
    def visit(u):
        if (r := result(u)) is not None: return r
        if u not in N:                                    # leaf: expand with the network
            P[u], v = f(u); N[u], W[u] = np.zeros(9), np.zeros(9); return v
        q = np.where(N[u] > 0, W[u] / np.maximum(N[u], 1), 0)
        score = q + c * P[u] * math.sqrt(N[u].sum() + 1) / (1 + N[u])
        a = int(np.where(np.array(u) == 0, score, -np.inf).argmax())
        v = -visit(play(u, a)); N[u][a] += 1; W[u][a] += v; return v
    visit(s); legal = np.array(s) == 0                    # expand root, then mix in root noise
    noise = np.zeros(9); noise[legal] = rng.dirichlet([alpha] * legal.sum())
    P[s] = (1 - eps) * P[s] + eps * noise
    for _ in range(sims): visit(s)
    return N[s]

def self_play_game(f, sims, explore_moves=2):
    s, hist = (0,) * 9, []
    while result(s) is None:
        n = search(s, sims, f); hist.append((s, n / n.sum()))       # target pi = visit share
        a = rng.choice(9, p=n / n.sum()) if len(hist) <= explore_moves else n.argmax()
        s = play(s, int(a))
    z, out = result(s), []
    for st, pi in reversed(hist): z = -z; out.append((st, pi, z))   # z from each mover's view
    return out

@functools.lru_cache(None)
def perfect(s):       # exact minimax value, only used to measure progress
    r = result(s)
    return r if r is not None else max(-perfect(play(s, a)) for a in range(9) if s[a] == 0)
def losses_vs_perfect(games=200):   # raw network, no search, against a perfect player
    lost = 0
    for g in range(games):
        s, me = (0,) * 9, g % 2 == 0
        while result(s) is None:
            legal = [a for a in range(9) if s[a] == 0]
            if me: a = max(legal, key=lambda a: net(enc([s]))[0, a].item())
            else: a = rng.choice([a for a in legal if -perfect(play(s, a)) == perfect(s)])
            s, me = play(s, int(a)), not me
        lost += me and result(s) == -1                # the net is to move and has lost
    return lost / games

g = np.arange(9).reshape(3, 3)                        # the 8 symmetries of the board
SYM = [np.rot90(m, r).ravel() for m in (g, g.T) for r in range(4)]
buffer = []
for it in range(10):
    @functools.lru_cache(None)                        # this round's network, frozen for self-play
    def f(s):
        with torch.no_grad(): o = net(enc([s]))[0]
        p = torch.softmax(o[:9].masked_fill(torch.tensor(s) != 0, -1e9), 0)
        return p.numpy(), math.tanh(o[9].item())
    games = [self_play_game(f, sims=64) for _ in range(60)]
    buffer = (buffer + games)[-240:]                  # replay window: last 240 games
    data = [ex for g in buffer for ex in g]
    for step in range(200):
        S, PI, Z = zip(*[data[i] for i in rng.integers(len(data), size=128)])
        perm = SYM[rng.integers(8)]                   # random rotation/reflection of the batch
        S, PI = np.array(S)[:, perm], np.array(PI)[:, perm]
        out = net(enc(S)); legal = torch.tensor(S == 0)
        logp = torch.log_softmax(out[:, :9].masked_fill(~legal, -1e9), 1)
        l_pol = -(torch.tensor(PI, dtype=torch.float32) * logp).sum(1).mean()
        l_val = ((torch.tensor(Z, dtype=torch.float32) - torch.tanh(out[:, 9])) ** 2).mean()
        opt.zero_grad(); (l_pol + l_val).backward(); opt.step()
    print(f"round {it}: policy CE {l_pol.item():.2f}  value MSE {l_val.item():.3f}  "
          f"lost to perfect player {losses_vs_perfect():.0%}")
# round 0: policy CE 1.44  value MSE 0.269  lost to perfect player 48%
# round 2: policy CE 1.40  value MSE 0.142  lost to perfect player 10%
# round 4: policy CE 1.03  value MSE 0.089  lost to perfect player 4%
# round 9: policy CE 0.85  value MSE 0.035  lost to perfect player 0%   (other rounds omitted)
```

The losses jump around because each round trains on a new mix of games; only the last column measures strength. For Go, every line keeps its job; the network (a ResNet), simulations (1,600), window (500,000 games) and hardware change.

## Reading the result

Training loss is the first number you see and the least informative.

~~~dg-fig 09.04-diagnostics.svg
The 64-simulation run (blue) and the 4-simulation run (orange). Left: both training losses fall, and the weak-search run even ends lower (0.86 against 0.92). Middle and right: only the run with real search keeps getting better moves and drives self-play towards the draw that perfect play produces.
~~~

With 4 simulations, $\pi \approx p$, so the policy target is easy to fit; the value loss fell to almost zero because repetitive weak games are easy to memorise. A low loss on self-generated data only says the network agrees with itself. Monitor instead:

- **Strength against fixed opponents** (earlier checkpoints, a reference engine, a perfect player): an Elo curve.
- **Search-network agreement**: if the raw network's top move almost always equals the search's, search has nothing left to teach.
- **Self-play statistics**: game length, draw or resignation rate, opening diversity.
- **Both loss terms separately** on held-out recent games.

## What goes wrong

| Problem | Symptom | Fix
| --- | --- | --- |
| Too few simulations | loss falls, strength plateaus (our 4-sim run still lost 70% at round 16) | more simulations, or KataGo's playout-cap randomisation: full search on a random subset of moves, used as policy targets
| Replay window too short | strength cycles: version B beats A, C beats B, A beats C | keep many past games; sample uniformly over a long window
| No symmetry augmentation | slower learning, lopsided play on mirrored positions | random rotation or reflection per sample (removing it, our code lost 24% at round 9 instead of 0%; one run)
| Resignation without checks | blind spots that never get fixed | play 10% of games to the end and cap false positives at 5%
| Sign error in $z$ or in the search backup | the network gets worse with training; it seems to try to lose | test on a position with a forced win in one
| Noise too strong or temperature kept at 1 all game | $z$ labels reflect random moves, value targets are biased | $\tau \to 0$ after the opening; noise only at the root
| Bug in the rules | the network finds and exploits it, for example illegal ko recaptures | test the rules engine separately (09.01)

## Going deeper

**How strict is the 55% gate?** Treat each evaluation game as an independent coin flip with the candidate's true win probability $p$ (Go with komi has no draws). With 400 games the win fraction has standard error $\sqrt{0.5 \times 0.5 / 400} = 0.025$ when $p = 0.5$, so 55% is two standard errors above even.

~~~dg-fig 09.04-gating.svg
Probability that a candidate is promoted, for 40, 100 and 400 evaluation games. With 400 games a network no better than the current one passes only 2.6% of the time, while a clearly better one (60% true win rate) passes 98% of the time.
~~~

Exact binomial values: an equal candidate passes 2.6% (400 games), 18% (100) or 32% (40) of the time. A true 55% candidate passes only about half the time, so small gains must accumulate before promotion: a conservative but sensible trade.

**Non-transitivity.** B can beat A and C beat B while A beats C. Training only against the latest self can chase such cycles; long windows and gating damp them, and AlphaStar later trained against leagues of past opponents for this reason.

**What changed in successors.**

| Choice | AlphaGo Zero | AlphaZero and later engines
| --- | --- | --- |
| Which network plays self-play | best checkpoint, gated at 55 percent | always the latest weights (AlphaZero)
| Simulations per move | 1,600 | 800 (AlphaZero); randomised caps (KataGo)
| Dirichlet $\alpha$ | 0.03 for Go | scaled to the typical number of legal moves (0.3 for chess)
| Targets | $\pi$ and $z$ | plus ownership, score and other auxiliary targets (KataGo)

KataGo reported roughly a 50x compute reduction versus AlphaGo Zero's approach from auxiliary targets, playout-cap randomisation and architecture changes. MuZero kept the loop but replaced the rules inside the search with a learned model.

**No guarantee.** Exact policy iteration converges on finite problems; the neural version, with approximation, finite search and shifting data, has no such theorem. It works in practice because each piece is conservative.

## Key terms

| Term | Meaning
| --- | --- |
| Self-play | The network plays both sides, MCTS choosing every move, to generate data
| Search policy $\pi$ | Root visit counts after MCTS, normalised; the target for the policy head
| Outcome $z$ | Final game result from the view of the player to move at that position
| Temperature $\tau$ | How moves are sampled from visits: 1 proportional, near 0 most visited
| Replay buffer | Recent self-play positions that training samples from (500,000 games)
| Evaluation gating | Promote a checkpoint only if it wins 55% of 400 games against the best
| Policy improvement operator | A procedure (here MCTS) that turns a policy into a better one
| Expert iteration | Alternating search (expert) and training the network to imitate it (apprentice)
| Resignation threshold | Value below which a self-play game is abandoned, checked with games played to the end

<div class="take">
<div class="lab">Key takeaway</div>

AlphaGo Zero alternates MCTS self-play, which turns the network's policy into a stronger search policy, with supervised training that copies that policy into the network and learns the game outcomes; temperature, root noise, a long replay window, symmetries, gated evaluation and checked resignation keep the loop exploring and improving, and its progress must be judged by strength, never by training loss.

</div>
