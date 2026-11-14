---
id: "09.01"
course: "deep-learning"
module: "09"
slug: "game-trees-and-the-rules-of-go"
title: "Game trees and the rules of Go"
keyIdeas: ["Go is two-player, zero-sum, deterministic and has perfect information, so **minimax** could in principle solve it. In practice its tree (about 250 moves per turn over about 150 moves, roughly $10^{360}$ sequences) is far too big, and there is no good hand-written way to score a position.", "Stones joined horizontally or vertically form a **group**; its **liberties** are the distinct empty points next to it. A group with no liberties is captured and removed.", "A move is processed in a fixed order: place the stone, remove opponent groups with no liberties, then reject the move if your own group has none (suicide) or if the board repeats an earlier position (**superko**). Checking suicide before captures is the classic bug.", "Two passes in a row end the game. **Area scoring** gives each player their stones plus the empty regions that touch only their colour, and White gets **komi** (7.5 in AlphaGo Zero) for moving second.", "For a learning system the board is a 361-cell array, the action space has 362 entries (every point plus pass), groups and liberties come from flood fill, and Zobrist hashing makes the repetition check a single set lookup."]
question: "Your Go engine sometimes reports that a move capturing an opponent group is illegal as suicide. The bug report shows the placed stone had no empty neighbours at the moment it was placed. Explain the bug in terms of the order of rule checks, and explain why this specific order is what the rules require."
answer: "The engine checks the new stone's liberties before removing captured opponent groups. When a stone is placed into a point surrounded by opponent stones, its group may have zero liberties at that instant, but if that placement takes the last liberty of an adjacent opponent group, that group is removed first, which empties points next to the new stone and gives it liberties. The rules define a move as: place the stone, remove all opponent groups with no liberties, then check your own group. Only if your group still has no liberties after captures is the move suicide (illegal under Chinese rules; under Tromp-Taylor rules it would instead remove your own group). Capturing therefore always takes precedence, which is what makes moves that fill an opponent's last liberty from 'inside' legal; the ko shape is the everyday example, where the capturing stone has a single liberty only because the captured stone has just left. Fix the order: place, scan the up-to-four adjacent opponent groups and remove any with no liberties (XOR-ing each removed stone out of the Zobrist hash), then flood-fill your own group and reject only if it has no liberties, then run the superko check on the resulting board. Add a regression test: a ko capture on a 5×5 board must be legal, and the immediate recapture must be rejected by superko, not by the suicide check."
sources: [{"title": "John Tromp: The Game of Go (Tromp-Taylor rules)", "url": "https://tromp.github.io/go.html", "note": "A complete, machine-friendly statement of the rules in ten short lines, with area scoring."}, {"title": "Sensei's Library: Superko", "url": "https://senseis.xmp.net/?Superko", "note": "Positional versus situational superko and why repetition rules exist."}, {"title": "Silver et al.: Mastering the game of Go without human knowledge", "url": "https://www.nature.com/articles/nature24270", "note": "AlphaGo Zero; the Methods section specifies the rules, komi and the 17-plane input encoding."}, {"title": "Chess Programming Wiki: Zobrist Hashing", "url": "https://www.chessprogramming.org/Zobrist_Hashing", "note": "Incremental XOR position hashing as used for repetition detection."}]
practice: ["apply-go-action-with-captures-and-repetition", "find-go-group-and-its-liberties", "build-legal-go-action-mask", "score-terminal-position-with-tromp-taylor-rules", "encode-alphago-zero-board-history"]
tags: ["game trees", "Go rules"]
tools: ["NumPy"]
---

## What you will learn

- What a game tree is, and how **minimax** finds perfect play by backing values up from the leaves.
- Why that recipe breaks down for Go: the tree is too wide and positions are too hard to score by hand.
- The complete rules of Go as a programmer needs them: groups, liberties, capture, suicide, ko, superko and area scoring.
- The order in which a move is processed, and the classic bug of checking suicide before captures.
- How a position is stored for learning: a flat array, a 362-action mask, flood fill and Zobrist hashes.

*Before you start:* you should know what a state, an action and a reward are (lesson 08.01) and what an exploration bonus does in a bandit (08.05). No knowledge of Go is assumed.

## Where this module starts

This module builds the AlphaGo Zero algorithm from the ground up: the game and its rules (this lesson), Monte Carlo tree search (09.02), the policy-value network that guides it (09.03), and the self-play training loop that ties them together (09.04). It leans on module 08 throughout: MCTS is a bandit (08.05) run at every node of a tree, the network is a two-headed actor-critic-style model (08.06), and the training loop is a form of policy iteration (08.02).

Everything later needs a correct **simulator**: a function that takes a position and a move and returns the next position, plus the list of legal moves. Search calls it millions of times, and one rule bug silently teaches the network nonsense. So this lesson is half "why is Go hard" and half "write the rules down exactly".

## The idea in one picture

Picture every possible game as a tree: the root is the empty board, each branch a move, each node a position, each leaf a finished game. With the whole tree you could play perfectly by looking ahead to the leaves.

How big is that tree? If a game offers about $b$ moves per turn (the **branching factor**) and lasts about $d$ moves (the **depth**, counted in **plies**, single moves by one player), the number of move sequences is roughly $b^d$:

| Game | Branching factor $b$ | Typical length $d$ | Rough tree size $b^d$
| --- | --- | --- | --- |
| Tic-tac-toe | about 4 on average | at most 9 plies | 255,168 complete games (exact count)
| Chess | about 35 | about 80 plies | $35^{80}\approx10^{123}$
| Go 19×19 | about 250 | about 150 plies | $250^{150}\approx10^{360}$

~~~dg-fig 09.01-tree-size.svg
The log of b to the power d grows in a straight line with depth, and the slope is log b. Go's line is steeper than chess's and runs nearly twice as long, so its tree is more than 200 orders of magnitude larger than chess's.
~~~

Each extra ply adds $\log_{10} b$ digits. Chess passes the number of atoms in the observable universe (about $10^{80}$, a standard published estimate) after about 52 plies; Go after 34. The branching factors and lengths in the table are the standard figures quoted in the AlphaGo papers.

You do not need the whole tree: you need to look at the *right* small part of it, and a good guess for the value of any position where you stop. AlphaGo Zero learns both.

## Step 1: minimax on a game tree

Go, chess and tic-tac-toe are **two-player**, **zero-sum** (one side's win is the other's loss), **deterministic** (no dice) games of **perfect information** (both see the whole board).

For such games a simple rule finds perfect play. Score each leaf from Black's point of view (+1 Black wins, −1 White wins). Then work upwards: where Black is to move, take the child with the **largest** value; where White is to move, the **smallest**. This is **minimax**, and the root's value is the result under perfect play.

$$V(s) = \begin{cases} \text{result}(s) & s \text{ is finished} \\ \max_{a} V(\text{next}(s, a)) & \text{Black to move} \\ \min_{a} V(\text{next}(s, a)) & \text{White to move} \end{cases}$$

Here $s$ is a position, $a$ a legal move, and $\text{next}(s, a)$ the position after $a$: exactly the simulator this lesson builds.

~~~dg-fig 09.01-minimax.svg
A two-ply tree. Each White (MIN) node takes its smallest leaf, the Black (MAX) root takes the largest of those, so the root is worth 3 and Black's best move is the left branch. Alpha-beta never needs to look at the two grey leaves.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**back up a two-ply tree by hand**

Black has three moves. After each, White has three replies, and the leaf numbers are Black's final margin in points.

Left branch: White picks $\min(3, 12, 8) = 3$. Middle branch: $\min(2, 4, 6) = 2$. Right branch: $\min(14, 5, 2) = 2$.

Black picks $\max(3, 2, 2) = 3$, so the left move is best and the position is worth +3 for Black. The right branch holds the best leaf (14), but White would never allow it.

Alpha-beta: after the left branch Black is already guaranteed 3. In the middle branch the first reply scores 2, so this branch is worth *at most* 2 to Black, which is worse than 3. The leaves 4 and 6 cannot change the decision and are skipped.

</div>

**Alpha-beta pruning** formalises that step: the same answer, minus branches that cannot affect it. On tic-tac-toe I measured plain minimax visiting 549,946 nodes to prove the game a draw, and alpha-beta 18,297, about 30 times fewer.

Chess programs add one ingredient: they stop at a fixed depth and score positions with a hand-written **evaluation function** (material, king safety, pawn structure), then minimax those scores.

## Step 2: why Go defeated that recipe

Go broke both halves of the chess recipe.

**The tree is too wide.** To see it, I played random games with this lesson's engine (random legal moves, never filling your own one-point eye) and counted legal actions every turn.

~~~dg-fig 09.01-branching.svg
Legal actions per turn in random-play games, measured with this lesson's engine. The count starts at 82 on 9×9 and 362 on 19×19 (every point plus pass) and falls roughly linearly as stones fill the board. The band shows the spread across games.
~~~

On 9×9, 200 random games averaged 34 legal actions per turn (median length 109 moves). On 19×19, 12 random games averaged about 150, but random games run far longer than real ones (median 466 moves) and spend half their time on a nearly full board. Expert games stay wide, hence the published "about 250".

**There is no good hand-written evaluation.** In Go every stone is worth the same, and whether a group of 20 stones is strong or already lost depends on tactical sequences many moves long. Hand-tuned Go evaluators stayed at amateur level.

AlphaGo's answer was to *learn* both missing pieces: a **policy** $p(a \mid s)$, a probability per move that narrows the tree, and a **value** $v(s)$, a predicted outcome that lets search stop early. **Monte Carlo tree search** (09.02) then spends computation selectively.

## Step 3: the board, groups and liberties

Go is played on the **intersections** of a 19×19 grid: 361 points (9×9 and smaller boards use identical rules). Black moves first. On each turn a player either places one stone of their colour on an empty point or **passes**. Stones never move; they leave only when captured.

Two ideas carry the whole game:

- A **group** (or chain) is a set of same-coloured stones joined by horizontal or vertical neighbours. Diagonal contact does not connect.
- A **liberty** of a group is an empty point orthogonally next to any stone in the group.

~~~dg-fig 09.01-liberties.svg
Crosses mark liberties (solid discs are Black, ringed discs are White). A lone stone has 4 liberties in the centre, 3 on the edge and 2 in the corner. The three connected white stones count their shared empty neighbours once each, giving 6.
~~~

Count the white group in the figure by hand. The stone at the left edge touches 2 empty points, the stone to its right touches 2 more, and the lower stone touches 3, one of which (directly below the edge stone) was already counted. So the group has $2 + 2 + 3 - 1 = 6$ distinct liberties. Liberties are a **set**, not a sum.

Computers find groups and liberties with a **flood fill**: start from one stone, keep visiting same-coloured orthogonal neighbours, and collect every empty neighbour touched on the way.

## Step 4: capture, suicide, ko and superko

**Capture.** After you place a stone, any *opponent* group left with zero liberties is removed, freeing its points.

**Suicide.** A move that leaves *your own* group with no liberties, after captures have been removed, is illegal under most rule sets, including the Chinese rules AlphaGo Zero used. (Tromp-Taylor rules, a minimal machine-friendly rule set, allow suicide and simply remove your own group.)

The words "after captures" decide everything. The rules process a move in a fixed order:

~~~dg-flow Order of operations for one move; checking suicide before removing captured stones is the classic bug.
Place stone -(remove opponent groups with 0 liberties)-> Check own group -(no liberties?)-> *Illegal: suicide
Check own group -(board seen before?)-> *Illegal: superko
~~~

Because capture comes first, you may play into a point surrounded by opponent stones *if* the move takes the last liberty of one of those groups.

**Ko.** That creates a loop. Below, Black captures one stone and Black's own stone is left with a single liberty: the point just emptied. White could capture straight back, recreating the earlier position, forever. The **ko rule** forbids the immediate recapture; White must play elsewhere first.

~~~dg-fig 09.01-capture-ko.svg
Left: the white stone has one liberty (green x). Middle: Black fills it, the white stone is removed, and Black's new stone (ringed) keeps one liberty. Right: White retaking at the red x would recreate the left board, so it is illegal for now.
~~~

**Superko** generalises ko. **Positional superko** says no move may recreate *any* earlier whole-board position, however long ago. It also catches rare cycles involving several kos. (**Situational superko** also compares who is to move.) For a program, superko is simpler than the special-case ko rule: keep a set of every position seen and reject any move whose result is in it.

<div class="ex">
<div class="lab">Worked example</div>

**process one move exactly as the engine does**

Position: the left board (5×5, rows and columns from 0). White's stone at (1, 1) has black stones above, below and left; its only liberty is (1, 2). Black plays there.

Step 1, place: the new stone's four neighbours are all white: (0, 2), (2, 2), (1, 3) and (1, 1). It has 0 liberties at this instant.

Step 2, captures: the white stone at (1, 1) now has 0 liberties and is removed. The other three still have liberties ((1, 3) has three), so they stay.

Step 3, suicide check: Black's stone now touches the emptied point (1, 1), so it has 1 liberty. The move is legal. Checked before step 2, it would wrongly be rejected.

Step 4, superko: this board is new, so the move stands. White's reply at (1, 1) would recreate the board from before step 1, which is in the history, so it is illegal. The code below confirms both.

</div>

## Step 5: ending the game and scoring

**Two consecutive passes end the game.** Then the board is scored.

AlphaGo Zero used **area scoring** (Chinese rules). Each player's score is:

$$\text{score} = \text{stones of your colour on the board} + \text{empty points whose region touches only your colour}.$$

An **empty region** is a connected set of empty points (flood fill again). If only Black stones border it, it counts for Black; if it touches both colours, for nobody. White also receives **komi** for moving second: 7.5 points in AlphaGo Zero's games. The half point rules out draws.

~~~dg-fig 09.01-scoring.svg
A finished 5×5 game. Blue empty points border only Black and orange ones only White. With a toy komi of 0.5 the final count is 13 to 12.5, so Black wins by half a point.
~~~

By hand: Black's empty region is the first two columns of the top three rows (6) plus the first column of the bottom two (2), so Black has $5 + 8 = 13$. White has $5 + 7 = 12$, plus 0.5 komi. Check: $13 + 12 = 25 = 5 \times 5$, since no region is shared. Result $13 - 12.5 = +0.5$.

For learning, the reward is the sign: $z = +1$ if the player to move wins, $-1$ otherwise; the value network (09.03) predicts $z$.

## Step 6: representing Go for a computer

- **The board** is a flat array of $N^2$ cells (361 for 19×19) holding 0 for empty, 1 for Black, 2 for White. Point $(r, c)$ lives at index $rN + c$.
- **The action space** has $N^2 + 1$ entries: one per point plus pass. On 19×19 that is 362, exactly the size of AlphaGo Zero's policy output.
- **The legal-move mask** is a boolean vector of length 362. Each empty point is tried (place, capture, suicide, repetition); pass is always legal. Illegal logits are set to $-\infty$ so search never tries them.

**Zobrist hashing** makes repetition checks cheap. Assign a random 64-bit number $z(p, c)$ to every (point, colour) pair; a position's hash is the XOR of its stones' numbers. XOR undoes itself, so adding or removing a stone is one update $h' = h \oplus z(p, c)$, and superko is one set lookup. With 4-bit toy numbers, $z(\text{A}, \text{black}) = 1011$ and $z(\text{B}, \text{white}) = 0110$: a board with both stones hashes to $1011 \oplus 0110 = 1101$, and capturing the white stone gives $1101 \oplus 0110 = 1011$, the black stone alone, as it should.

## Go rules in code

The class below implements every rule in this lesson: flood fill, the four-stage move order, superko via Zobrist hashes, a legal mask and area scoring. The tests reproduce the figures.

```python
import numpy as np

class Go:
    def __init__(self, n=9, komi=7.5, seed=0):
        self.n, self.komi, self.pass_move = n, komi, n * n
        self.board = np.zeros(n * n, dtype=np.int8)        # 0 empty, 1 black, 2 white
        self.to_play, self.passes = 1, 0
        self.zob = np.random.default_rng(seed).integers(0, 2**63, size=(n * n, 3), dtype=np.uint64)
        self.hash = np.uint64(0)
        self.seen = {self.hash}                             # hashes of every earlier position
        self.nbrs = [[r2 * n + c2 for r2, c2 in ((p // n - 1, p % n), (p // n + 1, p % n),
                      (p // n, p % n - 1), (p // n, p % n + 1)) if 0 <= r2 < n and 0 <= c2 < n]
                     for p in range(n * n)]

    def group(self, b, p):                                  # flood fill -> (stones, liberties)
        stack, grp, libs = [p], {p}, set()
        while stack:
            for m in self.nbrs[stack.pop()]:
                if b[m] == 0: libs.add(m)
                elif b[m] == b[p] and m not in grp: grp.add(m); stack.append(m)
        return grp, libs

    def try_move(self, p):                                  # new (board, hash), or None if illegal
        if p == self.pass_move: return self.board.copy(), self.hash
        if self.board[p] != 0: return None
        b, h, me = self.board.copy(), self.hash, self.to_play
        b[p] = me; h ^= self.zob[p, me]                     # 1. place the stone
        for q in self.nbrs[p]:                              # 2. remove opponent groups with 0 liberties
            if b[q] == 3 - me:
                grp, libs = self.group(b, q)
                if not libs:
                    for s in grp: b[s] = 0; h ^= self.zob[s, 3 - me]
        if not self.group(b, p)[1]: return None             # 3. suicide (checked AFTER captures)
        if h in self.seen: return None                      # 4. positional superko
        return b, h

    def legal_mask(self):                                   # 362 entries on 19x19, 82 on 9x9
        return np.array([self.try_move(p) is not None for p in range(self.n * self.n + 1)])

    def play(self, p):
        res = self.try_move(p)
        if res is None: raise ValueError(f"illegal move {p}")
        self.board, self.hash = res
        self.passes = self.passes + 1 if p == self.pass_move else 0
        self.seen.add(self.hash); self.to_play = 3 - self.to_play

    def area_score(self):                                   # > 0 means Black wins
        area = {1: int((self.board == 1).sum()), 2: int((self.board == 2).sum())}
        done = set()
        for p in np.flatnonzero(self.board == 0):
            if p in done: continue
            region, border, stack = {p}, set(), [p]
            while stack:
                for m in self.nbrs[stack.pop()]:
                    if self.board[m] == 0 and m not in region: region.add(m); stack.append(m)
                    elif self.board[m]: border.add(int(self.board[m]))
            done |= region
            if len(border) == 1: area[border.pop()] += len(region)
        return area[1] - area[2] - self.komi

P = lambda r, c: r * 5 + c                                  # (row, col) -> index on a 5x5 board
print(Go(n=9).legal_mask().sum())                          # 82  (81 points + pass)

g = Go(n=5)                                                 # the ko from the figure
for mv in [P(0,1), P(0,2), P(2,1), P(2,2), P(1,0), P(1,3), P(4,4), P(1,1), P(1,2)]: g.play(mv)
print(g.legal_mask()[P(1,1)])                               # False: superko
g.play(P(4,0)); g.play(P(3,3))                              # one exchange elsewhere
print(g.legal_mask()[P(1,1)])                               # True

g = Go(n=5, komi=0.5)                                       # the scoring example
for b, w in zip([(0,2),(1,2),(2,2),(3,1),(4,1)], [(0,3),(1,3),(2,3),(3,2),(4,2)]):
    g.play(P(*b)); g.play(P(*w))
g.play(25); g.play(25)
print(g.passes, g.area_score())                             # 2 0.5
```

This engine copies the board for every candidate move, which is fine for learning: one random 9×9 game with a full mask every turn took about 0.1 s of CPU here. Production engines update group and liberty counts incrementally, so a move touches only its neighbouring groups.

## Checking your engine

Rule bugs do not crash; they quietly produce wrong games. Test on tiny boards where you can draw the answer:

- **Empty board**: exactly $N^2 + 1$ legal actions (82 on 9×9).
- **Capture beats suicide**: a stone played into a surrounded point that captures is legal; the same point without a capture is not.
- **Ko**: immediate recapture illegal; legal again after one exchange elsewhere.
- **Scoring**: a hand-counted finished position, with Black area + White area + neutral points $= N^2$.
- **Hashes**: the incremental hash equals one recomputed from scratch after a long random game, and thousands of random games end with two passes and no exceptions.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| A legal capture is rejected as suicide | own liberties checked before removing captured groups | place, remove dead opponent groups, *then* check your own group
| Self-play games loop forever | no superko, or history not updated | add every new position's hash to the history
| Liberty count 7 instead of 6 | liberties summed per stone | collect liberties in a set
| Final score far off | dead stones left on the board | under area rules, capture them before passing
| Search picks an illegal move | policy not masked | set illegal logits to $-\infty$ before the softmax
| Random games never end | players fill their own eyes and get recaptured | forbid filling one-point eyes in random playouts (not a rule)

## Going deeper

**What the network actually sees.** A single board is not Markov (lesson 08.01) for Go: superko depends on history, and recent moves hint at intent. AlphaGo Zero's input is a $19\times19\times17$ stack of binary planes: 8 for the current player's stones over the last 8 positions, 8 for the opponent's over the same positions, and one constant plane that is all 1s when Black is to move and all 0s when White is. Encoding from the mover's perspective means the network always answers "how good is this for me?".

**Symmetry.** The rules are unchanged by the 8 rotations and reflections of the board (the dihedral group $D_4$), so each position gives 8 equivalent training examples and 8 ways to evaluate it (09.03).

**How big is Go really?** Many move sequences reach the same position. Each point is empty, black or white, so there are at most $3^{361} \approx 10^{172}$ boards; Tromp and colleagues computed (2016) that about $2.08 \times 10^{170}$ of them are legal, about 1.2% (a published value quoted from memory). Generalised to $N \times N$, Go with Japanese-style ko is EXPTIME-complete (Robson 1983, also from memory).

**Rule sets.** Japanese rules use **territory scoring** (empty territory plus captured stones, not stones on the board) and need agreement on which stones are dead. Area scoring lets a program settle everything by playing on, so research systems use it.

## Key terms

| Term | Meaning
| --- | --- |
| Game tree | Positions as nodes and moves as edges, from the start position to finished games
| Branching factor | The number of legal moves in a typical position; about 250 in 19×19 Go
| Minimax | Back up leaf results, taking the max on your turns and the min on the opponent's
| Alpha-beta pruning | Minimax that skips branches which cannot change the result
| Group | Same-coloured stones joined horizontally or vertically; lives or dies as one
| Liberty | An empty point orthogonally adjacent to a group; zero liberties means capture
| Ko / positional superko | Ko forbids an immediate recapture that restores the previous board; superko forbids repeating any earlier board
| Area scoring | Stones on the board plus empty regions bordered by only your colour, plus komi for White
| Komi | Points given to White to offset Black's first move (7.5 in AlphaGo Zero)
| Zobrist hash | XOR of random numbers per (point, colour); updated in one step per stone change

<div class="take">
<div class="lab">Key takeaway</div>

Go is a perfect-information game whose tree is too wide and whose positions are too hard to score by hand for minimax; its rules reduce to groups, liberties, capture-before-suicide, superko and area scoring, which a program implements with flood fill, a 362-action legal mask and Zobrist hashes over the game history.

</div>
