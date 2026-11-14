---
id: "09.01"
course: "deep-learning"
module: "09"
slug: "game-trees-and-the-rules-of-go"
title: "Game trees and the rules of Go"
keyIdeas: ["Go is a two-player, zero-sum, deterministic, perfect-information game: in principle minimax solves it, in practice its tree (branching factor around 250, games of about 150 moves) is far too big.", "Stones that touch orthogonally form a group; a group's liberties are the empty points next to it, and a group with no liberties is captured and removed.", "A move is legal only if the point is empty, it does not leave your own group without liberties after captures (in most rule sets), and it does not repeat an earlier board position (superko).", "The game ends after two consecutive passes; area scoring counts stones plus empty regions that reach only your colour, plus komi for White.", "A board is a 361-point array plus history; groups and liberties come from flood fill, and Zobrist hashing makes repetition checks cheap. The action space is 362 (every point plus pass)."]
question: "Your Go engine sometimes reports that a move capturing an opponent group is illegal as suicide. The bug report shows the placed stone had no empty neighbours at the moment it was placed. Explain the bug in terms of the order of rule checks, and explain why this specific order is what the rules require."
answer: "The engine checks the new stone's liberties before removing captured opponent groups. When a stone is placed into a point surrounded by opponent stones, its group may have zero liberties at that instant, but if that placement takes the last liberty of an adjacent opponent group, that group is removed first, which empties points next to the new stone and gives it liberties. The rules define a move as: place the stone, remove all opponent groups with no liberties, then check your own group. Only if your group still has no liberties after captures is the move suicide (illegal under Chinese rules). Capturing therefore always takes precedence, which is what makes moves that fill an opponent's last liberty from 'inside' legal. Fix the order: place, scan the up-to-four adjacent opponent groups and remove any with no liberties, then flood-fill your own group, then run the superko check on the resulting board."
sources: [{"title": "John Tromp: The Game of Go (Tromp-Taylor rules)", "url": "https://tromp.github.io/go.html", "note": "A complete, machine-friendly statement of the rules in ten short lines, with area scoring."}, {"title": "Sensei's Library: Superko", "url": "https://senseis.xmp.net/?Superko", "note": "Positional versus situational superko and why repetition rules exist."}, {"title": "Silver et al.: Mastering the game of Go without human knowledge", "url": "https://www.nature.com/articles/nature24270", "note": "AlphaGo Zero; the Methods section specifies the rules, komi and the 17-plane input encoding."}, {"title": "Chess Programming Wiki: Zobrist Hashing", "url": "https://www.chessprogramming.org/Zobrist_Hashing", "note": "Incremental XOR position hashing as used for repetition detection."}]
practice: ["apply-go-action-with-captures-and-repetition", "find-go-group-and-its-liberties", "build-legal-go-action-mask", "score-terminal-position-with-tromp-taylor-rules", "encode-alphago-zero-board-history"]
tags: ["game trees", "Go rules"]
tools: ["NumPy"]
---

## Where this module is going

This module builds the AlphaGo Zero algorithm from the ground up: the game and its rules (this lesson), Monte Carlo tree search (09.02), the policy-value network that guides it (09.03), and the self-play training loop that ties them together (09.04). It uses Module 08 throughout: MCTS is bandits (08.05) applied at every node of a tree, the network is an actor-critic-style two-headed model (08.06), and the training loop is a form of policy iteration (08.02).

## Game trees and why Go was hard

Go belongs to the cleanest class of games: two players alternate, both see everything, there is no chance, and one player's win is the other's loss. Such a game is a tree whose nodes are positions and whose edges are moves. **Minimax** assigns each leaf its result and backs values up, maximising at your turns and minimising at the opponent's; the root value is the game's exact outcome under perfect play. Alpha-beta pruning makes this practical for chess with a hand-built evaluation function at a depth limit.

| Game | Branching factor | Typical length | Rough tree size
| --- | --- | --- | --- |
| Chess | about 35 | about 80 plies | $35^{80}\approx10^{123}$
| Go 19x19 | about 250 | about 150 plies | $250^{150}\approx10^{360}$

Go defeated that approach for two reasons. The tree is far wider, so even a few plies of full-width search is expensive. And there was no good hand-written evaluation function: whether a group is alive or a region is territory depends on long tactical sequences, and simple material counting means nothing. AlphaGo's answer was to *learn* both a move prior (to narrow the tree) and an evaluation (to cut it off), and search selectively with MCTS.

## The rules, precisely

The board has $19\times19=361$ intersections. Black moves first; players alternately place a stone of their colour on an empty point or pass.

- **Groups.** Same-coloured stones connected horizontally or vertically (not diagonally) form a group, which lives or dies together.
- **Liberties.** The empty points orthogonally adjacent to any stone of the group. A lone stone in the centre has 4, on an edge 3, in a corner 2.
- **Capture.** After you place a stone, every *opponent* group with zero liberties is removed. Only then is your own group checked.
- **Suicide.** A move that leaves your own group with no liberties after captures is illegal under most rules, including the Chinese rules AlphaGo Zero used. (Tromp-Taylor rules allow it and simply remove your group.)
- **Ko and superko.** The simple ko rule forbids recapturing immediately when that would restore the previous position. **Positional superko** generalises it: no move may recreate any earlier board position. Without it, cycles of captures could repeat forever.
- **End and scoring.** Two consecutive passes end the game. **Area scoring** counts each player's stones on the board plus the empty regions that border only that player's stones. White receives **komi** (7.5 points in AlphaGo Zero's games) to offset Black's first-move advantage; the half point rules out draws.

~~~dg-flow Order of operations for one move; checking suicide before removing captured stones is the classic bug.
Place stone -(remove opponent groups with 0 liberties)-> Check own group -(no liberties?)-> *Illegal: suicide
Check own group -(board seen before?)-> *Illegal: superko
~~~

## Representing the board

A position is a flat array of 361 cells holding 0 (empty), 1 (black) or 2 (white), plus whose turn it is, the pass count and a history of earlier positions. The action space is 362: one action per point plus pass, which is exactly the size of AlphaGo Zero's policy output. Groups and liberties come from a flood fill:

```python
def group_and_liberties(board, p, N=19):          # board: list of N*N ints, p: index of a stone
    colour, stack, group, libs = board[p], [p], {p}, set()
    while stack:
        q = stack.pop(); r, c = divmod(q, N)
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= rr < N and 0 <= cc < N:
                n = rr * N + cc
                if board[n] == 0: libs.add(n)
                elif board[n] == colour and n not in group:
                    group.add(n); stack.append(n)
    return group, libs
```

A legal-move mask runs this logic for every empty point: tentatively place, remove captured opponent groups, reject suicide, reject repetition. For repetition you need to compare against every earlier position quickly. **Zobrist hashing** assigns a random 64-bit number to each (point, colour) pair; a position's hash is the XOR of the numbers of its stones, updated incrementally by XOR-ing stones in and out, so a superko check is a set lookup.

<div class="ex">
<div class="lab">Worked example</div>

**a capture and a ko**

A white stone sits on the edge with two black stones beside it along the edge, so its only liberty is the point directly inward from it.

Black plays on that liberty. Step 1: the white stone now has 0 liberties and is removed. Step 2: Black's new stone has a liberty again (the point just emptied), so the move is legal, not suicide.

If the captured stone could immediately recapture the black stone in the same way, recreating the position before Black's move, that recapture is forbidden by the ko rule; White must play elsewhere first.

</div>

## What the network sees

A single board does not satisfy the Markov property (lesson 08.01) for Go: legality depends on history through superko, and recent moves carry information about intent. AlphaGo Zero's input is a $19\times19\times17$ stack: 8 binary planes for the current player's stones over the last 8 positions, 8 for the opponent's, and one constant plane for the colour to move. Encoding is from the perspective of the player to move, so the network always answers "how good is this for me?". Because the rules are symmetric under the 8 rotations and reflections of the board (the dihedral group), every position also gives 8 equivalent training examples and 8 equivalent ways to evaluate it (lesson 09.03).

<div class="take">
<div class="lab">Key takeaway</div>

Go is a perfect-information game whose tree is too wide and whose positions are too hard to evaluate by hand for minimax; its rules reduce to groups, liberties, capture-before-suicide, superko and area scoring, implemented with flood fill, a 362-action mask and Zobrist hashes over a history of planes.

</div>
