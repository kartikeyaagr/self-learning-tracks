import numpy as np

META = dict(title='Capstone: PUCT search that plays tic-tac-toe', tags=['Reinforcement Learning'], diff='Hard', runtime='python',
            related=['tictactoe-legal-moves-and-winner', 'tictactoe-minimax', 'select-action-with-puct', 'turn-root-visits-into-search-policy'])

STATEMENT = r"""
Run AlphaZero's search on tic-tac-toe with a stand-in network that returns a **uniform prior** over legal moves and a **value of 0** for every non-terminal position. Only real game endings carry information, so this is a pure test of the search.

Write `puct_search(board, n_sims, c_puct, tau)`:

- **Tree.** Each edge from a node stores a prior $P$, a visit count $N$ and a total value $W$, with $W$ from the view of the player who makes that move. $Q = W/N$, or 0 if $N = 0$.
- **Root.** Expand the root before the first simulation (this does not count as one).
- **Select.** From the root, while the current node is expanded and not terminal, take the legal move maximising $Q + c\,P\sqrt{1 + \sum_b N_b}/(1 + N)$, where the sum is over the node's edges. Ties go to the lowest move index.
- **Evaluate.** At the reached node, with $v$ from the view of its player to move: if the game is over, $v = -1$ when there is a winner and $0$ for a draw (terminal nodes are never expanded). Otherwise expand it with uniform priors and use $v = 0$.
- **Back up.** Walk back to the root. The edge into the leaf gets $W \mathrel{+}= -v$; flip the sign at each level going up. Every edge on the path gets $N \mathrel{+}= 1$.

Return a dict with `visits` (9 root edge counts, 0 for occupied cells), `policy` (visits$^{1/\tau}$ normalised; for $\tau = 0$ a one-hot on the most visited move), `move` (most visited move, lowest index on ties) and `q` (9 root $Q$ values rounded to 6 decimals, `None` for occupied or unvisited cells). If the root is already finished, return all-zero `visits` and `policy`, `move` = `None`, and all-`None` `q`.
"""

STARTER = '''def puct_search(board, n_sims, c_puct, tau):
    """AlphaZero-style PUCT search with a uniform prior and zero value."""
    raise NotImplementedError
'''

SOLUTION = '''LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

def _winner(b):
    for x, y, z in LINES:
        if b[x] != "." and b[x] == b[y] == b[z]:
            return b[x]
    return None

class Node:
    def __init__(self, board):
        self.board = board
        self.win = _winner(board)
        self.terminal = self.win is not None or "." not in board
        self.edges = None

    def expand(self):
        me = "X" if self.board.count("X") == self.board.count("O") else "O"
        legal = [i for i, ch in enumerate(self.board) if ch == "."]
        self.edges = {m: {"P": 1.0 / len(legal), "N": 0, "W": 0.0,
                          "child": Node(self.board[:m] + me + self.board[m + 1:])} for m in legal}

def _select(node, c):
    tot = sum(e["N"] for e in node.edges.values())
    best, best_s = None, None
    for m in sorted(node.edges):
        e = node.edges[m]
        q = e["W"] / e["N"] if e["N"] else 0.0
        s = q + c * e["P"] * (1 + tot) ** 0.5 / (1 + e["N"])
        if best_s is None or s > best_s:
            best, best_s = m, s
    return best

def puct_search(board, n_sims, c_puct, tau):
    root = Node(board)
    if root.terminal:
        return {"visits": [0] * 9, "policy": [0.0] * 9, "move": None, "q": [None] * 9}
    root.expand()
    for _ in range(n_sims):
        node, path = root, []
        while node.edges is not None and not node.terminal:
            m = _select(node, c_puct)
            path.append(node.edges[m])
            node = node.edges[m]["child"]
        if node.terminal:
            v = -1.0 if node.win else 0.0
        else:
            node.expand()
            v = 0.0
        for e in reversed(path):
            e["N"] += 1
            e["W"] += -v
            v = -v
    visits = [root.edges[m]["N"] if m in root.edges else 0 for m in range(9)]
    move = max(range(9), key=lambda m: (visits[m], -m))
    if tau == 0:
        policy = [1.0 if m == move else 0.0 for m in range(9)]
    else:
        w = [v ** (1.0 / tau) for v in visits]
        s = sum(w)
        policy = [x / s for x in w]
    q = [round(root.edges[m]["W"] / root.edges[m]["N"], 6) if m in root.edges and root.edges[m]["N"] else None for m in range(9)]
    return {"visits": visits, "policy": policy, "move": move, "q": q}
'''

HINTS = [
    'Store W from the view of the player who chose the edge; then the parent can maximise Q directly without flipping signs.',
    'The value at a leaf is for its player to move, so the edge into the leaf receives minus that value.',
    'A terminal node is evaluated every time the search reaches it; it is never expanded.',
    'Compare the chosen move with the minimax solver from the earlier step on the same positions.',
]

TESTS = [
    ("XX.OO....", 50, 1.5, 1.0),
    ("XO.XO....", 100, 1.5, 0.0),
    ("XOX.O....", 400, 1.5, 1.0),
    ("X........", 200, 1.5, 1.0),
    ("XXXOO....", 10, 1.5, 1.0),
    ("XX.OO....", 50, 1.5, 0.5),
]

EXPLANATION = r"""
## What the search does here
With a uniform prior and a value of zero everywhere, the only signal comes from positions where the game actually ends. PUCT still finds wins and blocks: the exploration term sends some visits to every move, a move that wins immediately returns $+1$ every time it is tried, and its $Q$ quickly dominates. Visit counts then concentrate on it, and those counts are the training target for the policy network.

## Checking against minimax
On the first three positions the solver from the minimax step says the only good moves are 2, 6 and 7 respectively. The search picks the same moves. The third is harder: X must block a threat, and the danger only shows two plies down, which takes more simulations to see.

## Where it fails
In the fourth test X has taken a corner. Minimax says O must take the centre (cell 4) or lose. With 200 simulations the search never reaches a forced ending, every $Q$ stays 0 and the visits are spread evenly, so it picks cell 1, a losing move. Without a value estimate the search is blind until the end of the game is within reach.

## What a real network adds
- **The prior** concentrates visits on plausible moves, so the tree stays narrow in games with hundreds of legal moves.
- **The value** replaces the need to reach the end of the game; a single network evaluation stands in for a whole rollout.
- **Dirichlet noise** on the root priors during self-play makes sure every move is tried sometimes, even when the network is confident.

## Temperature
The policy target uses $\tau = 1$ early in self-play games (to keep variety) and $\tau \to 0$ later (to play the best move). The second test shows $\tau = 0$: a one-hot target.

## Interview follow-ups
- AlphaZero picks the most visited move rather than the one with the highest $Q$. Why is that more robust?
"""
