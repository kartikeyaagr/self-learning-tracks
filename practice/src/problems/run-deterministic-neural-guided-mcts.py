import numpy as np

META = dict(title='Run neural-guided MCTS on a fixed game tree', tags=['Reinforcement Learning'], diff='Hard',
            runtime='python', lessons=[],
            related=['select-action-with-puct', 'back-up-mcts-simulation-path', 'initialize-legal-mcts-edge-statistics',
                     'turn-root-visits-into-search-policy'])

STATEMENT = r"""
Put selection, expansion, evaluation and backup together into the full AlphaGo Zero search. To make it exactly reproducible, the game and the "network" are replaced by a fixed tree in which every node already carries the network's output.

## The tree
`tree` is a list of dicts; node `0` is the root. Node `i` has:
- `"children"`: list of child node ids, one per legal action (action `j` leads to `children[j]`). An empty list means the node is **terminal**.
- `"priors"`: list of floats, the policy output for the actions (same length as `children`, summing to 1).
- `"value"`: float, the evaluation **from the point of view of the player to move at this node**: the network's estimate, or the exact game result for a terminal node.

Players alternate at every edge. Each node has exactly one parent. The root is not terminal.

## Search
Keep, for every node, visit counts `N[node][j]` and value sums `W[node][j]` per action, all starting at 0, and a set of **expanded** nodes. Before the first simulation the root is expanded (its value is not backed up). Then run `num_sims` simulations, each as follows:

1. Start at the root with an empty path.
2. Repeat: if the current node is terminal, the leaf value is its `"value"`; stop. If it is not expanded, expand it (add it to the set); the leaf value is its `"value"`; stop. Otherwise choose action $j$ with PUCT, append `(node, j)` to the path and move to `children[j]`.
3. Back up: walking the path from the last edge to the first, negate the value and then add it to `W[node][j]` and add 1 to `N[node][j]`.

PUCT at node $s$ (all children are legal):

$$j^* = \arg\max_j \; Q(s, j) + c_\text{puct}\, P(s, j)\, \frac{\sqrt{\max(1, \sum_k N(s, k))}}{1 + N(s, j)}, \qquad Q = W / N, \text{ or } 0 \text{ if } N = 0.$$

Exact ties go to the smallest index $j$.

## Output
Return a dict:
- `"visits"`: list of ints, `N[0][j]` for each root action;
- `"q"`: list of floats, the root's $Q(0, j)$ (0 for unvisited actions);
- `"action"`: the root action with the most visits, ties to the smallest index.
"""

STARTER = '''import math

def run_mcts(tree, num_sims, c_puct):
    """tree: list of {"children": [...], "priors": [...], "value": float}; node 0 is the root.
    Return {"visits": [int], "q": [float], "action": int}."""
    raise NotImplementedError
'''

SOLUTION = '''import math

def run_mcts(tree, num_sims, c_puct):
    N = [[0] * len(nd['children']) for nd in tree]
    W = [[0.0] * len(nd['children']) for nd in tree]
    expanded = {0}                                  # the root is evaluated before any simulation

    def q(s, j):
        return W[s][j] / N[s][j] if N[s][j] > 0 else 0.0

    def select(s):
        sqrt_total = math.sqrt(max(1, sum(N[s])))
        best, best_score = 0, -math.inf
        for j, prior in enumerate(tree[s]['priors']):
            score = q(s, j) + c_puct * prior * sqrt_total / (1 + N[s][j])
            if score > best_score:                  # strict '>' keeps the smallest index on ties
                best, best_score = j, score
        return best

    for _ in range(num_sims):
        node, path = 0, []
        while True:
            if not tree[node]['children']:          # terminal: exact result
                value = tree[node]['value']
                break
            if node not in expanded:                # new leaf: expand and use the network value
                expanded.add(node)
                value = tree[node]['value']
                break
            j = select(node)
            path.append((node, j))
            node = tree[node]['children'][j]
        for s, j in reversed(path):                 # value flips to the parent's point of view
            value = -value
            N[s][j] += 1
            W[s][j] += value

    visits = N[0]
    action = max(range(len(visits)), key=lambda j: (visits[j], -j))
    return {'visits': list(visits), 'q': [q(0, j) for j in range(len(visits))], 'action': action}
'''

HINTS = [
    'Keep `N` and `W` as one list per node and `expanded` as a set containing the root. A simulation is: descend while the node is expanded and non-terminal, then stop at a terminal or freshly expanded node with its value.',
    'Selection is the PUCT formula with `sqrt(max(1, sum(N[s])))`; scan the children in order and only replace the best on a strictly larger score so ties keep the smallest index.',
    'Backup walks `reversed(path)`: `value = -value` first, then `N[s][j] += 1` and `W[s][j] += value`. The final action is `max(range(A), key=lambda j: (visits[j], -j))`.',
]

def _random_tree(seed, max_nodes, max_depth):
    rng = np.random.default_rng(seed)
    tree, depth, frontier = [None], [0], [0]
    while frontier:
        i = frontier.pop(0)
        k = 0 if depth[i] >= max_depth or len(tree) >= max_nodes else int(rng.integers(1, 4))
        if depth[i] > 0 and k and rng.random() < 0.15:
            k = 0                                    # some early terminal positions
        if k == 0 and i == 0:
            k = 2
        kids = list(range(len(tree), len(tree) + k))
        pri = rng.random(k) + 0.05
        pri = [float(x) for x in pri / pri.sum()]
        val = float(rng.integers(-1, 2)) if k == 0 else float(np.round(rng.uniform(-0.9, 0.9), 3))
        tree[i] = {'children': kids, 'priors': pri if k else [], 'value': val}
        for kid in kids:
            tree.append(None); depth.append(depth[i] + 1); frontier.append(kid)
    return tree

_two = [{'children': [1, 2], 'priors': [0.6, 0.4], 'value': 0.0},
        {'children': [], 'priors': [], 'value': 1.0},
        {'children': [], 'priors': [], 'value': -1.0}]
_deep = [{'children': [1, 2], 'priors': [0.5, 0.5], 'value': 0.1},
         {'children': [3, 4], 'priors': [0.7, 0.3], 'value': 0.2},
         {'children': [], 'priors': [], 'value': 0.0},
         {'children': [], 'priors': [], 'value': 1.0},
         {'children': [], 'priors': [], 'value': -1.0}]
_net = [{'children': [1, 2, 3], 'priors': [0.2, 0.5, 0.3], 'value': 0.0},
        {'children': [4], 'priors': [1.0], 'value': -0.37},
        {'children': [5, 6], 'priors': [0.5, 0.5], 'value': 0.23},
        {'children': [], 'priors': [], 'value': 0.0},
        {'children': [], 'priors': [], 'value': -1.0},
        {'children': [], 'priors': [], 'value': 1.0},
        {'children': [], 'priors': [], 'value': -1.0}]

TESTS = [
    (_two, 4, 1.0),
    (_deep, 6, 1.0),
    (_two, 0, 1.0),
    (_net, 25, 1.5),
    (_net, 25, 0.1),
    (_random_tree(71, 40, 4), 60, 1.25),
    (_random_tree(72, 80, 5), 200, 2.0),
]

EXPLANATION = r"""
## The four phases
1. **Select**: below the root, PUCT trades the network prior against the values backed up so far.
2. **Expand + evaluate**: the first time a simulation reaches a node it stops there and uses the network's value. AlphaGo Zero has no random rollouts; the value head replaces them. This is why one simulation adds at most one node to the tree.
3. **Terminal nodes** return the true result every time they are reached and are never expanded.
4. **Backup**: alternating signs, as in the backup exercise.

After many simulations, the root's visit counts concentrate on the moves whose subtrees hold up under the best replies, and $\pi \propto N^{1/\tau}$ becomes the move choice and the policy target.

## Determinism
Everything here is fixed: the priors and values are given, PUCT ties go to the smallest index, and the root's own value is not backed up. Change any one of those conventions (for example starting with $\sqrt{\sum N}$ instead of $\sqrt{\max(1, \sum N)}$, or backing up the root's value as a first visit) and the visit counts change, which is why papers and engines spell them out.

## Complexity
Each simulation costs $O(d \cdot b)$ for depth $d$ and branching $b$, plus one network evaluation, which in practice dominates by orders of magnitude.

## Common bugs
- Backing up without negating first, so the parent sees the leaf's own-perspective value.
- Expanding a node and then continuing to descend in the same simulation.
- Expanding terminal nodes, or using the network value instead of the true result for them.
- Breaking PUCT ties with `>=`, which picks the largest index.

## Interview follow-ups
- The real search reuses the subtree under the chosen move for the next turn. What has to be discarded, and why is Dirichlet noise added again to the new root?
- How would you batch leaf evaluations from many simulations at once without all of them choosing the same path?
"""
