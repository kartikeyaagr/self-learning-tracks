import numpy as np

META = dict(title='Move an agent on a grid', tags=['Reinforcement Learning'], diff='Easy', runtime='python')

STATEMENT = r"""
A grid world is a list of equal-length strings. `#` is a wall; every other character (`.`, `S`, `G`, `L`) is a cell the agent can stand on.

Write `grid_step(grid, pos, action)` where `pos = [row, col]` and the actions are `0` up, `1` right, `2` down, `3` left.

- Compute the target cell.
- If it is off the grid or a wall, the agent stays where it is.
- Return the new position as `[row, col]`.
"""

STARTER = '''def grid_step(grid, pos, action):
    """Deterministic move on a grid with walls."""
    dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
    return [pos[0] + dr, pos[1] + dc]
'''

SOLUTION = '''def grid_step(grid, pos, action):
    dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
    r, c = pos[0] + dr, pos[1] + dc
    if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != '#':
        return [r, c]
    return [pos[0], pos[1]]
'''

HINTS = [
    'Check the bounds before indexing; a negative index in Python silently wraps to the other side of the grid.',
    'Return a fresh list, not the caller’s pos object, so later code cannot mutate it by accident.',
]

G = ["S..#", ".#..", "...G"]

TESTS = [
    (G, [0, 0], 1),
    (G, [0, 0], 0),
    (G, [0, 0], 3),
    (G, [0, 2], 1),
    (G, [1, 0], 1),
    (G, [2, 2], 1),
    (G, [2, 3], 2),
]

EXPLANATION = r"""
## The environment contract
Every RL environment answers one question: given a state and an action, what happens next? Getting the edges right matters more than it looks. The negative-index bug in the starter teleports the agent across the map, and an agent will find and exploit any such bug, because it maximises reward under the rules you actually wrote rather than the ones you meant.

## Deterministic first
This world is deterministic. Adding a "slip" probability (the move goes sideways some of the time) turns it into a stochastic MDP, which is where value-based methods earn their keep over plain search.

## Interview follow-ups
- What should `reset` return, and why should an environment take a seed?
"""
