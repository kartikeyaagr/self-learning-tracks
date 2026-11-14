import numpy as np

META = dict(title='Capstone: solve your grid world and roll out the policy', tags=['Reinforcement Learning'], diff='Hard', runtime='numpy',
            related=['gridworld-step', 'reward-and-termination', 'td-target-terminal-vs-truncated'])

STATEMENT = r"""
Use the environment pieces to solve a grid world exactly, then check the result by running the greedy policy in the environment.

Grid rules (as in the steps): `#` wall, `G` goal (+1, terminates), `L` lava (−1, terminates), any other cell costs `step_cost`. Actions `0` up, `1` right, `2` down, `3` left; blocked moves stay in place.

Write `solve_grid(grid, start, gamma, step_cost, iters, max_steps)`:

1. **Value iteration.** Start with $V = 0$ everywhere. Run exactly `iters` synchronous sweeps. Goal and lava cells are terminal and keep $V = 0$. For every other open cell,
$$V(s) \leftarrow \max_a \big[r(s') + \gamma\,(1 - \text{term}(s'))\,V(s')\big],\quad s' = \text{step}(s, a).$$
2. **Greedy rollout.** From `start`, repeatedly take the action with the highest bracketed value under the final $V$ (ties go to the lowest action index). Stop when the agent enters a terminal cell, or after `max_steps` steps.

Return a dict: `V` (list of rows; walls are `None`, values rounded to 6 decimals), `path` (list of positions from `start` to the last position, inclusive), `return` (undiscounted sum of rewards, rounded to 6 decimals), `terminated`, `truncated`.
"""

STARTER = '''import numpy as np

def solve_grid(grid, start, gamma, step_cost, iters, max_steps):
    """Value iteration on a grid world, then a greedy rollout to check the policy."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def _step(grid, r, c, a):
    dr, dc = MOVES[a]
    nr, nc = r + dr, c + dc
    if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != '#':
        return nr, nc
    return r, c

def _rew(grid, r, c, step_cost):
    ch = grid[r][c]
    if ch == 'G':
        return 1.0, True
    if ch == 'L':
        return -1.0, True
    return 0.0 - step_cost, False

def _q(grid, V, r, c, a, gamma, step_cost):
    nr, nc = _step(grid, r, c, a)
    rew, term = _rew(grid, nr, nc, step_cost)
    return rew + gamma * (0.0 if term else V[nr, nc])

def solve_grid(grid, start, gamma, step_cost, iters, max_steps):
    H, W = len(grid), len(grid[0])
    V = np.zeros((H, W))
    open_cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] not in '#GL']
    for _ in range(iters):
        V2 = V.copy()
        for r, c in open_cells:
            V2[r, c] = max(_q(grid, V, r, c, a, gamma, step_cost) for a in range(4))
        V = V2
    r, c = start
    path, total, term, t = [[r, c]], 0.0, grid[r][c] in 'GL', 0
    while not term and t < max_steps:
        qs = [_q(grid, V, r, c, a, gamma, step_cost) for a in range(4)]
        a = int(np.argmax(qs))
        r, c = _step(grid, r, c, a)
        rew, term = _rew(grid, r, c, step_cost)
        total += rew; t += 1
        path.append([r, c])
    Vout = [[None if grid[i][j] == '#' else round(float(V[i, j]), 6) for j in range(W)] for i in range(H)]
    return {"V": Vout, "path": path, "return": round(total, 6), "terminated": bool(term),
            "truncated": bool(not term and t >= max_steps)}
'''

HINTS = [
    'Synchronous means every cell in a sweep reads the values from the previous sweep; update a copy.',
    'The bracketed value is exactly the TD target from the earlier step, with termination (not truncation) cutting the bootstrap.',
    'Too few sweeps leave far cells at 0 or at the step-cost floor, and the greedy rollout can wander until it is truncated.',
]

G1 = ["S..G",
      ".#.L",
      "...."]
G2 = ["S.L.G",
      ".....",]
G3 = ["S#G",
      ".#.",
      "..."]

TESTS = [
    (G1, [0, 0], 0.9, 0.0, 50, 20),
    (G1, [0, 0], 0.9, 0.04, 50, 20),
    (G2, [0, 0], 0.99, 0.01, 100, 20),
    (G3, [0, 0], 0.95, 0.0, 2, 10),
    (G3, [0, 0], 0.95, 0.0, 30, 10),
    (G2, [0, 0], 0.5, 0.6, 50, 20),
]

EXPLANATION = r"""
## What the capstone checks
The steps defined the dynamics, the rewards and the learning target. Value iteration applies that target to every state at once until the values settle, and the greedy rollout runs the result through the same environment code. If the rollout ends at the goal with the return the values predicted, the pieces agree with each other.

## Things the tests show
- **Discounting alone prefers short paths.** With no step cost, $\gamma < 1$ already makes a goal three steps away worth more than one five steps away.
- **A detour around lava.** In the third test the agent drops to the bottom row to pass the lava cell, then climbs back up to the goal.
- **Too few sweeps.** In the fourth test, two sweeps only carry the goal's value two cells back, so the start still has $V = 0$. All moves tie, the agent takes action 0 (up) forever against the edge, and the episode is truncated: an untrained policy looks exactly like this.
- **A step cost that is too high.** In the last test each step costs 0.6. Reaching the lava takes two steps and totals −1.6; the shortest safe route to the goal takes six steps and totals −2.0. The optimal policy walks into the lava. The environment did exactly what the reward said.

## From planning to learning
Value iteration needs the model. Q-learning uses the same target but estimates it from sampled transitions, which is why the environment pieces were written first: they are what the agent samples from.

## Interview follow-ups
- How many sweeps does value iteration need in the worst case, and how does $\gamma$ affect that?
"""
