import numpy as np

META = dict(title='Reward, termination and truncation', tags=['Reinforcement Learning'], diff='Easy', runtime='python',
            related=['gridworld-step'])

STATEMENT = r"""
Write `reward_done(grid, pos, t, max_steps, step_cost)` for the cell the agent has just moved into, after its `t`-th step (counting from 1).

- `G` (goal): reward `+1.0`, terminated.
- `L` (lava): reward `-1.0`, terminated.
- Any other cell: reward `-step_cost`, not terminated.
- Truncated: `t >= max_steps` **and** not terminated. Reaching the goal on the last allowed step is a termination, not a truncation.

Return `(reward, terminated, truncated)`.
"""

STARTER = '''def reward_done(grid, pos, t, max_steps, step_cost):
    """Reward plus separate termination and truncation flags."""
    cell = grid[pos[0]][pos[1]]
    done = cell in 'GL' or t >= max_steps
    return (1.0 if cell == 'G' else -step_cost), done, done
'''

SOLUTION = '''def reward_done(grid, pos, t, max_steps, step_cost):
    cell = grid[pos[0]][pos[1]]
    if cell == 'G':
        return 1.0, True, False
    if cell == 'L':
        return -1.0, True, False
    return 0.0 - step_cost, False, t >= max_steps
'''

HINTS = [
    'Termination is a property of the task (the agent reached an absorbing state). Truncation is a property of your training setup (you ran out of time).',
    'Check terminal cells first so that a goal on the last step is not also reported as truncated.',
]

G = ["S.L", "..G"]

TESTS = [
    (G, [0, 1], 1, 10, 0.01),
    (G, [1, 2], 3, 10, 0.01),
    (G, [0, 2], 2, 10, 0.01),
    (G, [1, 0], 10, 10, 0.01),
    (G, [1, 2], 10, 10, 0.01),
    (G, [0, 0], 4, 3, 0.0),
]

EXPLANATION = r"""
## Two different kinds of "done"
Gymnasium splits the old `done` flag into `terminated` and `truncated` because they mean different things for learning.

- **Terminated:** the episode really ended. The value of what comes next is zero.
- **Truncated:** you stopped watching. The agent would have kept collecting reward, so the value of the next state is not zero.

Mixing them up teaches the agent that running out of time is the same as falling into lava, which biases values near the time limit downward. The next step shows exactly where the flags enter the learning target.

## The step cost
A small negative reward per step makes shorter paths better. Too large, and walking into lava early beats a long walk to the goal.

## Interview follow-ups
- If the time limit is part of the task (a game clock the agent can see), should hitting it count as termination or truncation?
"""
