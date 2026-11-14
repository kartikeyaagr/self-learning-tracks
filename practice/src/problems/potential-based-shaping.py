import numpy as np

META = dict(title='A progress bonus that cannot be farmed', tags=['Reinforcement Learning'], diff='Medium', runtime='numpy',
            related=['reward-and-termination'])

STATEMENT = r"""
Reward shaping adds a bonus that guides the agent. Potential-based shaping uses a function $\Phi(s)$ of the state:
$$r'_t = r_t + \gamma\,\Phi(s_{t+1}) - \Phi(s_t).$$

Write `shape_path(phi, rewards, gamma, terminal)`:

- `phi` has length $T+1$: the potential of states $s_0, \dots, s_T$. `rewards` has length $T$.
- If `terminal` is true, the final state is absorbing and its potential is treated as 0.
- Return a dict with `shaped` (the list of $r'_t$), `ret` (discounted return of the original rewards, $\sum_t \gamma^t r_t$) and `shaped_ret` (the same for $r'$).

The identity to observe: $\text{shaped\_ret} = \text{ret} - \Phi(s_0) + \gamma^T \Phi(s_T)$, where the last term vanishes for a terminal end.
"""

STARTER = '''import numpy as np

def shape_path(phi, rewards, gamma, terminal):
    """Potential-based shaped rewards and both discounted returns."""
    phi, r = np.asarray(phi, float), np.asarray(rewards, float)
    shaped = r + phi[1:] - phi[:-1]
    disc = gamma ** np.arange(len(r))
    return {"shaped": shaped.tolist(), "ret": float(disc @ r), "shaped_ret": float(disc @ shaped)}
'''

SOLUTION = '''import numpy as np

def shape_path(phi, rewards, gamma, terminal):
    phi, r = np.array(phi, float), np.asarray(rewards, float)
    if terminal:
        phi[-1] = 0.0
    shaped = r + gamma * phi[1:] - phi[:-1]
    disc = gamma ** np.arange(len(r))
    return {"shaped": shaped.tolist(), "ret": float(disc @ r), "shaped_ret": float(disc @ shaped)}
'''

HINTS = [
    'The gamma in front of Phi(s_{t+1}) is what makes the sum telescope under discounting.',
    'Copy phi before zeroing its last entry so you do not change the caller’s data.',
    'Try a loop path that walks away from the goal and back: with potential-based shaping it earns no net bonus.',
]

TESTS = [
    ([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0], 1.0, True),
    ([-3.0, -2.0, -1.0, 0.0], [-0.1, -0.1, 1.0], 0.9, True),
    ([-2.0, -1.0, -2.0, -1.0, -2.0], [0.0, 0.0, 0.0, 0.0], 1.0, False),
    ([1.0, 2.0, 3.0], [0.5, 0.5], 0.5, False),
    ([0.0], [], 0.9, True),
]

EXPLANATION = r"""
## Why naive bonuses go wrong
A bonus of "+0.1 for moving closer to the goal" sounds harmless. But an agent that steps away and back collects the bonus again every lap, and if laps pay more than finishing, it never finishes. Agents find these loops reliably.

## Why potentials are safe
With $F = \gamma\Phi(s') - \Phi(s)$, the bonuses telescope. Any path's shaped return equals its original return minus $\Phi(s_0)$ (a constant for a given start) plus a term at the end. A loop returns to where it started and earns nothing. Ng, Harada and Russell (1999) showed that this form, and essentially only this form, leaves the optimal policy unchanged.

## The terminal convention
Setting $\Phi = 0$ at absorbing states is what makes the end term vanish; without it, the agent could prefer ending in high-potential non-goal states.

## Interview follow-ups
- Shaping does not change the optimal policy, so why does it help at all?
"""
