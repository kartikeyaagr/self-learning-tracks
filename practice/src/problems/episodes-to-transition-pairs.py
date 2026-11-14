import numpy as np

META = dict(title='Turn episodes into world-model training pairs', tags=['Reinforcement Learning', 'Data Processing'], diff='Easy', runtime='numpy')

STATEMENT = r"""
A dynamics model learns $(s_t, a_t) \mapsto (s_{t+1} - s_t,\ r_t,\ \text{done}_t)$. Write `make_pairs(episodes)`.

Each episode is a dict with `obs` (list of $T+1$ state vectors), `actions` ($T$ action vectors), `rewards` ($T$ floats) and `terminated` (bool).

- For every $t < T$ in every episode, emit input $[s_t, a_t]$ (concatenated), target delta $s_{t+1} - s_t$, reward $r_t$ and done flag.
- The done flag is 1 only for the **last** transition of an episode with `terminated = true`. A truncated episode's last transition has done 0.
- Never pair the last state of one episode with the first state of the next.

Return a dict with `X` (list of rows), `dS`, `r` and `done` (lists), in episode order.
"""

STARTER = '''import numpy as np

def make_pairs(episodes):
    """(state, action) -> (state delta, reward, done) rows from a list of episodes."""
    S = np.vstack([np.asarray(e["obs"], float) for e in episodes])
    A = np.vstack([np.asarray(e["actions"], float) for e in episodes])
    n = len(A)
    return {"X": np.hstack([S[:n], A]).tolist(), "dS": (S[1:n + 1] - S[:n]).tolist(),
            "r": [x for e in episodes for x in e["rewards"]], "done": [0] * n}
'''

SOLUTION = '''import numpy as np

def make_pairs(episodes):
    X, dS, r, done = [], [], [], []
    for e in episodes:
        S = np.asarray(e["obs"], float)
        A = np.asarray(e["actions"], float).reshape(len(S) - 1, -1)
        T = len(A)
        for t in range(T):
            X.append(np.r_[S[t], A[t]].tolist())
            dS.append((S[t + 1] - S[t]).tolist())
            r.append(float(e["rewards"][t]))
            done.append(int(bool(e["terminated"]) and t == T - 1))
    return {"X": X, "dS": dS, "r": r, "done": done}
'''

HINTS = [
    'Loop episode by episode; stacking all observations first is exactly how the cross-episode pair sneaks in.',
    'Predicting the change in state, not the next state, is easier for a network because most dimensions change little per step.',
]

E1 = {"obs": [[0.0], [1.0], [3.0]], "actions": [[1.0], [2.0]], "rewards": [0.0, 1.0], "terminated": True}
E2 = {"obs": [[10.0], [10.5]], "actions": [[0.5]], "rewards": [-1.0], "terminated": False}
E3 = {"obs": [[0.0, 0.0], [1.0, 0.5], [1.5, 1.5], [2.0, 2.0]], "actions": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], "rewards": [0.1, 0.2, 0.3], "terminated": False}

TESTS = [
    ([E1],),
    ([E1, E2],),
    ([E2, E1],),
    ([E3],),
    ([E3, E3],),
]

EXPLANATION = r"""
## The silent bug
Concatenating all observations into one array and taking consecutive pairs creates a transition from the end of one episode to the start of the next. The model then learns that the world occasionally teleports. In the second test the starter produces a jump from 3 to 10.

## Termination again
The done flag should mean the environment ended the episode. A model that learns to predict "done" from truncations will imagine episodes ending at arbitrary states during planning.

## Interview follow-ups
- Why do many world models (Dreamer, for example) learn a latent state instead of predicting raw observations, and what do they lose by doing that?
"""
