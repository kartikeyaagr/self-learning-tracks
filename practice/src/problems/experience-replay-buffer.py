import numpy as np

META = dict(title='Circular experience replay buffer', tags=['Reinforcement Learning'], diff='Easy', runtime='python',
            lessons=['air-fanout-v1:07.03'],
            related=['prioritized-experience-replay', 'dqn-loss-with-target-network'])

STATEMENT = r"""
DQN stores transitions in a fixed-capacity **replay buffer** and trains on random minibatches drawn from it. Once full, the newest transition overwrites the oldest.

Write `run_replay_buffer(capacity, ops)` that simulates such a buffer and returns `(samples, size, next_slot)`.

## Inputs
- `capacity`: int `>= 1`.
- `ops`: list of operations, applied in order. Each is a pair:
  - `("push", transition)`: store `transition` (any value; return it as given) in slot `next_slot`, then advance `next_slot = (next_slot + 1) % capacity`. Size grows by one until it reaches `capacity`.
  - `("sample", us)`: `us` is a list of uniform draws in `[0, 1)`. For each `u`, return the transition in **slot** `min(int(u * size), size - 1)`. Slots are storage positions, not ages. If the buffer is empty, the sample is `[]`.

## Output
- `samples`: a list with one entry per `"sample"` op, each a list of transitions in the order of `us`.
- `size`: the number of stored transitions at the end.
- `next_slot`: the slot the next push would write to.

Both slots and `next_slot` start at 0.
"""

STARTER = '''def run_replay_buffer(capacity, ops):
    """Return (list of sampled batches, final size, next slot to write)."""
    raise NotImplementedError
'''

SOLUTION = '''def run_replay_buffer(capacity, ops):
    storage = [None] * capacity          # preallocated ring; slot i holds whatever was written there last
    size = slot = 0
    samples = []
    for kind, arg in ops:
        if kind == "push":
            storage[slot] = arg
            slot = (slot + 1) % capacity
            size = min(size + 1, capacity)
        else:
            samples.append([storage[min(int(u * size), size - 1)] for u in arg] if size else [])
    return samples, size, slot
'''

HINTS = [
    'Preallocate a list of `capacity` slots and keep two integers: how many are filled and where the next write goes.',
    'A push writes at `next_slot`, then `next_slot = (next_slot + 1) % capacity` and `size = min(size + 1, capacity)`. That wrap is the whole eviction policy.',
    'A sample maps each `u` to `min(int(u * size), size - 1)`, an index into the slot list (not into insertion order). Handle `size == 0` first.',
]

_r = np.random.default_rng(161)
_ops = []
for i in range(30):
    if _r.uniform() < 0.7:
        _ops.append(("push", [i, round(float(_r.normal()), 2)]))
    else:
        _ops.append(("sample", [float(x) for x in _r.uniform(size=3)]))
TESTS = [
    (3, [("push", "a"), ("push", "b"), ("sample", [0.0, 0.6])]),
    (2, [("push", "a"), ("push", "b"), ("push", "c"), ("sample", [0.1, 0.9])]),
    (2, [("sample", [0.5]), ("push", 1), ("sample", [0.99, 0.0])]),
    (1, [("push", 10), ("push", 20), ("push", 30), ("sample", [0.3])]),
    (4, [("push", [0, 1.0]), ("sample", []), ("push", [1, -1.0])]),
    (5, [("push", x) for x in range(12)] + [("sample", [0.0, 0.2, 0.4, 0.6, 0.8])]),
    (6, _ops),
    (40, _ops),
]

EXPLANATION = r"""
## Why replay
Consecutive transitions are highly correlated, and a network trained on them in order overfits to whatever it saw last. Sampling uniformly from a large buffer breaks those correlations, makes minibatches closer to i.i.d., and reuses each transition many times, which matters when environment steps are expensive.

## Why a ring
A Python list with `append` and `pop(0)` would evict correctly, but `pop(0)` is $O(n)$. A ring buffer makes push and sample $O(1)$, and real implementations preallocate one NumPy array per field (states, actions, rewards, ...) so a batch is a single fancy-index gather. Because sampling is uniform, the storage order doesn't matter, which is why it can be slot-based.

## Edge cases
- Test 2: capacity 2, three pushes. `"c"` overwrote `"a"` in slot 0.
- Test 3: sampling an empty buffer gives `[]`.
- Test 4: capacity 1 keeps only the newest transition.

## Complexity
$O(1)$ per push, $O(k)$ per sample of $k$ draws, $O(\text{capacity})$ memory.

## Common bugs
- Sampling with `int(u * capacity)` before the buffer is full, which can return empty slots.
- Advancing `next_slot` without the modulo, or growing `size` past capacity.

## Interview follow-ups
- How would you store frames for Atari without keeping each frame four times (frame stacking)?
- How do you sample n-step returns from a ring buffer without crossing an episode boundary or the write pointer?
"""
