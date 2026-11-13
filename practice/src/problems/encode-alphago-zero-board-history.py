import numpy as np

META = dict(title='Encode Go history as input planes', tags=['Reinforcement Learning', 'Feature Engineering'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:04.02'],
            related=['apply-go-action-with-captures-and-repetition', 'apply-alphago-residual-block',
                     'apply-dihedral-go-symmetry'])

STATEMENT = r"""
The network never sees colours. It sees the board **from the point of view of the player to move**, plus a few past positions so it can recognise ko fights, plus one plane saying whose turn it is.

Write `encode_planes(history, to_play, k)` that builds a float array of shape `(2k + 1, N, N)`:

| Plane | Contents |
|---|---|
| `2i` for `i = 0..k-1` | `1.0` where the board `i` moves ago has a stone of colour `to_play` |
| `2i + 1` | `1.0` where that board has a stone of colour `-to_play` |
| `2k` | all `1.0` if `to_play == 1` (black to move), all `0.0` if `to_play == -1` |

## Inputs
- `history`: non-empty list of `(N, N)` int boards (`0` empty, `1` black, `-1` white), **oldest first**; `history[-1]` is the current position, so "the board `i` moves ago" is `history[-1 - i]`.
- `to_play`: `1` or `-1`, the player about to move.
- `k`: number of positions to encode, `k >= 1`.

If the history holds fewer than `k` boards, the planes for the missing older positions are all zeros. Every position is encoded relative to the **current** `to_play`, even though a different player was to move back then. Return `float64`.
"""

STARTER = '''import numpy as np

def encode_planes(history, to_play, k):
    """history: list of (N, N) boards, oldest first. Return float array (2k + 1, N, N)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def encode_planes(history, to_play, k):
    n = np.asarray(history[-1]).shape[0]
    planes = np.zeros((2 * k + 1, n, n))
    for i in range(min(k, len(history))):           # older-than-available planes stay zero
        board = np.asarray(history[-1 - i])
        planes[2 * i] = board == to_play            # "my" stones, from the mover's point of view
        planes[2 * i + 1] = board == -to_play       # "their" stones
    planes[2 * k] = 1.0 if to_play == 1 else 0.0    # colour plane: komi makes the two sides different
    return planes
'''

HINTS = [
    'Start from `np.zeros((2k + 1, N, N))`; then you only need to fill planes that exist in the history.',
    'Loop `i` from 0 to `min(k, len(history)) - 1` and read `history[-1 - i]`. A boolean comparison like `board == to_play` assigned into a float plane becomes 0.0/1.0.',
    'Plane `2i` is `board == to_play`, plane `2i + 1` is `board == -to_play`, and the last plane is filled with `1.0` only when black is to move.',
]

_b0 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
_b1 = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
_b2 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
_rng = np.random.default_rng(5)
_hist = [_rng.integers(-1, 2, size=(5, 5)) for _ in range(6)]

TESTS = [
    ([_b0, _b1], 1, 2),
    ([_b0], -1, 2),
    ([_b0, _b1, _b2], -1, 3),
    ([_b0, _b1, _b2], 1, 1),
    ([_b2], 1, 4),
    (_hist, -1, 4),
    (_hist, 1, 8),
]

EXPLANATION = r"""
## Why relative colours
Go is symmetric between the players except for komi. Encoding "my stones / their stones" lets one set of weights play both sides, and the network does not have to learn every pattern twice. The extra colour plane restores the one asymmetry: with komi, the same shape can be winning for black and losing for white.

## Why history
A single position does not determine the legal moves: whether a ko recapture is allowed depends on the previous move. The past planes also act as a cheap attention signal (the last move is visible as a difference between plane 0 and plane 2). AlphaGo Zero used 8 past positions, giving $2 \cdot 8 + 1 = 17$ planes.

## Complexity
$O(kN^2)$ to fill the planes; the array itself is the cost.

## Common bugs
- Reading the history in the wrong direction (`history[i]` instead of `history[-1 - i]`).
- Encoding old boards from the perspective of whoever was to move *then*. All planes use the current mover.
- Filling missing history with copies of the oldest board instead of zeros.
- Leaving the planes as `bool` or `int`: the convolution expects floats.

## Interview follow-ups
- Leela Zero and KataGo add planes for liberties, ladder status and legal moves. What is the trade-off between hand-made features and letting the network learn them?
- How would you encode a batch of positions from many games efficiently (stack into `(B, 2k+1, N, N)` with one vectorised comparison)?
"""
