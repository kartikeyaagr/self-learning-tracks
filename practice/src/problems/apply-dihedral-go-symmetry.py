import numpy as np

META = dict(title='Apply one of the 8 board symmetries', tags=['Reinforcement Learning', 'Data Processing'],
            diff='Medium', runtime='numpy', lessons=[],
            related=['encode-alphago-zero-board-history', 'sample-recent-self-play-replay-batch'])

STATEMENT = r"""
Go's rules do not change if the board is rotated or mirrored, so every training example gives 8 equally valid ones. The input planes and the policy target must be transformed **together**, and the pass entry, which has no location, stays where it is.

Number the 8 symmetries `k = 0..7`. Transform $T_k$ acts on any `(..., N, N)` stack:

1. if `k >= 4`, mirror left-right: `np.flip(x, axis=-1)`;
2. then rotate counterclockwise `k % 4` times by 90 degrees in the last two axes: `np.rot90(x, k % 4, axes=(-2, -1))`.

Write `apply_symmetry(planes, policy, k)` returning `(new_planes, new_policy, inverse_k)`.

## Inputs
- `planes`: float `(C, N, N)`.
- `policy`: float `(N*N + 1,)`, where entry `r * N + c` is the move at `(r, c)` and the last entry is pass.
- `k`: int in `0..7`.

## Output
- `new_planes`: $T_k$ applied to every plane, shape `(C, N, N)`.
- `new_policy`: the first $N^2$ entries reshaped to `(N, N)` (row-major), transformed by $T_k$, flattened back, with the pass entry appended unchanged.
- `inverse_k`: the Python int `j` in `0..7` such that $T_j(T_k(x)) = x$ for every board.
"""

STARTER = '''import numpy as np

def apply_symmetry(planes, policy, k):
    """planes: (C, N, N); policy: (N*N + 1,). Return (new_planes, new_policy, inverse_k)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _transform(x, k):
    if k >= 4:
        x = np.flip(x, axis=-1)
    return np.rot90(x, k % 4, axes=(-2, -1))

def apply_symmetry(planes, policy, k):
    planes = np.asarray(planes, dtype=float)
    policy = np.asarray(policy, dtype=float)
    n = planes.shape[-1]
    new_planes = np.ascontiguousarray(_transform(planes, k))
    board = _transform(policy[:-1].reshape(n, n), k).reshape(-1)
    new_policy = np.concatenate([board, policy[-1:]])        # pass has no location: unchanged
    # Pure rotations invert by rotating back; flip-then-rotate is its own inverse (a reflection).
    inverse_k = (4 - k) % 4 if k < 4 else k
    return new_planes, new_policy, int(inverse_k)
'''

HINTS = [
    'Write one helper `_transform(x, k)` that works on the last two axes, and use it for both the planes and the reshaped policy board.',
    'For the policy: `policy[:-1].reshape(N, N)`, transform, `reshape(-1)`, then `np.concatenate` with `policy[-1:]`.',
    'For the inverse: undoing $r$ counterclockwise turns takes $(4 - r) \\bmod 4$ turns. Every symmetry with a mirror (`k >= 4`) is a reflection of the square, and a reflection undoes itself.',
]

_rng = np.random.default_rng(59)
_pl = _rng.integers(0, 2, (3, 4, 4)).astype(float)
_po = _rng.random(17); _po /= _po.sum()

TESTS = [
    (np.arange(9.0).reshape(1, 3, 3), np.arange(10.0), 1),
    (np.arange(9.0).reshape(1, 3, 3), np.arange(10.0), 4),
    (np.arange(9.0).reshape(1, 3, 3), np.arange(10.0), 0),
    (np.arange(9.0).reshape(1, 3, 3), np.arange(10.0), 3),
    (np.arange(9.0).reshape(1, 3, 3), np.arange(10.0), 6),
    (_pl, _po, 2),
    (_pl, _po, 5),
    (_pl, _po, 7),
]

EXPLANATION = r"""
## Why it is valid
The rules of Go are invariant under the 8 symmetries of the square (the dihedral group $D_4$): legality, captures and the final score are all unchanged. If position $s$ has search policy $\pi$ and outcome $z$, then $T(s)$ has policy $T(\pi)$ and the same outcome $z$. AlphaGo Zero used this both for **training** (a random symmetry per sample, 8x more data for free) and for **inference** (evaluate a randomly transformed position and map the policy back with the inverse, which averages out orientation bias in MCTS).

## Why the inverse looks like that
Let $R$ be a 90 degree rotation and $F$ the mirror. Then $T_k = R^{k \bmod 4}$ for $k < 4$, whose inverse is $R^{-k} = R^{4-k}$. For $k \ge 4$, $T_k = R^r F$, and since $F R F = R^{-1}$ we get $(R^r F)^2 = R^r (F R^r F) = R^r R^{-r} = I$: every mirrored symmetry is its own inverse.

## Common bugs
- Transforming the planes but not the policy, or using a different convention for each.
- Applying the spatial transform to the full length-$(N^2+1)$ vector, which moves the pass probability onto the board.
- Assuming `rot90(k)` is the inverse of `rot90(k)` for mirrored cases, or that the inverse of $k=1$ is $1$.

## Interview follow-ups
- Komi breaks colour symmetry, so swapping colours is not a valid augmentation. Which of these 8 symmetries could a chess engine use, and why so few?
- How would you build the 8 permutation index arrays once, so transforming a policy is a single fancy-index `policy[perm]`?
"""
