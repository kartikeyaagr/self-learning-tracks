import numpy as np

META = dict(title='Policy and value heads with a legal-move mask', tags=['Convolutional Networks', 'Reinforcement Learning'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:02.02'],
            related=['apply-alphago-residual-block', 'assemble-compact-dual-head-network', 'build-legal-go-action-mask',
                     'softmax-rows'])

STATEMENT = r"""
The trunk produces features $F \in \mathbb{R}^{C \times N \times N}$. Two small heads turn them into a move distribution over $A = N^2 + 1$ actions (pass last) and a scalar evaluation in $[-1, 1]$.

Write `dual_heads(features, legal, params)` returning `(policy, value)`.

## Policy head
1. 1x1 convolution to 2 channels: $P_1[k, i, j] = \mathrm{relu}\big(\sum_c W^{p}_{k c} F[c, i, j] + b^{p}_k\big)$, with `p_conv_w` of shape `(2, C)` and `p_conv_b` of shape `(2,)`.
2. Flatten $P_1$ in C order (channel-major, `reshape(-1)`) to length $2N^2$.
3. Linear layer to logits: `p_fc_w` `(A, 2N^2)`, `p_fc_b` `(A,)`.
4. Masked softmax: illegal actions (`legal[a] == False`) get probability exactly `0`; legal ones get a softmax over the legal logits only. At least one action is legal.

## Value head
1. 1x1 convolution to 1 channel + relu: `v_conv_w` `(1, C)`, `v_conv_b` `(1,)`; flatten to length $N^2$.
2. Hidden layer: `v_fc1_w` `(H, N^2)`, `v_fc1_b` `(H,)`, then relu.
3. Output: $v = \tanh(w_2 \cdot h + b_2)$ with `v_fc2_w` `(H,)` and `v_fc2_b` a float.

## Inputs and output
- `features`: float `(C, N, N)`; `legal`: bool `(A,)`; `params`: dict with the keys above.
- Return `policy` as a float array `(A,)` and `value` as a Python float. (These heads use biases instead of batch norm to keep the exercise focused.)
"""

STARTER = '''import numpy as np

def dual_heads(features, legal, params):
    """features: (C, N, N); legal: bool (N*N + 1,). Return (policy (N*N + 1,), value float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def dual_heads(features, legal, params):
    F = np.asarray(features, dtype=float)
    legal = np.asarray(legal, dtype=bool)
    p = {k: np.asarray(v, dtype=float) for k, v in params.items()}

    # A 1x1 convolution is a channel-mixing matmul applied at every board point.
    ph = np.maximum(np.einsum('kc,chw->khw', p['p_conv_w'], F) + p['p_conv_b'][:, None, None], 0.0)
    logits = p['p_fc_w'] @ ph.reshape(-1) + p['p_fc_b']
    logits = np.where(legal, logits, -np.inf)       # exp(-inf) = 0: illegal moves get exactly zero mass
    e = np.exp(logits - logits.max())               # max is over legal entries, so it is finite
    policy = e / e.sum()

    vh = np.maximum(np.einsum('kc,chw->khw', p['v_conv_w'], F) + p['v_conv_b'][:, None, None], 0.0)
    h = np.maximum(p['v_fc1_w'] @ vh.reshape(-1) + p['v_fc1_b'], 0.0)
    value = float(np.tanh(p['v_fc2_w'] @ h + p['v_fc2_b']))
    return policy, value
'''

HINTS = [
    'A 1x1 convolution with weights `(K, C)` is `np.einsum("kc,chw->khw", W, F)` plus a bias broadcast as `b[:, None, None]`.',
    'Mask before the softmax: replace illegal logits with `-np.inf`, subtract the max (taken over what remains), exponentiate and normalise. `exp(-inf)` is exactly 0.',
    'Value head: 1x1 conv, relu, `reshape(-1)`, matmul + bias, relu, dot with `v_fc2_w`, add `v_fc2_b`, `tanh`. Convert the value to a Python float.',
]

def _params(rng, c, n, h, s=0.5):
    a = n * n + 1
    return {'p_conv_w': rng.normal(0, s, (2, c)), 'p_conv_b': rng.normal(0, 0.1, 2),
            'p_fc_w': rng.normal(0, s, (a, 2 * n * n)), 'p_fc_b': rng.normal(0, 0.1, a),
            'v_conv_w': rng.normal(0, s, (1, c)), 'v_conv_b': rng.normal(0, 0.1, 1),
            'v_fc1_w': rng.normal(0, s, (h, n * n)), 'v_fc1_b': rng.normal(0, 0.1, h),
            'v_fc2_w': rng.normal(0, s, h), 'v_fc2_b': float(rng.normal(0, 0.1))}

# Tiny hand-checkable case: 1 channel, 1x1 board (actions: the single point, pass).
_tiny = {'p_conv_w': np.array([[1.0], [-1.0]]), 'p_conv_b': np.zeros(2),
         'p_fc_w': np.array([[1.0, 0.0], [0.0, 1.0]]), 'p_fc_b': np.zeros(2),
         'v_conv_w': np.array([[1.0]]), 'v_conv_b': np.zeros(1),
         'v_fc1_w': np.array([[1.0]]), 'v_fc1_b': np.zeros(1),
         'v_fc2_w': np.array([0.5]), 'v_fc2_b': 0.0}
_rng = np.random.default_rng(17)
_p3 = _params(_rng, 3, 3, 4)
_f3 = _rng.normal(size=(3, 3, 3))
_p4 = _params(_rng, 5, 4, 6)
_f4 = _rng.normal(size=(5, 4, 4))
_m4 = _rng.integers(0, 2, 17).astype(bool); _m4[-1] = True

TESTS = [
    (np.array([[[2.0]]]), np.array([True, True]), _tiny),
    (np.array([[[2.0]]]), np.array([False, True]), _tiny),
    (np.array([[[-1.0]]]), np.array([True, True]), _tiny),
    (_f3, np.ones(10, dtype=bool), _p3),
    (_f3, np.array([True, False, True, False, False, True, True, False, True, False]), _p3),
    (_f4, _m4, _p4),
    (_f4 * 20.0, np.ones(17, dtype=bool), _p4),
]

EXPLANATION = r"""
## Why two heads on one trunk
Move prediction and position evaluation need the same understanding of the board (which groups are weak, where the territory is). Sharing the trunk regularises both: the policy target gives a dense learning signal for every move, and the value head piggybacks on those features. AlphaGo Zero found the combined network clearly stronger than separate policy and value networks.

## The masked softmax
Setting illegal logits to $-\infty$ makes their exponentials exactly 0, so the remaining probabilities form a softmax over legal moves only:

$$\pi_a = \frac{e^{z_a}}{\sum_{b \in \text{legal}} e^{z_b}} \quad (a \text{ legal}), \qquad \pi_a = 0 \text{ otherwise.}$$

Subtracting the max keeps `exp` from overflowing (test 7 scales the features by 20). The max must be finite, which is why at least one move has to be legal; in Go, pass always is.

## Complexity
The policy head's fully connected layer dominates: $(N^2 + 1) \cdot 2N^2$ weights, about 260k for 19x19.

## Common bugs
- Masking **after** the softmax and renormalising: fine mathematically, but zeroing without renormalising gives a distribution that does not sum to 1.
- Masking with a large negative number like `-1e9` and then subtracting the global max over *all* logits: works until an illegal logit is the max.
- Flattening in the wrong order (`(N, N, 2)` instead of `(2, N, N)`), which silently scrambles the weights.
- Returning a 0-d array instead of a float for the value.

## Interview follow-ups
- Why a `tanh` on the value head, and what would change if the target were a win probability in $[0, 1]$ instead of $z \in \{-1, 1\}$?
- The policy head's dense layer ties the network to one board size. How do fully convolutional policy heads (KataGo) avoid that?
"""
