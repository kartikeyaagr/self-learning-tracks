import numpy as np

META = dict(title='Sparse autoencoder forward pass', tags=['Neural Networks', 'Embeddings'], diff='Easy', runtime='numpy',
            related=['autoencoder-forward-pass'])

STATEMENT = r"""
A sparse autoencoder (SAE) rewrites a model activation $x \in \mathbb{R}^d$ as a sparse combination of $m \gg d$ learned feature directions.

Write `sae_forward(x, W_enc, b_enc, W_dec, b_dec)` for a batch `x (B, d)`, `W_enc (d, m)`, `b_enc (m,)`, `W_dec (m, d)`, `b_dec (d,)`:

- Encode: $f = \mathrm{ReLU}\big((x - b_{dec}) W_{enc} + b_{enc}\big)$, shape `(B, m)`.
- Decode: $\hat{x} = f W_{dec} + b_{dec}$, shape `(B, d)`.

Return `(f, x_hat)`.
"""

STARTER = '''import numpy as np

def sae_forward(x, W_enc, b_enc, W_dec, b_dec):
    """Encode activations into sparse features and decode them back."""
    f = x @ W_enc + b_enc
    return f, f @ W_dec
'''

SOLUTION = '''import numpy as np

def sae_forward(x, W_enc, b_enc, W_dec, b_dec):
    x, W_enc, b_enc, W_dec, b_dec = (np.asarray(a, float) for a in (x, W_enc, b_enc, W_dec, b_dec))
    f = np.maximum((x - b_dec) @ W_enc + b_enc, 0.0)
    return f, f @ W_dec + b_dec
'''

HINTS = [
    'Subtract the decoder bias before encoding; it acts as a learned centre of the activation cloud.',
    'The ReLU is what lets most features be exactly zero for a given input.',
]

def _case(seed, B, d, m):
    g = np.random.default_rng(seed)
    return (g.normal(size=(B, d)), g.normal(size=(d, m)) / np.sqrt(d), 0.1 * g.normal(size=m) - 0.2,
            g.normal(size=(m, d)) / np.sqrt(d), 0.1 * g.normal(size=d))

TESTS = [
    ([[1.0, 0.0]], [[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]], [0.0, 0.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.0, 0.0]),
    ([[1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]], [-0.5, -0.5], [[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0]),
    _case(1, 3, 4, 8),
    _case(2, 5, 6, 16),
    _case(3, 1, 2, 4),
]

EXPLANATION = r"""
## Why a sparse autoencoder
Neurons in a language model are often polysemantic: one neuron responds to several unrelated things. The superposition hypothesis says the model represents more features than it has dimensions by storing them as nearly orthogonal directions. An SAE tries to recover those directions: a wide dictionary ($m$ is often 8 to 64 times $d$) where each input uses only a handful of entries.

## The pieces
- The decoder rows are the feature directions. Reading what a feature means starts from them.
- The encoder decides how strongly each feature is present. It is not simply the decoder transposed, because the features overlap and the encoder has to filter out interference from the others.
- Subtracting $b_{dec}$ first centres the input, so the features describe deviations from a typical activation.

## Interview follow-ups
- Why is a ReLU encoder biased towards under-estimating feature magnitudes, and what do TopK or JumpReLU SAEs change?
"""
