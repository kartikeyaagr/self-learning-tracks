import numpy as np

META = dict(title='Sinusoidal positional encoding', tags=['Transformers', 'Linear Algebra'], diff='Medium', runtime='numpy',
            lessons=['aieng:04.01'], related=['scaled-dot-product-attention'])

STATEMENT = r"""
Attention by itself ignores token order. The original Transformer adds a fixed position signal to each embedding:

$$PE_{p,\,2i} = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad PE_{p,\,2i+1} = \cos\!\left(\frac{p}{10000^{2i/d}}\right)$$

Write `positional_encoding(seq_len, d_model)` that returns the `(seq_len, d_model)` matrix for positions $p = 0, \dots, \texttt{seq\_len}-1$.

## Requirements
- Even columns hold sines and odd columns hold cosines, and column pair $(2i, 2i+1)$ shares one frequency.
- If `d_model` is odd, the last column is a sine (its cosine partner is dropped).
- Vectorised: no loop over positions.
"""

STARTER = '''import numpy as np

def positional_encoding(seq_len, d_model):
    """Return PE with shape (seq_len, d_model)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]                     # (L, 1)
    i2 = np.arange(0, d_model, 2)[None, :]                # even column indices 2i, (1, ceil(d/2))
    angle = pos / np.power(10000.0, i2 / d_model)         # (L, ceil(d/2))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, : d_model // 2])        # odd d_model: one fewer cosine column
    return pe
'''

HINTS = [
    'Make positions a column `(L, 1)` and frequencies a row `(1, d/2)`. Broadcasting then gives every angle at once.',
    'The exponent uses the **even** index: column pair `(2i, 2i+1)` uses `10000 ** (2i / d_model)`.',
    'Assign the halves with slices: `pe[:, 0::2] = sin(...)`, `pe[:, 1::2] = cos(...)`. For odd d there is one fewer odd column.',
]

TESTS = [
    (1, 4),
    (4, 2),
    (6, 8),
    (5, 7),
    (50, 16),
    (3, 1),
]

EXPLANATION = r"""
## Why sinusoids
- **Bounded**: values stay in $[-1, 1]$ however long the sequence.
- **Many wavelengths**: the frequencies form a geometric series from $2\pi$ up to $10000\cdot 2\pi$, so low columns change fast (fine position) and high columns slowly (coarse position).
- **Relative offsets are linear**: for each frequency, $PE_{p+k}$ is a fixed rotation of $PE_p$. So "k tokens later" is a linear map the model can learn.

## Shape trick
`pos[:, None] / freq[None, :]` gives an `(L, d/2)` angle grid in one broadcast. Interleaving via strided slices avoids building separate sine and cosine arrays and then shuffling them.

## Where it went next
Learned absolute embeddings (GPT-2, BERT), then rotary embeddings (RoPE), which apply the same rotation idea to $q$ and $k$ inside attention so dot products depend only on relative position. Most modern LLMs use RoPE.

## Interview follow-ups
- Implement RoPE for a single head given this frequency schedule.
- Can sinusoidal encodings extrapolate to longer sequences than seen in training? Why does this often fail in practice?
"""
