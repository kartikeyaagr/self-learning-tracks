import numpy as np

META = dict(title='GPT-2 embeddings with a KV-cache offset and tied logits', tags=['Language Models', 'Embeddings', 'Transformers'], diff='Easy', runtime='numpy')

STATEMENT = r"""
GPT-2 adds a learned position table to token embeddings and reuses the token table as the output projection.

Write `gpt2_embed_and_logits(ids, wte, wpe, h_final, past_len)`:

- `wte` is `(V, d)`, `wpe` is `(n_ctx, d)`. `ids` are the **new** tokens; `past_len` tokens are already in the KV cache, so the new tokens sit at positions `past_len, past_len + 1, ...`.
- Raise `ValueError` if `past_len + len(ids) > n_ctx`.
- Input embeddings: `x = wte[ids] + wpe[positions]`.
- Logits: `h_final @ wte.T`, where `h_final` is the final hidden state for the new tokens (after the final LayerNorm, given to you).

Return `(x, logits)`.
"""

STARTER = '''import numpy as np

def gpt2_embed_and_logits(ids, wte, wpe, h_final, past_len):
    """Return (x, logits)."""
    wte, wpe = np.asarray(wte), np.asarray(wpe)
    x = wte[ids] + wpe[: len(ids)]
    return x, np.asarray(h_final) @ wte.T
'''

SOLUTION = '''import numpy as np

def gpt2_embed_and_logits(ids, wte, wpe, h_final, past_len):
    wte, wpe = np.asarray(wte, float), np.asarray(wpe, float)
    T = len(ids)
    if past_len + T > len(wpe):
        raise ValueError("sequence exceeds the context window")
    x = wte[np.asarray(ids, int)] + wpe[past_len:past_len + T]
    return x, np.asarray(h_final, float) @ wte.T
'''

HINTS = [
    'During cached decoding the new token is not at position 0; offset the position slice by past_len.',
    'Weight tying means there is no separate output matrix: the logits are dot products with every token embedding.',
]

def _case(seed, V, n_ctx, d, T, past):
    r = np.random.default_rng(seed)
    return (list(r.integers(0, V, T)), r.normal(size=(V, d)), r.normal(size=(n_ctx, d)), r.normal(size=(T, d)), past)

TESTS = [_case(1, 10, 8, 4, 3, 0), _case(2, 10, 8, 4, 1, 5), _case(3, 50, 16, 6, 4, 12),
         _case(4, 5, 4, 2, 4, 0), _case(5, 20, 32, 8, 2, 7)]

EXPLANATION = r"""
## Learned positions
GPT-2 learns one vector per absolute position up to 1024. This is simple and works, but the model cannot run past `n_ctx`, which is why later models moved to rotary or relative schemes that extrapolate further.

## The cache offset bug
With a KV cache each decoding step feeds one token. If you add `wpe[0]` instead of `wpe[past_len]`, every generated token believes it is at the start of the sequence, and output quality collapses after a few tokens even though nothing crashes.

## Weight tying
Sharing `wte` between input and output saves $Vd$ parameters (38M for GPT-2 small) and couples the two spaces: tokens with similar embeddings get similar logits. Most decoder models of that era tied weights; some larger later models untie them.

## Interview follow-ups
- What happens to the tied logits if you scale `wte` during training?
"""
