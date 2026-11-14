META = dict(title='Chunk text with overlap for retrieval', tags=['NLP', 'Data Processing'], diff='Easy', runtime='python')

STATEMENT = r"""
Write `chunk_tokens(tokens, size, overlap)` that splits a list of tokens into chunks for a retrieval index.

- Each chunk has at most `size` tokens. Consecutive chunks share exactly `overlap` tokens, so the start positions are `0, size - overlap, 2*(size - overlap), ...`.
- Stop once a chunk reaches the end of the list; never emit a chunk that is entirely contained in the previous one.
- Raise `ValueError` if `size <= 0` or not `0 <= overlap < size`. An empty input gives `[]`.

Return a list of `(start, chunk)` pairs, where `chunk` is a list.
"""

STARTER = '''def chunk_tokens(tokens, size, overlap):
    """Return [(start, chunk), ...] with the given overlap."""
    return [(0, list(tokens))]
'''

SOLUTION = '''def chunk_tokens(tokens, size, overlap):
    if size <= 0 or not (0 <= overlap < size):
        raise ValueError("need size > 0 and 0 <= overlap < size")
    tokens = list(tokens)
    step, out, start = size - overlap, [], 0
    while start < len(tokens):
        out.append((start, tokens[start:start + size]))
        if start + size >= len(tokens):
            break
        start += step
    return out
'''

HINTS = [
    'The stride between chunk starts is size - overlap.',
    'Check whether the current chunk already reaches the end before advancing; otherwise the last chunk is a strict suffix of the one before it.',
]

_w = "the quick brown fox jumps over the lazy dog again and again".split()
TESTS = [
    (_w, 4, 1),
    (_w, 5, 0),
    (_w, 20, 5),
    ([], 3, 1),
    (list(range(10)), 4, 2),
    (list(range(8)), 4, 2),
    (list(range(7)), 3, 2),
]

EXPLANATION = r"""
## Why overlap
Retrieval-augmented generation embeds chunks, not documents. If a sentence that answers a question straddles a chunk boundary, neither half may retrieve well. Overlap repeats the boundary region in both chunks so at least one contains it whole.

## The cost
Overlap $o$ with size $s$ multiplies the index by about $s/(s-o)$. 20% overlap is a common default; beyond that you pay storage and duplicate retrievals for little gain.

## Edge case worth getting right
Without the end check, a list of 8 with size 4 and overlap 2 would produce a final chunk `[6, 7]` that is already inside the chunk starting at 4. Duplicated suffix chunks inflate scores for the document tail.

## Interview follow-ups
- Why chunk on sentence or heading boundaries instead of fixed token counts?
- How would you attach the chunk's start offset to a citation in the answer?
"""
