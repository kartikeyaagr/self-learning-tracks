import numpy as np

META = dict(title='Multimodal RoPE position ids for text, images and video', tags=['Transformers', 'Embeddings'],
            diff='Medium', runtime='numpy', lessons=['aieng:04.01', 'ieng:26.03'],
            related=['rotary-position-embedding-rope', '2d-sinusoidal-positional-embedding', 'vit-patchify'])

STATEMENT = r"""
A vision-language model feeds text tokens and image patches through one decoder. Plain 1-D positions would flatten an image row by row and place vertically adjacent patches far apart. **Multimodal RoPE** gives every token three position ids, (temporal, height, width), and splits the rotary dimensions into three groups, one per axis. Text tokens use the same id on all three axes, so for pure text it reduces to ordinary RoPE.

Write `mrope_positions(segments, temporal_stride=1)` that returns the tuple `(pos, next_pos)`.

## Segments
`segments` is a list of tuples, processed in order with a running start position $p$ (initially 0):

- `("text", n)`: $n$ tokens. Token $k$ gets $(p + k,\ p + k,\ p + k)$. Afterwards $p \mathrel{+}= n$.
- `("image", h, w)`: an $h \times w$ grid of patch tokens in row-major order. Patch $(r, c)$ gets $(p,\ p + r,\ p + c)$. Afterwards $p \mathrel{+}= \max(h, w)$.
- `("video", f, h, w)`: $f$ frames of $h \times w$ patches, frame by frame, each frame row-major. Patch $(t, r, c)$ gets $(p + t \cdot s,\ p + r,\ p + c)$ with $s$ = `temporal_stride`. Afterwards $p \mathrel{+}= \max\big((f - 1)s + 1,\ h,\ w\big)$.

In every case the next segment starts one past the largest id used so far, so later text never collides with an image's ids.

## Output
- `pos`: int64 array `(3, N)`, rows = (temporal, height, width), one column per token in sequence order.
- `next_pos`: int, the final $p$ (where generated text would continue).

All counts are positive ints. An empty segment list returns a `(3, 0)` array and 0.
"""

STARTER = '''import numpy as np

def mrope_positions(segments, temporal_stride=1):
    """Return (pos int64 (3, N), next_pos int)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def mrope_positions(segments, temporal_stride=1):
    cols, p = [], 0
    for seg in segments:
        kind = seg[0]
        if kind == "text":
            k = np.arange(seg[1])
            cols.append(np.stack([p + k, p + k, p + k]))
            p += seg[1]
            continue
        f, h, w = (1, seg[1], seg[2]) if kind == "image" else seg[1:]
        t, r, c = np.meshgrid(np.arange(f), np.arange(h), np.arange(w), indexing="ij")   # frame-major, then row-major
        cols.append(np.stack([p + t.ravel() * temporal_stride, p + r.ravel(), p + c.ravel()]))
        p += max((f - 1) * temporal_stride + 1, h, w)     # an image is a one-frame video
    pos = np.concatenate(cols, axis=1) if cols else np.zeros((3, 0))
    return pos.astype(np.int64), int(p)
'''

HINTS = [
    'Treat an image as a video with one frame; then there are only two cases. The temporal id of an image patch is constant, $p$.',
    '`np.meshgrid(np.arange(f), np.arange(h), np.arange(w), indexing="ij")` followed by `.ravel()` enumerates patches frame by frame and row by row.',
    'Advance $p$ by the **extent** of the segment along its longest axis, not by its number of tokens: a $4 \\times 4$ image holds 16 tokens but only advances the position by 4.',
]

TESTS = [
    ([('text', 2), ('image', 2, 3), ('text', 2)],),
    ([('text', 3)],),
    ([('image', 3, 2)],),
    ([('text', 1), ('video', 2, 2, 2), ('text', 1)], 2),
    ([('text', 4), ('image', 4, 4), ('text', 3), ('image', 1, 5), ('text', 2)],),
    ([],),
    ([('video', 5, 2, 3), ('text', 2)], 1),
    ([('video', 3, 4, 2), ('image', 2, 2), ('text', 3)], 4),
]

EXPLANATION = r"""
## Why three axes
RoPE encodes position by rotating query and key vectors, so their dot product depends only on the position **difference**. Splitting the head dimension into temporal, height and width groups makes the attention score depend on $(\Delta t, \Delta r, \Delta c)$ separately: a patch and the patch below it differ by 1 on the height axis instead of by the image width. Text tokens use equal ids on all axes, which collapses back to 1-D RoPE, so a text-only model's behaviour is preserved.

## Why advance by the extent
After an image, the next text token gets id $p + \max(h, w)$: one past the largest id the image used on any axis. Advancing by the token count ($h \cdot w$) would also avoid collisions but would push later text far away and waste the position range on long, image-heavy inputs; advancing by less would let text ids overlap with image ids. The temporal stride (tests 4, 8) spaces frames to reflect real time between them.

## Complexity
$O(N)$ to build the ids.

## Common bugs
- Column-major patch order (`indexing="xy"`), which swaps the row and column ids.
- Starting the height and width ids at 0 instead of at $p$, so an image's ids collide with earlier text.
- Advancing $p$ for a video by $f$ instead of $(f-1)s + 1$.

## Interview follow-ups
- During generation with a KV cache, how do you compute the position ids of new text tokens after a prompt that contained images?
- How would you handle two images placed side by side in the same "row" of a document?
"""
